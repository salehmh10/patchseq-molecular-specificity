"""Post-hoc uncertainty, null tests, confounder comparisons, and sensitivities."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .evaluation import assert_exact_oof_once, performance_table, regression_metrics
from .models import elastic_net_candidates, select_on_inner_groups, xgboost_candidates
from .statistics import (
    cluster_bootstrap_metrics,
    donor_level_subclass_adjusted_spearman_permutation,
    stratified_spearman_permutation,
    summarize_bootstrap,
)
from .training import MODULES, load_feature_names


def _stable_seed(*values: object, base: int = 42) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_oof_artifacts(root: Path, oof: pd.DataFrame, table: pd.DataFrame) -> dict[str, object]:
    """Independently prove exact OOF grid, fold identity, truth, and metric coverage."""
    expected_keys: set[tuple[str, str, str]] = set()
    for module in MODULES:
        expected_keys.update(
            {
                (module, "ElasticNet", "ephys_only"),
                (module, "XGBoost", "ephys_only"),
                (module, "MLP", "ephys_only"),
                (module, "Dummy", "dummy"),
                (module, "SubclassRidge", "subclass_only"),
                (module, "EphysSubclassRidge", "ephys_plus_subclass"),
            }
        )
    expected_cells = set(table["canonical_cell_id"].astype(str))
    if len(expected_cells) != len(table):
        raise AssertionError("Modeling-table canonical IDs are not unique")
    assert_exact_oof_once(oof, expected_cells, expected_keys)

    frozen = pd.read_csv(root / "results/tables/cv_folds.csv")
    frozen["canonical_cell_id"] = frozen["canonical_cell_id"].astype(str)
    frozen["group_id"] = frozen["group_id"].astype(str)
    if not frozen["canonical_cell_id"].is_unique or set(frozen["canonical_cell_id"]) != expected_cells:
        raise AssertionError("Frozen fold cells differ from modeling-table cells")
    observed = oof[["canonical_cell_id", "group_id", "subclass", "fold"]].drop_duplicates()
    if len(observed) != len(frozen) or not observed["canonical_cell_id"].is_unique:
        raise AssertionError("OOF metadata does not collapse to one row per cell")
    joined = observed.merge(
        frozen,
        on="canonical_cell_id",
        validate="one_to_one",
        suffixes=("_oof", "_frozen"),
    )
    if not joined["fold_oof"].eq(joined["fold_frozen"]).all():
        raise AssertionError("OOF fold differs from frozen fold")
    if not joined["group_id_oof"].astype(str).eq(joined["group_id_frozen"].astype(str)).all():
        raise AssertionError("OOF donor/group differs from frozen folds")
    if not joined["subclass_oof"].astype(str).eq(joined["subclass_frozen"].astype(str)).all():
        raise AssertionError("OOF subclass differs from frozen folds")
    if oof.groupby(oof["group_id"].astype(str))["fold"].nunique().max() != 1:
        raise AssertionError("At least one donor/group crosses outer folds")
    if oof.groupby("canonical_cell_id")["fold"].nunique().max() != 1:
        raise AssertionError("At least one cell has multiple outer folds")

    targets = table[["canonical_cell_id"] + [f"target_{module}" for module in MODULES]].copy()
    targets["canonical_cell_id"] = targets["canonical_cell_id"].astype(str)
    for module in MODULES:
        observed_truth = oof.loc[
            oof["module"].eq(module), ["canonical_cell_id", "y_true"]
        ].drop_duplicates()
        if len(observed_truth) != len(table) or not observed_truth["canonical_cell_id"].is_unique:
            raise AssertionError(f"Truth coverage is not exact for {module}")
        comparison = observed_truth.merge(
            targets[["canonical_cell_id", f"target_{module}"]],
            on="canonical_cell_id",
            validate="one_to_one",
        )
        if not np.array_equal(
            comparison["y_true"].to_numpy(), comparison[f"target_{module}"].to_numpy()
        ):
            raise AssertionError(f"Saved OOF truth differs from modeling target for {module}")

    calculated = performance_table(oof).sort_values(["analysis", "module", "model"]).reset_index(drop=True)
    performance_path = root / "results/tables/model_performance.csv"
    if performance_path.exists():
        saved = pd.read_csv(performance_path).sort_values(["analysis", "module", "model"]).reset_index(drop=True)
        if not calculated[["module", "model", "analysis", "n"]].equals(
            saved[["module", "model", "analysis", "n"]]
        ) or not np.allclose(
            calculated[["r2", "mae", "rmse", "spearman"]],
            saved[["r2", "mae", "rmse", "spearman"]],
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        ):
            raise AssertionError("Saved model performance cannot be exactly recalculated from OOF")

    return {
        "status": "PASS",
        "oof_sha256": _sha256(root / "results/predictions/oof_predictions.parquet"),
        "rows": len(oof),
        "cells": len(expected_cells),
        "model_module_analysis_keys": len(expected_keys),
        "rows_per_cell": int(oof.groupby("canonical_cell_id").size().iloc[0]),
        "fold_cell_counts": {
            str(key): int(value) for key, value in frozen["fold"].value_counts().sort_index().items()
        },
        "donor_groups": int(frozen["group_id"].nunique()),
        "max_folds_per_group": int(oof.groupby(oof["group_id"].astype(str))["fold"].nunique().max()),
        "truth_exact_to_modeling_table": True,
        "metrics_exact_to_saved": True,
    }


def infer_from_oof(
    oof: pd.DataFrame,
    *,
    n_bootstrap: int,
    n_permutations: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    interval_rows: list[dict[str, object]] = []
    draw_parts: list[pd.DataFrame] = []
    for keys, frame in oof.groupby(["module", "model", "analysis"], sort=True):
        local_seed = _stable_seed(*keys, "bootstrap", base=seed)
        draws = cluster_bootstrap_metrics(
            frame,
            group_column="group_id",
            n_resamples=n_bootstrap,
            seed=local_seed,
        )
        draws.insert(0, "analysis", keys[2])
        draws.insert(0, "model", keys[1])
        draws.insert(0, "module", keys[0])
        draw_parts.append(draws)
        interval_rows.append(
            {
                "module": keys[0],
                "model": keys[1],
                "analysis": keys[2],
                "n_bootstrap": n_bootstrap,
                **summarize_bootstrap(draws),
            }
        )
    permutation_rows: list[dict[str, object]] = []
    primary = oof.loc[oof.analysis.eq("ephys_only")]
    for keys, frame in primary.groupby(["module", "model"], sort=True):
        result = stratified_spearman_permutation(
            frame,
            strata_column="subclass",
            n_permutations=n_permutations,
            seed=_stable_seed(*keys, "permutation", base=seed),
        )
        permutation_rows.append({"module": keys[0], "model": keys[1], **result})
    return pd.DataFrame(interval_rows), pd.concat(draw_parts, ignore_index=True), pd.DataFrame(permutation_rows)


def donor_aware_permutation_sensitivity(
    oof: pd.DataFrame, *, n_permutations: int, seed: int = 42
) -> pd.DataFrame:
    """Run the independent-donor, subclass-adjusted no-refit sensitivity test."""

    rows: list[dict[str, object]] = []
    primary = oof.loc[oof.analysis.eq("ephys_only")]
    for keys, frame in primary.groupby(["module", "model"], sort=True):
        result = donor_level_subclass_adjusted_spearman_permutation(
            frame,
            group_column="group_id",
            strata_column="subclass",
            n_permutations=n_permutations,
            seed=_stable_seed(*keys, "donor_permutation", base=seed),
        )
        rows.append({"module": keys[0], "model": keys[1], **result})
    return pd.DataFrame(rows)


def within_subclass_sensitivity(
    table: pd.DataFrame,
    features: list[str],
    *,
    level: int,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible: list[str] = []
    counts = table.subclass.value_counts()
    for subclass in counts.index:
        frame = table.loc[table.subclass.eq(subclass)]
        if len(frame) >= 150 and frame.group_id.astype(str).nunique() >= 3:
            eligible.append(str(subclass))
        if len(eligible) == 2:
            break
    prediction_parts: list[pd.DataFrame] = []
    eligibility_rows = []
    for subclass, count in counts.items():
        frame = table.loc[table.subclass.eq(subclass)]
        eligibility_rows.append(
            {
                "subclass": subclass,
                "n": int(count),
                "n_groups": int(frame.group_id.astype(str).nunique()),
                "eligible_and_selected": str(subclass) in eligible,
            }
        )
    for subclass in eligible:
        subset = table.loc[table.subclass.astype(str).eq(subclass)].reset_index(drop=True)
        groups = subset.group_id.astype(str).to_numpy()
        splitter = GroupKFold(n_splits=3)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(subset, groups=groups)):
            train, test = subset.iloc[train_idx], subset.iloc[test_idx]
            X_train, X_test = train[features], test[features]
            train_groups = train.group_id.astype(str).to_numpy()
            for module in MODULES:
                y_train = train[f"target_{module}"].to_numpy(dtype=float)
                selected = {
                    "ElasticNet": select_on_inner_groups(
                        elastic_net_candidates(features, seed=seed),
                        X_train,
                        y_train,
                        train_groups,
                        seed=seed + fold,
                    ),
                    "XGBoost": select_on_inner_groups(
                        xgboost_candidates(features, seed=seed, level=level),
                        X_train,
                        y_train,
                        train_groups,
                        seed=seed + fold,
                    ),
                }
                for model, fitted in selected.items():
                    prediction_parts.append(
                        pd.DataFrame(
                            {
                                "canonical_cell_id": test.canonical_cell_id.astype(str).to_numpy(),
                                "group_id": test.group_id.astype(str).to_numpy(),
                                "subclass": subclass,
                                "fold": fold,
                                "module": module,
                                "model": model,
                                "analysis": "within_subclass",
                                "y_true": test[f"target_{module}"].to_numpy(dtype=float),
                                "y_pred": fitted.pipeline.predict(X_test),
                            }
                        )
                    )
    predictions = pd.concat(prediction_parts, ignore_index=True) if prediction_parts else pd.DataFrame()
    return predictions, pd.DataFrame(eligibility_rows)


def within_subclass_performance(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, frame in predictions.groupby(["subclass", "module", "model", "analysis"], sort=True):
        rows.append(
            {
                "subclass": keys[0],
                "module": keys[1],
                "model": keys[2],
                "analysis": keys[3],
                "n": len(frame),
                "n_groups": frame["group_id"].astype(str).nunique(),
                **regression_metrics(frame["y_true"], frame["y_pred"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["subclass", "module", "model"]).reset_index(drop=True)


def dominance_loo_score_sensitivity(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate flagged-gene LOO scores against unchanged saved OOF predictions.

    This is deliberately a score/evaluation sensitivity only: no model is fit,
    selected, or tuned here, and the primary targets and OOF artifacts are not
    modified.
    """
    dominance = pd.read_csv(root / "results/tables/module_dominance.csv")
    flagged = dominance.loc[dominance["dominance_flag"].astype(bool), ["module", "gene"]]
    cpm = pd.read_parquet(root / "data/interim/module_gene_cpm.parquet")
    cpm.index = cpm.index.astype(str)
    crosswalk = pd.read_csv(
        root / "results/tables/cell_id_crosswalk.csv",
        dtype={"canonical_cell_id": str, "transcriptomics_sample_id": str},
    )
    scores = pd.read_parquet(root / "results/tables/module_scores.parquet")
    scores["canonical_cell_id"] = scores["canonical_cell_id"].astype(str)
    oof = pd.read_parquet(root / "results/predictions/oof_predictions.parquet")
    oof["canonical_cell_id"] = oof["canonical_cell_id"].astype(str)

    mapping = crosswalk[["canonical_cell_id", "transcriptomics_sample_id"]].merge(
        scores, on="canonical_cell_id", validate="one_to_one"
    )
    if len(mapping) != len(scores) or not mapping["transcriptomics_sample_id"].isin(cpm.index).all():
        raise AssertionError("Flagged-gene LOO sensitivity cannot exactly align scores to CPM")
    log_cpm = np.log2(cpm + 1.0)
    score_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for record in flagged.itertuples(index=False):
        module, gene = str(record.module), str(record.gene)
        module_genes = dominance.loc[dominance["module"].eq(module), "gene"].astype(str).tolist()
        missing = sorted(set(module_genes).difference(log_cpm.columns))
        if missing:
            raise AssertionError(f"Dominance table genes absent from selected CPM for {module}: {missing}")
        loo_genes = [name for name in module_genes if name != gene]
        if gene not in module_genes or len(loo_genes) < 1:
            raise AssertionError(f"Invalid flagged gene {module}/{gene}")
        full_by_sample = log_cpm[module_genes].mean(axis=1)
        loo_by_sample = log_cpm[loo_genes].mean(axis=1)
        local = mapping[["canonical_cell_id", "transcriptomics_sample_id", module]].copy()
        local["recomputed_primary"] = local["transcriptomics_sample_id"].map(full_by_sample)
        local["loo_score"] = local["transcriptomics_sample_id"].map(loo_by_sample)
        if local[["recomputed_primary", "loo_score"]].isna().any().any():
            raise AssertionError(f"Missing LOO score after exact sample alignment for {module}/{gene}")
        primary = local[module].to_numpy(dtype=float)
        if not np.allclose(primary, local["recomputed_primary"], rtol=1e-12, atol=1e-12):
            raise AssertionError(f"Saved primary score cannot be reproduced from CPM for {module}")
        loo = local["loo_score"].to_numpy(dtype=float)
        difference = loo - primary
        score_rows.append(
            {
                "module": module,
                "omitted_gene": gene,
                "n": len(local),
                "module_genes_present": len(module_genes),
                "loo_genes": len(loo_genes),
                "primary_vs_loo_pearson": float(np.corrcoef(primary, loo)[0, 1]),
                "primary_vs_loo_spearman": float(pd.Series(primary).corr(pd.Series(loo), method="spearman")),
                "loo_minus_primary_mean": float(np.mean(difference)),
                "mean_absolute_score_difference": float(np.mean(np.abs(difference))),
                "rmse_score_difference": float(np.sqrt(np.mean(difference**2))),
                "primary_score_mean": float(np.mean(primary)),
                "primary_score_sd": float(np.std(primary, ddof=1)),
                "loo_score_mean": float(np.mean(loo)),
                "loo_score_sd": float(np.std(loo, ddof=1)),
                "models_refit": False,
            }
        )
        truth = local[["canonical_cell_id", "loo_score"]]
        predictions = oof.loc[
            oof["module"].eq(module) & oof["analysis"].eq("ephys_only")
        ].merge(truth, on="canonical_cell_id", validate="many_to_one")
        for model, frame in predictions.groupby("model", sort=True):
            primary_metrics = regression_metrics(frame["y_true"], frame["y_pred"])
            loo_metrics = regression_metrics(frame["loo_score"], frame["y_pred"])
            metric_rows.append(
                {
                    "module": module,
                    "omitted_gene": gene,
                    "model": model,
                    "analysis": "flagged_gene_loo_score_evaluation",
                    "n": len(frame),
                    **{f"primary_{key}": value for key, value in primary_metrics.items()},
                    **{f"loo_{key}": value for key, value in loo_metrics.items()},
                    **{
                        f"loo_minus_primary_{key}": loo_metrics[key] - primary_metrics[key]
                        for key in primary_metrics
                    },
                    "models_refit": False,
                    "interpretation": "unchanged primary-target OOF predictions evaluated against an alternative LOO score",
                }
            )
    return pd.DataFrame(score_rows), pd.DataFrame(metric_rows)


def finalize_saved_level0_validation(
    root: str | Path, *, measured_core_runtime_seconds: float
) -> dict[str, object]:
    """Audit a completed Level-0 run and repair only derived reporting tables."""
    root = Path(root)
    started = time.perf_counter()
    oof_path = root / "results/predictions/oof_predictions.parquet"
    oof_sha256_before = _sha256(oof_path)
    oof = pd.read_parquet(oof_path)
    table = pd.read_parquet(root / "data/processed/modeling_table.parquet")
    coverage = audit_oof_artifacts(root, oof, table)
    (root / "results/tables/oof_coverage_audit.json").write_text(
        json.dumps(coverage, indent=2) + "\n", encoding="utf-8"
    )

    intervals = pd.read_csv(root / "results/tables/confidence_intervals.csv")
    draws = pd.read_parquet(root / "results/tables/bootstrap_draws.parquet")
    permutations = pd.read_csv(root / "results/tables/permutation_tests.csv")
    baseline = pd.read_csv(root / "results/tables/baseline_comparison.csv")
    expected_grid = set(oof[["module", "model", "analysis"]].drop_duplicates().itertuples(index=False, name=None))
    interval_grid = set(intervals[["module", "model", "analysis"]].itertuples(index=False, name=None))
    draw_grid = set(draws[["module", "model", "analysis"]].drop_duplicates().itertuples(index=False, name=None))
    if interval_grid != expected_grid or draw_grid != expected_grid or len(intervals) != 36:
        raise AssertionError("Bootstrap output does not cover the exact 36-key OOF grid")
    if not intervals["n_bootstrap"].eq(1000).all():
        raise AssertionError("Confidence intervals are not Level-0 1000-resample results")
    draw_counts = draws.groupby(["module", "model", "analysis"])["replicate"].agg(["size", "nunique", "min", "max"])
    if not (
        draw_counts["size"].eq(1000).all()
        and draw_counts["nunique"].eq(1000).all()
        and draw_counts["min"].eq(0).all()
        and draw_counts["max"].eq(999).all()
    ):
        raise AssertionError("Bootstrap draws do not contain exactly replicates 0..999 per grid key")
    for keys, frame in draws.groupby(["module", "model", "analysis"], sort=True):
        row = intervals.loc[
            intervals["module"].eq(keys[0])
            & intervals["model"].eq(keys[1])
            & intervals["analysis"].eq(keys[2])
        ].iloc[0]
        expected_interval = summarize_bootstrap(frame)
        for name, value in expected_interval.items():
            if not np.isclose(row[name], value, rtol=1e-12, atol=1e-12, equal_nan=True):
                raise AssertionError(f"Saved bootstrap interval differs from draws for {keys}/{name}")
    expected_permutation_grid = set(
        oof.loc[oof["analysis"].eq("ephys_only"), ["module", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    permutation_grid = set(permutations[["module", "model"]].itertuples(index=False, name=None))
    if (
        permutation_grid != expected_permutation_grid
        or len(permutations) != 18
        or not permutations["n_permutations"].eq(5000).all()
        or not permutations["test_type"].str.contains("no model refitting", regex=False).all()
    ):
        raise AssertionError("Permutation tests are not the required 18 Level-0 no-refit tests")
    if not (
        permutations["p_value_two_sided"].between(1 / 5001 - 1e-12, 1 + 1e-12).all()
        and np.allclose(
            permutations["p_value_two_sided"] * 5001,
            np.round(permutations["p_value_two_sided"] * 5001),
            rtol=0,
            atol=1e-12,
        )
        and np.isfinite(permutations[["observed_spearman", "null_mean", "null_sd"]]).all().all()
    ):
        raise AssertionError("Permutation summaries violate finite plus-one Monte Carlo constraints")
    for record in permutations.itertuples(index=False):
        frame = oof.loc[
            oof["analysis"].eq("ephys_only")
            & oof["module"].eq(record.module)
            & oof["model"].eq(record.model)
        ]
        observed = regression_metrics(frame["y_true"], frame["y_pred"])["spearman"]
        if not np.isclose(observed, record.observed_spearman, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"Permutation observed statistic differs from OOF for {record.module}/{record.model}")
    donor_permutations = donor_aware_permutation_sensitivity(
        oof, n_permutations=5000, seed=42
    )
    if (
        len(donor_permutations) != 18
        or set(donor_permutations[["module", "model"]].itertuples(index=False, name=None))
        != expected_permutation_grid
        or not donor_permutations["n_permutations"].eq(5000).all()
        or not donor_permutations["n_groups"].eq(871).all()
        or not donor_permutations["n_cells"].eq(3410).all()
        or not donor_permutations["p_value_two_sided"].between(1 / 5001 - 1e-12, 1 + 1e-12).all()
        or not np.isfinite(
            donor_permutations[["observed_spearman", "null_mean", "null_sd"]]
        ).all().all()
    ):
        raise AssertionError("Donor-level permutation sensitivity failed coverage/finite checks")
    donor_permutations.to_csv(
        root / "results/tables/donor_level_permutation_sensitivity.csv", index=False
    )
    recalculated = performance_table(oof).sort_values(["analysis", "module", "model"]).reset_index(drop=True)
    saved_baseline = baseline.sort_values(["analysis", "module", "model"]).reset_index(drop=True)
    if not recalculated[["module", "model", "analysis", "n"]].equals(
        saved_baseline[["module", "model", "analysis", "n"]]
    ) or not np.allclose(
        recalculated[["r2", "mae", "rmse", "spearman"]],
        saved_baseline[["r2", "mae", "rmse", "spearman"]],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    ):
        raise AssertionError("Baseline/comparison table is not exactly reproducible from OOF")

    features = load_feature_names(root / "config/ephys_features.yaml")
    correlations = pd.read_csv(root / "results/tables/ephys_feature_correlations.csv", index_col=0)
    if correlations.shape != (len(features), len(features)):
        raise AssertionError("Ephys correlation matrix has the wrong dimensions")
    expected_correlations = table[features].corr(method="spearman")
    if not np.allclose(correlations.loc[features, features], expected_correlations, rtol=1e-12, atol=1e-12):
        raise AssertionError("Saved ephys correlation matrix is not reproducible")
    high_pairs = pd.read_csv(root / "results/tables/high_correlation_pairs.csv")
    high_pair_correlation_column = (
        "spearman_rho" if "spearman_rho" in high_pairs.columns else "spearman"
    )
    if not high_pairs[high_pair_correlation_column].abs().gt(0.9).all():
        raise AssertionError("High-correlation table contains a pair at or below the threshold")
    expected_high_pairs = {
        tuple(sorted((left, right)))
        for index, left in enumerate(features)
        for right in features[index + 1 :]
        if abs(float(expected_correlations.loc[left, right])) > 0.9
    }
    observed_high_pairs = {
        tuple(sorted((str(row.feature_1), str(row.feature_2))))
        for row in high_pairs.itertuples(index=False)
    }
    if observed_high_pairs != expected_high_pairs:
        raise AssertionError("High-correlation table is not the exact >0.90 pair set")
    for row in high_pairs.itertuples(index=False):
        saved_rho = float(getattr(row, high_pair_correlation_column))
        expected_rho = float(expected_correlations.loc[row.feature_1, row.feature_2])
        if not np.isclose(saved_rho, expected_rho, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"High-correlation value is not reproducible for {row.feature_1}/{row.feature_2}")

    within = pd.read_parquet(root / "results/predictions/within_subclass_oof.parquet")
    eligibility = pd.read_csv(root / "results/tables/within_subclass_eligibility.csv")
    selected = eligibility.loc[eligibility["eligible_and_selected"].astype(bool), "subclass"].astype(str).tolist()
    expected_selected: list[str] = []
    for subclass in table["subclass"].value_counts().index:
        subset = table.loc[table["subclass"].eq(subclass)]
        if len(subset) >= 150 and subset["group_id"].astype(str).nunique() >= 3:
            expected_selected.append(str(subclass))
        if len(expected_selected) == 2:
            break
    if selected != expected_selected:
        raise AssertionError(f"Selected subclasses {selected} differ from two largest eligible {expected_selected}")
    if not np.isfinite(within[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise AssertionError("Within-subclass predictions contain non-finite values")
    within_groups = within.assign(_group_id=within["group_id"].astype(str))
    if within_groups.groupby(["subclass", "_group_id"])["fold"].nunique().max() != 1:
        raise AssertionError("A donor crosses folds inside a within-subclass analysis")
    for subclass in selected:
        expected_cells = set(table.loc[table["subclass"].astype(str).eq(subclass), "canonical_cell_id"].astype(str))
        parts = within.loc[within["subclass"].astype(str).eq(subclass)].groupby(
            ["module", "model", "analysis"]
        )
        expected_within_grid = {
            (module, model, "within_subclass")
            for module in MODULES
            for model in ("ElasticNet", "XGBoost")
        }
        observed_within_grid = set(parts.groups)
        if observed_within_grid != expected_within_grid:
            raise AssertionError(f"Wrong within-subclass model grid for {subclass}")
        for keys, frame in parts:
            identifiers = frame["canonical_cell_id"].astype(str)
            if set(identifiers) != expected_cells or identifiers.duplicated().any():
                raise AssertionError(f"Incomplete within-subclass coverage for {subclass}, {keys}")
            target = table[["canonical_cell_id", f"target_{keys[0]}"]].copy()
            target["canonical_cell_id"] = target["canonical_cell_id"].astype(str)
            comparison = frame[["canonical_cell_id", "y_true"]].assign(
                canonical_cell_id=identifiers.to_numpy()
            ).merge(target, on="canonical_cell_id", validate="one_to_one")
            if not np.array_equal(
                comparison["y_true"].to_numpy(), comparison[f"target_{keys[0]}"].to_numpy()
            ):
                raise AssertionError(f"Within-subclass truth differs from target for {subclass}, {keys}")
    within_performance = within_subclass_performance(within)
    within_performance.to_csv(root / "results/tables/within_subclass_performance.csv", index=False)

    score_sensitivity, prediction_sensitivity = dominance_loo_score_sensitivity(root)
    score_sensitivity.to_csv(
        root / "results/tables/dominance_loo_score_sensitivity.csv", index=False
    )
    prediction_sensitivity.to_csv(
        root / "results/tables/dominance_loo_oof_sensitivity.csv", index=False
    )
    oof_sha256_after = _sha256(oof_path)
    if oof_sha256_before != oof_sha256_after:
        raise AssertionError("Primary OOF predictions changed during validation finalization")
    finalization_seconds = time.perf_counter() - started
    summary: dict[str, object] = {
        "status": "PASS",
        "level": 0,
        "mandatory_core_runtime_seconds": measured_core_runtime_seconds,
        "mandatory_core_runtime_measurement": "shell wall time for scripts/05_validate.py --level 0",
        "posthoc_audit_and_reporting_seconds": finalization_seconds,
        "bootstrap_resamples_per_grid": 1000,
        "bootstrap_interval_rows": len(intervals),
        "bootstrap_rows": len(draws),
        "permutations_per_test": 5000,
        "permutation_tests": len(permutations),
        "donor_level_permutation_sensitivity_tests": len(donor_permutations),
        "donor_level_permutation_groups": int(donor_permutations["n_groups"].iloc[0]),
        "baseline_comparison_rows": len(baseline),
        "high_correlation_pairs": len(high_pairs),
        "within_subclass_prediction_rows": len(within),
        "within_subclasses": selected,
        "within_subclass_performance_rows": len(within_performance),
        "dominance_flagged_genes": len(score_sensitivity),
        "dominance_loo_oof_evaluations": len(prediction_sensitivity),
        "dominance_sensitivity_models_refit": False,
        "oof_sha256_before": oof_sha256_before,
        "oof_sha256_after": oof_sha256_after,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (root / "logs/validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_validation(root: str | Path, *, level: int, seed: int = 42) -> dict[str, object]:
    root = Path(root)
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    phase_started = started
    timings: dict[str, float] = {}
    oof = pd.read_parquet(root / "results/predictions/oof_predictions.parquet")
    table = pd.read_parquet(root / "data/processed/modeling_table.parquet")
    coverage_audit = audit_oof_artifacts(root, oof, table)
    (root / "results/tables/oof_coverage_audit.json").write_text(
        json.dumps(coverage_audit, indent=2) + "\n", encoding="utf-8"
    )
    timings["oof_audit_seconds"] = time.perf_counter() - phase_started
    settings = {
        0: {"bootstrap": 1000, "permutations": 5000},
        1: {"bootstrap": 500, "permutations": 2000},
        2: {"bootstrap": 300, "permutations": 1000},
    }[level]
    intervals, draws, permutations = infer_from_oof(
        oof,
        n_bootstrap=settings["bootstrap"],
        n_permutations=settings["permutations"],
        seed=seed,
    )
    timings["bootstrap_and_permutation_seconds"] = time.perf_counter() - phase_started - timings[
        "oof_audit_seconds"
    ]
    intervals.to_csv(root / "results/tables/confidence_intervals.csv", index=False)
    draws.to_parquet(root / "results/tables/bootstrap_draws.parquet", index=False)
    permutations.to_csv(root / "results/tables/permutation_tests.csv", index=False)
    donor_permutations = donor_aware_permutation_sensitivity(
        oof, n_permutations=settings["permutations"], seed=seed
    )
    donor_permutations.to_csv(
        root / "results/tables/donor_level_permutation_sensitivity.csv", index=False
    )
    performance = performance_table(oof)
    if len(performance) != 36:
        raise AssertionError(f"Expected 36 baseline/comparison rows, observed {len(performance)}")
    performance.to_csv(root / "results/tables/baseline_comparison.csv", index=False)

    features = load_feature_names(root / "config/ephys_features.yaml")
    correlation = table[features].corr(method="spearman")
    correlation.to_csv(root / "results/tables/ephys_feature_correlations.csv")
    pairs = []
    for i, left in enumerate(features):
        for right in features[i + 1 :]:
            value = float(correlation.loc[left, right])
            if abs(value) > 0.9:
                pairs.append({"feature_1": left, "feature_2": right, "spearman": value})
    pd.DataFrame(pairs, columns=["feature_1", "feature_2", "spearman"]).to_csv(
        root / "results/tables/high_correlation_pairs.csv", index=False
    )
    timings["baseline_and_correlation_seconds"] = (
        time.perf_counter() - phase_started - timings["oof_audit_seconds"]
        - timings["bootstrap_and_permutation_seconds"]
    )

    within_started = time.perf_counter()
    within, eligibility = within_subclass_sensitivity(table, features, level=level, seed=seed)
    eligibility.to_csv(root / "results/tables/within_subclass_eligibility.csv", index=False)
    if len(within):
        within_groups = within.assign(_group_id=within["group_id"].astype(str))
        # These are separate within-subclass analyses. The same donor may occur
        # in both subclasses and receive different fold numbers, but must never
        # cross folds inside either analysis.
        if within_groups.groupby(["subclass", "_group_id"])["fold"].nunique().max() != 1:
            raise AssertionError("Within-subclass donor/group crosses sensitivity folds")
        selected = eligibility.loc[eligibility["eligible_and_selected"], "subclass"].astype(str).tolist()
        if len(selected) != 2:
            raise AssertionError(f"Expected two selected eligible subclasses, observed {selected}")
        for subclass in selected:
            expected = set(table.loc[table["subclass"].astype(str).eq(subclass), "canonical_cell_id"].astype(str))
            for keys, frame in within.loc[within["subclass"].astype(str).eq(subclass)].groupby(
                ["module", "model", "analysis"]
            ):
                observed = set(frame["canonical_cell_id"].astype(str))
                if observed != expected or frame["canonical_cell_id"].astype(str).duplicated().any():
                    raise AssertionError(f"Incomplete within-subclass OOF coverage for {subclass}, {keys}")
        within.to_parquet(root / "results/predictions/within_subclass_oof.parquet", index=False)
        within_subclass_performance(within).to_csv(
            root / "results/tables/within_subclass_performance.csv", index=False
        )
    timings["within_subclass_seconds"] = time.perf_counter() - within_started
    total_seconds = time.perf_counter() - started
    oof_sha256_after = _sha256(root / "results/predictions/oof_predictions.parquet")
    if oof_sha256_after != coverage_audit["oof_sha256"]:
        raise AssertionError("Primary OOF predictions changed during validation")
    summary: dict[str, object] = {
        "status": "PASS",
        "level": level,
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "total_seconds": total_seconds,
        "timings": timings,
        "bootstrap_resamples_per_grid": settings["bootstrap"],
        "bootstrap_interval_rows": len(intervals),
        "bootstrap_rows": len(draws),
        "permutations_per_test": settings["permutations"],
        "permutation_tests": len(permutations),
        "donor_level_permutation_sensitivity_tests": len(donor_permutations),
        "donor_level_permutation_groups": int(donor_permutations["n_groups"].iloc[0]),
        "baseline_comparison_rows": len(performance),
        "high_correlation_pairs": len(pairs),
        "within_subclass_prediction_rows": len(within),
        "within_subclasses": eligibility.loc[
            eligibility["eligible_and_selected"], "subclass"
        ].astype(str).tolist(),
        "oof_sha256_before": coverage_audit["oof_sha256"],
        "oof_sha256_after": oof_sha256_after,
    }
    (root / "logs/validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
