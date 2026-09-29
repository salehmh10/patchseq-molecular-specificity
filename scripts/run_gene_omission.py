"""True target-refit sensitivity for dominance-flagged NaV/HCN genes only."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import regression_metrics
from src.models import (
    dummy_model,
    elastic_net_candidates,
    select_on_inner_groups,
    subclass_ridge_model,
    xgboost_candidates,
)
from src.training import load_and_validate_inputs


SEED = 42
LEVEL = 0
from src.reproduction import reference_or_run_digest

PRIMARY_OOF_SHA256 = reference_or_run_digest(ROOT, "primary", "results/predictions/oof_predictions.parquet", "9d0fb9edb89b3133fd1e356cbc437ca7aa3b985df4f810805b0cce74b96a4217")
VARIANTS = (
    ("NaV", "Scn9a"),
    ("NaV", "Scn3a"),
    ("HCN", "Hcn2"),
    ("HCN", "Hcn3"),
)
MODEL_ANALYSES = (
    ("ElasticNet", "ephys_only"),
    ("XGBoost", "ephys_only"),
    ("Dummy", "dummy"),
    ("SubclassRidge", "subclass_only"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")
    temporary.replace(path)


def classify(delta_r2: float, delta_spearman: float, alternative_r2: float) -> tuple[str, str]:
    if alternative_r2 <= 0 or delta_r2 <= -0.10 or delta_spearman <= -0.10:
        label = "weakened"
    elif delta_r2 > 0.05 and delta_spearman >= -0.05:
        label = "stronger_after_omission"
    elif abs(delta_r2) <= 0.05 and abs(delta_spearman) <= 0.05:
        label = "stable"
    else:
        label = "mixed_or_partial"
    rationale = (
        f"alternative R2={alternative_r2:.6f}; delta R2={delta_r2:+.6f}; "
        f"delta Spearman={delta_spearman:+.6f}. Rules: weakened if alternative R2<=0 "
        "or either dimensionless metric drops by >=0.10; stronger if delta R2>0.05 "
        "without Spearman dropping >0.05; stable if both absolute deltas<=0.05; "
        "otherwise mixed_or_partial."
    )
    return label, rationale


def prediction_rows(
    test: pd.DataFrame,
    target: pd.Series,
    prediction: np.ndarray,
    primary_prediction: pd.Series,
    *,
    module: str,
    omitted_gene: str,
    genes_used: list[str],
    model: str,
    analysis: str,
) -> pd.DataFrame:
    identifiers = test.canonical_cell_id.astype(str)
    return pd.DataFrame(
        {
            "canonical_cell_id": identifiers.to_numpy(),
            "group_id": test.group_id.astype(str).to_numpy(),
            "subclass": test.subclass.astype(str).to_numpy(),
            "fold": test.fold.to_numpy(dtype=int),
            "module": module,
            "omitted_gene": omitted_gene,
            "target_name": f"{module}_minus_{omitted_gene}",
            "genes_used": "|".join(genes_used),
            "gene_count": len(genes_used),
            "model": model,
            "analysis": analysis,
            "y_true": identifiers.map(target).to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
            "primary_y_true": test[f"target_{module}"].to_numpy(dtype=float),
            "primary_y_pred": identifiers.map(primary_prediction).to_numpy(dtype=float),
        }
    )


def main() -> None:
    overall_started = time.perf_counter()
    started_at = datetime.now().astimezone().isoformat()
    output_root = ROOT / "results/robustness_v2"
    model_dir = output_root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    primary_oof_path = ROOT / "results/predictions/oof_predictions.parquet"
    before_hash = sha256(primary_oof_path)
    if before_hash != PRIMARY_OOF_SHA256:
        raise AssertionError(f"Primary OOF hash changed before R2: {before_hash}")

    table, features = load_and_validate_inputs(
        ROOT / "data/processed/modeling_table.parquet",
        ROOT / "results/tables/cv_folds.csv",
        ROOT / "config/ephys_features.yaml",
    )
    table["canonical_cell_id"] = table.canonical_cell_id.astype(str)
    crosswalk = pd.read_csv(
        ROOT / "results/tables/cell_id_crosswalk.csv",
        dtype={"canonical_cell_id": str, "transcriptomics_sample_id": str},
    )
    cpm = pd.read_parquet(ROOT / "data/interim/module_gene_cpm.parquet")
    cpm.index = cpm.index.astype(str)
    if not cpm.index.is_unique or not np.isfinite(cpm.to_numpy(dtype=float)).all():
        raise AssertionError("Gene CPM matrix index must be unique and values finite")
    if (cpm.to_numpy(dtype=float) < 0).any():
        raise AssertionError("Gene CPM values cannot be negative")
    if crosswalk.canonical_cell_id.duplicated().any() or crosswalk.transcriptomics_sample_id.duplicated().any():
        raise AssertionError("Crosswalk identifiers must be one-to-one")
    if set(table.canonical_cell_id) != set(crosswalk.canonical_cell_id):
        raise AssertionError("Crosswalk/modeling cell sets differ")
    sample_by_cell = crosswalk.set_index("canonical_cell_id").transcriptomics_sample_id
    ordered_samples = table.canonical_cell_id.map(sample_by_cell)
    if ordered_samples.isna().any() or not set(ordered_samples).issubset(set(cpm.index)):
        raise AssertionError("Every modeling cell must map to one gene-CPM row")

    module_config = yaml.safe_load((ROOT / "config/gene_modules.yaml").read_text(encoding="utf-8"))
    dominance = pd.read_csv(ROOT / "results/tables/module_dominance.csv")
    observed_flags = set(
        dominance.loc[dominance.dominance_flag.astype(bool), ["module", "gene"]]
        .itertuples(index=False, name=None)
    )
    if observed_flags != set(VARIANTS):
        raise AssertionError(f"Dominance flags changed: {sorted(observed_flags)}")

    target_by_variant: dict[tuple[str, str], pd.Series] = {}
    target_audit: list[dict[str, object]] = []
    for module, omitted_gene in VARIANTS:
        requested = list(module_config["modules"][module]["genes"])
        present = [gene for gene in requested if gene in cpm.columns]
        missing = [gene for gene in requested if gene not in cpm.columns]
        if omitted_gene not in present:
            raise AssertionError(f"Flagged gene {omitted_gene} is absent from CPM")
        full_score_by_sample = np.log2(cpm[present] + 1.0).mean(axis=1)
        genes_used = [gene for gene in present if gene != omitted_gene]
        loo_score_by_sample = np.log2(cpm[genes_used] + 1.0).mean(axis=1)
        identifiers = table.canonical_cell_id.astype(str)
        full_score = pd.Series(
            ordered_samples.map(full_score_by_sample).to_numpy(dtype=float),
            index=identifiers,
        )
        loo_score = pd.Series(
            ordered_samples.map(loo_score_by_sample).to_numpy(dtype=float),
            index=identifiers,
        )
        primary = pd.Series(table[f"target_{module}"].to_numpy(dtype=float), index=identifiers)
        max_error = float(np.max(np.abs(full_score.to_numpy() - primary.to_numpy())))
        if max_error > 1e-12:
            raise AssertionError(f"Primary {module} target reconstruction error {max_error}")
        if not np.isfinite(loo_score.to_numpy()).all() or np.array_equal(
            loo_score.to_numpy(), primary.to_numpy()
        ):
            raise AssertionError("Alternative target must be finite and genuinely new")
        target_by_variant[(module, omitted_gene)] = loo_score
        target_audit.append(
            {
                "module": module,
                "omitted_gene": omitted_gene,
                "target_name": f"{module}_minus_{omitted_gene}",
                "requested_gene_count": len(requested),
                "present_primary_gene_count": len(present),
                "new_gene_count": len(genes_used),
                "genes_used": genes_used,
                "missing_primary_genes": missing,
                "primary_target_mean": float(primary.mean()),
                "primary_target_variance": float(primary.var(ddof=1)),
                "alternative_target_mean": float(loo_score.mean()),
                "alternative_target_variance": float(loo_score.var(ddof=1)),
                "alternative_target_sd": float(loo_score.std(ddof=1)),
                "primary_alternative_spearman": float(spearmanr(primary, loo_score).statistic),
                "primary_reconstruction_max_abs_error": max_error,
                "cells_different_from_primary": int(
                    (~np.isclose(loo_score.to_numpy(), primary.to_numpy(), rtol=0, atol=1e-12)).sum()
                ),
            }
        )

    primary_oof = pd.read_parquet(primary_oof_path)
    primary_performance = pd.read_csv(ROOT / "results/tables/model_performance.csv")
    prediction_parts: list[pd.DataFrame] = []
    selected_parameters: list[dict[str, object]] = []
    fold_timings: list[dict[str, object]] = []
    variant_timings: dict[str, float] = {}

    for module, omitted_gene in VARIANTS:
        variant_started = time.perf_counter()
        target_name = f"{module}_minus_{omitted_gene}"
        target = target_by_variant[(module, omitted_gene)]
        genes_used = next(
            row["genes_used"]
            for row in target_audit
            if row["module"] == module and row["omitted_gene"] == omitted_gene
        )
        for fold in (0, 1, 2):
            fold_started = time.perf_counter()
            train = table.loc[table.fold.ne(fold)].copy()
            test = table.loc[table.fold.eq(fold)].copy()
            train_groups = set(train.group_id.astype(str))
            test_groups = set(test.group_id.astype(str))
            if train_groups.intersection(test_groups):
                raise AssertionError(f"Donor leakage for {target_name} fold {fold}")
            X_train, X_test = train[features], test[features]
            y_train = train.canonical_cell_id.astype(str).map(target).to_numpy(dtype=float)
            groups = train.group_id.astype(str).to_numpy()

            elastic = select_on_inner_groups(
                elastic_net_candidates(features, seed=SEED),
                X_train,
                y_train,
                groups,
                seed=SEED + fold,
            )
            elastic_path = model_dir / f"{target_name}_fold{fold}_ElasticNet.joblib"
            joblib.dump(elastic.pipeline, elastic_path)
            selected_parameters.append(
                {
                    "target_name": target_name,
                    "module": module,
                    "omitted_gene": omitted_gene,
                    "fold": fold,
                    "model": "ElasticNet",
                    "inner_rmse": elastic.inner_rmse,
                    **elastic.parameters,
                }
            )

            boosted = select_on_inner_groups(
                xgboost_candidates(features, seed=SEED, level=LEVEL),
                X_train,
                y_train,
                groups,
                seed=SEED + fold,
            )
            boosted_path = model_dir / f"{target_name}_fold{fold}_XGBoost.joblib"
            joblib.dump(boosted.pipeline, boosted_path)
            selected_parameters.append(
                {
                    "target_name": target_name,
                    "module": module,
                    "omitted_gene": omitted_gene,
                    "fold": fold,
                    "model": "XGBoost",
                    "inner_rmse": boosted.inner_rmse,
                    **boosted.parameters,
                }
            )

            dummy = dummy_model(features).fit(X_train, y_train)
            dummy_path = model_dir / f"{target_name}_fold{fold}_Dummy.joblib"
            joblib.dump(dummy, dummy_path)
            subclass = subclass_ridge_model().fit(train[["subclass"]], y_train)
            subclass_path = model_dir / f"{target_name}_fold{fold}_SubclassRidge.joblib"
            joblib.dump(subclass, subclass_path)

            fitted = (
                ("ElasticNet", "ephys_only", elastic.pipeline.predict(X_test)),
                ("XGBoost", "ephys_only", boosted.pipeline.predict(X_test)),
                ("Dummy", "dummy", dummy.predict(X_test)),
                (
                    "SubclassRidge",
                    "subclass_only",
                    subclass.predict(test[["subclass"]]),
                ),
            )
            for model, analysis, prediction in fitted:
                primary_frame = primary_oof.loc[
                    primary_oof.module.eq(module)
                    & primary_oof.model.eq(model)
                    & primary_oof.analysis.eq(analysis)
                ]
                primary_prediction = primary_frame.set_index(
                    primary_frame.canonical_cell_id.astype(str)
                ).y_pred
                part = prediction_rows(
                    test,
                    target,
                    prediction,
                    primary_prediction,
                    module=module,
                    omitted_gene=omitted_gene,
                    genes_used=genes_used,
                    model=model,
                    analysis=analysis,
                )
                prediction_parts.append(part)
            fold_timings.append(
                {
                    "target_name": target_name,
                    "fold": fold,
                    "elapsed_seconds": time.perf_counter() - fold_started,
                }
            )
        variant_timings[target_name] = time.perf_counter() - variant_started
        print(f"{target_name}: {variant_timings[target_name]:.6f}s", flush=True)

    oof = pd.concat(prediction_parts, ignore_index=True)
    expected_cells = set(table.canonical_cell_id.astype(str))
    expected_keys = {
        (f"{module}_minus_{gene}", model, analysis)
        for module, gene in VARIANTS
        for model, analysis in MODEL_ANALYSES
    }
    observed_keys = set(
        oof[["target_name", "model", "analysis"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        raise AssertionError("Alternative-target model grid is incomplete")
    if oof.duplicated(["canonical_cell_id", "target_name", "model", "analysis"]).any():
        raise AssertionError("Duplicate alternative-target OOF prediction")
    if not np.isfinite(oof[["y_true", "y_pred", "primary_y_true", "primary_y_pred"]]).all().all():
        raise AssertionError("Alternative OOF values must be finite")
    frozen_fold = table.set_index("canonical_cell_id").fold.astype(int)
    fold_mismatch_count = int(
        (
            oof.canonical_cell_id.astype(str).map(frozen_fold).to_numpy(dtype=int)
            != oof.fold.to_numpy(dtype=int)
        ).sum()
    )
    if fold_mismatch_count:
        raise AssertionError("Alternative OOF fold mismatch")
    prediction_difference_audit: list[dict[str, object]] = []
    for key in sorted(expected_keys):
        frame = oof.loc[
            oof.target_name.eq(key[0]) & oof.model.eq(key[1]) & oof.analysis.eq(key[2])
        ]
        if set(frame.canonical_cell_id.astype(str)) != expected_cells or len(frame) != len(table):
            raise AssertionError(f"Incomplete exact-once coverage for {key}")
        different = ~np.isclose(
            frame.y_pred.to_numpy(dtype=float),
            frame.primary_y_pred.to_numpy(dtype=float),
            rtol=0,
            atol=1e-12,
        )
        if not different.any():
            raise AssertionError(f"Refit predictions were not new for {key}")
        prediction_difference_audit.append(
            {
                "target_name": key[0],
                "model": key[1],
                "analysis": key[2],
                "cells_with_new_prediction": int(different.sum()),
                "max_abs_prediction_change": float(
                    np.max(np.abs(frame.y_pred.to_numpy() - frame.primary_y_pred.to_numpy()))
                ),
            }
        )

    target_audit_by_name = {row["target_name"]: row for row in target_audit}
    performance_rows: list[dict[str, object]] = []
    for (target_name, model, analysis), frame in oof.groupby(
        ["target_name", "model", "analysis"], sort=True
    ):
        module = str(frame.module.iloc[0])
        omitted_gene = str(frame.omitted_gene.iloc[0])
        metrics = regression_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())
        primary_row = primary_performance.loc[
            primary_performance.module.eq(module)
            & primary_performance.model.eq(model)
            & primary_performance.analysis.eq(analysis)
        ].iloc[0]
        fold_values = [
            regression_metrics(part.y_true.to_numpy(), part.y_pred.to_numpy())
            for _, part in frame.groupby("fold", sort=True)
        ]
        delta_r2 = metrics["r2"] - float(primary_row.r2)
        delta_spearman = metrics["spearman"] - float(primary_row.spearman)
        if model == "Dummy":
            qualitative = "baseline_reference"
            rationale = (
                "Dummy is retained only as the mandatory fold-wise mean reference; "
                f"alternative R2={metrics['r2']:.6f}, delta R2={delta_r2:+.6f}."
            )
        else:
            qualitative, rationale = classify(delta_r2, delta_spearman, metrics["r2"])
        audit = target_audit_by_name[target_name]
        performance_rows.append(
            {
                "module": module,
                "omitted_gene": omitted_gene,
                "target_name": target_name,
                "genes_used": "|".join(audit["genes_used"]),
                "new_gene_count": audit["new_gene_count"],
                "n": len(frame),
                "target_mean": audit["alternative_target_mean"],
                "target_variance": audit["alternative_target_variance"],
                "target_sd": audit["alternative_target_sd"],
                "primary_target_variance": audit["primary_target_variance"],
                "target_primary_spearman": audit["primary_alternative_spearman"],
                "model": model,
                "analysis": analysis,
                **metrics,
                "fold_r2_min": min(value["r2"] for value in fold_values),
                "fold_r2_max": max(value["r2"] for value in fold_values),
                "fold_spearman_min": min(value["spearman"] for value in fold_values),
                "fold_spearman_max": max(value["spearman"] for value in fold_values),
                "primary_r2": float(primary_row.r2),
                "primary_mae": float(primary_row.mae),
                "primary_rmse": float(primary_row.rmse),
                "primary_spearman": float(primary_row.spearman),
                "delta_r2": delta_r2,
                "delta_mae": metrics["mae"] - float(primary_row.mae),
                "delta_rmse": metrics["rmse"] - float(primary_row.rmse),
                "delta_spearman": delta_spearman,
                "qualitative_classification": qualitative,
                "classification_rationale": rationale,
            }
        )
    performance = pd.DataFrame(performance_rows)

    target_conclusions: list[dict[str, object]] = []
    for target_name, frame in performance.loc[performance.analysis.eq("ephys_only")].groupby(
        "target_name", sort=True
    ):
        module = str(frame.module.iloc[0])
        alt_best = frame.sort_values(["r2", "spearman"], ascending=False).iloc[0]
        primary_ephys = primary_performance.loc[
            primary_performance.module.eq(module)
            & primary_performance.analysis.eq("ephys_only")
            & primary_performance.model.isin(["ElasticNet", "XGBoost"])
        ].sort_values(["r2", "spearman"], ascending=False)
        primary_best = primary_ephys.iloc[0]
        delta_r2 = float(alt_best.r2 - primary_best.r2)
        delta_spearman = float(alt_best.spearman - primary_best.spearman)
        classification, rationale = classify(delta_r2, delta_spearman, float(alt_best.r2))
        rationale = (
            f"Best alternative ephys model={alt_best.model}; best primary ephys model="
            f"{primary_best.model}. " + rationale
        )
        target_conclusions.append(
            {
                "target_name": target_name,
                "classification": classification,
                "best_alternative_model": alt_best.model,
                "best_alternative_r2": float(alt_best.r2),
                "best_alternative_spearman": float(alt_best.spearman),
                "best_primary_model": primary_best.model,
                "best_primary_r2": float(primary_best.r2),
                "best_primary_spearman": float(primary_best.spearman),
                "delta_best_r2": delta_r2,
                "delta_best_spearman": delta_spearman,
                "rationale": rationale,
            }
        )
    conclusion_by_target = {row["target_name"]: row for row in target_conclusions}
    performance["target_level_classification"] = performance.target_name.map(
        lambda name: conclusion_by_target[name]["classification"]
    )
    performance["target_level_rationale"] = performance.target_name.map(
        lambda name: conclusion_by_target[name]["rationale"]
    )
    performance = performance.sort_values(
        ["module", "omitted_gene", "analysis", "model"]
    ).reset_index(drop=True)
    oof = oof.sort_values(
        ["module", "omitted_gene", "analysis", "model", "fold", "canonical_cell_id"]
    ).reset_index(drop=True)

    prediction_path = output_root / "predictions/module_loo_oof.parquet"
    performance_path = output_root / "tables/module_loo_refit_performance.csv"
    atomic_parquet(oof, prediction_path)
    atomic_csv(performance, performance_path)

    after_hash = sha256(primary_oof_path)
    if after_hash != before_hash or after_hash != PRIMARY_OOF_SHA256:
        raise AssertionError("Primary OOF artifact changed during R2")
    elapsed = time.perf_counter() - overall_started
    if elapsed > 12 * 60:
        raise AssertionError(f"R2 runtime exceeded 12-minute target: {elapsed:.3f}s")
    log = {
        "status": "complete",
        "started_at": started_at,
        "completed_at": datetime.now().astimezone().isoformat(),
        "elapsed_seconds": elapsed,
        "runtime_target_seconds": 720,
        "seed": SEED,
        "runtime_level": LEVEL,
        "selection_policy": (
            "Fresh minimal Level-0 tuning for each alternative target using the same one "
            "donor-aware inner split (seed 42+outer fold) and the same frozen ElasticNet/XGBoost "
            "candidate grids as primary training. This is preferable to carrying primary-target "
            "hyperparameters into materially changed targets and never inspects outer-test data."
        ),
        "variants": [f"{module}_minus_{gene}" for module, gene in VARIANTS],
        "dominance_flags_verified": sorted([list(item) for item in observed_flags]),
        "target_audit": target_audit,
        "target_conclusions": target_conclusions,
        "selected_hyperparameters": selected_parameters,
        "variant_elapsed_seconds": variant_timings,
        "fold_timings": fold_timings,
        "prediction_difference_audit": prediction_difference_audit,
        "verification": {
            "expected_oof_rows": len(table) * len(VARIANTS) * len(MODEL_ANALYSES),
            "observed_oof_rows": len(oof),
            "expected_model_target_keys": len(expected_keys),
            "observed_model_target_keys": len(observed_keys),
            "duplicate_cell_target_model_rows": int(
                oof.duplicated(["canonical_cell_id", "target_name", "model", "analysis"]).sum()
            ),
            "fold_mismatch_count": fold_mismatch_count,
            "outer_donor_overlap_count_each_fold": {"0": 0, "1": 0, "2": 0},
            "nonfinite_prediction_or_target_count": int(
                (~np.isfinite(oof[["y_true", "y_pred"]].to_numpy(dtype=float))).sum()
            ),
            "primary_oof_sha256_before": before_hash,
            "primary_oof_sha256_after": after_hash,
            "fitted_model_count": len(list(model_dir.glob("*.joblib"))),
        },
        "outputs": {
            "predictions": str(prediction_path.relative_to(ROOT)),
            "performance": str(performance_path.relative_to(ROOT)),
            "prediction_sha256": sha256(prediction_path),
            "performance_sha256": sha256(performance_path),
        },
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    atomic_json(log, output_root / "logs/module_loo_refit_audit.json")
    print(json.dumps(jsonable({"elapsed_seconds": elapsed, "target_conclusions": target_conclusions}), indent=2))


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        main()
