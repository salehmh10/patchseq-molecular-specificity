"""Robustness V2 R3: duplicate removal and correlated-feature SHAP grouping.

This script writes only under ``results/robustness_v2`` and never mutates the
frozen primary folds, predictions, SHAP values, models, or manuscript.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import fold_metrics, regression_metrics  # noqa: E402
from src.interpretability import heldout_tree_shap, shap_stability  # noqa: E402
from src.preprocessing import make_preprocessor  # noqa: E402


MODULES = ("NaV", "Kv", "CaV", "HCN", "GABAA", "iGluR")
KEEP_FEATURE = "rheobase_i"
REMOVE_FEATURE = "stimulus_amplitude_0_long_square"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def feature_names() -> list[str]:
    payload = yaml.safe_load((ROOT / "config/ephys_features.yaml").read_text(encoding="utf-8"))
    return [str(record["name"]) for record in payload["feature_set"]["features"]]


def primary_hashes() -> dict[str, str]:
    paths = {
        "modeling_table": ROOT / "data/processed/modeling_table.parquet",
        "cv_folds": ROOT / "results/tables/cv_folds.csv",
        "primary_oof": ROOT / "results/predictions/oof_predictions.parquet",
        "primary_shap": ROOT / "results/shap/oof_shap.parquet",
        "primary_hyperparameters": ROOT / "results/tables/fitted_hyperparameters.csv",
        "manuscript": ROOT / "MANUSCRIPT.md",
    }
    return {name: sha256(path) for name, path in paths.items()}


class UnionFind:
    def __init__(self, values: list[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def correlated_components(features: list[str], pairs: pd.DataFrame) -> list[list[str]]:
    union = UnionFind(features)
    for row in pairs.itertuples(index=False):
        rho = float(getattr(row, "abs_spearman_rho", abs(getattr(row, "spearman_rho"))))
        if rho > 0.90:
            union.union(str(row.feature_1), str(row.feature_2))
    components: dict[str, list[str]] = {}
    for feature in features:
        components.setdefault(union.find(feature), []).append(feature)
    order = {feature: index for index, feature in enumerate(features)}
    return sorted(
        (sorted(members, key=order.get) for members in components.values()),
        key=lambda members: min(order[member] for member in members),
    )


def main() -> None:
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    output = ROOT / "results/robustness_v2"
    output.mkdir(parents=True, exist_ok=True)
    tables_output = output / "tables"
    predictions_output = output / "predictions"
    shap_output = output / "shap"
    # The shared robustness root is pre-created with these four coordinated
    # subdirectories. Keep R3 inside them to avoid colliding with other workers.
    figures_output = shap_output
    logs_output = tables_output
    for directory in (tables_output, predictions_output, shap_output):
        directory.mkdir(parents=True, exist_ok=True)
    hashes_before = primary_hashes()

    features = feature_names()
    if len(features) != 24 or KEEP_FEATURE not in features or REMOVE_FEATURE not in features:
        raise AssertionError("Frozen 24-feature set or duplicate pair is unexpected")
    reduced_features = [feature for feature in features if feature != REMOVE_FEATURE]
    table = pd.read_parquet(ROOT / "data/processed/modeling_table.parquet")
    folds = pd.read_csv(ROOT / "results/tables/cv_folds.csv")
    table["canonical_cell_id"] = table["canonical_cell_id"].astype(str)
    folds["canonical_cell_id"] = folds["canonical_cell_id"].astype(str)
    if table.canonical_cell_id.duplicated().any() or folds.canonical_cell_id.duplicated().any():
        raise AssertionError("Duplicate canonical IDs")
    check = table[["canonical_cell_id", "group_id", "subclass"]].merge(
        folds, on="canonical_cell_id", validate="one_to_one", suffixes=("_table", "_fold")
    )
    if len(check) != len(table) or not (
        check.group_id_table.astype(str).eq(check.group_id_fold.astype(str)).all()
        and check.subclass_table.astype(str).eq(check.subclass_fold.astype(str)).all()
    ):
        raise AssertionError("Frozen fold identity does not match modeling table")
    table = table.merge(folds[["canonical_cell_id", "fold"]], on="canonical_cell_id", validate="one_to_one")
    if table.groupby(table.group_id.astype(str)).fold.nunique().max() != 1:
        raise AssertionError("Donor crosses frozen folds")

    pairs = pd.read_csv(ROOT / "results/tables/high_correlation_pairs.csv")
    duplicate = pairs.loc[
        (
            pairs[["feature_1", "feature_2"]].astype(str).apply(set, axis=1)
            == {KEEP_FEATURE, REMOVE_FEATURE}
        )
        & pairs["exact_duplicate_on_complete_cases"].astype(bool)
    ]
    if len(duplicate) != 1 or not np.isclose(float(duplicate.iloc[0].abs_spearman_rho), 1.0):
        raise AssertionError("Expected exact duplicate pair was not uniquely verified")
    left = table[KEEP_FEATURE].to_numpy(dtype=float)
    right = table[REMOVE_FEATURE].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(left), np.isnan(right)) or not np.allclose(
        left[np.isfinite(left)], right[np.isfinite(right)], rtol=0, atol=0
    ):
        raise AssertionError("Duplicate columns are not exactly equal including missingness")

    parameters = pd.read_csv(ROOT / "results/tables/fitted_hyperparameters.csv")
    parameters = parameters.loc[parameters.model.eq("XGBoost")].copy()
    if len(parameters) != 18 or parameters.duplicated(["module", "fold"]).any():
        raise AssertionError("Expected exactly one saved XGBoost parameter row per module/fold")

    fit_started = time.perf_counter()
    prediction_parts: list[pd.DataFrame] = []
    shap_parts: list[pd.DataFrame] = []
    fit_rows: list[dict[str, object]] = []
    for module in MODULES:
        for fold in (0, 1, 2):
            fold_started = time.perf_counter()
            train = table.loc[table.fold.ne(fold)]
            test = table.loc[table.fold.eq(fold)]
            row = parameters.loc[parameters.module.eq(module) & parameters.fold.eq(fold)].iloc[0]
            selected = {
                "n_estimators": int(row.n_estimators),
                "max_depth": int(row.max_depth),
                "learning_rate": float(row.learning_rate),
                "reg_lambda": float(row.reg_lambda),
            }
            estimator = XGBRegressor(
                **selected,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=1,
                random_state=42,
                tree_method="hist",
            )
            pipeline = Pipeline(
                [
                    ("preprocessor", make_preprocessor(reduced_features, scale_numeric=False)),
                    ("estimator", estimator),
                ]
            )
            target = f"target_{module}"
            pipeline.fit(train[reduced_features], train[target].to_numpy(dtype=float))
            prediction = pipeline.predict(test[reduced_features])
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "canonical_cell_id": test.canonical_cell_id.to_numpy(),
                        "group_id": test.group_id.astype(str).to_numpy(),
                        "subclass": test.subclass.astype(str).to_numpy(),
                        "fold": fold,
                        "module": module,
                        "model": "XGBoost",
                        "analysis": "duplicate_removed",
                        "y_true": test[target].to_numpy(dtype=float),
                        "y_pred": prediction,
                    }
                )
            )
            shap_parts.append(
                heldout_tree_shap(
                    pipeline,
                    test[reduced_features],
                    cell_ids=test.canonical_cell_id,
                    module=module,
                    fold=fold,
                )
            )
            fit_rows.append(
                {
                    "module": module,
                    "fold": fold,
                    **selected,
                    "train_n": len(train),
                    "test_n": len(test),
                    "elapsed_seconds": time.perf_counter() - fold_started,
                    "parameter_policy": "reuse selected primary XGBoost hyperparameters; no retuning",
                }
            )
    refit_seconds = time.perf_counter() - fit_started

    oof = pd.concat(prediction_parts, ignore_index=True)
    duplicate_shap = pd.concat(shap_parts, ignore_index=True)
    expected_cells = set(table.canonical_cell_id)
    if len(oof) != len(table) * len(MODULES) or oof.duplicated(["canonical_cell_id", "module"]).any():
        raise AssertionError("Duplicate-removed OOF has incorrect shape or duplicates")
    if any(set(frame.canonical_cell_id) != expected_cells for _, frame in oof.groupby("module")):
        raise AssertionError("Duplicate-removed OOF cell coverage is incomplete")
    frozen_fold = table.set_index("canonical_cell_id").fold
    if not np.array_equal(
        oof.canonical_cell_id.map(frozen_fold).to_numpy(dtype=int), oof.fold.to_numpy(dtype=int)
    ):
        raise AssertionError("Duplicate-removed OOF does not use exact frozen folds")
    if REMOVE_FEATURE in duplicate_shap or set(duplicate_shap.columns) != {
        "canonical_cell_id", "module", "fold", *reduced_features
    }:
        raise AssertionError("Removed feature is present or SHAP feature set is incorrect")
    if len(duplicate_shap) != len(table) * len(MODULES) or duplicate_shap.duplicated(
        ["canonical_cell_id", "module"]
    ).any():
        raise AssertionError("Duplicate-removed held-out SHAP coverage is incorrect")
    if not np.array_equal(
        duplicate_shap.canonical_cell_id.map(frozen_fold).to_numpy(dtype=int),
        duplicate_shap.fold.to_numpy(dtype=int),
    ):
        raise AssertionError("Duplicate-removed SHAP does not use held-out frozen folds")

    primary_oof = pd.read_parquet(ROOT / "results/predictions/oof_predictions.parquet")
    primary_perf = primary_oof.loc[
        primary_oof.model.eq("XGBoost") & primary_oof.analysis.eq("ephys_only")
    ].groupby("module", sort=True).apply(
        lambda frame: pd.Series(regression_metrics(frame.y_true, frame.y_pred)), include_groups=False
    )
    performance_rows = []
    for module, frame in oof.groupby("module", sort=True):
        values = regression_metrics(frame.y_true, frame.y_pred)
        record: dict[str, object] = {
            "module": module,
            "model": "XGBoost",
            "analysis": "duplicate_removed",
            "n": len(frame),
            **values,
        }
        for metric in ("r2", "mae", "rmse", "spearman"):
            record[f"primary_{metric}"] = float(primary_perf.loc[module, metric])
            record[f"delta_{metric}"] = float(values[metric] - primary_perf.loc[module, metric])
        performance_rows.append(record)
    performance = pd.DataFrame(performance_rows).sort_values("module").reset_index(drop=True)
    duplicate_stability = shap_stability(duplicate_shap)

    grouped_started = time.perf_counter()
    components = correlated_components(features, pairs)
    component_rows = []
    for index, members in enumerate(components, start=1):
        component_rows.extend(
            {
                "group_id": f"G{index:02d}",
                "feature": feature,
                "n_features": len(members),
                "members": " | ".join(members),
            }
            for feature in members
        )
    component_table = pd.DataFrame(component_rows)
    if set(component_table.feature) != set(features) or component_table.feature.duplicated().any():
        raise AssertionError("Connected components do not partition the frozen features")

    primary_shap = pd.read_parquet(ROOT / "results/shap/oof_shap.parquet")
    grouped_wide = primary_shap[["canonical_cell_id", "module", "fold"]].copy()
    for group_id, frame in component_table.groupby("group_id", sort=True):
        members = frame.feature.tolist()
        grouped_wide[group_id] = primary_shap[members].abs().sum(axis=1)
    group_ids = sorted(component_table.group_id.unique())
    individual_total = primary_shap[features].abs().sum(axis=1).to_numpy(dtype=float)
    grouped_total = grouped_wide[group_ids].sum(axis=1).to_numpy(dtype=float)
    reconciliation_difference = np.abs(individual_total - grouped_total)
    reconciliation_max = float(reconciliation_difference.max())
    if not np.allclose(individual_total, grouped_total, rtol=1e-12, atol=1e-12):
        raise AssertionError("Grouped absolute SHAP does not reconcile to individual absolute SHAP")
    reconciliation = pd.DataFrame(
        {
            "module": primary_shap.module.astype(str),
            "abs_difference": reconciliation_difference,
        }
    ).groupby("module", as_index=False).agg(
        n_cell_module_rows=("abs_difference", "size"),
        max_abs_difference=("abs_difference", "max"),
        mean_abs_difference=("abs_difference", "mean"),
        sum_abs_difference=("abs_difference", "sum"),
    )
    reconciliation = pd.concat(
        [
            reconciliation,
            pd.DataFrame(
                [
                    {
                        "module": "ALL",
                        "n_cell_module_rows": len(reconciliation_difference),
                        "max_abs_difference": reconciliation_max,
                        "mean_abs_difference": float(reconciliation_difference.mean()),
                        "sum_abs_difference": float(reconciliation_difference.sum()),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    summary_rows: list[dict[str, object]] = []
    member_lookup = component_table.groupby("group_id").agg(
        n_features=("feature", "size"), members=("members", "first")
    )
    for module, module_frame in grouped_wide.groupby("module", sort=True):
        fold_means = module_frame.groupby("fold")[group_ids].mean()
        overall = module_frame[group_ids].mean()
        ranks = fold_means.rank(axis=1, ascending=False, method="min")
        for group_id in group_ids:
            summary_rows.append(
                {
                    "module": module,
                    "group_id": group_id,
                    "n_features": int(member_lookup.loc[group_id, "n_features"]),
                    "members": member_lookup.loc[group_id, "members"],
                    "mean_sum_abs_shap": float(overall[group_id]),
                    "fold0_mean_sum_abs_shap": float(fold_means.loc[0, group_id]),
                    "fold1_mean_sum_abs_shap": float(fold_means.loc[1, group_id]),
                    "fold2_mean_sum_abs_shap": float(fold_means.loc[2, group_id]),
                    "mean_fold_rank": float(ranks[group_id].mean()),
                    "top5_fold_frequency": float((ranks[group_id] <= 5).mean()),
                }
            )
    grouped_importance = pd.DataFrame(summary_rows)
    grouped_importance["overall_rank"] = grouped_importance.groupby("module")[
        "mean_sum_abs_shap"
    ].rank(ascending=False, method="min")
    grouped_importance = grouped_importance.sort_values(["module", "overall_rank", "group_id"])

    heatmap = grouped_importance.pivot(index="group_id", columns="module", values="mean_sum_abs_shap")
    heatmap = heatmap.reindex(columns=MODULES)
    order = heatmap.max(axis=1).sort_values(ascending=False).index
    heatmap = heatmap.loc[order]
    labels = []
    for group_id in heatmap.index:
        members = str(member_lookup.loc[group_id, "members"]).split(" | ")
        labels.append(group_id + ": " + (members[0] if len(members) == 1 else f"{members[0]} +{len(members)-1}"))
    fig, ax = plt.subplots(figsize=(8.4, 8.0))
    image = ax.imshow(heatmap.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(heatmap.columns)), heatmap.columns)
    ax.set_yticks(range(len(labels)), labels, fontsize=7)
    ax.set_xlabel("Molecular module")
    ax.set_ylabel("Correlated-feature component")
    ax.set_title("Primary held-out SHAP grouped by |Spearman rho| > 0.90 components")
    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Mean per-cell sum(|SHAP|)")
    fig.tight_layout()
    fig.savefig(figures_output / "grouped_shap_heatmap.png", dpi=300)
    fig.savefig(figures_output / "grouped_shap_heatmap.pdf")
    plt.close(fig)
    grouped_seconds = time.perf_counter() - grouped_started

    atomic_parquet(oof, predictions_output / "duplicate_removed_oof_predictions.parquet")
    atomic_parquet(duplicate_shap, shap_output / "duplicate_removed_oof_shap.parquet")
    atomic_csv(performance, tables_output / "duplicate_removed_performance.csv")
    atomic_csv(fold_metrics(oof), tables_output / "duplicate_removed_fold_performance.csv")
    atomic_csv(duplicate_stability, tables_output / "duplicate_removed_shap_stability.csv")
    atomic_csv(pd.DataFrame(fit_rows), tables_output / "duplicate_removed_fit_manifest.csv")
    atomic_csv(component_table, tables_output / "shap_feature_groups.csv")
    atomic_parquet(grouped_wide, shap_output / "primary_grouped_abs_shap.parquet")
    atomic_csv(grouped_importance, tables_output / "grouped_shap_importance.csv")
    atomic_csv(reconciliation, tables_output / "grouped_shap_reconciliation.csv")

    hashes_after = primary_hashes()
    if hashes_after != hashes_before:
        raise AssertionError("A frozen primary artifact changed during robustness analysis")
    elapsed = time.perf_counter() - started
    owned_outputs = [
        predictions_output / "duplicate_removed_oof_predictions.parquet",
        shap_output / "duplicate_removed_oof_shap.parquet",
        shap_output / "primary_grouped_abs_shap.parquet",
        figures_output / "grouped_shap_heatmap.png",
        figures_output / "grouped_shap_heatmap.pdf",
        tables_output / "duplicate_removed_performance.csv",
        tables_output / "duplicate_removed_fold_performance.csv",
        tables_output / "duplicate_removed_shap_stability.csv",
        tables_output / "duplicate_removed_fit_manifest.csv",
        tables_output / "shap_feature_groups.csv",
        tables_output / "grouped_shap_importance.csv",
        tables_output / "grouped_shap_reconciliation.csv",
    ]
    audit = {
        "status": "PASS",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "refit_and_heldout_shap_seconds": refit_seconds,
        "grouped_shap_seconds": grouped_seconds,
        "python": platform.python_version(),
        "duplicate_pair": [KEEP_FEATURE, REMOVE_FEATURE],
        "kept_feature": KEEP_FEATURE,
        "removed_feature": REMOVE_FEATURE,
        "duplicate_equal_including_missingness": True,
        "primary_feature_n": len(features),
        "reduced_feature_n": len(reduced_features),
        "cells": len(table),
        "donor_groups": int(table.group_id.astype(str).nunique()),
        "modules": len(MODULES),
        "outer_folds": 3,
        "oof_rows": len(oof),
        "shap_rows": len(duplicate_shap),
        "removed_feature_absent_from_shap": REMOVE_FEATURE not in duplicate_shap,
        "frozen_fold_exact": True,
        "donor_disjoint": True,
        "parameter_policy": "reused saved selected primary XGBoost parameters per module/fold; no retuning",
        "preprocessing": "fold-local median imputation; no scaling",
        "shap_policy": "fold-specific refitted model evaluated only on exact frozen outer-test cells",
        "correlation_threshold": "absolute Spearman rho > 0.90",
        "connected_components": len(components),
        "non_singleton_components": sum(len(component) > 1 for component in components),
        "component_sizes": [len(component) for component in components],
        "grouped_shap_rows": len(grouped_wide),
        "grouped_sum_reconciliation_max_abs_difference": reconciliation_max,
        "primary_hashes_before": hashes_before,
        "primary_hashes_after": hashes_after,
        "outputs": {
            str(path.relative_to(output)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in owned_outputs
        },
    }
    temporary = logs_output / "r3_duplicate_audit.json.tmp"
    temporary.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    temporary.replace(logs_output / "r3_duplicate_audit.json")
    print(json.dumps(audit, indent=2))
    print(performance.to_string(index=False))


if __name__ == "__main__":
    main()
