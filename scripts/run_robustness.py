#!/usr/bin/env python
"""Post-hoc robustness V2: R1 paired bootstrap and R4 repeated grouped CV.

This script writes exclusively below ``results/robustness_v2`` and treats all
primary analysis artifacts as immutable inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/robustness_v2"
sys.path.insert(0, str(ROOT))

from src.evaluation import regression_metrics
from src.models import elastic_net_candidates, xgboost_candidates
from src.training import MODULES, load_and_validate_inputs


SEEDS = (42, 123, 2026, 31415, 27182)
MODELS = ("ElasticNet", "XGBoost")
from src.reproduction import reference_or_run_digest

PRIMARY_OOF_SHA256 = reference_or_run_digest(ROOT, "primary", "results/predictions/oof_predictions.parquet", "9d0fb9edb89b3133fd1e356cbc437ca7aa3b985df4f810805b0cce74b96a4217")
HYPERPARAMETER_SELECTION_RULE = (
    "For each module/model, use the configuration selected by grouped inner validation "
    "inside primary outer fold 0 (the lowest frozen primary fold index). Hold it fixed "
    "for all five repeats and all three folds; do not use repeated-CV outcomes for selection."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stable_seed(*values: object, base: int = 42) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def validate_primary_inputs(
    oof: pd.DataFrame, table: pd.DataFrame, frozen_parameters: pd.DataFrame
) -> dict[str, object]:
    oof_hash = sha256(ROOT / "results/predictions/oof_predictions.parquet")
    if oof_hash != PRIMARY_OOF_SHA256:
        raise AssertionError(f"Primary OOF hash changed: {oof_hash}")
    table = table.copy()
    table["canonical_cell_id"] = table["canonical_cell_id"].astype(str)
    if len(table) != 3410 or not table["canonical_cell_id"].is_unique:
        raise AssertionError("Modeling table is not the expected 3,410 unique cells")
    expected_cells = set(table["canonical_cell_id"])
    if table["group_id"].astype(str).nunique() != 871:
        raise AssertionError("Modeling table is not the expected 871 donor groups")
    observed = oof[["canonical_cell_id", "group_id", "subclass", "fold"]].drop_duplicates().copy()
    observed["canonical_cell_id"] = observed["canonical_cell_id"].astype(str)
    if len(observed) != len(table) or set(observed["canonical_cell_id"]) != expected_cells:
        raise AssertionError("Primary OOF cell coverage differs from modeling table")
    comparison = observed.merge(
        table[["canonical_cell_id", "group_id", "subclass", "fold"]],
        on="canonical_cell_id",
        validate="one_to_one",
        suffixes=("_oof", "_table"),
    )
    if not comparison["group_id_oof"].astype(str).eq(comparison["group_id_table"].astype(str)).all():
        raise AssertionError("Primary OOF donor identity differs from modeling table")
    if not comparison["subclass_oof"].astype(str).eq(comparison["subclass_table"].astype(str)).all():
        raise AssertionError("Primary OOF subclass differs from modeling table")
    if not comparison["fold_oof"].eq(comparison["fold_table"]).all():
        raise AssertionError("Primary OOF fold identity differs from modeling table")
    if oof.groupby(oof["group_id"].astype(str))["fold"].nunique().max() != 1:
        raise AssertionError("Primary OOF contains donor leakage")
    if len(frozen_parameters) != len(MODULES) * len(MODELS):
        raise AssertionError("Frozen R4 parameter grid is incomplete")
    return {
        "primary_oof_sha256": oof_hash,
        "primary_oof_rows": len(oof),
        "cells": len(table),
        "donor_groups": int(table["group_id"].astype(str).nunique()),
        "primary_fold_cell_counts": {
            str(key): int(value)
            for key, value in table["fold"].value_counts().sort_index().items()
        },
        "primary_max_folds_per_donor": int(
            table.groupby(table["group_id"].astype(str))["fold"].nunique().max()
        ),
    }


def freeze_primary_parameters(parameters: pd.DataFrame) -> pd.DataFrame:
    selected = parameters.loc[
        parameters["fold"].eq(0) & parameters["model"].isin(MODELS)
    ].copy()
    selected = selected.sort_values(["module", "model"]).reset_index(drop=True)
    if selected.duplicated(["module", "model"]).any():
        raise AssertionError("Primary fold 0 has duplicate module/model parameter rows")
    if set(selected["module"].astype(str)) != set(MODULES):
        raise AssertionError("Primary fold 0 parameter modules are incomplete")
    selected.insert(0, "selection_rule", HYPERPARAMETER_SELECTION_RULE)
    selected["model_random_state"] = 42
    selected["tuning_in_r4"] = False
    return selected


def paired_module_frame(oof: pd.DataFrame, module: str, table: pd.DataFrame) -> pd.DataFrame:
    identity = ["canonical_cell_id", "group_id", "subclass", "fold", "y_true"]
    comparator = oof.loc[
        oof["module"].eq(module)
        & oof["model"].eq("SubclassRidge")
        & oof["analysis"].eq("subclass_only"),
        identity + ["y_pred"],
    ].rename(columns={"y_pred": "subclass_only_prediction"})
    augmented = oof.loc[
        oof["module"].eq(module)
        & oof["model"].eq("EphysSubclassRidge")
        & oof["analysis"].eq("ephys_plus_subclass"),
        identity + ["y_pred"],
    ].rename(
        columns={
            "y_true": "y_true_augmented",
            "group_id": "group_id_augmented",
            "subclass": "subclass_augmented",
            "fold": "fold_augmented",
            "y_pred": "ephys_plus_subclass_prediction",
        }
    )
    paired = comparator.merge(augmented, on="canonical_cell_id", validate="one_to_one")
    if len(paired) != len(table) or set(paired["canonical_cell_id"].astype(str)) != set(
        table["canonical_cell_id"].astype(str)
    ):
        raise AssertionError(f"R1 pair does not cover exact cells for {module}")
    if not np.array_equal(paired["y_true"], paired["y_true_augmented"]):
        raise AssertionError(f"R1 truth differs between paired models for {module}")
    for left, right in (
        ("group_id", "group_id_augmented"),
        ("subclass", "subclass_augmented"),
        ("fold", "fold_augmented"),
    ):
        if not paired[left].astype(str).eq(paired[right].astype(str)).all():
            raise AssertionError(f"R1 {left} differs between paired models for {module}")
    target = table.set_index(table["canonical_cell_id"].astype(str))[f"target_{module}"]
    expected_truth = paired["canonical_cell_id"].astype(str).map(target)
    if not np.array_equal(expected_truth.to_numpy(dtype=float), paired["y_true"].to_numpy(dtype=float)):
        raise AssertionError(f"R1 truth differs from frozen target for {module}")
    return paired


def improvement_deltas(
    truth: np.ndarray, comparator: np.ndarray, augmented: np.ndarray
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    baseline_metrics = regression_metrics(truth, comparator)
    augmented_metrics = regression_metrics(truth, augmented)
    delta = {
        "r2": augmented_metrics["r2"] - baseline_metrics["r2"],
        "mae": baseline_metrics["mae"] - augmented_metrics["mae"],
        "rmse": baseline_metrics["rmse"] - augmented_metrics["rmse"],
        "spearman": augmented_metrics["spearman"] - baseline_metrics["spearman"],
    }
    return baseline_metrics, augmented_metrics, delta


def run_r1(
    oof: pd.DataFrame, table: pd.DataFrame, *, n_bootstrap: int = 2000
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    started = time.perf_counter()
    summary_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []
    module_audit: dict[str, object] = {}
    formulas = {
        "r2": "EphysSubclassRidge R2 - SubclassRidge R2",
        "mae": "SubclassRidge MAE - EphysSubclassRidge MAE",
        "rmse": "SubclassRidge RMSE - EphysSubclassRidge RMSE",
        "spearman": "EphysSubclassRidge Spearman - SubclassRidge Spearman",
    }
    for module in MODULES:
        paired = paired_module_frame(oof, module, table)
        group_codes, unique_groups = pd.factorize(paired["group_id"].astype(str), sort=False)
        if len(unique_groups) != 871:
            raise AssertionError(f"R1 expected 871 donor groups for {module}")
        row_indices = np.arange(len(paired), dtype=np.int64)
        truth = paired["y_true"].to_numpy(dtype=float)
        comparator = paired["subclass_only_prediction"].to_numpy(dtype=float)
        augmented = paired["ephys_plus_subclass_prediction"].to_numpy(dtype=float)
        base_full, augmented_full, observed_delta = improvement_deltas(
            truth, comparator, augmented
        )
        rng = np.random.default_rng(stable_seed("R1", module))
        local_draws = {metric: np.empty(n_bootstrap, dtype=float) for metric in formulas}
        min_sampled_rows = len(paired)
        max_sampled_rows = 0
        for replicate in range(n_bootstrap):
            sampled_groups = rng.integers(0, len(unique_groups), size=len(unique_groups))
            multiplicity = np.bincount(sampled_groups, minlength=len(unique_groups))
            sampled_indices = np.repeat(row_indices, multiplicity[group_codes])
            min_sampled_rows = min(min_sampled_rows, len(sampled_indices))
            max_sampled_rows = max(max_sampled_rows, len(sampled_indices))
            _, _, delta = improvement_deltas(
                truth[sampled_indices], comparator[sampled_indices], augmented[sampled_indices]
            )
            for metric, value in delta.items():
                local_draws[metric][replicate] = value
        for metric, values in local_draws.items():
            for replicate, value in enumerate(values):
                draw_rows.append(
                    {"module": module, "replicate": replicate, "metric": metric, "delta": value}
                )
            summary_rows.append(
                {
                    "module": module,
                    "metric": metric,
                    "n_cells": len(paired),
                    "n_donor_groups": len(unique_groups),
                    "n_bootstrap": n_bootstrap,
                    "subclass_only": base_full[metric],
                    "ephys_plus_subclass": augmented_full[metric],
                    "delta": observed_delta[metric],
                    "ci_low": float(np.quantile(values, 0.025)),
                    "ci_high": float(np.quantile(values, 0.975)),
                    "p_delta_positive": float(np.mean(values > 0)),
                    "positive_means": "EphysSubclassRidge improves on SubclassRidge",
                    "delta_orientation": formulas[metric],
                    "bootstrap_unit": "donor/group; paired models use identical sampled donor multiplicities",
                }
            )
        module_audit[module] = {
            "cells": len(paired),
            "donor_groups": len(unique_groups),
            "folds": sorted(map(int, paired["fold"].unique())),
            "replicates": n_bootstrap,
            "minimum_rows_in_resample": min_sampled_rows,
            "maximum_rows_in_resample": max_sampled_rows,
            "paired_sampling": True,
        }
        print(f"R1 {module}: complete", flush=True)
    summary = pd.DataFrame(summary_rows)
    draws = pd.DataFrame(draw_rows)
    if len(summary) != len(MODULES) * 4 or len(draws) != len(MODULES) * 4 * n_bootstrap:
        raise AssertionError("R1 output grid is incomplete")
    return summary, draws, {
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "n_bootstrap": n_bootstrap,
        "metric_orientation": formulas,
        "modules": module_audit,
    }


def pipeline_for_frozen_configuration(
    module: str, model: str, features: list[str], frozen: pd.DataFrame
):
    row = frozen.loc[frozen["module"].eq(module) & frozen["model"].eq(model)]
    if len(row) != 1:
        raise AssertionError(f"Missing unique frozen parameters for {module}/{model}")
    record = row.iloc[0]
    if model == "ElasticNet":
        expected = {"alpha": float(record["alpha"]), "l1_ratio": float(record["l1_ratio"])}
        candidates = elastic_net_candidates(features, seed=42)
    elif model == "XGBoost":
        expected = {
            "n_estimators": int(record["n_estimators"]),
            "max_depth": int(record["max_depth"]),
            "learning_rate": float(record["learning_rate"]),
            "reg_lambda": float(record["reg_lambda"]),
        }
        candidates = xgboost_candidates(features, seed=42, level=0)
    else:
        raise ValueError(model)
    for parameters, pipeline in candidates:
        if parameters == expected:
            return clone(pipeline)
    raise AssertionError(f"Frozen configuration is outside the primary candidate grid: {module}/{model}")


def make_repeated_folds(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    assignment_parts: list[pd.DataFrame] = []
    composition_rows: list[dict[str, object]] = []
    global_proportion = table["subclass"].astype(str).value_counts(normalize=True)
    maximum_subclass_proportion_deviation = 0.0
    for repeat, seed in enumerate(SEEDS):
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
        local = table[["canonical_cell_id", "group_id", "subclass"]].copy()
        local["fold"] = -1
        for fold, (_, test_idx) in enumerate(
            splitter.split(
                np.zeros(len(table)),
                y=table["subclass"].astype(str),
                groups=table["group_id"].astype(str),
            )
        ):
            local.loc[local.index[test_idx], "fold"] = fold
        if local["fold"].lt(0).any():
            raise AssertionError(f"Unassigned R4 cells for repeat {repeat}")
        if local.groupby(local["group_id"].astype(str))["fold"].nunique().max() != 1:
            raise AssertionError(f"Donor leakage in R4 repeat {repeat}")
        if local["canonical_cell_id"].astype(str).duplicated().any():
            raise AssertionError(f"Duplicate R4 cell assignment in repeat {repeat}")
        local.insert(0, "seed", seed)
        local.insert(0, "repeat", repeat)
        assignment_parts.append(local)
        for fold, frame in local.groupby("fold", sort=True):
            fold_proportion = frame["subclass"].astype(str).value_counts(normalize=True)
            deviation = float(
                (fold_proportion.reindex(global_proportion.index, fill_value=0) - global_proportion)
                .abs()
                .max()
            )
            maximum_subclass_proportion_deviation = max(
                maximum_subclass_proportion_deviation, deviation
            )
            for subclass in global_proportion.index:
                composition_rows.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "fold": int(fold),
                        "subclass": subclass,
                        "n_cells": int(frame["subclass"].astype(str).eq(subclass).sum()),
                        "n_groups_total": int(frame["group_id"].astype(str).nunique()),
                        "fold_cell_n": len(frame),
                        "fold_proportion": float(fold_proportion.get(subclass, 0.0)),
                        "global_proportion": float(global_proportion[subclass]),
                        "absolute_proportion_deviation": abs(
                            float(fold_proportion.get(subclass, 0.0) - global_proportion[subclass])
                        ),
                    }
                )
    assignments = pd.concat(assignment_parts, ignore_index=True)
    composition = pd.DataFrame(composition_rows)
    return assignments, composition, {
        "fold_constructor": "sklearn.model_selection.StratifiedGroupKFold",
        "n_splits": 3,
        "seeds": list(SEEDS),
        "assignment_rows": len(assignments),
        "maximum_subclass_proportion_deviation": maximum_subclass_proportion_deviation,
        "max_folds_per_repeat_donor": int(
            assignments.assign(_group=assignments["group_id"].astype(str))
            .groupby(["repeat", "_group"])["fold"]
            .nunique()
            .max()
        ),
    }


def validate_r4_predictions(
    predictions: pd.DataFrame, table: pd.DataFrame, assignments: pd.DataFrame
) -> dict[str, object]:
    expected_cells = set(table["canonical_cell_id"].astype(str))
    expected_keys = {
        (repeat, module, model)
        for repeat in range(len(SEEDS))
        for module in MODULES
        for model in MODELS
    }
    observed_keys = set(
        predictions[["repeat", "module", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_keys != expected_keys:
        raise AssertionError("R4 prediction grid is incomplete")
    if not np.isfinite(predictions[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise AssertionError("R4 predictions contain non-finite values")
    for keys, frame in predictions.groupby(["repeat", "module", "model"], sort=False):
        identifiers = frame["canonical_cell_id"].astype(str)
        if identifiers.duplicated().any() or set(identifiers) != expected_cells:
            raise AssertionError(f"R4 does not have exact OOF coverage for {keys}")
        target = table.set_index(table["canonical_cell_id"].astype(str))[f"target_{keys[1]}"]
        expected_truth = identifiers.map(target)
        if not np.array_equal(expected_truth.to_numpy(dtype=float), frame["y_true"].to_numpy(dtype=float)):
            raise AssertionError(f"R4 truth differs from frozen target for {keys}")
    metadata = predictions[
        ["repeat", "seed", "canonical_cell_id", "group_id", "subclass", "fold"]
    ].drop_duplicates()
    expected_metadata = assignments.copy()
    for frame in (metadata, expected_metadata):
        frame["canonical_cell_id"] = frame["canonical_cell_id"].astype(str)
        frame["group_id"] = frame["group_id"].astype(str)
        frame["subclass"] = frame["subclass"].astype(str)
    merged = metadata.merge(
        expected_metadata,
        on=["repeat", "seed", "canonical_cell_id"],
        validate="one_to_one",
        suffixes=("_prediction", "_assignment"),
    )
    for column in ("group_id", "subclass", "fold"):
        if not merged[f"{column}_prediction"].astype(str).eq(
            merged[f"{column}_assignment"].astype(str)
        ).all():
            raise AssertionError(f"R4 prediction {column} differs from fold assignment")
    maximum_train_test_overlap = 0
    for repeat in range(len(SEEDS)):
        local = assignments.loc[assignments["repeat"].eq(repeat)]
        for fold in range(3):
            train_groups = set(local.loc[local["fold"].ne(fold), "group_id"].astype(str))
            test_groups = set(local.loc[local["fold"].eq(fold), "group_id"].astype(str))
            maximum_train_test_overlap = max(
                maximum_train_test_overlap, len(train_groups.intersection(test_groups))
            )
    if maximum_train_test_overlap != 0:
        raise AssertionError("R4 donor overlap detected")
    return {
        "status": "PASS",
        "prediction_rows": len(predictions),
        "expected_prediction_rows": len(table) * len(SEEDS) * len(MODULES) * len(MODELS),
        "grid_keys": len(observed_keys),
        "cells_per_grid_key": len(table),
        "exact_oof_once_per_repeat_module_model": True,
        "maximum_train_test_donor_overlap": maximum_train_test_overlap,
        "max_folds_per_repeat_donor": int(
            metadata.assign(_group=metadata["group_id"].astype(str))
            .groupby(["repeat", "_group"])["fold"]
            .nunique()
            .max()
        ),
        "truth_exact_to_frozen_targets": True,
    }


def run_r4(
    table: pd.DataFrame, features: list[str], frozen: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    started = time.perf_counter()
    assignments, composition, fold_audit = make_repeated_folds(table)
    prediction_parts: list[pd.DataFrame] = []
    repeat_timings: dict[str, float] = {}
    for repeat, seed in enumerate(SEEDS):
        repeat_started = time.perf_counter()
        local_assignments = assignments.loc[assignments["repeat"].eq(repeat), ["canonical_cell_id", "fold"]]
        local = table.merge(
            local_assignments,
            on="canonical_cell_id",
            validate="one_to_one",
            suffixes=("_primary", "_r4"),
        )
        for fold in range(3):
            train = local.loc[local["fold_r4"].ne(fold)]
            test = local.loc[local["fold_r4"].eq(fold)]
            if set(train["group_id"].astype(str)).intersection(test["group_id"].astype(str)):
                raise AssertionError(f"R4 donor overlap at repeat {repeat}, fold {fold}")
            X_train, X_test = train[features], test[features]
            for module in MODULES:
                y_train = train[f"target_{module}"].to_numpy(dtype=float)
                for model in MODELS:
                    pipeline = pipeline_for_frozen_configuration(module, model, features, frozen)
                    # Pipeline fitting makes imputation/scaling fold-local. No
                    # candidate selection or hyperparameter tuning occurs here.
                    pipeline.fit(X_train, y_train)
                    prediction_parts.append(
                        pd.DataFrame(
                            {
                                "repeat": repeat,
                                "seed": seed,
                                "canonical_cell_id": test["canonical_cell_id"].astype(str).to_numpy(),
                                "group_id": test["group_id"].astype(str).to_numpy(),
                                "subclass": test["subclass"].astype(str).to_numpy(),
                                "fold": fold,
                                "module": module,
                                "model": model,
                                "analysis": "R4_repeated_group_cv_fixed_primary_hyperparameters",
                                "y_true": test[f"target_{module}"].to_numpy(dtype=float),
                                "y_pred": pipeline.predict(X_test),
                            }
                        )
                    )
        repeat_timings[str(repeat)] = time.perf_counter() - repeat_started
        print(f"R4 repeat {repeat} seed {seed}: {repeat_timings[str(repeat)]:.3f}s", flush=True)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    prediction_audit = validate_r4_predictions(predictions, table, assignments)
    repeat_metric_rows: list[dict[str, object]] = []
    for keys, frame in predictions.groupby(["repeat", "seed", "module", "model"], sort=True):
        repeat_metric_rows.append(
            {
                "repeat": keys[0],
                "seed": keys[1],
                "module": keys[2],
                "model": keys[3],
                "n_cells": len(frame),
                "n_donor_groups": frame["group_id"].astype(str).nunique(),
                **regression_metrics(frame["y_true"], frame["y_pred"]),
            }
        )
    repeat_metrics = pd.DataFrame(repeat_metric_rows).sort_values(
        ["module", "model", "repeat"]
    ).reset_index(drop=True)
    if len(repeat_metrics) != len(SEEDS) * len(MODULES) * len(MODELS):
        raise AssertionError("R4 repeat-metric grid is incomplete")
    aggregate_rows: list[dict[str, object]] = []
    for keys, frame in repeat_metrics.groupby(["module", "model"], sort=True):
        row: dict[str, object] = {
            "module": keys[0],
            "model": keys[1],
            "n_repeats": len(frame),
            "fixed_hyperparameter_selection_rule": HYPERPARAMETER_SELECTION_RULE,
        }
        for metric in ("r2", "mae", "rmse", "spearman"):
            values = frame[metric].to_numpy(dtype=float)
            row.update(
                {
                    f"{metric}_mean": float(np.mean(values)),
                    f"{metric}_sd": float(np.std(values, ddof=1)),
                    f"{metric}_min": float(np.min(values)),
                    f"{metric}_max": float(np.max(values)),
                }
            )
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    return predictions, repeat_metrics, aggregate, composition, {
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "repeat_fit_seconds": repeat_timings,
        "fixed_hyperparameter_selection_rule": HYPERPARAMETER_SELECTION_RULE,
        "model_random_state": 42,
        "tuning_performed": False,
        "preprocessing": "fold-local Pipeline fit (median imputation; ElasticNet scaling; XGBoost unscaled)",
        "fold_audit": fold_audit,
        "prediction_audit": prediction_audit,
    }


def build_r4_compliance_summary(repeat_metrics: pd.DataFrame) -> pd.DataFrame:
    """Build the required 12-row R2 stability summary from saved repeat metrics."""
    primary = pd.read_csv(ROOT / "results/tables/model_performance.csv")
    primary = primary.loc[
        primary["analysis"].eq("ephys_only") & primary["model"].isin(MODELS),
        ["module", "model", "r2"],
    ].rename(columns={"r2": "primary_r2"})
    if len(primary) != len(MODULES) * len(MODELS) or primary.duplicated(
        ["module", "model"]
    ).any():
        raise AssertionError("Primary R2 comparison grid is incomplete")
    rows: list[dict[str, object]] = []
    for keys, frame in repeat_metrics.groupby(["module", "model"], sort=True):
        values = frame["r2"].to_numpy(dtype=float)
        if len(values) != len(SEEDS) or not np.isfinite(values).all():
            raise AssertionError(f"R4 requires five finite R2 values for {keys}")
        primary_r2 = float(
            primary.loc[
                primary["module"].eq(keys[0]) & primary["model"].eq(keys[1]), "primary_r2"
            ].iloc[0]
        )
        r2_min, r2_max = float(np.min(values)), float(np.max(values))
        if primary_r2 < r2_min:
            position = "below_repeated_min"
        elif primary_r2 > r2_max:
            position = "above_repeated_max"
        else:
            position = "within_repeated_range"
        median = float(np.median(values))
        q1, q3 = map(float, np.quantile(values, (0.25, 0.75)))
        rows.append(
            {
                "module": keys[0],
                "model": keys[1],
                "n_repeats": len(values),
                "r2_median": median,
                "r2_q1": q1,
                "r2_q3": q3,
                "r2_iqr": q3 - q1,
                "r2_min": r2_min,
                "r2_max": r2_max,
                "r2_range": r2_max - r2_min,
                "fraction_r2_gt_zero": float(np.mean(values > 0)),
                "primary_r2": primary_r2,
                "r2_median_minus_primary": median - primary_r2,
                "primary_r2_position": position,
                "primary_extreme_flag": position != "within_repeated_range",
                "primary_extreme_definition": "primary R2 below repeated minimum or above repeated maximum",
                "fixed_hyperparameter_selection_rule": HYPERPARAMETER_SELECTION_RULE,
            }
        )
    summary = pd.DataFrame(rows).sort_values(["module", "model"]).reset_index(drop=True)
    if len(summary) != len(MODULES) * len(MODELS):
        raise AssertionError("R4 compliance summary must contain 12 module/model rows")
    return summary


def plot_r2_distribution(summary: pd.DataFrame) -> None:
    modules = list(MODULES)
    colors = {"ElasticNet": "#3B82F6", "XGBoost": "#F97316"}
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.5), sharey=False)
    rng = np.random.default_rng(42)
    for axis, module in zip(axes.flat, modules):
        local = summary.loc[summary["module"].eq(module)]
        values = [local.loc[local["model"].eq(model), "r2"].to_numpy() for model in MODELS]
        box = axis.boxplot(values, positions=(1, 2), widths=0.5, patch_artist=True, showfliers=False)
        for patch, model in zip(box["boxes"], MODELS):
            patch.set_facecolor(colors[model])
            patch.set_alpha(0.28)
            patch.set_edgecolor(colors[model])
        for position, model, model_values in zip((1, 2), MODELS, values):
            jitter = rng.uniform(-0.07, 0.07, size=len(model_values))
            axis.scatter(
                np.full(len(model_values), position) + jitter,
                model_values,
                s=28,
                color=colors[model],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
        axis.axhline(0, color="#6B7280", linewidth=0.8, linestyle="--")
        axis.set_title(module)
        axis.set_xticks((1, 2), ("ElasticNet", "XGBoost"))
        axis.set_ylabel("Pooled OOF R² across 5 repeats")
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("R4: donor-disjoint repeated 3-fold CV with fixed primary hyperparameters", y=1.01)
    fig.tight_layout()
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "repeated_group_cv_r2_distribution.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / "repeated_group_cv_r2_distribution.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_audit_runtime_to_log_namespace() -> None:
    log_dir = ROOT / "logs/robustness_v2"
    log_dir.mkdir(parents=True, exist_ok=True)
    for name in ("audit.json", "runtime.json"):
        payload = json.loads((OUT / name).read_text(encoding="utf-8"))
        atomic_json(payload, log_dir / name)


def derive_r4_summary_only() -> dict[str, object]:
    """Repair R4 reporting schema from saved metrics without any model refit."""
    started = time.perf_counter()
    tables_dir = OUT / "tables"
    repeat_path = tables_dir / "repeated_group_cv_repeat_metrics.csv"
    current_summary_path = tables_dir / "repeated_group_cv_summary.csv"
    if repeat_path.exists():
        repeat_metrics = pd.read_csv(repeat_path)
    else:
        repeat_metrics = pd.read_csv(current_summary_path)
    required = {"repeat", "seed", "module", "model", "r2", "mae", "rmse", "spearman"}
    if len(repeat_metrics) != 60 or not required.issubset(repeat_metrics.columns):
        raise AssertionError("Cannot recover the required 60 saved R4 repeat metrics")
    summary = build_r4_compliance_summary(repeat_metrics)
    atomic_csv(repeat_metrics, repeat_path)
    atomic_csv(summary, current_summary_path)
    audit_path = OUT / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["r4"].update(
        {
            "repeat_metric_rows": len(repeat_metrics),
            "compliance_summary_rows": len(summary),
            "summary_primary_extreme_definition": "primary R2 below repeated minimum or above repeated maximum",
            "summary_derived_without_refitting": True,
        }
    )
    audit["output_sha256"][str(repeat_path.relative_to(ROOT)).replace("\\", "/")] = sha256(
        repeat_path
    )
    audit["output_sha256"][str(current_summary_path.relative_to(ROOT)).replace("\\", "/")] = sha256(
        current_summary_path
    )
    audit["primary_oof_sha256_after"] = sha256(
        ROOT / "results/predictions/oof_predictions.parquet"
    )
    audit["primary_oof_unchanged"] = audit["primary_oof_sha256_after"] == PRIMARY_OOF_SHA256
    if not audit["primary_oof_unchanged"]:
        raise AssertionError("Primary OOF changed during derived-summary repair")
    atomic_json(audit, audit_path)
    runtime_path = OUT / "runtime.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    elapsed = time.perf_counter() - started
    runtime.update(
        {
            "derived_summary_only_seconds": elapsed,
            "derived_summary_only_completed_utc": datetime.now(timezone.utc).isoformat(),
            "derived_summary_only_refits": 0,
        }
    )
    atomic_json(runtime, runtime_path)
    copy_audit_runtime_to_log_namespace()
    result = {
        "status": "PASS",
        "repeat_metric_rows": len(repeat_metrics),
        "summary_rows": len(summary),
        "elapsed_seconds": elapsed,
        "model_refits": 0,
        "primary_oof_sha256": audit["primary_oof_sha256_after"],
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def main() -> None:
    total_started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    tables_dir = OUT / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    oof = pd.read_parquet(ROOT / "results/predictions/oof_predictions.parquet")
    oof["canonical_cell_id"] = oof["canonical_cell_id"].astype(str)
    table, features = load_and_validate_inputs(
        ROOT / "data/processed/modeling_table.parquet",
        ROOT / "results/tables/cv_folds.csv",
        ROOT / "config/ephys_features.yaml",
    )
    parameter_source = pd.read_csv(ROOT / "results/tables/fitted_hyperparameters.csv")
    frozen = freeze_primary_parameters(parameter_source)
    primary_audit = validate_primary_inputs(oof, table, frozen)
    atomic_csv(frozen, tables_dir / "frozen_hyperparameters.csv")

    r1_summary, r1_draws, r1_audit = run_r1(oof, table, n_bootstrap=2000)
    atomic_csv(r1_summary, tables_dir / "incremental_subclass_comparison.csv")
    atomic_parquet(r1_draws, tables_dir / "incremental_subclass_bootstrap_draws.parquet")

    r4_predictions, r4_repeat_metrics, r4_aggregate, composition, r4_audit = run_r4(
        table, features, frozen
    )
    r4_summary = build_r4_compliance_summary(r4_repeat_metrics)
    atomic_csv(r4_predictions, tables_dir / "repeated_group_cv.csv")
    atomic_csv(r4_repeat_metrics, tables_dir / "repeated_group_cv_repeat_metrics.csv")
    atomic_csv(r4_summary, tables_dir / "repeated_group_cv_summary.csv")
    atomic_csv(r4_aggregate, tables_dir / "repeated_group_cv_aggregate.csv")
    atomic_csv(composition, tables_dir / "repeated_fold_composition.csv")
    plot_r2_distribution(r4_repeat_metrics)

    primary_oof_sha256_after = sha256(ROOT / "results/predictions/oof_predictions.parquet")
    if primary_oof_sha256_after != PRIMARY_OOF_SHA256:
        raise AssertionError("Primary OOF changed during robustness V2")
    output_hashes = {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256(path)
        for path in (
            tables_dir / "incremental_subclass_comparison.csv",
            tables_dir / "incremental_subclass_bootstrap_draws.parquet",
            tables_dir / "frozen_hyperparameters.csv",
            tables_dir / "repeated_group_cv.csv",
            tables_dir / "repeated_group_cv_repeat_metrics.csv",
            tables_dir / "repeated_group_cv_summary.csv",
            tables_dir / "repeated_group_cv_aggregate.csv",
            tables_dir / "repeated_fold_composition.csv",
            OUT / "figures/repeated_group_cv_r2_distribution.png",
            OUT / "figures/repeated_group_cv_r2_distribution.pdf",
        )
    }
    audit = {
        "status": "PASS",
        "scope": ["R1", "R4"],
        "namespace": "results/robustness_v2",
        "primary_inputs": {
            **primary_audit,
            "modeling_table_sha256": sha256(ROOT / "data/processed/modeling_table.parquet"),
            "fitted_hyperparameters_sha256": sha256(
                ROOT / "results/tables/fitted_hyperparameters.csv"
            ),
            "ephys_features_sha256": sha256(ROOT / "config/ephys_features.yaml"),
        },
        "r1": r1_audit,
        "r4": {
            **r4_audit,
            "repeat_metric_rows": len(r4_repeat_metrics),
            "compliance_summary_rows": len(r4_summary),
            "summary_primary_extreme_definition": "primary R2 below repeated minimum or above repeated maximum",
            "summary_derived_without_refitting": True,
        },
        "primary_oof_sha256_after": primary_oof_sha256_after,
        "primary_oof_unchanged": primary_oof_sha256_after == primary_audit["primary_oof_sha256"],
        "output_sha256": output_hashes,
    }
    atomic_json(audit, OUT / "audit.json")
    runtime = {
        "status": "PASS",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "r1_seconds": r1_audit["elapsed_seconds"],
        "r4_seconds": r4_audit["elapsed_seconds"],
        "r4_repeat_fit_seconds": r4_audit["repeat_fit_seconds"],
        "total_seconds": time.perf_counter() - total_started,
        "runtime_target_seconds": 900,
        "within_runtime_target": (time.perf_counter() - total_started) <= 900,
    }
    atomic_json(runtime, OUT / "runtime.json")
    copy_audit_runtime_to_log_namespace()
    print(json.dumps(runtime, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--derive-r4-summary-only",
        action="store_true",
        help="repair R4 summary schema from saved repeat metrics without refitting",
    )
    arguments = parser.parse_args()
    if arguments.derive_r4_summary_only:
        derive_r4_summary_only()
    else:
        main()
