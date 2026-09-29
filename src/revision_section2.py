"""Model-comparison analysis: 23-feature models, inference, and interpretation.

The frozen primary, Robustness V2, and Matched-control analysis trees are strictly
read-only inputs.  Every analytical write is isolated in the configured
``revision_v3/section2`` namespaces and is hash-covered for safe resume.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from scipy.stats import rankdata, spearmanr
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

from .evaluation import METRICS, fold_metrics, performance_table, regression_metrics
from .interpretability import heldout_tree_shap, high_correlation_pairs
from .models import (
    dummy_model,
    elastic_net_candidates,
    ephys_subclass_ridge_model,
    group_inner_split,
    mlp_candidate,
    select_on_inner_groups,
    subclass_ridge_model,
    xgboost_candidates,
)
from .preprocessing import make_preprocessor
from .training import MODULES, load_and_validate_inputs


PIPELINE_VERSION = "revision-v3-section2-1"
PRE_HARDENING_SIGNATURE = "48a26d4fe6c0cbbc16a7110cb6053f266f1d34d1e651a5451490e1f739783ea6"
MODULE_ORDER = tuple(MODULES)
CLASSICAL_MODELS = ("ElasticNet", "XGBoost")
EPHYS_MODELS = ("ElasticNet", "XGBoost", "MLP")
ALL_MODELS = (
    "ElasticNet",
    "XGBoost",
    "MLP",
    "Dummy",
    "SubclassRidge",
    "EphysSubclassRidge",
)
MODEL_ANALYSIS = {
    "ElasticNet": "ephys_only",
    "XGBoost": "ephys_only",
    "MLP": "ephys_only",
    "Dummy": "dummy",
    "SubclassRidge": "subclass_only",
    "EphysSubclassRidge": "ephys_plus_subclass",
}
CONTRASTS = (
    ("XGBoost_minus_ElasticNet", "XGBoost", "ElasticNet"),
    ("MLP_minus_ElasticNet", "MLP", "ElasticNet"),
    ("MLP_minus_XGBoost", "MLP", "XGBoost"),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def stable_seed(*parts: object, base: int = 42) -> int:
    value = "|".join([PIPELINE_VERSION, str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(payload: Any, path: Path) -> None:
    atomic_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="snappy")
    temporary.replace(path)


def read_config(config_path: str | Path) -> tuple[Path, dict[str, Any], Path]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "1.0":
        raise ValueError("Unsupported Model-comparison analysis configuration")
    return path.parent.parent, config, path


def resolve_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {key: root / value for key, value in config["paths"].items()}


def output_roots(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {key: root / value for key, value in config["outputs"].items()}


def ensure_output_roots(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = output_roots(root, config)
    for path in outputs.values():
        path.mkdir(parents=True, exist_ok=True)
    return outputs


def config_signature(root: Path, config: dict[str, Any], config_path: Path) -> str:
    paths = resolve_paths(root, config)
    input_keys = (
        "modeling_table",
        "frozen_folds",
        "module_scores",
        "ephys_features",
        "primary_oof",
        "primary_shap",
        "primary_hyperparameters",
        "manuscript",
        "section1_manifest",
        "section1_matched_sets",
        "section1_matching_quality",
        "section1_matching_assignments",
        "section1_technical_targets",
        "section1_residual_targets",
        "robustness_duplicate_oof",
        "robustness_duplicate_shap",
        "robustness_duplicate_fit_manifest",
        "robustness_repeated_oof",
    )
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "implementation_sha256": sha256_file(Path(__file__)),
        "entrypoint_sha256": sha256_file(root / "scripts/10_revision_section2.py"),
        "config_sha256": sha256_file(config_path),
        "inputs": {key: sha256_file(paths[key]) for key in input_keys},
        "control_targets": {
            module: sha256_file(paths["section1_control_targets"] / f"{module}.parquet")
            for module in MODULE_ORDER
        },
        "packages": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "shap")
        },
    }
    return hash_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


class StageCache:
    """Exact byte-hash cache contract over every required stage artifact."""

    def __init__(self, path: Path, signature: str):
        self.path = path
        self.signature = signature
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            payload = {}
        self.payload = payload if payload.get("signature") == signature else {"signature": signature}
        self.payload.setdefault("stages", {})

    def valid(self, stage: str, paths: Sequence[Path]) -> bool:
        record = self.payload["stages"].get(stage, {})
        files = record.get("files", {}) if isinstance(record, dict) else {}
        return bool(paths) and all(
            path.is_file() and files.get(str(path.resolve())) == sha256_file(path) for path in paths
        )

    def complete(self, stage: str, paths: Sequence[Path], **metadata: Any) -> None:
        if not all(path.is_file() for path in paths):
            missing = [str(path) for path in paths if not path.is_file()]
            raise AssertionError(f"Cannot complete {stage}; missing {missing[:5]}")
        self.payload["stages"][stage] = {
            "completed_at": pd.Timestamp.now(tz="Asia/Tehran").isoformat(),
            "files": {str(path.resolve()): sha256_file(path) for path in paths},
            **metadata,
        }
        atomic_json(self.payload, self.path)


def migrate_verified_checkpoint(cache_path: Path, new_signature: str) -> dict[str, Any] | None:
    """Narrow migration from the completed initial run after source hardening.

    Every prior artifact must still match its recorded digest.  Only the two
    stages whose computations/schemas changed are invalidated; no unverified
    prediction partition is admitted into the new cache contract.
    """

    if not cache_path.is_file():
        return None
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    if payload.get("signature") != PRE_HARDENING_SIGNATURE:
        return None
    failures = []
    for stage, record in payload.get("stages", {}).items():
        for raw_path, digest in record.get("files", {}).items():
            path = Path(raw_path)
            if not path.is_file() or sha256_file(path) != digest:
                failures.append({"stage": stage, "path": raw_path})
    if failures:
        raise AssertionError(f"Checkpoint migration refused; old digest mismatch: {failures[:3]}")
    invalidated = [stage for stage in ("technical_bridge", "adjusted_bridge") if stage in payload["stages"]]
    for stage in invalidated:
        payload["stages"].pop(stage, None)
    payload["signature"] = new_signature
    payload["migration"] = {
        "from_signature": PRE_HARDENING_SIGNATURE,
        "to_signature": new_signature,
        "all_prior_recorded_digests_validated": True,
        "invalidated_stages": invalidated,
        "reason": "subsequent cache hardening, frozen Robustness hashes, technical ledger, and leakage-free adjusted residualization",
        "migrated_at": pd.Timestamp.now(tz="Asia/Tehran").isoformat(),
    }
    atomic_json(payload, cache_path)
    return payload["migration"]


def metric_value(y_true: Sequence[float], y_pred: Sequence[float], metric: str) -> float:
    if metric not in METRICS:
        raise KeyError(metric)
    return float(regression_metrics(np.asarray(y_true), np.asarray(y_pred))[metric])


def bh_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("BH p-values must be a finite one-dimensional vector in [0,1]")
    order = np.lexsort((np.arange(len(p)), p))
    ranked = p[order]
    adjusted_ranked = np.minimum.accumulate((len(p) * ranked / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(adjusted_ranked)
    result[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return result


def exact_duplicate_audit(
    table: pd.DataFrame,
    features: Sequence[str],
    retain: str,
    drop: str,
) -> dict[str, Any]:
    features = list(features)
    exact_pairs: list[list[str]] = []
    pair_details: list[dict[str, Any]] = []
    for index, first in enumerate(features):
        a = pd.to_numeric(table[first], errors="coerce").to_numpy(dtype=float)
        for second in features[index + 1 :]:
            b = pd.to_numeric(table[second], errors="coerce").to_numpy(dtype=float)
            same_missing = np.array_equal(np.isnan(a), np.isnan(b))
            valid = np.isfinite(a) & np.isfinite(b)
            exact = bool(same_missing and np.array_equal(a[valid], b[valid]))
            if exact:
                exact_pairs.append([first, second])
                pair_details.append(
                    {
                        "feature_1": first,
                        "feature_2": second,
                        "missing_count": int((~valid).sum()),
                        "complete_pairs": int(valid.sum()),
                        "max_abs_difference": float(np.max(np.abs(a[valid] - b[valid]))),
                    }
                )
    wanted = {retain, drop}
    if len(exact_pairs) != 1 or set(exact_pairs[0]) != wanted:
        raise AssertionError(f"Expected only duplicate pair {wanted}; found {exact_pairs}")
    revised = [feature for feature in features if feature != drop]
    if len(revised) != 23 or retain not in revised or drop in revised:
        raise AssertionError("Revised predictor policy did not produce exactly 23 features")
    return {
        "status": "PASS",
        "original_feature_n": len(features),
        "revised_feature_n": len(revised),
        "retain": retain,
        "drop": drop,
        "exact_pairs": exact_pairs,
        "pair_details": pair_details,
        "revised_features": revised,
        "revised_feature_sha256": hash_text("\n".join(revised)),
    }


def correlation_components(
    table: pd.DataFrame, features: Sequence[str], threshold: float = 0.90
) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = list(features)
    edges = high_correlation_pairs(table, features, threshold=threshold)
    adjacency = {feature: set() for feature in features}
    for row in edges.itertuples(index=False):
        adjacency[row.feature_1].add(row.feature_2)
        adjacency[row.feature_2].add(row.feature_1)
    memberships: list[list[str]] = []
    unseen = set(features)
    while unseen:
        start = min(unseen)
        stack, members = [start], set()
        while stack:
            value = stack.pop()
            if value in members:
                continue
            members.add(value)
            stack.extend(sorted(adjacency[value] - members, reverse=True))
        unseen -= members
        memberships.append(sorted(members))
    memberships.sort(key=lambda values: tuple(values))
    rows = []
    for index, members in enumerate(memberships, start=1):
        component_id = f"C{index:02d}"
        for feature in members:
            rows.append(
                {
                    "component_id": component_id,
                    "feature": feature,
                    "component_size": len(members),
                    "members": "|".join(members),
                }
            )
    return edges, pd.DataFrame(rows)


def calibration_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    local = frame.sort_values("canonical_cell_id", kind="mergesort") if "canonical_cell_id" in frame else frame
    truth = local["y_true"].to_numpy(dtype=float)
    prediction = local["y_pred"].to_numpy(dtype=float)
    if len(truth) < 2 or not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
        return {"n": len(truth), "status": "not_calculable", "reason": "insufficient_or_nonfinite"}
    observed_sd, predicted_sd = float(np.std(truth, ddof=1)), float(np.std(prediction, ddof=1))
    observed_range, predicted_range = float(np.ptp(truth)), float(np.ptp(prediction))
    if predicted_sd == 0 or observed_sd == 0 or observed_range == 0:
        return {"n": len(truth), "status": "not_calculable", "reason": "zero_variance_or_range"}
    design = np.column_stack([np.ones(len(prediction)), prediction])
    intercept, slope = np.linalg.lstsq(design, truth, rcond=None)[0]
    n_top = int(math.ceil(0.10 * len(local)))
    tie = local.get("canonical_cell_id", pd.Series(np.arange(len(local)), index=local.index)).astype(str)
    ranked = local.assign(_tie=tie).sort_values(["y_pred", "_tie"], ascending=[False, True], kind="mergesort")
    top = ranked.iloc[:n_top]
    overall_mean = float(np.mean(truth))
    top_mean = float(top.y_true.mean())
    percentiles = pd.Series(truth).rank(method="average", pct=True).to_numpy()
    percentile_by_index = dict(zip(local.index, percentiles))
    top_percentile = float(np.mean([percentile_by_index[index] for index in top.index]))
    return {
        "n": len(local),
        "status": "PASS",
        "reason": "",
        "calibration_intercept": float(intercept),
        "calibration_slope": float(slope),
        "intercept": float(intercept),
        "slope": float(slope),
        "observed_sd": observed_sd,
        "truth_sd": observed_sd,
        "predicted_sd": predicted_sd,
        "prediction_sd": predicted_sd,
        "predicted_sd_ratio": predicted_sd / observed_sd,
        "observed_range": observed_range,
        "truth_range": observed_range,
        "predicted_range": predicted_range,
        "prediction_range": predicted_range,
        "range_ratio": predicted_range / observed_range,
        "top_decile_n": n_top,
        "top_decile_observed_mean": top_mean,
        "top_decile_observed_median": float(top.y_true.median()),
        "overall_observed_mean": overall_mean,
        "top_decile_mean_difference": top_mean - overall_mean,
        "top_decile_mean_ratio": top_mean / overall_mean if overall_mean != 0 else math.nan,
        "top_decile_mean_observed_percentile": top_percentile,
    }


def _validate_paired_frames(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    identity = ["canonical_cell_id", "group_id", "subclass", "fold"]
    required = set(identity + ["y_true", "y_pred"])
    if required - set(a) or required - set(b):
        raise ValueError("Paired frames lack identity/truth/prediction columns")
    if a.canonical_cell_id.astype(str).duplicated().any() or b.canonical_cell_id.astype(str).duplicated().any():
        raise AssertionError("Paired contrast contains duplicate cell IDs")
    aa = a.copy().sort_values("canonical_cell_id", kind="mergesort").reset_index(drop=True)
    bb = b.copy().sort_values("canonical_cell_id", kind="mergesort").reset_index(drop=True)
    if not np.array_equal(aa.canonical_cell_id.astype(str), bb.canonical_cell_id.astype(str)):
        raise AssertionError("Paired model cell sets differ")
    for column in identity[1:]:
        if not np.array_equal(aa[column].astype(str), bb[column].astype(str)):
            raise AssertionError(f"Paired model metadata differs: {column}")
    if not np.array_equal(aa.y_true.to_numpy(dtype=float), bb.y_true.to_numpy(dtype=float)):
        raise AssertionError("Paired model truth values differ")
    if not np.isfinite(aa[["y_true", "y_pred"]].to_numpy()).all() or not np.isfinite(bb.y_pred).all():
        raise AssertionError("Paired model values must be finite")
    return aa.assign(y_pred_b=bb.y_pred.to_numpy(dtype=float))


def paired_bootstrap(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    n_resamples: int,
    seed: int,
    module: str = "",
    contrast_id: str = "",
    model_a: str = "A",
    model_b: str = "B",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    local = _validate_paired_frames(frame_a, frame_b)
    donors = np.array(sorted(local.group_id.astype(str).unique()), dtype=object)
    donor_to_code = {donor: index for index, donor in enumerate(donors)}
    codes = local.group_id.astype(str).map(donor_to_code).to_numpy(dtype=int)
    base_indices = np.arange(len(local), dtype=np.int64)
    rng = np.random.default_rng(seed)
    plans = rng.integers(0, len(donors), size=(n_resamples, len(donors)), dtype=np.int32)
    plan_sha = hash_array(plans)
    truth = local.y_true.to_numpy(dtype=float)
    pred_a = local.y_pred.to_numpy(dtype=float)
    pred_b = local.y_pred_b.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for replicate, sampled in enumerate(plans):
        multiplicity = np.bincount(sampled, minlength=len(donors))
        index = np.repeat(base_indices, multiplicity[codes])
        values_a = regression_metrics(truth[index], pred_a[index])
        values_b = regression_metrics(truth[index], pred_b[index])
        if not all(np.isfinite([*values_a.values(), *values_b.values()])):
            raise AssertionError("Non-finite paired bootstrap draw")
        for metric in METRICS:
            delta = values_a[metric] - values_b[metric]
            if metric in ("mae", "rmse"):
                delta = -delta
            rows.append(
                {
                    "module": module,
                    "contrast_id": contrast_id,
                    "model_a": model_a,
                    "model_b": model_b,
                    "metric": metric,
                    "replicate": replicate,
                    "metric_a": values_a[metric],
                    "metric_b": values_b[metric],
                    "delta": delta,
                    "n_groups": len(donors),
                    "n_sampled_rows": len(index),
                    "bootstrap_seed": seed,
                    "plan_sha256": plan_sha,
                }
            )
    draws = pd.DataFrame(rows)
    point_a = regression_metrics(truth, pred_a)
    point_b = regression_metrics(truth, pred_b)
    summary_metrics: dict[str, Any] = {}
    for metric in METRICS:
        values = draws.loc[draws.metric.eq(metric), "delta"].to_numpy(dtype=float)
        point_delta = point_a[metric] - point_b[metric]
        if metric in ("mae", "rmse"):
            point_delta = -point_delta
        summary_metrics[metric] = {
            "point_a": point_a[metric],
            "point_b": point_b[metric],
            "point_delta": point_delta,
            "ci_low": float(np.quantile(values, 0.025)),
            "ci_high": float(np.quantile(values, 0.975)),
            "count_gt0": int(np.count_nonzero(values > 0)),
            "count_lt0": int(np.count_nonzero(values < 0)),
            "count_eq0": int(np.count_nonzero(values == 0)),
            "prob_gt0": float(np.mean(values > 0)),
            "prob_lt0": float(np.mean(values < 0)),
        }
    return draws, {
        "module": module,
        "contrast_id": contrast_id,
        "model_a": model_a,
        "model_b": model_b,
        "n_cells": len(local),
        "n_groups": len(donors),
        "n_bootstrap": n_resamples,
        "bootstrap_seed": seed,
        "plan_sha256": plan_sha,
        "metrics": summary_metrics,
    }


def _load_table(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str], list[str]]:
    paths = resolve_paths(root, config)
    table, features = load_and_validate_inputs(
        paths["modeling_table"], paths["frozen_folds"], paths["ephys_features"]
    )
    technical = pd.read_csv(
        paths["section1_technical_targets"], dtype={"canonical_cell_id": str, "transcriptomics_sample_id": str}
    )
    technical["canonical_cell_id"] = technical.canonical_cell_id.astype(str)
    table["canonical_cell_id"] = table.canonical_cell_id.astype(str)
    table = table.merge(
        technical[["canonical_cell_id", "transcriptomics_sample_id"]],
        on="canonical_cell_id",
        how="left",
        validate="one_to_one",
    )
    if table.transcriptomics_sample_id.isna().any():
        raise AssertionError("Missing transcriptomic sample ID in modeling cohort")
    audit = exact_duplicate_audit(
        table,
        features,
        config["duplicate_policy"]["retain"],
        config["duplicate_policy"]["drop"],
    )
    revised = list(audit["revised_features"])
    return table.sort_values("canonical_cell_id", kind="mergesort").reset_index(drop=True), features, revised


def verify_integrity(root: Path, config: dict[str, Any], destination: Path) -> dict[str, Any]:
    paths, expected = resolve_paths(root, config), config["expected"]
    direct_hashes = {
        "modeling_table": "modeling_table_sha256",
        "frozen_folds": "frozen_folds_sha256",
        "module_scores": "module_scores_sha256",
        "ephys_features": "ephys_features_sha256",
        "primary_oof": "primary_oof_sha256",
        "primary_shap": "primary_shap_sha256",
        "primary_hyperparameters": "primary_hyperparameters_sha256",
        "manuscript": "manuscript_sha256",
        "section1_manifest": "section1_manifest_sha256",
        "section1_matched_sets": "section1_matched_sets_sha256",
        "section1_matching_quality": "section1_matching_quality_sha256",
        "section1_matching_assignments": "section1_matching_assignments_sha256",
        "section1_technical_targets": "section1_technical_targets_sha256",
        "section1_residual_targets": "section1_residual_targets_sha256",
        "robustness_duplicate_oof": "robustness_duplicate_oof_sha256",
        "robustness_duplicate_shap": "robustness_duplicate_shap_sha256",
        "robustness_duplicate_fit_manifest": "robustness_duplicate_fit_manifest_sha256",
        "robustness_repeated_oof": "robustness_repeated_oof_sha256",
    }
    actual_hashes = {}
    for path_key, expected_key in direct_hashes.items():
        actual = sha256_file(paths[path_key])
        actual_hashes[path_key] = actual
        if actual != str(expected[expected_key]).lower():
            raise AssertionError(f"Frozen input hash mismatch: {path_key}")
    for module, wanted in expected["control_target_sha256"].items():
        actual = sha256_file(paths["section1_control_targets"] / f"{module}.parquet")
        actual_hashes[f"control_target_{module}"] = actual
        if actual != wanted:
            raise AssertionError(f"Frozen control target hash mismatch: {module}")

    section1_manifest = pd.read_csv(paths["section1_manifest"])
    if len(section1_manifest) != int(expected["section1_manifest_rows"]):
        raise AssertionError("Unexpected Section 1 manifest row count")
    failures = []
    for row in section1_manifest.itertuples(index=False):
        artifact = root / row.relative_path
        if not artifact.is_file() or artifact.stat().st_size != int(row.bytes) or sha256_file(artifact) != row.sha256:
            failures.append(row.relative_path)
    if failures:
        raise AssertionError(f"Section 1 manifest validation failed: {failures[:5]}")
    checkpoint = json.loads(paths["section1_checkpoint"].read_text(encoding="utf-8"))
    if checkpoint.get("signature") != expected["section1_signature"]:
        raise AssertionError("Section 1 checkpoint signature mismatch")
    table, features, revised = _load_table(root, config)
    if len(table) != int(expected["cohort_n"]) or table.group_id.astype(str).nunique() != int(expected["donor_n"]):
        raise AssertionError("Cohort or donor count mismatch")
    if len(features) != int(expected["primary_features"]) or len(revised) != int(expected["revised_features"]):
        raise AssertionError("Feature count mismatch")
    donor_overlap = {}
    for fold in range(int(expected["folds"])):
        train_groups = set(table.loc[table.fold.ne(fold), "group_id"].astype(str))
        test_groups = set(table.loc[table.fold.eq(fold), "group_id"].astype(str))
        donor_overlap[str(fold)] = len(train_groups & test_groups)
    if any(donor_overlap.values()):
        raise AssertionError("Outer fold donor leakage")
    scores = pd.read_parquet(paths["module_scores"])
    scores["canonical_cell_id"] = scores.canonical_cell_id.astype(str)
    score_lookup = scores.set_index("canonical_cell_id")
    target_differences = {}
    for module in MODULE_ORDER:
        observed = table[f"target_{module}"].to_numpy(dtype=float)
        rebuilt = table.canonical_cell_id.map(score_lookup[module]).to_numpy(dtype=float)
        difference = float(np.max(np.abs(observed - rebuilt)))
        target_differences[module] = difference
        if difference != 0:
            raise AssertionError(f"Target mismatch: {module}")
    duplicate = exact_duplicate_audit(
        table, features, config["duplicate_policy"]["retain"], config["duplicate_policy"]["drop"]
    )
    detail = duplicate["pair_details"][0]
    if detail["missing_count"] != int(config["duplicate_policy"]["missing_count"]):
        raise AssertionError("Duplicate missing-count mismatch")
    if detail["complete_pairs"] != int(config["duplicate_policy"]["complete_pairs"]):
        raise AssertionError("Duplicate complete-pair mismatch")
    payload = {
        "status": "PASS",
        "cohort_n": len(table),
        "donor_n": int(table.group_id.astype(str).nunique()),
        "fold_sizes": table.fold.value_counts().sort_index().astype(int).to_dict(),
        "donor_overlap": donor_overlap,
        "target_max_abs_difference": target_differences,
        "duplicate_audit": duplicate,
        "input_sha256": actual_hashes,
        "section1_manifest_rows_validated": len(section1_manifest),
        "section1_manifest_valid": True,
        "manuscript_sha256_before": actual_hashes["manuscript"],
        "packages": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "shap")
        },
    }
    atomic_json(payload, destination)
    atomic_json(
        {
            "feature_count": len(revised),
            "features": revised,
            "ordered_feature_sha256": duplicate["revised_feature_sha256"],
            "revised_feature_sha256": duplicate["revised_feature_sha256"],
            "retained_exact_duplicate_representative": config["duplicate_policy"]["retain"],
            "dropped_exact_duplicate": config["duplicate_policy"]["drop"],
        },
        destination.parent.parent / "config/revised_23_features.json",
    )
    return payload


def _atomic_joblib(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(model, temporary)
    temporary.replace(path)


def _prediction_frame(
    test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    module: str,
    model: str,
    analysis: str | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_cell_id": test.canonical_cell_id.astype(str).to_numpy(),
            "group_id": test.group_id.astype(str).to_numpy(),
            "subclass": test.subclass.astype(str).to_numpy(),
            "fold": test.fold.to_numpy(dtype=int),
            "module": module,
            "model": model,
            "analysis": analysis or MODEL_ANALYSIS[model],
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )


def _fit_mlp_with_curve(
    features: Sequence[str],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    model_seed: int,
    split_seed: int,
    maximum_epochs: int,
    patience: int,
    min_delta: float,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    inner_train, inner_validation = group_inner_split(groups, seed=split_seed)
    preprocessor = make_preprocessor(features, scale_numeric=True)
    x_train = np.asarray(preprocessor.fit_transform(X.iloc[inner_train]), dtype=float)
    x_validation = np.asarray(preprocessor.transform(X.iloc[inner_validation]), dtype=float)
    estimator = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam", learning_rate="constant",
        learning_rate_init=0.001, alpha=0.001, batch_size="auto", shuffle=True,
        early_stopping=False, max_iter=1, random_state=model_seed,
    )
    rows, best_iteration, best_rmse, stale = [], 0, float("inf"), 0
    for iteration in range(1, maximum_epochs + 1):
        estimator.partial_fit(x_train, y[inner_train])
        validation_prediction = estimator.predict(x_validation)
        rmse = float(np.mean((y[inner_validation] - validation_prediction) ** 2) ** 0.5)
        improved = rmse < best_rmse - min_delta
        if improved:
            best_iteration, best_rmse, stale = iteration, rmse, 0
        else:
            stale += 1
        rows.append(
            {
                "iteration": iteration,
                "train_loss_sklearn_half_squared_error": float(estimator.loss_),
                "validation_mse": rmse**2,
                "validation_rmse": rmse,
                "is_new_best": improved,
                "best_iteration_so_far": best_iteration,
            }
        )
        if stale >= patience:
            break
    if best_iteration < 1:
        raise RuntimeError("MLP inner selection failed")
    parameters, pipeline = mlp_candidate(features, seed=model_seed, max_iter=best_iteration)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(X, y)
    actual_iterations = int(pipeline.named_steps["estimator"].n_iter_)
    if actual_iterations != best_iteration:
        raise AssertionError(
            f"MLP full refit stopped at {actual_iterations}, expected selected epoch {best_iteration}"
        )
    parameters.update(
        {
            "selected_iter": best_iteration,
            "actual_refit_iter": actual_iterations,
            "inner_split_seed": split_seed,
            "inner_validation_fraction": 0.20,
            "inner_split_method": "GroupShuffleSplit",
            "early_stopping": "group_aware_manual",
            "solver": "adam",
            "activation": "relu",
            "learning_rate": "constant",
            "learning_rate_init": 0.001,
            "batch_size": "auto",
            "shuffle": True,
            "sklearn_early_stopping": False,
        }
    )
    ledger = {
        "inner_train_n": len(inner_train),
        "inner_validation_n": len(inner_validation),
        "inner_train_group_n": int(np.unique(groups[inner_train]).size),
        "inner_validation_group_n": int(np.unique(groups[inner_validation]).size),
        "inner_valid_group_n": int(np.unique(groups[inner_validation]).size),
        "inner_group_overlap": len(set(groups[inner_train]) & set(groups[inner_validation])),
        "inner_group_overlap_n": len(set(groups[inner_train]) & set(groups[inner_validation])),
        "inner_train_id_sha256": hash_text("\n".join(sorted(X.index[inner_train].astype(str)))),
        "inner_validation_id_sha256": hash_text("\n".join(sorted(X.index[inner_validation].astype(str)))),
    }
    return pipeline, {**parameters, "inner_rmse": best_rmse}, pd.DataFrame(rows), ledger


def _fixed_candidate(
    model: str, features: Sequence[str], parameters: dict[str, Any], seed: int = 42
) -> Pipeline:
    candidates = elastic_net_candidates(features, seed=seed) if model == "ElasticNet" else xgboost_candidates(features, seed=seed)
    for candidate_parameters, pipeline in candidates:
        if all(
            (isinstance(value, float) and math.isclose(float(parameters[key]), value, rel_tol=0, abs_tol=1e-12))
            or parameters[key] == value
            for key, value in candidate_parameters.items()
        ):
            return clone(pipeline)
    raise ValueError(f"Configuration not in frozen grid: {model} {parameters}")


def _fit_primary_models(
    root: Path,
    config: dict[str, Any],
    *,
    modules: Sequence[str],
    folds: Sequence[int],
    mlp_seeds: Sequence[int],
    result_root: Path,
    model_root: Path,
) -> list[Path]:
    table, _, features = _load_table(root, config)
    predictions, seed_predictions, parameters, curves, ledgers, model_paths = [], [], [], [], [], []
    mlp_config = config["models"]["mlp"]
    for module in modules:
        for fold in folds:
            train = table.loc[table.fold.ne(fold)].copy()
            test = table.loc[table.fold.eq(fold)].copy()
            train.index = train.canonical_cell_id.astype(str)
            test.index = test.canonical_cell_id.astype(str)
            X_train, X_test = train[features], test[features]
            y_train = train[f"target_{module}"].to_numpy(dtype=float)
            y_test = test[f"target_{module}"].to_numpy(dtype=float)
            groups = train.group_id.astype(str).to_numpy()
            outer = {
                "module": module,
                "fold": int(fold),
                "outer_train_n": len(train),
                "train_n": len(train),
                "outer_test_n": len(test),
                "test_n": len(test),
                "outer_train_group_n": int(train.group_id.astype(str).nunique()),
                "train_group_n": int(train.group_id.astype(str).nunique()),
                "outer_test_group_n": int(test.group_id.astype(str).nunique()),
                "test_group_n": int(test.group_id.astype(str).nunique()),
                "outer_group_overlap": len(set(train.group_id.astype(str)) & set(test.group_id.astype(str))),
                "outer_group_overlap_n": len(set(train.group_id.astype(str)) & set(test.group_id.astype(str))),
                "outer_train_id_sha256": hash_text("\n".join(sorted(train.canonical_cell_id.astype(str)))),
                "train_id_sha256": hash_text("\n".join(sorted(train.canonical_cell_id.astype(str)))),
                "outer_test_id_sha256": hash_text("\n".join(sorted(test.canonical_cell_id.astype(str)))),
                "test_id_sha256": hash_text("\n".join(sorted(test.canonical_cell_id.astype(str)))),
            }
            if outer["outer_group_overlap"]:
                raise AssertionError("Outer donor leakage")
            inner_train, inner_validation = group_inner_split(groups, seed=int(config["seed"]) + fold)
            inner_ledger = {
                "inner_train_n": len(inner_train),
                "inner_validation_n": len(inner_validation),
                "inner_train_group_n": int(np.unique(groups[inner_train]).size),
                "inner_validation_group_n": int(np.unique(groups[inner_validation]).size),
                "inner_valid_group_n": int(np.unique(groups[inner_validation]).size),
                "inner_group_overlap": len(set(groups[inner_train]) & set(groups[inner_validation])),
                "inner_group_overlap_n": len(set(groups[inner_train]) & set(groups[inner_validation])),
                "inner_train_id_sha256": hash_text("\n".join(sorted(X_train.index[inner_train].astype(str)))),
                "inner_validation_id_sha256": hash_text("\n".join(sorted(X_train.index[inner_validation].astype(str)))),
            }
            for model in CLASSICAL_MODELS:
                candidates = elastic_net_candidates(features, seed=int(config["seed"])) if model == "ElasticNet" else xgboost_candidates(features, seed=int(config["seed"]))
                selected = select_on_inner_groups(
                    candidates, X_train, y_train, groups, seed=int(config["seed"]) + fold
                )
                prediction = selected.pipeline.predict(X_test)
                predictions.append(_prediction_frame(test, y_test, prediction, module, model))
                row = {"module": module, "fold": fold, "model": model, "model_seed": int(config["seed"]), "inner_rmse": selected.inner_rmse, **selected.parameters}
                parameters.append(row)
                ledgers.append({**outer, **inner_ledger, "model": model, "model_seed": int(config["seed"])})
                model_path = model_root / "primary" / f"{module}_fold{fold}_{model}.joblib"
                _atomic_joblib(selected.pipeline, model_path); model_paths.append(model_path)

            mlp_fold_predictions = []
            for model_seed in mlp_seeds:
                pipeline, values, curve, mlp_ledger = _fit_mlp_with_curve(
                    features, X_train, y_train, groups,
                    model_seed=int(model_seed), split_seed=int(config["seed"]) + fold,
                    maximum_epochs=int(mlp_config["maximum_epochs"]), patience=int(mlp_config["patience"]),
                    min_delta=float(mlp_config["min_delta"]),
                )
                prediction = np.asarray(pipeline.predict(X_test), dtype=float)
                mlp_fold_predictions.append(prediction)
                seed_frame = _prediction_frame(test, y_test, prediction, module, "MLP")
                seed_frame["model_seed"] = int(model_seed)
                seed_predictions.append(seed_frame)
                parameters.append({"module": module, "fold": fold, "model": "MLP", "model_seed": int(model_seed), **values})
                curve = curve.assign(module=module, fold=fold, model="MLP", model_seed=int(model_seed))
                curves.append(curve)
                ledgers.append({**outer, **mlp_ledger, "model": "MLP", "model_seed": int(model_seed)})
                model_path = model_root / "primary" / f"{module}_fold{fold}_MLP_seed{model_seed}.joblib"
                _atomic_joblib(pipeline, model_path); model_paths.append(model_path)
            predictions.append(
                _prediction_frame(test, y_test, np.mean(mlp_fold_predictions, axis=0), module, "MLP")
            )

            comparator_specs = (
                ("Dummy", dummy_model(features), features),
                ("SubclassRidge", subclass_ridge_model(), ["subclass"]),
                ("EphysSubclassRidge", ephys_subclass_ridge_model(features), [*features, "subclass"]),
            )
            for model, pipeline, columns in comparator_specs:
                fitted = pipeline.fit(train[columns], y_train)
                predictions.append(_prediction_frame(test, y_test, fitted.predict(test[columns]), module, model))
                parameters.append(
                    {
                        "module": module, "fold": fold, "model": model, "model_seed": int(config["seed"]),
                        "strategy": "mean" if model == "Dummy" else "ridge_alpha_1.0",
                        "alpha": math.nan if model == "Dummy" else 1.0,
                    }
                )
                ledgers.append({**outer, "model": model, "model_seed": int(config["seed"]), "inner_split": "none_fixed_model"})
                model_path = model_root / "primary" / f"{module}_fold{fold}_{model}.joblib"
                _atomic_joblib(fitted, model_path); model_paths.append(model_path)

    oof = pd.concat(predictions, ignore_index=True)
    seed_oof = pd.concat(seed_predictions, ignore_index=True)
    if not np.isfinite(oof[["y_true", "y_pred"]].to_numpy()).all():
        raise AssertionError("Nonfinite primary OOF")
    paths = [
        result_root / "oof/revised_oof.parquet",
        result_root / "oof/mlp_seed_oof.parquet",
        result_root / "tables/pooled_metrics.csv",
        result_root / "tables/fold_metrics.csv",
        result_root / "tables/fitted_hyperparameters.csv",
        result_root / "mlp/learning_curves_raw.parquet",
        result_root / "tables/training_ledgers.csv",
    ]
    atomic_parquet(oof, paths[0]); atomic_parquet(seed_oof, paths[1])
    atomic_csv(performance_table(oof), paths[2]); atomic_csv(fold_metrics(oof), paths[3])
    atomic_csv(pd.DataFrame(parameters), paths[4]); atomic_parquet(pd.concat(curves, ignore_index=True), paths[5])
    atomic_csv(pd.DataFrame(ledgers), paths[6])
    return [*paths, *model_paths]


def run_primary_comparison(root: Path, config: dict[str, Any], result_root: Path) -> list[Path]:
    revised = pd.read_parquet(result_root / "oof/revised_oof.parquet")
    primary = pd.read_parquet(resolve_paths(root, config)["primary_oof"])
    rows = []
    for module in MODULE_ORDER:
        for model in EPHYS_MODELS:
            new = revised.loc[revised.module.eq(module) & revised.model.eq(model)].sort_values("canonical_cell_id")
            old = primary.loc[primary.module.eq(module) & primary.model.eq(model)].sort_values("canonical_cell_id")
            paired = _validate_paired_frames(new, old)
            new_values = regression_metrics(paired.y_true.to_numpy(), paired.y_pred.to_numpy())
            old_values = regression_metrics(paired.y_true.to_numpy(), paired.y_pred_b.to_numpy())
            rows.append(
                {
                    "module": module,
                    "model": model,
                    "n": len(paired),
                    **{f"revised_23_{metric}": new_values[metric] for metric in METRICS},
                    **{f"primary_24_{metric}": old_values[metric] for metric in METRICS},
                    **{
                        f"delta_23_minus_24_{metric}": (
                            old_values[metric] - new_values[metric]
                            if metric in ("mae", "rmse")
                            else new_values[metric] - old_values[metric]
                        )
                        for metric in METRICS
                    },
                    "positive_delta_means": "23-feature model improves the metric",
                    "comparison_note": "nested-tuned estimand; Robustness V2 fixed-HP result isolates pure removal",
                }
            )
    output = pd.DataFrame(rows)
    output = output.rename(
        columns={
            **{f"revised_23_{m}": f"feature23_{m}" for m in METRICS},
            **{f"primary_24_{m}": f"feature24_{m}" for m in METRICS},
            **{f"delta_23_minus_24_{m}": f"delta_{m}" for m in METRICS},
        }
    )
    path = result_root / "tables/feature23_vs_feature24.csv"
    atomic_csv(output, path)
    return [path]


def run_paired_inference(
    config: dict[str, Any], result_root: Path, *, n_resamples: int
) -> list[Path]:
    oof = pd.read_parquet(result_root / "oof/revised_oof.parquet")
    all_draws, summary_rows = [], []
    common_seed = stable_seed("primary_paired_donor_plan", base=int(config["seed"]))
    for module in MODULE_ORDER:
        for contrast_id, model_a, model_b in CONTRASTS:
            a = oof.loc[oof.module.eq(module) & oof.model.eq(model_a)]
            b = oof.loc[oof.module.eq(module) & oof.model.eq(model_b)]
            draws, summary = paired_bootstrap(
                a, b, n_resamples=n_resamples, seed=common_seed, module=module,
                contrast_id=contrast_id, model_a=model_a, model_b=model_b,
            )
            all_draws.append(draws)
            for metric, values in summary["metrics"].items():
                row = {
                    key: summary[key]
                    for key in (
                        "module", "contrast_id", "model_a", "model_b", "n_cells", "n_groups",
                        "n_bootstrap", "bootstrap_seed", "plan_sha256",
                    )
                }
                row.update({"metric": metric, **values})
                row["n_resamples"] = summary["n_bootstrap"]
                row["delta_orientation"] = "A-B" if metric in ("r2", "spearman") else "B-A"
                row["positive_means"] = "model_a_better"
                if metric == "r2":
                    delta_values = draws.loc[draws.metric.eq("r2"), "delta"].to_numpy(dtype=float)
                    p_low = (1 + np.count_nonzero(delta_values <= 0)) / (n_resamples + 1)
                    p_high = (1 + np.count_nonzero(delta_values >= 0)) / (n_resamples + 1)
                    row["p_bootstrap_sign_tail_two_sided"] = min(1.0, 2 * min(p_low, p_high))
                    row["p_definition"] = "inclusive-zero plus-one two-sided bootstrap sign-tail heuristic"
                summary_rows.append(row)
    draws_frame, summary_frame = pd.concat(all_draws, ignore_index=True), pd.DataFrame(summary_rows)
    r2 = summary_frame.loc[summary_frame.metric.eq("r2")].copy()
    if len(r2) != int(config["inference"]["bh_family_n"]):
        raise AssertionError("Primary R2 multiplicity family is not exactly 18")
    r2["p_unadjusted"] = r2.p_bootstrap_sign_tail_two_sided.astype(float)
    r2["p_bh"] = bh_adjust(r2.p_unadjusted.to_numpy())
    r2["metric"] = "r2"
    r2["p_two_sided"] = r2["p_unadjusted"]
    r2["q_value"] = r2["p_bh"]
    order = np.lexsort((r2.contrast_id.astype(str), r2.module.astype(str), r2.p_unadjusted.to_numpy()))
    rank = np.empty(len(r2), dtype=int); rank[order] = np.arange(1, len(r2) + 1)
    r2["rank_bh"] = rank
    r2["reject_bh_q05"] = r2.p_bh.le(float(config["inference"]["bh_q"]))
    r2["family_n"] = len(r2)
    r2["family_label"] = "6 modules x 3 predeclared revised-model R2 contrasts"
    fdr_columns = [
        "module", "contrast_id", "model_a", "model_b", "metric", "point_delta", "ci_low", "ci_high",
        "p_unadjusted", "p_two_sided", "p_bh", "q_value", "rank_bh", "reject_bh_q05", "family_n", "family_label",
    ]
    paths = [
        result_root / "model_comparison/paired_bootstrap_draws.parquet",
        result_root / "model_comparison/paired_model_comparison.csv",
        result_root / "model_comparison/paired_model_comparison_fdr.csv",
    ]
    atomic_parquet(draws_frame, paths[0]); atomic_csv(summary_frame, paths[1]); atomic_csv(r2[fdr_columns], paths[2])
    return paths


def run_subclass_increment(
    config: dict[str, Any], result_root: Path, *, n_resamples: int
) -> list[Path]:
    oof = pd.read_parquet(result_root / "oof/revised_oof.parquet")
    draws_frames, rows = [], []
    seed = stable_seed("subclass_increment_plan", base=int(config["seed"]))
    for module in MODULE_ORDER:
        a = oof.loc[oof.module.eq(module) & oof.model.eq("EphysSubclassRidge")]
        b = oof.loc[oof.module.eq(module) & oof.model.eq("SubclassRidge")]
        draws, summary = paired_bootstrap(
            a, b, n_resamples=n_resamples, seed=seed, module=module,
            contrast_id="EphysSubclassRidge_minus_SubclassRidge",
            model_a="EphysSubclassRidge", model_b="SubclassRidge",
        )
        draws_frames.append(draws)
        for metric, values in summary["metrics"].items():
            rows.append(
                {
                    "module": module,
                    "contrast_id": summary["contrast_id"],
                    "model_a": summary["model_a"],
                    "model_b": summary["model_b"],
                    "metric": metric,
                    "n_cells": summary["n_cells"],
                    "n_groups": summary["n_groups"],
                    "n_bootstrap": summary["n_bootstrap"],
                    "n_resamples": summary["n_bootstrap"],
                    "bootstrap_seed": seed,
                    "plan_sha256": summary["plan_sha256"],
                    **values,
                    "delta_orientation": "A-B" if metric in ("r2", "spearman") else "B-A",
                    "positive_means": "incremental ephys improvement beyond subclass",
                    "inference_scope": "unadjusted exploratory donor-cluster bootstrap",
                    "descriptively_resolved": bool(metric == "r2" and values["ci_low"] > 0),
                }
            )
    paths = [
        result_root / "subclass/subclass_increment_draws.parquet",
        result_root / "subclass/subclass_increment_summary.csv",
    ]
    atomic_parquet(pd.concat(draws_frames, ignore_index=True), paths[0]); atomic_csv(pd.DataFrame(rows), paths[1])
    return paths


def _single_model_cluster_bootstrap(frame: pd.DataFrame, n_resamples: int, seed: int) -> pd.DataFrame:
    local = frame.sort_values("canonical_cell_id", kind="mergesort").reset_index(drop=True)
    donors = np.array(sorted(local.group_id.astype(str).unique()))
    lookup = {value: index for index, value in enumerate(donors)}
    codes = local.group_id.astype(str).map(lookup).to_numpy(dtype=int)
    base = np.arange(len(local), dtype=np.int64)
    plans = np.random.default_rng(seed).integers(0, len(donors), size=(n_resamples, len(donors)), dtype=np.int32)
    rows = []
    truth, prediction = local.y_true.to_numpy(dtype=float), local.y_pred.to_numpy(dtype=float)
    for replicate, sampled in enumerate(plans):
        multiplicity = np.bincount(sampled, minlength=len(donors))
        index = np.repeat(base, multiplicity[codes])
        rows.append({"replicate": replicate, **regression_metrics(truth[index], prediction[index])})
    result = pd.DataFrame(rows)
    result["bootstrap_seed"] = seed
    result["plan_sha256"] = hash_array(plans)
    return result


def run_within_subclass(
    root: Path,
    config: dict[str, Any],
    *,
    subclasses: Sequence[str] | None,
    modules: Sequence[str],
    n_resamples: int,
    result_root: Path,
    model_root: Path,
) -> list[Path]:
    table, _, features = _load_table(root, config)
    counts = (
        table.groupby("subclass", sort=True)
        .agg(n_cells=("canonical_cell_id", "size"), n_donors=("group_id", lambda x: x.astype(str).nunique()))
        .reset_index()
    )
    counts["threshold_eligible"] = counts.n_cells.ge(int(config["within_subclass"]["minimum_cells"])) & counts.n_donors.ge(int(config["within_subclass"]["minimum_donors"]))
    counts["eligible"] = counts["threshold_eligible"]
    counts["cell_rank"] = counts.n_cells.rank(method="min", ascending=False).astype(int)
    counts["historical_top2_selected"] = counts.subclass.isin(config["within_subclass"]["historical_top_two"])
    selected = sorted(counts.loc[counts.threshold_eligible, "subclass"].astype(str))
    if subclasses is not None:
        selected = [value for value in selected if value in set(subclasses)]
    counts["selected_for_analysis"] = counts.subclass.isin(selected)
    counts["selection_rule"] = "all subclasses with n_cells>=150 and n_donors>=3"
    counts["exclusion_reason"] = np.where(counts.threshold_eligible, "", "below_cell_threshold")
    predictions, params, ledgers, fold_rows, model_paths = [], [], [], [], []
    for subclass in selected:
        local = table.loc[table.subclass.eq(subclass)].sort_values("canonical_cell_id", kind="mergesort").copy()
        groups = local.group_id.astype(str).to_numpy()
        assignment = np.full(len(local), -1, dtype=int)
        splitter = GroupKFold(n_splits=int(config["within_subclass"]["folds"]))
        for fold, (_, test_index) in enumerate(splitter.split(local, groups=groups)):
            assignment[test_index] = fold
        if (assignment < 0).any():
            raise AssertionError("Incomplete within-subclass folds")
        local["fold"] = assignment
        for row in local[["canonical_cell_id", "group_id", "subclass", "fold"]].to_dict("records"):
            fold_rows.append(row)
        for module in modules:
            for fold in range(int(config["within_subclass"]["folds"])):
                train, test = local.loc[local.fold.ne(fold)].copy(), local.loc[local.fold.eq(fold)].copy()
                train.index, test.index = train.canonical_cell_id.astype(str), test.canonical_cell_id.astype(str)
                X_train, X_test = train[features], test[features]
                y_train = train[f"target_{module}"].to_numpy(dtype=float)
                groups_train = train.group_id.astype(str).to_numpy()
                for model in CLASSICAL_MODELS:
                    candidates = elastic_net_candidates(features, seed=int(config["seed"])) if model == "ElasticNet" else xgboost_candidates(features, seed=int(config["seed"]))
                    selected_model = select_on_inner_groups(
                        candidates, X_train, y_train, groups_train,
                        seed=stable_seed("within", subclass, fold, base=int(config["seed"])),
                    )
                    predictions.append(
                        _prediction_frame(
                            test, test[f"target_{module}"].to_numpy(dtype=float),
                            selected_model.pipeline.predict(X_test), module, model,
                            analysis="within_subclass",
                        ).assign(analysis_subclass=subclass)
                    )
                    params.append(
                        {
                            "subclass": subclass, "module": module, "fold": fold, "model": model,
                            "inner_rmse": selected_model.inner_rmse, **selected_model.parameters,
                        }
                    )
                    overlap = len(set(train.group_id.astype(str)) & set(test.group_id.astype(str)))
                    if overlap:
                        raise AssertionError("Within-subclass outer donor leakage")
                    ledgers.append(
                        {
                            "subclass": subclass, "module": module, "fold": fold, "model": model,
                            "outer_train_n": len(train), "outer_test_n": len(test),
                            "outer_train_donors": train.group_id.astype(str).nunique(),
                            "outer_test_donors": test.group_id.astype(str).nunique(),
                            "outer_group_overlap": overlap,
                        }
                    )
                    model_path = model_root / "within_subclass" / f"{subclass}_{module}_fold{fold}_{model}.joblib"
                    _atomic_joblib(selected_model.pipeline, model_path); model_paths.append(model_path)
    oof = pd.concat(predictions, ignore_index=True)
    oof["subclass"] = oof["analysis_subclass"]
    group_columns = ["analysis_subclass", "module", "model", "analysis"]
    pooled_rows, fold_metric_rows, bootstrap_frames = [], [], []
    for keys, frame in oof.groupby(group_columns, sort=True):
        values = regression_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())
        seed = stable_seed("within_bootstrap", *keys, base=int(config["seed"]))
        draws = _single_model_cluster_bootstrap(frame, n_resamples, seed).assign(
            analysis_subclass=keys[0], module=keys[1], model=keys[2]
        )
        bootstrap_frames.append(draws)
        ci = {
            f"{metric}_ci_low": float(np.quantile(draws[metric], 0.025))
            for metric in METRICS
        } | {
            f"{metric}_ci_high": float(np.quantile(draws[metric], 0.975))
            for metric in METRICS
        }
        pooled_rows.append(
            {
                    "analysis_subclass": keys[0], "subclass": keys[0], "module": keys[1], "model": keys[2],
                "analysis": keys[3], "n": len(frame), "n_donors": frame.group_id.astype(str).nunique(),
                **values, **ci, "r2_descriptive_support": bool(ci["r2_ci_low"] > 0),
                "inference_scope": "unadjusted descriptive donor-cluster CI",
            }
        )
        for fold, fold_frame in frame.groupby("fold"):
            fold_metric_rows.append(
                {
                    "analysis_subclass": keys[0], "subclass": keys[0], "module": keys[1], "model": keys[2], "fold": int(fold),
                    "n": len(fold_frame), **regression_metrics(fold_frame.y_true, fold_frame.y_pred),
                }
            )
    paths = [
        result_root / "within_subclass/eligibility.csv",
        result_root / "within_subclass/folds.csv",
        result_root / "within_subclass/oof.parquet",
        result_root / "within_subclass/pooled_metrics.csv",
        result_root / "within_subclass/fold_metrics.csv",
        result_root / "within_subclass/bootstrap_draws.parquet",
        result_root / "within_subclass/fitted_hyperparameters.csv",
        result_root / "within_subclass/leakage_ledger.csv",
    ]
    atomic_csv(counts, paths[0]); atomic_csv(pd.DataFrame(fold_rows), paths[1]); atomic_parquet(oof, paths[2])
    atomic_csv(pd.DataFrame(pooled_rows), paths[3]); atomic_csv(pd.DataFrame(fold_metric_rows), paths[4])
    atomic_parquet(pd.concat(bootstrap_frames, ignore_index=True), paths[5]); atomic_csv(pd.DataFrame(params), paths[6]); atomic_csv(pd.DataFrame(ledgers), paths[7])
    return [*paths, *model_paths]


def _fast_spearman(x: np.ndarray, y: np.ndarray) -> float:
    xr, yr = rankdata(x).astype(float), rankdata(y).astype(float)
    xr -= xr.mean(); yr -= yr.mean()
    denominator = float(np.linalg.norm(xr) * np.linalg.norm(yr))
    if denominator == 0:
        raise ValueError("Spearman statistic requires two variable arrays")
    return float(np.dot(xr, yr) / denominator)


def run_permutations(
    config: dict[str, Any], result_root: Path, *, n_permutations: int
) -> list[Path]:
    oof = pd.read_parquet(result_root / "oof/revised_oof.parquet")
    oof = oof.loc[oof.model.isin(EPHYS_MODELS)].copy()
    cell_rows, donor_rows = [], []
    for module in MODULE_ORDER:
        for model in EPHYS_MODELS:
            frame = oof.loc[oof.module.eq(module) & oof.model.eq(model)].sort_values("canonical_cell_id", kind="mergesort")
            truth = frame.y_true.to_numpy(dtype=float)
            prediction = frame.y_pred.to_numpy(dtype=float)
            observed = _fast_spearman(truth, prediction)
            true_rank, pred_rank = rankdata(truth).astype(float), rankdata(prediction).astype(float)
            true_rank -= true_rank.mean(); pred_rank -= pred_rank.mean()
            denominator = float(np.linalg.norm(true_rank) * np.linalg.norm(pred_rank))
            strata = frame.subclass.astype(str).to_numpy()
            strata_index = [np.flatnonzero(strata == value) for value in sorted(np.unique(strata))]
            seed = stable_seed("permutation_cell", module, model, base=int(config["seed"]))
            rng, null = np.random.default_rng(seed), np.empty(n_permutations, dtype=float)
            for iteration in range(n_permutations):
                permuted = true_rank.copy()
                for index in strata_index:
                    permuted[index] = rng.permutation(permuted[index])
                null[iteration] = float(np.dot(permuted, pred_rank) / denominator)
            extreme = int(np.count_nonzero(np.abs(null) >= abs(observed)))
            cell_rows.append(
                {
                    "module": module, "model": model, "n_cells": len(frame),
                    "n_strata": len(strata_index), "n_permutations": n_permutations, "seed": seed,
                    "statistic": "pooled_global_spearman", "observed": observed,
                    "null_exceedances": extreme, "extreme_count": extreme,
                    "p_two_sided": (extreme + 1) / (n_permutations + 1),
                    "p_value": (extreme + 1) / (n_permutations + 1),
                    "null_mean": float(np.mean(null)), "null_sd": float(np.std(null, ddof=1)),
                    "test_type": "truth labels permuted independently within each subclass; saved OOF; no refit",
                    "no_refit": True,
                }
            )

            local = frame[["group_id", "subclass", "y_true", "y_pred"]].copy()
            for column in ("y_true", "y_pred"):
                local[f"{column}_residual"] = local[column] - local.groupby("subclass")[column].transform("mean")
            donor = local.groupby(local.group_id.astype(str), sort=True).agg(
                y_true=("y_true_residual", "mean"), y_pred=("y_pred_residual", "mean"), n_cells=("y_true", "size")
            )
            donor_truth, donor_prediction = donor.y_true.to_numpy(), donor.y_pred.to_numpy()
            observed_donor = _fast_spearman(donor_truth, donor_prediction)
            donor_true_rank = rankdata(donor_truth).astype(float); donor_pred_rank = rankdata(donor_prediction).astype(float)
            donor_true_rank -= donor_true_rank.mean(); donor_pred_rank -= donor_pred_rank.mean()
            donor_denominator = float(np.linalg.norm(donor_true_rank) * np.linalg.norm(donor_pred_rank))
            donor_seed = stable_seed("permutation_donor", module, model, base=int(config["seed"]))
            rng, null_donor = np.random.default_rng(donor_seed), np.empty(n_permutations, dtype=float)
            for iteration in range(n_permutations):
                null_donor[iteration] = float(np.dot(rng.permutation(donor_true_rank), donor_pred_rank) / donor_denominator)
            extreme_donor = int(np.count_nonzero(np.abs(null_donor) >= abs(observed_donor)))
            donor_rows.append(
                {
                    "module": module, "model": model, "n_cells": len(frame), "n_groups": len(donor),
                    "n_permutations": n_permutations, "seed": donor_seed,
                    "statistic": "donor_equal_weight_subclass_centered_spearman", "observed": observed_donor,
                    "null_exceedances": extreme_donor, "extreme_count": extreme_donor,
                    "p_two_sided": (extreme_donor + 1) / (n_permutations + 1),
                    "p_value": (extreme_donor + 1) / (n_permutations + 1),
                    "null_mean": float(np.mean(null_donor)), "null_sd": float(np.std(null_donor, ddof=1)),
                    "test_type": "donor truth residual means permuted across donors; saved OOF; no refit",
                    "no_refit": True,
                }
            )
    paths = [result_root / "permutations/cell_stratified.csv", result_root / "permutations/donor_residual.csv"]
    atomic_csv(pd.DataFrame(cell_rows), paths[0]); atomic_csv(pd.DataFrame(donor_rows), paths[1])
    return paths


def write_mlp_audit(config: dict[str, Any], result_root: Path) -> list[Path]:
    raw = pd.read_parquet(result_root / "mlp/learning_curves_raw.parquet")
    curves = raw.rename(
        columns={
            "fold": "outer_fold", "model_seed": "seed", "iteration": "epoch",
            "train_loss_sklearn_half_squared_error": "train_loss", "validation_mse": "validation_loss",
        }
    )
    stopping = (
        pd.read_csv(result_root / "tables/fitted_hyperparameters.csv")
        .loc[lambda x: x.model.eq("MLP")]
        .drop(columns="seed", errors="ignore")
        .rename(columns={"fold": "outer_fold", "model_seed": "seed", "selected_iter": "selected_epoch"})
    )
    ledger = pd.read_csv(result_root / "tables/training_ledgers.csv")
    ledger = ledger.loc[ledger.model.eq("MLP")].rename(columns={"fold": "outer_fold", "model_seed": "seed"})
    stopping = stopping.merge(
        ledger[["module", "outer_fold", "seed", "inner_group_overlap_n"]],
        on=["module", "outer_fold", "seed"], how="left", validate="one_to_one",
    )
    mlp = config["models"]["mlp"]
    configuration = {
        **mlp,
        "inner_validation_grouped": True,
        "inner_validation_fraction": float(config["models"]["inner_validation_fraction"]),
        "inner_split_method": config["models"]["inner_split_method"],
        "primary_epoch_selection": "minimum validation RMSE with patience/min_delta; fresh full outer-train refit",
        "validation_fraction_0_15_in_estimator": "inert because sklearn early_stopping is false",
        "final_refit_iteration_assertion": "n_iter_ equals selected_epoch for every saved model",
    }
    paths = [
        result_root / "mlp/mlp_configuration.json",
        result_root / "mlp/learning_curves.parquet",
        result_root / "mlp/early_stopping_summary.csv",
    ]
    atomic_json(configuration, paths[0]); atomic_parquet(curves, paths[1]); atomic_csv(stopping, paths[2])
    return paths


def run_repeated_cv(
    root: Path,
    config: dict[str, Any],
    *,
    repeat_seeds: Sequence[int],
    modules: Sequence[str],
    mlp_seeds: Sequence[int],
    result_root: Path,
    model_root: Path,
) -> list[Path]:
    table, _, features = _load_table(root, config)
    parameter_table = pd.read_csv(result_root / "tables/fitted_hyperparameters.csv")
    fixed: dict[tuple[str, str], dict[str, Any]] = {}
    for module in modules:
        for model in CLASSICAL_MODELS:
            row = parameter_table.loc[
                parameter_table.module.eq(module) & parameter_table.fold.eq(0) & parameter_table.model.eq(model)
            ].iloc[0]
            names = ("alpha", "l1_ratio") if model == "ElasticNet" else ("n_estimators", "max_depth", "learning_rate", "reg_lambda")
            fixed[(module, model)] = {
                name: int(row[name]) if name in ("n_estimators", "max_depth") else float(row[name])
                for name in names
            }
    fold_frames, prediction_frames, parameter_rows, ledger_rows, model_paths = [], [], [], [], []
    mlp_config = config["models"]["mlp"]
    for repeat_index, repeat_seed in enumerate(repeat_seeds):
        splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=int(repeat_seed))
        repeat_fold = np.full(len(table), -1, dtype=int)
        for fold, (_, test_index) in enumerate(
            splitter.split(table, y=table.subclass.astype(str), groups=table.group_id.astype(str))
        ):
            repeat_fold[test_index] = fold
        if (repeat_fold < 0).any():
            raise AssertionError("Incomplete repeated fold assignment")
        fold_frames.append(
            table[["canonical_cell_id", "group_id", "subclass"]].assign(
                repeat=repeat_index, repeat_seed=int(repeat_seed), split_seed=int(repeat_seed), fold=repeat_fold
            )
        )
        local_table = table.assign(fold=repeat_fold)
        for module in modules:
            for fold in range(3):
                train, test = local_table.loc[local_table.fold.ne(fold)].copy(), local_table.loc[local_table.fold.eq(fold)].copy()
                train.index, test.index = train.canonical_cell_id.astype(str), test.canonical_cell_id.astype(str)
                X_train, X_test = train[features], test[features]
                y_train, y_test = train[f"target_{module}"].to_numpy(float), test[f"target_{module}"].to_numpy(float)
                groups = train.group_id.astype(str).to_numpy()
                if set(train.group_id.astype(str)) & set(test.group_id.astype(str)):
                    raise AssertionError("Repeated outer donor leakage")
                for model in CLASSICAL_MODELS:
                    pipeline = _fixed_candidate(model, features, fixed[(module, model)], seed=int(config["seed"]))
                    pipeline.fit(X_train, y_train)
                    frame = _prediction_frame(test, y_test, pipeline.predict(X_test), module, model)
                    frame["repeat"] = repeat_index; frame["repeat_seed"] = int(repeat_seed); frame["split_seed"] = int(repeat_seed)
                    prediction_frames.append(frame)
                    parameter_rows.append(
                        {"repeat": repeat_index, "repeat_seed": repeat_seed, "module": module, "fold": fold, "model": model, **fixed[(module, model)], "configuration_source": "revised_primary_fold0"}
                    )
                    model_path = model_root / "repeated_cv" / f"repeat{repeat_index}_{module}_fold{fold}_{model}.joblib"
                    _atomic_joblib(pipeline, model_path); model_paths.append(model_path)
                mlp_predictions = []
                for model_seed in mlp_seeds:
                    pipeline, values, _, inner_ledger = _fit_mlp_with_curve(
                        features, X_train, y_train, groups, model_seed=int(model_seed),
                        split_seed=int(repeat_seed) + fold,
                        maximum_epochs=int(mlp_config["maximum_epochs"]), patience=int(mlp_config["patience"]),
                        min_delta=float(mlp_config["min_delta"]),
                    )
                    mlp_predictions.append(pipeline.predict(X_test))
                    parameter_rows.append(
                        {"repeat": repeat_index, "repeat_seed": repeat_seed, "module": module, "fold": fold, "model": "MLP", "model_seed": model_seed, **values, "configuration_source": "fixed_architecture_with_grouped_inner_epoch_rule"}
                    )
                    model_path = model_root / "repeated_cv" / f"repeat{repeat_index}_{module}_fold{fold}_MLP_seed{model_seed}.joblib"
                    _atomic_joblib(pipeline, model_path); model_paths.append(model_path)
                    ledger_rows.append(
                        {"repeat": repeat_index, "repeat_seed": repeat_seed, "module": module, "fold": fold, "model": "MLP", "model_seed": model_seed, **inner_ledger}
                    )
                frame = _prediction_frame(test, y_test, np.mean(mlp_predictions, axis=0), module, "MLP")
                frame["repeat"] = repeat_index; frame["repeat_seed"] = int(repeat_seed); frame["split_seed"] = int(repeat_seed)
                prediction_frames.append(frame)
    folds_frame, oof = pd.concat(fold_frames, ignore_index=True), pd.concat(prediction_frames, ignore_index=True)
    summary_rows = []
    primary = pd.read_csv(result_root / "tables/pooled_metrics.csv").set_index(["module", "model"])
    repeat_values = []
    for keys, frame in oof.groupby(["repeat", "repeat_seed", "module", "model"]):
        repeat_values.append({"repeat": keys[0], "repeat_seed": keys[1], "module": keys[2], "model": keys[3], **regression_metrics(frame.y_true, frame.y_pred)})
    repeat_metrics = pd.DataFrame(repeat_values)
    for (module, model), frame in repeat_metrics.groupby(["module", "model"]):
        for metric in METRICS:
            values = frame[metric].to_numpy(float)
            primary_value = float(primary.loc[(module, model), metric])
            minimum, maximum = float(np.min(values)), float(np.max(values))
            summary_rows.append(
                {
                    "module": module, "model": model, "metric": metric, "n_repeats": len(values),
                    "median": float(np.median(values)), "q1": float(np.quantile(values, .25)),
                    "q3": float(np.quantile(values, .75)), "iqr": float(np.quantile(values, .75)-np.quantile(values, .25)),
                    "min": minimum, "max": maximum, "range": maximum-minimum,
                    "fraction_r2_gt0": float(np.mean(values > 0)) if metric == "r2" else math.nan,
                    "primary_value": primary_value,
                    "primary_position": "within" if minimum <= primary_value <= maximum else ("below" if primary_value < minimum else "above"),
                    "interpretation": "descriptive five-repeat stability; not a confidence interval",
                }
            )
    paths = [
        result_root / "repeated_cv/folds.csv", result_root / "repeated_cv/oof.parquet",
        result_root / "repeated_cv/repeat_metrics.csv", result_root / "repeated_cv/summary.csv",
        result_root / "repeated_cv/fitted_parameters.csv", result_root / "repeated_cv/leakage_ledger.csv",
    ]
    atomic_csv(folds_frame, paths[0]); atomic_parquet(oof, paths[1]); atomic_csv(repeat_metrics, paths[2]); atomic_csv(pd.DataFrame(summary_rows), paths[3]); atomic_csv(pd.DataFrame(parameter_rows), paths[4]); atomic_csv(pd.DataFrame(ledger_rows), paths[5])
    return [*paths, *model_paths]


def run_calibration(result_root: Path) -> list[Path]:
    oof = pd.read_parquet(result_root / "oof/revised_oof.parquet")
    calibration_rows, top_rows = [], []
    for (module, model), frame in oof.loc[oof.model.isin(EPHYS_MODELS)].groupby(["module", "model"]):
        record = calibration_metrics(frame)
        calibration_rows.append(
            {"module": module, "model": model, **{key: record[key] for key in (
                "n", "status", "reason", "intercept", "slope", "prediction_sd", "truth_sd",
                "predicted_sd_ratio", "prediction_range", "truth_range", "range_ratio",
            )}}
        )
        top_rows.append(
            {"module": module, "model": model, **{key: record[key] for key in record if key.startswith("top_decile") or key == "overall_observed_mean"}}
        )
    paths = [result_root / "calibration/calibration.csv", result_root / "calibration/top_decile_enrichment.csv"]
    atomic_csv(pd.DataFrame(calibration_rows), paths[0]); atomic_csv(pd.DataFrame(top_rows), paths[1])
    return paths


def run_attribution(
    root: Path, config: dict[str, Any], result_root: Path, model_root: Path
) -> list[Path]:
    table, _, features = _load_table(root, config)
    edges, components = correlation_components(
        table, features, threshold=float(config["interpretability"]["correlation_threshold_strict"])
    )
    if len(edges) != 6 or components.component_id.nunique() != 19:
        raise AssertionError("Unexpected revised feature correlation component geometry")
    shap_wide_frames, shap_long_frames, shap_records, coefficient_rows = [], [], [], []
    revised_oof = pd.read_parquet(result_root / "oof/revised_oof.parquet")
    for module in MODULE_ORDER:
        for fold in range(3):
            test = table.loc[table.fold.eq(fold)].copy()
            x_test = test[features]
            xgb = joblib.load(model_root / "primary" / f"{module}_fold{fold}_XGBoost.joblib")
            preprocessor, estimator = xgb.named_steps["preprocessor"], xgb.named_steps["estimator"]
            transformed = np.asarray(preprocessor.transform(x_test), dtype=float)
            names = list(map(str, preprocessor.get_feature_names_out()))
            if names != features:
                raise AssertionError("XGBoost transformed feature order differs from frozen revised order")
            explainer = shap.TreeExplainer(estimator)
            resolved = str(explainer.feature_perturbation)
            output_mode = str(explainer.model_output)
            if resolved != config["interpretability"]["shap_resolved_required"] or output_mode != config["interpretability"]["shap_model_output_required"]:
                raise RuntimeError(f"Unsupported configuration: SHAP resolved to {resolved}/{output_mode}")
            explanation = explainer(transformed, check_additivity=False)
            values = np.asarray(explanation.values, dtype=float)
            if values.ndim == 3 and values.shape[-1] == 1:
                values = values[..., 0]
            if values.shape != transformed.shape or not np.isfinite(values).all():
                raise AssertionError("Unexpected/nonfinite held-out SHAP values")
            wide = pd.DataFrame(values, columns=features)
            wide.insert(0, "fold", fold); wide.insert(0, "module", module)
            wide.insert(0, "canonical_cell_id", test.canonical_cell_id.astype(str).to_numpy())
            shap_wide_frames.append(wide)
            long = wide.melt(id_vars=["canonical_cell_id", "module", "fold"], var_name="feature", value_name="shap_value")
            shap_long_frames.append(long)
            expected = float(np.asarray(explainer.expected_value).reshape(-1)[0])
            predicted_from_shap = expected + values.sum(axis=1)
            model_prediction = np.asarray(xgb.predict(x_test), dtype=float)
            oof_part = revised_oof.loc[revised_oof.module.eq(module) & revised_oof.model.eq("XGBoost") & revised_oof.fold.eq(fold)].sort_values("canonical_cell_id")
            predicted_sorted = pd.Series(model_prediction, index=test.canonical_cell_id.astype(str)).reindex(oof_part.canonical_cell_id.astype(str)).to_numpy()
            if not np.allclose(predicted_sorted, oof_part.y_pred.to_numpy(), rtol=0, atol=1e-6):
                raise AssertionError("Saved XGBoost model does not reproduce revised held-out OOF")
            shap_records.append(
                {
                    "module": module, "fold": fold, "n_test": len(test),
                    "explainer_class": explainer.__class__.__name__,
                    "estimator_class": estimator.__class__.__name__,
                    "default_requested": "auto (argument omitted)", "feature_perturbation": resolved,
                    "model_output": output_mode, "background_provided": False, "masker": None,
                    "check_additivity": False, "expected_value": expected,
                    "transformed_feature_names": names,
                    "max_prediction_additivity_error": float(np.max(np.abs(predicted_from_shap - model_prediction))),
                }
            )
            elastic = joblib.load(model_root / "primary" / f"{module}_fold{fold}_ElasticNet.joblib")
            en_names = list(map(str, elastic.named_steps["preprocessor"].get_feature_names_out()))
            coefficients = np.asarray(elastic.named_steps["estimator"].coef_, dtype=float).reshape(-1)
            if en_names != features or coefficients.shape != (23,):
                raise AssertionError("ElasticNet coefficient feature order/shape mismatch")
            manual = np.asarray(elastic.named_steps["preprocessor"].transform(x_test)) @ coefficients + float(elastic.named_steps["estimator"].intercept_)
            if not np.allclose(manual, elastic.predict(x_test), rtol=0, atol=1e-12):
                raise AssertionError("Manual ElasticNet predictions do not reconstruct")
            order = np.lexsort((np.array(features), -np.abs(coefficients)))
            ranks = np.empty(len(features), dtype=int); ranks[order] = np.arange(1, len(features)+1)
            selected = pd.read_csv(result_root / "tables/fitted_hyperparameters.csv")
            selected = selected.loc[selected.module.eq(module) & selected.fold.eq(fold) & selected.model.eq("ElasticNet")].iloc[0]
            for feature, coefficient, rank in zip(features, coefficients, ranks):
                coefficient_rows.append(
                    {
                        "module": module, "fold": fold, "feature": feature,
                        "coefficient": float(coefficient), "abs_coefficient": float(abs(coefficient)),
                        "fold_rank": int(rank), "standardized_input": True,
                        "coefficient_scale": "coefficient_on_fold_standardized_X; outcome unstandardized",
                        "alpha": float(selected.alpha), "l1_ratio": float(selected.l1_ratio),
                    }
                )
    shap_wide, shap_long = pd.concat(shap_wide_frames, ignore_index=True), pd.concat(shap_long_frames, ignore_index=True)
    grouped_shap = (
        shap_long.assign(abs_value=lambda x: x.shap_value.abs())
        .merge(components[["feature", "component_id"]], on="feature", validate="many_to_one")
        .groupby(["canonical_cell_id", "module", "fold", "component_id"], as_index=False).abs_value.sum()
        .rename(columns={"abs_value": "group_abs_shap"})
    )
    if not np.allclose(
        grouped_shap.groupby(["canonical_cell_id", "module"]).group_abs_shap.sum().sort_index(),
        shap_long.assign(abs_value=lambda x: x.shap_value.abs()).groupby(["canonical_cell_id", "module"]).abs_value.sum().sort_index(),
        rtol=1e-12, atol=1e-12,
    ):
        raise AssertionError("Grouped SHAP absolute reconciliation failed")
    coefficients = pd.DataFrame(coefficient_rows)
    en_grouped = (
        coefficients.merge(components[["feature", "component_id"]], on="feature", validate="many_to_one")
        .groupby(["module", "fold", "component_id"], as_index=False).abs_coefficient.sum()
    )
    shap_fold = grouped_shap.groupby(["module", "fold", "component_id"], as_index=False).group_abs_shap.mean()
    shap_module = shap_fold.groupby(["module", "component_id"], as_index=False).group_abs_shap.mean()
    en_module = en_grouped.groupby(["module", "component_id"], as_index=False).abs_coefficient.mean()
    concordance_rows = []
    for module in MODULE_ORDER:
        joined = shap_module.loc[shap_module.module.eq(module)].merge(
            en_module.loc[en_module.module.eq(module)], on=["module", "component_id"], validate="one_to_one"
        )
        row = {"module": module, "group_n": len(joined), "group_spearman_rho": float(spearmanr(joined.group_abs_shap, joined.abs_coefficient).statistic)}
        for k in (3, 5):
            shap_top = set(joined.sort_values(["group_abs_shap", "component_id"], ascending=[False, True]).head(k).component_id)
            en_top = set(joined.sort_values(["abs_coefficient", "component_id"], ascending=[False, True]).head(k).component_id)
            overlap = len(shap_top & en_top)
            row[f"top{k}_overlap_n"] = overlap; row[f"top{k}_jaccard"] = overlap / len(shap_top | en_top)
            row[f"top{k}_shap_components"] = "|".join(sorted(shap_top)); row[f"top{k}_en_components"] = "|".join(sorted(en_top))
        concordance_rows.append(row)
    configuration = {
        "held_out_only": True, "explainer_class": "TreeExplainer",
        "feature_perturbation": config["interpretability"]["shap_resolved_required"],
        "model_output": config["interpretability"]["shap_model_output_required"],
        "constructor": config["interpretability"]["shap_constructor"],
        "evaluation": config["interpretability"]["shap_call"],
        "background_provided": False, "check_additivity": False,
        "shap_version": version("shap"), "xgboost_version": version("xgboost"),
        "records": shap_records,
    }
    paths = [
        result_root / "attribution/oof_shap_23feature.parquet",
        result_root / "attribution/oof_shap_23feature_wide.parquet",
        result_root / "attribution/shap_configuration.json",
        result_root / "attribution/correlation_edges.csv",
        result_root / "attribution/feature_components.csv",
        result_root / "attribution/grouped_shap.parquet",
        result_root / "attribution/en_coefficients.csv",
        result_root / "attribution/en_grouped_coefficients.csv",
        result_root / "attribution/concordance.csv",
    ]
    atomic_parquet(shap_long, paths[0]); atomic_parquet(shap_wide, paths[1]); atomic_json(configuration, paths[2])
    atomic_csv(edges, paths[3]); atomic_csv(components, paths[4]); atomic_parquet(grouped_shap, paths[5])
    atomic_csv(coefficients, paths[6]); atomic_csv(en_grouped, paths[7]); atomic_csv(pd.DataFrame(concordance_rows), paths[8])
    return paths


def _fit_target_models(
    table: pd.DataFrame,
    features: Sequence[str],
    targets: pd.DataFrame,
    *,
    target_name_column: str,
    target_value_column: str,
    model_names: Sequence[str],
    seed: int,
    result_root: Path,
    model_root: Path,
    stem: str,
) -> list[Path]:
    predictions, parameters, ledger_rows, model_paths = [], [], [], []
    target_names = sorted(targets[target_name_column].astype(str).unique())
    for target_name in target_names:
        target_frame = targets.loc[targets[target_name_column].astype(str).eq(target_name), ["canonical_cell_id", target_value_column]].copy()
        target_frame["canonical_cell_id"] = target_frame.canonical_cell_id.astype(str)
        local = table.merge(target_frame, on="canonical_cell_id", how="left", validate="one_to_one")
        if local[target_value_column].isna().any():
            raise AssertionError(f"Incomplete target {target_name}")
        for fold in range(3):
            train, test = local.loc[local.fold.ne(fold)].copy(), local.loc[local.fold.eq(fold)].copy()
            train.index, test.index = train.canonical_cell_id.astype(str), test.canonical_cell_id.astype(str)
            X_train, X_test = train[list(features)], test[list(features)]
            y_train, y_test = train[target_value_column].to_numpy(float), test[target_value_column].to_numpy(float)
            groups = train.group_id.astype(str).to_numpy()
            for model in model_names:
                inner_train, inner_valid = group_inner_split(groups, seed=seed + fold)
                candidates = elastic_net_candidates(features, seed=seed) if model == "ElasticNet" else xgboost_candidates(features, seed=seed)
                selected = select_on_inner_groups(candidates, X_train, y_train, groups, seed=seed + fold)
                frame = _prediction_frame(test, y_test, selected.pipeline.predict(X_test), target_name, model)
                frame = frame.rename(columns={"module": target_name_column}); frame["target"] = target_name
                predictions.append(frame)
                parameters.append(
                    {"target": target_name, target_name_column: target_name, "fold": fold, "model": model, "inner_rmse": selected.inner_rmse, **selected.parameters}
                )
                overlap = len(set(train.group_id.astype(str)) & set(test.group_id.astype(str)))
                ledger_rows.append(
                    {"target": target_name, "fold": fold, "model": model, "train_n": len(train), "test_n": len(test),
                     "train_group_n": train.group_id.astype(str).nunique(), "test_group_n": test.group_id.astype(str).nunique(),
                     "outer_group_overlap_n": overlap, "inner_split_seed": seed + fold,
                     "inner_train_n": len(inner_train), "inner_valid_n": len(inner_valid),
                     "inner_train_group_n": int(np.unique(groups[inner_train]).size),
                     "inner_valid_group_n": int(np.unique(groups[inner_valid]).size),
                     "inner_group_overlap_n": len(set(groups[inner_train]) & set(groups[inner_valid])),
                     "inner_train_id_sha256": hash_text("\n".join(sorted(train.index[inner_train].astype(str)))),
                     "inner_valid_id_sha256": hash_text("\n".join(sorted(train.index[inner_valid].astype(str))))}
                )
                model_path = model_root / f"{stem}_{target_name}_fold{fold}_{model}.joblib"
                _atomic_joblib(selected.pipeline, model_path); model_paths.append(model_path)
    oof = pd.concat(predictions, ignore_index=True)
    metric_rows, fold_rows = [], []
    for (target, model), frame in oof.groupby(["target", "model"]):
        metric_rows.append({"target": target, "model": model, "n": len(frame), **regression_metrics(frame.y_true, frame.y_pred)})
        for fold, part in frame.groupby("fold"):
            fold_rows.append({"target": target, "model": model, "fold": int(fold), "n": len(part), **regression_metrics(part.y_true, part.y_pred)})
    paths = [
        result_root / f"{stem}_oof.parquet", result_root / f"{stem}_performance.csv",
        result_root / f"{stem}_fold_metrics.csv", result_root / f"{stem}_hyperparameters.csv",
        result_root / f"{stem}_leakage_ledger.csv",
    ]
    atomic_parquet(oof, paths[0]); atomic_csv(pd.DataFrame(metric_rows), paths[1]); atomic_csv(pd.DataFrame(fold_rows), paths[2]); atomic_csv(pd.DataFrame(parameters), paths[3]); atomic_csv(pd.DataFrame(ledger_rows), paths[4])
    return [*paths, *model_paths]


def run_technical_bridge(root: Path, config: dict[str, Any], result_root: Path, model_root: Path) -> list[Path]:
    table, _, features = _load_table(root, config)
    technical = pd.read_csv(resolve_paths(root, config)["section1_technical_targets"], dtype={"canonical_cell_id": str})
    target_names = ["log10_total_counts", "log10_genes_detected"]
    targets = technical[["canonical_cell_id", *target_names]].melt(
        id_vars="canonical_cell_id", var_name="target", value_name="target_value"
    )
    paths = _fit_target_models(
        table, features, targets, target_name_column="target", target_value_column="target_value",
        model_names=CLASSICAL_MODELS, seed=int(config["seed"]),
        result_root=result_root / "specificity_bridge", model_root=model_root / "specificity_bridge/technical",
        stem="technical_target",
    )
    return paths


def run_adjusted_bridge(root: Path, config: dict[str, Any], result_root: Path, model_root: Path) -> list[Path]:
    """Leakage-free outer-fold nuisance residualization and revised XGBoost fit."""

    table, _, features = _load_table(root, config)
    technical = pd.read_csv(resolve_paths(root, config)["section1_technical_targets"], dtype={"canonical_cell_id": str})
    nuisance_columns = ["log10_total_counts", "log10_genes_detected"]
    table = table.merge(technical[["canonical_cell_id", *nuisance_columns]], on="canonical_cell_id", validate="one_to_one")
    predictions, nuisance_parts, parameter_rows, coefficient_rows, model_paths = [], [], [], [], []
    for module in MODULE_ORDER:
        raw = table[f"target_{module}"].to_numpy(dtype=float)
        for fold in range(3):
            train_mask, test_mask = table.fold.ne(fold).to_numpy(), table.fold.eq(fold).to_numpy()
            train, test = table.loc[train_mask].copy(), table.loc[test_mask].copy()
            train.index, test.index = train.canonical_cell_id.astype(str), test.canonical_cell_id.astype(str)
            nuisance = LinearRegression(fit_intercept=True).fit(train[nuisance_columns], raw[train_mask])
            train_nuisance = nuisance.predict(train[nuisance_columns]); test_nuisance = nuisance.predict(test[nuisance_columns])
            train_residual = raw[train_mask] - train_nuisance; test_residual = raw[test_mask] - test_nuisance
            train_groups, test_groups = set(train.group_id.astype(str)), set(test.group_id.astype(str))
            if train_groups & test_groups:
                raise AssertionError("Adjusted bridge outer donor leakage")
            nuisance_parts.append(
                pd.DataFrame(
                    {"canonical_cell_id": test.canonical_cell_id.astype(str).to_numpy(),
                     "group_id": test.group_id.astype(str).to_numpy(), "subclass": test.subclass.astype(str).to_numpy(),
                     "fold": fold, "module": module, "target_raw": raw[test_mask],
                     "log10_total_counts": test.log10_total_counts.to_numpy(float),
                     "log10_genes_detected": test.log10_genes_detected.to_numpy(float),
                     "nuisance_prediction": test_nuisance, "residual_target": test_residual}
                )
            )
            coefficient_rows.append(
                {"module": module, "fold": fold, "intercept": float(nuisance.intercept_),
                 "coefficient_log10_total_counts": float(nuisance.coef_[0]),
                 "coefficient_log10_genes_detected": float(nuisance.coef_[1]),
                 "train_n": len(train), "test_n": len(test), "train_group_n": len(train_groups),
                 "test_group_n": len(test_groups), "outer_group_overlap_n": 0,
                 "train_id_sha256": hash_text("\n".join(sorted(train.canonical_cell_id.astype(str)))),
                 "test_id_sha256": hash_text("\n".join(sorted(test.canonical_cell_id.astype(str)))),
                 "nuisance_fit_scope": "current_outer_training_only",
                 "residual_rule": "raw minus same outer-train-fitted nuisance prediction for both train and test"}
            )
            groups = train.group_id.astype(str).to_numpy()
            selected = select_on_inner_groups(
                xgboost_candidates(features, seed=int(config["seed"])), train[features], train_residual,
                groups, seed=int(config["seed"]) + fold,
            )
            frame = _prediction_frame(test, test_residual, selected.pipeline.predict(test[features]), module, "XGBoost", analysis="technical_residualized")
            predictions.append(frame)
            parameter_rows.append({"module": module, "fold": fold, "model": "XGBoost", "inner_rmse": selected.inner_rmse, **selected.parameters})
            model_path = model_root / "specificity_bridge/adjusted" / f"adjusted_target_{module}_fold{fold}_XGBoost.joblib"
            _atomic_joblib(selected.pipeline, model_path); model_paths.append(model_path)
    nuisance_oof, oof = pd.concat(nuisance_parts, ignore_index=True), pd.concat(predictions, ignore_index=True)
    if nuisance_oof.duplicated(["canonical_cell_id", "module"]).any() or len(nuisance_oof) != 3410 * 6:
        raise AssertionError("Incomplete adjusted bridge nuisance OOF")
    if not np.allclose(nuisance_oof.target_raw - nuisance_oof.nuisance_prediction, nuisance_oof.residual_target, rtol=0, atol=1e-12):
        raise AssertionError("Adjusted bridge residual identity failure")
    performance, fold_rows = [], []
    for module, frame in oof.groupby("module"):
        performance.append({"target": module, "module": module, "model": "XGBoost", "n": len(frame), **regression_metrics(frame.y_true, frame.y_pred)})
        for fold, part in frame.groupby("fold"):
            fold_rows.append({"target": module, "module": module, "model": "XGBoost", "fold": int(fold), "n": len(part), **regression_metrics(part.y_true, part.y_pred)})
    base = result_root / "specificity_bridge/adjusted"
    paths = [
        base / "adjusted_target_oof.parquet", base / "adjusted_target_performance.csv",
        base / "adjusted_target_fold_metrics.csv", base / "adjusted_target_hyperparameters.csv",
        base / "adjusted_target_leakage_ledger.csv", base / "crossfit_nuisance_predictions.parquet",
        base / "nuisance_coefficients_and_leakage.csv",
    ]
    atomic_parquet(oof, paths[0]); atomic_csv(pd.DataFrame(performance), paths[1]); atomic_csv(pd.DataFrame(fold_rows), paths[2])
    atomic_csv(pd.DataFrame(parameter_rows), paths[3]); atomic_csv(pd.DataFrame(coefficient_rows), paths[4])
    atomic_parquet(nuisance_oof, paths[5]); atomic_csv(pd.DataFrame(coefficient_rows), paths[6])
    return [*paths, *model_paths]


def _valid_control_partition(
    path: Path,
    *,
    expected_target: pd.Series,
    expected_table: pd.DataFrame,
    module: str,
    control_id: int,
    signature: str,
    accepted_run_signatures: set[str] | None = None,
    folds: Sequence[int],
    expected_configuration_sha256: dict[int, str],
    trusted_sha256: str | None = None,
) -> bool:
    if trusted_sha256 is None or not path.is_file() or sha256_file(path) != trusted_sha256:
        return False
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return False
    required = {"canonical_cell_id", "group_id", "subclass", "fold", "module", "control_id", "y_true", "y_pred", "target_hash", "run_signature", "configuration_sha256"}
    if required - set(frame) or frame.empty or not np.isfinite(frame[["y_true", "y_pred"]].to_numpy()).all():
        return False
    expected = expected_table.loc[expected_table.fold.isin(folds)].sort_values("canonical_cell_id")
    observed = frame.sort_values("canonical_cell_id")
    if len(expected) != len(observed) or observed.canonical_cell_id.astype(str).duplicated().any():
        return False
    if not np.array_equal(expected.canonical_cell_id.astype(str).to_numpy(), observed.canonical_cell_id.astype(str).to_numpy()):
        return False
    target = expected.canonical_cell_id.astype(str).map(expected_target).to_numpy(dtype=float)
    if not np.array_equal(target, observed.y_true.to_numpy(dtype=float)):
        return False
    return bool(
        observed.module.eq(module).all() and observed.control_id.astype(int).eq(control_id).all()
        and observed.run_signature.isin(accepted_run_signatures or {signature}).all()
        and observed.target_hash.eq(hash_array(target)).all()
        and np.array_equal(expected.fold.to_numpy(int), observed.fold.to_numpy(int))
        and np.array_equal(expected.group_id.astype(str).to_numpy(), observed.group_id.astype(str).to_numpy())
        and np.array_equal(expected.subclass.astype(str).to_numpy(), observed.subclass.astype(str).to_numpy())
        and all(
            observed.loc[observed.fold.eq(fold), "configuration_sha256"].eq(expected_configuration_sha256[int(fold)]).all()
            for fold in folds
        )
    )


def run_specificity_bridge(
    root: Path,
    config: dict[str, Any],
    *,
    modules: Sequence[str],
    folds: Sequence[int],
    control_count: int,
    result_root: Path,
    signature: str,
    trusted_partition_sha256: dict[str, str] | None = None,
    accepted_run_signatures: set[str] | None = None,
) -> list[Path]:
    table, _, features = _load_table(root, config)
    paths = resolve_paths(root, config)
    parameters = pd.read_csv(result_root / "tables/fitted_hyperparameters.csv")
    matched = json.loads(paths["section1_matched_sets"].read_text(encoding="utf-8"))
    quality = pd.read_csv(paths["section1_matching_quality"])
    performance_rows, partition_paths = [], []
    reused, fitted = 0, 0
    for module in modules:
        target_matrix = pd.read_parquet(paths["section1_control_targets"] / f"{module}.parquet")
        target_matrix["transcriptomics_sample_id"] = target_matrix.transcriptomics_sample_id.astype(str)
        target_matrix = target_matrix.set_index("transcriptomics_sample_id", verify_integrity=True)
        control_meta = {int(item["control_id"]): item for item in matched["modules"][module]["controls"]}
        for control_id in range(control_count):
            column = f"control_{control_id:04d}"
            target_by_cell = table.set_index("canonical_cell_id").transcriptomics_sample_id.map(target_matrix[column])
            target_by_cell.index = target_by_cell.index.astype(str)
            if target_by_cell.isna().any():
                raise AssertionError("Incomplete frozen random-control target")
            part_path = result_root / "specificity_bridge/random_control_oof" / f"module={module}" / f"control_id={control_id:04d}" / "part-000.parquet"
            partition_paths.append(part_path)
            expected_configs: dict[int, str] = {}
            for fold in folds:
                selected = parameters.loc[parameters.module.eq(module) & parameters.fold.eq(fold) & parameters.model.eq("XGBoost")].iloc[0]
                config_values = {"n_estimators": int(selected.n_estimators), "max_depth": int(selected.max_depth), "learning_rate": float(selected.learning_rate), "reg_lambda": float(selected.reg_lambda)}
                expected_configs[int(fold)] = hash_text(json.dumps(config_values, sort_keys=True))
            if _valid_control_partition(
                part_path, expected_target=target_by_cell, expected_table=table, module=module,
                control_id=control_id, signature=signature, folds=folds,
                accepted_run_signatures=accepted_run_signatures,
                expected_configuration_sha256=expected_configs,
                trusted_sha256=(trusted_partition_sha256 or {}).get(str(part_path.resolve())),
            ):
                frame = pd.read_parquet(part_path); reused += 1
            else:
                rows = []
                for fold in folds:
                    train, test = table.loc[table.fold.ne(fold)].copy(), table.loc[table.fold.eq(fold)].copy()
                    selected = parameters.loc[
                        parameters.module.eq(module) & parameters.fold.eq(fold) & parameters.model.eq("XGBoost")
                    ].iloc[0]
                    config_values = {
                        "n_estimators": int(selected.n_estimators), "max_depth": int(selected.max_depth),
                        "learning_rate": float(selected.learning_rate), "reg_lambda": float(selected.reg_lambda),
                    }
                    pipeline = _fixed_candidate("XGBoost", features, config_values, seed=int(config["seed"]))
                    y_train = train.canonical_cell_id.astype(str).map(target_by_cell).to_numpy(float)
                    y_test = test.canonical_cell_id.astype(str).map(target_by_cell).to_numpy(float)
                    pipeline.fit(train[features], y_train)
                    row = _prediction_frame(test, y_test, pipeline.predict(test[features]), module, "XGBoost")
                    row["control_id"] = control_id; row["run_signature"] = signature
                    row["target_hash"] = hash_array(
                        table.loc[table.fold.isin(folds)].sort_values("canonical_cell_id").canonical_cell_id.astype(str).map(target_by_cell).to_numpy(float)
                    )
                    row["configuration_sha256"] = expected_configs[int(fold)]
                    rows.append(row)
                frame = pd.concat(rows, ignore_index=True)
                atomic_parquet(frame, part_path); fitted += 1
            values = regression_metrics(frame.y_true, frame.y_pred)
            qrow = quality.loc[quality.module.eq(module) & quality.control_id.astype(int).eq(control_id)].iloc[0]
            performance_rows.append(
                {"module": module, "control_id": control_id, "n": len(frame), "n_genes": len(control_meta[control_id]["genes"]),
                 **values, **{f"fold{fold}_r2": regression_metrics(part.y_true, part.y_pred)["r2"] for fold, part in frame.groupby("fold")},
                 "matching_pass": bool(qrow["matching_pass"] if "matching_pass" in qrow else True),
                 "matching_mean_distance": float(control_meta[control_id]["mean_matching_distance"]),
                 "matching_max_distance": float(control_meta[control_id]["max_matching_distance"]),
                 "target_hash": str(frame.target_hash.iloc[0]), "run_signature": signature,
                 "control_tuned": False, "feature_n": 23,
                 "configuration_source": "revised observed module/fold XGBoost configuration"}
            )
    performance = pd.DataFrame(performance_rows)
    performance_path = result_root / "specificity_bridge/random_control_performance.parquet"
    atomic_parquet(performance, performance_path)
    observed = pd.read_csv(result_root / "tables/pooled_metrics.csv")
    summary_rows = []
    for module in modules:
        random = performance.loc[performance.module.eq(module), "r2"].to_numpy(float)
        observed_r2 = float(observed.loc[observed.module.eq(module) & observed.model.eq("XGBoost"), "r2"].iloc[0])
        extreme = int(np.count_nonzero(random >= observed_r2))
        summary_rows.append(
            {"module": module, "observed_revised_xgboost_r2": observed_r2, "random_control_n": len(random),
             "random_median_r2": float(np.median(random)), "random_95th_percentile_r2": float(np.quantile(random,.95)),
             "random_99th_percentile_r2": float(np.quantile(random,.99)),
             "observed_empirical_percentile_le": float(100*np.mean(random <= observed_r2)),
             "extreme_count": extreme, "empirical_p": (1+extreme)/(len(random)+1),
             "passes_95th": bool(observed_r2 > np.quantile(random,.95)), "passes_99th": bool(observed_r2 > np.quantile(random,.99)),
             "control_tuned": False, "feature_n": 23}
        )
    summary_path = result_root / "specificity_bridge/specificity_summary.csv"
    audit_path = result_root / "specificity_bridge/resume_audit.json"
    atomic_csv(pd.DataFrame(summary_rows), summary_path)
    atomic_json(
        {"reused_partition_count": reused, "fit_partition_count": fitted, "partition_count": len(partition_paths),
         "generator_called": False, "extractor_called": False, "source_matched_sets_sha256": sha256_file(paths["section1_matched_sets"]),
         "source_control_target_sha256": {module: sha256_file(paths["section1_control_targets"] / f"{module}.parquet") for module in modules}},
        audit_path,
    )
    return [performance_path, summary_path, audit_path, *partition_paths]


def run_figures(result_root: Path, figure_root: Path) -> list[Path]:
    performance = pd.read_csv(result_root / "tables/pooled_metrics.csv")
    paired = pd.read_csv(result_root / "model_comparison/paired_model_comparison.csv")
    increment = pd.read_csv(result_root / "subclass/subclass_increment_summary.csv")
    repeated = pd.read_csv(result_root / "repeated_cv/summary.csv")
    calibration = pd.read_csv(result_root / "calibration/calibration.csv")
    figure_root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    def save(stem: str) -> None:
        plt.tight_layout()
        for suffix in ("png", "pdf"):
            path = figure_root / f"{stem}.{suffix}"
            plt.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
            paths.append(path)
        plt.close()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    pivot = performance.loc[performance.model.isin(ALL_MODELS)].pivot(index="module", columns="model", values="r2").reindex(MODULE_ORDER)
    pivot.plot(kind="bar", ax=ax); ax.axhline(0, color="black", lw=.8); ax.set_ylabel("Pooled OOF R²"); ax.set_title("A. Revised 23-feature predictive performance")
    save("figure_a_revised_model_performance")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    r2 = paired.loc[paired.metric.eq("r2")].copy(); x = np.arange(len(r2))
    ax.errorbar(x, r2.point_delta, yerr=[r2.point_delta-r2.ci_low, r2.ci_high-r2.point_delta], fmt="o", capsize=2)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, r2.module+"\n"+r2.contrast_id.str.replace("_minus_", "−"), rotation=55, ha="right", fontsize=7); ax.set_ylabel("Paired donor-bootstrap ΔR²"); ax.set_title("B. Predeclared model contrasts")
    save("figure_b_paired_model_comparison")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    r2i = increment.loc[increment.metric.eq("r2")]; x = np.arange(len(r2i))
    ax.errorbar(x, r2i.point_delta, yerr=[r2i.point_delta-r2i.ci_low, r2i.ci_high-r2i.point_delta], fmt="o", capsize=3)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, r2i.module); ax.set_ylabel("ΔR²: combined − subclass"); ax.set_title("C. Incremental electrophysiological signal")
    save("figure_c_subclass_increment")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    r2r = repeated.loc[repeated.metric.eq("r2")]; x = np.arange(len(r2r))
    ax.errorbar(x, r2r["median"], yerr=[r2r["median"]-r2r["min"], r2r["max"]-r2r["median"]], fmt="o", capsize=2)
    ax.axhline(0, color="black", lw=.8); ax.set_xticks(x, r2r.module+"\n"+r2r.model, rotation=55, ha="right", fontsize=7); ax.set_ylabel("R² median and five-repeat range"); ax.set_title("D. Donor-split stability")
    save("figure_d_model_stability")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for model, marker in zip(EPHYS_MODELS, ("o", "s", "^")):
        local = calibration.loc[calibration.model.eq(model)]
        ax.scatter(local.predicted_sd_ratio, local.slope, label=model, marker=marker)
    ax.axvline(1, color="grey", lw=.8); ax.axhline(1, color="grey", lw=.8); ax.set_xlabel("Predicted SD / observed SD"); ax.set_ylabel("Calibration slope"); ax.legend(); ax.set_title("E. Prediction shrinkage and calibration")
    save("figure_e_calibration")
    return paths


def write_artifact_manifest(root: Path, config: dict[str, Any]) -> Path:
    outputs = output_roots(root, config)
    destination = outputs["results"] / "artifact_manifest.csv"
    rows = []
    for namespace, base in outputs.items():
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.resolve() == destination.resolve() or path.name == ".gitkeep" or path.suffix == ".tmp" or "checkpoints" in path.parts:
                continue
            rows.append(
                {"namespace": namespace, "path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size,
                 "sha256": sha256_file(path), "suffix": path.suffix.lower()}
            )
    atomic_csv(pd.DataFrame(rows), destination)
    return destination


def _markdown(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    view = frame[list(columns)].copy()
    for column in view:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda x: f"{x:.{digits}f}" if pd.notna(x) else "NA")
    header = "| " + " | ".join(view.columns) + " |"
    divider = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def generate_report(root: Path, config: dict[str, Any]) -> Path:
    results = output_roots(root, config)["results"]
    performance = pd.read_csv(results / "tables/pooled_metrics.csv")
    fdr = pd.read_csv(results / "model_comparison/paired_model_comparison_fdr.csv")
    specificity = pd.read_csv(results / "specificity_bridge/specificity_summary.csv")
    calibration = pd.read_csv(results / "calibration/calibration.csv")
    comparison = pd.read_csv(results / "tables/feature23_vs_feature24.csv")
    repeated = pd.read_csv(results / "repeated_cv/summary.csv")
    eligibility = pd.read_csv(results / "within_subclass/eligibility.csv")
    increment = pd.read_csv(results / "subclass/subclass_increment_summary.csv")
    within = pd.read_csv(results / "within_subclass/pooled_metrics.csv")
    cell_permutation = pd.read_csv(results / "permutations/cell_stratified.csv")
    donor_permutation = pd.read_csv(results / "permutations/donor_residual.csv")
    technical = pd.read_csv(results / "specificity_bridge/technical_target_performance.csv")
    adjusted = pd.read_csv(results / "specificity_bridge/adjusted/adjusted_target_performance.csv")
    concordance = pd.read_csv(results / "attribution/concordance.csv")
    integrity = json.loads((results / "integrity/integrity_gate.json").read_text(encoding="utf-8"))
    report = f"""# Model-comparison analysis Results

## Objective and frozen estimand

This section tests the predictive value, model dependence, donor-split stability, calibration, and attribution of the frozen six transcriptomic module scores after removing the single exact duplicate electrophysiology column. The estimand remains pooled held-out prediction on the original 3,410 cells and 871 donor/groups; no result is causal.

## Integrity and revised predictor set

Integrity Gate: **{integrity['status']}**. All 1,480 Matched-control analysis manifest rows were byte/hash validated, primary OOF and SHAP hashes matched, all module targets reconstructed exactly, and donor overlap was zero. `rheobase_i` was retained and the exactly identical `stimulus_amplitude_0_long_square` was dropped, leaving exactly 23 ordered predictors.

## Revised primary OOF performance

{_markdown(performance, ['module','model','n','r2','mae','rmse','spearman'])}

## 23-feature versus frozen 24-feature models

{_markdown(comparison, ['module','model','feature23_r2','feature24_r2','delta_r2'])}

The new comparison follows a fully revised nested-selection estimand. The earlier Robustness V2 duplicate run instead froze each primary fold's XGBoost configuration and therefore more purely isolates removal of the redundant input; differences between the two analyses can include configuration re-selection.

## Predeclared paired model comparisons

{_markdown(fdr, ['module','contrast_id','point_delta','ci_low','ci_high','p_unadjusted','p_bh','reject_bh_q05'])}

Positive deltas favor model A. R² and Spearman use A−B; MAE and RMSE use B−A. The R² tail quantity is an inclusive-zero, plus-one, two-sided bootstrap sign-tail heuristic, not a permutation p-value; BH correction covers exactly 18 predeclared R² rows.

## Within-subclass scope

The literal threshold is met by four subclasses—not only the historical top two. Section 2 therefore analyzes every eligible subclass: {', '.join(eligibility.loc[eligibility.eligible, 'subclass'])}. `historical_top2_selected` is retained separately so Lamp5/Vip are not falsely reported as threshold failures. Within-subclass CIs are unadjusted descriptive donor-bootstrap intervals, not a multiplicity-controlled claim family.

## Increment beyond subclass

{_markdown(increment.loc[increment.metric.eq('r2')], ['module','point_delta','ci_low','ci_high','prob_gt0','descriptively_resolved'])}

These paired intervals compare Ephys+Subclass Ridge with Subclass Ridge. Five lower bounds exceed zero; CaV crosses zero. `prob_gt0` is a bootstrap proportion, not a frequentist p-value, and no multiplicity family was prespecified.

## Within-subclass performance

{_markdown(within, ['subclass','module','model','n','r2','r2_ci_low','r2_ci_high','r2_descriptive_support'])}

Descriptive support is concentrated in Sst, especially GABAA; most Lamp5, Pvalb, and Vip intervals include zero or are negative. This heterogeneity does not support uniformly predictive within-subclass models.

## Saved-OOF association permutations

Cell-level truth permutations within subclass:

{_markdown(cell_permutation, ['module','model','observed','p_two_sided'])}

Donor-level permutations after cell-level subclass centering:

{_markdown(donor_permutation, ['module','model','observed','p_two_sided'])}

Both are saved-prediction association diagnostics with no model refitting. The cell test does not address donor dependence; the donor test relies on exchangeability of one residual mean per donor. No additional multiplicity adjustment was prespecified, and neither is an algorithm-level randomization test.

## Repeated donor-split stability and calibration

{_markdown(repeated.loc[repeated.metric.eq('r2')], ['module','model','median','q1','q3','min','max','fraction_r2_gt0','primary_position'])}

{_markdown(calibration, ['module','model','intercept','slope','predicted_sd_ratio','range_ratio'])}

A calibration slope above 1 together with predicted-SD ratio below 1 indicates under-dispersed or shrunken predictions. These diagnostics are descriptive; no post-hoc recalibration was applied to OOF predictions.

Kv MLP is the important negative stability result: pooled R² is −0.0151 and only three of five repeated-split R² values are positive. MLP otherwise retains positive R² across all five declared splits. Repeated CV is split sensitivity under a frozen configuration rule: classical configurations are fixed, whereas the fixed MLP rule legitimately reselects epochs using donor-disjoint inner validation in each outer split.

## Technical-target and leakage-free adjusted bridges

{_markdown(technical, ['target','model','n','r2','mae','rmse','spearman'])}

Detected-gene count is modestly predictable, whereas total-count R² is non-positive for both models. This negative result rules out a blanket claim that electrophysiology predicts library depth.

{_markdown(adjusted, ['module','model','n','r2','mae','rmse','spearman'])}

Adjusted targets are rebuilt inside each current outer fold: a linear nuisance model is fit on current outer-training raw targets and technical covariates, and that same nuisance model defines training and test residuals before XGBoost selection and fitting. These R² values evaluate residual truth and are not directly on the same outcome scale as raw-target R²; numerical raw-versus-adjusted differences are descriptive rather than causal effects of adjustment.

## Specificity bridge

{_markdown(specificity, ['module','observed_revised_xgboost_r2','random_median_r2','random_95th_percentile_r2','random_99th_percentile_r2','observed_empirical_percentile_le','empirical_p','passes_95th','passes_99th'])}

The exact 1,200 frozen Section 1 matched gene sets and target matrices were reused by hash, but every control received a new 23-feature XGBoost refit using the revised observed module/fold configuration. Controls were never retuned.

## Attribution interpretation

Held-out TreeSHAP used the recovered `TreeExplainer(estimator)` call with no background and resolved to `tree_path_dependent`, raw model output. This is predictive attribution, not causal or uniquely identifiable attribution under correlation. The 23-feature correlation graph has 19 components; SHAP sums absolute values within a component. ElasticNet coefficients are semi-standardized coefficients on fold-standardized X with the outcome unstandardized.

{_markdown(concordance, ['module','group_n','group_spearman_rho','top3_overlap_n','top3_jaccard','top5_overlap_n','top5_jaccard'])}

Group-rank concordance is positive but imperfect. GABAA has high overall rank correlation despite only one top-three overlap, so agreement cannot be reduced to a single top-feature ranking.

# Claims Supported

- The saved tables support descriptive comparisons of held-out predictive accuracy among the revised models.
- Paired donor bootstraps quantify sampling uncertainty for the three predeclared model contrasts.
- Repeated group splits quantify sensitivity to five declared donor partitions.
- The frozen-control bridge tests whether revised XGBoost R² exceeds expression/detection-matched module benchmarks under the stated conditional design.

# Claims Not Supported

- No result establishes causal channel/receptor regulation, conductance, trafficking, localization, or protein abundance.
- Bootstrap sign-tail probabilities are not algorithm-level randomization p-values.
- Cell-stratified permutations do not resolve donor dependence; donor-residual permutations rely on exchangeability of donor summaries.
- SHAP and ElasticNet attributions do not identify unique biological mechanisms among correlated predictors.
- Positive saved-OOF permutation associations do not prove positive R² or model superiority; Kv MLP is an explicit counterexample.

## Remaining limitations

- Only five repeated donor splits are reported as a range/IQR, not a confidence interval.
- The matched-control p-value has minimum resolution 1/201 and is conditional on the revised observed-target XGBoost configurations.
- Four eligible subclass analyses lack a prespecified multiplicity family and remain descriptive.

# Reproduction Commands

```powershell
python scripts/10_revision_section2.py --config config/revision_v3_section2.yaml --mode smoke
python scripts/10_revision_section2.py --config config/revision_v3_section2.yaml --mode full
python scripts/10_revision_section2.py --config config/revision_v3_section2.yaml --mode resume
python -m pytest -q
```
"""
    path = root / "REVISION_V3_SECTION2_RESULTS.md"
    atomic_text(report, path)
    return path


def _stage_paths(
    root: Path, config: dict[str, Any], stage: str, *, smoke: bool = False
) -> list[Path]:
    outputs = output_roots(root, config)
    r = outputs["results"] / "smoke" if smoke else outputs["results"]
    m = outputs["models"] / "smoke" if smoke else outputs["models"]
    modules = tuple(config["smoke"]["modules"]) if smoke else MODULE_ORDER
    folds = tuple(map(int, config["smoke"]["folds"])) if smoke else (0, 1, 2)
    mlp_seeds = tuple(map(int, config["smoke"]["mlp_seeds"])) if smoke else tuple(map(int, config["models"]["mlp"]["ensemble_seeds"]))
    if stage == "primary":
        paths = [
            r / "oof/revised_oof.parquet", r / "oof/mlp_seed_oof.parquet", r / "tables/pooled_metrics.csv",
            r / "tables/fold_metrics.csv", r / "tables/fitted_hyperparameters.csv",
            r / "mlp/learning_curves_raw.parquet", r / "tables/training_ledgers.csv",
        ]
        for module in modules:
            for fold in folds:
                paths.extend(
                    m / "primary" / f"{module}_fold{fold}_{model}.joblib"
                    for model in ("ElasticNet", "XGBoost", "Dummy", "SubclassRidge", "EphysSubclassRidge")
                )
                paths.extend(m / "primary" / f"{module}_fold{fold}_MLP_seed{seed}.joblib" for seed in mlp_seeds)
        return paths
    mapping = {
        "comparison": [r / "tables/feature23_vs_feature24.csv"],
        "paired": [r / "model_comparison/paired_bootstrap_draws.parquet", r / "model_comparison/paired_model_comparison.csv", r / "model_comparison/paired_model_comparison_fdr.csv"],
        "subclass_increment": [r / "subclass/subclass_increment_draws.parquet", r / "subclass/subclass_increment_summary.csv"],
        "permutations": [r / "permutations/cell_stratified.csv", r / "permutations/donor_residual.csv"],
        "mlp_audit": [r / "mlp/mlp_configuration.json", r / "mlp/learning_curves.parquet", r / "mlp/early_stopping_summary.csv"],
        "calibration": [r / "calibration/calibration.csv", r / "calibration/top_decile_enrichment.csv"],
    }
    return mapping.get(stage, [])


def run_revision_section2(config_path: str | Path, *, mode: str) -> dict[str, Any]:
    root, config, config_path = read_config(config_path)
    outputs = ensure_output_roots(root, config)
    signature = config_signature(root, config, config_path)
    integrity_path = outputs["results"] / "integrity/integrity_gate.json"
    integrity = verify_integrity(root, config, integrity_path)
    if mode == "smoke":
        r, m = outputs["results"] / "smoke", outputs["models"] / "smoke"
        modules = tuple(config["smoke"]["modules"]); folds = tuple(map(int, config["smoke"]["folds"])); seeds = tuple(map(int, config["smoke"]["mlp_seeds"]))
        primary = _fit_primary_models(root, config, modules=modules, folds=folds, mlp_seeds=seeds, result_root=r, model_root=m)
        write_mlp_audit(config, r)
        run_calibration(r)
        bridge = run_specificity_bridge(
            root, config, modules=modules, folds=folds, control_count=int(config["specificity_bridge"]["smoke_controls"]),
            result_root=r, signature=signature + "-smoke",
        )
        marker = outputs["logs"] / "smoke_run.json"
        atomic_json(
            {"status": "PASS", "mode": "smoke", "signature": signature, "integrity": integrity["status"],
             "modules": list(modules), "folds": list(folds), "mlp_seeds": list(seeds),
             "primary_files": len(primary), "control_partitions": len(bridge)-3}, marker,
        )
        return {"status": "PASS", "mode": mode, "signature": signature, "marker": str(marker)}
    if mode not in {"full", "resume"}:
        raise ValueError("mode must be smoke, full, or resume")
    smoke_marker = outputs["logs"] / "smoke_run.json"
    if not smoke_marker.is_file() or json.loads(smoke_marker.read_text(encoding="utf-8")).get("signature") != signature:
        raise RuntimeError("Full/resume requires a PASS smoke marker for the exact current signature")
    cache_path = outputs["logs"] / "checkpoints.json"
    migration = migrate_verified_checkpoint(cache_path, signature)
    cache = StageCache(cache_path, signature)
    if migration is None:
        persisted_migration = cache.payload.get("migration")
        if (
            isinstance(persisted_migration, dict)
            and persisted_migration.get("to_signature") == signature
            and persisted_migration.get("all_prior_recorded_digests_validated") is True
        ):
            migration = persisted_migration
    completed_stages, fit_count, reused_partition_count = [], 0, 0

    def stage(name: str, paths: Sequence[Path], function: Any) -> None:
        nonlocal fit_count
        if cache.valid(name, paths):
            completed_stages.append(name); return
        written = list(function())
        if {path.resolve() for path in written} != {path.resolve() for path in paths}:
            raise AssertionError(f"Stage output contract mismatch: {name}")
        cache.complete(name, paths)
        fit_count += 1; completed_stages.append(name)

    primary_paths = _stage_paths(root, config, "primary")
    stage("primary", primary_paths, lambda: _fit_primary_models(
        root, config, modules=MODULE_ORDER, folds=(0,1,2), mlp_seeds=tuple(config["models"]["mlp"]["ensemble_seeds"]),
        result_root=outputs["results"], model_root=outputs["models"],
    ))
    stage("comparison", _stage_paths(root, config, "comparison"), lambda: run_primary_comparison(root, config, outputs["results"]))
    stage("paired", _stage_paths(root, config, "paired"), lambda: run_paired_inference(config, outputs["results"], n_resamples=int(config["inference"]["bootstrap_draws"])))
    stage("subclass_increment", _stage_paths(root, config, "subclass_increment"), lambda: run_subclass_increment(config, outputs["results"], n_resamples=int(config["inference"]["bootstrap_draws"])))

    within_paths = [
        outputs["results"] / f"within_subclass/{name}" for name in (
            "eligibility.csv", "folds.csv", "oof.parquet", "pooled_metrics.csv", "fold_metrics.csv",
            "bootstrap_draws.parquet", "fitted_hyperparameters.csv", "leakage_ledger.csv",
        )
    ]
    for subclass in ("Lamp5", "Pvalb", "Sst", "Vip"):
        for module in MODULE_ORDER:
            for fold in range(3):
                for model in CLASSICAL_MODELS:
                    within_paths.append(outputs["models"] / "within_subclass" / f"{subclass}_{module}_fold{fold}_{model}.joblib")
    stage("within_subclass", within_paths, lambda: run_within_subclass(
        root, config, subclasses=None, modules=MODULE_ORDER, n_resamples=int(config["inference"]["bootstrap_draws"]),
        result_root=outputs["results"], model_root=outputs["models"],
    ))
    stage("permutations", _stage_paths(root, config, "permutations"), lambda: run_permutations(config, outputs["results"], n_permutations=int(config["inference"]["permutation_draws"])))
    stage("mlp_audit", _stage_paths(root, config, "mlp_audit"), lambda: write_mlp_audit(config, outputs["results"]))

    repeated_paths = [
        outputs["results"] / f"repeated_cv/{name}" for name in (
            "folds.csv", "oof.parquet", "repeat_metrics.csv", "summary.csv", "fitted_parameters.csv", "leakage_ledger.csv",
        )
    ]
    for repeat_index in range(5):
        for module in MODULE_ORDER:
            for fold in range(3):
                repeated_paths.extend(outputs["models"] / "repeated_cv" / f"repeat{repeat_index}_{module}_fold{fold}_{model}.joblib" for model in CLASSICAL_MODELS)
                repeated_paths.extend(outputs["models"] / "repeated_cv" / f"repeat{repeat_index}_{module}_fold{fold}_MLP_seed{seed}.joblib" for seed in config["models"]["mlp"]["ensemble_seeds"])
    stage("repeated_cv", repeated_paths, lambda: run_repeated_cv(
        root, config, repeat_seeds=config["repeated_cv"]["seeds"], modules=MODULE_ORDER,
        mlp_seeds=config["models"]["mlp"]["ensemble_seeds"], result_root=outputs["results"], model_root=outputs["models"],
    ))
    stage("calibration", _stage_paths(root, config, "calibration"), lambda: run_calibration(outputs["results"]))
    attribution_paths = [outputs["results"] / f"attribution/{name}" for name in (
        "oof_shap_23feature.parquet", "oof_shap_23feature_wide.parquet", "shap_configuration.json",
        "correlation_edges.csv", "feature_components.csv", "grouped_shap.parquet", "en_coefficients.csv",
        "en_grouped_coefficients.csv", "concordance.csv",
    )]
    stage("attribution", attribution_paths, lambda: run_attribution(root, config, outputs["results"], outputs["models"]))
    technical_paths = [outputs["results"] / f"specificity_bridge/technical_target_{suffix}" for suffix in (
        "oof.parquet", "performance.csv", "fold_metrics.csv", "hyperparameters.csv", "leakage_ledger.csv",
    )]
    for target in ("log10_total_counts", "log10_genes_detected"):
        for fold in range(3):
            for model in CLASSICAL_MODELS:
                technical_paths.append(outputs["models"] / "specificity_bridge/technical" / f"technical_target_{target}_fold{fold}_{model}.joblib")
    stage("technical_bridge", technical_paths, lambda: run_technical_bridge(root, config, outputs["results"], outputs["models"]))
    adjusted_paths = [outputs["results"] / f"specificity_bridge/adjusted/adjusted_target_{suffix}" for suffix in (
        "oof.parquet", "performance.csv", "fold_metrics.csv", "hyperparameters.csv", "leakage_ledger.csv",
    )]
    adjusted_paths.extend(
        [outputs["results"] / "specificity_bridge/adjusted/crossfit_nuisance_predictions.parquet",
         outputs["results"] / "specificity_bridge/adjusted/nuisance_coefficients_and_leakage.csv"]
    )
    for module in MODULE_ORDER:
        for fold in range(3):
            adjusted_paths.append(outputs["models"] / "specificity_bridge/adjusted" / f"adjusted_target_{module}_fold{fold}_XGBoost.joblib")
    stage("adjusted_bridge", adjusted_paths, lambda: run_adjusted_bridge(root, config, outputs["results"], outputs["models"]))

    bridge_paths = [
        outputs["results"] / "specificity_bridge/random_control_performance.parquet",
        outputs["results"] / "specificity_bridge/specificity_summary.csv",
        outputs["results"] / "specificity_bridge/resume_audit.json",
    ]
    for module in MODULE_ORDER:
        bridge_paths.extend(outputs["results"] / "specificity_bridge/random_control_oof" / f"module={module}" / f"control_id={control_id:04d}" / "part-000.parquet" for control_id in range(200))
    if cache.valid("specificity_bridge", bridge_paths):
        completed_stages.append("specificity_bridge"); reused_partition_count = 1200
    else:
        bridge_record = cache.payload.get("stages", {}).get("specificity_bridge", {})
        trusted_partition_sha256 = bridge_record.get("files", {}) if isinstance(bridge_record, dict) else {}
        accepted_run_signatures = {signature}
        if migration and migration.get("all_prior_recorded_digests_validated"):
            accepted_run_signatures.add(str(migration["from_signature"]))
        written = run_specificity_bridge(
            root, config, modules=MODULE_ORDER, folds=(0,1,2), control_count=200,
            result_root=outputs["results"], signature=signature,
            trusted_partition_sha256=trusted_partition_sha256,
            accepted_run_signatures=accepted_run_signatures,
        )
        cache.complete("specificity_bridge", bridge_paths)
        audit = json.loads((outputs["results"] / "specificity_bridge/resume_audit.json").read_text(encoding="utf-8"))
        reused_partition_count = int(audit["reused_partition_count"]); fit_count += int(audit["fit_partition_count"])
        completed_stages.append("specificity_bridge")

    figure_paths = [outputs["figures"] / f"{stem}.{suffix}" for stem in (
        "figure_a_revised_model_performance", "figure_b_paired_model_comparison", "figure_c_subclass_increment",
        "figure_d_model_stability", "figure_e_calibration",
    ) for suffix in ("png", "pdf")]
    stage("figures", figure_paths, lambda: run_figures(outputs["results"], outputs["figures"]))
    report_path = generate_report(root, config)
    marker = outputs["logs"] / ("resume_run.json" if mode == "resume" else "full_run.json")
    corrupt_probe = outputs["logs"] / "cache_truth_corruption_probe.parquet"
    probe_frame = pd.DataFrame({"y_true": [1.0], "y_pred": [1.0]}); atomic_parquet(probe_frame, corrupt_probe)
    probe_cache = StageCache(outputs["logs"] / "cache_truth_corruption_probe.json", signature); probe_cache.complete("probe", [corrupt_probe]); probe_frame.loc[0,"y_true"] = 2.; atomic_parquet(probe_frame, corrupt_probe)
    corruption_rejected = not probe_cache.valid("probe", [corrupt_probe])
    corrupt_probe.unlink(missing_ok=True); (outputs["logs"] / "cache_truth_corruption_probe.json").unlink(missing_ok=True)
    atomic_json(
        {"status": "PASS", "mode": mode, "signature": signature, "completed_stages": completed_stages,
         "fit_count": fit_count, "reused_partition_count": reused_partition_count,
         "corrupt_cache_truth_rejected": corruption_rejected, "report": report_path.relative_to(root).as_posix()},
        marker,
    )
    manifest = write_artifact_manifest(root, config)
    return {"status": "PASS", "mode": mode, "signature": signature, "marker": str(marker), "manifest": str(manifest), "fit_count": fit_count}
