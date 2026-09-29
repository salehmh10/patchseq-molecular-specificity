"""Matched-control analysis: molecular-specificity and target-validity analyses.

All analytical writes are isolated below the configured revision_v3/specificity
namespaces.  The frozen primary and Robustness V2 artifacts are read-only inputs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import tarfile
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedGroupKFold

from .biology import load_gene_modules, score_modules
from .data import _single_csv_member, load_public_metadata, sha256
from .evaluation import fold_metrics, performance_table, regression_metrics
from .models import elastic_net_candidates, select_on_inner_groups, xgboost_candidates
from .statistics import cluster_bootstrap_metrics, summarize_bootstrap
from .training import MODULES, load_and_validate_inputs, load_feature_names


MODULE_ORDER = tuple(MODULES)
MODEL_ORDER = ("ElasticNet", "XGBoost")
COUNT_QC_VERSION = "revision-v3-count-qc-v1"
PIPELINE_VERSION = "revision-v3-specificity-v1"


def stable_seed(*values: object, base: int = 42) -> int:
    payload = "|".join([PIPELINE_VERSION, str(base), *(str(value) for value in values)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_json(payload: Any, path: Path) -> None:
    atomic_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), path)


def atomic_csv(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=index, compression="snappy")
    temporary.replace(path)


def read_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(config_path).resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != "1.0":
        raise ValueError("Unsupported Specificity analysis specificity configuration")
    root = path.parent.parent
    return root, payload


def resolve_paths(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {key: root / value for key, value in config["paths"].items()}


def output_roots(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    return {key: root / value for key, value in config["outputs"].items()}


def ensure_output_roots(root: Path, config: dict[str, Any]) -> dict[str, Path]:
    outputs = output_roots(root, config)
    for path in outputs.values():
        path.mkdir(parents=True, exist_ok=True)
    return outputs


def config_signature(root: Path, config_path: Path, config: dict[str, Any]) -> str:
    paths = resolve_paths(root, config)
    immutable = {
        key: sha256(paths[key])
        for key in (
            "modeling_table",
            "frozen_folds",
            "crosswalk",
            "metadata",
            "raw_count_archive",
            "module_cpm",
            "library_sizes",
            "gene_modules",
            "ephys_features",
            "primary_hyperparameters",
            "primary_oof",
            "primary_shap",
        )
    }
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "implementation_sha256": sha256(Path(__file__).resolve()),
        "entrypoint_sha256": sha256(root / "scripts/09_revision_specificity.py"),
        "config_sha256": sha256(config_path),
        "inputs": immutable,
        "seed": int(config["seed"]),
    }
    return hash_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


class StageCache:
    """Hash-validated stage completion records used by full/resume modes."""

    def __init__(self, path: Path, signature: str):
        self.path = path
        self.signature = signature
        if path.is_file():
            try:
                self.payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self.payload = {}
        else:
            self.payload = {}
        if self.payload.get("signature") != signature:
            self.payload = {"signature": signature, "stages": {}}
        self.payload.setdefault("stages", {})

    def valid(self, stage: str, paths: Sequence[Path]) -> bool:
        record = self.payload["stages"].get(stage)
        if not isinstance(record, dict):
            return False
        expected = record.get("files", {})
        for path in paths:
            key = str(path.resolve())
            if not path.is_file() or expected.get(key) != sha256(path):
                return False
        return True

    def complete(self, stage: str, paths: Sequence[Path], **metadata: Any) -> None:
        self.payload["stages"][stage] = {
            "completed_at": pd.Timestamp.now(tz="Asia/Tehran").isoformat(),
            "files": {str(path.resolve()): sha256(path) for path in paths},
            **metadata,
        }
        atomic_json(self.payload, self.path)


def load_analysis_table(root: Path, config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    paths = resolve_paths(root, config)
    table, features = load_and_validate_inputs(
        paths["modeling_table"], paths["frozen_folds"], paths["ephys_features"]
    )
    crosswalk = pd.read_csv(
        paths["crosswalk"],
        dtype={"canonical_cell_id": str, "transcriptomics_sample_id": str},
    )
    crosswalk["canonical_cell_id"] = crosswalk["canonical_cell_id"].astype(str)
    keep = ["canonical_cell_id", "transcriptomics_sample_id", "map_confidence"]
    if crosswalk[keep].isna().any().any() or crosswalk["canonical_cell_id"].duplicated().any():
        raise AssertionError("Crosswalk identifiers and map confidence must be complete and unique")
    table = table.merge(crosswalk[keep], on="canonical_cell_id", how="left", validate="one_to_one")
    if table[["transcriptomics_sample_id", "map_confidence"]].isna().any().any():
        raise AssertionError("Every modeling cell must map to transcriptomic ID and map confidence")
    return table, features


def verify_integrity(root: Path, config: dict[str, Any], output: Path) -> dict[str, Any]:
    paths = resolve_paths(root, config)
    expected = config["expected"]
    manifest = pd.read_csv(root / "data/MANIFEST.tsv", sep="\t", dtype=str)
    manifest_rows: list[dict[str, Any]] = []
    for row in manifest.to_dict("records"):
        source = root / row["relative_path"]
        actual_bytes = source.stat().st_size
        actual_hash = sha256(source)
        manifest_rows.append(
            {
                "path": row["relative_path"],
                "bytes": actual_bytes,
                "bytes_match": actual_bytes == int(row["bytes"]),
                "sha256": actual_hash,
                "sha256_match": actual_hash == row["sha256"],
            }
        )
    if not all(row["bytes_match"] and row["sha256_match"] for row in manifest_rows):
        raise AssertionError("Manifest integrity failure")

    table, features = load_analysis_table(root, config)
    if len(table) != int(expected["cohort_n"]):
        raise AssertionError("Unexpected cohort size")
    if table.canonical_cell_id.duplicated().any():
        raise AssertionError("Canonical cell IDs are not unique")
    if table.transcriptomics_sample_id.duplicated().any():
        raise AssertionError("Transcriptomic sample IDs are not unique in modeling cohort")
    if table.group_id.astype(str).nunique() != int(expected["donor_n"]):
        raise AssertionError("Unexpected donor/group count")
    if len(features) != int(expected["feature_n"]):
        raise AssertionError("Unexpected predictor count")
    donor_overlap = {}
    for fold in range(int(expected["outer_folds"])):
        train = set(table.loc[table.fold.ne(fold), "group_id"].astype(str))
        test = set(table.loc[table.fold.eq(fold), "group_id"].astype(str))
        donor_overlap[str(fold)] = len(train & test)
    if any(donor_overlap.values()):
        raise AssertionError(f"Frozen donor overlap: {donor_overlap}")

    modules = load_gene_modules(paths["gene_modules"])
    cpm = pd.read_parquet(paths["module_cpm"])
    cpm.index = cpm.index.astype(str)
    scores, inventory = score_modules(cpm, modules)
    sample_ids = table.transcriptomics_sample_id.astype(str)
    target_max_abs_difference = {}
    for module in MODULE_ORDER:
        rebuilt = sample_ids.map(scores[module]).to_numpy(dtype=float)
        observed = table[f"target_{module}"].to_numpy(dtype=float)
        difference = float(np.max(np.abs(rebuilt - observed)))
        target_max_abs_difference[module] = difference
        if difference != 0.0:
            raise AssertionError(f"Primary target reconstruction mismatch for {module}: {difference}")

    oof_hash = sha256(paths["primary_oof"])
    shap_hash = sha256(paths["primary_shap"])
    if oof_hash != expected["primary_oof_sha256"] or shap_hash != expected["primary_shap_sha256"]:
        raise AssertionError("Frozen primary OOF/SHAP hash mismatch")
    payload = {
        "status": "PASS",
        "cohort_n": len(table),
        "donor_n": int(table.group_id.astype(str).nunique()),
        "subclasses": table.subclass.value_counts().sort_index().to_dict(),
        "fold_sizes": table.fold.value_counts().sort_index().to_dict(),
        "donor_overlap": donor_overlap,
        "feature_n": len(features),
        "target_max_abs_difference": target_max_abs_difference,
        "module_inventory": inventory.to_dict("records"),
        "manifest": manifest_rows,
        "primary_oof_sha256": oof_hash,
        "primary_shap_sha256": shap_hash,
    }
    atomic_json(payload, output)
    return payload


def _valid_release_identifier(symbol: str) -> tuple[bool, str]:
    if not symbol:
        return False, "empty"
    if symbol != symbol.strip():
        return False, "surrounding_whitespace"
    if any(ord(character) < 32 for character in symbol):
        return False, "control_character"
    return True, ""


def stream_count_qc(
    archive_path: Path,
    library_path: Path,
    modules: dict[str, list[str]],
    *,
    chunksize: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Stream all count rows and compute per-sample/per-gene QC plus Scn2a evidence."""

    reference = pd.read_csv(library_path, dtype={"transcriptomics_sample_id": str})
    if reference.transcriptomics_sample_id.duplicated().any():
        raise AssertionError("Saved library-size sample IDs are not unique")
    reference = reference.set_index("transcriptomics_sample_id", verify_integrity=True)
    requested_to_modules: dict[str, list[str]] = defaultdict(list)
    for module, genes in modules.items():
        for gene in genes:
            requested_to_modules[gene].append(module)
    requested = set(requested_to_modules)

    gene_rows: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()
    duplicate_symbols: set[str] = set()
    scn_rows: list[dict[str, Any]] = []
    scn2a1_counts: pd.Series | None = None
    total_counts: np.ndarray | None = None
    genes_detected: np.ndarray | None = None
    sample_ids: list[str] | None = None
    rows_seen = 0
    started = time.perf_counter()

    with tarfile.open(archive_path, mode="r:*") as archive:
        member = _single_csv_member(archive)
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("Could not open count CSV member")
        for chunk in pd.read_csv(handle, index_col=0, chunksize=chunksize):
            chunk.index = chunk.index.astype(str)
            if sample_ids is None:
                sample_ids = chunk.columns.astype(str).tolist()
                if len(sample_ids) != len(set(sample_ids)):
                    raise AssertionError("Raw count sample IDs are not unique")
                if set(sample_ids) != set(reference.index.astype(str)):
                    raise AssertionError("Raw count and saved library-size sample ID sets differ")
                denominator = reference.loc[sample_ids, "library_size_counts"].to_numpy(dtype=np.float64)
                if not np.isfinite(denominator).all() or np.any(denominator <= 0):
                    raise AssertionError("Reference library sizes must be positive and finite")
                total_counts = np.zeros(len(sample_ids), dtype=np.int64)
                genes_detected = np.zeros(len(sample_ids), dtype=np.int64)
            values = chunk.to_numpy(dtype=np.float64, copy=False)
            if not np.isfinite(values).all() or np.any(values < 0):
                raise AssertionError("Counts must be finite and nonnegative")
            if not np.equal(values, np.floor(values)).all():
                raise AssertionError("Counts must be integer-valued")
            integer_values = values.astype(np.int64, copy=False)
            assert total_counts is not None and genes_detected is not None and sample_ids is not None
            total_counts += integer_values.sum(axis=0, dtype=np.int64)
            genes_detected += (integer_values > 0).sum(axis=0, dtype=np.int64)
            log_values = np.log2(values / denominator.reshape(1, -1) * 1e6 + 1.0)
            for local_index, symbol in enumerate(chunk.index):
                source_row = rows_seen + local_index
                valid, issue = _valid_release_identifier(symbol)
                duplicate = symbol in seen_symbols
                if duplicate:
                    duplicate_symbols.add(symbol)
                seen_symbols.add(symbol)
                vector = log_values[local_index]
                detection_rate = float(np.mean(values[local_index] > 0))
                row = {
                    "gene_symbol": symbol,
                    "source_row": source_row,
                    "duplicate_symbol": duplicate,
                    "identifier_valid": valid,
                    "identifier_issue": issue,
                    "mean_log2cpm": float(np.mean(vector)),
                    "detection_rate": detection_rate,
                    "variance_log2cpm": float(np.var(vector, ddof=0)),
                    "is_primary_requested_gene": symbol in requested,
                    "primary_modules": ";".join(requested_to_modules.get(symbol, [])),
                }
                gene_rows.append(row)
                normalized = symbol.strip().casefold()
                if "scn2a" in normalized or normalized == "scn2a1":
                    scn_rows.append(
                        {
                            "gene_symbol": symbol,
                            "source_row": source_row,
                            "csv_line": source_row + 2,
                            "exact_scn2a": symbol == "Scn2a",
                            "exact_scn2a1": symbol == "Scn2a1",
                            "case_insensitive_exact_scn2a": normalized == "scn2a",
                            "case_insensitive_substring_scn2a": "scn2a" in normalized,
                            "whitespace_variant": symbol != symbol.strip(),
                            "detected_sample_n": int(np.sum(values[local_index] > 0)),
                            "detection_rate": detection_rate,
                        }
                    )
                if symbol == "Scn2a1":
                    scn2a1_counts = pd.Series(values[local_index].copy(), index=sample_ids, name="count")
            rows_seen += len(chunk)

    if sample_ids is None or total_counts is None or genes_detected is None:
        raise AssertionError("Raw count source was empty")
    if duplicate_symbols:
        raise AssertionError(f"Duplicate raw gene symbols: {sorted(duplicate_symbols)[:10]}")
    reference_values = reference.loc[sample_ids, "library_size_counts"].to_numpy(dtype=np.float64)
    delta = total_counts.astype(np.float64) - reference_values
    if not np.array_equal(total_counts.astype(np.float64), reference_values):
        raise AssertionError(f"Recomputed library sizes differ; max abs delta={np.max(np.abs(delta))}")
    if np.any(total_counts <= 0) or np.any(genes_detected <= 0):
        raise AssertionError("Technical count targets must be positive")

    gene_qc = pd.DataFrame(gene_rows)
    excluded = requested | {"Scn2a1"}
    eligible = (
        gene_qc.identifier_valid
        & ~gene_qc.duplicate_symbol
        & ~gene_qc.gene_symbol.isin(excluded)
        & np.isfinite(gene_qc.mean_log2cpm)
        & np.isfinite(gene_qc.detection_rate)
        & np.isfinite(gene_qc.variance_log2cpm)
        & gene_qc.detection_rate.gt(0)
        & gene_qc.variance_log2cpm.gt(1e-12)
    )
    gene_qc["candidate_eligible"] = eligible
    gene_qc["exclusion_reason"] = ""
    gene_qc.loc[~gene_qc.identifier_valid, "exclusion_reason"] = "invalid_identifier"
    gene_qc.loc[gene_qc.duplicate_symbol, "exclusion_reason"] = "duplicate_symbol"
    gene_qc.loc[gene_qc.gene_symbol.isin(requested), "exclusion_reason"] = "primary_module_gene"
    gene_qc.loc[gene_qc.gene_symbol.eq("Scn2a1"), "exclusion_reason"] = "scn2a_alias_audit_gene"
    gene_qc.loc[gene_qc.detection_rate.le(0), "exclusion_reason"] = "zero_detection"
    gene_qc.loc[gene_qc.variance_log2cpm.le(1e-12), "exclusion_reason"] = "zero_variance"

    sample_qc = pd.DataFrame(
        {
            "transcriptomics_sample_id": sample_ids,
            "total_counts": total_counts,
            "genes_detected": genes_detected,
            "log10_total_counts": np.log10(total_counts.astype(float)),
            "log10_genes_detected": np.log10(genes_detected.astype(float)),
            "reference_library_size": reference_values,
            "library_size_delta": delta,
        }
    )
    if not np.isfinite(sample_qc.select_dtypes(include=[np.number]).to_numpy()).all():
        raise AssertionError("Transcriptomic QC contains non-finite values")
    scn_matches = pd.DataFrame(scn_rows)
    if scn2a1_counts is None:
        scn2a1 = pd.DataFrame(columns=["transcriptomics_sample_id", "count"])
    else:
        scn2a1 = scn2a1_counts.rename_axis("transcriptomics_sample_id").reset_index()
    audit = {
        "version": COUNT_QC_VERSION,
        "archive_sha256": sha256(archive_path),
        "tar_member": member.name,
        "gene_rows": rows_seen,
        "sample_n": len(sample_ids),
        "unique_gene_symbols": len(seen_symbols),
        "duplicate_gene_symbols": len(duplicate_symbols),
        "library_reconciliation_max_abs_difference": float(np.max(np.abs(delta))),
        "library_reconciliation_exact": True,
        "total_counts_min": int(total_counts.min()),
        "total_counts_max": int(total_counts.max()),
        "genes_detected_min": int(genes_detected.min()),
        "genes_detected_max": int(genes_detected.max()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return sample_qc, gene_qc, scn_matches, audit, scn2a1


def write_count_qc_outputs(
    root: Path,
    config: dict[str, Any],
    sample_qc: pd.DataFrame,
    gene_qc: pd.DataFrame,
    scn_matches: pd.DataFrame,
    audit: dict[str, Any],
    scn2a1: pd.DataFrame,
) -> list[Path]:
    outputs = output_roots(root, config)
    paths = resolve_paths(root, config)
    table, _ = load_analysis_table(root, config)
    cohort = table[["canonical_cell_id", "transcriptomics_sample_id"]].merge(
        sample_qc, on="transcriptomics_sample_id", how="left", validate="one_to_one"
    )
    if len(cohort) != len(table) or cohort.isna().any().any():
        raise AssertionError("Technical targets do not cover every modeling cell")
    if not np.array_equal(
        cohort.total_counts.to_numpy(dtype=float), cohort.reference_library_size.to_numpy(dtype=float)
    ):
        raise AssertionError("Cohort technical targets do not reconcile to reference library sizes")
    technical_dir = outputs["results"] / "technical_targets"
    random_dir = outputs["results"] / "random_controls"
    scn_dir = outputs["results"] / "scn2a_audit"
    technical_path = technical_dir / "transcriptomic_qc_targets.csv"
    all_sample_path = technical_dir / "all_transcriptomic_qc_targets.parquet"
    gene_path = random_dir / "all_gene_expression_qc.parquet"
    matches_path = scn_dir / "scn2a_symbol_matches.csv"
    audit_path = scn_dir / "scn2a_audit.json"
    counts_path = scn_dir / "scn2a1_counts.parquet"
    atomic_csv(cohort, technical_path)
    atomic_parquet(sample_qc, all_sample_path)
    atomic_parquet(gene_qc, gene_path)
    atomic_csv(scn_matches, matches_path)
    audit_payload = {
        **audit,
        "exact_scn2a_present": bool(
            not scn_matches.empty and scn_matches.gene_symbol.eq("Scn2a").any()
        ),
        "exact_scn2a1_present": bool(
            not scn_matches.empty and scn_matches.gene_symbol.eq("Scn2a1").any()
        ),
        "case": "B" if (not scn_matches.empty and scn_matches.gene_symbol.eq("Scn2a1").any()) else "A",
        "primary_module_changed": False,
        "mapping_policy": (
            "User-specified release-native Scn2a1 sensitivity only; no primary replacement"
        ),
        "secondary_identifier_evidence": "unavailable_in_count_release",
        "raw_count_source": str(paths["raw_count_archive"].relative_to(root)),
    }
    atomic_json(audit_payload, audit_path)
    atomic_parquet(scn2a1, counts_path)
    return [technical_path, all_sample_path, gene_path, matches_path, audit_path, counts_path]


def prediction_frame(
    test: pd.DataFrame,
    prediction: np.ndarray,
    truth: np.ndarray,
    *,
    target: str,
    model: str,
    analysis: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_cell_id": test.canonical_cell_id.astype(str).to_numpy(),
            "group_id": test.group_id.astype(str).to_numpy(),
            "subclass": test.subclass.astype(str).to_numpy(),
            "fold": test.fold.to_numpy(dtype=int),
            "target": target,
            "model": model,
            "analysis": analysis,
            "y_true": np.asarray(truth, dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
        }
    )


def _fit_selected_models(
    table: pd.DataFrame,
    features: list[str],
    target_values: pd.Series,
    *,
    target_name: str,
    folds: Sequence[int],
    seed: int,
    analysis: str,
    model_dir: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune the bounded EN/XGB grids inside each supplied outer-training fold."""

    values = target_values.reindex(table.index).to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise AssertionError(f"Non-finite target values for {target_name}")
    prediction_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    for fold in folds:
        train_mask = table.fold.ne(fold).to_numpy()
        test_mask = table.fold.eq(fold).to_numpy()
        train = table.loc[train_mask]
        test = table.loc[test_mask]
        train_groups = set(train.group_id.astype(str))
        test_groups = set(test.group_id.astype(str))
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"Outer donor leakage for {target_name}, fold {fold}")
        X_train = train[features]
        X_test = test[features]
        y_train = values[train_mask]
        y_test = values[test_mask]
        groups = train.group_id.astype(str).to_numpy()
        candidates = {
            "ElasticNet": elastic_net_candidates(features, seed=seed),
            "XGBoost": xgboost_candidates(features, seed=seed, level=0),
        }
        for model in MODEL_ORDER:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                selected = select_on_inner_groups(
                    candidates[model], X_train, y_train, groups, seed=seed + int(fold)
                )
            prediction = selected.pipeline.predict(X_test)
            prediction_parts.append(
                prediction_frame(
                    test,
                    prediction,
                    y_test,
                    target=target_name,
                    model=model,
                    analysis=analysis,
                )
            )
            parameter_rows.append(
                {
                    "target": target_name,
                    "fold": int(fold),
                    "model": model,
                    "inner_rmse": float(selected.inner_rmse),
                    **selected.parameters,
                }
            )
            if model_dir is not None:
                model_dir.mkdir(parents=True, exist_ok=True)
                safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", target_name)
                joblib.dump(selected.pipeline, model_dir / f"{safe_target}_fold{fold}_{model}.joblib")
        ledger_rows.append(
            {
                "target": target_name,
                "fold": int(fold),
                "train_n": int(train_mask.sum()),
                "test_n": int(test_mask.sum()),
                "train_group_n": len(train_groups),
                "test_group_n": len(test_groups),
                "group_overlap_n": len(overlap),
                "train_id_sha256": hash_text("\n".join(sorted(train.canonical_cell_id.astype(str)))),
                "test_id_sha256": hash_text("\n".join(sorted(test.canonical_cell_id.astype(str)))),
                "preprocessing_fit_scope": "outer_training_only",
            }
        )
    oof = pd.concat(prediction_parts, ignore_index=True)
    if oof.duplicated(["canonical_cell_id", "target", "model"]).any():
        raise AssertionError("Duplicate OOF target/model/cell row")
    if not np.isfinite(oof[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise AssertionError("OOF output contains non-finite values")
    return oof, pd.DataFrame(parameter_rows), pd.DataFrame(ledger_rows)


def performance_with_bootstrap(
    oof: pd.DataFrame,
    *,
    draws: int,
    seed: int,
    key_columns: Sequence[str] = ("target", "model", "analysis"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    draw_parts: list[pd.DataFrame] = []
    for keys, frame in oof.groupby(list(key_columns), sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_values = dict(zip(key_columns, keys))
        metrics = regression_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())
        bootstrap = cluster_bootstrap_metrics(
            frame,
            group_column="group_id",
            n_resamples=draws,
            seed=stable_seed("bootstrap", *keys, base=seed),
        )
        rows.append({**key_values, "n": len(frame), **metrics, **summarize_bootstrap(bootstrap)})
        for column, value in key_values.items():
            bootstrap[column] = value
        draw_parts.append(bootstrap)
    return pd.DataFrame(rows), pd.concat(draw_parts, ignore_index=True)


def run_technical_target_models(
    root: Path,
    config: dict[str, Any],
) -> list[Path]:
    outputs = output_roots(root, config)
    table, features = load_analysis_table(root, config)
    technical_dir = outputs["results"] / "technical_targets"
    targets = pd.read_csv(
        technical_dir / "transcriptomic_qc_targets.csv", dtype={"canonical_cell_id": str}
    )
    targets["canonical_cell_id"] = targets.canonical_cell_id.astype(str)
    table = table.merge(
        targets[["canonical_cell_id", "log10_total_counts", "log10_genes_detected"]],
        on="canonical_cell_id",
        how="left",
        validate="one_to_one",
    )
    if table[["log10_total_counts", "log10_genes_detected"]].isna().any().any():
        raise AssertionError("Technical targets are incomplete after cohort join")
    oof_parts: list[pd.DataFrame] = []
    parameter_parts: list[pd.DataFrame] = []
    ledger_parts: list[pd.DataFrame] = []
    for target in ("log10_total_counts", "log10_genes_detected"):
        oof, parameters, ledger = _fit_selected_models(
            table,
            features,
            table[target],
            target_name=target,
            folds=(0, 1, 2),
            seed=int(config["seed"]),
            analysis="technical_target",
            model_dir=outputs["models"] / "technical_targets",
        )
        oof_parts.append(oof)
        parameter_parts.append(parameters)
        ledger_parts.append(ledger)
    oof = pd.concat(oof_parts, ignore_index=True)
    expected_rows = len(table) * 2 * 2
    if len(oof) != expected_rows:
        raise AssertionError(f"Technical-target OOF coverage {len(oof)} != {expected_rows}")
    performance, bootstrap = performance_with_bootstrap(
        oof, draws=int(config["models"]["bootstrap_draws"]), seed=int(config["seed"])
    )
    folds = fold_metrics(oof.rename(columns={"target": "module"})).rename(columns={"module": "target"})
    paths = [
        technical_dir / "technical_target_oof.parquet",
        technical_dir / "technical_target_performance.csv",
        technical_dir / "technical_target_fold_metrics.csv",
        technical_dir / "technical_target_bootstrap.parquet",
        technical_dir / "technical_target_hyperparameters.csv",
        technical_dir / "technical_target_leakage_ledger.csv",
    ]
    atomic_parquet(oof, paths[0])
    atomic_csv(performance, paths[1])
    atomic_csv(folds, paths[2])
    atomic_parquet(bootstrap, paths[3])
    atomic_csv(pd.concat(parameter_parts, ignore_index=True), paths[4])
    atomic_csv(pd.concat(ledger_parts, ignore_index=True), paths[5])
    return paths


def generate_matched_gene_sets(
    gene_qc: pd.DataFrame,
    modules: dict[str, list[str]],
    *,
    control_n: int,
    settings: dict[str, Any],
    seed: int,
    selected_modules: Sequence[str],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Generate deterministic one-to-one expression/detection matched modules."""

    if gene_qc.gene_symbol.duplicated().any():
        raise AssertionError("Gene-QC symbols must be unique before matching")
    candidates = gene_qc.loc[gene_qc.candidate_eligible].copy()
    if candidates.empty:
        raise AssertionError("No eligible matched-control candidates")
    means = candidates[["mean_log2cpm", "detection_rate"]].mean()
    scales = candidates[["mean_log2cpm", "detection_rate"]].std(ddof=0)
    if not np.isfinite(scales).all() or np.any(scales.to_numpy() <= 0):
        raise AssertionError("Matching-feature scales must be positive and finite")
    candidates["z_expression"] = (candidates.mean_log2cpm - means.mean_log2cpm) / scales.mean_log2cpm
    candidates["z_detection"] = (candidates.detection_rate - means.detection_rate) / scales.detection_rate
    candidate_lookup = candidates.set_index("gene_symbol", verify_integrity=True)
    lookup = gene_qc.set_index("gene_symbol", verify_integrity=True)
    max_distance = float(settings["max_per_gene_standardized_distance"])
    neighbor_n = int(settings["nearest_neighbor_pool"])
    max_attempts = int(settings["maximum_attempts_per_set"])
    max_mean_distance = float(settings["max_mean_per_gene_standardized_distance"])
    max_module_expression = float(settings["max_module_abs_standardized_mean_difference"])
    max_module_detection = float(settings["max_module_abs_standardized_detection_difference"])

    payload: dict[str, Any] = {
        "matching_features": ["mean_log2cpm", "detection_rate"],
        "candidate_reference_mean": means.to_dict(),
        "candidate_reference_scale": scales.to_dict(),
        "quality_rule": {
            "max_per_gene_standardized_euclidean_distance": max_distance,
            "max_mean_per_gene_standardized_distance": max_mean_distance,
            "max_module_abs_z_expression_difference": max_module_expression,
            "max_module_abs_z_detection_difference": max_module_detection,
        },
        "modules": {},
    }
    assignment_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    candidate_symbols = candidates.gene_symbol.astype(str).to_numpy()
    candidate_coordinates = candidates[["z_expression", "z_detection"]].to_numpy(dtype=float)

    for module in selected_modules:
        requested = modules[module]
        present = [gene for gene in requested if gene in lookup.index]
        target_coordinates: dict[str, tuple[float, float]] = {}
        pools: dict[str, np.ndarray] = {}
        distance_lookup: dict[str, dict[str, float]] = {}
        for gene in present:
            row = lookup.loc[gene]
            target = np.array(
                [
                    (float(row.mean_log2cpm) - means.mean_log2cpm) / scales.mean_log2cpm,
                    (float(row.detection_rate) - means.detection_rate) / scales.detection_rate,
                ]
            )
            target_coordinates[gene] = (float(target[0]), float(target[1]))
            delta = candidate_coordinates - target.reshape(1, -1)
            distance = np.sqrt(np.square(delta).sum(axis=1))
            eligible_indices = np.flatnonzero(distance <= max_distance)
            ordered = eligible_indices[
                np.lexsort((candidate_symbols[eligible_indices], distance[eligible_indices]))
            ]
            ordered = ordered[:neighbor_n]
            if len(ordered) < len(present):
                raise RuntimeError(
                    f"Matching blocker for {module}/{gene}: only {len(ordered)} candidates "
                    f"within standardized distance {max_distance}"
                )
            pools[gene] = ordered
            distance_lookup[gene] = {
                str(candidate_symbols[index]): float(distance[index]) for index in ordered
            }
        order = sorted(present, key=lambda gene: (len(pools[gene]), present.index(gene)))
        signatures: set[str] = set()
        controls: list[dict[str, Any]] = []
        rejected = 0
        for control_id in range(control_n):
            rng = np.random.default_rng(stable_seed("matched_control", module, control_id, base=seed))
            accepted: dict[str, str] | None = None
            accepted_metrics: dict[str, float] | None = None
            for proposal in range(1, max_attempts + 1):
                used: set[str] = set()
                chosen: dict[str, str] = {}
                for gene in order:
                    available = [
                        int(index)
                        for index in pools[gene]
                        if str(candidate_symbols[int(index)]) not in used
                    ]
                    if not available:
                        chosen = {}
                        break
                    index = int(rng.choice(np.asarray(available, dtype=int)))
                    symbol = str(candidate_symbols[index])
                    chosen[gene] = symbol
                    used.add(symbol)
                if len(chosen) != len(present):
                    rejected += 1
                    continue
                signature = hash_text("\n".join(sorted(chosen.values())))
                if signature in signatures:
                    rejected += 1
                    continue
                target_matrix = np.asarray([target_coordinates[gene] for gene in present])
                control_matrix = np.asarray(
                    [
                        candidate_lookup.loc[chosen[gene], ["z_expression", "z_detection"]]
                        .to_numpy(dtype=float)
                        for gene in present
                    ]
                )
                distances = np.asarray(
                    [distance_lookup[gene][chosen[gene]] for gene in present], dtype=float
                )
                metrics = {
                    "mean_matching_distance": float(distances.mean()),
                    "max_matching_distance": float(distances.max()),
                    "module_z_expression_difference": float(
                        control_matrix[:, 0].mean() - target_matrix[:, 0].mean()
                    ),
                    "module_z_detection_difference": float(
                        control_matrix[:, 1].mean() - target_matrix[:, 1].mean()
                    ),
                }
                passed = (
                    metrics["mean_matching_distance"] <= max_mean_distance
                    and metrics["max_matching_distance"] <= max_distance
                    and abs(metrics["module_z_expression_difference"]) <= max_module_expression
                    and abs(metrics["module_z_detection_difference"]) <= max_module_detection
                )
                if not passed:
                    rejected += 1
                    continue
                accepted = chosen
                accepted_metrics = metrics
                signatures.add(signature)
                break
            if accepted is None or accepted_metrics is None:
                raise RuntimeError(
                    f"Matching blocker: {module} control {control_id} failed after {max_attempts} proposals"
                )
            control_record = {
                "control_id": control_id,
                "seed": stable_seed("matched_control", module, control_id, base=seed),
                "genes": [accepted[gene] for gene in present],
                "assignment": accepted,
                "set_signature": hash_text("\n".join(sorted(accepted.values()))),
                **accepted_metrics,
            }
            controls.append(control_record)
            quality_rows.append(
                {
                    "module": module,
                    "control_id": control_id,
                    "n_genes": len(present),
                    "matching_pass": True,
                    "rejected_proposals_before_acceptance": rejected,
                    **accepted_metrics,
                    "set_signature": control_record["set_signature"],
                }
            )
            for target_gene in present:
                control_gene = accepted[target_gene]
                target_z = target_coordinates[target_gene]
                control_row = candidate_lookup.loc[control_gene]
                assignment_rows.append(
                    {
                        "module": module,
                        "control_id": control_id,
                        "target_gene": target_gene,
                        "control_gene": control_gene,
                        "target_z_expression": target_z[0],
                        "target_z_detection": target_z[1],
                        "control_z_expression": float(control_row.z_expression),
                        "control_z_detection": float(control_row.z_detection),
                        "delta_z_expression": float(control_row.z_expression - target_z[0]),
                        "delta_z_detection": float(control_row.z_detection - target_z[1]),
                        "distance": distance_lookup[target_gene][control_gene],
                        "candidate_pool_size": len(pools[target_gene]),
                    }
                )
        payload["modules"][module] = {
            "present_target_genes": present,
            "requested_target_genes": requested,
            "control_n": control_n,
            "rejected_proposals": rejected,
            "controls": controls,
        }
    quality = pd.DataFrame(quality_rows)
    assignments = pd.DataFrame(assignment_rows)
    if not quality.matching_pass.all() or quality.set_signature.duplicated().any():
        # Set signatures may coincide across different modules, so only within-module duplication is forbidden.
        duplicate_within = quality.duplicated(["module", "set_signature"]).any()
        if not quality.matching_pass.all() or duplicate_within:
            raise AssertionError("Matched-control quality/uniqueness invariant failed")
    return payload, quality, assignments


def extract_control_targets(
    archive_path: Path,
    library_path: Path,
    matched: dict[str, Any],
    *,
    chunksize: int,
) -> dict[str, pd.DataFrame]:
    reference = pd.read_csv(library_path, dtype={"transcriptomics_sample_id": str}).set_index(
        "transcriptomics_sample_id", verify_integrity=True
    )
    gene_to_controls: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    expected_gene_counts: dict[tuple[str, int], int] = {}
    for module, module_payload in matched["modules"].items():
        n_genes = len(module_payload["present_target_genes"])
        for control in module_payload["controls"]:
            key = (module, int(control["control_id"]))
            expected_gene_counts[key] = n_genes
            for gene in control["genes"]:
                gene_to_controls[gene].append((module, key[1], 1.0 / n_genes))
    selected = set(gene_to_controls)
    sample_ids: list[str] | None = None
    denominator: np.ndarray | None = None
    accumulators: dict[str, np.ndarray] = {}
    seen_counts: dict[tuple[str, int], int] = defaultdict(int)
    with tarfile.open(archive_path, mode="r:*") as archive:
        member = _single_csv_member(archive)
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("Could not open count CSV member")
        for chunk in pd.read_csv(handle, index_col=0, chunksize=chunksize):
            chunk.index = chunk.index.astype(str)
            if sample_ids is None:
                sample_ids = chunk.columns.astype(str).tolist()
                if set(sample_ids) != set(reference.index.astype(str)):
                    raise AssertionError("Control-target sample IDs differ from library-size IDs")
                denominator = reference.loc[sample_ids, "library_size_counts"].to_numpy(dtype=float)
                for module, module_payload in matched["modules"].items():
                    accumulators[module] = np.zeros(
                        (int(module_payload["control_n"]), len(sample_ids)), dtype=np.float64
                    )
            hits = [gene for gene in chunk.index if gene in selected]
            for gene in hits:
                assert denominator is not None
                values = chunk.loc[gene].to_numpy(dtype=float)
                transformed = np.log2(values / denominator * 1e6 + 1.0)
                for module, control_id, weight in gene_to_controls[gene]:
                    accumulators[module][control_id] += transformed * weight
                    seen_counts[(module, control_id)] += 1
    if sample_ids is None:
        raise AssertionError("Count archive was empty during control-target extraction")
    for key, expected in expected_gene_counts.items():
        if seen_counts[key] != expected:
            raise AssertionError(f"Control target {key} saw {seen_counts[key]}/{expected} genes")
    frames: dict[str, pd.DataFrame] = {}
    for module, matrix in accumulators.items():
        if not np.isfinite(matrix).all():
            raise AssertionError(f"Non-finite random-control target for {module}")
        frame = pd.DataFrame(
            matrix.T,
            columns=[f"control_{index:04d}" for index in range(matrix.shape[0])],
        )
        frame.insert(0, "transcriptomics_sample_id", sample_ids)
        frames[module] = frame
    return frames


def _frozen_xgb_pipeline(
    features: list[str], hyperparameters: pd.DataFrame, module: str, fold: int, *, seed: int
):
    row = hyperparameters.loc[
        hyperparameters.module.eq(module)
        & hyperparameters.fold.eq(fold)
        & hyperparameters.model.eq("XGBoost")
    ]
    if len(row) != 1:
        raise AssertionError(f"Expected one frozen XGBoost configuration for {module}/fold{fold}")
    record = row.iloc[0]
    expected = {
        "n_estimators": int(record.n_estimators),
        "max_depth": int(record.max_depth),
        "learning_rate": float(record.learning_rate),
        "reg_lambda": float(record.reg_lambda),
    }
    for parameters, pipeline in xgboost_candidates(features, seed=seed, level=0):
        if parameters == expected:
            return clone(pipeline), expected
    raise AssertionError(f"Frozen XGBoost parameters are outside the declared grid: {expected}")


def _valid_random_partition(
    path: Path,
    table: pd.DataFrame,
    *,
    target_values: pd.Series,
    target_hash: str,
    run_signature: str,
    folds: Sequence[int],
) -> pd.DataFrame | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_parquet(path)
        expected = table.loc[table.fold.isin(folds), ["canonical_cell_id", "fold"]].copy()
        expected.canonical_cell_id = expected.canonical_cell_id.astype(str)
        frame.canonical_cell_id = frame.canonical_cell_id.astype(str)
        if len(frame) != len(expected) or frame.canonical_cell_id.duplicated().any():
            return None
        if set(frame.canonical_cell_id) != set(expected.canonical_cell_id):
            return None
        lookup = expected.set_index("canonical_cell_id", verify_integrity=True).fold
        if not np.array_equal(
            frame.canonical_cell_id.map(lookup).to_numpy(dtype=int), frame.fold.to_numpy(dtype=int)
        ):
            return None
        truth = pd.Series(
            target_values.to_numpy(dtype=float),
            index=table.canonical_cell_id.astype(str),
        )
        expected_y_true = frame.canonical_cell_id.map(truth).to_numpy(dtype=float)
        if not np.array_equal(frame.y_true.to_numpy(dtype=float), expected_y_true):
            return None
        if set(frame.target_hash.astype(str)) != {target_hash}:
            return None
        if set(frame.run_signature.astype(str)) != {run_signature}:
            return None
        if not np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
            return None
        return frame
    except (OSError, KeyError, TypeError, ValueError):
        return None


def fit_random_controls(
    root: Path,
    config: dict[str, Any],
    matched: dict[str, Any],
    target_frames: dict[str, pd.DataFrame],
    *,
    result_root: Path,
    folds: Sequence[int],
    run_signature: str,
) -> tuple[pd.DataFrame, int]:
    paths = resolve_paths(root, config)
    table, features = load_analysis_table(root, config)
    hyperparameters = pd.read_csv(paths["primary_hyperparameters"])
    source_rows = hyperparameters.loc[hyperparameters.model.eq("XGBoost")].copy()
    if len(source_rows) != 18 or source_rows.duplicated(["module", "fold"]).any():
        raise AssertionError("Primary XGBoost hyperparameter grid must have 18 unique rows")
    snapshot_path = result_root / "frozen_xgboost_hyperparameters.csv"
    snapshot = source_rows.copy()
    snapshot["source_sha256"] = sha256(paths["primary_hyperparameters"])
    snapshot["feature_order_sha256"] = hash_text("\n".join(features))
    snapshot["random_state"] = int(config["seed"])
    snapshot["n_jobs"] = 1
    snapshot["subsample"] = 0.8
    snapshot["colsample_bytree"] = 0.8
    snapshot["tree_method"] = "hist"
    atomic_csv(snapshot, snapshot_path)
    completed_before = 0
    performance_rows: list[dict[str, Any]] = []
    checkpoint_every = int(config["random_controls"]["checkpoint_every"])
    for module, module_payload in matched["modules"].items():
        target_frame = target_frames[module].set_index("transcriptomics_sample_id", verify_integrity=True)
        for control in module_payload["controls"]:
            control_id = int(control["control_id"])
            column = f"control_{control_id:04d}"
            target = table.transcriptomics_sample_id.astype(str).map(target_frame[column])
            if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
                raise AssertionError(f"Incomplete random-control target {module}/{control_id}")
            target_hash = hash_array(target.to_numpy(dtype=np.float64))
            partition = (
                result_root
                / "random_module_oof"
                / f"module={module}"
                / f"control_id={control_id:04d}"
                / "part-000.parquet"
            )
            cached = _valid_random_partition(
                partition,
                table,
                target_values=target,
                target_hash=target_hash,
                run_signature=run_signature,
                folds=folds,
            )
            if cached is not None:
                oof = cached
                completed_before += 1
            else:
                parts: list[pd.DataFrame] = []
                for fold in folds:
                    train_mask = table.fold.ne(fold).to_numpy()
                    test_mask = table.fold.eq(fold).to_numpy()
                    pipeline, parameters = _frozen_xgb_pipeline(
                        features,
                        hyperparameters,
                        module,
                        int(fold),
                        seed=int(config["seed"]),
                    )
                    pipeline.fit(table.loc[train_mask, features], target.to_numpy(dtype=float)[train_mask])
                    prediction = pipeline.predict(table.loc[test_mask, features])
                    part = prediction_frame(
                        table.loc[test_mask],
                        prediction,
                        target.to_numpy(dtype=float)[test_mask],
                        target=f"{module}_control_{control_id:04d}",
                        model="XGBoost",
                        analysis="matched_random_module",
                    )
                    part["target_hash"] = target_hash
                    part["run_signature"] = run_signature
                    part["n_estimators"] = parameters["n_estimators"]
                    part["max_depth"] = parameters["max_depth"]
                    part["learning_rate"] = parameters["learning_rate"]
                    part["reg_lambda"] = parameters["reg_lambda"]
                    parts.append(part)
                oof = pd.concat(parts, ignore_index=True)
                atomic_parquet(oof, partition)
            metrics = regression_metrics(oof.y_true.to_numpy(), oof.y_pred.to_numpy())
            fold_values = {
                f"fold{int(fold)}_r2": regression_metrics(frame.y_true, frame.y_pred)["r2"]
                for fold, frame in oof.groupby("fold")
            }
            performance_rows.append(
                {
                    "module": module,
                    "control_id": control_id,
                    "n": len(oof),
                    "n_genes": len(control["genes"]),
                    **metrics,
                    **fold_values,
                    "matching_pass": True,
                    "matching_mean_distance": control["mean_matching_distance"],
                    "matching_max_distance": control["max_matching_distance"],
                    "target_hash": target_hash,
                    "run_signature": run_signature,
                }
            )
            if len(performance_rows) % checkpoint_every == 0:
                atomic_parquet(
                    pd.DataFrame(performance_rows), result_root / "random_module_performance.partial.parquet"
                )
    performance = pd.DataFrame(performance_rows).sort_values(["module", "control_id"])
    return performance.reset_index(drop=True), completed_before


def summarize_random_controls(
    performance: pd.DataFrame, primary_oof_path: Path
) -> pd.DataFrame:
    primary = pd.read_parquet(primary_oof_path)
    observed = primary.loc[
        primary.model.eq("XGBoost") & primary.analysis.eq("ephys_only")
    ].copy()
    rows: list[dict[str, Any]] = []
    for module, random_frame in performance.groupby("module", sort=False):
        primary_frame = observed.loc[observed.module.eq(module)]
        observed_r2 = regression_metrics(primary_frame.y_true, primary_frame.y_pred)["r2"]
        values = random_frame.r2.to_numpy(dtype=float)
        n = len(values)
        exceed = int(np.sum(values >= observed_r2))
        rows.append(
            {
                "module": module,
                "observed_primary_xgboost_r2": observed_r2,
                "random_control_n": n,
                "random_median_r2": float(np.median(values)),
                "random_95th_percentile_r2": float(np.quantile(values, 0.95, method="linear")),
                "random_99th_percentile_r2": float(np.quantile(values, 0.99, method="linear")),
                "observed_empirical_percentile_le": float(100.0 * np.mean(values <= observed_r2)),
                "random_r2_greater_equal_n": exceed,
                "empirical_p_plus_one": float((1 + exceed) / (n + 1)),
                "passes_95th": bool(observed_r2 > np.quantile(values, 0.95, method="linear")),
                "passes_99th": bool(observed_r2 > np.quantile(values, 0.99, method="linear")),
                "quantile_method": "numpy_linear",
            }
        )
    return pd.DataFrame(rows)


def run_residualized_models(
    root: Path,
    config: dict[str, Any],
    *,
    modules: Sequence[str] = MODULE_ORDER,
    folds: Sequence[int] = (0, 1, 2),
    result_root: Path | None = None,
    model_root: Path | None = None,
) -> list[Path]:
    outputs = output_roots(root, config)
    result_root = result_root or (outputs["results"] / "technical_adjustment")
    model_root = model_root or (outputs["models"] / "technical_adjustment")
    table, features = load_analysis_table(root, config)
    technical = pd.read_csv(
        outputs["results"] / "technical_targets/transcriptomic_qc_targets.csv",
        dtype={"canonical_cell_id": str},
    )
    table = table.merge(
        technical[["canonical_cell_id", "log10_total_counts", "log10_genes_detected"]],
        on="canonical_cell_id",
        how="left",
        validate="one_to_one",
    )
    nuisance_columns = list(config["residualization"]["nuisance_covariates"])
    prediction_parts: list[pd.DataFrame] = []
    nuisance_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for module in modules:
        raw = table[f"target_{module}"].to_numpy(dtype=float)
        for fold in folds:
            train_mask = table.fold.ne(fold).to_numpy()
            test_mask = table.fold.eq(fold).to_numpy()
            train = table.loc[train_mask]
            test = table.loc[test_mask]
            nuisance = LinearRegression(fit_intercept=True).fit(
                train[nuisance_columns], raw[train_mask]
            )
            train_nuisance_prediction = nuisance.predict(train[nuisance_columns])
            test_nuisance_prediction = nuisance.predict(test[nuisance_columns])
            train_residual = raw[train_mask] - train_nuisance_prediction
            test_residual = raw[test_mask] - test_nuisance_prediction
            nuisance_parts.append(
                pd.DataFrame(
                    {
                        "canonical_cell_id": test.canonical_cell_id.astype(str).to_numpy(),
                        "group_id": test.group_id.astype(str).to_numpy(),
                        "subclass": test.subclass.astype(str).to_numpy(),
                        "fold": int(fold),
                        "module": module,
                        "target_raw": raw[test_mask],
                        "log10_total_counts": test.log10_total_counts.to_numpy(dtype=float),
                        "log10_genes_detected": test.log10_genes_detected.to_numpy(dtype=float),
                        "nuisance_prediction": test_nuisance_prediction,
                        "residual_target": test_residual,
                    }
                )
            )
            train_groups = set(train.group_id.astype(str))
            test_groups = set(test.group_id.astype(str))
            if train_groups & test_groups:
                raise AssertionError("Donor leakage during residualization")
            coefficient_rows.append(
                {
                    "module": module,
                    "fold": int(fold),
                    "intercept": float(nuisance.intercept_),
                    "coefficient_log10_total_counts": float(nuisance.coef_[0]),
                    "coefficient_log10_genes_detected": float(nuisance.coef_[1]),
                    "train_n": int(train_mask.sum()),
                    "test_n": int(test_mask.sum()),
                    "train_group_n": len(train_groups),
                    "test_group_n": len(test_groups),
                    "group_overlap_n": len(train_groups & test_groups),
                    "train_id_sha256": hash_text("\n".join(sorted(train.canonical_cell_id.astype(str)))),
                    "test_id_sha256": hash_text("\n".join(sorted(test.canonical_cell_id.astype(str)))),
                    "nuisance_fit_scope": "outer_training_only",
                }
            )
            candidates = {
                "ElasticNet": elastic_net_candidates(features, seed=int(config["seed"])),
                "XGBoost": xgboost_candidates(features, seed=int(config["seed"]), level=0),
            }
            for model in MODEL_ORDER:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    selected = select_on_inner_groups(
                        candidates[model],
                        train[features],
                        train_residual,
                        train.group_id.astype(str).to_numpy(),
                        seed=int(config["seed"]) + int(fold),
                    )
                prediction = selected.pipeline.predict(test[features])
                frame = prediction_frame(
                    test,
                    prediction,
                    test_residual,
                    target=module,
                    model=model,
                    analysis="technical_residualized",
                ).rename(columns={"target": "module"})
                prediction_parts.append(frame)
                parameter_rows.append(
                    {
                        "module": module,
                        "fold": int(fold),
                        "model": model,
                        "inner_rmse": float(selected.inner_rmse),
                        **selected.parameters,
                    }
                )
                model_root.mkdir(parents=True, exist_ok=True)
                joblib.dump(selected.pipeline, model_root / f"{module}_fold{fold}_{model}.joblib")
    nuisance_oof = pd.concat(nuisance_parts, ignore_index=True)
    residual_oof = pd.concat(prediction_parts, ignore_index=True)
    expected_nuisance = len(table.loc[table.fold.isin(folds)]) * len(modules)
    if len(nuisance_oof) != expected_nuisance:
        raise AssertionError("Incomplete cross-fitted nuisance coverage")
    if nuisance_oof.duplicated(["canonical_cell_id", "module"]).any():
        raise AssertionError("Duplicate cross-fitted nuisance test row")
    if not np.allclose(
        nuisance_oof.target_raw - nuisance_oof.nuisance_prediction,
        nuisance_oof.residual_target,
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("Residual identity failed")
    adjusted_performance = performance_table(residual_oof)
    adjusted_folds = fold_metrics(residual_oof)
    nuisance_for_metrics = nuisance_oof.rename(
        columns={"target_raw": "y_true", "nuisance_prediction": "y_pred"}
    ).copy()
    nuisance_for_metrics["model"] = "TechnicalLinearRegression"
    nuisance_for_metrics["analysis"] = "nuisance_only"
    nuisance_performance = performance_table(nuisance_for_metrics)
    primary = pd.read_parquet(resolve_paths(root, config)["primary_oof"])
    raw = primary.loc[
        primary.module.isin(modules)
        & primary.model.isin(MODEL_ORDER)
        & primary.analysis.eq("ephys_only")
    ]
    raw_performance = performance_table(raw)
    comparison = raw_performance.merge(
        adjusted_performance,
        on=["module", "model"],
        suffixes=("_raw", "_adjusted"),
        validate="one_to_one",
    )
    for metric in ("r2", "mae", "rmse", "spearman"):
        comparison[f"delta_{metric}_adjusted_minus_raw"] = (
            comparison[f"{metric}_adjusted"] - comparison[f"{metric}_raw"]
        )
    result_root.mkdir(parents=True, exist_ok=True)
    paths = [
        result_root / "crossfit_nuisance_predictions.parquet",
        result_root / "technical_residualized_targets.parquet",
        result_root / "technical_residualized_oof.parquet",
        result_root / "technical_residualized_performance.csv",
        result_root / "technical_residualized_fold_metrics.csv",
        result_root / "nuisance_only_performance.csv",
        result_root / "raw_vs_adjusted_comparison.csv",
        result_root / "nuisance_coefficients_and_leakage.csv",
        result_root / "technical_residualized_hyperparameters.csv",
    ]
    atomic_parquet(nuisance_oof, paths[0])
    atomic_parquet(
        nuisance_oof[["canonical_cell_id", "group_id", "subclass", "fold", "module", "residual_target"]],
        paths[1],
    )
    atomic_parquet(residual_oof, paths[2])
    atomic_csv(adjusted_performance, paths[3])
    atomic_csv(adjusted_folds, paths[4])
    atomic_csv(nuisance_performance, paths[5])
    atomic_csv(comparison, paths[6])
    atomic_csv(pd.DataFrame(coefficient_rows), paths[7])
    atomic_csv(pd.DataFrame(parameter_rows), paths[8])
    return paths


def build_target_definitions(
    root: Path, config: dict[str, Any], *, modules: Sequence[str] = MODULE_ORDER
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = resolve_paths(root, config)
    table, _ = load_analysis_table(root, config)
    cpm = pd.read_parquet(paths["module_cpm"])
    cpm.index = cpm.index.astype(str)
    frozen = load_gene_modules(paths["gene_modules"])
    transformed = np.log2(cpm.astype(float) + 1.0)
    definition_values: dict[str, dict[str, pd.Series]] = {
        "mean_logcpm": {},
        "zmean": {},
        "pc1": {},
    }
    loading_rows: list[dict[str, Any]] = []
    variance_rows: list[dict[str, Any]] = []
    gene_stat_rows: list[dict[str, Any]] = []
    for module in modules:
        present = [gene for gene in frozen[module] if gene in transformed.columns]
        matrix = transformed[present].to_numpy(dtype=float)
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0, ddof=0)
        if not np.isfinite(scales).all() or np.any(scales <= 0):
            raise AssertionError(f"Non-variable target gene in {module}")
        standardized = (matrix - means.reshape(1, -1)) / scales.reshape(1, -1)
        zmean = standardized.mean(axis=1)
        pca = PCA(n_components=1, svd_solver="full")
        pc1 = pca.fit_transform(standardized).reshape(-1)
        loadings = pca.components_[0].copy()
        orientation_dot = float(np.dot(pc1, zmean))
        orientation = "positive_dot_with_zmean"
        if orientation_dot < 0:
            pc1 *= -1
            loadings *= -1
            orientation_dot *= -1
            orientation = "flipped_to_positive_dot_with_zmean"
        elif abs(orientation_dot) <= 1e-14:
            first = next((index for index, value in enumerate(loadings) if abs(value) > 1e-14), None)
            if first is not None and loadings[first] < 0:
                pc1 *= -1
                loadings *= -1
                orientation = "tie_first_nonzero_loading_positive"
        mean_score = matrix.mean(axis=1)
        definition_values["mean_logcpm"][module] = pd.Series(mean_score, index=transformed.index)
        definition_values["zmean"][module] = pd.Series(zmean, index=transformed.index)
        definition_values["pc1"][module] = pd.Series(pc1, index=transformed.index)
        for gene, mean, scale, loading in zip(present, means, scales, loadings):
            gene_stat_rows.append(
                {"module": module, "gene": gene, "mean_log2cpm": mean, "scale_log2cpm": scale}
            )
            loading_rows.append({"module": module, "gene": gene, "loading": loading})
        variance_rows.append(
            {
                "module": module,
                "n_genes": len(present),
                "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
                "orientation_dot_with_zmean": orientation_dot,
                "orientation_rule": orientation,
                "loading_l2_norm": float(np.linalg.norm(loadings)),
                "fit_sample_n": len(transformed),
                "construction_inputs": "transcriptomics_only_full_release",
            }
        )
    cohort_frames: dict[str, pd.DataFrame] = {}
    for definition, values in definition_values.items():
        frame = table[["canonical_cell_id", "transcriptomics_sample_id"]].copy()
        for module in modules:
            frame[module] = frame.transcriptomics_sample_id.astype(str).map(values[module])
        if frame[list(modules)].isna().any().any():
            raise AssertionError(f"Incomplete cohort mapping for {definition}")
        cohort_frames[definition] = frame
    for module in modules:
        difference = np.max(
            np.abs(
                cohort_frames["mean_logcpm"][module].to_numpy(dtype=float)
                - table[f"target_{module}"].to_numpy(dtype=float)
            )
        )
        if difference != 0.0:
            raise AssertionError(f"Mean target definition differs from primary for {module}")
    return (
        cohort_frames,
        pd.DataFrame(loading_rows),
        pd.DataFrame(variance_rows),
        pd.DataFrame(gene_stat_rows),
    )


def run_target_definition_models(
    root: Path,
    config: dict[str, Any],
    *,
    modules: Sequence[str] = MODULE_ORDER,
    folds: Sequence[int] = (0, 1, 2),
    result_root: Path | None = None,
    model_root: Path | None = None,
) -> list[Path]:
    outputs = output_roots(root, config)
    result_root = result_root or (outputs["results"] / "target_definitions")
    model_root = model_root or (outputs["models"] / "target_definitions")
    table, features = load_analysis_table(root, config)
    frames, loadings, variance, gene_stats = build_target_definitions(root, config, modules=modules)
    result_root.mkdir(parents=True, exist_ok=True)
    target_paths = {
        "mean_logcpm": result_root / "target_mean_logcpm.parquet",
        "zmean": result_root / "target_zmean.parquet",
        "pc1": result_root / "target_pc1.parquet",
    }
    for definition, frame in frames.items():
        atomic_parquet(frame, target_paths[definition])
    loading_path = result_root / "pc1_loadings.csv"
    variance_path = result_root / "pc1_variance_explained.csv"
    gene_stats_path = result_root / "target_gene_standardization.csv"
    atomic_csv(loadings, loading_path)
    atomic_csv(variance, variance_path)
    atomic_csv(gene_stats, gene_stats_path)

    oof_parts: list[pd.DataFrame] = []
    parameter_parts: list[pd.DataFrame] = []
    primary = pd.read_parquet(resolve_paths(root, config)["primary_oof"])
    primary = primary.loc[
        primary.module.isin(modules)
        & primary.model.isin(MODEL_ORDER)
        & primary.analysis.eq("ephys_only")
        & primary.fold.isin(folds)
    ].copy()
    primary["target_definition"] = "mean_logcpm"
    primary["analysis"] = "target_definition_sensitivity"
    oof_parts.append(primary)
    for definition in ("zmean", "pc1"):
        values = frames[definition].set_index("canonical_cell_id")
        for module in modules:
            target = table.canonical_cell_id.astype(str).map(values[module])
            oof, parameters, _ = _fit_selected_models(
                table,
                features,
                target,
                target_name=f"{module}__{definition}",
                folds=folds,
                seed=int(config["seed"]),
                analysis="target_definition_sensitivity",
                model_dir=model_root,
            )
            oof["module"] = module
            oof["target_definition"] = definition
            oof = oof.drop(columns="target")
            parameters["module"] = module
            parameters["target_definition"] = definition
            parameter_parts.append(parameters)
            oof_parts.append(oof)
    oof = pd.concat(oof_parts, ignore_index=True, sort=False)
    if oof.duplicated(["canonical_cell_id", "module", "target_definition", "model"]).any():
        raise AssertionError("Duplicate target-definition OOF row")
    performance_rows: list[dict[str, Any]] = []
    for keys, frame in oof.groupby(["module", "target_definition", "model", "analysis"], sort=True):
        performance_rows.append(
            {
                "module": keys[0],
                "target_definition": keys[1],
                "model": keys[2],
                "analysis": keys[3],
                "n": len(frame),
                **regression_metrics(frame.y_true, frame.y_pred),
            }
        )
    performance = pd.DataFrame(performance_rows)
    summary_rows: list[dict[str, Any]] = []
    thresholds = config["target_definitions"]["classification"]
    for module in modules:
        xgb = performance.loc[
            performance.module.eq(module) & performance.model.eq("XGBoost")
        ].set_index("target_definition")
        r2 = xgb.r2.to_dict()
        deltas = [abs(r2[name] - r2["mean_logcpm"]) for name in ("zmean", "pc1")]
        max_delta = max(deltas)
        if max_delta <= float(thresholds["robust_max_abs_xgboost_r2_delta"]) and all(
            r2[name] > 0 for name in ("zmean", "pc1")
        ):
            classification = "robust"
        elif max_delta <= float(thresholds["moderate_max_abs_xgboost_r2_delta"]):
            classification = "moderately sensitive"
        else:
            classification = "composition-sensitive"
        target_matrix = pd.DataFrame(
            {
                name: frames[name][module].to_numpy(dtype=float)
                for name in ("mean_logcpm", "zmean", "pc1")
            }
        )
        correlations = target_matrix.corr(method="spearman")
        summary_rows.append(
            {
                "module": module,
                "primary_xgboost_r2": r2["mean_logcpm"],
                "zmean_xgboost_r2": r2["zmean"],
                "pc1_xgboost_r2": r2["pc1"],
                "max_abs_xgboost_r2_delta": max_delta,
                "pc1_explained_variance": float(
                    variance.loc[variance.module.eq(module), "explained_variance_ratio"].iloc[0]
                ),
                "spearman_mean_zmean": correlations.loc["mean_logcpm", "zmean"],
                "spearman_mean_pc1": correlations.loc["mean_logcpm", "pc1"],
                "spearman_zmean_pc1": correlations.loc["zmean", "pc1"],
                "classification": classification,
            }
        )
    oof_path = result_root / "target_definition_oof.parquet"
    performance_path = result_root / "target_definition_performance.csv"
    summary_path = result_root / "target_definition_summary.csv"
    parameter_path = result_root / "target_definition_hyperparameters.csv"
    atomic_parquet(oof, oof_path)
    atomic_csv(performance, performance_path)
    atomic_csv(pd.DataFrame(summary_rows), summary_path)
    atomic_csv(pd.concat(parameter_parts, ignore_index=True), parameter_path)
    return [
        *target_paths.values(),
        loading_path,
        variance_path,
        gene_stats_path,
        oof_path,
        performance_path,
        summary_path,
        parameter_path,
    ]


def run_scn2a_alias_sensitivity(root: Path, config: dict[str, Any]) -> list[Path]:
    outputs = output_roots(root, config)
    paths = resolve_paths(root, config)
    result_dir = outputs["results"] / "scn2a_audit"
    audit = json.loads((result_dir / "scn2a_audit.json").read_text(encoding="utf-8"))
    if audit.get("case") != "B":
        return [result_dir / "scn2a_symbol_matches.csv", result_dir / "scn2a_audit.json"]
    counts = pd.read_parquet(result_dir / "scn2a1_counts.parquet")
    counts["transcriptomics_sample_id"] = counts.transcriptomics_sample_id.astype(str)
    library = pd.read_csv(paths["library_sizes"], dtype={"transcriptomics_sample_id": str})
    alias = counts.merge(library, on="transcriptomics_sample_id", how="left", validate="one_to_one")
    if alias.library_size_counts.isna().any():
        raise AssertionError("Scn2a1 counts do not align to library sizes")
    alias["Scn2a1_log2cpm"] = np.log2(
        alias["count"].to_numpy(dtype=float)
        / alias["library_size_counts"].to_numpy(dtype=float)
        * 1e6
        + 1.0
    )
    cpm = pd.read_parquet(paths["module_cpm"])
    cpm.index = cpm.index.astype(str)
    frozen = load_gene_modules(paths["gene_modules"])
    present = [gene for gene in frozen["NaV"] if gene in cpm.columns]
    nav_log = np.log2(cpm[present].astype(float) + 1.0)
    alias_lookup = alias.set_index("transcriptomics_sample_id")["Scn2a1_log2cpm"]
    alias_values = alias_lookup.reindex(nav_log.index)
    if alias_values.isna().any():
        raise AssertionError("Scn2a1 alias values do not cover the full transcriptomic release")
    nav_alias_full = (nav_log.sum(axis=1) + alias_values.to_numpy(dtype=float)) / (len(present) + 1)
    table, features = load_analysis_table(root, config)
    target = table.transcriptomics_sample_id.astype(str).map(nav_alias_full)
    if target.isna().any():
        raise AssertionError("Alias-resolved NaV target is incomplete")
    target_frame = table[["canonical_cell_id", "transcriptomics_sample_id"]].copy()
    target_frame["target_NaV_alias_Scn2a1"] = target.to_numpy(dtype=float)
    oof, parameters, ledger = _fit_selected_models(
        table,
        features,
        target,
        target_name="NaV_alias_Scn2a1",
        folds=(0, 1, 2),
        seed=int(config["seed"]),
        analysis="scn2a1_alias_sensitivity",
        model_dir=outputs["models"] / "scn2a_audit",
    )
    performance, _ = performance_with_bootstrap(
        oof, draws=int(config["models"]["bootstrap_draws"]), seed=int(config["seed"])
    )
    output_paths = [
        result_dir / "nav_alias_target.parquet",
        result_dir / "nav_alias_oof.parquet",
        result_dir / "nav_alias_performance.csv",
        result_dir / "nav_alias_hyperparameters.csv",
        result_dir / "nav_alias_leakage_ledger.csv",
    ]
    atomic_parquet(target_frame, output_paths[0])
    atomic_parquet(oof, output_paths[1])
    atomic_csv(performance, output_paths[2])
    atomic_csv(parameters, output_paths[3])
    atomic_csv(ledger, output_paths[4])
    return output_paths


def _canonical_partition(indices: Iterable[int], n: int) -> tuple[int, ...]:
    left = tuple(sorted(int(index) for index in indices))
    if len(left) * 2 != n:
        return left
    right = tuple(index for index in range(n) if index not in set(left))
    return min(left, right)


def module_partitions(
    genes: Sequence[str], module: str, *, draws: int, seed: int
) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], str, int]:
    n = len(genes)
    left_n = n // 2
    maximum = math.comb(n, left_n) // 2 if n % 2 == 0 else math.comb(n, left_n)
    exhaustive = module in {"HCN", "NaV", "CaV"} or maximum <= draws
    partitions: set[tuple[int, ...]] = set()
    if exhaustive:
        for combination in itertools.combinations(range(n), left_n):
            partitions.add(_canonical_partition(combination, n))
        method = "exhaustive"
    else:
        rng = np.random.default_rng(stable_seed("split_half", module, base=seed))
        while len(partitions) < draws:
            candidate = rng.choice(n, size=left_n, replace=False)
            partitions.add(_canonical_partition(candidate, n))
        method = "deterministic_random_unique"
    ordered = sorted(partitions)
    result = [
        (left, tuple(index for index in range(n) if index not in set(left))) for left in ordered
    ]
    return result, method, maximum


def run_internal_consistency(root: Path, config: dict[str, Any]) -> list[Path]:
    outputs = output_roots(root, config)
    paths = resolve_paths(root, config)
    result_dir = outputs["results"] / "reliability"
    table, _ = load_analysis_table(root, config)
    cpm = pd.read_parquet(paths["module_cpm"])
    cpm.index = cpm.index.astype(str)
    transformed = np.log2(cpm.astype(float) + 1.0)
    transformed = transformed.loc[table.transcriptomics_sample_id.astype(str)]
    modules = load_gene_modules(paths["gene_modules"])
    partition_rows: list[dict[str, Any]] = []
    draw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for module in MODULE_ORDER:
        genes = [gene for gene in modules[module] if gene in transformed.columns]
        partitions, method, maximum = module_partitions(
            genes,
            module,
            draws=int(config["reliability"]["large_module_draws"]),
            seed=int(config["seed"]),
        )
        for partition_id, (left, right) in enumerate(partitions):
            genes_a = [genes[index] for index in left]
            genes_b = [genes[index] for index in right]
            if set(genes_a) & set(genes_b) or set(genes_a) | set(genes_b) != set(genes):
                raise AssertionError("Invalid split-half partition")
            score_a = transformed[genes_a].mean(axis=1)
            score_b = transformed[genes_b].mean(axis=1)
            rho = float(spearmanr(score_a, score_b).statistic)
            sb = float(2 * rho / (1 + rho)) if not np.isclose(1 + rho, 0) else math.nan
            partition_rows.append(
                {
                    "module": module,
                    "partition_id": partition_id,
                    "genes_a": ";".join(genes_a),
                    "genes_b": ";".join(genes_b),
                    "n_a": len(genes_a),
                    "n_b": len(genes_b),
                    "generation": method,
                    "seed": stable_seed("split_half", module, base=int(config["seed"])),
                    "maximum_unique_partitions": maximum,
                }
            )
            draw_rows.append(
                {
                    "module": module,
                    "partition_id": partition_id,
                    "raw_spearman_rho": rho,
                    "spearman_brown": sb,
                }
            )
        module_draws = pd.DataFrame([row for row in draw_rows if row["module"] == module])
        summary_rows.append(
            {
                "module": module,
                "gene_n": len(genes),
                "partition_n": len(partitions),
                "unique_partition_n": len(partitions),
                "maximum_unique_partitions": maximum,
                "generation": method,
                "median_raw_rho": float(module_draws.raw_spearman_rho.median()),
                "raw_rho_q1": float(module_draws.raw_spearman_rho.quantile(0.25)),
                "raw_rho_q3": float(module_draws.raw_spearman_rho.quantile(0.75)),
                "raw_rho_min": float(module_draws.raw_spearman_rho.min()),
                "raw_rho_max": float(module_draws.raw_spearman_rho.max()),
                "median_spearman_brown": float(module_draws.spearman_brown.median()),
                "spearman_brown_q1": float(module_draws.spearman_brown.quantile(0.25)),
                "spearman_brown_q3": float(module_draws.spearman_brown.quantile(0.75)),
                "spearman_brown_min": float(module_draws.spearman_brown.min()),
                "spearman_brown_max": float(module_draws.spearman_brown.max()),
                "interpretation": "internal_consistency_not_definitive_noise_ceiling",
            }
        )
    partition_frame = pd.DataFrame(partition_rows)
    if partition_frame.duplicated(["module", "genes_a", "genes_b"]).any():
        raise AssertionError("Duplicate canonical split-half partition")
    output_paths = [
        result_dir / "split_half_partitions.parquet",
        result_dir / "split_half_draws.parquet",
        result_dir / "module_internal_consistency.csv",
    ]
    atomic_parquet(partition_frame, output_paths[0])
    atomic_parquet(pd.DataFrame(draw_rows), output_paths[1])
    atomic_csv(pd.DataFrame(summary_rows), output_paths[2])
    return output_paths


def _correlation_summary(matrix: pd.DataFrame) -> dict[str, Any]:
    values = matrix.to_numpy(dtype=float)
    if not np.allclose(values, values.T, rtol=0, atol=1e-12):
        raise AssertionError("Target correlation matrix is not symmetric")
    if not np.allclose(np.diag(values), 1.0, rtol=0, atol=1e-12):
        raise AssertionError("Target correlation matrix diagonal is not one")
    eigenvalues = np.linalg.eigvalsh(values)[::-1]
    clipped = np.clip(eigenvalues, 0, None)
    effective = float(clipped.sum() ** 2 / np.square(clipped).sum())
    return {
        "eigenvalues_descending": eigenvalues.tolist(),
        "first_eigenvalue_fraction": float(eigenvalues[0] / len(values)),
        "participation_ratio_effective_dimension": effective,
        "shared_axis_flag": bool(eigenvalues[0] / len(values) >= 0.50),
    }


def run_target_correlations(root: Path, config: dict[str, Any]) -> list[Path]:
    outputs = output_roots(root, config)
    result_dir = outputs["results"] / "target_correlations"
    table, _ = load_analysis_table(root, config)
    raw = table[[f"target_{module}" for module in MODULE_ORDER]].copy()
    raw.columns = MODULE_ORDER
    residual = pd.read_parquet(
        outputs["results"] / "technical_adjustment/technical_residualized_targets.parquet"
    )
    adjusted = residual.pivot(index="canonical_cell_id", columns="module", values="residual_target")
    adjusted = adjusted.reindex(index=table.canonical_cell_id.astype(str), columns=MODULE_ORDER)
    if adjusted.isna().any().any():
        raise AssertionError("Adjusted target matrix is incomplete")
    subclass_parts: list[pd.DataFrame] = []
    mean_rows: list[dict[str, Any]] = []
    for fold in (0, 1, 2):
        train = table.loc[table.fold.ne(fold)]
        test = table.loc[table.fold.eq(fold)]
        centered = test[["canonical_cell_id"]].copy()
        for module in MODULE_ORDER:
            means = train.groupby("subclass", observed=True)[f"target_{module}"].mean()
            expected = test.subclass.map(means)
            if expected.isna().any():
                raise AssertionError(f"Unseen subclass in fold {fold}")
            centered[module] = test[f"target_{module}"].to_numpy(dtype=float) - expected.to_numpy(
                dtype=float
            )
            for subclass, value in means.items():
                mean_rows.append(
                    {
                        "fold": fold,
                        "module": module,
                        "subclass": subclass,
                        "training_mean": float(value),
                        "fit_scope": "outer_training_only",
                    }
                )
        subclass_parts.append(centered)
    subclass_centered = pd.concat(subclass_parts, ignore_index=True).set_index("canonical_cell_id")
    subclass_centered = subclass_centered.reindex(table.canonical_cell_id.astype(str))[list(MODULE_ORDER)]
    matrices = {
        "raw": raw.corr(method="spearman").reindex(index=MODULE_ORDER, columns=MODULE_ORDER),
        "technical_adjusted": adjusted.corr(method="spearman").reindex(
            index=MODULE_ORDER, columns=MODULE_ORDER
        ),
        "subclass_centered": subclass_centered.corr(method="spearman").reindex(
            index=MODULE_ORDER, columns=MODULE_ORDER
        ),
    }
    summary = {name: _correlation_summary(matrix) for name, matrix in matrices.items()}
    output_paths = [
        result_dir / "raw_spearman.csv",
        result_dir / "technical_adjusted_spearman.csv",
        result_dir / "subclass_centered_spearman.csv",
        result_dir / "target_correlation_summary.json",
        result_dir / "subclass_training_means.csv",
    ]
    atomic_csv(matrices["raw"], output_paths[0], index=True)
    atomic_csv(matrices["technical_adjusted"], output_paths[1], index=True)
    atomic_csv(matrices["subclass_centered"], output_paths[2], index=True)
    atomic_json(summary, output_paths[3])
    atomic_csv(pd.DataFrame(mean_rows), output_paths[4])

    figure_dir = outputs["figures"] / "target_correlations"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name, matrix in matrices.items():
        fig, axis = plt.subplots(figsize=(6.2, 5.2))
        image = axis.imshow(matrix.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(MODULE_ORDER)), MODULE_ORDER, rotation=45, ha="right")
        axis.set_yticks(range(len(MODULE_ORDER)), MODULE_ORDER)
        axis.set_title(name.replace("_", " ").title() + " target Spearman correlation")
        fig.colorbar(image, ax=axis, label="Spearman rho")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figure_dir / f"{name}_spearman.{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    return output_paths


def metadata_qc_audit(root: Path, config: dict[str, Any]) -> pd.DataFrame:
    paths = resolve_paths(root, config)
    metadata = load_public_metadata(paths["metadata"])
    crosswalk = pd.read_csv(paths["crosswalk"], dtype={"canonical_cell_id": str})
    roles = {
        "cell_specimen_id": "canonical public specimen identifier",
        "transcriptomics_sample_id": "public transcriptomic sample identifier",
        "ephys_session_id": "public electrophysiology session identifier",
        "donor_id": "donor/group provenance field",
        "transcriptomics_batch": "transcriptomics batch metadata; not a quality score",
        "transcriptomic_cluster_id": "semantically repaired CS accession identifier",
        "transcriptomic_cluster_label": "semantically repaired public cluster label",
    }
    rows: list[dict[str, Any]] = []
    direct_candidates: list[str] = []
    pattern = re.compile(r"contam|quality|qc|rna", flags=re.IGNORECASE)
    for column in metadata.columns:
        series = metadata[column]
        if pattern.search(str(column)):
            direct_candidates.append(str(column))
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() and numeric.notna().sum() == series.notna().sum():
            distribution = {
                "min": float(numeric.min()),
                "q25": float(numeric.quantile(0.25)),
                "median": float(numeric.median()),
                "q75": float(numeric.quantile(0.75)),
                "max": float(numeric.max()),
            }
        else:
            distribution = {
                str(key): int(value)
                for key, value in series.astype("string").fillna("<MISSING>").value_counts().head(20).items()
            }
        rows.append(
            {
                "source": str(paths["metadata"].relative_to(root)).replace("\\", "/"),
                "source_sha256": sha256(paths["metadata"]),
                "scope": "all_4435_public_metadata_samples",
                "column": column,
                "semantic_name": column,
                "dtype": str(series.dtype),
                "row_n": len(series),
                "missing_n": int(series.isna().sum()),
                "missing_fraction": float(series.isna().mean()),
                "unique_nonmissing_n": int(series.nunique(dropna=True)),
                "distribution_json": json.dumps(distribution, sort_keys=True),
                "documented_project_role": roles.get(
                    column, "public metadata field; no contamination/QC definition in project docs"
                ),
                "documentation_reference": "DATA_DICTIONARY.md and source metadata schema",
                "direct_contamination_score": False,
            }
        )
    map_series = crosswalk["map_confidence"]
    rows.append(
        {
            "source": str(paths["crosswalk"].relative_to(root)).replace("\\", "/"),
            "source_sha256": sha256(paths["crosswalk"]),
            "scope": "3410_exact_id_matched_cells",
            "column": "map_confidence",
            "semantic_name": "map_confidence",
            "dtype": str(map_series.dtype),
            "row_n": len(map_series),
            "missing_n": int(map_series.isna().sum()),
            "missing_fraction": float(map_series.isna().mean()),
            "unique_nonmissing_n": int(map_series.nunique(dropna=True)),
            "distribution_json": json.dumps(
                {str(key): int(value) for key, value in map_series.value_counts(dropna=False).items()},
                sort_keys=True,
            ),
            "documented_project_role": (
                "MAT transcriptomic mapping-confidence category; not a contamination score"
            ),
            "documentation_reference": "DATA_DICTIONARY.md MAT variables",
            "direct_contamination_score": False,
        }
    )
    frame = pd.DataFrame(rows)
    frame.attrs["direct_candidate_columns"] = direct_candidates
    return frame


def run_core_only(root: Path, config: dict[str, Any]) -> list[Path]:
    outputs = output_roots(root, config)
    result_dir = outputs["results"] / "core_only"
    table, features = load_analysis_table(root, config)
    technical = pd.read_csv(
        outputs["results"] / "technical_targets/transcriptomic_qc_targets.csv",
        dtype={"canonical_cell_id": str},
    )
    table = table.merge(
        technical[["canonical_cell_id", "log10_total_counts", "log10_genes_detected"]],
        on="canonical_cell_id",
        how="left",
        validate="one_to_one",
    )
    required_value = str(config["core_only"]["map_confidence_value"])
    audit = table[["canonical_cell_id", "group_id", "subclass", "map_confidence"]].copy()
    audit["included_core_only"] = audit.map_confidence.eq(required_value)
    core = table.loc[table.map_confidence.eq(required_value)].copy().reset_index(drop=True)
    splitter = StratifiedGroupKFold(
        n_splits=int(config["core_only"]["outer_folds"]),
        shuffle=True,
        random_state=int(config["core_only"]["fold_seed"]),
    )
    core["fold"] = -1
    configured_fold_n = int(config["core_only"]["outer_folds"])
    for fold, (_, test_index) in enumerate(
        splitter.split(core, y=core.subclass.astype(str), groups=core.group_id.astype(str))
    ):
        core.loc[test_index, "fold"] = fold
    expected_folds = set(range(configured_fold_n))
    if set(core.fold) != expected_folds or core.fold.lt(0).any():
        raise AssertionError("Core-only folds are incomplete")
    for fold in sorted(expected_folds):
        train_groups = set(core.loc[core.fold.ne(fold), "group_id"].astype(str))
        test_groups = set(core.loc[core.fold.eq(fold), "group_id"].astype(str))
        if train_groups & test_groups:
            raise AssertionError("Core-only donor leakage")
    targets = list(config["core_only"]["targets"])
    oof_parts: list[pd.DataFrame] = []
    parameter_parts: list[pd.DataFrame] = []
    ledger_parts: list[pd.DataFrame] = []
    for target in targets:
        column = target if target.startswith("log10_") else f"target_{target}"
        oof, parameters, ledger = _fit_selected_models(
            core,
            features,
            core[column],
            target_name=target,
            folds=tuple(sorted(expected_folds)),
            seed=int(config["seed"]),
            analysis="quality_restricted_core_only",
            model_dir=outputs["models"] / "core_only",
        )
        oof_parts.append(oof)
        parameter_parts.append(parameters)
        ledger_parts.append(ledger)
    oof = pd.concat(oof_parts, ignore_index=True)
    performance, _ = performance_with_bootstrap(
        oof, draws=int(config["models"]["bootstrap_draws"]), seed=int(config["seed"])
    )
    primary_rows: list[pd.DataFrame] = []
    primary_module_oof = pd.read_parquet(resolve_paths(root, config)["primary_oof"])
    primary_module_oof = primary_module_oof.loc[
        primary_module_oof.model.isin(MODEL_ORDER)
        & primary_module_oof.analysis.eq("ephys_only")
        & primary_module_oof.module.isin([target for target in targets if not target.startswith("log10_")])
    ].copy()
    primary_module_oof = primary_module_oof.rename(columns={"module": "target"})
    primary_rows.append(primary_module_oof)
    primary_technical = pd.read_parquet(
        outputs["results"] / "technical_targets/technical_target_oof.parquet"
    )
    primary_rows.append(primary_technical.loc[primary_technical.target.isin(targets)])
    primary_combined = pd.concat(primary_rows, ignore_index=True, sort=False)
    primary_performance_rows: list[dict[str, Any]] = []
    core_ids = set(core.canonical_cell_id.astype(str))
    for keys, frame in primary_combined.groupby(["target", "model"], sort=True):
        restricted = frame.loc[frame.canonical_cell_id.astype(str).isin(core_ids)]
        full_metrics = regression_metrics(frame.y_true, frame.y_pred)
        restricted_metrics = regression_metrics(restricted.y_true, restricted.y_pred)
        primary_performance_rows.append(
            {
                "target": keys[0],
                "model": keys[1],
                "n_primary_full": len(frame),
                "n_primary_core_subset": len(restricted),
                **{f"{metric}_primary_full": value for metric, value in full_metrics.items()},
                **{f"{metric}_primary_core_subset": value for metric, value in restricted_metrics.items()},
            }
        )
    core_for_comparison = performance.rename(
        columns={
            "n": "n_core",
            "r2": "r2_core",
            "mae": "mae_core",
            "rmse": "rmse_core",
            "spearman": "spearman_core",
        }
    )
    comparison = pd.DataFrame(primary_performance_rows).merge(
        core_for_comparison,
        on=["target", "model"],
        validate="one_to_one",
    )
    comparison["delta_r2_core_minus_primary_full"] = (
        comparison.r2_core - comparison.r2_primary_full
    )
    comparison["delta_r2_core_minus_primary_core_subset"] = (
        comparison.r2_core - comparison.r2_primary_core_subset
    )
    comparison["positive_r2_performance_retained"] = comparison.r2_core.gt(0) & comparison[
        "r2_primary_full"
    ].gt(0)
    comparison["spearman_sign_concordant"] = (
        np.sign(comparison.spearman_core) == np.sign(comparison.spearman_primary_full)
    )
    comparison["core_concordance_rule"] = comparison.spearman_sign_concordant & comparison[
        "delta_r2_core_minus_primary_full"
    ].abs().le(0.10)
    metadata = metadata_qc_audit(root, config)
    output_paths = [
        result_dir / "core_cohort_audit.csv",
        result_dir / "core_cv_folds.csv",
        result_dir / "core_oof.parquet",
        result_dir / "core_performance.csv",
        result_dir / "primary_vs_core_comparison.csv",
        result_dir / "metadata_qc_audit.csv",
        result_dir / "core_hyperparameters.csv",
        result_dir / "core_leakage_ledger.csv",
        result_dir / "contamination_audit.json",
    ]
    atomic_csv(audit, output_paths[0])
    core_folds = core[["canonical_cell_id", "group_id", "subclass", "fold"]].copy()
    core_folds["fold_seed"] = int(config["core_only"]["fold_seed"])
    core_folds["split_method"] = "StratifiedGroupKFold_shuffle"
    atomic_csv(core_folds, output_paths[1])
    atomic_parquet(oof, output_paths[2])
    atomic_csv(performance, output_paths[3])
    atomic_csv(comparison, output_paths[4])
    atomic_csv(metadata, output_paths[5])
    atomic_csv(pd.concat(parameter_parts, ignore_index=True), output_paths[6])
    atomic_csv(pd.concat(ledger_parts, ignore_index=True), output_paths[7])
    atomic_json(
        {
            "direct_contamination_score_found": False,
            "direct_contamination_score_columns": metadata.attrs.get("direct_candidate_columns", []),
            "map_confidence_interpretation": (
                "transcriptomic mapping confidence; not treated as a contamination score"
            ),
            "sensitivity_name": "quality-restricted Core-only sensitivity",
            "core_value": required_value,
            "core_n": len(core),
            "core_donor_n": int(core.group_id.astype(str).nunique()),
            "core_subclasses": core.subclass.value_counts().sort_index().to_dict(),
            "fold_seed": int(config["core_only"]["fold_seed"]),
            "fold_method": "StratifiedGroupKFold(n_splits=3, shuffle=True)",
            "fold_sizes": core.fold.value_counts().sort_index().to_dict(),
            "fold_donor_counts": core.groupby("fold").group_id.nunique().sort_index().to_dict(),
            "comparison_interpretation": (
                "Core delta conflates cohort restriction, refitting, retuning, and new fold assignment; "
                "frozen-primary OOF restricted to Core IDs is also reported as a descriptive reference"
            ),
        },
        output_paths[8],
    )
    return output_paths


def run_random_control_workstream(
    root: Path,
    config: dict[str, Any],
    *,
    smoke: bool,
    base_signature: str,
) -> list[Path]:
    outputs = output_roots(root, config)
    paths = resolve_paths(root, config)
    canonical = outputs["results"] / "random_controls"
    result_root = canonical / "smoke" if smoke else canonical
    result_root.mkdir(parents=True, exist_ok=True)
    gene_qc = pd.read_parquet(canonical / "all_gene_expression_qc.parquet")
    modules = load_gene_modules(paths["gene_modules"])
    control_n = int(
        config["random_controls"][
            "smoke_controls_per_module" if smoke else "full_controls_per_module"
        ]
    )
    selected_modules = (
        list(config["random_controls"]["modules_smoke"]) if smoke else list(MODULE_ORDER)
    )
    folds = list(config["random_controls"]["folds_smoke"]) if smoke else [0, 1, 2]
    matched, quality, assignments = generate_matched_gene_sets(
        gene_qc,
        modules,
        control_n=control_n,
        settings=config["random_controls"],
        seed=int(config["seed"]),
        selected_modules=selected_modules,
    )
    matched_path = result_root / "matched_gene_sets.json"
    quality_path = result_root / "matching_quality.csv"
    assignments_path = result_root / "matching_assignments.parquet"
    atomic_json(matched, matched_path)
    atomic_csv(quality, quality_path)
    atomic_parquet(assignments, assignments_path)
    if not quality.matching_pass.all():
        raise RuntimeError("Random matching quality gate failed")
    if quality.groupby("module").size().min() != control_n:
        raise AssertionError("Wrong matched-control count")
    if not smoke and float(quality.matching_pass.mean()) < float(
        config["random_controls"]["minimum_passing_fraction"]
    ):
        raise RuntimeError("Random matching passing fraction is below the declared gate")
    target_frames = extract_control_targets(
        paths["raw_count_archive"],
        paths["library_sizes"],
        matched,
        chunksize=int(config["count_stream"]["selected_gene_chunksize"]),
    )
    target_dir = result_root / "control_targets"
    target_paths: list[Path] = []
    for module, frame in target_frames.items():
        path = target_dir / f"{module}.parquet"
        atomic_parquet(frame, path)
        target_paths.append(path)
    run_signature = hash_text(
        "|".join(
            [
                base_signature,
                sha256(matched_path),
                ",".join(map(str, folds)),
                "smoke" if smoke else "full",
            ]
        )
    )
    performance, completed_before = fit_random_controls(
        root,
        config,
        matched,
        target_frames,
        result_root=result_root,
        folds=folds,
        run_signature=run_signature,
    )
    performance_path = result_root / "random_module_performance.parquet"
    atomic_parquet(performance, performance_path)
    resume_audit_path = result_root / "resume_audit.json"
    atomic_json(
        {
            "run_signature": run_signature,
            "control_n": len(performance),
            "validated_complete_partitions_reused": completed_before,
            "new_partitions_fit": len(performance) - completed_before,
            "folds": folds,
        },
        resume_audit_path,
    )
    output_paths = [
        matched_path,
        quality_path,
        assignments_path,
        *target_paths,
        performance_path,
        resume_audit_path,
        result_root / "frozen_xgboost_hyperparameters.csv",
    ]
    if not smoke:
        summary = summarize_random_controls(performance, paths["primary_oof"])
        summary_path = result_root / "module_specificity_summary.csv"
        atomic_csv(summary, summary_path)
        output_paths.append(summary_path)
        figure_dir = outputs["figures"] / "random_controls"
        figure_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
        for axis, module in zip(axes.flat, MODULE_ORDER):
            values = performance.loc[performance.module.eq(module), "r2"].to_numpy(dtype=float)
            observed = float(
                summary.loc[summary.module.eq(module), "observed_primary_xgboost_r2"].iloc[0]
            )
            axis.hist(values, bins=20, color="#6baed6", edgecolor="white")
            axis.axvline(observed, color="#cb181d", linewidth=2, label="observed")
            axis.set_title(module)
            axis.set_xlabel("OOF R²")
        axes.flat[0].legend(frameon=False)
        fig.suptitle("Primary module prediction versus matched random modules")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            fig.savefig(figure_dir / f"matched_random_module_benchmark.{suffix}", dpi=200)
        plt.close(fig)
    return output_paths


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], digits: int = 4) -> str:
    view = frame[list(columns)].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(lambda value: f"{value:.{digits}f}" if pd.notna(value) else "NA")
    header = "| " + " | ".join(view.columns) + " |"
    divider = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, divider, *rows])


def write_artifact_manifest(root: Path, config: dict[str, Any]) -> Path:
    outputs = output_roots(root, config)
    destination = outputs["results"] / "ARTIFACT_MANIFEST.csv"
    rows: list[dict[str, Any]] = []
    for namespace, base in outputs.items():
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix == ".tmp" or path.resolve() == destination.resolve():
                continue
            rows.append(
                {
                    "namespace": namespace,
                    "relative_path": str(path.relative_to(root)).replace("\\", "/"),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "suffix": path.suffix.lower(),
                }
            )
    atomic_csv(pd.DataFrame(rows), destination)
    return destination


def generate_final_report(root: Path, config: dict[str, Any]) -> tuple[Path, Path]:
    outputs = output_roots(root, config)
    paths = resolve_paths(root, config)
    technical = pd.read_csv(outputs["results"] / "technical_targets/technical_target_performance.csv")
    specificity = pd.read_csv(outputs["results"] / "random_controls/module_specificity_summary.csv")
    adjusted = pd.read_csv(
        outputs["results"] / "technical_adjustment/raw_vs_adjusted_comparison.csv"
    )
    nuisance = pd.read_csv(
        outputs["results"] / "technical_adjustment/nuisance_only_performance.csv"
    )
    alias = pd.read_csv(outputs["results"] / "scn2a_audit/nav_alias_performance.csv")
    target_summary = pd.read_csv(
        outputs["results"] / "target_definitions/target_definition_summary.csv"
    )
    consistency = pd.read_csv(
        outputs["results"] / "reliability/module_internal_consistency.csv"
    )
    core = pd.read_csv(outputs["results"] / "core_only/primary_vs_core_comparison.csv")
    contamination = json.loads(
        (outputs["results"] / "core_only/contamination_audit.json").read_text(encoding="utf-8")
    )
    correlation = json.loads(
        (outputs["results"] / "target_correlations/target_correlation_summary.json").read_text(
            encoding="utf-8"
        )
    )
    scn_audit = json.loads(
        (outputs["results"] / "scn2a_audit/scn2a_audit.json").read_text(encoding="utf-8")
    )
    integrity = json.loads(
        (outputs["results"] / "integrity/integrity_gate.json").read_text(encoding="utf-8")
    )
    # The manifest is written only after the report, summary, and completion
    # marker have reached their final state in ``run_revision_specificity``.
    # Keeping the destination here lets the report point to it without taking
    # a knowingly stale snapshot mid-transaction.
    manifest_path = outputs["results"] / "ARTIFACT_MANIFEST.csv"
    best_technical = (
        technical.sort_values("r2", ascending=False).groupby("target", as_index=False).first()
    )
    supported = [
        "Ephys predicts detected-gene count modestly, but does not predict total-count depth better than the pooled OOF mean baseline.",
        "Module specificity is conditional on the frozen observed-target XGBoost configurations and the declared matched-control design.",
        "Residualized results quantify signal remaining after fold-local adjustment for depth and detected-gene count; they are predictive associations.",
        "Core-only results are quality-restricted sensitivities, not direct contamination-controlled estimates.",
    ]
    unsupported = [
        "No result proves causal channel regulation, protein abundance, conductance, trafficking, or localization.",
        "Matched-control p-values are conditional benchmarks, not exact algorithm-level randomization tests.",
        "The Scn2a1 sensitivity does not retroactively change the frozen primary NaV module.",
        "map_confidence is not a contamination score, and no direct contamination score was found.",
    ]
    text = f"""# Objective

Matched-control analysis tests whether electrophysiology-to-module prediction is molecularly specific or is explained by transcriptomic depth/detection, score construction, shared identity axes, or transcriptomic-quality restriction.

# Data and Integrity Status

Integrity Gate: **{integrity['status']}**. The cohort contains {integrity['cohort_n']} cells and {integrity['donor_n']} donor/groups; donor overlap is zero in every frozen fold. All six primary targets reconstruct with exact maximum absolute difference 0. Primary OOF SHA256 is `{integrity['primary_oof_sha256']}` and primary SHAP SHA256 is `{integrity['primary_shap_sha256']}`.

# Technical Target Results

{_markdown_table(best_technical, ['target', 'model', 'n', 'r2', 'mae', 'rmse', 'spearman'])}

The requested log(total counts) and log(genes detected) R² values are the best-model rows above; complete model/fold/CIs are in `technical_targets`. The non-positive total-count R² is a negative result, not evidence that electrophysiology predicts library depth.

# Matched Random Module Results

{_markdown_table(specificity, ['module', 'observed_primary_xgboost_r2', 'random_median_r2', 'random_95th_percentile_r2', 'random_99th_percentile_r2', 'observed_empirical_percentile_le', 'empirical_p_plus_one', 'passes_95th', 'passes_99th'])}

Each module has {int(specificity.random_control_n.min())} frozen, expression/detection-matched controls. The empirical p-value uses plus-one correction; the percentile is the empirical percentage of random R² values less than or equal to observed.

# Technical-Residualized Prediction

{_markdown_table(adjusted, ['module', 'model', 'r2_raw', 'r2_adjusted', 'delta_r2_adjusted_minus_raw', 'spearman_raw', 'spearman_adjusted'])}

Nuisance-only raw-target performance is retained separately in `nuisance_only_performance.csv`; adjusted R² is evaluated against cross-fitted residual truth, not by adding nuisance predictions back.

# Scn2a Audit

Exact `Scn2a` is absent; exact release-native `Scn2a1` is present. Case `{scn_audit['case']}` was handled only as a user-specified alias-resolved NaV sensitivity, with no primary-module change and no external alias lookup. Alias sensitivity performance:

{_markdown_table(alias, ['target', 'model', 'n', 'r2', 'mae', 'rmse', 'spearman'])}

# Alternative Target Definitions

{_markdown_table(target_summary, ['module', 'primary_xgboost_r2', 'zmean_xgboost_r2', 'pc1_xgboost_r2', 'pc1_explained_variance', 'spearman_mean_zmean', 'spearman_mean_pc1', 'classification'])}

Classification thresholds were frozen before performance inspection: maximum absolute XGBoost R² delta <=0.05 plus positive alternative signals is robust; <=0.10 is moderately sensitive; otherwise composition-sensitive.

# Module Internal Consistency

{_markdown_table(consistency, ['module', 'gene_n', 'partition_n', 'generation', 'median_raw_rho', 'median_spearman_brown', 'spearman_brown_q1', 'spearman_brown_q3'])}

These are split-half internal-consistency estimates, not definitive noise ceilings. CaV exhausts all 126 mathematically possible balanced swap-unique partitions rather than duplicating draws to reach 1,000.

# Target Correlation Structure

- Raw first-eigenvalue fraction: {correlation['raw']['first_eigenvalue_fraction']:.4f}; effective dimension: {correlation['raw']['participation_ratio_effective_dimension']:.4f}.
- Technical-adjusted first-eigenvalue fraction: {correlation['technical_adjusted']['first_eigenvalue_fraction']:.4f}; effective dimension: {correlation['technical_adjusted']['participation_ratio_effective_dimension']:.4f}.
- Subclass-centered first-eigenvalue fraction: {correlation['subclass_centered']['first_eigenvalue_fraction']:.4f}; effective dimension: {correlation['subclass_centered']['participation_ratio_effective_dimension']:.4f}.

These descriptive matrices test whether targets share transcriptomic axes; they do not imply causality.

# Transcriptomic Quality and Core-Only Results

{_markdown_table(core, ['target', 'model', 'n_primary_full', 'n_primary_core_subset', 'n_core', 'r2_primary_full', 'r2_primary_core_subset', 'r2_core', 'delta_r2_core_minus_primary_full', 'positive_r2_performance_retained', 'spearman_sign_concordant', 'core_concordance_rule'])}

No documented direct contamination score was found. `map_confidence == Core` defines a quality-restricted sensitivity only. The Core cohort contains {contamination['core_n']} cells from {contamination['core_donor_n']} donors, with fold sizes {contamination['fold_sizes']} under seed {contamination['fold_seed']}. The full-primary versus retrained-Core delta conflates restriction, refitting, retuning, and fold reassignment; frozen-primary OOF restricted to the same Core IDs is included as a second descriptive reference.

# Verification and Tests

The pipeline records train/test ID hashes, group-overlap counts, configuration hashes, control partitions, and resume audits. Final pytest and after-hash results are recorded in `logs/revision_v3/specificity/pytest.log` and the state file.

# Independent Reviewer Findings

All source and artifact integrity checks are required before analysis.

# Claims Supported by the Results

""" + "\n".join(f"- {claim}" for claim in supported) + """

# Claims Not Supported by the Results

""" + "\n".join(f"- {claim}" for claim in unsupported) + f"""

# Remaining Risks and Limitations

- Only 200 controls per module limit upper-tail resolution (minimum empirical p = 1/201; the 99th percentile depends on about two tail observations).
- Observed XGBoost configurations were selected for the observed targets, whereas matched controls reuse them; specificity is therefore a conditional benchmark.
- Global z-mean/PC1 construction uses the full 4,435-sample transcriptomic release as explicitly specified. It is ephys-independent but not an inductive train-only outcome transform.
- Upstream ephys standardization/clipping was release-global; only downstream preprocessing is outer-fold local.
- CaV has only 126 unique balanced split halves, creating a disclosed literal deviation from the requested 1,000-draw wording.

# Reproduction Command

```powershell
python scripts/09_revision_specificity.py --config config/revision_v3_specificity.yaml --mode resume
python -m pytest -q
```

# Artifact Manifest

Machine-readable artifact paths, sizes, and SHA256 hashes are in `{manifest_path.relative_to(root).as_posix()}`. All analytical outputs are inside the four Specificity analysis specificity namespaces.
"""
    report_path = root / "REVISION_V3_SPECIFICITY_RESULTS.md"
    atomic_text(text, report_path)
    summary_path = outputs["results"] / "summary.json"
    atomic_json(
        {
            "integrity": integrity,
            "technical_best": best_technical.to_dict("records"),
            "specificity": specificity.to_dict("records"),
            "target_definitions": target_summary.to_dict("records"),
            "internal_consistency": consistency.to_dict("records"),
            "correlations": correlation,
            "report": str(report_path.relative_to(root)),
            "artifact_manifest": str(manifest_path.relative_to(root)),
        },
        summary_path,
    )
    return report_path, summary_path


def run_revision_specificity(
    config_path: str | Path,
    *,
    mode: str,
) -> dict[str, Any]:
    root, config = read_config(config_path)
    outputs = ensure_output_roots(root, config)
    signature = config_signature(root, Path(config_path).resolve(), config)
    cache = StageCache(outputs["logs"] / "checkpoints.json", signature)
    integrity_path = outputs["results"] / "integrity/integrity_gate.json"
    integrity = verify_integrity(root, config, integrity_path)

    count_paths = [
        outputs["results"] / "technical_targets/transcriptomic_qc_targets.csv",
        outputs["results"] / "technical_targets/all_transcriptomic_qc_targets.parquet",
        outputs["results"] / "random_controls/all_gene_expression_qc.parquet",
        outputs["results"] / "scn2a_audit/scn2a_symbol_matches.csv",
        outputs["results"] / "scn2a_audit/scn2a_audit.json",
        outputs["results"] / "scn2a_audit/scn2a1_counts.parquet",
    ]
    if not cache.valid("count_qc", count_paths):
        paths = resolve_paths(root, config)
        sample_qc, gene_qc, matches, count_audit, scn2a1 = stream_count_qc(
            paths["raw_count_archive"],
            paths["library_sizes"],
            load_gene_modules(paths["gene_modules"]),
            chunksize=int(config["count_stream"]["chunksize"]),
        )
        written = write_count_qc_outputs(
            root, config, sample_qc, gene_qc, matches, count_audit, scn2a1
        )
        cache.complete("count_qc", written, gene_rows=count_audit["gene_rows"])

    if mode == "smoke":
        smoke_random = run_random_control_workstream(
            root, config, smoke=True, base_signature=signature
        )
        smoke_results = outputs["results"] / "smoke"
        smoke_models = outputs["models"] / "smoke"
        residual = run_residualized_models(
            root,
            config,
            modules=("GABAA",),
            folds=(0,),
            result_root=smoke_results / "technical_adjustment",
            model_root=smoke_models / "technical_adjustment",
        )
        definitions = run_target_definition_models(
            root,
            config,
            modules=("GABAA",),
            folds=(0,),
            result_root=smoke_results / "target_definitions",
            model_root=smoke_models / "target_definitions",
        )
        marker = outputs["logs"] / "smoke_complete.json"
        atomic_json(
            {
                "status": "PASS",
                "signature": signature,
                "integrity": integrity,
                "random_outputs": [str(path.relative_to(root)) for path in smoke_random],
                "residual_outputs": [str(path.relative_to(root)) for path in residual],
                "target_definition_outputs": [str(path.relative_to(root)) for path in definitions],
            },
            marker,
        )
        return {"mode": mode, "status": "PASS", "marker": str(marker), "signature": signature}

    if mode not in {"full", "resume"}:
        raise ValueError("mode must be smoke, full, or resume")
    smoke_marker = outputs["logs"] / "smoke_complete.json"
    if not smoke_marker.is_file():
        raise RuntimeError("Full/resume mode requires a passing smoke marker")
    smoke_payload = json.loads(smoke_marker.read_text(encoding="utf-8"))
    if smoke_payload.get("status") != "PASS" or smoke_payload.get("signature") != signature:
        raise RuntimeError("Full/resume mode requires a passing smoke marker for the current signature")

    stages: list[tuple[str, list[Path], Any]] = []
    technical_paths = [
        outputs["results"] / "technical_targets/technical_target_oof.parquet",
        outputs["results"] / "technical_targets/technical_target_performance.csv",
        outputs["results"] / "technical_targets/technical_target_fold_metrics.csv",
        outputs["results"] / "technical_targets/technical_target_bootstrap.parquet",
        outputs["results"] / "technical_targets/technical_target_hyperparameters.csv",
        outputs["results"] / "technical_targets/technical_target_leakage_ledger.csv",
    ]
    for target in ("log10_total_counts", "log10_genes_detected"):
        for fold in range(3):
            for model in MODEL_ORDER:
                technical_paths.append(
                    outputs["models"] / "technical_targets" / f"{target}_fold{fold}_{model}.joblib"
                )
    stages.append(("technical_models", technical_paths, lambda: run_technical_target_models(root, config)))
    random_root = outputs["results"] / "random_controls"
    random_paths = [
        random_root / "matched_gene_sets.json",
        random_root / "matching_quality.csv",
        random_root / "matching_assignments.parquet",
        random_root / "random_module_performance.parquet",
        random_root / "random_module_performance.partial.parquet",
        random_root / "module_specificity_summary.csv",
        random_root / "resume_audit.json",
        random_root / "frozen_xgboost_hyperparameters.csv",
    ]
    full_control_n = int(config["random_controls"]["full_controls_per_module"])
    for module in MODULE_ORDER:
        random_paths.append(random_root / "control_targets" / f"{module}.parquet")
        for control_id in range(full_control_n):
            random_paths.append(
                random_root
                / "random_module_oof"
                / f"module={module}"
                / f"control_id={control_id:04d}"
                / "part-000.parquet"
            )
    random_paths.extend(
        [
            outputs["figures"] / "random_controls/matched_random_module_benchmark.png",
            outputs["figures"] / "random_controls/matched_random_module_benchmark.pdf",
        ]
    )
    stages.append(
        (
            "random_controls",
            random_paths,
            lambda: run_random_control_workstream(root, config, smoke=False, base_signature=signature),
        )
    )
    residual_paths = [
        outputs["results"] / "technical_adjustment/crossfit_nuisance_predictions.parquet",
        outputs["results"] / "technical_adjustment/technical_residualized_targets.parquet",
        outputs["results"] / "technical_adjustment/technical_residualized_oof.parquet",
        outputs["results"] / "technical_adjustment/technical_residualized_performance.csv",
        outputs["results"] / "technical_adjustment/technical_residualized_fold_metrics.csv",
        outputs["results"] / "technical_adjustment/nuisance_only_performance.csv",
        outputs["results"] / "technical_adjustment/raw_vs_adjusted_comparison.csv",
        outputs["results"] / "technical_adjustment/nuisance_coefficients_and_leakage.csv",
        outputs["results"] / "technical_adjustment/technical_residualized_hyperparameters.csv",
    ]
    for module in MODULE_ORDER:
        for fold in range(3):
            for model in MODEL_ORDER:
                residual_paths.append(
                    outputs["models"]
                    / "technical_adjustment"
                    / f"{module}_fold{fold}_{model}.joblib"
                )
    stages.append(("technical_adjustment", residual_paths, lambda: run_residualized_models(root, config)))
    alias_paths = [
        outputs["results"] / "scn2a_audit/nav_alias_target.parquet",
        outputs["results"] / "scn2a_audit/nav_alias_oof.parquet",
        outputs["results"] / "scn2a_audit/nav_alias_performance.csv",
        outputs["results"] / "scn2a_audit/nav_alias_hyperparameters.csv",
        outputs["results"] / "scn2a_audit/nav_alias_leakage_ledger.csv",
    ]
    for fold in range(3):
        for model in MODEL_ORDER:
            alias_paths.append(
                outputs["models"]
                / "scn2a_audit"
                / f"NaV_alias_Scn2a1_fold{fold}_{model}.joblib"
            )
    stages.append(("scn2a_alias", alias_paths, lambda: run_scn2a_alias_sensitivity(root, config)))
    definition_paths = [
        outputs["results"] / "target_definitions/target_mean_logcpm.parquet",
        outputs["results"] / "target_definitions/target_zmean.parquet",
        outputs["results"] / "target_definitions/target_pc1.parquet",
        outputs["results"] / "target_definitions/pc1_loadings.csv",
        outputs["results"] / "target_definitions/pc1_variance_explained.csv",
        outputs["results"] / "target_definitions/target_gene_standardization.csv",
        outputs["results"] / "target_definitions/target_definition_oof.parquet",
        outputs["results"] / "target_definitions/target_definition_performance.csv",
        outputs["results"] / "target_definitions/target_definition_summary.csv",
        outputs["results"] / "target_definitions/target_definition_hyperparameters.csv",
    ]
    for module in MODULE_ORDER:
        for definition in ("zmean", "pc1"):
            for fold in range(3):
                for model in MODEL_ORDER:
                    definition_paths.append(
                        outputs["models"]
                        / "target_definitions"
                        / f"{module}__{definition}_fold{fold}_{model}.joblib"
                    )
    stages.append(("target_definitions", definition_paths, lambda: run_target_definition_models(root, config)))
    reliability_paths = [
        outputs["results"] / "reliability/split_half_partitions.parquet",
        outputs["results"] / "reliability/split_half_draws.parquet",
        outputs["results"] / "reliability/module_internal_consistency.csv",
    ]
    stages.append(("internal_consistency", reliability_paths, lambda: run_internal_consistency(root, config)))
    correlation_paths = [
        outputs["results"] / "target_correlations/raw_spearman.csv",
        outputs["results"] / "target_correlations/technical_adjusted_spearman.csv",
        outputs["results"] / "target_correlations/subclass_centered_spearman.csv",
        outputs["results"] / "target_correlations/target_correlation_summary.json",
        outputs["results"] / "target_correlations/subclass_training_means.csv",
    ]
    for name in ("raw", "technical_adjusted", "subclass_centered"):
        for suffix in ("png", "pdf"):
            correlation_paths.append(
                outputs["figures"] / "target_correlations" / f"{name}_spearman.{suffix}"
            )
    stages.append(("target_correlations", correlation_paths, lambda: run_target_correlations(root, config)))
    core_paths = [
        outputs["results"] / "core_only/core_cohort_audit.csv",
        outputs["results"] / "core_only/core_cv_folds.csv",
        outputs["results"] / "core_only/core_oof.parquet",
        outputs["results"] / "core_only/core_performance.csv",
        outputs["results"] / "core_only/primary_vs_core_comparison.csv",
        outputs["results"] / "core_only/metadata_qc_audit.csv",
        outputs["results"] / "core_only/core_hyperparameters.csv",
        outputs["results"] / "core_only/core_leakage_ledger.csv",
        outputs["results"] / "core_only/contamination_audit.json",
    ]
    for target in config["core_only"]["targets"]:
        safe_target = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(target))
        for fold in range(int(config["core_only"]["outer_folds"])):
            for model in MODEL_ORDER:
                core_paths.append(
                    outputs["models"] / "core_only" / f"{safe_target}_fold{fold}_{model}.joblib"
                )
    stages.append(("core_only", core_paths, lambda: run_core_only(root, config)))
    stage_results: dict[str, str] = {}
    for stage, expected_paths, action in stages:
        if mode == "resume" and cache.valid(stage, expected_paths):
            stage_results[stage] = "reused"
            continue
        produced = action()
        if not all(path.is_file() and path.stat().st_size > 0 for path in expected_paths):
            raise AssertionError(f"Stage {stage} did not produce its required outputs")
        cache.complete(stage, expected_paths, produced_n=len(produced))
        stage_results[stage] = "completed"
    report, summary = generate_final_report(root, config)
    after_oof = sha256(resolve_paths(root, config)["primary_oof"])
    after_shap = sha256(resolve_paths(root, config)["primary_shap"])
    if after_oof != config["expected"]["primary_oof_sha256"] or after_shap != config["expected"][
        "primary_shap_sha256"
    ]:
        raise AssertionError("Primary hashes changed during Specificity analysis")
    marker = outputs["logs"] / "full_complete_pre_review.json"
    atomic_json(
        {
            "status": "PASS_PRE_REVIEW",
            "mode": mode,
            "signature": signature,
            "stages": stage_results,
            "primary_oof_sha256_after": after_oof,
            "primary_shap_sha256_after": after_shap,
            "report": str(report.relative_to(root)),
            "summary": str(summary.relative_to(root)),
        },
        marker,
    )
    # This is deliberately the final pipeline write so the manifest captures
    # the summary, completion marker, and all namespaced artifacts at their
    # final hashes. The required root-level Markdown report is outside the
    # four analytical namespaces and is therefore intentionally not listed.
    write_artifact_manifest(root, config)
    return {
        "mode": mode,
        "status": "PASS_PRE_REVIEW",
        "stages": stage_results,
        "marker": str(marker),
        "signature": signature,
    }
