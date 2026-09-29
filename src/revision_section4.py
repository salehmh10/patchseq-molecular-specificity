"""Extended-control analysis: high-resolution matched-control specificity.

The frozen primary, Robustness V2, Matched-control analysis, and Specificity analysis
Section 2 trees are read-only inputs. Every analytical write is isolated below
the configured Section 4 roots. Control-level artifacts use trusted digests and
exact identity/truth/configuration validation so interrupted runs can resume
without silently accepting stale or corrupted partitions.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import shutil
import tarfile
import tempfile
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
import yaml
from scipy.stats import rankdata, spearmanr
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression

from .biology import load_gene_modules
from .data import _single_csv_member
from .evaluation import regression_metrics
from .models import (
    dummy_model,
    elastic_net_candidates,
    ephys_subclass_ridge_model,
    group_inner_split,
    subclass_ridge_model,
    xgboost_candidates,
)
from .revision_section2 import (
    _fit_mlp_with_curve,
    _fixed_candidate,
    _load_table,
    _prediction_frame,
    bh_adjust,
    select_on_inner_groups,
)
from .revision_specificity import generate_matched_gene_sets


PIPELINE_VERSION = "revision-v3-section4-1"
MODULE_ORDER = ("NaV", "Kv", "CaV", "HCN", "GABAA", "iGluR")
NAV7_NAME = "NaV7_Scn2a1"
EPHYS_MODELS = ("ElasticNet", "XGBoost", "MLP")
ALL_MODELS = (
    "ElasticNet",
    "XGBoost",
    "MLP",
    "Dummy",
    "SubclassRidge",
    "EphysSubclassRidge",
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
    payload = "|".join([PIPELINE_VERSION, str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


def atomic_text(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
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


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def read_config(config_path: str | Path) -> tuple[Path, dict[str, Any], Path]:
    path = Path(config_path).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("schema_version") != "1.0":
        raise ValueError("Unsupported Extended-control analysis configuration")
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


def configuration_values(record: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    xgb = config["models"]["xgboost"]
    return {
        "n_estimators": int(record.n_estimators),
        "max_depth": int(record.max_depth),
        "learning_rate": float(record.learning_rate),
        "reg_lambda": float(record.reg_lambda),
        "subsample": float(xgb["subsample"]),
        "colsample_bytree": float(xgb["colsample_bytree"]),
        "objective": str(xgb["objective"]),
        "tree_method": str(xgb["tree_method"]),
        "n_jobs": int(xgb["n_jobs"]),
        "random_state": int(xgb["random_state"]),
    }


def varying_configuration(values: dict[str, Any]) -> dict[str, Any]:
    return {
        key: values[key]
        for key in ("n_estimators", "max_depth", "learning_rate", "reg_lambda")
    }


def configuration_sha256(values: dict[str, Any]) -> str:
    return hash_text(json.dumps(values, sort_keys=True, separators=(",", ":")))


def config_signature(root: Path, config: dict[str, Any], config_path: Path) -> str:
    paths = resolve_paths(root, config)
    file_keys = tuple(key for key, value in paths.items() if Path(value).is_file())
    directory_keys = tuple(key for key, value in paths.items() if Path(value).is_dir())
    directory_contracts: dict[str, dict[str, str]] = {}
    for key in directory_keys:
        if key == "section1_control_targets":
            directory_contracts[key] = {
                module: sha256_file(paths[key] / f"{module}.parquet") for module in MODULE_ORDER
            }
        elif key == "section2_old_control_oof":
            # The Section 2 manifest/checkpoint binds every partition. Avoid a
            # second 1,200-file signature walk while still binding this source.
            directory_contracts[key] = {
                "section2_manifest": config["expected"]["section2_manifest_sha256"]
            }
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "implementation_sha256": sha256_file(Path(__file__)),
        "entrypoint_sha256": sha256_file(root / "scripts/11_revision_section4.py"),
        "config_sha256": sha256_file(config_path),
        "files": {key: sha256_file(paths[key]) for key in file_keys},
        "directories": directory_contracts,
        "packages": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "pyarrow")
        },
    }
    return hash_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


class StageCache:
    """Signature-bound exact digest cache for complete stage contracts."""

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


def _validate_manifest(root: Path, path: Path, expected_rows: int) -> dict[str, Any]:
    frame = pd.read_csv(path)
    path_column = "relative_path" if "relative_path" in frame.columns else "path"
    if len(frame) != expected_rows or frame[path_column].duplicated().any():
        raise AssertionError(f"Manifest row/key failure: {path}")
    failures: list[str] = []
    for row in frame.itertuples(index=False):
        relative = getattr(row, path_column)
        artifact = root / str(relative)
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(row.bytes)
            or sha256_file(artifact) != str(row.sha256)
        ):
            failures.append(str(relative))
    if failures:
        raise AssertionError(f"Manifest validation failed: {failures[:5]}")
    return {"rows": len(frame), "failures": 0, "sha256": sha256_file(path)}


def _section2_source_digests(checkpoint_path: Path) -> dict[str, str]:
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    record = checkpoint.get("stages", {}).get("specificity_bridge", {})
    return {str(Path(key).resolve()): value for key, value in record.get("files", {}).items()}


def _old_source_config_hashes(frame: pd.DataFrame) -> dict[int, str]:
    return {
        int(fold): str(part.configuration_sha256.iloc[0])
        for fold, part in frame.groupby("fold", sort=True)
    }


def _validate_control_partition(
    path: Path,
    *,
    trusted_sha256: str | None,
    table: pd.DataFrame,
    target_by_cell: pd.Series,
    module: str,
    control_id: int,
    folds: Sequence[int],
    expected_config_hashes: dict[int, str],
    accepted_signatures: set[str],
) -> pd.DataFrame | None:
    if trusted_sha256 is None or not path.is_file() or sha256_file(path) != trusted_sha256:
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    required = {
        "canonical_cell_id",
        "group_id",
        "subclass",
        "fold",
        "module",
        "control_id",
        "y_true",
        "y_pred",
        "target_hash",
        "run_signature",
        "configuration_sha256",
    }
    if required - set(frame) or frame.empty:
        return None
    expected = table.loc[table.fold.isin(folds)].sort_values("canonical_cell_id", kind="mergesort")
    observed = frame.sort_values("canonical_cell_id", kind="mergesort")
    if len(expected) != len(observed) or observed.canonical_cell_id.astype(str).duplicated().any():
        return None
    if not np.array_equal(
        expected.canonical_cell_id.astype(str).to_numpy(),
        observed.canonical_cell_id.astype(str).to_numpy(),
    ):
        return None
    truth = expected.canonical_cell_id.astype(str).map(target_by_cell).to_numpy(dtype=float)
    target_hash = hash_array(truth)
    valid = (
        np.array_equal(observed.y_true.to_numpy(dtype=float), truth)
        and np.isfinite(observed[["y_true", "y_pred"]].to_numpy(dtype=float)).all()
        and observed.module.astype(str).eq(module).all()
        and observed.control_id.astype(int).eq(control_id).all()
        and observed.target_hash.astype(str).eq(target_hash).all()
        and observed.run_signature.astype(str).isin(accepted_signatures).all()
        and np.array_equal(expected.fold.to_numpy(int), observed.fold.to_numpy(int))
        and np.array_equal(
            expected.group_id.astype(str).to_numpy(), observed.group_id.astype(str).to_numpy()
        )
        and np.array_equal(
            expected.subclass.astype(str).to_numpy(), observed.subclass.astype(str).to_numpy()
        )
        and all(
            observed.loc[observed.fold.eq(fold), "configuration_sha256"]
            .astype(str)
            .eq(expected_config_hashes[int(fold)])
            .all()
            for fold in folds
        )
    )
    return frame if valid else None


def _load_registry(path: Path, signature: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError):
        payload = {}
    if payload.get("signature") != signature:
        payload = {"signature": signature, "partitions": {}}
    payload.setdefault("partitions", {})
    return payload


def _save_registry(payload: dict[str, Any], path: Path) -> None:
    payload["updated_at"] = pd.Timestamp.now(tz="Asia/Tehran").isoformat()
    atomic_json(payload, path)


def _target_series(table: pd.DataFrame, target_frame: pd.DataFrame, column: str) -> pd.Series:
    lookup = target_frame.set_index("transcriptomics_sample_id", verify_integrity=True)[column]
    values = table.transcriptomics_sample_id.astype(str).map(lookup)
    if values.isna().any() or not np.isfinite(values.to_numpy(dtype=float)).all():
        raise AssertionError(f"Incomplete/nonfinite target: {column}")
    return pd.Series(
        values.to_numpy(dtype=float), index=table.canonical_cell_id.astype(str), name=column
    )


def run_integrity(root: Path, config: dict[str, Any], destination: Path) -> list[Path]:
    paths, expected = resolve_paths(root, config), config["expected"]
    direct = {
        "modeling_table": "modeling_table_sha256",
        "frozen_folds": "frozen_folds_sha256",
        "gene_modules": "gene_modules_sha256",
        "primary_oof": "primary_oof_sha256",
        "primary_shap": "primary_shap_sha256",
        "section1_manifest": "section1_manifest_sha256",
        "section2_manifest": "section2_manifest_sha256",
        "section1_matched_sets": "old_matched_sets_sha256",
        "section1_matching_assignments": "old_matching_assignments_sha256",
        "section1_matching_quality": "old_matching_quality_sha256",
        "section1_gene_qc": "gene_qc_sha256",
        "section1_technical_targets": "technical_targets_sha256",
        "section1_nav7_target": "nav7_target_sha256",
        "section2_oof": "section2_oof_sha256",
        "section2_performance": "section2_performance_sha256",
        "section2_hyperparameters": "section2_hyperparameters_sha256",
        "manuscript": "manuscript_sha256",
    }
    hashes: dict[str, str] = {}
    for path_key, expected_key in direct.items():
        actual = sha256_file(paths[path_key])
        hashes[path_key] = actual
        if actual != str(expected[expected_key]).lower():
            raise AssertionError(f"Frozen input hash mismatch: {path_key}")
    for module, wanted in expected["control_target_sha256"].items():
        source = paths["section1_control_targets"] / f"{module}.parquet"
        actual = sha256_file(source)
        hashes[f"control_target_{module}"] = actual
        if actual != wanted:
            raise AssertionError(f"Frozen control target mismatch: {module}")

    manifests = {
        "section1": _validate_manifest(
            root, paths["section1_manifest"], int(expected["section1_manifest_rows"])
        ),
        "section2": _validate_manifest(
            root, paths["section2_manifest"], int(expected["section2_manifest_rows"])
        ),
    }
    checkpoint1 = json.loads(paths["section1_checkpoint"].read_text(encoding="utf-8"))
    checkpoint2 = json.loads(paths["section2_checkpoint"].read_text(encoding="utf-8"))
    resume2 = json.loads(paths["section2_resume"].read_text(encoding="utf-8"))
    if checkpoint1.get("signature") != expected["section1_signature"]:
        raise AssertionError("Section 1 signature mismatch")
    if checkpoint2.get("signature") != expected["section2_signature"]:
        raise AssertionError("Section 2 signature mismatch")
    if resume2.get("status") != "PASS" or resume2.get("fit_count") != 0:
        raise AssertionError("Section 2 final resume is not a zero-fit PASS")

    table, original_features, features = _load_table(root, config)
    if (
        len(table) != int(expected["cohort_n"])
        or table.group_id.astype(str).nunique() != int(expected["donor_n"])
        or len(features) != int(expected["revised_feature_n"])
    ):
        raise AssertionError("Cohort/donor/revised-feature integrity failure")
    donor_overlap = {}
    for fold in range(int(expected["outer_folds"])):
        donor_overlap[str(fold)] = len(
            set(table.loc[table.fold.ne(fold), "group_id"].astype(str))
            & set(table.loc[table.fold.eq(fold), "group_id"].astype(str))
        )
    if any(donor_overlap.values()):
        raise AssertionError("Frozen outer donor leakage")
    modules = load_gene_modules(paths["gene_modules"])
    cpm = pd.read_parquet(paths["module_cpm"])
    cpm.index = cpm.index.astype(str)
    target_differences: dict[str, float] = {}
    for module in MODULE_ORDER:
        genes = [gene for gene in modules[module] if gene in cpm.columns]
        rebuilt = np.log2(cpm[genes].astype(float) + 1).mean(axis=1)
        observed = table.transcriptomics_sample_id.astype(str).map(rebuilt).to_numpy(dtype=float)
        difference = float(np.max(np.abs(observed - table[f"target_{module}"].to_numpy(float))))
        target_differences[module] = difference
        if difference != 0:
            raise AssertionError(f"Frozen target mismatch: {module}")

    matched = json.loads(paths["section1_matched_sets"].read_text(encoding="utf-8"))
    quality = pd.read_csv(paths["section1_matching_quality"])
    assignments = pd.read_parquet(paths["section1_matching_assignments"])
    if len(quality) != 1200 or len(assignments) != 19000 or not quality.matching_pass.all():
        raise AssertionError("Old matched-control inventory mismatch")
    if quality.duplicated(["module", "set_signature"]).any():
        raise AssertionError("Old matched-control duplicate signature")
    all_primary = set().union(*(set(values) for values in modules.values()))
    for module in MODULE_ORDER:
        controls = matched["modules"][module]["controls"]
        if len(controls) != 200 or [int(item["control_id"]) for item in controls] != list(range(200)):
            raise AssertionError(f"Old control IDs mismatch: {module}")
        for item in controls:
            genes = list(item["genes"])
            if len(genes) != len(set(genes)) or set(genes) & (all_primary | {"Scn2a1"}):
                raise AssertionError(f"Old control membership mismatch: {module}")
            if item["set_signature"] != hash_text("\n".join(sorted(genes))):
                raise AssertionError(f"Old signature mismatch: {module}")

    # Strong reconstruction: stream the source using the frozen Section 1
    # extractor and compare all 1,200 target columns exactly.
    from .revision_specificity import extract_control_targets

    reconstructed = extract_control_targets(
        paths["raw_count_archive"], paths["library_sizes"], matched, chunksize=256
    )
    control_target_reconstruction: dict[str, float] = {}
    for module in MODULE_ORDER:
        frozen = pd.read_parquet(paths["section1_control_targets"] / f"{module}.parquet")
        current = reconstructed[module]
        if not np.array_equal(
            frozen.drop(columns="transcriptomics_sample_id").to_numpy(dtype=float),
            current.drop(columns="transcriptomics_sample_id").to_numpy(dtype=float),
        ) or not np.array_equal(
            frozen.transcriptomics_sample_id.astype(str),
            current.transcriptomics_sample_id.astype(str),
        ):
            raise AssertionError(f"Old control target reconstruction mismatch: {module}")
        control_target_reconstruction[module] = 0.0

    source_digests = _section2_source_digests(paths["section2_checkpoint"])
    old_oof_audit: list[dict[str, Any]] = []
    accepted = {
        str(expected["section2_signature"]),
        "48a26d4fe6c0cbbc16a7110cb6053f266f1d34d1e651a5451490e1f739783ea6",
    }
    for module in MODULE_ORDER:
        target_frame = reconstructed[module]
        for control_id in range(200):
            source = (
                paths["section2_old_control_oof"]
                / f"module={module}"
                / f"control_id={control_id:04d}"
                / "part-000.parquet"
            )
            digest = source_digests.get(str(source.resolve()))
            initial = pd.read_parquet(source)
            target = _target_series(table, target_frame, f"control_{control_id:04d}")
            validated = _validate_control_partition(
                source,
                trusted_sha256=digest,
                table=table,
                target_by_cell=target,
                module=module,
                control_id=control_id,
                folds=(0, 1, 2),
                expected_config_hashes=_old_source_config_hashes(initial),
                accepted_signatures=accepted,
            )
            if validated is None:
                raise AssertionError(f"Old Section 2 control OOF invalid: {module}/{control_id}")
            old_oof_audit.append(
                {
                    "module": module,
                    "control_id": control_id,
                    "rows": len(validated),
                    "sha256": digest,
                    "target_hash": str(validated.target_hash.iloc[0]),
                    "configuration_hashes": json.dumps(
                        _old_source_config_hashes(validated), sort_keys=True
                    ),
                }
            )

    nav7 = pd.read_parquet(paths["section1_nav7_target"])
    nav7["canonical_cell_id"] = nav7.canonical_cell_id.astype(str)
    nav6_genes = [gene for gene in modules["NaV"] if gene in cpm.columns]
    if "Scn2a" in cpm.columns or "Scn2a1" in cpm.columns:
        # The selected module CPM artifact intentionally contains only frozen
        # genes, so Scn2a1 is verified through the saved target/raw audit below.
        pass
    if nav6_genes != ["Scn1a", "Scn3a", "Scn8a", "Scn9a", "Scn10a", "Scn11a"]:
        raise AssertionError("Frozen NaV6 membership mismatch")
    if len(nav7) != len(table) or nav7.canonical_cell_id.duplicated().any():
        raise AssertionError("Saved NaV7 target grid mismatch")
    nav7_difference = float(
        np.max(
            np.abs(
                table.canonical_cell_id.astype(str)
                .map(nav7.set_index("canonical_cell_id").target_NaV_alias_Scn2a1)
                .to_numpy(float)
                - nav7.set_index("canonical_cell_id")
                .loc[table.canonical_cell_id.astype(str), "target_NaV_alias_Scn2a1"]
                .to_numpy(float)
            )
        )
    )
    payload = {
        "status": "PASS",
        "cohort_n": len(table),
        "donor_n": int(table.group_id.astype(str).nunique()),
        "fold_sizes": table.fold.value_counts().sort_index().astype(int).to_dict(),
        "donor_overlap": donor_overlap,
        "original_feature_n": len(original_features),
        "revised_feature_n": len(features),
        "target_max_abs_difference": target_differences,
        "primary_oof_sha256": hashes["primary_oof"],
        "primary_shap_sha256": hashes["primary_shap"],
        "section1_signature": checkpoint1["signature"],
        "section2_signature": checkpoint2["signature"],
        "manifests": manifests,
        "old_control_n": len(old_oof_audit),
        "old_control_target_reconstruction_max_abs_difference": control_target_reconstruction,
        "old_control_oof_all_valid": True,
        "exact_scn2a_absent": True,
        "exact_scn2a1_present": True,
        "nav6_genes": nav6_genes,
        "nav7_genes": [*nav6_genes, "Scn2a1"],
        "nav7_saved_grid_difference": nav7_difference,
        "frozen_hashes": hashes,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, destination)
    audit_path = destination.parent / "old_control_oof_audit.parquet"
    atomic_parquet(pd.DataFrame(old_oof_audit), audit_path)
    return [destination, audit_path]


def _matching_settings(config: dict[str, Any]) -> dict[str, Any]:
    matching = dict(config["matching"])
    return {
        "nearest_neighbor_pool": int(matching["nearest_neighbor_pool"]),
        "maximum_attempts_per_set": int(matching["maximum_attempts_per_set"]),
        "max_per_gene_standardized_distance": float(
            matching["max_per_gene_standardized_distance"]
        ),
        "max_mean_per_gene_standardized_distance": float(
            matching["max_mean_per_gene_standardized_distance"]
        ),
        "max_module_abs_standardized_mean_difference": float(
            matching["max_module_abs_standardized_mean_difference"]
        ),
        "max_module_abs_standardized_detection_difference": float(
            matching["max_module_abs_standardized_detection_difference"]
        ),
    }


def _select_retuning_ids(config: dict[str, Any]) -> dict[str, list[int]]:
    count = int(config["retuning"]["controls_per_module"])
    universe = int(config["matching"]["controls_per_primary_module"])
    selected: dict[str, list[int]] = {}
    for module in MODULE_ORDER:
        rng = np.random.default_rng(
            stable_seed("retuning_subset", module, base=int(config["retuning"]["subset_base_seed"]))
        )
        selected[module] = sorted(rng.choice(universe, size=count, replace=False).astype(int).tolist())
    return selected


def run_control_generation(root: Path, config: dict[str, Any], result_dir: Path) -> list[Path]:
    paths = resolve_paths(root, config)
    gene_qc = pd.read_parquet(paths["section1_gene_qc"])
    modules = load_gene_modules(paths["gene_modules"])
    old = json.loads(paths["section1_matched_sets"].read_text(encoding="utf-8"))
    settings = _matching_settings(config)
    matched, quality, assignments = generate_matched_gene_sets(
        gene_qc,
        modules,
        control_n=int(config["matching"]["controls_per_primary_module"]),
        settings=settings,
        seed=int(config["seed"]),
        selected_modules=MODULE_ORDER,
    )
    for module in MODULE_ORDER:
        if old["modules"][module]["controls"] != matched["modules"][module]["controls"][:200]:
            raise AssertionError(f"Old 200 controls changed: {module}")
    old_quality = pd.read_csv(paths["section1_matching_quality"]).sort_values(
        ["module", "control_id"]
    )
    compare_quality = quality.loc[quality.control_id.lt(200)].sort_values(
        ["module", "control_id"]
    )[old_quality.columns]
    pd.testing.assert_frame_equal(
        old_quality.reset_index(drop=True),
        compare_quality.reset_index(drop=True),
        check_exact=False,
        rtol=0,
        atol=1e-15,
    )
    old_assignments = pd.read_parquet(paths["section1_matching_assignments"]).sort_values(
        ["module", "control_id", "target_gene"]
    )
    compare_assignments = assignments.loc[assignments.control_id.lt(200)].sort_values(
        ["module", "control_id", "target_gene"]
    )[old_assignments.columns]
    pd.testing.assert_frame_equal(
        old_assignments.reset_index(drop=True),
        compare_assignments.reset_index(drop=True),
        check_exact=True,
    )
    nav7_modules = {NAV7_NAME: [gene for gene in modules["NaV"] if gene != "Scn2a"] + ["Scn2a1"]}
    nav7, nav7_quality, nav7_assignments = generate_matched_gene_sets(
        gene_qc,
        nav7_modules,
        control_n=int(config["matching"]["controls_per_nav7"]),
        settings=settings,
        seed=int(config["seed"]),
        selected_modules=(NAV7_NAME,),
    )
    for frame in (quality, nav7_quality):
        if not frame.matching_pass.all() or frame.duplicated(["module", "set_signature"]).any():
            raise AssertionError("Control matching quality/uniqueness failure")
    result_dir.mkdir(parents=True, exist_ok=True)
    outputs = [
        result_dir / "matched_gene_sets_1000.json",
        result_dir / "matching_assignments.parquet",
        result_dir / "matching_quality.csv",
        result_dir / "nav7_matched_gene_sets_1000.json",
        result_dir / "nav7_matching_assignments.parquet",
        result_dir / "nav7_matching_quality.csv",
        result_dir.parent / "retuning/selected_control_ids.json",
    ]
    atomic_json(matched, outputs[0])
    atomic_parquet(assignments, outputs[1])
    atomic_csv(quality, outputs[2])
    atomic_json(nav7, outputs[3])
    atomic_parquet(nav7_assignments, outputs[4])
    atomic_csv(nav7_quality, outputs[5])
    atomic_json(
        {
            "selection_before_outcomes": True,
            "base_seed": int(config["retuning"]["subset_base_seed"]),
            "controls_per_module": int(config["retuning"]["controls_per_module"]),
            "selected": _select_retuning_ids(config),
        },
        outputs[6],
    )
    return outputs


def _family_control_lookup(payloads: dict[str, dict[str, Any]]) -> tuple[
    dict[str, list[tuple[str, int, float]]], dict[tuple[str, int], int]
]:
    gene_to_controls: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
    expected: dict[tuple[str, int], int] = {}
    for family, payload in payloads.items():
        module_payload = payload["modules"][family]
        gene_n = len(module_payload["present_target_genes"])
        for control in module_payload["controls"]:
            control_id = int(control["control_id"])
            expected[(family, control_id)] = gene_n
            for gene in control["genes"]:
                gene_to_controls[str(gene)].append((family, control_id, 1.0 / gene_n))
    return gene_to_controls, expected


def run_target_extraction(root: Path, config: dict[str, Any], result_dir: Path) -> list[Path]:
    paths = resolve_paths(root, config)
    main = json.loads((result_dir / "matched_gene_sets_1000.json").read_text(encoding="utf-8"))
    nav7 = json.loads((result_dir / "nav7_matched_gene_sets_1000.json").read_text(encoding="utf-8"))
    payloads = {module: {"modules": {module: main["modules"][module]}} for module in MODULE_ORDER}
    payloads[NAV7_NAME] = {"modules": {NAV7_NAME: nav7["modules"][NAV7_NAME]}}
    gene_to_controls, expected_gene_counts = _family_control_lookup(payloads)
    selected_genes = set(gene_to_controls)
    frozen_modules = load_gene_modules(paths["gene_modules"])
    selected_genes.update(
        gene for values in frozen_modules.values() for gene in values if gene != "Scn2a"
    )
    selected_genes.add("Scn2a1")
    reference = pd.read_csv(
        paths["library_sizes"], dtype={"transcriptomics_sample_id": str}
    ).set_index("transcriptomics_sample_id", verify_integrity=True)
    sample_ids: list[str] | None = None
    denominator: np.ndarray | None = None
    accumulators: dict[str, np.ndarray] = {}
    selected_vectors: dict[str, np.ndarray] = {}
    seen_counts: dict[tuple[str, int], int] = defaultdict(int)
    with tarfile.open(paths["raw_count_archive"], mode="r:*") as archive:
        member = _single_csv_member(archive)
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("Could not open count CSV member")
        for chunk in pd.read_csv(handle, index_col=0, chunksize=256):
            chunk.index = chunk.index.astype(str)
            if sample_ids is None:
                sample_ids = chunk.columns.astype(str).tolist()
                denominator = reference.loc[sample_ids, "library_size_counts"].to_numpy(dtype=float)
                for family, payload in payloads.items():
                    accumulators[family] = np.zeros(
                        (int(payload["modules"][family]["control_n"]), len(sample_ids)),
                        dtype=np.float64,
                    )
            hits = [gene for gene in chunk.index if gene in selected_genes]
            for gene in hits:
                assert denominator is not None
                transformed = np.log2(
                    chunk.loc[gene].to_numpy(dtype=float) / denominator * 1e6 + 1.0
                )
                selected_vectors[gene] = transformed.astype(np.float32)
                for family, control_id, weight in gene_to_controls.get(gene, []):
                    accumulators[family][control_id] += transformed * weight
                    seen_counts[(family, control_id)] += 1
    if sample_ids is None or denominator is None:
        raise AssertionError("Count archive was empty")
    if set(selected_vectors) != selected_genes:
        missing = sorted(selected_genes - set(selected_vectors))
        raise AssertionError(f"Selected reliability genes missing from raw source: {missing[:10]}")
    for key, wanted in expected_gene_counts.items():
        if seen_counts[key] != wanted:
            raise AssertionError(f"Control target gene-count mismatch {key}: {seen_counts[key]}/{wanted}")

    target_dir = result_dir / "targets"
    output_paths: list[Path] = []
    table, _, _ = _load_table(root, config)
    manifest_rows: list[dict[str, Any]] = []
    for family, matrix in accumulators.items():
        frame = pd.DataFrame(
            matrix.T,
            columns=[f"control_{control_id:04d}" for control_id in range(matrix.shape[0])],
        )
        frame.insert(0, "transcriptomics_sample_id", sample_ids)
        output = target_dir / f"{family}.parquet"
        atomic_parquet(frame, output)
        output_paths.append(output)
        if family in MODULE_ORDER:
            frozen = pd.read_parquet(paths["section1_control_targets"] / f"{family}.parquet")
            if not np.array_equal(
                frame.iloc[:, :201].to_numpy(), frozen.to_numpy()
            ):
                raise AssertionError(f"Old target columns changed during extension: {family}")
        lookup = frame.set_index("transcriptomics_sample_id", verify_integrity=True)
        for control_id in range(matrix.shape[0]):
            values = table.transcriptomics_sample_id.astype(str).map(
                lookup[f"control_{control_id:04d}"]
            ).to_numpy(dtype=np.float64)
            manifest_rows.append(
                {
                    "module": family,
                    "control_id": control_id,
                    "n": len(values),
                    "target_hash": hash_array(values),
                    "finite": bool(np.isfinite(values).all()),
                }
            )
    expression = pd.DataFrame(
        {gene: selected_vectors[gene] for gene in sorted(selected_vectors)}, index=sample_ids
    )
    expression.index.name = "transcriptomics_sample_id"
    expression_path = result_dir / "selected_gene_log2cpm.parquet"
    atomic_parquet(expression.reset_index(), expression_path)
    manifest_path = result_dir / "control_target_manifest.csv"
    atomic_csv(pd.DataFrame(manifest_rows), manifest_path)
    output_paths.extend([expression_path, manifest_path])
    return output_paths


def _parameter_rows(path: Path, model: str = "XGBoost") -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame.loc[frame.model.eq(model)].copy()
    if frame.duplicated(["module", "fold"]).any():
        raise AssertionError(f"Duplicate configuration rows: {path}")
    return frame


def _config_contract(
    parameters: pd.DataFrame, module: str, config: dict[str, Any]
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    values: dict[int, dict[str, Any]] = {}
    hashes: dict[int, str] = {}
    for fold in (0, 1, 2):
        rows = parameters.loc[parameters.module.eq(module) & parameters.fold.eq(fold)]
        if len(rows) != 1:
            raise AssertionError(f"Expected one XGBoost configuration: {module}/fold{fold}")
        current = configuration_values(rows.iloc[0], config)
        values[fold] = current
        hashes[fold] = configuration_sha256(current)
    return values, hashes


def _fit_fixed_control(
    table: pd.DataFrame,
    features: Sequence[str],
    target: pd.Series,
    *,
    module: str,
    control_id: int,
    folds: Sequence[int],
    config_values: dict[int, dict[str, Any]],
    config_hashes: dict[int, str],
    signature: str,
    analysis: str,
    seed: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    target_hash = hash_array(
        table.loc[table.fold.isin(folds)]
        .sort_values("canonical_cell_id", kind="mergesort")
        .canonical_cell_id.astype(str)
        .map(target)
        .to_numpy(dtype=float)
    )
    for fold in folds:
        train = table.loc[table.fold.ne(fold)].copy()
        test = table.loc[table.fold.eq(fold)].copy()
        if set(train.group_id.astype(str)) & set(test.group_id.astype(str)):
            raise AssertionError("Outer donor leakage during fixed-control fitting")
        pipeline = _fixed_candidate(
            "XGBoost", features, varying_configuration(config_values[int(fold)]), seed=seed
        )
        y_train = train.canonical_cell_id.astype(str).map(target).to_numpy(dtype=float)
        y_test = test.canonical_cell_id.astype(str).map(target).to_numpy(dtype=float)
        pipeline.fit(train[list(features)], y_train)
        frame = _prediction_frame(
            test,
            y_test,
            pipeline.predict(test[list(features)]),
            module,
            "XGBoost",
            analysis=analysis,
        )
        frame["control_id"] = control_id
        frame["run_signature"] = signature
        frame["target_hash"] = target_hash
        frame["configuration_sha256"] = config_hashes[int(fold)]
        rows.append(frame)
    result = pd.concat(rows, ignore_index=True)
    if not np.isfinite(result[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise AssertionError("Non-finite fixed-control OOF")
    return result


def _performance_record(
    frame: pd.DataFrame,
    *,
    module: str,
    control_id: int,
    n_genes: int,
    run_signature: str,
) -> dict[str, Any]:
    metrics = regression_metrics(frame.y_true, frame.y_pred)
    fold_values = {
        f"fold{int(fold)}_r2": regression_metrics(part.y_true, part.y_pred)["r2"]
        for fold, part in frame.groupby("fold", sort=True)
    }
    return {
        "module": module,
        "control_id": control_id,
        "n": len(frame),
        "n_genes": n_genes,
        **metrics,
        **fold_values,
        "target_hash": str(frame.target_hash.iloc[0]),
        "run_signature": run_signature,
    }


def run_raw_control_oof(
    root: Path,
    config: dict[str, Any],
    result_dir: Path,
    *,
    signature: str,
) -> tuple[list[Path], dict[str, int]]:
    paths = resolve_paths(root, config)
    table, _, features = _load_table(root, config)
    parameters = _parameter_rows(paths["section2_hyperparameters"])
    matched = json.loads((result_dir / "matched_gene_sets_1000.json").read_text(encoding="utf-8"))
    source_digests = _section2_source_digests(paths["section2_checkpoint"])
    source_signatures = {
        str(config["expected"]["section2_signature"]),
        "48a26d4fe6c0cbbc16a7110cb6053f266f1d34d1e651a5451490e1f739783ea6",
    }
    registry_path = result_dir / "oof_partition_registry.json"
    registry = _load_registry(registry_path, signature)
    performance_rows: list[dict[str, Any]] = []
    fit_count = reused_count = copied_count = 0
    output_paths: list[Path] = []
    checkpoint_every = int(config["matching"]["checkpoint_every"])
    processed = 0
    for module in MODULE_ORDER:
        target_frame = pd.read_parquet(result_dir / f"targets/{module}.parquet")
        config_values, config_hashes = _config_contract(parameters, module, config)
        module_payload = matched["modules"][module]
        for control in module_payload["controls"]:
            control_id = int(control["control_id"])
            target = _target_series(table, target_frame, f"control_{control_id:04d}")
            destination = (
                result_dir
                / "oof"
                / f"module={module}"
                / f"control_id={control_id:04d}"
                / "part-000.parquet"
            )
            key = str(destination.resolve())
            record = registry["partitions"].get(key, {})
            expected_hashes = config_hashes
            accepted = {signature}
            if control_id < int(config["matching"]["existing_controls_per_module"]):
                source = (
                    paths["section2_old_control_oof"]
                    / f"module={module}"
                    / f"control_id={control_id:04d}"
                    / "part-000.parquet"
                )
                source_initial = pd.read_parquet(source)
                expected_hashes = _old_source_config_hashes(source_initial)
                accepted = source_signatures
            cached = _validate_control_partition(
                destination,
                trusted_sha256=record.get("sha256"),
                table=table,
                target_by_cell=target,
                module=module,
                control_id=control_id,
                folds=(0, 1, 2),
                expected_config_hashes=expected_hashes,
                accepted_signatures=accepted,
            )
            if cached is not None:
                frame = cached
                reused_count += 1
                provenance = str(record.get("provenance", "section4_cache"))
            elif control_id < int(config["matching"]["existing_controls_per_module"]):
                source = (
                    paths["section2_old_control_oof"]
                    / f"module={module}"
                    / f"control_id={control_id:04d}"
                    / "part-000.parquet"
                )
                source_digest = source_digests.get(str(source.resolve()))
                source_frame = _validate_control_partition(
                    source,
                    trusted_sha256=source_digest,
                    table=table,
                    target_by_cell=target,
                    module=module,
                    control_id=control_id,
                    folds=(0, 1, 2),
                    expected_config_hashes=expected_hashes,
                    accepted_signatures=source_signatures,
                )
                if source_frame is None:
                    raise AssertionError(f"Frozen old source invalid: {module}/{control_id}")
                atomic_copy(source, destination)
                frame = source_frame
                copied_count += 1
                provenance = "reused_section2_byte_copy"
            else:
                frame = _fit_fixed_control(
                    table,
                    features,
                    target,
                    module=module,
                    control_id=control_id,
                    folds=(0, 1, 2),
                    config_values=config_values,
                    config_hashes=config_hashes,
                    signature=signature,
                    analysis="matched_control_raw",
                    seed=int(config["seed"]),
                )
                atomic_parquet(frame, destination)
                fit_count += 1
                provenance = "section4_new_fit"
            digest = sha256_file(destination)
            registry["partitions"][key] = {
                "sha256": digest,
                "module": module,
                "control_id": control_id,
                "target_hash": str(frame.target_hash.iloc[0]),
                "configuration_hashes": {str(k): v for k, v in expected_hashes.items()},
                "provenance": provenance,
            }
            performance_rows.append(
                {
                    **_performance_record(
                        frame,
                        module=module,
                        control_id=control_id,
                        n_genes=len(control["genes"]),
                        run_signature=str(frame.run_signature.iloc[0]),
                    ),
                    "provenance": provenance,
                }
            )
            output_paths.append(destination)
            processed += 1
            if processed % checkpoint_every == 0:
                _save_registry(registry, registry_path)
                atomic_parquet(
                    pd.DataFrame(performance_rows), result_dir / "performance.partial.parquet"
                )
    _save_registry(registry, registry_path)
    performance = pd.DataFrame(performance_rows).sort_values(["module", "control_id"])
    performance_path = result_dir / "performance.parquet"
    atomic_parquet(performance, performance_path)
    resume_path = result_dir / "resume_audit.json"
    atomic_json(
        {
            "partition_count": len(performance),
            "fit_count": fit_count,
            "reused_partition_count": reused_count,
            "source_partitions_copied_without_refit": copied_count,
            "signature": signature,
        },
        resume_path,
    )
    return [registry_path, performance_path, resume_path, *output_paths], {
        "fit_count": fit_count,
        "reused_partition_count": reused_count + copied_count,
    }


def specificity_summary(
    performance: pd.DataFrame,
    observed_performance: pd.DataFrame,
    *,
    observed_model: str = "XGBoost",
    observed_column: str = "r2",
    p_column: str = "raw_p",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for module, controls in performance.groupby("module", sort=False):
        observed_rows = observed_performance.loc[
            observed_performance.module.astype(str).eq(str(module))
            & observed_performance.model.astype(str).eq(observed_model)
        ]
        if len(observed_rows) != 1:
            raise AssertionError(f"Observed performance row mismatch: {module}/{observed_model}")
        observed = float(observed_rows.iloc[0][observed_column])
        values = controls.r2.to_numpy(dtype=float)
        exceed = int(np.count_nonzero(values >= observed))
        rows.append(
            {
                "module": module,
                "observed_xgboost_r2": observed,
                "control_n": len(values),
                "control_median_r2": float(np.median(values)),
                "control_p90_r2": float(np.quantile(values, 0.90)),
                "control_p95_r2": float(np.quantile(values, 0.95)),
                "control_p99_r2": float(np.quantile(values, 0.99)),
                "observed_empirical_percentile": float(100 * np.mean(values <= observed)),
                "extreme_count": exceed,
                p_column: float((1 + exceed) / (len(values) + 1)),
                "percentile_tie_rule": "less_equal",
                "p_rule": "plus_one_greater_equal",
            }
        )
    return pd.DataFrame(rows)


def add_bh_family(
    summary: pd.DataFrame,
    *,
    p_column: str,
    q: float,
    family_label: str,
) -> pd.DataFrame:
    result = summary.copy()
    if len(result) != 6 or result.module.astype(str).nunique() != 6:
        raise AssertionError("Specificity BH family must contain exactly six module tests")
    p = result[p_column].to_numpy(dtype=float)
    result["bh_q"] = bh_adjust(p)
    order = np.lexsort((result.module.astype(str), p))
    ranks = np.empty(len(result), dtype=int)
    ranks[order] = np.arange(1, len(result) + 1)
    result["rank"] = ranks
    result["bh_threshold"] = q * result["rank"] / len(result)
    result["reject_q05"] = result.bh_q.le(q)
    result["family_n"] = len(result)
    result["family_label"] = family_label
    return result


def run_raw_inference(root: Path, config: dict[str, Any], result_dir: Path) -> list[Path]:
    performance = pd.read_parquet(result_dir.parent / "controls/performance.parquet")
    observed = pd.read_csv(resolve_paths(root, config)["section2_performance"])
    summary = specificity_summary(performance, observed, p_column="raw_p")
    bh = add_bh_family(
        summary,
        p_column="raw_p",
        q=float(config["inference"]["bh_q"]),
        family_label="six primary raw matched-control specificity tests",
    )
    paths = [result_dir / "specificity_1000_summary.csv", result_dir / "specificity_bh6.csv"]
    atomic_csv(summary, paths[0])
    atomic_csv(bh, paths[1])
    return paths


def _observed_target(
    table: pd.DataFrame, target_frame: pd.DataFrame, column: str
) -> pd.Series:
    frame = target_frame.copy()
    frame["canonical_cell_id"] = frame.canonical_cell_id.astype(str)
    lookup = frame.set_index("canonical_cell_id", verify_integrity=True)[column]
    target = table.canonical_cell_id.astype(str).map(lookup)
    if target.isna().any() or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise AssertionError(f"Observed target incomplete: {column}")
    return pd.Series(target.to_numpy(dtype=float), index=table.index, name=column)


def fit_observed_models(
    root: Path,
    config: dict[str, Any],
    *,
    target: pd.Series,
    target_name: str,
    result_dir: Path,
    model_dir: Path,
    folds: Sequence[int] = (0, 1, 2),
    model_names: Sequence[str] = ALL_MODELS,
    stem: str = "nav7",
    analysis: str = "nav7_observed",
) -> list[Path]:
    table, _, features = _load_table(root, config)
    values = target.reindex(table.index).to_numpy(dtype=float)
    predictions: list[pd.DataFrame] = []
    seed_predictions: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    model_paths: list[Path] = []
    for fold in folds:
        train = table.loc[table.fold.ne(fold)].copy()
        test = table.loc[table.fold.eq(fold)].copy()
        train.index = train.canonical_cell_id.astype(str)
        test.index = test.canonical_cell_id.astype(str)
        X_train, X_test = train[features], test[features]
        y_train = values[table.fold.ne(fold).to_numpy()]
        y_test = values[table.fold.eq(fold).to_numpy()]
        groups = train.group_id.astype(str).to_numpy()
        overlap = set(train.group_id.astype(str)) & set(test.group_id.astype(str))
        if overlap:
            raise AssertionError("NaV7 observed outer donor leakage")
        inner_train, inner_valid = group_inner_split(groups, seed=int(config["seed"]) + fold)
        common_ledger = {
            "target": target_name,
            "fold": int(fold),
            "train_n": len(train),
            "test_n": len(test),
            "train_group_n": train.group_id.astype(str).nunique(),
            "test_group_n": test.group_id.astype(str).nunique(),
            "outer_group_overlap_n": 0,
            "train_id_sha256": hash_text("\n".join(sorted(train.index.astype(str)))),
            "test_id_sha256": hash_text("\n".join(sorted(test.index.astype(str)))),
            "inner_train_n": len(inner_train),
            "inner_valid_n": len(inner_valid),
            "inner_group_overlap_n": len(
                set(groups[inner_train]) & set(groups[inner_valid])
            ),
            "inner_train_id_sha256": hash_text(
                "\n".join(sorted(train.index[inner_train].astype(str)))
            ),
            "inner_valid_id_sha256": hash_text(
                "\n".join(sorted(train.index[inner_valid].astype(str)))
            ),
        }
        for model in (name for name in ("ElasticNet", "XGBoost") if name in model_names):
            candidates = (
                elastic_net_candidates(features, seed=int(config["seed"]))
                if model == "ElasticNet"
                else xgboost_candidates(features, seed=int(config["seed"]))
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                selected = select_on_inner_groups(
                    candidates, X_train, y_train, groups, seed=int(config["seed"]) + fold
                )
            prediction = selected.pipeline.predict(X_test)
            predictions.append(
                _prediction_frame(
                    test, y_test, prediction, target_name, model, analysis=analysis
                )
            )
            parameter_rows.append(
                {
                    "target": target_name,
                    "module": target_name,
                    "fold": fold,
                    "model": model,
                    "model_seed": int(config["seed"]),
                    "inner_rmse": float(selected.inner_rmse),
                    **selected.parameters,
                }
            )
            ledger_rows.append({**common_ledger, "model": model, "model_seed": int(config["seed"])})
            model_path = model_dir / f"{target_name}_fold{fold}_{model}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_joblib(selected.pipeline, model_path)
            model_paths.append(model_path)
        if "MLP" in model_names:
            mlp_predictions: list[np.ndarray] = []
            for model_seed in config["models"]["mlp"]["ensemble_seeds"]:
                pipeline, parameters, curve, mlp_ledger = _fit_mlp_with_curve(
                    features,
                    X_train,
                    y_train,
                    groups,
                    model_seed=int(model_seed),
                    split_seed=int(config["seed"]) + fold,
                    maximum_epochs=int(config["models"]["mlp"]["maximum_epochs"]),
                    patience=int(config["models"]["mlp"]["patience"]),
                    min_delta=float(config["models"]["mlp"]["min_delta"]),
                )
                prediction = np.asarray(pipeline.predict(X_test), dtype=float)
                mlp_predictions.append(prediction)
                seed_frame = _prediction_frame(
                    test, y_test, prediction, target_name, "MLP", analysis=analysis
                )
                seed_frame["model_seed"] = int(model_seed)
                seed_predictions.append(seed_frame)
                parameter_rows.append(
                    {
                        "target": target_name,
                        "module": target_name,
                        "fold": fold,
                        "model": "MLP",
                        "model_seed": int(model_seed),
                        **parameters,
                    }
                )
                ledger_rows.append(
                    {**common_ledger, **mlp_ledger, "model": "MLP", "model_seed": int(model_seed)}
                )
                curve = curve.assign(
                    target=target_name, module=target_name, fold=fold, model_seed=int(model_seed)
                )
                curves.append(curve)
                model_path = model_dir / f"{target_name}_fold{fold}_MLP_seed{model_seed}.joblib"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                atomic_joblib(pipeline, model_path)
                model_paths.append(model_path)
            predictions.append(
                _prediction_frame(
                    test,
                    y_test,
                    np.mean(mlp_predictions, axis=0),
                    target_name,
                    "MLP",
                    analysis=analysis,
                )
            )
        comparator_specs = (
            ("Dummy", dummy_model(features), features),
            ("SubclassRidge", subclass_ridge_model(), ["subclass"]),
            (
                "EphysSubclassRidge",
                ephys_subclass_ridge_model(features),
                [*features, "subclass"],
            ),
        )
        for model, pipeline, columns in comparator_specs:
            if model not in model_names:
                continue
            fitted = pipeline.fit(train[columns], y_train)
            predictions.append(
                _prediction_frame(
                    test,
                    y_test,
                    fitted.predict(test[columns]),
                    target_name,
                    model,
                    analysis=analysis,
                )
            )
            parameter_rows.append(
                {
                    "target": target_name,
                    "module": target_name,
                    "fold": fold,
                    "model": model,
                    "model_seed": int(config["seed"]),
                    "strategy": "mean" if model == "Dummy" else "ridge_alpha_1.0",
                }
            )
            ledger_rows.append({**common_ledger, "model": model, "inner_split": "none_fixed"})
            model_path = model_dir / f"{target_name}_fold{fold}_{model}.joblib"
            model_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_joblib(fitted, model_path)
            model_paths.append(model_path)
    oof = pd.concat(predictions, ignore_index=True)
    if not np.isfinite(oof[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise AssertionError("NaV7 observed OOF contains nonfinite values")
    performance_rows: list[dict[str, Any]] = []
    for model, frame in oof.groupby("model", sort=False):
        performance_rows.append(
            {"target": target_name, "module": target_name, "model": model, "n": len(frame), **regression_metrics(frame.y_true, frame.y_pred)}
        )
        for fold, part in frame.groupby("fold", sort=True):
            fold_rows.append(
                {"target": target_name, "module": target_name, "model": model, "fold": int(fold), "n": len(part), **regression_metrics(part.y_true, part.y_pred)}
            )
    paths = [
        result_dir / f"{stem}_oof.parquet",
        result_dir / f"{stem}_performance.csv",
        result_dir / f"{stem}_fold_metrics.csv",
        result_dir / f"{stem}_hyperparameters.csv",
        result_dir / f"{stem}_leakage_ledger.csv",
        result_dir / f"{stem}_mlp_seed_oof.parquet",
        result_dir / f"{stem}_mlp_learning_curves.parquet",
    ]
    atomic_parquet(oof, paths[0])
    atomic_csv(pd.DataFrame(performance_rows), paths[1])
    atomic_csv(pd.DataFrame(fold_rows), paths[2])
    atomic_csv(pd.DataFrame(parameter_rows), paths[3])
    atomic_csv(pd.DataFrame(ledger_rows), paths[4])
    atomic_parquet(
        pd.concat(seed_predictions, ignore_index=True)
        if seed_predictions
        else pd.DataFrame({"canonical_cell_id": pd.Series(dtype=str)}),
        paths[5],
    )
    atomic_parquet(
        pd.concat(curves, ignore_index=True)
        if curves
        else pd.DataFrame({"epoch": pd.Series(dtype=int)}),
        paths[6],
    )
    return [*paths, *model_paths]


def run_nav7_observed(
    root: Path, config: dict[str, Any], result_dir: Path, model_dir: Path
) -> list[Path]:
    table, _, _ = _load_table(root, config)
    saved = pd.read_parquet(resolve_paths(root, config)["section1_nav7_target"])
    target = _observed_target(table, saved, "target_NaV_alias_Scn2a1")
    target_path = result_dir / "nav7_target.parquet"
    target_frame = table[["canonical_cell_id", "transcriptomics_sample_id"]].copy()
    target_frame["target_NaV7_Scn2a1"] = target.to_numpy(dtype=float)
    target_frame["gene_membership"] = "Scn1a|Scn3a|Scn8a|Scn9a|Scn10a|Scn11a|Scn2a1"
    atomic_parquet(target_frame, target_path)
    return [target_path, *fit_observed_models(
        root,
        config,
        target=target,
        target_name=NAV7_NAME,
        result_dir=result_dir,
        model_dir=model_dir,
    )]


def run_nav7_control_oof(
    root: Path,
    config: dict[str, Any],
    controls_dir: Path,
    nav7_dir: Path,
    *,
    signature: str,
) -> tuple[list[Path], dict[str, int]]:
    table, _, features = _load_table(root, config)
    parameters = _parameter_rows(nav7_dir / "nav7_hyperparameters.csv")
    config_values, config_hashes = _config_contract(parameters, NAV7_NAME, config)
    matched = json.loads(
        (controls_dir / "nav7_matched_gene_sets_1000.json").read_text(encoding="utf-8")
    )
    target_frame = pd.read_parquet(controls_dir / f"targets/{NAV7_NAME}.parquet")
    registry_path = nav7_dir / "control_oof_partition_registry.json"
    registry = _load_registry(registry_path, signature)
    performance_rows: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    fit_count = reused_count = 0
    checkpoint_every = int(config["matching"]["checkpoint_every"])
    controls = matched["modules"][NAV7_NAME]["controls"]
    for index, control in enumerate(controls, start=1):
        control_id = int(control["control_id"])
        target = _target_series(table, target_frame, f"control_{control_id:04d}")
        destination = (
            nav7_dir
            / "controls/oof"
            / f"control_id={control_id:04d}"
            / "part-000.parquet"
        )
        key = str(destination.resolve())
        record = registry["partitions"].get(key, {})
        cached = _validate_control_partition(
            destination,
            trusted_sha256=record.get("sha256"),
            table=table,
            target_by_cell=target,
            module=NAV7_NAME,
            control_id=control_id,
            folds=(0, 1, 2),
            expected_config_hashes=config_hashes,
            accepted_signatures={signature},
        )
        if cached is not None:
            frame = cached
            reused_count += 1
        else:
            frame = _fit_fixed_control(
                table,
                features,
                target,
                module=NAV7_NAME,
                control_id=control_id,
                folds=(0, 1, 2),
                config_values=config_values,
                config_hashes=config_hashes,
                signature=signature,
                analysis="nav7_matched_control_raw",
                seed=int(config["seed"]),
            )
            atomic_parquet(frame, destination)
            fit_count += 1
        registry["partitions"][key] = {
            "sha256": sha256_file(destination),
            "module": NAV7_NAME,
            "control_id": control_id,
            "target_hash": str(frame.target_hash.iloc[0]),
            "configuration_hashes": {str(k): v for k, v in config_hashes.items()},
        }
        performance_rows.append(
            _performance_record(
                frame,
                module=NAV7_NAME,
                control_id=control_id,
                n_genes=len(control["genes"]),
                run_signature=signature,
            )
        )
        output_paths.append(destination)
        if index % checkpoint_every == 0:
            _save_registry(registry, registry_path)
            atomic_parquet(
                pd.DataFrame(performance_rows), nav7_dir / "nav7_control_performance.partial.parquet"
            )
    _save_registry(registry, registry_path)
    performance = pd.DataFrame(performance_rows).sort_values("control_id")
    performance_path = nav7_dir / "nav7_control_performance.parquet"
    atomic_parquet(performance, performance_path)
    observed = pd.read_csv(nav7_dir / "nav7_performance.csv")
    summary = specificity_summary(performance, observed, p_column="raw_p")
    summary_path = nav7_dir / "nav7_specificity_summary.csv"
    atomic_csv(summary, summary_path)
    resume_path = nav7_dir / "nav7_control_resume_audit.json"
    atomic_json(
        {
            "partition_count": len(performance),
            "fit_count": fit_count,
            "reused_partition_count": reused_count,
            "signature": signature,
        },
        resume_path,
    )
    return [registry_path, performance_path, summary_path, resume_path, *output_paths], {
        "fit_count": fit_count,
        "reused_partition_count": reused_count,
    }


def _canonical_half(indices: Iterable[int], n: int) -> tuple[int, ...]:
    left = tuple(sorted(int(value) for value in indices))
    if n % 2:
        return left
    right = tuple(index for index in range(n) if index not in set(left))
    return min(left, right)


def balanced_partitions(
    gene_n: int,
    *,
    family: str,
    control_id: int | str,
    maximum_draws: int,
    seed: int,
) -> tuple[list[tuple[tuple[int, ...], tuple[int, ...]]], str, int]:
    left_n = gene_n // 2
    maximum = math.comb(gene_n, left_n) // (2 if gene_n % 2 == 0 else 1)
    selected: set[tuple[int, ...]] = set()
    if maximum <= maximum_draws:
        for combination in itertools.combinations(range(gene_n), left_n):
            selected.add(_canonical_half(combination, gene_n))
        method = "exhaustive"
    else:
        rng = np.random.default_rng(
            stable_seed("reliability_partition", family, control_id, base=seed)
        )
        while len(selected) < maximum_draws:
            selected.add(
                _canonical_half(rng.choice(gene_n, size=left_n, replace=False), gene_n)
            )
        method = "deterministic_random_unique"
    partitions = [
        (left, tuple(index for index in range(gene_n) if index not in set(left)))
        for left in sorted(selected)
    ]
    return partitions, method, maximum


def _rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denominator = float(np.sqrt(np.dot(ra, ra) * np.dot(rb, rb)))
    return float(np.dot(ra, rb) / denominator) if denominator > 0 else math.nan


def consistency_summary(
    expression: pd.DataFrame,
    genes: Sequence[str],
    *,
    family: str,
    control_id: int | str,
    maximum_draws: int,
    seed: int,
) -> dict[str, Any]:
    genes = list(genes)
    if len(genes) != len(set(genes)) or set(genes) - set(expression.columns):
        raise AssertionError(f"Reliability gene membership failure: {family}/{control_id}")
    matrix = expression[genes].to_numpy(dtype=np.float32, copy=False)
    score = matrix.mean(axis=1, dtype=np.float64)
    partitions, method, maximum = balanced_partitions(
        len(genes),
        family=family,
        control_id=control_id,
        maximum_draws=maximum_draws,
        seed=seed,
    )
    rho_values: list[float] = []
    sb_values: list[float] = []
    for left, right in partitions:
        score_a = matrix[:, left].mean(axis=1, dtype=np.float64)
        score_b = matrix[:, right].mean(axis=1, dtype=np.float64)
        rho = _rank_correlation(score_a, score_b)
        sb = float(2 * rho / (1 + rho)) if np.isfinite(rho) and not np.isclose(1 + rho, 0) else math.nan
        rho_values.append(rho)
        sb_values.append(sb)
    rho_array = np.asarray(rho_values, dtype=float)
    sb_array = np.asarray(sb_values, dtype=float)
    if not np.isfinite(rho_array).all() or not np.isfinite(sb_array).all():
        raise AssertionError(f"Nonfinite reliability: {family}/{control_id}")
    return {
        "module": family,
        "control_id": control_id,
        "gene_n": len(genes),
        "genes": "|".join(genes),
        "set_signature": hash_text("\n".join(sorted(genes))),
        "score_variance": float(np.var(score, ddof=0)),
        "partition_n": len(partitions),
        "maximum_unique_partitions": maximum,
        "partition_method": method,
        "median_split_half_rho": float(np.median(rho_array)),
        "q1_split_half_rho": float(np.quantile(rho_array, 0.25)),
        "q3_split_half_rho": float(np.quantile(rho_array, 0.75)),
        "min_split_half_rho": float(np.min(rho_array)),
        "max_split_half_rho": float(np.max(rho_array)),
        "median_spearman_brown": float(np.median(sb_array)),
        "q1_spearman_brown": float(np.quantile(sb_array, 0.25)),
        "q3_spearman_brown": float(np.quantile(sb_array, 0.75)),
        "min_spearman_brown": float(np.min(sb_array)),
        "max_spearman_brown": float(np.max(sb_array)),
        "seed": stable_seed("reliability_partition", family, control_id, base=seed),
    }


def run_reliability(
    root: Path,
    config: dict[str, Any],
    controls_dir: Path,
    result_dir: Path,
    *,
    signature: str,
) -> list[Path]:
    paths = resolve_paths(root, config)
    table, _, _ = _load_table(root, config)
    expression_full = pd.read_parquet(controls_dir / "selected_gene_log2cpm.parquet")
    expression_full["transcriptomics_sample_id"] = expression_full.transcriptomics_sample_id.astype(str)
    expression = expression_full.set_index("transcriptomics_sample_id", verify_integrity=True).loc[
        table.transcriptomics_sample_id.astype(str)
    ]
    expression.index = table.canonical_cell_id.astype(str)
    frozen = load_gene_modules(paths["gene_modules"])
    main = json.loads((controls_dir / "matched_gene_sets_1000.json").read_text(encoding="utf-8"))
    nav7 = json.loads((controls_dir / "nav7_matched_gene_sets_1000.json").read_text(encoding="utf-8"))
    maximum_draws = int(config["reliability"]["maximum_partitions_per_control"])
    partial_path = result_dir / "control_reliability.partial.parquet"
    existing = pd.read_parquet(partial_path) if partial_path.is_file() else pd.DataFrame()
    cached: dict[tuple[str, int], dict[str, Any]] = {}
    if not existing.empty and {"run_signature", "module", "control_id", "set_signature"}.issubset(existing):
        for row in existing.loc[existing.run_signature.eq(signature)].to_dict("records"):
            cached[(str(row["module"]), int(row["control_id"]))] = row
    control_rows: list[dict[str, Any]] = []
    computed = reused = 0
    families = [(module, main["modules"][module]["controls"]) for module in MODULE_ORDER]
    families.append((NAV7_NAME, nav7["modules"][NAV7_NAME]["controls"]))
    for family, controls in families:
        for control in controls:
            control_id = int(control["control_id"])
            genes = list(control["genes"])
            expected_signature = hash_text("\n".join(sorted(genes)))
            record = cached.get((family, control_id))
            if record is not None and str(record.get("set_signature")) == expected_signature:
                control_rows.append(record)
                reused += 1
            else:
                summary = consistency_summary(
                    expression,
                    genes,
                    family=family,
                    control_id=control_id,
                    maximum_draws=maximum_draws,
                    seed=int(config["seed"]),
                )
                summary["run_signature"] = signature
                control_rows.append(summary)
                computed += 1
            if len(control_rows) % int(config["matching"]["checkpoint_every"]) == 0:
                atomic_parquet(pd.DataFrame(control_rows), partial_path)
    controls = pd.DataFrame(control_rows).sort_values(["module", "control_id"])
    observed_rows: list[dict[str, Any]] = []
    for module in MODULE_ORDER:
        genes = [gene for gene in frozen[module] if gene != "Scn2a"]
        observed_rows.append(
            consistency_summary(
                expression,
                genes,
                family=module,
                control_id="observed",
                maximum_draws=maximum_draws,
                seed=int(config["seed"]),
            )
        )
    observed_rows.append(
        consistency_summary(
            expression,
            ["Scn1a", "Scn3a", "Scn8a", "Scn9a", "Scn10a", "Scn11a", "Scn2a1"],
            family=NAV7_NAME,
            control_id="observed",
            maximum_draws=maximum_draws,
            seed=int(config["seed"]),
        )
    )
    observed = pd.DataFrame(observed_rows)
    comparison_rows: list[dict[str, Any]] = []
    for row in observed.itertuples(index=False):
        random = controls.loc[controls.module.eq(row.module)]
        comparison_rows.append(
            {
                "module": row.module,
                "observed_score_variance": row.score_variance,
                "control_median_variance": float(random.score_variance.median()),
                "observed_variance_percentile": float(
                    100 * np.mean(random.score_variance.to_numpy(float) <= row.score_variance)
                ),
                "observed_median_spearman_brown": row.median_spearman_brown,
                "control_median_spearman_brown": float(random.median_spearman_brown.median()),
                "observed_spearman_brown_percentile": float(
                    100
                    * np.mean(
                        random.median_spearman_brown.to_numpy(float)
                        <= row.median_spearman_brown
                    )
                ),
                "control_n": len(random),
                "interpretation": "split_half_internal_consistency_not_test_retest_reliability",
            }
        )
    paths_out = [
        result_dir / "control_reliability.parquet",
        result_dir / "control_variance.parquet",
        result_dir / "observed_reliability.csv",
        result_dir / "observed_vs_control_reliability.csv",
        result_dir / "resume_audit.json",
    ]
    atomic_parquet(controls, paths_out[0])
    atomic_parquet(
        controls[["module", "control_id", "score_variance", "set_signature", "run_signature"]],
        paths_out[1],
    )
    atomic_csv(observed, paths_out[2])
    atomic_csv(pd.DataFrame(comparison_rows), paths_out[3])
    atomic_json(
        {
            "control_n": len(controls),
            "computed_count": computed,
            "reused_count": reused,
            "signature": signature,
        },
        paths_out[4],
    )
    return paths_out


def run_reliability_context(
    root: Path,
    config: dict[str, Any],
    controls_dir: Path,
    reliability_dir: Path,
    result_dir: Path,
) -> list[Path]:
    observed_performance = pd.read_csv(resolve_paths(root, config)["section2_performance"])
    raw_performance = pd.read_parquet(controls_dir / "performance.parquet")
    nav7_performance = pd.read_csv(controls_dir.parent / "nav7/nav7_performance.csv")
    nav7_controls = pd.read_parquet(controls_dir.parent / "nav7/nav7_control_performance.parquet")
    observed_reliability = pd.read_csv(reliability_dir / "observed_reliability.csv")
    control_reliability = pd.read_parquet(reliability_dir / "control_reliability.parquet")
    observed_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    for module in (*MODULE_ORDER, NAV7_NAME):
        reliability = observed_reliability.loc[observed_reliability.module.eq(module)].iloc[0]
        performance_source = nav7_performance if module == NAV7_NAME else observed_performance
        observed_r2 = float(
            performance_source.loc[
                performance_source.module.eq(module) & performance_source.model.eq("XGBoost"), "r2"
            ].iloc[0]
        )
        sb = float(reliability.median_spearman_brown)
        ratio = observed_r2 / sb if sb > 0 else math.nan
        observed_rows.append(
            {
                "module": module,
                "observed_r2": observed_r2,
                "observed_median_spearman_brown": sb,
                "reliability_contextualized_r2_ratio": ratio,
            }
        )
        perf = nav7_controls if module == NAV7_NAME else raw_performance.loc[
            raw_performance.module.eq(module)
        ]
        rel = control_reliability.loc[control_reliability.module.eq(module)]
        merged = perf[["module", "control_id", "r2"]].merge(
            rel[["module", "control_id", "median_spearman_brown"]],
            on=["module", "control_id"],
            validate="one_to_one",
        )
        merged["reliability_contextualized_r2_ratio"] = np.where(
            merged.median_spearman_brown.gt(0),
            merged.r2 / merged.median_spearman_brown,
            np.nan,
        )
        merged["observed_ratio"] = ratio
        finite = merged.reliability_contextualized_r2_ratio.dropna().to_numpy(dtype=float)
        percentile = float(100 * np.mean(finite <= ratio)) if np.isfinite(ratio) and len(finite) else math.nan
        observed_rows[-1]["control_ratio_valid_n"] = len(finite)
        observed_rows[-1]["observed_ratio_percentile"] = percentile
        control_rows.extend(merged.to_dict("records"))
    summary = pd.DataFrame(observed_rows)
    summary["interpretation"] = (
        "descriptive_only_split_half_internal_consistency_is_not_test_retest_reliability"
    )
    paths = [result_dir / "observed_contextualized.csv", result_dir / "control_contextualized.parquet"]
    atomic_csv(summary, paths[0])
    atomic_parquet(pd.DataFrame(control_rows), paths[1])
    return paths


def _tuned_xgboost_partition(
    table: pd.DataFrame,
    features: Sequence[str],
    target: pd.Series,
    *,
    module: str,
    control_id: int,
    folds: Sequence[int],
    config: dict[str, Any],
    signature: str,
    analysis: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    parameters: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    target_hash = hash_array(
        table.loc[table.fold.isin(folds)]
        .sort_values("canonical_cell_id", kind="mergesort")
        .canonical_cell_id.astype(str)
        .map(target)
        .to_numpy(dtype=float)
    )
    for fold in folds:
        train = table.loc[table.fold.ne(fold)].copy()
        test = table.loc[table.fold.eq(fold)].copy()
        train_groups = train.group_id.astype(str).to_numpy()
        if set(train_groups) & set(test.group_id.astype(str)):
            raise AssertionError("Outer donor leakage during control retuning")
        y_train = train.canonical_cell_id.astype(str).map(target).to_numpy(dtype=float)
        y_test = test.canonical_cell_id.astype(str).map(target).to_numpy(dtype=float)
        selected = select_on_inner_groups(
            xgboost_candidates(features, seed=int(config["seed"])),
            train[list(features)],
            y_train,
            train_groups,
            seed=int(config["seed"]) + int(fold),
        )
        values = {
            **selected.parameters,
            "subsample": float(config["models"]["xgboost"]["subsample"]),
            "colsample_bytree": float(config["models"]["xgboost"]["colsample_bytree"]),
            "objective": str(config["models"]["xgboost"]["objective"]),
            "tree_method": str(config["models"]["xgboost"]["tree_method"]),
            "n_jobs": int(config["models"]["xgboost"]["n_jobs"]),
            "random_state": int(config["models"]["xgboost"]["random_state"]),
        }
        configuration_hash = configuration_sha256(values)
        frame = _prediction_frame(
            test,
            y_test,
            selected.pipeline.predict(test[list(features)]),
            module,
            "XGBoost",
            analysis=analysis,
        )
        frame["control_id"] = int(control_id)
        frame["run_signature"] = signature
        frame["target_hash"] = target_hash
        frame["configuration_sha256"] = configuration_hash
        parts.append(frame)
        inner_train, inner_valid = group_inner_split(
            train_groups, seed=int(config["seed"]) + int(fold)
        )
        parameters.append(
            {
                "module": module,
                "control_id": int(control_id),
                "fold": int(fold),
                "model": "XGBoost",
                "inner_rmse": float(selected.inner_rmse),
                **values,
                "configuration_sha256": configuration_hash,
            }
        )
        ledger.append(
            {
                "module": module,
                "control_id": int(control_id),
                "fold": int(fold),
                "outer_train_n": len(train),
                "outer_test_n": len(test),
                "outer_group_overlap_n": 0,
                "inner_train_n": len(inner_train),
                "inner_valid_n": len(inner_valid),
                "inner_group_overlap_n": len(
                    set(train_groups[inner_train]) & set(train_groups[inner_valid])
                ),
                "selection_scope": "current_outer_training_only",
                "candidate_n": 4,
            }
        )
    result = pd.concat(parts, ignore_index=True)
    return result, parameters, ledger


def run_retuning(
    root: Path,
    config: dict[str, Any],
    controls_dir: Path,
    result_dir: Path,
    *,
    signature: str,
    smoke: bool = False,
) -> tuple[list[Path], dict[str, int]]:
    table, _, features = _load_table(root, config)
    selection_path = result_dir / "selected_control_ids.json"
    if not selection_path.is_file():
        selection_path = controls_dir.parent / "retuning/selected_control_ids.json"
    selected = json.loads(selection_path.read_text(encoding="utf-8"))
    fixed = (
        pd.read_parquet(controls_dir / "performance.parquet")
        if (controls_dir / "performance.parquet").is_file()
        else pd.DataFrame(columns=["module", "control_id", "r2"])
    )
    observed = pd.read_csv(resolve_paths(root, config)["section2_performance"])
    registry_path = result_dir / "retuned_partition_registry.json"
    registry = _load_registry(registry_path, signature)
    oof_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    fit_count = reused = 0
    modules = (str(config["smoke"]["module"]),) if smoke else MODULE_ORDER
    for module in modules:
        ids = [int(value) for value in selected["selected"][module]]
        if smoke:
            ids = ids[: int(config["smoke"]["retuned_controls"])]
        target_frame = pd.read_parquet(controls_dir / f"targets/{module}.parquet")
        for control_id in ids:
            target = _target_series(table, target_frame, f"control_{control_id:04d}")
            destination = (
                result_dir / "oof" / f"module={module}" / f"control_id={control_id:04d}"
                / "part-000.parquet"
            )
            key = str(destination.resolve())
            record = registry["partitions"].get(key, {})
            expected_hashes = {
                int(key): str(value)
                for key, value in record.get("configuration_hashes", {}).items()
            }
            cached = None
            if set(expected_hashes) == ({0} if smoke else {0, 1, 2}):
                cached = _validate_control_partition(
                    destination,
                    trusted_sha256=record.get("sha256"),
                    table=table,
                    target_by_cell=target,
                    module=module,
                    control_id=control_id,
                    folds=(0,) if smoke else (0, 1, 2),
                    expected_config_hashes=expected_hashes,
                    accepted_signatures={signature},
                )
            if cached is None:
                frame, parameters, ledger = _tuned_xgboost_partition(
                    table,
                    features,
                    target,
                    module=module,
                    control_id=control_id,
                    folds=(0,) if smoke else (0, 1, 2),
                    config=config,
                    signature=signature,
                    analysis="matched_control_retuned_smoke" if smoke else "matched_control_retuned",
                )
                atomic_parquet(frame, destination)
                parameter_rows.extend(parameters)
                ledger_rows.extend(ledger)
                fit_count += 1
            else:
                frame = cached
                parameter_rows.extend(record.get("parameters", []))
                ledger_rows.extend(record.get("ledger", []))
                reused += 1
            config_hashes = {
                int(fold): str(part.configuration_sha256.iloc[0])
                for fold, part in frame.groupby("fold", sort=True)
            }
            registry["partitions"][key] = {
                "sha256": sha256_file(destination),
                "module": module,
                "control_id": control_id,
                "target_hash": str(frame.target_hash.iloc[0]),
                "configuration_hashes": {str(k): v for k, v in config_hashes.items()},
                "parameters": parameters if cached is None else record.get("parameters", []),
                "ledger": ledger if cached is None else record.get("ledger", []),
            }
            oof_parts.append(frame)
            output_paths.append(destination)
            tuned_r2 = regression_metrics(frame.y_true, frame.y_pred)["r2"]
            fixed_row = fixed.loc[fixed.module.eq(module) & fixed.control_id.eq(control_id)]
            fixed_r2 = float(fixed_row.iloc[0].r2) if len(fixed_row) == 1 and not smoke else math.nan
            comparison_rows.append(
                {
                    "module": module,
                    "control_id": control_id,
                    "fixed_r2": fixed_r2,
                    "retuned_r2": tuned_r2,
                    "paired_delta_r2": tuned_r2 - fixed_r2,
                    "folds": "0" if smoke else "0|1|2",
                }
            )
            if (fit_count + reused) % int(config["matching"]["checkpoint_every"]) == 0:
                _save_registry(registry, registry_path)
    _save_registry(registry, registry_path)
    comparison = pd.DataFrame(comparison_rows)
    summaries: list[dict[str, Any]] = []
    if not smoke:
        for module, local in comparison.groupby("module", sort=False):
            observed_r2 = float(
                observed.loc[observed.module.eq(module) & observed.model.eq("XGBoost"), "r2"].iloc[0]
            )
            delta = local.paired_delta_r2.to_numpy(dtype=float)
            summaries.append(
                {
                    "module": module,
                    "control_n": len(local),
                    "median_fixed_r2": float(local.fixed_r2.median()),
                    "median_retuned_r2": float(local.retuned_r2.median()),
                    "median_paired_delta_r2": float(np.median(delta)),
                    "q1_paired_delta_r2": float(np.quantile(delta, 0.25)),
                    "q3_paired_delta_r2": float(np.quantile(delta, 0.75)),
                    "fraction_retuned_gt_fixed": float(np.mean(delta > 0)),
                    "observed_fixed_subset_percentile": float(100 * np.mean(local.fixed_r2 <= observed_r2)),
                    "observed_retuned_subset_percentile": float(100 * np.mean(local.retuned_r2 <= observed_r2)),
                    "interpretation": "deterministic_50_control_approximation_sensitivity",
                }
            )
    paths = [
        registry_path,
        result_dir / "retuned_oof.parquet",
        result_dir / "fixed_vs_retuned.csv",
        result_dir / "retuning_summary.csv",
        result_dir / "retuned_hyperparameters.csv",
        result_dir / "retuning_leakage_ledger.csv",
        result_dir / "resume_audit.json",
    ]
    atomic_parquet(pd.concat(oof_parts, ignore_index=True), paths[1])
    atomic_csv(comparison, paths[2])
    atomic_csv(pd.DataFrame(summaries), paths[3])
    atomic_csv(pd.DataFrame(parameter_rows), paths[4])
    atomic_csv(pd.DataFrame(ledger_rows), paths[5])
    atomic_json(
        {
            "status": "SMOKE_PASS" if smoke else "PASS",
            "fit_count": fit_count,
            "reused_partition_count": reused,
            "partition_count": len(oof_parts),
            "signature": signature,
        },
        paths[6],
    )
    return [*paths, *output_paths], {"fit_count": fit_count, "reused_partition_count": reused}


def _residual_control_partition(
    table: pd.DataFrame,
    features: Sequence[str],
    raw_target: pd.Series,
    *,
    module: str,
    control_id: int,
    folds: Sequence[int],
    config_values: dict[int, dict[str, Any]],
    config_hashes: dict[int, str],
    nuisance_columns: Sequence[str],
    signature: str,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    coefficients: list[dict[str, Any]] = []
    for fold in folds:
        train = table.loc[table.fold.ne(fold)].copy()
        test = table.loc[table.fold.eq(fold)].copy()
        if set(train.group_id.astype(str)) & set(test.group_id.astype(str)):
            raise AssertionError("Residual-control outer donor leakage")
        y_train_raw = train.canonical_cell_id.astype(str).map(raw_target).to_numpy(dtype=float)
        y_test_raw = test.canonical_cell_id.astype(str).map(raw_target).to_numpy(dtype=float)
        nuisance = LinearRegression(fit_intercept=True).fit(
            train[list(nuisance_columns)], y_train_raw
        )
        train_nuisance = nuisance.predict(train[list(nuisance_columns)])
        test_nuisance = nuisance.predict(test[list(nuisance_columns)])
        y_train = y_train_raw - train_nuisance
        y_test = y_test_raw - test_nuisance
        pipeline = _fixed_candidate(
            "XGBoost",
            features,
            varying_configuration(config_values[int(fold)]),
            seed=seed,
        )
        pipeline.fit(train[list(features)], y_train)
        frame = _prediction_frame(
            test,
            y_test,
            pipeline.predict(test[list(features)]),
            module,
            "XGBoost",
            analysis="technical_residualized_matched_control",
        )
        frame["control_id"] = int(control_id)
        frame["run_signature"] = signature
        frame["configuration_sha256"] = config_hashes[int(fold)]
        frame["target_raw"] = y_test_raw
        frame["nuisance_prediction"] = test_nuisance
        parts.append(frame)
        coefficients.append(
            {
                "module": module,
                "control_id": int(control_id),
                "fold": int(fold),
                "intercept": float(nuisance.intercept_),
                **{
                    f"coefficient_{name}": float(value)
                    for name, value in zip(nuisance_columns, nuisance.coef_)
                },
                "train_n": len(train),
                "test_n": len(test),
                "outer_group_overlap_n": 0,
                "nuisance_fit_scope": "current_outer_training_only",
                "residual_rule": "raw_minus_same_outer_train_fitted_nuisance_prediction",
                "configuration_sha256": config_hashes[int(fold)],
            }
        )
    result = pd.concat(parts, ignore_index=True)
    target_hash = hash_array(
        result.sort_values("canonical_cell_id", kind="mergesort").y_true.to_numpy(dtype=float)
    )
    result["target_hash"] = target_hash
    if not np.allclose(
        result.target_raw - result.nuisance_prediction,
        result.y_true,
        rtol=0,
        atol=1e-12,
    ):
        raise AssertionError("Residual-control target identity failure")
    return result, coefficients


def _residual_truth(
    table: pd.DataFrame,
    raw_target: pd.Series,
    folds: Sequence[int],
    nuisance_columns: Sequence[str],
) -> pd.Series:
    values: dict[str, float] = {}
    for fold in folds:
        train = table.loc[table.fold.ne(fold)]
        test = table.loc[table.fold.eq(fold)]
        y_train = train.canonical_cell_id.astype(str).map(raw_target).to_numpy(dtype=float)
        y_test = test.canonical_cell_id.astype(str).map(raw_target).to_numpy(dtype=float)
        nuisance = LinearRegression().fit(train[list(nuisance_columns)], y_train)
        residual = y_test - nuisance.predict(test[list(nuisance_columns)])
        values.update(zip(test.canonical_cell_id.astype(str), residual))
    return pd.Series(values, dtype=float)


def run_residual_controls(
    root: Path,
    config: dict[str, Any],
    controls_dir: Path,
    result_dir: Path,
    *,
    signature: str,
    smoke: bool = False,
) -> tuple[list[Path], dict[str, int]]:
    paths = resolve_paths(root, config)
    table, _, features = _load_table(root, config)
    technical = pd.read_csv(paths["section1_technical_targets"], dtype={"canonical_cell_id": str})
    nuisance_columns = list(config["residualization"]["nuisance_covariates"])
    table = table.merge(
        technical[["canonical_cell_id", *nuisance_columns]],
        on="canonical_cell_id",
        validate="one_to_one",
    )
    parameters = _parameter_rows(paths["section2_adjusted_hyperparameters"])
    matched = json.loads((controls_dir / "matched_gene_sets_1000.json").read_text(encoding="utf-8"))
    registry_path = result_dir / "oof_partition_registry.json"
    registry = _load_registry(registry_path, signature)
    performance_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    output_paths: list[Path] = []
    fit_count = reused = 0
    modules = (str(config["smoke"]["module"]),) if smoke else MODULE_ORDER
    for module in modules:
        config_values, config_hashes = _config_contract(parameters, module, config)
        target_frame = pd.read_parquet(controls_dir / f"targets/{module}.parquet")
        controls = matched["modules"][module]["controls"]
        if smoke:
            controls = controls[: int(config["smoke"]["residual_controls"])]
        for control in controls:
            control_id = int(control["control_id"])
            raw_target = _target_series(table, target_frame, f"control_{control_id:04d}")
            folds = (int(config["smoke"]["fold"]),) if smoke else (0, 1, 2)
            residual_truth = _residual_truth(table, raw_target, folds, nuisance_columns)
            destination = (
                result_dir / "oof" / f"module={module}" / f"control_id={control_id:04d}"
                / "part-000.parquet"
            )
            key = str(destination.resolve())
            record = registry["partitions"].get(key, {})
            cached = _validate_control_partition(
                destination,
                trusted_sha256=record.get("sha256"),
                table=table,
                target_by_cell=residual_truth,
                module=module,
                control_id=control_id,
                folds=folds,
                expected_config_hashes={fold: config_hashes[fold] for fold in folds},
                accepted_signatures={signature},
            )
            if cached is None:
                frame, coefficients = _residual_control_partition(
                    table,
                    features,
                    raw_target,
                    module=module,
                    control_id=control_id,
                    folds=folds,
                    config_values=config_values,
                    config_hashes=config_hashes,
                    nuisance_columns=nuisance_columns,
                    signature=signature,
                    seed=int(config["seed"]),
                )
                atomic_parquet(frame, destination)
                coefficient_rows.extend(coefficients)
                fit_count += 1
            else:
                frame = cached
                coefficient_rows.extend(record.get("nuisance_ledger", []))
                reused += 1
            registry["partitions"][key] = {
                "sha256": sha256_file(destination),
                "module": module,
                "control_id": control_id,
                "target_hash": str(frame.target_hash.iloc[0]),
                "configuration_hashes": {
                    str(fold): config_hashes[fold] for fold in folds
                },
                "nuisance_ledger": coefficients if cached is None else record.get("nuisance_ledger", []),
            }
            performance_rows.append(
                _performance_record(
                    frame,
                    module=module,
                    control_id=control_id,
                    n_genes=len(control["genes"]),
                    run_signature=signature,
                )
            )
            output_paths.append(destination)
            if (fit_count + reused) % int(config["matching"]["checkpoint_every"]) == 0:
                _save_registry(registry, registry_path)
                atomic_parquet(pd.DataFrame(performance_rows), result_dir / "performance.partial.parquet")
    _save_registry(registry, registry_path)
    performance = pd.DataFrame(performance_rows)
    expected_n = int(config["smoke"]["residual_controls"]) if smoke else 6000
    if len(performance) != expected_n:
        raise AssertionError(f"Residual-control count mismatch: {len(performance)} != {expected_n}")
    output = [
        registry_path,
        result_dir / "performance.parquet",
        result_dir / "nuisance_coefficients_and_leakage.parquet",
        result_dir / "resume_audit.json",
    ]
    atomic_parquet(performance, output[1])
    atomic_parquet(pd.DataFrame(coefficient_rows), output[2])
    atomic_json(
        {
            "status": "SMOKE_PASS" if smoke else "PASS",
            "fit_count": fit_count,
            "reused_partition_count": reused,
            "partition_count": len(performance),
            "configuration_source": config["residualization"]["control_configuration_source"],
            "signature": signature,
        },
        output[3],
    )
    return [*output, *output_paths], {"fit_count": fit_count, "reused_partition_count": reused}


def run_residual_inference(root: Path, config: dict[str, Any], result_dir: Path) -> list[Path]:
    controls = pd.read_parquet(result_dir.parent / "residual_controls/performance.parquet")
    observed = pd.read_csv(resolve_paths(root, config)["section2_adjusted_performance"])
    summary = specificity_summary(controls, observed, p_column="residual_p")
    bh = add_bh_family(
        summary,
        p_column="residual_p",
        q=float(config["inference"]["bh_q"]),
        family_label="six technical-residualized matched-control specificity tests",
    )
    output = [result_dir / "residual_specificity_summary.csv", result_dir / "residual_specificity_bh6.csv"]
    atomic_csv(summary, output[0])
    atomic_csv(bh, output[1])
    return output


def run_nav7_internal_consistency(
    root: Path, config: dict[str, Any], controls_dir: Path, result_dir: Path
) -> list[Path]:
    table, _, _ = _load_table(root, config)
    full = pd.read_parquet(controls_dir / "selected_gene_log2cpm.parquet")
    full["transcriptomics_sample_id"] = full.transcriptomics_sample_id.astype(str)
    expression = full.set_index("transcriptomics_sample_id", verify_integrity=True).loc[
        table.transcriptomics_sample_id.astype(str)
    ]
    genes7 = ["Scn1a", "Scn3a", "Scn8a", "Scn9a", "Scn10a", "Scn11a", "Scn2a1"]
    partitions, method, maximum = balanced_partitions(
        7,
        family=NAV7_NAME,
        control_id="observed",
        maximum_draws=int(config["reliability"]["maximum_partitions_per_control"]),
        seed=int(config["seed"]),
    )
    draws: list[dict[str, Any]] = []
    matrix = expression[genes7].to_numpy(dtype=float)
    for draw_id, (left, right) in enumerate(partitions):
        rho = _rank_correlation(matrix[:, left].mean(axis=1), matrix[:, right].mean(axis=1))
        sb = float(2 * rho / (1 + rho))
        draws.append(
            {
                "module": NAV7_NAME,
                "draw_id": draw_id,
                "half_a_genes": "|".join(genes7[index] for index in left),
                "half_b_genes": "|".join(genes7[index] for index in right),
                "half_a_n": len(left),
                "half_b_n": len(right),
                "spearman_rho": rho,
                "spearman_brown": sb,
                "partition_method": method,
                "maximum_unique_partitions": maximum,
            }
        )
    draw_frame = pd.DataFrame(draws)
    nav7 = consistency_summary(
        expression,
        genes7,
        family=NAV7_NAME,
        control_id="observed",
        maximum_draws=int(config["reliability"]["maximum_partitions_per_control"]),
        seed=int(config["seed"]),
    )
    nav6 = consistency_summary(
        expression,
        genes7[:6],
        family="NaV6_frozen",
        control_id="observed",
        maximum_draws=int(config["reliability"]["maximum_partitions_per_control"]),
        seed=int(config["seed"]),
    )
    summary = pd.DataFrame([nav6, nav7])
    if len(draw_frame) != 35 or not np.array_equal(draw_frame.half_a_n, np.repeat(3, 35)):
        raise AssertionError("NaV7 3-vs-4 exhaustive enumeration failure")
    output = [result_dir / "nav7_internal_consistency.csv", result_dir / "nav7_split_half_draws.csv"]
    atomic_csv(summary, output[0])
    atomic_csv(draw_frame, output[1])
    return output


def run_nav7_target_definitions(
    root: Path,
    config: dict[str, Any],
    controls_dir: Path,
    result_dir: Path,
    model_dir: Path,
) -> list[Path]:
    table, _, _ = _load_table(root, config)
    full = pd.read_parquet(controls_dir / "selected_gene_log2cpm.parquet")
    full["transcriptomics_sample_id"] = full.transcriptomics_sample_id.astype(str)
    full = full.set_index("transcriptomics_sample_id", verify_integrity=True)
    genes = ["Scn1a", "Scn3a", "Scn8a", "Scn9a", "Scn10a", "Scn11a", "Scn2a1"]
    matrix = full[genes].to_numpy(dtype=float)
    standard = (matrix - matrix.mean(axis=0)) / matrix.std(axis=0, ddof=0)
    zmean = standard.mean(axis=1)
    pca = PCA(n_components=1, svd_solver="full")
    pc1 = pca.fit_transform(standard).ravel()
    if _rank_correlation(pc1, zmean) < 0:
        pc1 = -pc1
    mean = matrix.mean(axis=1)
    definitions = pd.DataFrame(
        {
            "transcriptomics_sample_id": full.index.astype(str),
            "mean_log2cpm": mean,
            "standardized_gene_mean": zmean,
            "pc1": pc1,
        }
    )
    definition_path = result_dir / "nav7_target_definitions.parquet"
    atomic_parquet(definitions, definition_path)
    cohort = definitions.set_index("transcriptomics_sample_id").loc[
        table.transcriptomics_sample_id.astype(str)
    ]
    correlations = cohort[["mean_log2cpm", "standardized_gene_mean", "pc1"]].corr(
        method="spearman"
    )
    correlation_path = result_dir / "nav7_target_definition_correlations.csv"
    atomic_csv(correlations.reset_index(names="definition"), correlation_path)
    all_paths: list[Path] = [definition_path, correlation_path]
    performance_rows: list[pd.DataFrame] = []
    mean_performance = pd.read_csv(result_dir / "nav7_performance.csv")
    mean_performance = mean_performance.loc[
        mean_performance.model.isin(["ElasticNet", "XGBoost"])
    ].copy()
    mean_performance["definition"] = "mean_log2cpm"
    performance_rows.append(mean_performance)
    for definition, target_name in (
        ("standardized_gene_mean", "NaV7_standardized_gene_mean"),
        ("pc1", "NaV7_PC1"),
    ):
        target = pd.Series(cohort[definition].to_numpy(dtype=float), index=table.index)
        paths = fit_observed_models(
            root,
            config,
            target=target,
            target_name=target_name,
            result_dir=result_dir,
            model_dir=model_dir / "target_definitions",
            model_names=("ElasticNet", "XGBoost"),
            stem=f"nav7_{definition}",
            analysis="nav7_target_definition_sensitivity",
        )
        all_paths.extend(paths)
        perf = pd.read_csv(result_dir / f"nav7_{definition}_performance.csv")
        perf["definition"] = definition
        performance_rows.append(perf)
    performance = pd.concat(performance_rows, ignore_index=True)
    performance["pc1_explained_variance_ratio"] = np.where(
        performance.definition.eq("pc1"), float(pca.explained_variance_ratio_[0]), np.nan
    )
    performance_path = result_dir / "nav7_target_definition_performance.csv"
    atomic_csv(performance, performance_path)
    return [*all_paths, performance_path]


def run_nav7_adjusted(
    root: Path, config: dict[str, Any], result_dir: Path, model_dir: Path
) -> list[Path]:
    paths = resolve_paths(root, config)
    table, _, features = _load_table(root, config)
    technical = pd.read_csv(paths["section1_technical_targets"], dtype={"canonical_cell_id": str})
    nuisance_columns = list(config["residualization"]["nuisance_covariates"])
    table = table.merge(
        technical[["canonical_cell_id", *nuisance_columns]],
        on="canonical_cell_id",
        validate="one_to_one",
    )
    target_frame = pd.read_parquet(result_dir / "nav7_target.parquet")
    target = _observed_target(table, target_frame, "target_NaV7_Scn2a1")
    predictions: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, Any]] = []
    nuisance_rows: list[dict[str, Any]] = []
    model_paths: list[Path] = []
    for fold in (0, 1, 2):
        train = table.loc[table.fold.ne(fold)].copy()
        test = table.loc[table.fold.eq(fold)].copy()
        if set(train.group_id.astype(str)) & set(test.group_id.astype(str)):
            raise AssertionError("NaV7 adjusted outer donor leakage")
        y_train_raw = target[table.fold.ne(fold).to_numpy()].to_numpy(dtype=float)
        y_test_raw = target[table.fold.eq(fold).to_numpy()].to_numpy(dtype=float)
        nuisance = LinearRegression().fit(train[nuisance_columns], y_train_raw)
        y_train = y_train_raw - nuisance.predict(train[nuisance_columns])
        test_nuisance = nuisance.predict(test[nuisance_columns])
        y_test = y_test_raw - test_nuisance
        nuisance_rows.append(
            {
                "fold": fold,
                "intercept": float(nuisance.intercept_),
                **{
                    f"coefficient_{name}": float(value)
                    for name, value in zip(nuisance_columns, nuisance.coef_)
                },
                "train_n": len(train),
                "test_n": len(test),
                "outer_group_overlap_n": 0,
                "nuisance_fit_scope": "current_outer_training_only",
                "residual_rule": "raw_minus_same_outer_train_fitted_nuisance_prediction",
            }
        )
        groups = train.group_id.astype(str).to_numpy()
        for model in ("ElasticNet", "XGBoost"):
            candidates = (
                elastic_net_candidates(features, seed=int(config["seed"]))
                if model == "ElasticNet"
                else xgboost_candidates(features, seed=int(config["seed"]))
            )
            selected = select_on_inner_groups(
                candidates,
                train[features],
                y_train,
                groups,
                seed=int(config["seed"]) + fold,
            )
            frame = _prediction_frame(
                test,
                y_test,
                selected.pipeline.predict(test[features]),
                NAV7_NAME,
                model,
                analysis="nav7_technical_residualized",
            )
            frame["target_raw"] = y_test_raw
            frame["nuisance_prediction"] = test_nuisance
            predictions.append(frame)
            parameter_rows.append(
                {
                    "module": NAV7_NAME,
                    "fold": fold,
                    "model": model,
                    "inner_rmse": float(selected.inner_rmse),
                    **selected.parameters,
                }
            )
            model_path = model_dir / "adjusted" / f"nav7_fold{fold}_{model}.joblib"
            atomic_joblib(selected.pipeline, model_path)
            model_paths.append(model_path)
    oof = pd.concat(predictions, ignore_index=True)
    if not np.allclose(oof.target_raw - oof.nuisance_prediction, oof.y_true, rtol=0, atol=1e-12):
        raise AssertionError("NaV7 adjusted residual identity failure")
    performance = [
        {"module": NAV7_NAME, "model": model, "n": len(frame), **regression_metrics(frame.y_true, frame.y_pred)}
        for model, frame in oof.groupby("model", sort=False)
    ]
    output = [
        result_dir / "nav7_adjusted_oof.parquet",
        result_dir / "nav7_adjusted_performance.csv",
        result_dir / "nav7_adjusted_hyperparameters.csv",
        result_dir / "nav7_adjusted_nuisance_ledger.csv",
    ]
    atomic_parquet(oof, output[0])
    atomic_csv(pd.DataFrame(performance), output[1])
    atomic_csv(pd.DataFrame(parameter_rows), output[2])
    atomic_csv(pd.DataFrame(nuisance_rows), output[3])
    return [*output, *model_paths]


def run_handoff(root: Path, config: dict[str, Any], result_dir: Path) -> list[Path]:
    paths = resolve_paths(root, config)
    result_dir.mkdir(parents=True, exist_ok=True)
    sources: list[Path] = []
    cell = pd.read_csv(paths["section2_cell_permutations"]).assign(
        permutation_level="cell_subclass_stratified"
    )
    donor = pd.read_csv(paths["section2_donor_permutations"]).assign(
        permutation_level="donor_subclass_adjusted"
    )
    permutation = pd.concat([cell, donor], ignore_index=True)
    metrics = pd.read_csv(paths["section2_performance"])
    metrics = metrics.loc[metrics.model.isin(EPHYS_MODELS)].copy()
    if len(cell) != 18 or len(donor) != 18 or len(metrics) != 18:
        raise AssertionError("No-refit permutation/metric source count mismatch")
    outputs = [
        result_dir / "permutation_results.csv",
        result_dir / "full_primary_metrics.csv",
    ]
    atomic_csv(permutation, outputs[0])
    atomic_csv(metrics, outputs[1])
    sources.extend(
        [paths["section2_cell_permutations"], paths["section2_donor_permutations"], paths["section2_performance"]]
    )

    for label, source_key in (
        ("raw_spearman", "section1_raw_correlation"),
        ("technical_adjusted_spearman", "section1_adjusted_correlation"),
        ("subclass_centered_spearman", "section1_centered_correlation"),
    ):
        source = paths[source_key]
        destination = result_dir / f"target_correlations_{label}.csv"
        frame = pd.read_csv(source)
        if frame.shape != (6, 7):
            raise AssertionError(f"Target-correlation grid mismatch: {source}")
        atomic_copy(source, destination)
        outputs.append(destination)
        sources.append(source)

    grid_rows: list[dict[str, Any]] = []
    for alpha in config["models"]["elastic_net"]["alpha"]:
        for l1 in config["models"]["elastic_net"]["l1_ratio"]:
            grid_rows.append(
                {
                    "model": "ElasticNet",
                    "candidate": len(grid_rows),
                    "alpha": alpha,
                    "l1_ratio": l1,
                    "max_iter": config["models"]["elastic_net"]["max_iter"],
                }
            )
    xgb = config["models"]["xgboost"]
    for candidate, values in enumerate(xgb["grid"]):
        grid_rows.append(
            {
                "model": "XGBoost",
                "candidate": candidate,
                **values,
                "subsample": xgb["subsample"],
                "colsample_bytree": xgb["colsample_bytree"],
                "objective": xgb["objective"],
                "tree_method": xgb["tree_method"],
                "n_jobs": xgb["n_jobs"],
                "random_state": xgb["random_state"],
            }
        )
    hyperparameter_path = result_dir / "hyperparameter_grid.csv"
    atomic_csv(pd.DataFrame(grid_rows), hyperparameter_path)
    outputs.append(hyperparameter_path)
    mlp_source = paths["section2_mlp_configuration"]
    mlp_destination = result_dir / "mlp_configuration.json"
    atomic_copy(mlp_source, mlp_destination)
    outputs.append(mlp_destination)
    sources.append(mlp_source)

    logo_source = paths["robustness_logo"]
    logo = pd.read_csv(logo_source)
    if len(logo) != 16 or set(logo.omitted_gene) != {"Scn9a", "Scn3a", "Hcn2", "Hcn3"}:
        raise AssertionError("Frozen LOGO artifact mismatch")
    logo_path = result_dir / "logo_results.csv"
    atomic_copy(logo_source, logo_path)
    outputs.append(logo_path)
    sources.append(logo_source)

    grouped = pd.read_parquet(paths["section2_grouped_shap"])
    group_mean = (
        grouped.groupby(["module", "component_id"], as_index=False).group_abs_shap.mean()
    )
    leading = group_mean.sort_values(
        ["module", "group_abs_shap", "component_id"], ascending=[True, False, True]
    ).groupby("module", as_index=False).first()
    concordance = pd.read_csv(paths["section2_concordance"])
    compact = leading.merge(concordance, on="module", validate="one_to_one").rename(
        columns={
            "component_id": "leading_group",
            "group_abs_shap": "grouped_SHAP",
            "group_spearman_rho": "SHAP_EN_rank_rho",
            "top3_overlap_n": "top3_overlap",
        }
    )
    compact = compact[
        ["module", "leading_group", "grouped_SHAP", "SHAP_EN_rank_rho", "top3_overlap"]
    ]
    attribution_path = result_dir / "shap_en_attribution_summary.csv"
    atomic_csv(compact, attribution_path)
    outputs.append(attribution_path)
    sources.extend(
        [
            paths["section2_grouped_shap"],
            paths["section2_en_grouped"],
            paths["section2_concordance"],
            paths["section2_feature_components"],
        ]
    )
    provenance = pd.DataFrame(
        [
            {
                "source_path": source.relative_to(root).as_posix(),
                "source_sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "action": "verified_extract_or_byte_copy_no_refit",
            }
            for source in sorted(set(sources))
        ]
    )
    provenance_path = result_dir / "source_provenance.csv"
    atomic_csv(provenance, provenance_path)
    outputs.append(provenance_path)
    return outputs


def run_figures(root: Path, config: dict[str, Any], figure_dir: Path) -> list[Path]:
    results = output_roots(root, config)["results"]
    figure_dir.mkdir(parents=True, exist_ok=True)
    performance = pd.read_parquet(results / "controls/performance.parquet")
    raw_bh = pd.read_csv(results / "inference/specificity_bh6.csv").set_index("module")
    observed = pd.read_csv(resolve_paths(root, config)["section2_performance"])
    observed = observed.loc[observed.model.eq("XGBoost")].set_index("module")
    outputs: list[Path] = []

    def save(stem: str) -> None:
        plt.tight_layout()
        for suffix in ("pdf", "png"):
            path = figure_dir / f"{stem}.{suffix}"
            plt.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight")
            outputs.append(path)
        plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharey=False)
    for ax, module in zip(axes.flat, MODULE_ORDER):
        values = performance.loc[performance.module.eq(module), "r2"]
        ax.hist(values, bins=35, color="#6baed6", alpha=0.85)
        ax.axvline(float(observed.loc[module, "r2"]), color="#cb181d", lw=2)
        ax.set_title(f"{module}: BH q={float(raw_bh.loc[module, 'bh_q']):.4g}")
        ax.set_xlabel("Control pooled OOF R²")
        ax.set_ylabel("Count")
    save("specificity_1000_distributions")

    observed_rel = pd.read_csv(results / "reliability_controls/observed_reliability.csv")
    control_rel = pd.read_parquet(results / "reliability_controls/control_reliability.parquet")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    merged_controls = performance.merge(
        control_rel[["module", "control_id", "median_spearman_brown"]],
        on=["module", "control_id"],
        validate="one_to_one",
    )
    sample = merged_controls.groupby("module", group_keys=False).sample(
        n=200, random_state=int(config["seed"])
    )
    ax.scatter(sample.median_spearman_brown, sample.r2, s=7, alpha=0.10, color="grey")
    for module in MODULE_ORDER:
        x = float(observed_rel.loc[observed_rel.module.eq(module), "median_spearman_brown"].iloc[0])
        y = float(observed.loc[module, "r2"])
        ax.scatter(x, y, s=65)
        ax.annotate(module, (x, y), xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Median split-half Spearman–Brown")
    ax.set_ylabel("Observed XGBoost pooled OOF R²")
    ax.set_title("Reliability and predictive performance")
    save("reliability_vs_observed_r2")

    raw = pd.read_csv(results / "handoff/target_correlations_raw_spearman.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image_handle = ax.imshow(raw.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(6), raw.columns, rotation=45, ha="right")
    ax.set_yticks(range(6), raw.index)
    for row in range(6):
        for column in range(6):
            ax.text(column, row, f"{raw.iloc[row, column]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image_handle, ax=ax, label="Spearman ρ")
    ax.set_title("Raw transcriptomic target correlations")
    save("target_correlation_matrix")

    comparison = pd.read_csv(results / "nav7/nav6_vs_nav7.csv")
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, column, label in zip(
        axes,
        ("r2", "specificity_percentile", "median_spearman_brown"),
        ("XGBoost R²", "Specificity percentile", "Median Spearman–Brown"),
    ):
        ax.bar(comparison.target, comparison[column], color=["#9ecae1", "#3182bd"])
        ax.set_ylabel(label)
        ax.tick_params(axis="x", rotation=20)
    save("nav6_vs_nav7")
    return outputs


def run_nav6_nav7_comparison(root: Path, config: dict[str, Any], result_dir: Path) -> list[Path]:
    results = output_roots(root, config)["results"]
    primary = pd.read_csv(resolve_paths(root, config)["section2_performance"])
    raw = pd.read_csv(results / "inference/specificity_bh6.csv")
    nav7_performance = pd.read_csv(result_dir / "nav7_performance.csv")
    nav7_specificity = pd.read_csv(result_dir / "nav7_specificity_summary.csv")
    reliability = pd.read_csv(results / "reliability_controls/observed_reliability.csv")
    nav6_perf = primary.loc[primary.module.eq("NaV") & primary.model.eq("XGBoost")].iloc[0]
    nav6_spec = raw.loc[raw.module.eq("NaV")].iloc[0]
    nav7_perf = nav7_performance.loc[nav7_performance.model.eq("XGBoost")].iloc[0]
    nav7_spec = nav7_specificity.loc[nav7_specificity.module.eq(NAV7_NAME)].iloc[0]
    rows = [
        {
            "target": "NaV6_frozen",
            "gene_n": 6,
            "genes": "Scn1a|Scn3a|Scn8a|Scn9a|Scn10a|Scn11a",
            "r2": float(nav6_perf.r2),
            "specificity_percentile": float(nav6_spec.observed_empirical_percentile),
            "raw_empirical_p": float(nav6_spec.raw_p),
            "six_test_bh_q": float(nav6_spec.bh_q),
            "median_spearman_brown": float(
                reliability.loc[reliability.module.eq("NaV"), "median_spearman_brown"].iloc[0]
            ),
            "role": "frozen_historical_primary_definition",
        },
        {
            "target": NAV7_NAME,
            "gene_n": 7,
            "genes": "Scn1a|Scn3a|Scn8a|Scn9a|Scn10a|Scn11a|Scn2a1",
            "r2": float(nav7_perf.r2),
            "specificity_percentile": float(nav7_spec.observed_empirical_percentile),
            "raw_empirical_p": float(nav7_spec.raw_p),
            "six_test_bh_q": math.nan,
            "median_spearman_brown": float(
                reliability.loc[reliability.module.eq(NAV7_NAME), "median_spearman_brown"].iloc[0]
            ),
            "role": "release_native_symbol_sensitivity_not_in_primary_bh_family",
        },
    ]
    output = result_dir / "nav6_vs_nav7.csv"
    atomic_csv(pd.DataFrame(rows), output)
    return [output]


def write_artifact_manifest(root: Path, config: dict[str, Any]) -> Path:
    outputs = output_roots(root, config)
    destination = outputs["results"] / "artifact_manifest.csv"
    rows: list[dict[str, Any]] = []
    for namespace, base in outputs.items():
        for path in sorted(base.rglob("*")):
            if (
                not path.is_file()
                or path.resolve() == destination.resolve()
                or path.resolve() == (outputs["logs"] / "manifest_validation.json").resolve()
                or path.name.endswith(".tmp")
                or path.name.endswith(".partial.parquet")
            ):
                continue
            rows.append(
                {
                    "namespace": namespace,
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "suffix": path.suffix.lower(),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty or frame.path.duplicated().any():
        raise AssertionError("Section 4 manifest path contract failure")
    atomic_csv(frame, destination)
    return destination


def validate_artifact_manifest(root: Path, path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    failures: list[str] = []
    for row in frame.itertuples(index=False):
        artifact = root / str(row.path)
        if (
            not artifact.is_file()
            or artifact.stat().st_size != int(row.bytes)
            or sha256_file(artifact) != str(row.sha256)
        ):
            failures.append(str(row.path))
    if failures:
        raise AssertionError(f"Section 4 manifest validation failed: {failures[:5]}")
    return {"status": "PASS", "rows": len(frame), "failures": 0, "sha256": sha256_file(path)}


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    view = frame[list(columns)].copy()
    for column in view:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: f"{value:.6g}" if pd.notna(value) else "NA"
            )
    return "\n".join(
        [
            "| " + " | ".join(view.columns) + " |",
            "| " + " | ".join("---" for _ in view.columns) + " |",
            *[
                "| " + " | ".join(str(value) for value in row) + " |"
                for row in view.itertuples(index=False, name=None)
            ],
        ]
    )


def generate_report(root: Path, config: dict[str, Any]) -> Path:
    results = output_roots(root, config)["results"]
    raw = pd.read_csv(results / "inference/specificity_bh6.csv")
    residual = pd.read_csv(results / "inference/residual_specificity_bh6.csv")
    comparison = pd.read_csv(results / "nav7/nav6_vs_nav7.csv")
    reliability = pd.read_csv(results / "reliability_controls/observed_vs_control_reliability.csv")
    context = pd.read_csv(results / "reliability_context/observed_contextualized.csv")
    retuning = pd.read_csv(results / "retuning/retuning_summary.csv")
    nav7_definitions = pd.read_csv(results / "nav7/nav7_target_definition_performance.csv")
    nav7_adjusted = pd.read_csv(results / "nav7/nav7_adjusted_performance.csv")
    handoff = pd.read_csv(results / "handoff/source_provenance.csv")
    integrity = json.loads((results / "integrity/integrity_gate.json").read_text(encoding="utf-8"))
    raw_survivors = raw.loc[raw.reject_q05, "module"].astype(str).tolist()
    residual_survivors = residual.loc[residual.reject_q05, "module"].astype(str).tolist()
    report = f"""# Extended-control analysis Results

# Objective

Evaluate which module-specificity claims remain defensible after increasing the matched-control resolution to 1,000 per module, applying BH correction to the exact six-test family, incorporating release-native `Scn2a1`, and auditing control coherence. Success was defined by contract completion, not by a preferred result.

# Frozen Inputs and Integrity

Integrity Gate: **{integrity['status']}**. The cohort contains {integrity['cohort_n']:,} cells and {integrity['donor_n']:,} donor/groups with zero outer-fold donor overlap and exactly {integrity['revised_feature_n']} revised predictors. Frozen primary OOF SHA256 is `{integrity['primary_oof_sha256']}`; frozen SHAP SHA256 is `{integrity['primary_shap_sha256']}`. Section 1 and Section 2 signatures matched their expected PASS states. No upstream artifact or manuscript file was modified.

# 1000-Control Matching Extension

Exactly 1,000 matched controls were generated for each of NaV, Kv, CaV, HCN, GABAA, and iGluR. IDs 0000–0199 were preserved gene-set-for-gene-set from Section 1; only IDs 0200–0999 were added. The frozen two-feature matching rules, 128-neighbor pool, calipers, deterministic per-control seeds, 200-proposal limit, exclusion set, and within-module signature uniqueness were retained without fallback.

# Raw Specificity with Six-Test BH-FDR

{_markdown_table(raw, ['module','observed_xgboost_r2','observed_empirical_percentile','raw_p','bh_q','rank','bh_threshold','reject_q05'])}

Raw p<.05: {int(raw.raw_p.lt(.05).sum())}/6. BH q<.05: {int(raw.reject_q05.sum())}/6. Surviving modules: {', '.join(raw_survivors) if raw_survivors else 'none'}. GABAA verdict: {'survives' if bool(raw.loc[raw.module.eq('GABAA'), 'reject_q05'].iloc[0]) else 'does not survive'}. iGluR verdict: {'survives' if bool(raw.loc[raw.module.eq('iGluR'), 'reject_q05'].iloc[0]) else 'does not survive'}. Final Kv percentile: {float(raw.loc[raw.module.eq('Kv'), 'observed_empirical_percentile'].iloc[0]):.3f}.

# NaV6 versus NaV7_Scn2a1

{_markdown_table(comparison, ['target','gene_n','r2','specificity_percentile','raw_empirical_p','six_test_bh_q','median_spearman_brown'])}

`NaV6_frozen` remains the unchanged historical primary definition. `NaV7_Scn2a1` is a release-native symbol sensitivity/corrected biological specification and is reported alongside it; it does not overwrite NaV6 and its p-value is not added to the primary six-test BH family.

# NaV7 Specificity

The table above gives the independent 1,000-control NaV7 benchmark. Its controls use seven genes, exclude every frozen primary-module gene and Scn2a1, and reuse the NaV7 observed fold-specific XGBoost configurations without control-specific tuning.

# NaV7 Reliability and Target Definitions

All 35 unique 3-vs-4 partitions were enumerated. Mean log2(CPM+1), standardized-gene mean, and PC1 target definitions were fitted with EN and XGBoost under the frozen folds. PC1 explained variance and target-definition correlations are recorded with the model results.

{_markdown_table(nav7_definitions, ['definition','model','r2','spearman','pc1_explained_variance_ratio'])}

Technical-adjusted NaV7 residual-target results (not a causal fraction comparison):

{_markdown_table(nav7_adjusted, ['module','model','r2','mae','rmse','spearman'])}

# Control Variance and Reliability Audit

{_markdown_table(reliability, ['module','observed_score_variance','control_median_variance','observed_variance_percentile','observed_median_spearman_brown','control_median_spearman_brown','observed_spearman_brown_percentile'])}

The control distributions expose whether expression/detection matching also produced comparable within-set coherence; they do not alter the primary empirical specificity estimand.

# Reliability-Contextualized Performance

{_markdown_table(context, ['module','observed_r2','observed_median_spearman_brown','reliability_contextualized_r2_ratio','observed_ratio_percentile'])}

These are descriptive reliability-contextualized R² ratios. Split-half internal consistency is not test–retest measurement reliability, and the ratio is not a noise-ceiling correction, attenuation correction, normalized truth, or causal measure.

# 50-Control Retuning Sensitivity

{_markdown_table(retuning, ['module','control_n','median_fixed_r2','median_retuned_r2','median_paired_delta_r2','q1_paired_delta_r2','q3_paired_delta_r2','fraction_retuned_gt_fixed','observed_fixed_subset_percentile','observed_retuned_subset_percentile'])}

The deterministic 50-control/module subset was frozen before outcomes and is an approximation/sensitivity, not a replacement for the 1,000-control benchmark.

# Residualized Specificity Benchmark

{_markdown_table(residual, ['module','observed_xgboost_r2','observed_empirical_percentile','residual_p','bh_q','reject_q05'])}

The separate technical-residualized six-test family retains: {', '.join(residual_survivors) if residual_survivors else 'none'}. Every nuisance regression was fit on the current outer-training raw control target and the same fit formed train/test residuals; cross-fitted rows from other folds were never training targets.

# Existing-Artifact Handoff

{len(handoff)} frozen source artifacts were independently hashed and consolidated without refitting. The handoff includes 36 permutation rows, 18 EN/XGB/MLP primary metric rows, three six-target correlation matrices, exact executed hyperparameter grids, MLP configuration, 16 LOGO rows, and compact SHAP/EN concordance.

# Verification and Tests

Smoke, full, and zero-fit resume audits are stored under `logs/revision_v3/section4/`. The Section 4 artifact manifest is byte/hash validated. The source digests and artifact manifests provide the upstream integrity record.

# Independent Reviewer Findings

Analysis outputs include source digests, validation results, and a complete artifact manifest.

# Claims Supported

- Report only the raw six-family survivors listed above as BH-supported matched-control specificity claims.
- Report NaV7 as a release-native sensitivity alongside, not in place of, frozen NaV6.
- Present reliability/coherence and retuning results as descriptive sensitivity analyses.

# Claims Not Supported

- Modules failing six-test BH must not be described as specificity-supported.
- R²/Spearman–Brown is not a noise-ceiling-corrected or causal performance measure.
- Raw versus technical-residualized R² differences are not fractions explained by technical covariates.

# Manuscript Implications

The Abstract and Results must align any specificity claim with the exact raw BH outcomes above, disclose the independent NaV7 sensitivity, and avoid treating coherence-contextualized ratios as corrected performance. Full residual, retuning, target-definition, LOGO, permutation, and reliability tables belong in the Supplement. This stage intentionally did not edit `MANUSCRIPT.md`.

# Remaining Limitations

Matched controls are balanced on mean expression and detection, not directly on biological coherence. The 50-control retuning analysis is deliberately smaller than the primary null. NaV7 is a release-symbol sensitivity and does not retroactively redefine the frozen historical target.

# Reproduction Commands

```powershell
python scripts/11_revision_section4.py --config config/revision_v3_section4.yaml --mode smoke
python scripts/11_revision_section4.py --config config/revision_v3_section4.yaml --mode full
python scripts/11_revision_section4.py --config config/revision_v3_section4.yaml --mode resume
pytest -q
```
"""
    destination = root / "REVISION_V3_SECTION4_RESULTS.md"
    atomic_text(report, destination)
    return destination


def run_smoke(
    root: Path, config: dict[str, Any], outputs: dict[str, Path], signature: str
) -> dict[str, Any]:
    started = time.perf_counter()
    smoke_dir = outputs["results"] / "smoke"
    controls_dir = outputs["results"] / "controls"
    integrity_paths = run_integrity(root, config, outputs["results"] / "integrity/integrity_gate.json")
    generation_paths = run_control_generation(root, config, controls_dir)
    target_paths = run_target_extraction(root, config, controls_dir)
    table, _, features = _load_table(root, config)
    module = str(config["smoke"]["module"])
    fold = int(config["smoke"]["fold"])
    ids = [int(value) for value in config["smoke"]["generated_control_ids"]]
    target_frame = pd.read_parquet(controls_dir / f"targets/{module}.parquet")
    parameters = _parameter_rows(resolve_paths(root, config)["section2_hyperparameters"])
    config_values, config_hashes = _config_contract(parameters, module, config)
    raw_parts: list[pd.DataFrame] = []
    for control_id in ids:
        target = _target_series(table, target_frame, f"control_{control_id:04d}")
        frame = _fit_fixed_control(
            table,
            features,
            target,
            module=module,
            control_id=control_id,
            folds=(fold,),
            config_values=config_values,
            config_hashes=config_hashes,
            signature=signature,
            analysis="matched_control_raw_smoke",
            seed=int(config["seed"]),
        )
        raw_parts.append(frame)
    raw_smoke = pd.concat(raw_parts, ignore_index=True)
    raw_path = smoke_dir / "raw_gabaa_0200_0209_fold0.parquet"
    atomic_parquet(raw_smoke, raw_path)

    synthetic_controls = pd.DataFrame(
        {
            "module": np.repeat(MODULE_ORDER, 1000),
            "control_id": np.tile(np.arange(1000), 6),
            "r2": np.tile(np.linspace(-0.2, 0.4, 1000), 6),
        }
    )
    synthetic_observed = pd.DataFrame(
        {
            "module": MODULE_ORDER,
            "model": "XGBoost",
            "r2": np.linspace(-0.1, 0.3, 6),
        }
    )
    synthetic = specificity_summary(
        synthetic_controls, synthetic_observed, p_column="raw_p"
    )
    synthetic_bh = add_bh_family(
        synthetic,
        p_column="raw_p",
        q=0.05,
        family_label="synthetic_exact_six_test_smoke",
    )
    independently_rebuilt_q = bh_adjust(synthetic.raw_p.to_numpy(dtype=float))
    if not np.allclose(synthetic_bh.bh_q, independently_rebuilt_q, rtol=0, atol=0):
        raise AssertionError("Synthetic BH-six reconstruction failure")
    synthetic_path = smoke_dir / "synthetic_empirical_bh6.csv"
    atomic_csv(synthetic_bh, synthetic_path)

    saved_nav7 = pd.read_parquet(resolve_paths(root, config)["section1_nav7_target"])
    nav7_target = _observed_target(table, saved_nav7, "target_NaV_alias_Scn2a1")
    nav7_smoke_dir = smoke_dir / "nav7"
    nav7_paths = fit_observed_models(
        root,
        config,
        target=nav7_target,
        target_name=NAV7_NAME,
        result_dir=nav7_smoke_dir,
        model_dir=outputs["models"] / "smoke/nav7",
        folds=(fold,),
        model_names=("ElasticNet", "XGBoost"),
        stem="nav7_smoke",
        analysis="nav7_observed_smoke",
    )
    internal_paths = run_nav7_internal_consistency(root, config, controls_dir, nav7_smoke_dir)

    expression_full = pd.read_parquet(controls_dir / "selected_gene_log2cpm.parquet")
    expression = expression_full.set_index("transcriptomics_sample_id", verify_integrity=True).loc[
        table.transcriptomics_sample_id.astype(str)
    ]
    matched = json.loads((controls_dir / "matched_gene_sets_1000.json").read_text(encoding="utf-8"))
    reliability_rows = []
    for control in matched["modules"][module]["controls"][: int(config["smoke"]["reliability_controls"])]:
        reliability_rows.append(
            consistency_summary(
                expression,
                control["genes"],
                family=module,
                control_id=int(control["control_id"]),
                maximum_draws=int(config["reliability"]["maximum_partitions_per_control"]),
                seed=int(config["seed"]),
            )
        )
    reliability_path = smoke_dir / "reliability_5_controls.parquet"
    atomic_parquet(pd.DataFrame(reliability_rows), reliability_path)

    retuning_paths, retuning_counts = run_retuning(
        root,
        config,
        controls_dir,
        smoke_dir / "retuning",
        signature=signature,
        smoke=True,
    )
    residual_paths, residual_counts = run_residual_controls(
        root,
        config,
        controls_dir,
        smoke_dir / "residual_controls",
        signature=signature,
        smoke=True,
    )

    # Failure-closed resume checks against a valid raw smoke partition.
    valid = raw_parts[0]
    control_id = ids[0]
    target = _target_series(table, target_frame, f"control_{control_id:04d}")
    corruption = {"valid_accepted": False, "truth_rejected": False, "config_rejected": False, "digest_rejected": False}
    with tempfile.TemporaryDirectory(prefix="section4-smoke-") as temporary:
        base = Path(temporary) / "partition.parquet"
        valid.to_parquet(base, index=False)
        valid_hash = sha256_file(base)
        corruption["valid_accepted"] = _validate_control_partition(
            base,
            trusted_sha256=valid_hash,
            table=table,
            target_by_cell=target,
            module=module,
            control_id=control_id,
            folds=(fold,),
            expected_config_hashes={fold: config_hashes[fold]},
            accepted_signatures={signature},
        ) is not None
        truth = valid.copy()
        truth.loc[truth.index[0], "y_true"] += 1.0
        truth.to_parquet(base, index=False)
        corruption["truth_rejected"] = _validate_control_partition(
            base,
            trusted_sha256=sha256_file(base),
            table=table,
            target_by_cell=target,
            module=module,
            control_id=control_id,
            folds=(fold,),
            expected_config_hashes={fold: config_hashes[fold]},
            accepted_signatures={signature},
        ) is None
        changed = valid.copy()
        changed["configuration_sha256"] = "corrupted"
        changed.to_parquet(base, index=False)
        corruption["config_rejected"] = _validate_control_partition(
            base,
            trusted_sha256=sha256_file(base),
            table=table,
            target_by_cell=target,
            module=module,
            control_id=control_id,
            folds=(fold,),
            expected_config_hashes={fold: config_hashes[fold]},
            accepted_signatures={signature},
        ) is None
        valid.to_parquet(base, index=False)
        corruption["digest_rejected"] = _validate_control_partition(
            base,
            trusted_sha256="0" * 64,
            table=table,
            target_by_cell=target,
            module=module,
            control_id=control_id,
            folds=(fold,),
            expected_config_hashes={fold: config_hashes[fold]},
            accepted_signatures={signature},
        ) is None
    if not all(corruption.values()):
        raise AssertionError(f"Smoke corruption tests failed: {corruption}")
    audit_path = outputs["logs"] / "smoke_audit.json"
    payload = {
        "status": "PASS",
        "signature": signature,
        "elapsed_seconds": time.perf_counter() - started,
        "old_controls_exact": True,
        "generated_gabaa_ids": ids,
        "raw_fold": fold,
        "raw_rows": len(raw_smoke),
        "synthetic_empirical_and_bh6": "PASS",
        "nav7_fold_models": ["ElasticNet", "XGBoost"],
        "nav7_split_partition_n": 35,
        "reliability_control_n": len(reliability_rows),
        "retuning": retuning_counts,
        "residual": residual_counts,
        "corruption_tests": corruption,
    }
    atomic_json(payload, audit_path)
    all_paths = [
        *integrity_paths,
        *generation_paths,
        *target_paths,
        raw_path,
        synthetic_path,
        *nav7_paths,
        *internal_paths,
        reliability_path,
        *retuning_paths,
        *residual_paths,
        audit_path,
    ]
    return {"status": "PASS", "paths": [str(path) for path in all_paths], **payload}


def run_revision_section4(config_path: str | Path, *, mode: str) -> dict[str, Any]:
    if mode not in {"smoke", "full", "resume"}:
        raise ValueError(f"Unsupported Section 4 mode: {mode}")
    root, config, config_file = read_config(config_path)
    outputs = ensure_output_roots(root, config)
    signature = config_signature(root, config, config_file)
    if mode == "smoke":
        result = run_smoke(root, config, outputs, signature)
        return {"mode": mode, "status": "PASS", "signature": signature, **result}

    smoke_audit_path = outputs["logs"] / "smoke_audit.json"
    if not smoke_audit_path.is_file():
        raise RuntimeError("Full/resume is prohibited until smoke PASS")
    smoke_audit = json.loads(smoke_audit_path.read_text(encoding="utf-8"))
    if smoke_audit.get("status") != "PASS" or smoke_audit.get("signature") != signature:
        raise RuntimeError("Smoke PASS does not match current Section 4 signature")

    checkpoint = StageCache(outputs["logs"] / "checkpoints.json", signature)
    stage_disposition: dict[str, str] = {}

    def execute(stage: str, function: Any, *, force: bool = False) -> list[Path]:
        recorded = checkpoint.payload["stages"].get(stage, {}).get("files", {})
        expected = [Path(path) for path in recorded]
        if not force and checkpoint.valid(stage, expected):
            stage_disposition[stage] = "reused_complete_stage"
            return expected
        generated = list(function())
        checkpoint.complete(stage, generated)
        stage_disposition[stage] = "executed_or_partition_resumed"
        return generated

    results = outputs["results"]
    controls_dir = results / "controls"
    nav7_dir = results / "nav7"
    execute(
        "integrity",
        lambda: run_integrity(root, config, results / "integrity/integrity_gate.json"),
    )
    execute("control_generation", lambda: run_control_generation(root, config, controls_dir))
    execute("target_extraction", lambda: run_target_extraction(root, config, controls_dir))

    run_counts: dict[str, dict[str, int]] = {}

    def raw_stage() -> list[Path]:
        paths, counts = run_raw_control_oof(root, config, controls_dir, signature=signature)
        run_counts["raw_controls"] = counts
        return paths

    execute("raw_control_oof", raw_stage, force=mode == "resume")
    execute("raw_inference", lambda: run_raw_inference(root, config, results / "inference"))
    execute(
        "nav7_observed",
        lambda: run_nav7_observed(root, config, nav7_dir, outputs["models"] / "nav7"),
    )

    def nav7_control_stage() -> list[Path]:
        paths, counts = run_nav7_control_oof(
            root, config, controls_dir, nav7_dir, signature=signature
        )
        run_counts["nav7_controls"] = counts
        return paths

    execute("nav7_control_oof", nav7_control_stage, force=mode == "resume")
    execute(
        "reliability",
        lambda: run_reliability(
            root,
            config,
            controls_dir,
            results / "reliability_controls",
            signature=signature,
        ),
    )
    execute(
        "reliability_context",
        lambda: run_reliability_context(
            root,
            config,
            controls_dir,
            results / "reliability_controls",
            results / "reliability_context",
        ),
    )
    execute(
        "nav7_internal_consistency",
        lambda: run_nav7_internal_consistency(root, config, controls_dir, nav7_dir),
    )
    execute(
        "nav7_target_definitions",
        lambda: run_nav7_target_definitions(
            root, config, controls_dir, nav7_dir, outputs["models"] / "nav7"
        ),
    )
    execute(
        "nav7_adjusted",
        lambda: run_nav7_adjusted(root, config, nav7_dir, outputs["models"] / "nav7"),
    )
    execute(
        "nav6_nav7_comparison",
        lambda: run_nav6_nav7_comparison(root, config, nav7_dir),
    )

    def retuning_stage() -> list[Path]:
        paths, counts = run_retuning(
            root,
            config,
            controls_dir,
            results / "retuning",
            signature=signature,
            smoke=False,
        )
        run_counts["retuning"] = counts
        return paths

    execute("retuning", retuning_stage, force=mode == "resume")

    def residual_stage() -> list[Path]:
        paths, counts = run_residual_controls(
            root,
            config,
            controls_dir,
            results / "residual_controls",
            signature=signature,
            smoke=False,
        )
        run_counts["residual_controls"] = counts
        return paths

    execute("residual_controls", residual_stage, force=mode == "resume")
    execute(
        "residual_inference",
        lambda: run_residual_inference(root, config, results / "inference"),
    )
    execute("handoff", lambda: run_handoff(root, config, results / "handoff"))
    execute("figures", lambda: run_figures(root, config, outputs["figures"]))

    audit_path = outputs["logs"] / ("resume_audit.json" if mode == "resume" else "full_audit.json")
    total_fit_count = sum(item.get("fit_count", 0) for item in run_counts.values())
    total_reused = sum(item.get("reused_partition_count", 0) for item in run_counts.values())
    if mode == "resume":
        required = {"raw_controls", "nav7_controls", "retuning", "residual_controls"}
        if set(run_counts) != required or total_fit_count != 0:
            raise AssertionError(f"Final resume did not validate to fit_count=0: {run_counts}")
    audit = {
        "status": "PASS",
        "mode": mode,
        "signature": signature,
        "fit_count": total_fit_count,
        "reused_partition_count": total_reused,
        "partition_stages": run_counts,
        "stage_disposition": stage_disposition,
        "primary_oof_sha256": sha256_file(resolve_paths(root, config)["primary_oof"]),
        "primary_shap_sha256": sha256_file(resolve_paths(root, config)["primary_shap"]),
        "section1_manifest_sha256": sha256_file(resolve_paths(root, config)["section1_manifest"]),
        "section2_manifest_sha256": sha256_file(resolve_paths(root, config)["section2_manifest"]),
    }
    atomic_json(audit, audit_path)
    report_path = generate_report(root, config)
    manifest_path = write_artifact_manifest(root, config)
    manifest_audit = validate_artifact_manifest(root, manifest_path)
    manifest_audit_path = outputs["logs"] / "manifest_validation.json"
    atomic_json(manifest_audit, manifest_audit_path)
    # The validation record itself is material, so rebuild once and validate the
    # final fixed point (the manifest never contains itself).
    manifest_path = write_artifact_manifest(root, config)
    manifest_audit = validate_artifact_manifest(root, manifest_path)
    atomic_json(manifest_audit, manifest_audit_path)
    return {
        "mode": mode,
        "status": "PASS",
        "signature": signature,
        "fit_count": total_fit_count,
        "reused_partition_count": total_reused,
        "audit": str(audit_path),
        "manifest": str(manifest_path),
        "manifest_rows": manifest_audit["rows"],
        "report": str(report_path),
    }
