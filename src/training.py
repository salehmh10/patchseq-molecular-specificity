"""Grouped outer-CV training, mandatory baselines, and held-out SHAP."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.exceptions import ConvergenceWarning

from .evaluation import assert_exact_oof_once, fold_metrics, performance_table
from .interpretability import heldout_tree_shap, shap_stability
from .models import (
    dummy_model,
    elastic_net_candidates,
    ephys_subclass_ridge_model,
    fit_mlp_on_inner_groups,
    select_on_inner_groups,
    subclass_ridge_model,
    xgboost_candidates,
)
from .runtime import RuntimeGovernor


MODULES = ("NaV", "Kv", "CaV", "HCN", "GABAA", "iGluR")
MODEL_ANALYSES = (
    ("ElasticNet", "ephys_only"),
    ("XGBoost", "ephys_only"),
    ("MLP", "ephys_only"),
    ("Dummy", "dummy"),
    ("SubclassRidge", "subclass_only"),
    ("EphysSubclassRidge", "ephys_plus_subclass"),
)
TRAINING_CACHE_VERSION = "grouped-oof-v2"


def expected_oof_keys(modules: tuple[str, ...]) -> set[tuple[str, str, str]]:
    return {(module, model, analysis) for module in modules for model, analysis in MODEL_ANALYSES}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _training_signature(root: Path, *, level: int, seed: int) -> str:
    payload = {
        "cache_version": TRAINING_CACHE_VERSION,
        "level": level,
        "seed": seed,
        "modules": MODULES,
        "python": sys.version,
        "packages": {
            package: version(package)
            for package in ("numpy", "pandas", "scikit-learn", "xgboost", "shap")
        },
        "inputs": {
            relative: _sha256(root / relative)
            for relative in (
                "data/processed/modeling_table.parquet",
                "results/tables/cv_folds.csv",
                "config/ephys_features.yaml",
            )
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_feature_names(path: str | Path) -> list[str]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload.get("feature_set"), dict) and "features" in payload["feature_set"]:
        raw = payload["feature_set"]["features"]
    else:
        raw = payload.get("features", payload.get("primary_features", payload))
    if isinstance(raw, dict):
        raw = [dict({"name": name}, **(meta if isinstance(meta, dict) else {})) for name, meta in raw.items()]
    names: list[str] = []
    for item in raw:
        if isinstance(item, str):
            names.append(item)
        elif item.get("include_primary", item.get("selected", True)):
            names.append(str(item["name"]))
    if not 15 <= len(names) <= 24:
        raise ValueError(f"Expected 15–24 frozen primary ephys features, found {len(names)}")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate frozen ephys feature")
    return names


def load_and_validate_inputs(
    modeling_path: str | Path,
    folds_path: str | Path,
    features_path: str | Path,
) -> tuple[pd.DataFrame, list[str]]:
    table = pd.read_parquet(modeling_path)
    folds = pd.read_csv(folds_path, dtype={"canonical_cell_id": str})
    fold_required = {"canonical_cell_id", "group_id", "subclass", "fold"}
    missing_fold_columns = fold_required.difference(folds.columns)
    if missing_fold_columns:
        raise ValueError(f"Fold table lacks columns: {sorted(missing_fold_columns)}")
    table["canonical_cell_id"] = table["canonical_cell_id"].astype(str)
    folds["canonical_cell_id"] = folds["canonical_cell_id"].astype(str)
    if table.canonical_cell_id.duplicated().any() or folds.canonical_cell_id.duplicated().any():
        raise AssertionError("Cell IDs must be unique in modeling table and fold table")
    if set(table.canonical_cell_id) != set(folds.canonical_cell_id):
        raise AssertionError("Modeling and frozen-fold cell ID sets differ")
    if table[["canonical_cell_id", "group_id", "subclass"]].isna().any().any():
        raise AssertionError("Cell ID, donor/group ID, and subclass must be present for every cell")
    if folds[list(fold_required)].isna().any().any():
        raise AssertionError("Frozen fold identity fields must be complete")
    if "fold" in table:
        table = table.drop(columns="fold")
    table = table.merge(
        folds[["canonical_cell_id", "group_id", "subclass", "fold"]],
        on="canonical_cell_id",
        how="left",
        validate="one_to_one",
        suffixes=("", "_frozen"),
    )
    if not np.array_equal(
        table.group_id.astype(str).to_numpy(), table.group_id_frozen.astype(str).to_numpy()
    ):
        raise AssertionError("Frozen-fold group IDs disagree with the modeling table")
    if not np.array_equal(
        table.subclass.astype(str).to_numpy(), table.subclass_frozen.astype(str).to_numpy()
    ):
        raise AssertionError("Frozen-fold subclasses disagree with the modeling table")
    table = table.drop(columns=["group_id_frozen", "subclass_frozen"])
    numeric_folds = pd.to_numeric(table.fold, errors="coerce")
    if (
        numeric_folds.isna().any()
        or not np.equal(numeric_folds, np.floor(numeric_folds)).all()
        or set(numeric_folds.astype(int)) != {0, 1, 2}
    ):
        raise AssertionError("Every cell must have exactly one of three frozen folds")
    table["fold"] = numeric_folds.astype(int)
    features = load_feature_names(features_path)
    required = {"canonical_cell_id", "group_id", "subclass", *features}
    required.update(f"target_{module}" for module in MODULES)
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Modeling table lacks columns: {sorted(missing)}")
    target_values = table[[f"target_{module}" for module in MODULES]].to_numpy(dtype=float)
    if not np.isfinite(target_values).all():
        raise AssertionError("Primary targets contain non-finite values")
    if any(column in features for column in ("subclass", "group_id", *[f"target_{m}" for m in MODULES])):
        raise AssertionError("Predictor list contains metadata or targets")
    feature_values = table[features].to_numpy(dtype=float)
    if np.isinf(feature_values).any():
        raise AssertionError("Predictors contain infinite values")
    for fold in (0, 1, 2):
        train_groups = set(table.loc[table.fold.ne(fold), "group_id"].astype(str))
        test_groups = set(table.loc[table.fold.eq(fold), "group_id"].astype(str))
        if train_groups.intersection(test_groups):
            raise AssertionError(f"Outer group leakage in fold {fold}")
    return table, features


def _prediction_rows(
    test: pd.DataFrame,
    prediction: np.ndarray,
    *,
    module: str,
    model: str,
    analysis: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_cell_id": test.canonical_cell_id.astype(str).to_numpy(),
            "group_id": test.group_id.astype(str).to_numpy(),
            "subclass": test.subclass.astype(str).to_numpy(),
            "fold": test.fold.to_numpy(dtype=int),
            "module": module,
            "model": model,
            "analysis": analysis,
            "y_true": test[f"target_{module}"].to_numpy(dtype=float),
            "y_pred": np.asarray(prediction, dtype=float),
        }
    )


def _fit_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    module: str,
    *,
    fold: int,
    level: int,
    seed: int,
    model_dir: Path,
) -> tuple[list[pd.DataFrame], list[dict[str, object]], pd.DataFrame]:
    X_train, X_test = train[features], test[features]
    y_train = train[f"target_{module}"].to_numpy(dtype=float)
    groups = train.group_id.astype(str).to_numpy()
    predictions: list[pd.DataFrame] = []
    parameters: list[dict[str, object]] = []

    elastic = select_on_inner_groups(
        elastic_net_candidates(features, seed=seed), X_train, y_train, groups, seed=seed + fold
    )
    predictions.append(
        _prediction_rows(
            test, elastic.pipeline.predict(X_test), module=module, model="ElasticNet", analysis="ephys_only"
        )
    )
    parameters.append(
        {"module": module, "fold": fold, "model": "ElasticNet", "inner_rmse": elastic.inner_rmse, **elastic.parameters}
    )
    joblib.dump(elastic.pipeline, model_dir / f"{module}_fold{fold}_ElasticNet.joblib")

    boosted = select_on_inner_groups(
        xgboost_candidates(features, seed=seed, level=level),
        X_train,
        y_train,
        groups,
        seed=seed + fold,
    )
    predictions.append(
        _prediction_rows(
            test, boosted.pipeline.predict(X_test), module=module, model="XGBoost", analysis="ephys_only"
        )
    )
    parameters.append(
        {"module": module, "fold": fold, "model": "XGBoost", "inner_rmse": boosted.inner_rmse, **boosted.parameters}
    )
    joblib.dump(boosted.pipeline, model_dir / f"{module}_fold{fold}_XGBoost.joblib")
    heldout_shap = heldout_tree_shap(
        boosted.pipeline,
        X_test,
        cell_ids=test.canonical_cell_id,
        module=module,
        fold=fold,
    )

    mlp_seeds = (42, 123, 2026)[: (3, 2, 1)[level]]
    mlp_predictions = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        for mlp_seed in mlp_seeds:
            mlp = fit_mlp_on_inner_groups(
                features,
                X_train,
                y_train,
                groups,
                model_seed=mlp_seed,
                split_seed=seed + fold,
            )
            mlp_predictions.append(mlp.pipeline.predict(X_test))
            joblib.dump(mlp.pipeline, model_dir / f"{module}_fold{fold}_MLP_seed{mlp_seed}.joblib")
            parameters.append(
                {
                    "module": module,
                    "fold": fold,
                    "model": "MLP",
                    "inner_rmse": mlp.inner_rmse,
                    **mlp.parameters,
                }
            )
    predictions.append(
        _prediction_rows(
            test,
            np.mean(mlp_predictions, axis=0),
            module=module,
            model="MLP",
            analysis="ephys_only",
        )
    )

    dummy = dummy_model(features).fit(X_train, y_train)
    predictions.append(
        _prediction_rows(test, dummy.predict(X_test), module=module, model="Dummy", analysis="dummy")
    )
    subclass = subclass_ridge_model().fit(train[["subclass"]], y_train)
    predictions.append(
        _prediction_rows(
            test,
            subclass.predict(test[["subclass"]]),
            module=module,
            model="SubclassRidge",
            analysis="subclass_only",
        )
    )
    combined_columns = [*features, "subclass"]
    combined = ephys_subclass_ridge_model(features).fit(train[combined_columns], y_train)
    predictions.append(
        _prediction_rows(
            test,
            combined.predict(test[combined_columns]),
            module=module,
            model="EphysSubclassRidge",
            analysis="ephys_plus_subclass",
        )
    )
    return predictions, parameters, heldout_shap


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _validate_shap(
    shap_values: pd.DataFrame,
    table: pd.DataFrame,
    features: list[str],
    modules: tuple[str, ...],
) -> None:
    identity = {"canonical_cell_id", "module", "fold"}
    if set(shap_values.columns) != identity | set(features):
        raise AssertionError("SHAP columns do not exactly match held-out identity plus frozen features")
    if not np.isfinite(shap_values[features].to_numpy(dtype=float)).all():
        raise AssertionError("SHAP values must all be finite")
    if shap_values.duplicated(["canonical_cell_id", "module"]).any():
        raise AssertionError("Duplicate held-out SHAP row")
    if set(shap_values.module.astype(str)) != set(modules):
        raise AssertionError("SHAP module set is incomplete")
    expected_cells = set(table.canonical_cell_id.astype(str))
    frozen_fold = table.set_index(table.canonical_cell_id.astype(str))["fold"].astype(int)
    for module in modules:
        frame = shap_values.loc[shap_values.module.eq(module)].copy()
        if set(frame.canonical_cell_id.astype(str)) != expected_cells:
            raise AssertionError(f"Incomplete held-out SHAP coverage for {module}")
        expected_fold = frame.canonical_cell_id.astype(str).map(frozen_fold)
        if expected_fold.isna().any() or not np.array_equal(
            expected_fold.to_numpy(dtype=int), frame.fold.to_numpy(dtype=int)
        ):
            raise AssertionError(f"SHAP fold identity mismatch for {module}")


def _validate_oof_identity(oof: pd.DataFrame, table: pd.DataFrame) -> None:
    lookup = table.copy()
    lookup["canonical_cell_id"] = lookup.canonical_cell_id.astype(str)
    lookup = lookup.set_index("canonical_cell_id", verify_integrity=True)
    identifiers = oof.canonical_cell_id.astype(str)
    expected_fold = identifiers.map(lookup.fold.astype(int))
    expected_group = identifiers.map(lookup.group_id.astype(str))
    expected_subclass = identifiers.map(lookup.subclass.astype(str))
    if expected_fold.isna().any() or not np.array_equal(
        expected_fold.to_numpy(dtype=int), oof.fold.to_numpy(dtype=int)
    ):
        raise AssertionError("OOF rows do not use the frozen outer folds")
    if not np.array_equal(expected_group.to_numpy(dtype=str), oof.group_id.astype(str).to_numpy()):
        raise AssertionError("OOF group identities disagree with the modeling table")
    if not np.array_equal(
        expected_subclass.to_numpy(dtype=str), oof.subclass.astype(str).to_numpy()
    ):
        raise AssertionError("OOF subclasses disagree with the modeling table")
    for module, frame in oof.groupby("module", sort=False):
        target_column = f"target_{module}"
        if target_column not in lookup:
            raise AssertionError(f"Unknown OOF module {module}")
        expected_truth = frame.canonical_cell_id.astype(str).map(lookup[target_column])
        if not np.array_equal(
            expected_truth.to_numpy(dtype=float), frame.y_true.to_numpy(dtype=float)
        ):
            raise AssertionError(f"OOF outcomes disagree with frozen target {module}")


def _expected_parameter_rows(level: int, module_count: int) -> int:
    return module_count * 3 * (2 + (3, 2, 1)[level])


def _model_cache_complete(root: Path, modules: tuple[str, ...], level: int) -> bool:
    model_dir = root / "results/models"
    seeds = (42, 123, 2026)[: (3, 2, 1)[level]]
    expected = []
    for module in modules:
        for fold in (0, 1, 2):
            expected.extend(
                [
                    model_dir / f"{module}_fold{fold}_ElasticNet.joblib",
                    model_dir / f"{module}_fold{fold}_XGBoost.joblib",
                    *(model_dir / f"{module}_fold{fold}_MLP_seed{seed}.joblib" for seed in seeds),
                ]
            )
    return all(path.is_file() and path.stat().st_size > 0 for path in expected)


def _load_full_cache(
    root: Path,
    table: pd.DataFrame,
    features: list[str],
    *,
    signature: str,
    level: int,
) -> dict[str, object] | None:
    paths = {
        "summary": root / "logs/training_summary.json",
        "oof": root / "results/predictions/oof_predictions.parquet",
        "performance": root / "results/tables/model_performance.csv",
        "fold_performance": root / "results/tables/fold_performance.csv",
        "parameters": root / "results/tables/fitted_hyperparameters.csv",
        "shap": root / "results/shap/oof_shap.parquet",
        "stability": root / "results/tables/shap_stability.csv",
    }
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
        return None
    try:
        summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
        if summary.get("cache_signature") != signature or int(summary.get("level", -1)) != level:
            return None
        oof = pd.read_parquet(paths["oof"])
        assert_exact_oof_once(
            oof,
            set(table.canonical_cell_id.astype(str)),
            expected_oof_keys(MODULES),
        )
        _validate_oof_identity(oof, table)
        shap_values = pd.read_parquet(paths["shap"])
        _validate_shap(shap_values, table, features, MODULES)
        parameters = pd.read_csv(paths["parameters"])
        if len(parameters) != _expected_parameter_rows(level, len(MODULES)):
            raise AssertionError("Cached hyperparameter row count is incomplete")
        performance = pd.read_csv(paths["performance"])
        if len(performance) != len(expected_oof_keys(MODULES)):
            raise AssertionError("Cached performance grid is incomplete")
        fold_performance = pd.read_csv(paths["fold_performance"])
        if len(fold_performance) != 3 * len(expected_oof_keys(MODULES)):
            raise AssertionError("Cached fold-performance grid is incomplete")
        stability = pd.read_csv(paths["stability"])
        if len(stability) != len(MODULES) * len(features):
            raise AssertionError("Cached SHAP stability grid is incomplete")
        if not _model_cache_complete(root, MODULES, level):
            raise AssertionError("Cached primary fitted-model set is incomplete")
    except (AssertionError, KeyError, OSError, TypeError, ValueError):
        return None
    return {**summary, "cached": True}


def _load_partial_cache(
    root: Path,
    table: pd.DataFrame,
    features: list[str],
    *,
    signature: str,
    level: int,
) -> tuple[list[pd.DataFrame], list[dict[str, object]], list[pd.DataFrame], set[str]]:
    manifest_path = root / "logs/training_partial_manifest.json"
    oof_path = root / "results/predictions/oof_predictions.partial.parquet"
    shap_path = root / "results/shap/oof_shap.partial.parquet"
    parameter_path = root / "results/tables/fitted_hyperparameters.partial.csv"
    paths = (manifest_path, oof_path, shap_path, parameter_path)
    if not all(path.is_file() and path.stat().st_size > 0 for path in paths):
        return [], [], [], set()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("cache_signature") != signature:
            return [], [], [], set()
        oof = pd.read_parquet(oof_path)
        shap_values = pd.read_parquet(shap_path)
        parameters = pd.read_csv(parameter_path)
    except (OSError, TypeError, ValueError):
        return [], [], [], set()

    prediction_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    shap_parts: list[pd.DataFrame] = []
    completed: set[str] = set()
    expected_cells = set(table.canonical_cell_id.astype(str))
    for module in MODULES:
        module_oof = oof.loc[oof.module.eq(module)].copy()
        module_shap = shap_values.loc[shap_values.module.eq(module)].copy()
        module_parameters = parameters.loc[parameters.module.eq(module)].copy()
        try:
            assert_exact_oof_once(
                module_oof,
                expected_cells,
                expected_oof_keys((module,)),
            )
            _validate_oof_identity(module_oof, table)
            _validate_shap(module_shap, table, features, (module,))
            if len(module_parameters) != _expected_parameter_rows(level, 1):
                raise AssertionError("Incomplete per-module hyperparameter cache")
            if not _model_cache_complete(root, (module,), level):
                raise AssertionError("Incomplete per-module fitted-model cache")
        except (AssertionError, KeyError, TypeError, ValueError):
            continue
        prediction_parts.append(module_oof)
        shap_parts.append(module_shap)
        parameter_rows.extend(module_parameters.to_dict("records"))
        completed.add(module)
    return prediction_parts, parameter_rows, shap_parts, completed


def _write_partial_cache(
    root: Path,
    prediction_parts: list[pd.DataFrame],
    parameter_rows: list[dict[str, object]],
    shap_parts: list[pd.DataFrame],
    *,
    signature: str,
    completed_modules: set[str],
) -> None:
    _atomic_parquet(
        pd.concat(prediction_parts, ignore_index=True),
        root / "results/predictions/oof_predictions.partial.parquet",
    )
    _atomic_parquet(
        pd.concat(shap_parts, ignore_index=True),
        root / "results/shap/oof_shap.partial.parquet",
    )
    _atomic_csv(
        pd.DataFrame(parameter_rows),
        root / "results/tables/fitted_hyperparameters.partial.csv",
    )
    _atomic_json(
        {
            "cache_signature": signature,
            "completed_modules": [module for module in MODULES if module in completed_modules],
        },
        root / "logs/training_partial_manifest.json",
    )


def run_training(
    root: str | Path,
    *,
    level: int,
    smoke: bool = False,
    seed: int = 42,
    use_cache: bool = True,
) -> dict[str, object]:
    if level not in (0, 1, 2):
        raise ValueError("RuntimeGovernor level must be 0, 1, or 2")
    root = Path(root)
    table, features = load_and_validate_inputs(
        root / "data/processed/modeling_table.parquet",
        root / "results/tables/cv_folds.csv",
        root / "config/ephys_features.yaml",
    )
    governor = RuntimeGovernor(
        150,
        os.environ.get("PATCHSEQ_START_TIME"),
        root / "logs/runtime_checkpoints.jsonl",
    )
    governor.level = level
    modules = ("HCN",) if smoke else MODULES
    folds = (0,) if smoke else (0, 1, 2)
    signature = _training_signature(root, level=level, seed=seed)
    if not smoke and use_cache:
        cached = _load_full_cache(
            root,
            table,
            features,
            signature=signature,
            level=level,
        )
        if cached is not None:
            governor.checkpoint("training_cache_validated")
            return cached
    if not smoke and governor.elapsed_minutes() >= 145.0:
        state = {
            "status": "not_started_hard_compute_stop",
            "elapsed_minutes": governor.elapsed_minutes(),
            "level": level,
            "cache_signature": signature,
        }
        _atomic_json(state, root / "logs/training_deferred.json")
        raise RuntimeError("Hard compute stop reached before full training could start")
    model_dir = root / "results/models"
    model_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    prediction_parts: list[pd.DataFrame] = []
    parameter_rows: list[dict[str, object]] = []
    shap_parts: list[pd.DataFrame] = []
    completed_modules: set[str] = set()
    if not smoke and use_cache:
        prediction_parts, parameter_rows, shap_parts, completed_modules = _load_partial_cache(
            root,
            table,
            features,
            signature=signature,
            level=level,
        )
    resumed_modules = set(completed_modules)
    for module in modules:
        module_started = time.perf_counter()
        if module in completed_modules:
            print(f"{module}: cached", flush=True)
            governor.checkpoint(f"training_{module}_cache_validated")
            continue
        if not smoke and governor.elapsed_minutes() >= 145.0:
            _write_partial_cache(
                root,
                prediction_parts,
                parameter_rows,
                shap_parts,
                signature=signature,
                completed_modules=completed_modules,
            )
            raise RuntimeError(f"Hard compute stop reached before starting mandatory module {module}")
        for fold in folds:
            train, test = table.loc[table.fold.ne(fold)].copy(), table.loc[table.fold.eq(fold)].copy()
            prediction, parameters, heldout_shap = _fit_fold(
                train,
                test,
                features,
                module,
                fold=fold,
                level=level,
                seed=seed,
                model_dir=model_dir,
            )
            prediction_parts.extend(prediction)
            parameter_rows.extend(parameters)
            shap_parts.append(heldout_shap)
        if not smoke:
            completed_modules.add(module)
            _write_partial_cache(
                root,
                prediction_parts,
                parameter_rows,
                shap_parts,
                signature=signature,
                completed_modules=completed_modules,
            )
        governor.checkpoint(f"training_{module}_complete")
        print(f"{module}: {time.perf_counter() - module_started:.3f}s", flush=True)

    elapsed = time.perf_counter() - started
    oof = pd.concat(prediction_parts, ignore_index=True)
    shap_values = pd.concat(shap_parts, ignore_index=True)
    if smoke:
        smoke_cells = set(table.loc[table.fold.eq(0), "canonical_cell_id"].astype(str))
        assert_exact_oof_once(oof, smoke_cells, expected_oof_keys(("HCN",)))
        smoke_table = table.loc[table.fold.eq(0)].copy()
        _validate_oof_identity(oof, smoke_table)
        _validate_shap(shap_values, smoke_table, features, ("HCN",))
        result = {
            "module": "HCN",
            "fold": 0,
            "level": level,
            "test_n": int(table.fold.eq(0).sum()),
            "elapsed_seconds": elapsed,
            "projected_primary_seconds": elapsed * 18,
            "shap_shape": list(shap_values.shape),
            "models": sorted(oof.model.unique().tolist()),
            "primary_models": ["ElasticNet", "XGBoost", "MLP"],
            "cache_signature": signature,
        }
        _atomic_json(result, root / "logs/smoke_benchmark.json")
        return result

    expected_cells = set(table.canonical_cell_id.astype(str))
    assert_exact_oof_once(oof, expected_cells, expected_oof_keys(MODULES))
    _validate_oof_identity(oof, table)
    _validate_shap(shap_values, table, features, MODULES)
    performance = performance_table(oof)
    folds_table = fold_metrics(oof)
    stability = shap_stability(shap_values)
    _atomic_parquet(oof, root / "results/predictions/oof_predictions.parquet")
    _atomic_csv(performance, root / "results/tables/model_performance.csv")
    _atomic_csv(folds_table, root / "results/tables/fold_performance.csv")
    _atomic_csv(
        pd.DataFrame(parameter_rows),
        root / "results/tables/fitted_hyperparameters.csv",
    )
    _atomic_parquet(shap_values, root / "results/shap/oof_shap.parquet")
    _atomic_csv(stability, root / "results/tables/shap_stability.csv")
    summary = {
        "elapsed_seconds": elapsed,
        "n_cells": len(table),
        "n_groups": int(table.group_id.nunique()),
        "n_features": len(features),
        "modules": list(MODULES),
        "outer_folds": 3,
        "level": level,
        "seed": seed,
        "cache_signature": signature,
        "resumed_modules": [module for module in MODULES if module in resumed_modules],
        "cached": False,
    }
    _atomic_json(summary, root / "logs/training_summary.json")
    return summary
