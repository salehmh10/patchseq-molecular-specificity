#!/usr/bin/env python
"""Audit held-out XGBoost SHAP and recalculate fold stability."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.interpretability import heldout_tree_shap, high_correlation_pairs, shap_stability
from src.training import MODULES, load_and_validate_inputs


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def _same_stability(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    sort_columns = ["module", "feature"]
    left = left.sort_values(sort_columns).reset_index(drop=True)
    right = right.sort_values(sort_columns).reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left:
        if pd.api.types.is_numeric_dtype(left[column]):
            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                equal_nan=True,
                rtol=0.0,
                atol=1e-12,
            ):
                return False
        elif not left[column].equals(right[column]):
            return False
    return True


def main() -> None:
    started = time.perf_counter()
    table, features = load_and_validate_inputs(
        ROOT / "data/processed/modeling_table.parquet",
        ROOT / "results/tables/cv_folds.csv",
        ROOT / "config/ephys_features.yaml",
    )
    values = pd.read_parquet(ROOT / "results/shap/oof_shap.parquet")
    values["canonical_cell_id"] = values["canonical_cell_id"].astype(str)
    required = {"canonical_cell_id", "module", "fold", *features}
    if set(values.columns) != required:
        raise AssertionError("OOF SHAP schema does not match the frozen feature set")
    if set(values.module) != set(MODULES):
        raise AssertionError("OOF SHAP does not cover all frozen modules")
    if values.duplicated(["canonical_cell_id", "module"]).any():
        raise AssertionError("A cell/module has more than one OOF SHAP row")
    if not np.isfinite(values[features].to_numpy(dtype=float)).all():
        raise AssertionError("OOF SHAP contains non-finite values")

    expected_cells = set(table.canonical_cell_id.astype(str))
    expected_rows = len(expected_cells) * len(MODULES)
    if len(values) != expected_rows:
        raise AssertionError(f"Expected {expected_rows} held-out SHAP rows; found {len(values)}")
    frozen_fold = table.set_index(table.canonical_cell_id.astype(str))["fold"].astype(int)
    expected_keys = {
        (cell_id, module, int(frozen_fold.loc[cell_id]))
        for cell_id in expected_cells
        for module in MODULES
    }
    actual_keys = set(
        values[["canonical_cell_id", "module", "fold"]].itertuples(index=False, name=None)
    )
    if actual_keys != expected_keys:
        raise AssertionError("SHAP cell/module/fold keys differ from the frozen outer-fold grid")

    oof = pd.read_parquet(ROOT / "results/predictions/oof_predictions.parquet")
    oof["canonical_cell_id"] = oof["canonical_cell_id"].astype(str)
    xgboost = oof.loc[
        oof.model.eq("XGBoost") & oof.analysis.eq("ephys_only"),
        ["canonical_cell_id", "module", "fold"],
    ]
    xgboost_keys = set(xgboost.itertuples(index=False, name=None))
    if actual_keys != xgboost_keys:
        raise AssertionError("SHAP keys differ from the held-out XGBoost OOF prediction keys")

    max_abs_difference = 0.0
    compared_values = 0
    models_recomputed = 0
    for module in MODULES:
        for fold in (0, 1, 2):
            test = table.loc[table.fold.eq(fold)]
            pipeline = joblib.load(ROOT / f"results/models/{module}_fold{fold}_XGBoost.joblib")
            recalculated = heldout_tree_shap(
                pipeline,
                test[features],
                cell_ids=test.canonical_cell_id,
                module=module,
                fold=fold,
            ).sort_values("canonical_cell_id").reset_index(drop=True)
            cached = values.loc[
                values.module.eq(module) & values.fold.eq(fold)
            ].sort_values("canonical_cell_id").reset_index(drop=True)
            identity = ["canonical_cell_id", "module", "fold"]
            if not cached[identity].equals(recalculated[identity]):
                raise AssertionError(f"Cached SHAP identities differ for {module} fold {fold}")
            difference = np.abs(
                cached[features].to_numpy(dtype=float)
                - recalculated[features].to_numpy(dtype=float)
            )
            max_abs_difference = max(max_abs_difference, float(difference.max()))
            compared_values += int(difference.size)
            models_recomputed += 1
    if max_abs_difference > 1e-12:
        raise AssertionError(
            f"Cached SHAP differs from held-out model recomputation: {max_abs_difference}"
        )

    summary = shap_stability(values)
    stability_path = ROOT / "results/tables/shap_stability.csv"
    previous = pd.read_csv(stability_path) if stability_path.exists() else None
    matches_previous = previous is not None and _same_stability(previous, summary)
    summary.to_csv(stability_path, index=False)

    correlations = high_correlation_pairs(table, features, threshold=0.90)
    correlations.to_csv(ROOT / "results/tables/high_correlation_pairs.csv", index=False)
    audit = {
        "audit_version": "heldout-shap-audit-v1",
        "checked_at": datetime.now().astimezone().isoformat(),
        "status": "PASS",
        "elapsed_seconds": time.perf_counter() - started,
        "n_cells": len(expected_cells),
        "n_modules": len(MODULES),
        "n_outer_folds": 3,
        "oof_shap_rows": len(values),
        "expected_oof_shap_rows": expected_rows,
        "duplicate_cell_module_rows": int(values.duplicated(["canonical_cell_id", "module"]).sum()),
        "nonfinite_shap_values": int(
            values[features].size - np.isfinite(values[features].to_numpy(dtype=float)).sum()
        ),
        "frozen_cell_module_fold_keys_exact": actual_keys == expected_keys,
        "xgboost_oof_keys_exact": actual_keys == xgboost_keys,
        "stored_fold_models_recomputed": models_recomputed,
        "shap_values_recomputed": compared_values,
        "max_abs_difference_from_model_recomputation": max_abs_difference,
        "stability_rows": len(summary),
        "stability_matches_previous_at_1e-12": matches_previous,
        "stable_top_features": int(summary.stable_top_feature.astype(bool).sum()),
        "high_correlation_pairs_abs_spearman_gt_0_90": len(correlations),
        "exact_duplicate_feature_pairs": int(
            correlations.exact_duplicate_on_complete_cases.astype(bool).sum()
        ),
    }
    _atomic_json(audit, ROOT / "results/tables/shap_audit.json")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
