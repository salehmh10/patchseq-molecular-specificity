"""Integrity gates for isolated additional robustness analyses."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import regression_metrics
from src.training import MODULES


ROOT = Path(__file__).resolve().parents[1]
ROBUST = ROOT / "results/robustness_v2"
PRIMARY_OOF_SHA256 = "9d0fb9edb89b3133fd1e356cbc437ca7aa3b985df4f810805b0cce74b96a4217"
PRIMARY_SHAP_SHA256 = "a6a0c7dc5df16106882c1da74faa043e8a54ccf82d2a73325cd4eabf3094453b"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_seed(*values: object, base: int = 42) -> int:
    digest = hashlib.sha256("|".join(map(str, values)).encode("utf-8")).digest()
    return (base + int.from_bytes(digest[:4], "little")) % (2**32 - 1)


def test_primary_hash_unchanged() -> None:
    assert _sha256(ROOT / "results/predictions/oof_predictions.parquet") == PRIMARY_OOF_SHA256
    assert _sha256(ROOT / "results/shap/oof_shap.parquet") == PRIMARY_SHAP_SHA256


def test_incremental_bootstrap_paired() -> None:
    summary = pd.read_csv(ROBUST / "tables/incremental_subclass_comparison.csv")
    draws = pd.read_parquet(ROBUST / "tables/incremental_subclass_bootstrap_draws.parquet")
    required = {
        "module", "metric", "subclass_only", "ephys_plus_subclass", "delta",
        "ci_low", "ci_high", "p_delta_positive",
    }
    assert required.issubset(summary.columns)
    assert len(summary) == 24
    assert summary.n_bootstrap.eq(2000).all()
    assert summary.bootstrap_unit.str.contains("paired", case=False).all()
    assert draws.groupby(["module", "metric"]).size().eq(2000).all()

    # Independently reconstruct one stored replicate. Both predictions use the
    # same sampled donor multiplicities, which is the defining paired property.
    oof = pd.read_parquet(ROOT / "results/predictions/oof_predictions.parquet")
    columns = ["canonical_cell_id", "group_id", "y_true", "y_pred"]
    base = oof.loc[
        oof.module.eq("NaV") & oof.model.eq("SubclassRidge") & oof.analysis.eq("subclass_only"),
        columns,
    ].rename(columns={"y_pred": "base"})
    augmented = oof.loc[
        oof.module.eq("NaV")
        & oof.model.eq("EphysSubclassRidge")
        & oof.analysis.eq("ephys_plus_subclass"),
        ["canonical_cell_id", "y_true", "y_pred"],
    ].rename(columns={"y_true": "y_true_aug", "y_pred": "augmented"})
    paired = base.merge(augmented, on="canonical_cell_id", validate="one_to_one")
    assert np.array_equal(paired.y_true, paired.y_true_aug)
    codes, groups = pd.factorize(paired.group_id.astype(str), sort=False)
    rng = np.random.default_rng(_stable_seed("R1", "NaV"))
    sampled = rng.integers(0, len(groups), size=len(groups))
    multiplicity = np.bincount(sampled, minlength=len(groups))
    indices = np.repeat(np.arange(len(paired)), multiplicity[codes])
    base_r2 = regression_metrics(paired.y_true.to_numpy()[indices], paired.base.to_numpy()[indices])["r2"]
    augmented_r2 = regression_metrics(
        paired.y_true.to_numpy()[indices], paired.augmented.to_numpy()[indices]
    )["r2"]
    saved = draws.loc[
        draws.module.eq("NaV") & draws.metric.eq("r2") & draws.replicate.eq(0), "delta"
    ].item()
    assert np.isclose(saved, augmented_r2 - base_r2, rtol=0, atol=1e-12)


def test_loo_targets_differ_from_primary() -> None:
    oof = pd.read_parquet(ROBUST / "predictions/module_loo_oof.parquet")
    representative = oof.loc[oof.model.eq("ElasticNet")]
    for target_name, frame in representative.groupby("target_name"):
        assert frame.canonical_cell_id.astype(str).is_unique, target_name
        assert not np.array_equal(frame.y_true.to_numpy(), frame.primary_y_true.to_numpy())
        assert frame.y_true.corr(frame.primary_y_true, method="spearman") < 0.90


def test_loo_predictions_are_new() -> None:
    oof = pd.read_parquet(ROBUST / "predictions/module_loo_oof.parquet")
    assert len(oof) == 3410 * 4 * 4
    assert not oof.duplicated(["canonical_cell_id", "target_name", "model"]).any()
    for keys, frame in oof.loc[oof.model.isin(["ElasticNet", "XGBoost"])].groupby(
        ["target_name", "model"]
    ):
        assert np.mean(np.abs(frame.y_pred - frame.primary_y_pred)) > 1e-6, keys


def test_repeated_cv_donor_disjoint() -> None:
    repeated = pd.read_csv(
        ROBUST / "tables/repeated_group_cv.csv", dtype={"canonical_cell_id": str, "group_id": str}
    )
    assert len(repeated) == 5 * 6 * 2 * 3410
    assert repeated.groupby(["repeat", "group_id"]).fold.nunique().max() == 1
    assert repeated.groupby(["repeat", "module", "model", "canonical_cell_id"]).size().eq(1).all()
    assert set(repeated.repeat) == {0, 1, 2, 3, 4}


def test_duplicate_feature_removed() -> None:
    removed = "stimulus_amplitude_0_long_square"
    sensitivity_shap = pd.read_parquet(ROBUST / "shap/duplicate_removed_oof_shap.parquet")
    assert removed not in sensitivity_shap.columns
    assert "rheobase_i" in sensitivity_shap.columns
    assert len(sensitivity_shap.columns) == 23 + 3


def test_duplicate_sensitivity_oof_complete() -> None:
    sensitivity = pd.read_parquet(ROBUST / "predictions/duplicate_removed_oof_predictions.parquet")
    frozen = pd.read_csv(ROOT / "results/tables/cv_folds.csv", dtype={"canonical_cell_id": str})
    sensitivity["canonical_cell_id"] = sensitivity.canonical_cell_id.astype(str)
    assert len(sensitivity) == 3410 * 6
    assert not sensitivity.duplicated(["canonical_cell_id", "module"]).any()
    assert set(sensitivity.module) == set(MODULES)
    expected_fold = sensitivity.canonical_cell_id.map(frozen.set_index("canonical_cell_id").fold)
    assert np.array_equal(sensitivity.fold.to_numpy(), expected_fold.to_numpy())


def test_grouped_shap_reconciles() -> None:
    primary = pd.read_parquet(ROOT / "results/shap/oof_shap.parquet")
    grouped = pd.read_parquet(ROBUST / "shap/primary_grouped_abs_shap.parquet")
    keys = ["canonical_cell_id", "module", "fold"]
    feature_columns = [name for name in primary.columns if name not in keys]
    group_columns = [name for name in grouped.columns if name not in keys]
    left = primary.assign(_sum=primary[feature_columns].abs().sum(axis=1))[keys + ["_sum"]]
    right = grouped.assign(_sum_grouped=grouped[group_columns].sum(axis=1))[keys + ["_sum_grouped"]]
    joined = left.merge(right, on=keys, validate="one_to_one")
    assert len(joined) == 3410 * 6
    assert np.allclose(joined._sum, joined._sum_grouped, rtol=0, atol=1e-12)


def test_robustness_tables_complete() -> None:
    required = {
        "incremental_subclass_comparison.csv",
        "module_loo_refit_performance.csv",
        "duplicate_removed_performance.csv",
        "duplicate_removed_shap_stability.csv",
        "grouped_shap_importance.csv",
        "repeated_group_cv.csv",
        "repeated_group_cv_summary.csv",
    }
    observed = {path.name for path in (ROBUST / "tables").glob("*")}
    assert required.issubset(observed)
    assert len(pd.read_csv(ROBUST / "tables/duplicate_removed_performance.csv")) == 6
    assert len(pd.read_csv(ROBUST / "tables/grouped_shap_importance.csv")) == 6 * 19
    loo = pd.read_csv(ROBUST / "tables/module_loo_refit_performance.csv")
    assert set(loo.target_name) == {
        "NaV_minus_Scn9a", "NaV_minus_Scn3a", "HCN_minus_Hcn2", "HCN_minus_Hcn3"
    }
    assert set(loo.model) == {"Dummy", "SubclassRidge", "ElasticNet", "XGBoost"}
