"""Paper-grade identity, leakage, OOF, metric, and SHAP invariants."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

from src.evaluation import regression_metrics
from src.training import MODEL_ANALYSES, MODULES, load_feature_names


ROOT = Path(__file__).resolve().parents[1]


def _table() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "data/processed/modeling_table.parquet")


def _folds() -> pd.DataFrame:
    return pd.read_csv(ROOT / "results/tables/cv_folds.csv", dtype={"canonical_cell_id": str})


def _oof() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "results/predictions/oof_predictions.parquet")


def test_mat_loads() -> None:
    mat = loadmat(ROOT / "data/raw/PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat", squeeze_me=True)
    required = {"E_feature", "E_spec_id_label", "T_dat", "T_spec_id_label", "feature_name", "feature_mean", "feature_std"}
    assert required.issubset(mat)
    assert mat["E_feature"].shape[0] == len(mat["E_spec_id_label"])


def test_feature_names_match() -> None:
    mat = loadmat(ROOT / "data/raw/PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat", squeeze_me=True)
    observed = [str(value) for value in np.asarray(mat["feature_name"]).ravel()]
    frozen = load_feature_names(ROOT / "config/ephys_features.yaml")
    assert observed == frozen
    assert mat["E_feature"].shape[1] == len(frozen)


def test_cell_ids_unique() -> None:
    table = _table()
    assert table.canonical_cell_id.notna().all()
    assert table.canonical_cell_id.astype(str).is_unique


def test_crosswalk_one_to_one() -> None:
    crosswalk = pd.read_csv(ROOT / "results/tables/cell_id_crosswalk.csv", dtype=str)
    assert crosswalk.canonical_cell_id.notna().all()
    assert crosswalk.canonical_cell_id.is_unique
    for column in ("cell_specimen_id", "transcriptomics_sample_id"):
        if column in crosswalk:
            assert crosswalk[column].notna().all()
            assert crosswalk[column].is_unique


def test_targets_exist() -> None:
    table = _table()
    target_columns = [f"target_{module}" for module in MODULES]
    assert set(target_columns).issubset(table)
    assert np.isfinite(table[target_columns].to_numpy(dtype=float)).all()


def test_no_x_target_contamination() -> None:
    features = load_feature_names(ROOT / "config/ephys_features.yaml")
    forbidden = {f"target_{module}" for module in MODULES} | {"group_id", "subclass", "canonical_cell_id"}
    assert not forbidden.intersection(features)
    assert set(features).issubset(_table().columns)


def test_cv_cell_disjoint() -> None:
    folds = _folds()
    assert folds.canonical_cell_id.is_unique
    assert set(folds.fold.astype(int)) == {0, 1, 2}
    test_sets = [set(folds.loc[folds.fold.eq(fold), "canonical_cell_id"]) for fold in (0, 1, 2)]
    assert not test_sets[0].intersection(test_sets[1])
    assert not test_sets[0].intersection(test_sets[2])
    assert not test_sets[1].intersection(test_sets[2])


def test_cv_group_disjoint() -> None:
    table = _table().copy()
    table["canonical_cell_id"] = table.canonical_cell_id.astype(str)
    merged = table.drop(columns=["fold"], errors="ignore").merge(
        _folds()[["canonical_cell_id", "fold"]], on="canonical_cell_id", validate="one_to_one"
    )
    for fold in (0, 1, 2):
        training = set(merged.loc[merged.fold.ne(fold), "group_id"].astype(str))
        testing = set(merged.loc[merged.fold.eq(fold), "group_id"].astype(str))
        assert training.isdisjoint(testing)


def test_same_folds_all_models() -> None:
    oof = _oof().copy()
    oof["canonical_cell_id"] = oof.canonical_cell_id.astype(str)
    frozen = _folds().set_index("canonical_cell_id")["fold"].astype(int)
    expected = oof.canonical_cell_id.map(frozen)
    assert expected.notna().all()
    assert np.array_equal(oof.fold.to_numpy(dtype=int), expected.to_numpy(dtype=int))


def test_oof_exactly_once() -> None:
    oof = _oof()
    expected = set(_table().canonical_cell_id.astype(str))
    expected_keys = {
        (module, model, analysis)
        for module in MODULES
        for model, analysis in MODEL_ANALYSES
    }
    observed_keys = set(
        oof[["module", "model", "analysis"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    assert observed_keys == expected_keys
    assert np.isfinite(oof[["y_true", "y_pred"]].to_numpy(dtype=float)).all()
    for keys in expected_keys:
        frame = oof.loc[
            oof.module.eq(keys[0]) & oof.model.eq(keys[1]) & oof.analysis.eq(keys[2])
        ]
        ids = frame.canonical_cell_id.astype(str)
        assert ids.is_unique
        assert set(ids) == expected


def test_metric_recalculation() -> None:
    oof = _oof()
    saved = pd.read_csv(ROOT / "results/tables/model_performance.csv").set_index(["module", "model", "analysis"])
    for keys, frame in oof.groupby(["module", "model", "analysis"]):
        recalculated = regression_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())
        for metric, value in recalculated.items():
            assert np.isclose(value, saved.loc[keys, metric], rtol=1e-10, atol=1e-12, equal_nan=True)


def test_shap_shape() -> None:
    shap_values = pd.read_parquet(ROOT / "results/shap/oof_shap.parquet")
    features = load_feature_names(ROOT / "config/ephys_features.yaml")
    table = _table()
    assert set(features) == set(shap_values.columns) - {"canonical_cell_id", "module", "fold"}
    assert len(shap_values) == len(table) * len(MODULES)
    assert not shap_values.duplicated(["canonical_cell_id", "module"]).any()
    assert np.isfinite(shap_values[features].to_numpy(dtype=float)).all()


def test_final_n_consistent() -> None:
    n = len(_table())
    assert len(_folds()) == n
    assert pd.read_csv(ROOT / "results/tables/cell_id_crosswalk.csv").shape[0] == n
    oof = _oof()
    primary = oof.loc[oof.model.eq("ElasticNet") & oof.analysis.eq("ephys_only")]
    assert len(primary) == n * len(MODULES)
    donor_permutations = pd.read_csv(
        ROOT / "results/tables/donor_level_permutation_sensitivity.csv"
    )
    assert len(donor_permutations) == len(MODULES) * 3
    assert donor_permutations.n_groups.eq(_table().group_id.astype(str).nunique()).all()
    assert donor_permutations.n_cells.eq(n).all()
    assert donor_permutations.n_permutations.eq(5000).all()
    assert donor_permutations.p_value_two_sided.between(1 / 5001 - 1e-12, 1).all()
