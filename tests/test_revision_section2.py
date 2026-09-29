"""Model-comparison analysis synthetic contracts and strict full-run gates.

Fast tests exercise the statistical primitives without fitting the analysis::

    python -m pytest -q tests/test_revision_section2.py -k synthetic

The ``test_full_*`` tests are deliberately strict.  They are the post-run
definition-of-done gate, so a missing mandatory artifact is a failure rather
than a skip.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.revision_section2 as rs
from src.training import MODULES, load_feature_names


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/revision_v3/section2"
FIGURES = ROOT / "figures/revision_v3/section2"
LOGS = ROOT / "logs/revision_v3/section2"
MODELS = ROOT / "models/revision_v3/section2"

MODULE_ORDER = tuple(MODULES)
PRIMARY_MODELS = ("ElasticNet", "XGBoost", "MLP")
ALL_MODELS = (
    "ElasticNet",
    "XGBoost",
    "MLP",
    "Dummy",
    "SubclassRidge",
    "EphysSubclassRidge",
)
CONTRASTS = (
    ("XGBoost", "ElasticNet"),
    ("MLP", "ElasticNet"),
    ("MLP", "XGBoost"),
)
METRICS = ("r2", "mae", "rmse", "spearman")
REPEAT_SEEDS = (42, 123, 2026, 31415, 27182)
MLP_SEEDS = (42, 123, 2026)
EXPECTED_PRIMARY_OOF_HASH = (
    "9d0fb9edb89b3133fd1e356cbc437ca7aa3b985df4f810805b0cce74b96a4217"
)
EXPECTED_PRIMARY_SHAP_HASH = (
    "a6a0c7dc5df16106882c1da74faa043e8a54ccf82d2a73325cd4eabf3094453b"
)
EXPECTED_SECTION1_MANIFEST_HASH = (
    "1cd9397323b7f6f758def2da60d72d1df11c20145c6ff64628e6ab237381581b"
)


def _require_files(*paths: Path) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    empty = [
        str(path.relative_to(ROOT))
        for path in paths
        if path.is_file() and path.stat().st_size == 0
    ]
    assert not missing, "Required Section 2 artifacts are missing: " + ", ".join(missing)
    assert not empty, "Required Section 2 artifacts are empty: " + ", ".join(empty)


def _modeling_table() -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / "data/processed/modeling_table.parquet").copy()
    frame["canonical_cell_id"] = frame.canonical_cell_id.astype(str)
    frame["group_id"] = frame.group_id.astype(str)
    return frame


def _folds() -> pd.DataFrame:
    frame = pd.read_csv(
        ROOT / "results/tables/cv_folds.csv", dtype={"canonical_cell_id": str}
    )
    frame["fold"] = frame.fold.astype(int)
    return frame


def _metric_column(frame: pd.DataFrame, name: str) -> str:
    aliases = {
        "r2": ("r2", "R2", "R_squared"),
        "mae": ("mae", "MAE"),
        "rmse": ("rmse", "RMSE"),
        "spearman": ("spearman", "spearman_rho", "rho"),
    }
    for candidate in aliases[name]:
        if candidate in frame.columns:
            return candidate
    raise AssertionError(f"Missing {name} metric column from {frame.columns.tolist()}")


def _model_column(frame: pd.DataFrame) -> str:
    for column in ("model", "model_name"):
        if column in frame:
            return column
    raise AssertionError("No model column")


def _draw_column(frame: pd.DataFrame) -> str:
    for column in ("draw", "replicate", "bootstrap_id", "resample"):
        if column in frame:
            return column
    raise AssertionError("No bootstrap draw identifier")


def _permutation_p_column(frame: pd.DataFrame) -> str:
    for column in ("p_value", "empirical_p", "p_plus_one"):
        if column in frame:
            return column
    raise AssertionError("No permutation p-value column")


def _assert_exact_oof_grid(
    frame: pd.DataFrame,
    *,
    models: tuple[str, ...] = ALL_MODELS,
) -> None:
    required = {
        "canonical_cell_id",
        "group_id",
        "subclass",
        "fold",
        "module",
        "model",
        "y_true",
        "y_pred",
    }
    assert required.issubset(frame.columns)
    assert np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all()
    observed = set(
        frame[["module", "model"]].drop_duplicates().itertuples(index=False, name=None)
    )
    expected = {(module, model) for module in MODULE_ORDER for model in models}
    assert observed == expected
    expected_ids = set(_modeling_table().canonical_cell_id)
    assert not frame.duplicated(["canonical_cell_id", "module", "model"]).any()
    for _, part in frame.groupby(["module", "model"], sort=False):
        assert len(part) == 3410
        assert set(part.canonical_cell_id.astype(str)) == expected_ids


def _assert_saved_metric_reconstruction(
    predictions: pd.DataFrame,
    saved: pd.DataFrame,
    keys: tuple[str, ...],
) -> None:
    indexed = saved.set_index(list(keys))
    for raw_key, part in predictions.groupby(list(keys), sort=True):
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        for metric in METRICS:
            observed = float(indexed.loc[key, _metric_column(saved, metric)])
            expected = rs.metric_value(
                part.y_true.to_numpy(dtype=float),
                part.y_pred.to_numpy(dtype=float),
                metric,
            )
            assert np.isclose(observed, expected, rtol=1e-10, atol=1e-12, equal_nan=True), (
                key,
                metric,
                observed,
                expected,
            )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Fast synthetic contracts.


def test_synthetic_stable_seed_and_file_hash(tmp_path: Path) -> None:
    expected = int.from_bytes(
        hashlib.sha256(
            f"{rs.PIPELINE_VERSION}|42|bootstrap|GABAA|0".encode("utf-8")
        ).digest()[:4],
        "big",
    )
    assert rs.stable_seed("bootstrap", "GABAA", 0, base=42) == expected
    assert rs.stable_seed("bootstrap", "GABAA", 0, base=42) == expected
    assert rs.stable_seed("bootstrap", "GABAA", 1, base=42) != expected
    assert 0 <= expected < 2**32

    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"section-2\x00contract")
    assert rs.sha256_file(payload) == hashlib.sha256(payload.read_bytes()).hexdigest()


def test_synthetic_bh_adjust_is_exact_monotone_and_rejects_nan() -> None:
    p_values = np.array([0.01, 0.04, 0.03, 0.002, 0.5])
    adjusted = np.asarray(rs.bh_adjust(p_values), dtype=float)
    expected = np.array([0.025, 0.05, 0.05, 0.01, 0.5])
    np.testing.assert_allclose(adjusted, expected, rtol=0, atol=1e-15)
    finite_order = np.argsort(p_values, kind="mergesort")
    assert np.all(np.diff(adjusted[finite_order]) >= -1e-15)
    assert np.all((adjusted >= p_values) & (adjusted <= 1.0))
    with pytest.raises(ValueError):
        rs.bh_adjust([0.1, np.nan])


def test_synthetic_metric_values_match_hand_calculation() -> None:
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    prediction = np.array([0.0, 1.0, 1.0, 4.0])
    residual = truth - prediction
    expected_r2 = 1.0 - np.sum(residual**2) / np.sum((truth - truth.mean()) ** 2)
    assert rs.metric_value(truth, prediction, "r2") == pytest.approx(expected_r2)
    assert rs.metric_value(truth, prediction, "mae") == pytest.approx(0.5)
    assert rs.metric_value(truth, prediction, "rmse") == pytest.approx(math.sqrt(0.5))
    assert rs.metric_value(truth, prediction, "spearman") == pytest.approx(
        pd.Series(truth).corr(pd.Series(prediction), method="spearman")
    )


def test_synthetic_exact_duplicate_audit_checks_values_masks_and_only_drop() -> None:
    frame = pd.DataFrame({f"other_{i}": np.arange(4, dtype=float) + 10 * i for i in range(22)})
    frame["rheobase_i"] = [1.0, np.nan, 3.0, 4.0]
    frame["stimulus_amplitude_0_long_square"] = [1.0, np.nan, 3.0, 4.0]
    features = [
        "rheobase_i",
        "stimulus_amplitude_0_long_square",
        *(f"other_{i}" for i in range(22)),
    ]
    report = rs.exact_duplicate_audit(
        frame,
        features,
        retain="rheobase_i",
        drop="stimulus_amplitude_0_long_square",
    )
    assert report["status"] == "PASS"
    assert report["original_feature_n"] == 24
    assert report["revised_feature_n"] == 23
    assert report["exact_pairs"] == [["rheobase_i", "stimulus_amplitude_0_long_square"]]
    assert report["pair_details"][0]["complete_pairs"] == 3
    assert report["pair_details"][0]["missing_count"] == 1
    assert report["revised_features"] == ["rheobase_i", *(f"other_{i}" for i in range(22))]

    unequal = frame.copy()
    unequal.loc[2, "stimulus_amplitude_0_long_square"] = 3.1
    with pytest.raises((AssertionError, ValueError)):
        rs.exact_duplicate_audit(
            unequal,
            features,
            retain="rheobase_i",
            drop="stimulus_amplitude_0_long_square",
        )

    mask_mismatch = frame.copy()
    mask_mismatch.loc[1, "stimulus_amplitude_0_long_square"] = 2.0
    with pytest.raises((AssertionError, ValueError)):
        rs.exact_duplicate_audit(
            mask_mismatch,
            features,
            retain="rheobase_i",
            drop="stimulus_amplitude_0_long_square",
        )


def test_synthetic_paired_bootstrap_is_reproducible_paired_and_oriented() -> None:
    # Each donor has two cells. A is perfect; B is deliberately worse.
    base = pd.DataFrame(
        {
            "canonical_cell_id": [f"c{i}" for i in range(8)],
            "group_id": np.repeat(["d0", "d1", "d2", "d3"], 2),
            "subclass": np.repeat(["S0", "S1", "S2", "S3"], 2),
            "fold": np.repeat([0, 1, 2, 0], 2),
            "module": "M",
            "y_true": np.arange(8, dtype=float),
        }
    )
    a = base.assign(model="A", y_pred=base.y_true)
    b = base.assign(model="B", y_pred=base.y_true.to_numpy()[::-1])
    draws_a, summary_a = rs.paired_bootstrap(
        a,
        b,
        n_resamples=97,
        seed=123,
        model_a="A",
        model_b="B",
        module="M",
    )
    draws_b, summary_b = rs.paired_bootstrap(
        a.sample(frac=1.0, random_state=9),
        b.sample(frac=1.0, random_state=11),
        n_resamples=97,
        seed=123,
        model_a="A",
        model_b="B",
        module="M",
    )
    pd.testing.assert_frame_equal(
        draws_a.sort_index(axis=1).reset_index(drop=True),
        draws_b.sort_index(axis=1).reset_index(drop=True),
    )
    assert summary_a == summary_b
    assert draws_a[_draw_column(draws_a)].nunique() == 97
    if "metric" in draws_a:
        for metric in METRICS:
            assert (draws_a.loc[draws_a.metric.eq(metric), "delta"] > 0).all()
    else:
        assert all((draws_a[metric] > 0).all() for metric in METRICS)

    # Pairing guard: the same cells may not carry different truth vectors.
    bad = b.copy()
    bad.loc[bad.index[0], "y_true"] += 0.25
    with pytest.raises((AssertionError, ValueError)):
        rs.paired_bootstrap(a, bad, n_resamples=5, seed=1)


def test_synthetic_calibration_recovers_known_line_and_tie_safe_decile() -> None:
    prediction = np.arange(20, dtype=float)
    frame = pd.DataFrame(
        {
            "canonical_cell_id": [f"c{i:02d}" for i in range(20)],
            "y_pred": prediction,
            "y_true": 2.0 + 1.5 * prediction,
        }
    )
    record = rs.calibration_metrics(frame)
    assert record["calibration_intercept"] == pytest.approx(2.0, abs=1e-12)
    assert record["calibration_slope"] == pytest.approx(1.5, abs=1e-12)
    assert record["predicted_sd"] / record["observed_sd"] == pytest.approx(2 / 3)
    assert record["predicted_range"] / record["observed_range"] == pytest.approx(2 / 3)
    assert record["top_decile_n"] == math.ceil(0.1 * len(frame)) == 2


def test_synthetic_correlation_components_use_absolute_strict_threshold() -> None:
    x = np.arange(12, dtype=float)
    frame = pd.DataFrame(
        {
            "a": x,
            "b": -x,  # abs(rho)=1, so connected to a
            "c": np.array([0, 2, 1, 4, 3, 7, 5, 9, 6, 11, 8, 10], dtype=float),
            "d": np.repeat([0.0, 1.0, 0.0], 4),
        }
    )
    edges, components = rs.correlation_components(
        frame, ["a", "b", "c", "d"], threshold=0.90
    )
    assert set(components.feature) == {"a", "b", "c", "d"}
    lookup = components.set_index("feature").component_id
    assert lookup["a"] == lookup["b"]
    assert lookup["a"] != lookup["d"]
    assert ((edges.feature_1.eq("a") & edges.feature_2.eq("b")) | (
        edges.feature_1.eq("b") & edges.feature_2.eq("a")
    )).any()
    assert (edges.abs_spearman_rho > 0.90).all()


def test_synthetic_frozen_fold_geometry_is_donor_disjoint() -> None:
    table = _modeling_table()
    folds = _folds()
    merged = table.drop(columns="fold", errors="ignore").merge(
        folds[["canonical_cell_id", "fold"]], on="canonical_cell_id", validate="one_to_one"
    )
    assert len(merged) == 3410
    assert merged.group_id.nunique() == 871
    assert merged.groupby("fold").size().sort_index().tolist() == [1137, 1136, 1137]
    for fold in (0, 1, 2):
        train = set(merged.loc[merged.fold.ne(fold), "group_id"])
        test = set(merged.loc[merged.fold.eq(fold), "group_id"])
        assert train.isdisjoint(test)


def test_synthetic_stage_cache_detects_byte_corruption(tmp_path: Path) -> None:
    artifact = tmp_path / "partition.parquet"
    artifact.write_bytes(b"valid-prediction-and-truth")
    cache = rs.StageCache(tmp_path / "cache.json", "synthetic-signature")
    cache.complete("partition", [artifact], fit_count=1)
    assert cache.valid("partition", [artifact])
    artifact.write_bytes(b"corrupt-prediction-and-truth")
    assert not cache.valid("partition", [artifact])
    reloaded = rs.StageCache(tmp_path / "cache.json", "synthetic-signature")
    assert not reloaded.valid("partition", [artifact])


def test_synthetic_control_partition_requires_trusted_digest_and_rejects_finite_corruption(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {"canonical_cell_id": ["c0", "c1"], "group_id": ["g0", "g1"],
         "subclass": ["Sst", "Vip"], "fold": [0, 1]}
    )
    target = pd.Series([1.0, 2.0], index=["c0", "c1"])
    configs = {0: rs.hash_text("cfg0"), 1: rs.hash_text("cfg1")}
    values = np.array([1.0, 2.0])
    frame = table.assign(
        module="M", control_id=0, y_true=values, y_pred=[1.1, 1.9],
        target_hash=rs.hash_array(values), run_signature="sig",
        configuration_sha256=[configs[0], configs[1]],
    )
    path = tmp_path / "part.parquet"
    frame.to_parquet(path, index=False)
    trusted = rs.sha256_file(path)
    kwargs = dict(
        expected_target=target, expected_table=table, module="M", control_id=0,
        signature="sig", folds=(0, 1), expected_configuration_sha256=configs,
    )
    assert rs._valid_control_partition(path, trusted_sha256=trusted, **kwargs)
    assert not rs._valid_control_partition(path, trusted_sha256=None, **kwargs)
    frame.loc[0, "y_pred"] = 999.0
    frame.to_parquet(path, index=False)
    assert not rs._valid_control_partition(path, trusted_sha256=trusted, **kwargs)
    frame.loc[0, "y_pred"] = 1.1
    frame.loc[0, "subclass"] = "Pvalb"
    frame.to_parquet(path, index=False)
    assert not rs._valid_control_partition(path, trusted_sha256=rs.sha256_file(path), **kwargs)
    frame.loc[0, "subclass"] = "Sst"
    frame.loc[0, "configuration_sha256"] = "bad"
    frame.to_parquet(path, index=False)
    assert not rs._valid_control_partition(path, trusted_sha256=rs.sha256_file(path), **kwargs)


# ---------------------------------------------------------------------------
# Strict full-run gates.


def test_full_mandatory_artifacts_exist_in_section2_namespaces() -> None:
    _require_files(
        RESULTS / "config/revised_23_features.json",
        RESULTS / "integrity/integrity_gate.json",
        RESULTS / "oof/revised_oof.parquet",
        RESULTS / "tables/pooled_metrics.csv",
        RESULTS / "tables/fold_metrics.csv",
        RESULTS / "tables/fitted_hyperparameters.csv",
        RESULTS / "tables/training_ledgers.csv",
        RESULTS / "tables/feature23_vs_feature24.csv",
        RESULTS / "model_comparison/paired_bootstrap_draws.parquet",
        RESULTS / "model_comparison/paired_model_comparison.csv",
        RESULTS / "model_comparison/paired_model_comparison_fdr.csv",
        RESULTS / "subclass/subclass_increment_draws.parquet",
        RESULTS / "subclass/subclass_increment_summary.csv",
        RESULTS / "within_subclass/eligibility.csv",
        RESULTS / "within_subclass/oof.parquet",
        RESULTS / "within_subclass/pooled_metrics.csv",
        RESULTS / "within_subclass/fold_metrics.csv",
        RESULTS / "within_subclass/bootstrap_draws.parquet",
        RESULTS / "permutations/cell_stratified.csv",
        RESULTS / "permutations/donor_residual.csv",
        RESULTS / "mlp/mlp_configuration.json",
        RESULTS / "mlp/learning_curves.parquet",
        RESULTS / "mlp/early_stopping_summary.csv",
        RESULTS / "repeated_cv/folds.csv",
        RESULTS / "repeated_cv/oof.parquet",
        RESULTS / "repeated_cv/summary.csv",
        RESULTS / "calibration/calibration.csv",
        RESULTS / "calibration/top_decile_enrichment.csv",
        RESULTS / "attribution/oof_shap_23feature.parquet",
        RESULTS / "attribution/shap_configuration.json",
        RESULTS / "attribution/correlation_edges.csv",
        RESULTS / "attribution/feature_components.csv",
        RESULTS / "attribution/grouped_shap.parquet",
        RESULTS / "attribution/en_coefficients.csv",
        RESULTS / "attribution/en_grouped_coefficients.csv",
        RESULTS / "attribution/concordance.csv",
        RESULTS / "specificity_bridge/random_control_performance.parquet",
        RESULTS / "specificity_bridge/specificity_summary.csv",
        RESULTS / "specificity_bridge/technical_target_performance.csv",
        RESULTS / "artifact_manifest.csv",
        LOGS / "full_run.json",
        LOGS / "resume_run.json",
        ROOT / "REVISION_V3_SECTION2_RESULTS.md",
    )
    allowed_roots = (RESULTS, FIGURES, LOGS, MODELS)
    manifest = pd.read_csv(RESULTS / "artifact_manifest.csv")
    for relative in manifest.path.astype(str):
        resolved = (ROOT / relative).resolve()
        assert any(_inside(resolved, root) for root in allowed_roots), relative


def test_full_frozen_inputs_and_section1_are_byte_unchanged() -> None:
    assert rs.sha256_file(ROOT / "results/predictions/oof_predictions.parquet") == (
        EXPECTED_PRIMARY_OOF_HASH
    )
    assert rs.sha256_file(ROOT / "results/shap/oof_shap.parquet") == (
        EXPECTED_PRIMARY_SHAP_HASH
    )
    section1_manifest = ROOT / "results/revision_v3/specificity/ARTIFACT_MANIFEST.csv"
    assert rs.sha256_file(section1_manifest) == EXPECTED_SECTION1_MANIFEST_HASH
    integrity = json.loads((RESULTS / "integrity/integrity_gate.json").read_text("utf-8"))
    assert integrity["status"] == "PASS"
    assert integrity["cohort_n"] == 3410
    assert integrity["donor_n"] == 871
    assert integrity["section1_manifest_valid"] is True


def test_full_revised_feature_definition_is_exact_and_hashed() -> None:
    original = load_feature_names(ROOT / "config/ephys_features.yaml")
    artifact = json.loads((RESULTS / "config/revised_23_features.json").read_text("utf-8"))
    features = artifact["features"]
    assert len(original) == 24
    assert len(features) == len(set(features)) == 23
    assert "rheobase_i" in features
    assert "stimulus_amplitude_0_long_square" not in features
    assert set(original) - set(features) == {"stimulus_amplitude_0_long_square"}
    assert set(features) - set(original) == set()
    saved_hash = artifact.get("sha256", artifact.get("revised_feature_sha256"))
    assert saved_hash == rs.hash_text("\n".join(features))

    table = _modeling_table()
    left = table.rheobase_i
    right = table.stimulus_amplitude_0_long_square
    assert np.array_equal(left.isna(), right.isna())
    complete = left.notna()
    assert np.array_equal(left[complete].to_numpy(), right[complete].to_numpy())


def test_full_revised_oof_grid_truth_folds_and_donors_are_exact() -> None:
    oof = pd.read_parquet(RESULTS / "oof/revised_oof.parquet")
    _assert_exact_oof_grid(oof)
    assert len(oof) == 3410 * 6 * 6
    table = _modeling_table().set_index("canonical_cell_id")
    fold_lookup = _folds().set_index("canonical_cell_id").fold
    ids = oof.canonical_cell_id.astype(str)
    assert np.array_equal(oof.fold.to_numpy(dtype=int), ids.map(fold_lookup).to_numpy())
    assert np.array_equal(oof.group_id.astype(str), ids.map(table.group_id).astype(str))
    for module in MODULE_ORDER:
        part = oof.loc[oof.module.eq(module)]
        expected_truth = ids[part.index].map(table[f"target_{module}"]).to_numpy(dtype=float)
        assert np.array_equal(part.y_true.to_numpy(dtype=float), expected_truth)
    assert oof.groupby(["module", "model"]).fold.nunique().eq(3).all()


def test_full_pooled_and_fold_metrics_reconstruct_exactly() -> None:
    oof = pd.read_parquet(RESULTS / "oof/revised_oof.parquet")
    pooled = pd.read_csv(RESULTS / "tables/pooled_metrics.csv")
    fold = pd.read_csv(RESULTS / "tables/fold_metrics.csv")
    assert len(pooled) == 36
    assert len(fold) == 108
    _assert_saved_metric_reconstruction(oof, pooled, ("module", "model"))
    _assert_saved_metric_reconstruction(oof, fold, ("module", "model", "fold"))
    assert fold.groupby(["module", "model"]).fold.nunique().eq(3).all()


def test_full_training_ledgers_prove_outer_and_inner_donor_separation() -> None:
    ledger = pd.read_csv(RESULTS / "tables/training_ledgers.csv")
    required = {
        "module",
        "model",
        "fold",
        "train_n",
        "test_n",
        "train_group_n",
        "test_group_n",
        "outer_group_overlap_n",
        "train_id_sha256",
        "test_id_sha256",
    }
    assert required.issubset(ledger.columns)
    assert ledger.outer_group_overlap_n.eq(0).all()
    assert ledger.fold.isin([0, 1, 2]).all()
    assert ledger.train_n.add(ledger.test_n).eq(3410).all()
    tuning = ledger.loc[ledger.model.isin(PRIMARY_MODELS)]
    assert {"inner_train_group_n", "inner_valid_group_n", "inner_group_overlap_n"}.issubset(
        tuning.columns
    )
    assert tuning.inner_group_overlap_n.eq(0).all()
    assert tuning.inner_train_group_n.gt(0).all()
    assert tuning.inner_valid_group_n.gt(0).all()


def test_full_primary_model_partitions_are_all_persisted() -> None:
    primary_root = MODELS / "primary"
    assert primary_root.is_dir()
    saved = list(primary_root.rglob("*.joblib"))
    # EN/XGB/comparators: 5 families x 6 modules x 3 folds = 90.
    # MLP: 3 seeds x 6 modules x 3 folds = 54.  Total = 144.
    assert len(saved) == 144
    lowered = [str(path.relative_to(primary_root)).lower() for path in saved]
    for module in MODULE_ORDER:
        assert sum(module.lower() in name for name in lowered) == 24


def test_full_feature23_vs_24_is_complete_same_cell_sensitivity() -> None:
    sensitivity = pd.read_csv(RESULTS / "tables/feature23_vs_feature24.csv")
    assert len(sensitivity) == 18
    expected = {(m, model) for m in MODULE_ORDER for model in PRIMARY_MODELS}
    observed = set(
        sensitivity[["module", _model_column(sensitivity)]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    assert observed == expected
    for metric in METRICS:
        assert f"feature23_{metric}" in sensitivity
        assert f"feature24_{metric}" in sensitivity
        assert f"delta_{metric}" in sensitivity


def test_full_pairwise_bootstrap_has_exact_predeclared_grid_and_fdr_family() -> None:
    draws = pd.read_parquet(
        RESULTS / "model_comparison/paired_bootstrap_draws.parquet"
    )
    summary = pd.read_csv(RESULTS / "model_comparison/paired_model_comparison.csv")
    fdr = pd.read_csv(RESULTS / "model_comparison/paired_model_comparison_fdr.csv")
    required_pairs = {(m, a, b) for m in MODULE_ORDER for a, b in CONTRASTS}
    assert set(
        summary[["module", "model_a", "model_b"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ) == required_pairs
    assert "best classical" not in " ".join(summary.astype(str).to_numpy().ravel()).lower()
    assert len(summary) == 18 * 4
    assert set(summary.metric) == set(METRICS)
    assert summary.n_resamples.eq(5000).all()
    assert (summary.ci_low <= summary.point_delta).all()
    assert (summary.point_delta <= summary.ci_high).all()

    draw_id = _draw_column(draws)
    if "metric" in draws:
        assert draws.groupby(["module", "model_a", "model_b", "metric"])[draw_id].nunique().eq(5000).all()
    else:
        assert draws.groupby(["module", "model_a", "model_b"])[draw_id].nunique().eq(5000).all()
        assert set(METRICS).issubset(draws.columns)

    assert len(fdr) == 18
    assert set(fdr.metric) == {"r2"}
    assert set(
        fdr[["module", "model_a", "model_b"]].itertuples(index=False, name=None)
    ) == required_pairs
    assert fdr.q_value.between(0, 1).all()
    np.testing.assert_allclose(
        fdr.q_value.to_numpy(dtype=float),
        rs.bh_adjust(fdr.p_two_sided.to_numpy(dtype=float)),
        rtol=0,
        atol=1e-15,
    )


def test_full_subclass_increment_is_paired_complete_and_oriented() -> None:
    draws = pd.read_parquet(RESULTS / "subclass/subclass_increment_draws.parquet")
    summary = pd.read_csv(RESULTS / "subclass/subclass_increment_summary.csv")
    assert len(summary) == 6 * 4
    assert set(summary.module) == set(MODULE_ORDER)
    assert set(summary.metric) == set(METRICS)
    assert summary.n_resamples.eq(5000).all()
    assert set(summary.model_a) == {"EphysSubclassRidge"}
    assert set(summary.model_b) == {"SubclassRidge"}
    draw_id = _draw_column(draws)
    if "metric" in draws:
        assert draws.groupby(["module", "metric"])[draw_id].nunique().eq(5000).all()
    else:
        assert draws.groupby("module")[draw_id].nunique().eq(5000).all()


def test_full_within_subclass_eligibility_and_complete_grid() -> None:
    eligibility = pd.read_csv(RESULTS / "within_subclass/eligibility.csv")
    expected_eligible = {"Sst", "Pvalb", "Vip", "Lamp5"}
    assert set(eligibility.subclass) == set(_modeling_table().subclass.unique())
    assert set(eligibility.loc[eligibility.eligible, "subclass"]) == expected_eligible
    assert eligibility.loc[eligibility.eligible, "n_cells"].ge(150).all()
    assert eligibility.loc[eligibility.eligible, "n_donors"].ge(3).all()

    oof = pd.read_parquet(RESULTS / "within_subclass/oof.parquet")
    expected_grid = {
        (subclass, module, model)
        for subclass in expected_eligible
        for module in MODULE_ORDER
        for model in ("ElasticNet", "XGBoost")
    }
    assert set(
        oof[["subclass", "module", "model"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ) == expected_grid
    assert not oof.duplicated(["canonical_cell_id", "subclass", "module", "model"]).any()
    cohort = _modeling_table()
    for key, part in oof.groupby(["subclass", "module", "model"]):
        subclass, _, _ = key
        expected_ids = set(cohort.loc[cohort.subclass.eq(subclass), "canonical_cell_id"])
        assert set(part.canonical_cell_id.astype(str)) == expected_ids
        assert part.fold.nunique() == 3
        donor_folds = part[["group_id", "fold"]].drop_duplicates()
        assert donor_folds.groupby("group_id").fold.nunique().eq(1).all()

    pooled = pd.read_csv(RESULTS / "within_subclass/pooled_metrics.csv")
    fold = pd.read_csv(RESULTS / "within_subclass/fold_metrics.csv")
    assert len(pooled) == len(expected_grid)
    assert len(fold) == len(expected_grid) * 3
    _assert_saved_metric_reconstruction(oof, pooled, ("subclass", "module", "model"))
    _assert_saved_metric_reconstruction(oof, fold, ("subclass", "module", "model", "fold"))


@pytest.mark.parametrize("filename", ["cell_stratified.csv", "donor_residual.csv"])
def test_full_permutation_families_are_complete_plus_one(filename: str) -> None:
    frame = pd.read_csv(RESULTS / "permutations" / filename)
    expected = {(m, model) for m in MODULE_ORDER for model in PRIMARY_MODELS}
    assert set(
        frame[["module", _model_column(frame)]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ) == expected
    assert len(frame) == 18
    assert frame.n_permutations.eq(5000).all()
    p_column = _permutation_p_column(frame)
    assert frame[p_column].between(1 / 5001 - 1e-12, 1.0).all()
    if "extreme_count" in frame:
        expected_p = (frame.extreme_count.to_numpy(dtype=float) + 1.0) / 5001.0
        np.testing.assert_allclose(frame[p_column], expected_p, rtol=0, atol=1e-15)


def test_full_mlp_configuration_curves_and_inner_split_audit() -> None:
    config = json.loads((RESULTS / "mlp/mlp_configuration.json").read_text("utf-8"))
    assert tuple(config["hidden_layer_sizes"]) == (64, 32)
    assert config["activation"] == "relu"
    assert config["solver"] == "adam"
    assert tuple(config["ensemble_seeds"]) == MLP_SEEDS
    assert config["patience"] == 20
    assert config["min_delta"] == pytest.approx(1e-5)
    assert config["inner_validation_grouped"] is True

    curves = pd.read_parquet(RESULTS / "mlp/learning_curves.parquet")
    required = {
        "module",
        "outer_fold",
        "seed",
        "epoch",
        "train_loss",
        "validation_loss",
    }
    assert required.issubset(curves.columns)
    assert "test_loss" not in curves.columns
    assert set(curves.module) == set(MODULE_ORDER)
    assert set(curves.outer_fold) == {0, 1, 2}
    assert set(curves.seed) == set(MLP_SEEDS)
    assert np.isfinite(curves[["train_loss", "validation_loss"]]).all().all()

    stopping = pd.read_csv(RESULTS / "mlp/early_stopping_summary.csv")
    assert len(stopping) == 6 * 3 * 3
    assert stopping.inner_group_overlap_n.eq(0).all()
    assert stopping.selected_epoch.ge(1).all()
    max_epoch = curves.groupby(["module", "outer_fold", "seed"]).epoch.max()
    selected = stopping.set_index(["module", "outer_fold", "seed"]).selected_epoch.sort_index()
    assert selected.index.equals(max_epoch.index)
    assert (selected <= max_epoch).all()


def test_full_repeated_cv_has_exact_seeds_complete_oof_and_no_donor_leakage() -> None:
    folds = pd.read_csv(RESULTS / "repeated_cv/folds.csv", dtype={"canonical_cell_id": str})
    assert set(folds.repeat_seed.astype(int)) == set(REPEAT_SEEDS)
    assert not folds.duplicated(["repeat_seed", "canonical_cell_id"]).any()
    assert folds.groupby("repeat_seed").size().eq(3410).all()
    merged = folds.merge(
        _modeling_table()[["canonical_cell_id", "group_id"]],
        on="canonical_cell_id",
        validate="many_to_one",
        suffixes=("", "_frozen"),
    )
    assert merged.groupby(["repeat_seed", "group_id"]).fold.nunique().eq(1).all()

    oof = pd.read_parquet(RESULTS / "repeated_cv/oof.parquet")
    assert set(oof.repeat_seed.astype(int)) == set(REPEAT_SEEDS)
    assert set(oof.model) == set(PRIMARY_MODELS)
    assert not oof.duplicated(
        ["repeat_seed", "canonical_cell_id", "module", "model"]
    ).any()
    assert oof.groupby(["repeat_seed", "module", "model"]).size().eq(3410).all()
    assert len(oof) == 5 * 6 * 3 * 3410
    expected_fold = folds.set_index(["repeat_seed", "canonical_cell_id"]).fold
    index = pd.MultiIndex.from_frame(oof[["repeat_seed", "canonical_cell_id"]])
    assert np.array_equal(oof.fold.to_numpy(dtype=int), expected_fold.reindex(index).to_numpy())
    summary = pd.read_csv(RESULTS / "repeated_cv/summary.csv")
    assert len(summary) == 6 * 3 * 4
    assert set(summary.metric) == set(METRICS)
    assert summary.n_repeats.eq(5).all()


def test_full_calibration_and_top_decile_reconstruct() -> None:
    oof = pd.read_parquet(RESULTS / "oof/revised_oof.parquet")
    saved = pd.read_csv(RESULTS / "calibration/calibration.csv")
    top = pd.read_csv(RESULTS / "calibration/top_decile_enrichment.csv")
    assert len(saved) == len(top) == 18
    expected_grid = {(m, model) for m in MODULE_ORDER for model in PRIMARY_MODELS}
    assert set(saved[["module", "model"]].itertuples(index=False, name=None)) == expected_grid
    indexed = saved.set_index(["module", "model"])
    for key, part in oof.loc[oof.model.isin(PRIMARY_MODELS)].groupby(["module", "model"]):
        record = rs.calibration_metrics(part)
        for column in (
            "intercept",
            "slope",
            "prediction_sd",
            "truth_sd",
            "prediction_range",
            "truth_range",
        ):
            assert indexed.loc[key, column] == pytest.approx(record[column], rel=1e-10, abs=1e-12)
    assert top.top_decile_n.eq(math.ceil(3410 * 0.1)).all()


def test_full_shap_is_heldout_complete_and_configuration_is_observed() -> None:
    shap = pd.read_parquet(RESULTS / "attribution/oof_shap_23feature.parquet")
    config = json.loads((RESULTS / "attribution/shap_configuration.json").read_text("utf-8"))
    feature_artifact = json.loads(
        (RESULTS / "config/revised_23_features.json").read_text("utf-8")
    )
    features = set(feature_artifact["features"])
    required = {"canonical_cell_id", "module", "fold", "feature", "shap_value"}
    assert required.issubset(shap.columns)
    assert set(shap.feature) == features
    assert len(shap) == 6 * 3410 * 23
    assert not shap.duplicated(["canonical_cell_id", "module", "feature"]).any()
    assert shap.groupby(["module", "feature"]).size().eq(3410).all()
    fold_lookup = _folds().set_index("canonical_cell_id").fold
    expected = shap.canonical_cell_id.astype(str).map(fold_lookup)
    assert np.array_equal(shap.fold.to_numpy(dtype=int), expected.to_numpy(dtype=int))
    assert config["held_out_only"] is True
    assert config["explainer_class"] == "TreeExplainer"
    assert isinstance(config["feature_perturbation"], str)
    assert config["feature_perturbation"] not in {"", "unknown", "assumed", "null"}


def test_full_grouped_shap_and_en_coefficients_reconcile_exactly() -> None:
    shap = pd.read_parquet(RESULTS / "attribution/oof_shap_23feature.parquet")
    components = pd.read_csv(RESULTS / "attribution/feature_components.csv")
    grouped = pd.read_parquet(RESULTS / "attribution/grouped_shap.parquet")
    assert len(components) == 23
    assert components.feature.is_unique
    expected = (
        shap.assign(abs_value=shap.shap_value.abs())
        .merge(components[["feature", "component_id"]], on="feature", validate="many_to_one")
        .groupby(["canonical_cell_id", "module", "fold", "component_id"], as_index=False)
        .abs_value.sum()
        .rename(columns={"abs_value": "group_abs_shap"})
    )
    observed = grouped[
        ["canonical_cell_id", "module", "fold", "component_id", "group_abs_shap"]
    ]
    merged = expected.merge(
        observed,
        on=["canonical_cell_id", "module", "fold", "component_id"],
        how="outer",
        validate="one_to_one",
        suffixes=("_expected", "_saved"),
        indicator=True,
    )
    assert merged._merge.eq("both").all()
    np.testing.assert_allclose(
        merged.group_abs_shap_expected,
        merged.group_abs_shap_saved,
        rtol=1e-12,
        atol=1e-12,
    )

    coefficients = pd.read_csv(RESULTS / "attribution/en_coefficients.csv")
    en_grouped = pd.read_csv(RESULTS / "attribution/en_grouped_coefficients.csv")
    assert len(coefficients) == 6 * 3 * 23
    assert coefficients.groupby(["module", "fold"]).feature.nunique().eq(23).all()
    assert coefficients.standardized_input.eq(True).all()
    expected_en = (
        coefficients.assign(abs_coefficient=coefficients.coefficient.abs())
        .merge(components[["feature", "component_id"]], on="feature", validate="many_to_one")
        .groupby(["module", "fold", "component_id"], as_index=False)
        .abs_coefficient.sum()
    )
    observed_en = en_grouped[["module", "fold", "component_id", "abs_coefficient"]]
    merged_en = expected_en.merge(
        observed_en,
        on=["module", "fold", "component_id"],
        validate="one_to_one",
        suffixes=("_expected", "_saved"),
    )
    np.testing.assert_allclose(
        merged_en.abs_coefficient_expected,
        merged_en.abs_coefficient_saved,
        rtol=1e-12,
        atol=1e-12,
    )


def test_full_attribution_concordance_is_all_module_complete() -> None:
    concordance = pd.read_csv(RESULTS / "attribution/concordance.csv")
    assert len(concordance) == 6
    assert set(concordance.module) == set(MODULE_ORDER)
    assert concordance.group_spearman_rho.between(-1, 1).all()
    for k in (3, 5):
        assert concordance[f"top{k}_overlap_n"].between(0, k).all()
        assert concordance[f"top{k}_jaccard"].between(0, 1).all()


def test_full_specificity_bridge_reuses_all_frozen_controls_and_p_reconstructs() -> None:
    bridge = pd.read_parquet(
        RESULTS / "specificity_bridge/random_control_performance.parquet"
    )
    summary = pd.read_csv(RESULTS / "specificity_bridge/specificity_summary.csv")
    frozen_sets = json.loads(
        (
            ROOT
            / "results/revision_v3/specificity/random_controls/matched_gene_sets.json"
        ).read_text("utf-8")
    )
    assert len(bridge) == 1200
    assert bridge.groupby("module").control_id.nunique().eq(200).all()
    assert set(bridge.module) == set(MODULE_ORDER)
    for module in MODULE_ORDER:
        expected = {
            str(item["control_id"])
            for item in frozen_sets["modules"][module]["controls"]
        }
        observed = set(bridge.loc[bridge.module.eq(module), "control_id"].astype(str))
        assert observed == expected
    assert bridge.control_tuned.eq(False).all()
    assert len(summary) == 6
    assert set(summary.module) == set(MODULE_ORDER)
    if {"extreme_count", "empirical_p"}.issubset(summary.columns):
        expected_p = (summary.extreme_count.to_numpy(dtype=float) + 1.0) / 201.0
        np.testing.assert_allclose(summary.empirical_p, expected_p, rtol=0, atol=1e-15)

    technical = pd.read_csv(
        RESULTS / "specificity_bridge/technical_target_performance.csv"
    )
    assert set(technical.model) == {"ElasticNet", "XGBoost"}
    assert technical.target.nunique() >= 2
    assert technical.groupby(["target", "model"]).size().eq(1).all()


def test_full_figures_have_pdf_and_300dpi_png_variants() -> None:
    stems = (
        "figure_a_revised_model_performance",
        "figure_b_paired_model_comparison",
        "figure_c_subclass_increment",
        "figure_d_model_stability",
        "figure_e_calibration",
    )
    paths = [FIGURES / f"{stem}.{suffix}" for stem in stems for suffix in ("pdf", "png")]
    _require_files(*paths)
    from PIL import Image

    for stem in stems:
        with Image.open(FIGURES / f"{stem}.png") as image:
            dpi = image.info.get("dpi", (0, 0))
            assert min(dpi) >= 299.0, (stem, dpi)


def test_full_manifest_validates_bytes_hashes_and_exact_namespace_coverage() -> None:
    manifest_path = RESULTS / "artifact_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    assert {"path", "bytes", "sha256"}.issubset(manifest.columns)
    assert manifest.path.is_unique
    allowed_roots = (RESULTS, FIGURES, LOGS, MODELS)
    for row in manifest.itertuples(index=False):
        path = (ROOT / str(row.path)).resolve()
        assert any(_inside(path, root) for root in allowed_roots), row.path
        assert path.is_file(), row.path
        assert path.stat().st_size == int(row.bytes), row.path
        assert rs.sha256_file(path) == row.sha256, row.path

    declared = {str((ROOT / path).resolve()) for path in manifest.path.astype(str)}
    actual = {
        str(path.resolve())
        for root in allowed_roots
        for path in root.rglob("*")
        if path.is_file()
        and path != manifest_path
        and path.name != ".gitkeep"
        and "checkpoints" not in path.parts
    }
    assert declared == actual


def test_full_resume_reuses_every_stage_and_does_not_refit() -> None:
    full = json.loads((LOGS / "full_run.json").read_text("utf-8"))
    resume = json.loads((LOGS / "resume_run.json").read_text("utf-8"))
    assert full["status"] == "PASS"
    assert resume["status"] == "PASS"
    assert resume["signature"] == full["signature"]
    assert resume["fit_count"] == 0
    assert resume["reused_partition_count"] > 0
    assert set(resume["completed_stages"]) == set(full["completed_stages"])
    assert resume["corrupt_cache_truth_rejected"] is True


def test_full_report_and_manuscript_freeze_are_closed() -> None:
    report = (ROOT / "REVISION_V3_SECTION2_RESULTS.md").read_text("utf-8")
    assert "# Claims Supported" in report
    assert "# Claims Not Supported" in report
    assert "# Reproduction Commands" in report
    integrity = json.loads((RESULTS / "integrity/integrity_gate.json").read_text("utf-8"))
    assert rs.sha256_file(ROOT / "MANUSCRIPT.md") == integrity["manuscript_sha256_before"]
