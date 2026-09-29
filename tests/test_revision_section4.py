from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.revision_section4 import (
    MODULE_ORDER,
    _rank_correlation,
    _select_retuning_ids,
    _validate_control_partition,
    add_bh_family,
    balanced_partitions,
    bh_adjust,
    configuration_sha256,
    consistency_summary,
    hash_array,
    read_config,
    specificity_summary,
    stable_seed,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/revision_v3_section4.yaml"


def test_empirical_plus_one_and_exact_six_bh() -> None:
    controls = pd.DataFrame(
        {
            "module": np.repeat(MODULE_ORDER, 1000),
            "control_id": np.tile(np.arange(1000), 6),
            "r2": np.tile(np.linspace(-1, 1, 1000), 6),
        }
    )
    observed = pd.DataFrame(
        {"module": MODULE_ORDER, "model": "XGBoost", "r2": [0, .1, .2, .3, .4, .5]}
    )
    summary = specificity_summary(controls, observed, p_column="raw_p")
    for row in summary.itertuples(index=False):
        values = controls.loc[controls.module.eq(row.module), "r2"].to_numpy()
        expected = (1 + np.count_nonzero(values >= row.observed_xgboost_r2)) / 1001
        assert row.raw_p == expected
    adjusted = add_bh_family(
        summary, p_column="raw_p", q=.05, family_label="unit-test-six"
    )
    np.testing.assert_array_equal(adjusted.bh_q, bh_adjust(summary.raw_p.to_numpy()))
    assert adjusted.family_n.eq(6).all()


def test_balanced_partition_counts_and_determinism() -> None:
    seven_a = balanced_partitions(7, family="NaV7", control_id=0, maximum_draws=100, seed=42)
    seven_b = balanced_partitions(7, family="NaV7", control_id=0, maximum_draws=100, seed=42)
    assert seven_a == seven_b
    assert len(seven_a[0]) == 35
    assert all(len(left) == 3 and len(right) == 4 for left, right in seven_a[0])
    large = balanced_partitions(19, family="GABAA", control_id=4, maximum_draws=100, seed=42)
    assert len(large[0]) == 100
    assert len(set(large[0])) == 100


def test_spearman_brown_summary_formula() -> None:
    rng = np.random.default_rng(7)
    expression = pd.DataFrame(rng.normal(size=(120, 7)), columns=list("abcdefg"))
    summary = consistency_summary(
        expression,
        list("abcdefg"),
        family="synthetic",
        control_id=3,
        maximum_draws=100,
        seed=42,
    )
    assert summary["partition_n"] == 35
    partitions, _, _ = balanced_partitions(
        7, family="synthetic", control_id=3, maximum_draws=100, seed=42
    )
    rho = []
    for left, right in partitions:
        value = _rank_correlation(
            expression.iloc[:, list(left)].mean(axis=1).to_numpy(),
            expression.iloc[:, list(right)].mean(axis=1).to_numpy(),
        )
        rho.append(2 * value / (1 + value))
    assert summary["median_spearman_brown"] == float(np.median(rho))


def test_retuning_subset_is_frozen_before_outcomes() -> None:
    _, config, _ = read_config(CONFIG)
    first = _select_retuning_ids(config)
    second = _select_retuning_ids(config)
    assert first == second
    assert set(first) == set(MODULE_ORDER)
    assert all(len(ids) == 50 and len(set(ids)) == 50 for ids in first.values())
    assert all(min(ids) >= 0 and max(ids) < 1000 for ids in first.values())


def test_configuration_hash_binds_static_fields() -> None:
    base = {
        "n_estimators": 250,
        "max_depth": 2,
        "learning_rate": .05,
        "reg_lambda": 1.,
        "subsample": .8,
        "colsample_bytree": .8,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": 1,
        "random_state": 42,
    }
    changed = dict(base, subsample=.9)
    assert configuration_sha256(base) != configuration_sha256(changed)


def test_section4_seed_namespace_is_stable() -> None:
    expected = int.from_bytes(
        hashlib.sha256(b"revision-v3-section4-1|42|retuning_subset|NaV").digest()[:4],
        "big",
    )
    assert stable_seed("retuning_subset", "NaV", base=42) == expected


def _synthetic_partition() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str, str]:
    table = pd.DataFrame(
        {
            "canonical_cell_id": ["a", "b", "c"],
            "group_id": ["g1", "g2", "g3"],
            "subclass": ["x", "y", "z"],
            "fold": [0, 0, 1],
        }
    )
    truth = pd.Series([1., 2.], index=["a", "b"])
    target_hash = hash_array(np.array([1., 2.]))
    config_hash = "configuration"
    frame = pd.DataFrame(
        {
            "canonical_cell_id": ["a", "b"],
            "group_id": ["g1", "g2"],
            "subclass": ["x", "y"],
            "fold": [0, 0],
            "module": ["NaV", "NaV"],
            "control_id": [2, 2],
            "y_true": [1., 2.],
            "y_pred": [1.1, 1.9],
            "target_hash": [target_hash, target_hash],
            "run_signature": ["signature", "signature"],
            "configuration_sha256": [config_hash, config_hash],
        }
    )
    return table, frame, truth, target_hash, config_hash


def test_partition_validator_rejects_truth_config_and_digest_corruption(tmp_path: Path) -> None:
    table, frame, truth, _, config_hash = _synthetic_partition()
    path = tmp_path / "part.parquet"
    frame.to_parquet(path, index=False)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    def validate(current_digest: str):
        return _validate_control_partition(
            path,
            trusted_sha256=current_digest,
            table=table,
            target_by_cell=truth,
            module="NaV",
            control_id=2,
            folds=(0,),
            expected_config_hashes={0: config_hash},
            accepted_signatures={"signature"},
        )

    assert validate(digest) is not None
    assert validate("0" * 64) is None
    corrupted = frame.copy()
    corrupted.loc[0, "y_true"] += 1
    corrupted.to_parquet(path, index=False)
    assert validate(hashlib.sha256(path.read_bytes()).hexdigest()) is None
    corrupted = frame.copy()
    corrupted["configuration_sha256"] = "wrong"
    corrupted.to_parquet(path, index=False)
    assert validate(hashlib.sha256(path.read_bytes()).hexdigest()) is None
