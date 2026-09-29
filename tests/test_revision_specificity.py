"""Specificity analysis specificity unit tests and paper-grade artifact gates.

The ``test_synthetic_*`` tests are intentionally independent of a completed
pipeline and form the fast smoke/collection target::

    python -m pytest -q tests/test_revision_specificity.py -k synthetic

The ``test_full_artifacts_*`` tests are strict post-run gates.  Missing full
artifacts are failures, not skips, because those files are part of the stated
definition of done.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import math
import re
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import src.revision_specificity as rs
from src.biology import load_gene_modules
from src.data import load_public_metadata, sha256
from src.evaluation import regression_metrics
from src.training import MODULES


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config/revision_v3_specificity.yaml"
ROOT_FROM_CONFIG, CONFIG = rs.read_config(CONFIG_PATH)
OUTPUTS = rs.output_roots(ROOT, CONFIG)
RESULTS = OUTPUTS["results"]
FIGURES = OUTPUTS["figures"]
LOGS = OUTPUTS["logs"]
MODELS = OUTPUTS["models"]
MODULE_ORDER = tuple(MODULES)
MODEL_ORDER = ("ElasticNet", "XGBoost")


def _require_files(*paths: Path) -> None:
    missing = [str(path.relative_to(ROOT)) for path in paths if not path.is_file()]
    empty = [
        str(path.relative_to(ROOT))
        for path in paths
        if path.is_file() and path.stat().st_size == 0
    ]
    assert not missing, "Required full-run artifacts are missing: " + ", ".join(missing)
    assert not empty, "Required full-run artifacts are empty: " + ", ".join(empty)


def _modeling_table() -> pd.DataFrame:
    table, _ = rs.load_analysis_table(ROOT, CONFIG)
    table = table.copy()
    table["canonical_cell_id"] = table.canonical_cell_id.astype(str)
    table["group_id"] = table.group_id.astype(str)
    table["transcriptomics_sample_id"] = table.transcriptomics_sample_id.astype(str)
    return table


def _assert_fold_mapping(frame: pd.DataFrame, fold_table: pd.DataFrame) -> None:
    lookup = fold_table.copy()
    lookup["canonical_cell_id"] = lookup.canonical_cell_id.astype(str)
    lookup = lookup.set_index("canonical_cell_id", verify_integrity=True).fold.astype(int)
    ids = frame.canonical_cell_id.astype(str)
    expected = ids.map(lookup)
    assert expected.notna().all()
    assert np.array_equal(frame.fold.to_numpy(dtype=int), expected.to_numpy(dtype=int))


def _assert_complete_oof(
    frame: pd.DataFrame,
    expected_ids: set[str],
    key_columns: tuple[str, ...],
    expected_keys: set[tuple[object, ...]],
) -> None:
    assert set(frame.columns).issuperset(
        {"canonical_cell_id", "fold", "y_true", "y_pred", *key_columns}
    )
    assert np.isfinite(frame[["y_true", "y_pred"]].to_numpy(dtype=float)).all()
    observed_keys = set(
        frame[list(key_columns)].drop_duplicates().itertuples(index=False, name=None)
    )
    assert observed_keys == expected_keys
    assert not frame.duplicated(["canonical_cell_id", *key_columns]).any()
    for keys, part in frame.groupby(list(key_columns), sort=False):
        del keys
        ids = part.canonical_cell_id.astype(str)
        assert ids.is_unique
        assert set(ids) == expected_ids


def _assert_leakage_ledger(ledger: pd.DataFrame, cohort: pd.DataFrame) -> None:
    required = {
        "fold",
        "train_n",
        "test_n",
        "train_group_n",
        "test_group_n",
        "group_overlap_n",
        "train_id_sha256",
        "test_id_sha256",
    }
    assert required.issubset(ledger.columns)
    cohort = cohort.copy()
    cohort["canonical_cell_id"] = cohort.canonical_cell_id.astype(str)
    cohort["group_id"] = cohort.group_id.astype(str)
    for row in ledger.itertuples(index=False):
        train = cohort.loc[cohort.fold.ne(int(row.fold))]
        test = cohort.loc[cohort.fold.eq(int(row.fold))]
        train_groups = set(train.group_id)
        test_groups = set(test.group_id)
        assert train_groups.isdisjoint(test_groups)
        assert row.group_overlap_n == 0
        assert row.train_n == len(train)
        assert row.test_n == len(test)
        assert row.train_group_n == len(train_groups)
        assert row.test_group_n == len(test_groups)
        train_ids = "\n".join(sorted(train.canonical_cell_id))
        test_ids = "\n".join(sorted(test.canonical_cell_id))
        assert row.train_id_sha256 == rs.hash_text(train_ids)
        assert row.test_id_sha256 == rs.hash_text(test_ids)


def _assert_metrics_reconstruct(
    oof: pd.DataFrame, saved: pd.DataFrame, key_columns: tuple[str, ...]
) -> None:
    indexed = saved.set_index(list(key_columns))
    for keys, frame in oof.groupby(list(key_columns), sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        calculated = regression_metrics(frame.y_true.to_numpy(), frame.y_pred.to_numpy())
        for metric, value in calculated.items():
            assert np.isclose(
                value,
                indexed.loc[keys, metric],
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ), (keys, metric)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Fast synthetic contracts (safe before the computational pipeline is run).


def test_synthetic_stable_seed_and_hash_contracts() -> None:
    payload = "|".join([rs.PIPELINE_VERSION, "42", "matched_control", "NaV", "7"])
    expected = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")
    assert rs.stable_seed("matched_control", "NaV", 7, base=42) == expected
    assert rs.stable_seed("matched_control", "NaV", 7, base=42) == expected
    assert rs.stable_seed("matched_control", "NaV", 8, base=42) != expected
    assert 0 <= expected < 2**32

    vector = np.array([1.0, 2.0], dtype=np.float64)
    assert rs.hash_array(vector) == rs.hash_array(vector.copy())
    assert rs.hash_array(vector) != rs.hash_array(vector.astype(np.float32))
    assert rs.hash_array(vector) != rs.hash_array(vector.reshape(1, 2))
    assert rs.hash_text("a\nb") != rs.hash_text("ab")


def test_synthetic_module_partitions_are_deterministic_balanced_and_swap_unique() -> None:
    genes = [f"g{index}" for index in range(6)]
    partitions, method, maximum = rs.module_partitions(
        genes, "CaV", draws=1000, seed=42
    )
    assert method == "exhaustive"
    assert maximum == math.comb(6, 3) // 2 == 10
    assert len(partitions) == maximum
    assert len(set(partitions)) == maximum
    for left, right in partitions:
        assert len(left) == len(right) == 3
        assert set(left).isdisjoint(right)
        assert set(left) | set(right) == set(range(6))
        assert left <= right  # canonical representative of the swap pair

    random_a = rs.module_partitions([f"g{i}" for i in range(8)], "Kv", draws=7, seed=42)
    random_b = rs.module_partitions([f"g{i}" for i in range(8)], "Kv", draws=7, seed=42)
    assert random_a == random_b
    assert random_a[1] == "deterministic_random_unique"
    assert len(random_a[0]) == len(set(random_a[0])) == 7

    odd, odd_method, odd_maximum = rs.module_partitions(
        [f"g{i}" for i in range(5)], "HCN", draws=2, seed=42
    )
    assert odd_method == "exhaustive"
    assert odd_maximum == math.comb(5, 2) == len(odd)
    assert all(len(left) == 2 and len(right) == 3 for left, right in odd)


def test_synthetic_matching_is_deterministic_unique_and_excludes_targets() -> None:
    rows = [
        {"gene_symbol": "T1", "mean_log2cpm": -0.5, "detection_rate": 0.15,
         "candidate_eligible": False},
        {"gene_symbol": "T2", "mean_log2cpm": 0.5, "detection_rate": 0.85,
         "candidate_eligible": False},
    ]
    rows.extend(
        {
            "gene_symbol": f"C{index}",
            "mean_log2cpm": float(index - 4),
            "detection_rate": 0.1 * index,
            "candidate_eligible": True,
        }
        for index in range(1, 9)
    )
    gene_qc = pd.DataFrame(rows)
    settings = {
        "max_per_gene_standardized_distance": 10.0,
        "nearest_neighbor_pool": 8,
        "maximum_attempts_per_set": 1000,
        "max_mean_per_gene_standardized_distance": 10.0,
        "max_module_abs_standardized_mean_difference": 10.0,
        "max_module_abs_standardized_detection_difference": 10.0,
    }
    args = dict(
        gene_qc=gene_qc,
        modules={"Synthetic": ["T1", "T2"]},
        control_n=4,
        settings=settings,
        seed=42,
        selected_modules=["Synthetic"],
    )
    matched_a, quality_a, assignments_a = rs.generate_matched_gene_sets(**args)
    matched_b, quality_b, assignments_b = rs.generate_matched_gene_sets(**args)
    assert matched_a == matched_b
    pd.testing.assert_frame_equal(quality_a, quality_b)
    pd.testing.assert_frame_equal(assignments_a, assignments_b)

    controls = matched_a["modules"]["Synthetic"]["controls"]
    assert len(controls) == 4
    assert quality_a.matching_pass.all()
    assert not quality_a.duplicated(["module", "set_signature"]).any()
    for control in controls:
        assert len(control["genes"]) == len(set(control["genes"])) == 2
        assert set(control["genes"]).isdisjoint({"T1", "T2"})
        expected_signature = rs.hash_text("\n".join(sorted(control["genes"])))
        assert control["set_signature"] == expected_signature
    grouped = assignments_a.groupby(["module", "control_id"])
    assert grouped.target_gene.nunique().eq(2).all()
    assert grouped.control_gene.nunique().eq(2).all()


def test_synthetic_count_stream_reconciles_and_records_scn2a(tmp_path: Path) -> None:
    counts = pd.DataFrame(
        {"s1": [1, 2, 0, 0], "s2": [2, 0, 3, 0]},
        index=["GeneA", "Scn2a1", "GeneB", "Zero"],
    )
    csv_bytes = counts.to_csv(index_label="gene_symbol").encode("utf-8")
    archive_path = tmp_path / "counts.csv.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("counts.csv")
        member.size = len(csv_bytes)
        archive.addfile(member, io.BytesIO(csv_bytes))
    library_path = tmp_path / "library.csv"
    pd.DataFrame(
        {
            "transcriptomics_sample_id": ["s1", "s2"],
            "library_size_counts": counts.sum(axis=0).to_numpy(),
        }
    ).to_csv(library_path, index=False)

    sample_qc, gene_qc, matches, audit, scn2a1 = rs.stream_count_qc(
        archive_path,
        library_path,
        {"NaV": ["GeneA"]},
        chunksize=2,
    )
    assert sample_qc.total_counts.tolist() == [3, 5]
    assert sample_qc.genes_detected.tolist() == [2, 2]
    assert sample_qc.library_size_delta.eq(0).all()
    assert np.allclose(sample_qc.log10_total_counts, np.log10([3, 5]))
    assert np.isfinite(sample_qc.select_dtypes(include=[np.number])).all().all()
    assert audit["gene_rows"] == audit["unique_gene_symbols"] == 4
    assert audit["sample_n"] == 2
    assert audit["library_reconciliation_exact"] is True
    assert matches.gene_symbol.tolist() == ["Scn2a1"]
    assert bool(matches.exact_scn2a1.iloc[0])
    assert not bool(matches.exact_scn2a.iloc[0])
    assert scn2a1.set_index("transcriptomics_sample_id")["count"].to_dict() == {
        "s1": 2.0,
        "s2": 0.0,
    }
    exclusions = gene_qc.set_index("gene_symbol").exclusion_reason.to_dict()
    assert exclusions["GeneA"] == "primary_module_gene"
    assert exclusions["Scn2a1"] == "scn2a_alias_audit_gene"
    assert exclusions["Zero"] == "zero_variance"
    assert bool(gene_qc.set_index("gene_symbol").loc["GeneB", "candidate_eligible"])

    bad_library = tmp_path / "bad_library.csv"
    bad = pd.read_csv(library_path)
    bad.loc[0, "library_size_counts"] += 1
    bad.to_csv(bad_library, index=False)
    with pytest.raises(AssertionError, match="Recomputed library sizes differ"):
        rs.stream_count_qc(archive_path, bad_library, {"NaV": ["GeneA"]}, chunksize=3)


def test_synthetic_fit_rejects_group_leakage_and_writes_exact_ledger(monkeypatch) -> None:
    leaking = pd.DataFrame(
        {
            "canonical_cell_id": ["a", "b", "c", "d"],
            "group_id": ["shared", "g1", "shared", "g2"],
            "subclass": ["S"] * 4,
            "fold": [0, 0, 1, 1],
            "feature": [0.0, 1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(AssertionError, match="Outer donor leakage"):
        rs._fit_selected_models(
            leaking,
            ["feature"],
            pd.Series(np.arange(4.0)),
            target_name="synthetic",
            folds=(0,),
            seed=42,
            analysis="synthetic",
            model_dir=None,
        )

    class ZeroPipeline:
        def predict(self, frame: pd.DataFrame) -> np.ndarray:
            return np.zeros(len(frame), dtype=float)

    monkeypatch.setattr(rs, "elastic_net_candidates", lambda features, seed: [("EN", None)])
    monkeypatch.setattr(
        rs, "xgboost_candidates", lambda features, seed, level: [("XGB", None)]
    )
    monkeypatch.setattr(
        rs,
        "select_on_inner_groups",
        lambda candidates, X, y, groups, seed: SimpleNamespace(
            pipeline=ZeroPipeline(), inner_rmse=1.0, parameters={"choice": candidates[0][0]}
        ),
    )
    clean = pd.DataFrame(
        {
            "canonical_cell_id": [f"cell-{i}" for i in range(6)],
            "group_id": [f"donor-{i}" for i in range(6)],
            "subclass": ["S0", "S0", "S1", "S1", "S2", "S2"],
            "fold": [0, 0, 0, 1, 1, 1],
            "feature": np.arange(6.0),
        }
    )
    oof, parameters, ledger = rs._fit_selected_models(
        clean,
        ["feature"],
        pd.Series(np.arange(6.0)),
        target_name="synthetic",
        folds=(0, 1),
        seed=42,
        analysis="synthetic",
        model_dir=None,
    )
    assert len(oof) == 6 * 2
    assert len(parameters) == 2 * 2
    assert len(ledger) == 2
    assert ledger.preprocessing_fit_scope.eq("outer_training_only").all()
    _assert_leakage_ledger(ledger, clean)


def test_synthetic_random_partition_validation_is_fail_closed(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "canonical_cell_id": ["a", "b", "c", "d"],
            "group_id": ["g0", "g1", "g2", "g3"],
            "subclass": ["S"] * 4,
            "fold": [0, 0, 1, 1],
        }
    )
    part = rs.prediction_frame(
        table.loc[table.fold.eq(0)],
        np.array([0.1, 0.2]),
        np.array([0.0, 1.0]),
        target="control",
        model="XGBoost",
        analysis="matched_random_module",
    )
    part["target_hash"] = "target-hash"
    part["run_signature"] = "run-signature"
    path = tmp_path / "part.parquet"
    part.to_parquet(path, index=False)
    target_values = pd.Series([0.0, 1.0, 2.0, 3.0], index=table.index)
    valid = rs._valid_random_partition(
        path,
        table,
        target_values=target_values,
        target_hash="target-hash",
        run_signature="run-signature",
        folds=(0,),
    )
    assert valid is not None and len(valid) == 2
    assert rs._valid_random_partition(
        path,
        table,
        target_values=target_values,
        target_hash="wrong",
        run_signature="run-signature",
        folds=(0,),
    ) is None
    corrupted = part.copy()
    corrupted.loc[0, "fold"] = 1
    corrupted.to_parquet(path, index=False)
    assert rs._valid_random_partition(
        path,
        table,
        target_values=target_values,
        target_hash="target-hash",
        run_signature="run-signature",
        folds=(0,),
    ) is None
    corrupted = part.copy()
    corrupted.loc[0, "y_true"] = 99.0
    corrupted.to_parquet(path, index=False)
    assert rs._valid_random_partition(
        path,
        table,
        target_values=target_values,
        target_hash="target-hash",
        run_signature="run-signature",
        folds=(0,),
    ) is None


def test_synthetic_stage_cache_invalidates_changed_files_and_signatures(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("version one", encoding="utf-8")
    cache_path = tmp_path / "checkpoints.json"
    cache = rs.StageCache(cache_path, "signature-a")
    assert not cache.valid("stage", [artifact])
    cache.complete("stage", [artifact], row_n=1)
    assert cache.valid("stage", [artifact])
    reloaded = rs.StageCache(cache_path, "signature-a")
    assert reloaded.valid("stage", [artifact])
    artifact.write_text("version two", encoding="utf-8")
    assert not reloaded.valid("stage", [artifact])
    assert not rs.StageCache(cache_path, "signature-b").valid("stage", [artifact])


def test_synthetic_output_namespace_and_manifest_contract(tmp_path: Path) -> None:
    config = {
        "outputs": {
            "results": "results/revision_v3/specificity",
            "figures": "figures/revision_v3/specificity",
            "logs": "logs/revision_v3/specificity",
            "models": "models/revision_v3/specificity",
        }
    }
    roots = rs.output_roots(tmp_path, config)
    for namespace, base in roots.items():
        base.mkdir(parents=True)
        (base / f"{namespace}.txt").write_text(namespace, encoding="utf-8")
    (roots["results"] / "ignored.tmp").write_text("temporary", encoding="utf-8")
    manifest_path = rs.write_artifact_manifest(tmp_path, config)
    manifest = pd.read_csv(manifest_path)
    assert set(manifest.namespace) == set(roots)
    assert len(manifest) == 4
    assert not manifest.relative_path.str.endswith(".tmp").any()
    for row in manifest.itertuples(index=False):
        path = tmp_path / row.relative_path
        assert _inside(path, roots[row.namespace])
        assert row.bytes == path.stat().st_size
        assert row.sha256 == sha256(path)

    assert ROOT_FROM_CONFIG == ROOT
    assert OUTPUTS == {
        "results": ROOT / "results/revision_v3/specificity",
        "figures": ROOT / "figures/revision_v3/specificity",
        "logs": ROOT / "logs/revision_v3/specificity",
        "models": ROOT / "models/revision_v3/specificity",
    }


# ---------------------------------------------------------------------------
# Strict full-run artifact gates.  These deliberately do not skip when absent.


def test_full_artifacts_required_files_exist() -> None:
    required = [
        RESULTS / "integrity/integrity_gate.json",
        RESULTS / "technical_targets/transcriptomic_qc_targets.csv",
        RESULTS / "technical_targets/all_transcriptomic_qc_targets.parquet",
        RESULTS / "technical_targets/technical_target_oof.parquet",
        RESULTS / "technical_targets/technical_target_performance.csv",
        RESULTS / "technical_targets/technical_target_fold_metrics.csv",
        RESULTS / "technical_targets/technical_target_leakage_ledger.csv",
        RESULTS / "random_controls/all_gene_expression_qc.parquet",
        RESULTS / "random_controls/matched_gene_sets.json",
        RESULTS / "random_controls/matching_quality.csv",
        RESULTS / "random_controls/matching_assignments.parquet",
        RESULTS / "random_controls/random_module_performance.parquet",
        RESULTS / "random_controls/module_specificity_summary.csv",
        RESULTS / "random_controls/resume_audit.json",
        RESULTS / "technical_adjustment/crossfit_nuisance_predictions.parquet",
        RESULTS / "technical_adjustment/technical_residualized_oof.parquet",
        RESULTS / "technical_adjustment/nuisance_coefficients_and_leakage.csv",
        RESULTS / "scn2a_audit/scn2a_audit.json",
        RESULTS / "scn2a_audit/scn2a_symbol_matches.csv",
        RESULTS / "scn2a_audit/nav_alias_oof.parquet",
        RESULTS / "target_definitions/target_zmean.parquet",
        RESULTS / "target_definitions/target_pc1.parquet",
        RESULTS / "target_definitions/target_definition_oof.parquet",
        RESULTS / "reliability/split_half_partitions.parquet",
        RESULTS / "reliability/split_half_draws.parquet",
        RESULTS / "reliability/module_internal_consistency.csv",
        RESULTS / "target_correlations/target_correlation_summary.json",
        RESULTS / "core_only/core_cv_folds.csv",
        RESULTS / "core_only/core_oof.parquet",
        RESULTS / "core_only/metadata_qc_audit.csv",
        RESULTS / "core_only/core_leakage_ledger.csv",
        RESULTS / "ARTIFACT_MANIFEST.csv",
        RESULTS / "summary.json",
        LOGS / "checkpoints.json",
        LOGS / "full_complete_pre_review.json",
        ROOT / "REVISION_V3_SPECIFICITY_RESULTS.md",
    ]
    _require_files(*required)


def test_full_artifacts_integrity_cohort_and_primary_hashes() -> None:
    integrity_path = RESULTS / "integrity/integrity_gate.json"
    _require_files(integrity_path)
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    expected = CONFIG["expected"]
    assert integrity["status"] == "PASS"
    assert integrity["cohort_n"] == expected["cohort_n"] == 3410
    assert integrity["donor_n"] == expected["donor_n"] == 871
    assert len(integrity["subclasses"]) == 6
    assert set(map(int, integrity["fold_sizes"])) == {0, 1, 2}
    assert sum(integrity["fold_sizes"].values()) == 3410
    assert set(map(int, integrity["donor_overlap"])) == {0, 1, 2}
    assert set(integrity["donor_overlap"].values()) == {0}
    assert set(integrity["target_max_abs_difference"]) == set(MODULE_ORDER)
    assert set(integrity["target_max_abs_difference"].values()) == {0.0}

    table = _modeling_table()
    assert len(table) == 3410
    assert table.canonical_cell_id.is_unique
    assert table.transcriptomics_sample_id.is_unique
    assert table.group_id.nunique() == 871
    assert set(table.fold.astype(int)) == {0, 1, 2}
    for fold in (0, 1, 2):
        train_groups = set(table.loc[table.fold.ne(fold), "group_id"])
        test_groups = set(table.loc[table.fold.eq(fold), "group_id"])
        assert train_groups.isdisjoint(test_groups)

    paths = rs.resolve_paths(ROOT, CONFIG)
    assert sha256(paths["primary_oof"]) == expected["primary_oof_sha256"]
    assert sha256(paths["primary_shap"]) == expected["primary_shap_sha256"]
    assert integrity["primary_oof_sha256"] == expected["primary_oof_sha256"]
    assert integrity["primary_shap_sha256"] == expected["primary_shap_sha256"]


def test_full_artifacts_count_qc_and_technical_oof() -> None:
    cohort_path = RESULTS / "technical_targets/transcriptomic_qc_targets.csv"
    all_samples_path = RESULTS / "technical_targets/all_transcriptomic_qc_targets.parquet"
    gene_qc_path = RESULTS / "random_controls/all_gene_expression_qc.parquet"
    oof_path = RESULTS / "technical_targets/technical_target_oof.parquet"
    performance_path = RESULTS / "technical_targets/technical_target_performance.csv"
    ledger_path = RESULTS / "technical_targets/technical_target_leakage_ledger.csv"
    _require_files(
        cohort_path, all_samples_path, gene_qc_path, oof_path, performance_path, ledger_path
    )
    expected = CONFIG["expected"]
    cohort = pd.read_csv(cohort_path, dtype={"canonical_cell_id": str,
                                             "transcriptomics_sample_id": str})
    all_samples = pd.read_parquet(all_samples_path)
    gene_qc = pd.read_parquet(gene_qc_path)
    assert len(cohort) == expected["cohort_n"]
    assert len(all_samples) == expected["transcriptomic_sample_n"]
    assert len(gene_qc) == expected["gene_row_n"]
    assert cohort.canonical_cell_id.is_unique
    assert cohort.transcriptomics_sample_id.is_unique
    numeric = ["total_counts", "genes_detected", "log10_total_counts",
               "log10_genes_detected", "reference_library_size", "library_size_delta"]
    assert np.isfinite(cohort[numeric].to_numpy(dtype=float)).all()
    assert cohort.total_counts.gt(0).all()
    assert cohort.genes_detected.gt(0).all()
    assert np.array_equal(
        cohort.total_counts.to_numpy(dtype=float),
        cohort.reference_library_size.to_numpy(dtype=float),
    )
    assert cohort.library_size_delta.eq(0).all()
    assert np.allclose(cohort.log10_total_counts, np.log10(cohort.total_counts), atol=1e-12)
    assert np.allclose(
        cohort.log10_genes_detected, np.log10(cohort.genes_detected), atol=1e-12
    )
    assert all_samples.transcriptomics_sample_id.astype(str).is_unique
    assert gene_qc.gene_symbol.astype(str).is_unique

    table = _modeling_table()
    oof = pd.read_parquet(oof_path)
    expected_keys = set(
        itertools.product(
            ("log10_total_counts", "log10_genes_detected"), MODEL_ORDER, ("technical_target",)
        )
    )
    _assert_complete_oof(
        oof,
        set(table.canonical_cell_id),
        ("target", "model", "analysis"),
        expected_keys,
    )
    assert len(oof) == 3410 * 2 * 2
    _assert_fold_mapping(oof, table[["canonical_cell_id", "fold"]])
    _assert_metrics_reconstruct(
        oof, pd.read_csv(performance_path), ("target", "model", "analysis")
    )
    ledger = pd.read_csv(ledger_path)
    assert len(ledger) == 2 * 3
    assert ledger.preprocessing_fit_scope.eq("outer_training_only").all()
    _assert_leakage_ledger(ledger, table)


def test_full_artifacts_random_controls_matching_oof_and_empirical_pvalues() -> None:
    matched_path = RESULTS / "random_controls/matched_gene_sets.json"
    quality_path = RESULTS / "random_controls/matching_quality.csv"
    assignments_path = RESULTS / "random_controls/matching_assignments.parquet"
    performance_path = RESULTS / "random_controls/random_module_performance.parquet"
    summary_path = RESULTS / "random_controls/module_specificity_summary.csv"
    _require_files(matched_path, quality_path, assignments_path, performance_path, summary_path)
    matched = json.loads(matched_path.read_text(encoding="utf-8"))
    quality = pd.read_csv(quality_path)
    assignments = pd.read_parquet(assignments_path)
    performance = pd.read_parquet(performance_path)
    gene_qc = pd.read_parquet(RESULTS / "random_controls/all_gene_expression_qc.parquet")
    modules = load_gene_modules(rs.resolve_paths(ROOT, CONFIG)["gene_modules"])
    all_primary_genes = set().union(*(set(genes) for genes in modules.values()))
    control_n = int(CONFIG["random_controls"]["full_controls_per_module"])
    assert control_n == 200
    assert set(matched["modules"]) == set(MODULE_ORDER)
    assert quality.groupby("module").size().to_dict() == {
        module: control_n for module in MODULE_ORDER
    }
    assert quality.matching_pass.astype(bool).all()
    assert not quality.duplicated(["module", "set_signature"]).any()
    limits = CONFIG["random_controls"]
    assert quality.max_matching_distance.le(
        limits["max_per_gene_standardized_distance"] + 1e-12
    ).all()
    assert quality.mean_matching_distance.le(
        limits["max_mean_per_gene_standardized_distance"] + 1e-12
    ).all()
    assert quality.module_z_expression_difference.abs().le(
        limits["max_module_abs_standardized_mean_difference"] + 1e-12
    ).all()
    assert quality.module_z_detection_difference.abs().le(
        limits["max_module_abs_standardized_detection_difference"] + 1e-12
    ).all()

    eligible = set(gene_qc.loc[gene_qc.candidate_eligible, "gene_symbol"].astype(str))
    for module, payload in matched["modules"].items():
        assert payload["control_n"] == control_n
        assert len(payload["controls"]) == control_n
        present = payload["present_target_genes"]
        assert set(present).issubset(modules[module])
        for expected_id, control in enumerate(payload["controls"]):
            genes = control["genes"]
            assert control["control_id"] == expected_id
            assert len(genes) == len(set(genes)) == len(present)
            assert set(genes).issubset(eligible)
            assert set(genes).isdisjoint(all_primary_genes)
            assert set(control["assignment"]) == set(present)
            assert set(control["assignment"].values()) == set(genes)
            assert control["set_signature"] == rs.hash_text("\n".join(sorted(genes)))

    expected_assignment_n = sum(
        control_n * len(matched["modules"][module]["present_target_genes"])
        for module in MODULE_ORDER
    )
    assert len(assignments) == expected_assignment_n
    grouped = assignments.groupby(["module", "control_id"])
    assert grouped.control_gene.nunique().eq(grouped.size()).all()
    assert grouped.target_gene.nunique().eq(grouped.size()).all()
    assert assignments.control_gene.astype(str).isin(eligible).all()
    assert assignments.distance.le(limits["max_per_gene_standardized_distance"] + 1e-12).all()

    assert len(performance) == control_n * len(MODULE_ORDER)
    assert not performance.duplicated(["module", "control_id"]).any()
    assert performance.groupby("module").size().eq(control_n).all()
    assert performance.n.eq(3410).all()
    assert performance.matching_pass.astype(bool).all()
    metric_columns = ["r2", "mae", "rmse", "spearman", "fold0_r2", "fold1_r2", "fold2_r2"]
    assert np.isfinite(performance[metric_columns].to_numpy(dtype=float)).all()
    assert performance.run_signature.astype(str).nunique() == 1

    partition_paths = sorted(
        (RESULTS / "random_controls/random_module_oof").glob(
            "module=*/control_id=*/part-000.parquet"
        )
    )
    assert len(partition_paths) == control_n * len(MODULE_ORDER)
    assert all(path.stat().st_size > 0 for path in partition_paths)
    for module in MODULE_ORDER:
        module_paths = [path for path in partition_paths if path.parent.parent.name == f"module={module}"]
        assert len(module_paths) == control_n
        control_ids = {int(path.parent.name.split("=", 1)[1]) for path in module_paths}
        assert control_ids == set(range(control_n))

    # Deep-validate one complete checkpoint per module; the file-count/key gate
    # above covers every partition while avoiding 1,200 repeated parquet loads.
    table = _modeling_table()
    for module in MODULE_ORDER:
        row = performance.loc[performance.module.eq(module)].sort_values("control_id").iloc[0]
        target_frame = pd.read_parquet(
            RESULTS / f"random_controls/control_targets/{module}.parquet"
        ).set_index("transcriptomics_sample_id", verify_integrity=True)
        target_values = table.transcriptomics_sample_id.astype(str).map(
            target_frame[f"control_{int(row.control_id):04d}"]
        )
        assert target_values.notna().all()
        path = (
            RESULTS
            / "random_controls/random_module_oof"
            / f"module={module}"
            / f"control_id={int(row.control_id):04d}"
            / "part-000.parquet"
        )
        validated = rs._valid_random_partition(
            path,
            table,
            target_values=target_values,
            target_hash=str(row.target_hash),
            run_signature=str(row.run_signature),
            folds=(0, 1, 2),
        )
        assert validated is not None and len(validated) == 3410

    summary = pd.read_csv(summary_path).set_index("module")
    primary = pd.read_parquet(rs.resolve_paths(ROOT, CONFIG)["primary_oof"])
    for module in MODULE_ORDER:
        values = performance.loc[performance.module.eq(module), "r2"].to_numpy(dtype=float)
        observed_oof = primary.loc[
            primary.module.eq(module)
            & primary.model.eq("XGBoost")
            & primary.analysis.eq("ephys_only")
        ]
        observed = regression_metrics(observed_oof.y_true, observed_oof.y_pred)["r2"]
        row = summary.loc[module]
        exceed = int(np.sum(values >= observed))
        assert row.random_control_n == control_n
        assert np.isclose(row.observed_primary_xgboost_r2, observed, atol=1e-12)
        assert np.isclose(row.random_median_r2, np.median(values), atol=1e-12)
        assert np.isclose(row.random_95th_percentile_r2, np.quantile(values, 0.95), atol=1e-12)
        assert np.isclose(row.random_99th_percentile_r2, np.quantile(values, 0.99), atol=1e-12)
        assert row.random_r2_greater_equal_n == exceed
        assert np.isclose(row.empirical_p_plus_one, (1 + exceed) / (control_n + 1), atol=1e-12)
        assert np.isclose(
            row.observed_empirical_percentile_le, 100 * np.mean(values <= observed), atol=1e-12
        )


def test_full_artifacts_residualization_identity_oof_and_leakage() -> None:
    nuisance_path = RESULTS / "technical_adjustment/crossfit_nuisance_predictions.parquet"
    residual_path = RESULTS / "technical_adjustment/technical_residualized_oof.parquet"
    performance_path = RESULTS / "technical_adjustment/technical_residualized_performance.csv"
    ledger_path = RESULTS / "technical_adjustment/nuisance_coefficients_and_leakage.csv"
    _require_files(nuisance_path, residual_path, performance_path, ledger_path)
    table = _modeling_table()
    nuisance = pd.read_parquet(nuisance_path)
    residual = pd.read_parquet(residual_path)
    assert len(nuisance) == 3410 * len(MODULE_ORDER)
    assert not nuisance.duplicated(["canonical_cell_id", "module"]).any()
    assert set(nuisance.module) == set(MODULE_ORDER)
    nuisance_numeric = [
        "target_raw", "log10_total_counts", "log10_genes_detected",
        "nuisance_prediction", "residual_target",
    ]
    assert np.isfinite(nuisance[nuisance_numeric].to_numpy(dtype=float)).all()
    assert np.allclose(
        nuisance.target_raw - nuisance.nuisance_prediction,
        nuisance.residual_target,
        rtol=0,
        atol=1e-12,
    )
    _assert_fold_mapping(nuisance, table[["canonical_cell_id", "fold"]])

    expected_keys = set(
        itertools.product(MODULE_ORDER, MODEL_ORDER, ("technical_residualized",))
    )
    _assert_complete_oof(
        residual,
        set(table.canonical_cell_id),
        ("module", "model", "analysis"),
        expected_keys,
    )
    assert len(residual) == 3410 * len(MODULE_ORDER) * len(MODEL_ORDER)
    _assert_fold_mapping(residual, table[["canonical_cell_id", "fold"]])
    truths = nuisance.set_index(["canonical_cell_id", "module"], verify_integrity=True).residual_target
    residual_index = pd.MultiIndex.from_frame(
        residual[["canonical_cell_id", "module"]].astype(str)
    )
    assert np.allclose(residual.y_true, truths.reindex(residual_index), rtol=0, atol=1e-12)
    _assert_metrics_reconstruct(residual, pd.read_csv(performance_path), ("module", "model"))

    ledger = pd.read_csv(ledger_path)
    assert len(ledger) == len(MODULE_ORDER) * 3
    assert ledger.nuisance_fit_scope.eq("outer_training_only").all()
    _assert_leakage_ledger(ledger, table)


def test_full_artifacts_target_definitions_reconstruct_and_oof_is_complete() -> None:
    target_dir = RESULTS / "target_definitions"
    paths = {
        "mean_logcpm": target_dir / "target_mean_logcpm.parquet",
        "zmean": target_dir / "target_zmean.parquet",
        "pc1": target_dir / "target_pc1.parquet",
    }
    _require_files(
        *paths.values(),
        target_dir / "target_gene_standardization.csv",
        target_dir / "pc1_loadings.csv",
        target_dir / "pc1_variance_explained.csv",
        target_dir / "target_definition_oof.parquet",
    )
    table = _modeling_table()
    frames = {name: pd.read_parquet(path) for name, path in paths.items()}
    for frame in frames.values():
        assert len(frame) == 3410
        assert frame.canonical_cell_id.astype(str).is_unique
        assert np.isfinite(frame[list(MODULE_ORDER)].to_numpy(dtype=float)).all()
    for module in MODULE_ORDER:
        assert np.array_equal(
            frames["mean_logcpm"][module].to_numpy(dtype=float),
            table[f"target_{module}"].to_numpy(dtype=float),
        )

    source_paths = rs.resolve_paths(ROOT, CONFIG)
    cpm = pd.read_parquet(source_paths["module_cpm"])
    cpm.index = cpm.index.astype(str)
    transformed = np.log2(cpm.astype(float) + 1.0)
    stats = pd.read_csv(target_dir / "target_gene_standardization.csv")
    loadings = pd.read_csv(target_dir / "pc1_loadings.csv")
    variance = pd.read_csv(target_dir / "pc1_variance_explained.csv").set_index("module")
    for module in MODULE_ORDER:
        module_stats = stats.loc[stats.module.eq(module)].set_index("gene")
        genes = module_stats.index.tolist()
        matrix = transformed[genes].to_numpy(dtype=float)
        standardized = (
            matrix - module_stats.mean_log2cpm.to_numpy().reshape(1, -1)
        ) / module_stats.scale_log2cpm.to_numpy().reshape(1, -1)
        zmean = pd.Series(standardized.mean(axis=1), index=transformed.index)
        loading = (
            loadings.loc[loadings.module.eq(module)].set_index("gene").loc[genes, "loading"]
            .to_numpy(dtype=float)
        )
        pc1 = pd.Series(standardized @ loading, index=transformed.index)
        mean = pd.Series(matrix.mean(axis=1), index=transformed.index)
        for name, reconstructed in (("mean_logcpm", mean), ("zmean", zmean), ("pc1", pc1)):
            observed = frames[name].transcriptomics_sample_id.astype(str).map(reconstructed)
            assert observed.notna().all()
            assert np.allclose(frames[name][module], observed, rtol=0, atol=1e-11)
        row = variance.loc[module]
        assert 0 < row.explained_variance_ratio <= 1
        assert row.orientation_dot_with_zmean >= -1e-12
        assert np.isclose(row.loading_l2_norm, 1.0, atol=1e-12)
        assert row.fit_sample_n == CONFIG["expected"]["transcriptomic_sample_n"]
        assert row.construction_inputs == "transcriptomics_only_full_release"

    oof = pd.read_parquet(target_dir / "target_definition_oof.parquet")
    expected_keys = set(
        itertools.product(
            MODULE_ORDER,
            ("mean_logcpm", "zmean", "pc1"),
            MODEL_ORDER,
            ("target_definition_sensitivity",),
        )
    )
    _assert_complete_oof(
        oof,
        set(table.canonical_cell_id),
        ("module", "target_definition", "model", "analysis"),
        expected_keys,
    )
    assert len(oof) == 3410 * len(MODULE_ORDER) * 3 * 2
    _assert_fold_mapping(oof, table[["canonical_cell_id", "fold"]])


def test_full_artifacts_scn2a_search_alias_target_oof_and_ledger() -> None:
    result_dir = RESULTS / "scn2a_audit"
    audit_path = result_dir / "scn2a_audit.json"
    matches_path = result_dir / "scn2a_symbol_matches.csv"
    counts_path = result_dir / "scn2a1_counts.parquet"
    target_path = result_dir / "nav_alias_target.parquet"
    oof_path = result_dir / "nav_alias_oof.parquet"
    ledger_path = result_dir / "nav_alias_leakage_ledger.csv"
    _require_files(audit_path, matches_path, counts_path, target_path, oof_path, ledger_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    matches = pd.read_csv(matches_path)
    gene_qc = pd.read_parquet(RESULTS / "random_controls/all_gene_expression_qc.parquet")
    expected_matches = gene_qc.loc[
        gene_qc.gene_symbol.astype(str).str.strip().str.casefold().str.contains("scn2a"),
        ["gene_symbol", "source_row"],
    ].sort_values("source_row").reset_index(drop=True)
    observed_matches = matches[["gene_symbol", "source_row"]].sort_values(
        "source_row"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(observed_matches, expected_matches, check_dtype=False)
    assert audit["case"] == "B"
    assert audit["exact_scn2a_present"] is False
    assert audit["exact_scn2a1_present"] is True
    assert audit["primary_module_changed"] is False
    assert "no primary replacement" in audit["mapping_policy"].lower()
    assert not matches.gene_symbol.astype(str).eq("Scn2a").any()
    assert matches.gene_symbol.astype(str).eq("Scn2a1").any()

    counts = pd.read_parquet(counts_path)
    assert len(counts) == CONFIG["expected"]["transcriptomic_sample_n"]
    assert counts.transcriptomics_sample_id.astype(str).is_unique
    assert np.isfinite(counts["count"].to_numpy(dtype=float)).all()
    assert counts["count"].ge(0).all()
    assert np.equal(counts["count"], np.floor(counts["count"])).all()

    source_paths = rs.resolve_paths(ROOT, CONFIG)
    library = pd.read_csv(source_paths["library_sizes"], dtype={"transcriptomics_sample_id": str})
    alias = counts.assign(
        transcriptomics_sample_id=counts.transcriptomics_sample_id.astype(str)
    ).merge(library, on="transcriptomics_sample_id", validate="one_to_one")
    alias["log2cpm"] = np.log2(alias["count"] / alias.library_size_counts * 1e6 + 1)
    alias_lookup = alias.set_index("transcriptomics_sample_id", verify_integrity=True).log2cpm
    cpm = pd.read_parquet(source_paths["module_cpm"])
    cpm.index = cpm.index.astype(str)
    modules = load_gene_modules(source_paths["gene_modules"])
    genes = [gene for gene in modules["NaV"] if gene in cpm.columns]
    nav = np.log2(cpm[genes].astype(float) + 1.0)
    reconstructed = (nav.sum(axis=1) + alias_lookup.reindex(nav.index)) / (len(genes) + 1)
    target = pd.read_parquet(target_path)
    observed = target.transcriptomics_sample_id.astype(str).map(reconstructed)
    assert observed.notna().all()
    assert np.allclose(target.target_NaV_alias_Scn2a1, observed, rtol=0, atol=1e-12)

    table = _modeling_table()
    oof = pd.read_parquet(oof_path)
    expected_keys = set(itertools.product(("NaV_alias_Scn2a1",), MODEL_ORDER,
                                          ("scn2a1_alias_sensitivity",)))
    _assert_complete_oof(
        oof,
        set(table.canonical_cell_id),
        ("target", "model", "analysis"),
        expected_keys,
    )
    assert len(oof) == 3410 * 2
    _assert_fold_mapping(oof, table[["canonical_cell_id", "fold"]])
    ledger = pd.read_csv(ledger_path)
    assert len(ledger) == 3
    assert ledger.preprocessing_fit_scope.eq("outer_training_only").all()
    _assert_leakage_ledger(ledger, table)


def test_full_artifacts_reliability_partitions_and_spearman_brown_reconstruct() -> None:
    partition_path = RESULTS / "reliability/split_half_partitions.parquet"
    draws_path = RESULTS / "reliability/split_half_draws.parquet"
    summary_path = RESULTS / "reliability/module_internal_consistency.csv"
    _require_files(partition_path, draws_path, summary_path)
    partitions = pd.read_parquet(partition_path)
    draws = pd.read_parquet(draws_path)
    summary = pd.read_csv(summary_path).set_index("module")
    assert set(summary.index) == set(MODULE_ORDER)
    assert not partitions.duplicated(["module", "partition_id"]).any()
    assert not draws.duplicated(["module", "partition_id"]).any()
    assert len(partitions) == len(draws) == int(summary.partition_n.sum())
    seen: set[tuple[str, str, str]] = set()
    for row in partitions.itertuples(index=False):
        genes_a = str(row.genes_a).split(";")
        genes_b = str(row.genes_b).split(";")
        assert set(genes_a).isdisjoint(genes_b)
        assert len(genes_a) == row.n_a
        assert len(genes_b) == row.n_b
        assert abs(row.n_a - row.n_b) <= 1
        left = ";".join(sorted(genes_a))
        right = ";".join(sorted(genes_b))
        key = (row.module, *sorted((left, right)))
        assert key not in seen
        seen.add(key)
    for module, part in partitions.groupby("module", sort=False):
        row = summary.loc[module]
        assert len(part) == row.partition_n == row.unique_partition_n
        assert set(part.partition_id) == set(range(len(part)))
        assert part.generation.nunique() == 1
        assert part.generation.iloc[0] == row.generation
        assert part.maximum_unique_partitions.nunique() == 1
        assert part.maximum_unique_partitions.iloc[0] == row.maximum_unique_partitions
        if row.generation == "exhaustive":
            assert row.partition_n == row.maximum_unique_partitions
        else:
            assert row.partition_n == CONFIG["reliability"]["large_module_draws"]
    merged = draws.merge(
        partitions[["module", "partition_id"]],
        on=["module", "partition_id"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    assert merged._merge.eq("both").all()
    denominator = 1 + merged.raw_spearman_rho.to_numpy(dtype=float)
    reconstructed = np.divide(
        2 * merged.raw_spearman_rho.to_numpy(dtype=float),
        denominator,
        out=np.full(len(merged), np.nan),
        where=~np.isclose(denominator, 0),
    )
    assert np.allclose(merged.spearman_brown, reconstructed, atol=1e-12, equal_nan=True)


def test_full_artifacts_core_only_cohort_folds_oof_and_leakage() -> None:
    result_dir = RESULTS / "core_only"
    audit_path = result_dir / "core_cohort_audit.csv"
    folds_path = result_dir / "core_cv_folds.csv"
    oof_path = result_dir / "core_oof.parquet"
    performance_path = result_dir / "core_performance.csv"
    comparison_path = result_dir / "primary_vs_core_comparison.csv"
    metadata_path = result_dir / "metadata_qc_audit.csv"
    hyperparameters_path = result_dir / "core_hyperparameters.csv"
    ledger_path = result_dir / "core_leakage_ledger.csv"
    contamination_path = result_dir / "contamination_audit.json"
    _require_files(
        audit_path, folds_path, oof_path, performance_path, comparison_path,
        metadata_path, hyperparameters_path, ledger_path, contamination_path,
    )
    audit = pd.read_csv(audit_path, dtype={"canonical_cell_id": str, "group_id": str})
    folds = pd.read_csv(folds_path, dtype={"canonical_cell_id": str, "group_id": str})
    required_value = str(CONFIG["core_only"]["map_confidence_value"])
    assert len(audit) == 3410
    assert audit.canonical_cell_id.is_unique
    assert np.array_equal(
        audit.included_core_only.astype(bool).to_numpy(),
        audit.map_confidence.astype(str).eq(required_value).to_numpy(),
    )
    expected_ids = set(audit.loc[audit.included_core_only.astype(bool), "canonical_cell_id"])
    assert len(expected_ids) == 1964
    assert audit.loc[audit.included_core_only.astype(bool), "group_id"].nunique() == 748
    assert (
        audit.loc[audit.included_core_only.astype(bool), "subclass"]
        .value_counts()
        .sort_index()
        .to_dict()
        == {"Lamp5": 260, "Pvalb": 440, "Serpinf1": 6, "Sncg": 76, "Sst": 856, "Vip": 326}
    )
    assert set(folds.canonical_cell_id) == expected_ids
    assert folds.canonical_cell_id.is_unique
    assert set(folds.fold.astype(int)) == {0, 1, 2}
    assert folds.fold.value_counts().sort_index().to_dict() == {0: 655, 1: 655, 2: 654}
    assert folds.groupby("fold").group_id.nunique().sort_index().to_dict() == {
        0: 247, 1: 251, 2: 250
    }
    assert folds.fold_seed.eq(CONFIG["core_only"]["fold_seed"]).all()
    assert folds.split_method.eq("StratifiedGroupKFold_shuffle").all()
    assert folds.groupby("group_id").fold.nunique().max() == 1
    for fold in (0, 1, 2):
        train_groups = set(folds.loc[folds.fold.ne(fold), "group_id"])
        test_groups = set(folds.loc[folds.fold.eq(fold), "group_id"])
        assert train_groups.isdisjoint(test_groups)

    oof = pd.read_parquet(oof_path)
    targets = tuple(CONFIG["core_only"]["targets"])
    expected_keys = set(
        itertools.product(targets, MODEL_ORDER, ("quality_restricted_core_only",))
    )
    _assert_complete_oof(
        oof,
        expected_ids,
        ("target", "model", "analysis"),
        expected_keys,
    )
    assert len(oof) == len(folds) * len(targets) * len(MODEL_ORDER)
    _assert_fold_mapping(oof, folds)
    _assert_metrics_reconstruct(
        oof, pd.read_csv(performance_path), ("target", "model", "analysis")
    )
    ledger = pd.read_csv(ledger_path)
    assert len(ledger) == len(targets) * 3
    assert ledger.preprocessing_fit_scope.eq("outer_training_only").all()
    _assert_leakage_ledger(ledger, folds)

    # Five targets x three folds x two estimators must represent 30 distinct
    # fitted/tuned models, both in the parameter ledger and on disk.
    hyperparameters = pd.read_csv(hyperparameters_path)
    fitted_keys = set(
        hyperparameters[["target", "fold", "model"]]
        .itertuples(index=False, name=None)
    )
    expected_fitted_keys = set(itertools.product(targets, (0, 1, 2), MODEL_ORDER))
    assert len(hyperparameters) == len(fitted_keys) == 30
    assert fitted_keys == expected_fitted_keys
    expected_model_names = {
        f"{target}_fold{fold}_{model}.joblib"
        for target, fold, model in expected_fitted_keys
    }
    model_files = set((MODELS / "core_only").glob("*.joblib"))
    assert {path.name for path in model_files} == expected_model_names
    assert all(path.stat().st_size > 0 for path in model_files)

    comparison = pd.read_csv(comparison_path)
    assert len(comparison) == len(targets) * len(MODEL_ORDER)
    required_comparison = {
        "n_primary_full", "n_primary_core_subset", "n_core",
        "r2_primary_full", "r2_primary_core_subset", "r2_core",
        "delta_r2_core_minus_primary_full",
        "delta_r2_core_minus_primary_core_subset",
        "positive_r2_performance_retained", "spearman_sign_concordant",
        "core_concordance_rule",
    }
    assert required_comparison.issubset(comparison.columns)
    assert np.isfinite(
        comparison[
            [
                "r2_primary_full", "r2_primary_core_subset", "r2_core",
                "delta_r2_core_minus_primary_full",
                "delta_r2_core_minus_primary_core_subset",
            ]
        ].to_numpy()
    ).all()
    assert np.allclose(
        comparison.delta_r2_core_minus_primary_full,
        comparison.r2_core - comparison.r2_primary_full,
        atol=1e-12,
    )
    assert np.allclose(
        comparison.delta_r2_core_minus_primary_core_subset,
        comparison.r2_core - comparison.r2_primary_core_subset,
        atol=1e-12,
    )
    assert np.array_equal(
        comparison.positive_r2_performance_retained.astype(bool),
        comparison.r2_core.gt(0) & comparison.r2_primary_full.gt(0),
    )
    assert np.array_equal(
        comparison.spearman_sign_concordant.astype(bool),
        np.sign(comparison.spearman_core) == np.sign(comparison.spearman_primary_full),
    )
    assert np.array_equal(
        comparison.core_concordance_rule.astype(bool),
        comparison.spearman_sign_concordant.astype(bool)
        & comparison.delta_r2_core_minus_primary_full.abs().le(0.10),
    )

    primary = pd.read_parquet(rs.resolve_paths(ROOT, CONFIG)["primary_oof"])
    primary_modules = primary.loc[
        primary.module.isin([target for target in targets if not target.startswith("log10_")])
        & primary.model.isin(MODEL_ORDER)
        & primary.analysis.eq("ephys_only")
    ].rename(columns={"module": "target"})
    primary_technical = pd.read_parquet(
        RESULTS / "technical_targets/technical_target_oof.parquet"
    )
    primary_combined = pd.concat(
        [primary_modules, primary_technical.loc[primary_technical.target.isin(targets)]],
        ignore_index=True,
        sort=False,
    )
    comparison_indexed = comparison.set_index(["target", "model"])
    for keys, primary_frame in primary_combined.groupby(["target", "model"], sort=True):
        core_subset = primary_frame.loc[
            primary_frame.canonical_cell_id.astype(str).isin(expected_ids)
        ]
        core_frame = oof.loc[oof.target.eq(keys[0]) & oof.model.eq(keys[1])]
        saved = comparison_indexed.loc[keys]
        assert saved.n_primary_full == len(primary_frame) == 3410
        assert saved.n_primary_core_subset == len(core_subset) == 1964
        assert saved.n_core == len(core_frame) == 1964
        for suffix, frame in (
            ("primary_full", primary_frame),
            ("primary_core_subset", core_subset),
            ("core", core_frame),
        ):
            metrics = regression_metrics(frame.y_true, frame.y_pred)
            assert np.isclose(saved[f"r2_{suffix}"], metrics["r2"], atol=1e-12)
            assert np.isclose(saved[f"spearman_{suffix}"], metrics["spearman"], atol=1e-12)

    contamination = json.loads(contamination_path.read_text(encoding="utf-8"))
    assert contamination["direct_contamination_score_found"] is False
    assert contamination["core_value"] == required_value
    assert contamination["core_n"] == len(folds) == 1964
    assert contamination["core_donor_n"] == 748
    assert contamination["core_subclasses"] == {
        "Lamp5": 260, "Pvalb": 440, "Serpinf1": 6, "Sncg": 76, "Sst": 856, "Vip": 326
    }
    assert contamination["fold_seed"] == CONFIG["core_only"]["fold_seed"] == 42
    assert contamination["fold_method"] == "StratifiedGroupKFold(n_splits=3, shuffle=True)"
    assert {int(key): value for key, value in contamination["fold_sizes"].items()} == {
        0: 655, 1: 655, 2: 654
    }
    assert {int(key): value for key, value in contamination["fold_donor_counts"].items()} == {
        0: 247, 1: 251, 2: 250
    }
    assert "not treated as a contamination score" in contamination[
        "map_confidence_interpretation"
    ]


def test_full_artifacts_metadata_qc_semantics_and_distributions() -> None:
    audit_path = RESULTS / "core_only/metadata_qc_audit.csv"
    contamination_path = RESULTS / "core_only/contamination_audit.json"
    _require_files(audit_path, contamination_path)
    audit = pd.read_csv(audit_path)
    required = {
        "source", "source_sha256", "scope", "column", "semantic_name", "dtype",
        "row_n", "missing_n", "missing_fraction", "unique_nonmissing_n",
        "distribution_json", "documented_project_role", "documentation_reference",
        "direct_contamination_score",
    }
    assert required.issubset(audit.columns)
    assert audit.column.is_unique
    assert audit.semantic_name.equals(audit.column)
    assert not {"Tree_first_cl_label", "Unnamed: 21"}.intersection(audit.column)
    assert {"transcriptomic_cluster_id", "transcriptomic_cluster_label", "map_confidence"}.issubset(
        audit.column
    )
    assert not audit.direct_contamination_score.astype(bool).any()
    assert audit.documentation_reference.notna().all()
    assert np.allclose(audit.missing_fraction, audit.missing_n / audit.row_n, atol=1e-15)
    distributions = audit.distribution_json.map(json.loads)
    assert distributions.map(lambda value: isinstance(value, dict) and bool(value)).all()

    source_paths = rs.resolve_paths(ROOT, CONFIG)
    metadata = load_public_metadata(source_paths["metadata"])
    metadata_rows = audit.loc[audit.scope.eq("all_4435_public_metadata_samples")]
    assert len(metadata_rows) == len(metadata.columns)
    assert set(metadata_rows.column) == set(metadata.columns)
    assert metadata_rows.row_n.eq(4435).all()
    assert metadata_rows.source.nunique() == 1
    assert metadata_rows.source_sha256.nunique() == 1
    assert metadata_rows.source_sha256.iloc[0] == sha256(source_paths["metadata"])

    cluster_id = audit.set_index("column").loc["transcriptomic_cluster_id"]
    cluster_id_distribution = json.loads(cluster_id.distribution_json)
    assert "semantically repaired CS accession identifier" in cluster_id.documented_project_role
    assert cluster_id_distribution
    assert all(
        value == "<MISSING>" or re.fullmatch(r"CS\d+", value)
        for value in cluster_id_distribution
    )
    assert cluster_id_distribution.get("<MISSING>", 0) == int(cluster_id.missing_n)
    cluster_label = audit.set_index("column").loc["transcriptomic_cluster_label"]
    cluster_label_distribution = json.loads(cluster_label.distribution_json)
    assert "semantically repaired public cluster label" in cluster_label.documented_project_role
    assert cluster_label_distribution
    assert not any(re.fullmatch(r"CS\d+", value) for value in cluster_label_distribution)

    map_row = audit.set_index("column").loc["map_confidence"]
    assert map_row.scope == "3410_exact_id_matched_cells"
    assert map_row.row_n == 3410
    assert map_row.missing_n == 0
    assert map_row.unique_nonmissing_n == 4
    assert json.loads(map_row.distribution_json) == {
        "Core": 1964, "I1": 780, "I2": 602, "I3": 64
    }
    assert "not a contamination score" in map_row.documented_project_role
    assert map_row.source_sha256 == sha256(source_paths["crosswalk"])

    contamination = json.loads(contamination_path.read_text(encoding="utf-8"))
    candidates = sorted(
        column for column in metadata.columns if re.search(r"contam|quality|qc|rna", column, re.I)
    )
    assert sorted(contamination["direct_contamination_score_columns"]) == candidates


def test_full_artifacts_target_correlations_are_valid_and_reconstruct_summary() -> None:
    result_dir = RESULTS / "target_correlations"
    matrix_paths = {
        "raw": result_dir / "raw_spearman.csv",
        "technical_adjusted": result_dir / "technical_adjusted_spearman.csv",
        "subclass_centered": result_dir / "subclass_centered_spearman.csv",
    }
    summary_path = result_dir / "target_correlation_summary.json"
    means_path = result_dir / "subclass_training_means.csv"
    _require_files(*matrix_paths.values(), summary_path, means_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert set(summary) == set(matrix_paths)
    for name, path in matrix_paths.items():
        matrix = pd.read_csv(path, index_col=0).reindex(index=MODULE_ORDER, columns=MODULE_ORDER)
        assert matrix.shape == (6, 6)
        assert np.isfinite(matrix.to_numpy(dtype=float)).all()
        assert np.allclose(matrix, matrix.T, atol=1e-12)
        assert np.allclose(np.diag(matrix), 1.0, atol=1e-12)
        reconstructed = rs._correlation_summary(matrix)
        assert np.allclose(
            summary[name]["eigenvalues_descending"],
            reconstructed["eigenvalues_descending"],
            atol=1e-12,
        )
        assert np.isclose(
            summary[name]["first_eigenvalue_fraction"],
            reconstructed["first_eigenvalue_fraction"],
            atol=1e-12,
        )
    means = pd.read_csv(means_path)
    assert means.fit_scope.eq("outer_training_only").all()
    assert set(means.fold.astype(int)) == {0, 1, 2}
    assert set(means.module) == set(MODULE_ORDER)


def test_full_artifacts_resume_cache_manifest_and_namespace_contracts() -> None:
    cache_path = LOGS / "checkpoints.json"
    marker_path = LOGS / "full_complete_pre_review.json"
    manifest_path = RESULTS / "ARTIFACT_MANIFEST.csv"
    summary_path = RESULTS / "summary.json"
    report_path = ROOT / "REVISION_V3_SPECIFICITY_RESULTS.md"
    _require_files(cache_path, marker_path, manifest_path, summary_path, report_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    expected_stages = {
        "count_qc", "technical_models", "random_controls", "technical_adjustment",
        "scn2a_alias", "target_definitions", "internal_consistency",
        "target_correlations", "core_only",
    }
    assert marker["status"] == "PASS_PRE_REVIEW"
    assert marker["mode"] == "resume"
    assert set(marker["stages"]) == expected_stages - {"count_qc"}
    assert set(marker["stages"].values()) == {"reused"}
    assert cache["signature"] == marker["signature"]
    assert set(cache["stages"]) == expected_stages
    assert {stage: len(record["files"]) for stage, record in cache["stages"].items()} == {
        "count_qc": 6,
        "technical_models": 18,
        "random_controls": 1216,
        "technical_adjustment": 45,
        "scn2a_alias": 11,
        "target_definitions": 82,
        "internal_consistency": 3,
        "target_correlations": 11,
        "core_only": 39,
    }
    assert marker["primary_oof_sha256_after"] == CONFIG["expected"]["primary_oof_sha256"]
    assert marker["primary_shap_sha256_after"] == CONFIG["expected"]["primary_shap_sha256"]
    allowed_roots = tuple(OUTPUTS.values())
    for stage, record in cache["stages"].items():
        assert stage in expected_stages
        assert record["files"]
        for path_text, expected_hash in record["files"].items():
            path = Path(path_text)
            assert path.is_file()
            assert any(_inside(path, base) for base in allowed_roots)
            assert sha256(path) == expected_hash

    random_cached = {Path(path) for path in cache["stages"]["random_controls"]["files"]}
    random_partitions = {
        path for path in random_cached if "random_module_oof" in path.parts
    }
    assert len(random_partitions) == len(MODULE_ORDER) * int(
        CONFIG["random_controls"]["full_controls_per_module"]
    )
    for module in MODULE_ORDER:
        assert (RESULTS / f"random_controls/control_targets/{module}.parquet").resolve() in random_cached
    assert (RESULTS / "random_controls/matching_assignments.parquet").resolve() in random_cached

    resume = json.loads(
        (RESULTS / "random_controls/resume_audit.json").read_text(encoding="utf-8")
    )
    assert resume["control_n"] == len(MODULE_ORDER) * int(
        CONFIG["random_controls"]["full_controls_per_module"]
    )
    assert resume["validated_complete_partitions_reused"] + resume["new_partitions_fit"] == resume[
        "control_n"
    ]
    assert resume["folds"] == [0, 1, 2]

    manifest = pd.read_csv(manifest_path)
    assert not manifest.empty
    assert set(manifest.namespace) == set(OUTPUTS)
    assert manifest.relative_path.astype(str).is_unique
    assert not manifest.relative_path.str.endswith(".tmp").any()
    manifest_paths = set(manifest.relative_path.astype(str))
    late_written_paths = {(LOGS / "pytest.log").relative_to(ROOT).as_posix()}
    # pytest.log can still be open during this test. The pipeline performs
    # one final manifest regeneration after the logged test run; every pipeline
    # artifact, including summary and completion marker, must already be exact.
    stale_late_written: set[str] = set()
    for row in manifest.itertuples(index=False):
        path = ROOT / row.relative_path
        assert path.is_file()
        assert _inside(path, OUTPUTS[row.namespace])
        bytes_match = path.stat().st_size == row.bytes
        hash_match = sha256(path) == row.sha256
        if not (bytes_match and hash_match):
            assert row.relative_path in late_written_paths
            stale_late_written.add(row.relative_path)
        assert path.suffix.lower() == row.suffix
    forbidden_primary = {
        "results/predictions/oof_predictions.parquet",
        "results/shap/oof_shap.parquet",
    }
    assert forbidden_primary.isdisjoint(manifest_paths)
    assert stale_late_written.issubset(late_written_paths)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["report"] == report_path.relative_to(ROOT).as_posix()
    assert Path(summary["artifact_manifest"]) == manifest_path.relative_to(ROOT)
