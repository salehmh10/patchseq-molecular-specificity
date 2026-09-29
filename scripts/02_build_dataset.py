#!/usr/bin/env python
"""Build exact-ID matched module targets, modeling table, and frozen grouped folds."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.biology import load_gene_modules, module_dominance, score_modules
from src.data import extract_selected_counts
from src.training import MODULES, load_feature_names


def _load_cpm(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    for identifier in ("transcriptomics_sample_id", "sample_id"):
        if identifier in frame:
            frame = frame.set_index(identifier)
            break
    if isinstance(frame.index, pd.RangeIndex):
        raise ValueError("Selected-gene CPM artifact lacks transcriptomics sample identifiers")
    frame.index = frame.index.astype(str)
    frame.index.name = "transcriptomics_sample_id"
    if frame.index.duplicated().any():
        raise AssertionError("Duplicate transcriptomics sample ID in CPM artifact")
    return frame


def _metadata(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    unnamed = [column for column in frame if str(column).startswith("Unnamed")]
    # The official release has a trailing unlabeled field: the preceding named
    # column contains CS... cluster IDs and this field contains human-readable labels.
    if unnamed:
        frame = frame.rename(
            columns={"Tree_first_cl_label": "transcriptomic_cluster_id", unnamed[-1]: "metadata_cluster"}
        )
    else:
        frame = frame.rename(columns={"Tree_first_cl_label": "metadata_cluster"})
        frame["transcriptomic_cluster_id"] = pd.NA
    required = {
        "cell_specimen_id",
        "donor_id",
        "ephys_session_id",
        "transcriptomics_sample_id",
        "metadata_cluster",
    }
    if missing := required.difference(frame):
        raise ValueError(f"Metadata missing columns: {sorted(missing)}")
    if frame.cell_specimen_id.duplicated().any():
        raise AssertionError("Metadata cell_specimen_id is not unique")
    return frame


def main() -> int:
    raw = ROOT / "data/raw"
    interim = ROOT / "data/interim"
    processed = ROOT / "data/processed"
    tables = ROOT / "results/tables"
    processed.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)

    modules = load_gene_modules(ROOT / "config/gene_modules.yaml")
    cpm_path = interim / "module_gene_cpm.parquet"
    if not cpm_path.is_file():
        archive = raw / "20200513_Mouse_PatchSeq_Release_count.v2.csv.tar"
        if not archive.is_file():
            raise FileNotFoundError(
                f"{archive.relative_to(ROOT)} is required; run 00_download.py --include-transcriptomics"
            )
        requested = [gene for module in modules.values() for gene in module]
        counts_gene_by_sample, library_sizes, extraction_audit = extract_selected_counts(
            archive, requested
        )
        if (library_sizes <= 0).any():
            raise AssertionError("Count release contains a zero/negative library size")
        counts = counts_gene_by_sample.T
        counts.index.name = "transcriptomics_sample_id"
        cpm_generated = counts_gene_by_sample.div(library_sizes, axis=1).mul(1_000_000.0).T
        cpm_generated.index.name = "transcriptomics_sample_id"
        counts.to_parquet(interim / "module_gene_counts.parquet")
        cpm_generated.to_parquet(cpm_path)
        library_sizes.rename("library_size_counts").rename_axis("transcriptomics_sample_id").reset_index().to_csv(
            interim / "library_sizes.csv", index=False
        )
        (interim / "count_extraction_audit.json").write_text(
            json.dumps(extraction_audit, indent=2), encoding="utf-8"
        )
    cpm = _load_cpm(cpm_path)
    features = load_feature_names(ROOT / "config/ephys_features.yaml")

    mat = loadmat(raw / "PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat", squeeze_me=True, struct_as_record=False)
    mat_names = [str(value) for value in np.asarray(mat["feature_name"]).ravel()]
    if mat_names != features:
        raise AssertionError("Frozen feature order does not exactly match MAT E_feature columns")
    standardized = np.asarray(mat["E_feature"], dtype=float)
    feature_mean = np.asarray(mat["feature_mean"], dtype=float).reshape(1, -1)
    feature_std = np.asarray(mat["feature_std"], dtype=float).reshape(1, -1)
    if feature_mean.shape[1] != len(mat_names) or feature_std.shape[1] != len(mat_names):
        raise AssertionError("MAT inversion parameters do not match E_feature columns")
    # Official preprocessing was z=(x-mean)/std followed by clipping to [-6, 6].
    # Physical units are recovered except that clipped extrema remain censored.
    physical_values = standardized * feature_std + feature_mean
    ephys = pd.DataFrame(physical_values, columns=mat_names)
    ephys.insert(0, "cell_specimen_id", np.asarray(mat["E_spec_id_label"], dtype=np.int64))
    ephys["mat_sample_id"] = np.asarray(mat["sample_id"]).astype(str)
    ephys["mat_cluster"] = np.asarray(mat["cluster"]).astype(str)
    if ephys.cell_specimen_id.duplicated().any():
        raise AssertionError("MAT E specimen IDs are not unique")

    metadata_candidates = sorted(raw.glob("*patchseq_metadata_mouse.csv"))
    if len(metadata_candidates) != 1:
        raise FileNotFoundError(f"Expected one extracted mouse metadata CSV, found {metadata_candidates}")
    metadata = _metadata(metadata_candidates[0])
    merged = ephys.merge(metadata, on="cell_specimen_id", how="left", validate="one_to_one", indicator=True)
    metadata_missing = merged.loc[merged._merge.ne("both")].copy()
    matched = merged.loc[merged._merge.eq("both")].drop(columns="_merge").copy()
    matched["transcriptomics_sample_id"] = matched.transcriptomics_sample_id.astype(str)
    count_missing = matched.loc[~matched.transcriptomics_sample_id.isin(cpm.index)].copy()
    matched = matched.loc[matched.transcriptomics_sample_id.isin(cpm.index)].copy()
    matched["canonical_cell_id"] = matched.cell_specimen_id.astype(str)
    matched["group_id"] = matched.donor_id.astype(str)
    matched["subclass"] = matched.mat_cluster.str.split().str[0]
    public_available = matched.metadata_cluster.notna()
    matched["metadata_subclass"] = matched.metadata_cluster.str.split().str[0]
    matched["cluster_label_agreement"] = pd.Series(pd.NA, index=matched.index, dtype="boolean")
    matched.loc[public_available, "cluster_label_agreement"] = matched.loc[
        public_available, "mat_cluster"
    ].eq(matched.loc[public_available, "metadata_cluster"])
    matched["subclass_agreement"] = pd.Series(pd.NA, index=matched.index, dtype="boolean")
    matched.loc[public_available, "subclass_agreement"] = matched.loc[
        public_available, "subclass"
    ].eq(matched.loc[public_available, "metadata_subclass"])
    if matched.canonical_cell_id.duplicated().any():
        raise AssertionError("Final canonical cell IDs are not unique")
    if matched.transcriptomics_sample_id.duplicated().any():
        raise AssertionError("Final transcriptomics sample IDs are not unique")

    crosswalk_columns = [
        "canonical_cell_id",
        "cell_specimen_id",
        "transcriptomics_sample_id",
        "ephys_session_id",
        "donor_id",
        "group_id",
        "mat_sample_id",
        "mat_cluster",
        "transcriptomic_cluster_id",
        "metadata_cluster",
        "cluster_label_agreement",
        "subclass",
        "metadata_subclass",
        "subclass_agreement",
    ]
    matched[crosswalk_columns].to_csv(tables / "cell_id_crosswalk.csv", index=False)
    mismatch_parts = []
    if len(metadata_missing):
        mismatch = metadata_missing[["cell_specimen_id", "mat_sample_id", "mat_cluster"]].copy()
        mismatch["reason"] = "cell_specimen_id absent from official metadata"
        mismatch_parts.append(mismatch)
    if len(count_missing):
        mismatch = count_missing[
            ["cell_specimen_id", "mat_sample_id", "mat_cluster", "transcriptomics_sample_id"]
        ].copy()
        mismatch["reason"] = "transcriptomics_sample_id absent from full count release"
        mismatch_parts.append(mismatch)
    mismatches = pd.concat(mismatch_parts, ignore_index=True) if mismatch_parts else pd.DataFrame(columns=["reason"])
    mismatches.to_csv(tables / "cell_id_mismatches.csv", index=False)
    matched.loc[matched.cluster_label_agreement.eq(False), crosswalk_columns].to_csv(
        tables / "cluster_label_mismatches.csv", index=False
    )

    selected_cpm = cpm.loc[matched.transcriptomics_sample_id].copy()
    selected_cpm.index = matched.canonical_cell_id.to_numpy()
    selected_cpm.index.name = "canonical_cell_id"
    scores, inventory = score_modules(selected_cpm, modules)
    dominance = module_dominance(selected_cpm, scores, modules)
    scores.reset_index().to_parquet(tables / "module_scores.parquet", index=False)
    inventory.to_csv(tables / "module_score_summary.csv", index=False)
    dominance.to_csv(tables / "module_dominance.csv", index=False)

    modeling = matched[
        [
            "canonical_cell_id",
            "group_id",
            "donor_id",
            "subclass",
            "mat_cluster",
            "metadata_cluster",
            "biological_sex",
            "age",
            "structure",
            *features,
        ]
    ].copy()
    score_lookup = scores.copy()
    for module in MODULES:
        modeling[f"target_{module}"] = modeling.canonical_cell_id.map(score_lookup[module])
    if modeling.canonical_cell_id.duplicated().any():
        raise AssertionError("Duplicate modeling row")
    if modeling[[f"target_{module}" for module in MODULES]].isna().any().any():
        raise AssertionError("Target missing after exact-ID alignment")
    modeling.to_parquet(processed / "modeling_table.parquet", index=False)

    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    fold_assignment = np.full(len(modeling), -1, dtype=int)
    for fold, (_, test_idx) in enumerate(
        splitter.split(modeling[features], y=modeling.subclass, groups=modeling.group_id)
    ):
        fold_assignment[test_idx] = fold
    if set(fold_assignment) != {0, 1, 2}:
        raise AssertionError("Frozen fold assignment is incomplete")
    folds = modeling[["canonical_cell_id", "group_id", "subclass"]].copy()
    folds["fold"] = fold_assignment
    if folds.groupby("group_id").fold.nunique().max() != 1:
        raise AssertionError("Donor group spans multiple outer folds")
    folds.to_csv(tables / "cv_folds.csv", index=False)
    folds.groupby(["fold", "subclass"]).size().rename("n").reset_index().to_csv(
        tables / "fold_composition.csv", index=False
    )

    summary = {
        "mat_cells": len(ephys),
        "metadata_matches": len(merged) - len(metadata_missing),
        "count_matches": len(matched),
        "final_n": len(modeling),
        "groups": int(modeling.group_id.nunique()),
        "subclasses": modeling.subclass.value_counts().to_dict(),
        "features": len(features),
        "modules": list(MODULES),
        "metadata_mismatches": len(metadata_missing),
        "count_mismatches": len(count_missing),
        "public_cluster_labels_missing": int(matched.metadata_cluster.isna().sum()),
        "cluster_label_mismatches": int(matched.cluster_label_agreement.eq(False).sum()),
        "broad_subclass_mismatches": int(matched.subclass_agreement.eq(False).sum()),
    }
    (tables / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(main())
