"""Verified data loading, identifier joins, and memory-bounded count extraction."""

from __future__ import annotations

import hashlib
import json
import re
import tarfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.io import loadmat


MAT_FILENAME = "PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat"
PUBLIC_METADATA_FILENAME = "20200711_patchseq_metadata_mouse.csv"
COUNT_ARCHIVE_FILENAME = "20200513_Mouse_PatchSeq_Release_count.v2.csv.tar"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_release_mat(path: Path) -> dict[str, np.ndarray]:
    """Load the MATLAB v5 release while squeezing singleton dimensions."""
    data = loadmat(Path(path), squeeze_me=True, struct_as_record=False)
    return {key: value for key, value in data.items() if not key.startswith("__")}


def one_dimensional_strings(value: np.ndarray) -> np.ndarray:
    return np.asarray([str(item) for item in np.atleast_1d(value)], dtype=object)


def load_ephys_name_map(path: Path) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        mapping = json.load(handle)
    if not isinstance(mapping, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in mapping.items()
    ):
        raise ValueError("E_names.json must be a string-to-string JSON object")
    return mapping


def inverse_upstream_ephys_zscore(data: dict[str, np.ndarray]) -> np.ndarray:
    """Invert the recorded affine transform for E_feature.

    The official upstream implementation first estimates mean/std from values
    between each feature's 1st and 99th percentiles, z-scores the full cohort,
    and clips z to [-6, 6]. Thus the inverse is exact except that observations
    at either boundary remain censored and cannot recover the original outlier.
    """
    z = np.asarray(data["E_feature"], dtype=np.float64)
    mean = np.asarray(data["feature_mean"], dtype=np.float64).reshape(1, -1)
    std = np.asarray(data["feature_std"], dtype=np.float64).reshape(1, -1)
    if z.shape[1] != mean.shape[1] or z.shape[1] != std.shape[1]:
        raise ValueError("E_feature/feature_mean/feature_std dimensions disagree")
    if np.any(~np.isfinite(std)) or np.any(std <= 0):
        raise ValueError("feature_std must contain positive finite values")
    return z * std + mean


def ephys_frame(data: dict[str, np.ndarray], raw_units: bool = False) -> pd.DataFrame:
    names = one_dimensional_strings(data["feature_name"])
    values = inverse_upstream_ephys_zscore(data) if raw_units else np.asarray(data["E_feature"])
    frame = pd.DataFrame(values, columns=names)
    frame.insert(0, "canonical_cell_id", np.asarray(data["E_spec_id_label"], dtype=np.int64))
    if frame["canonical_cell_id"].duplicated().any():
        raise ValueError("E_spec_id_label is not unique")
    return frame


def load_public_metadata(path: Path) -> pd.DataFrame:
    """Load and semantically repair the malformed final header fields.

    The archived official CSV has a trailing comma in its header. Data rows
    contain two final fields: an accession such as CS180626100018 and a
    cluster label such as ``Vip Gpc3 Slc18a3``. Pandas consequently labels
    them ``Tree_first_cl_label`` and ``Unnamed: 21``. We rename only after
    validating those observed value patterns; no positional cell matching is
    performed.
    """
    metadata = pd.read_csv(Path(path))
    expected_tail = {"Tree_first_cl_label", "Unnamed: 21"}
    if expected_tail.issubset(metadata.columns):
        accession = metadata["Tree_first_cl_label"].dropna().astype(str)
        labels = metadata["Unnamed: 21"].dropna().astype(str)
        if not accession.str.fullmatch(r"CS\d+").all():
            raise ValueError("Cannot safely interpret Tree_first_cl_label as accession IDs")
        if labels.empty or labels.str.fullmatch(r"CS\d+").any():
            raise ValueError("Cannot safely interpret trailing values as cluster labels")
        metadata = metadata.rename(
            columns={
                "Tree_first_cl_label": "transcriptomic_cluster_id",
                "Unnamed: 21": "transcriptomic_cluster_label",
            }
        )
    required = {
        "cell_specimen_id",
        "donor_id",
        "ephys_session_id",
        "transcriptomics_sample_id",
        "transcriptomic_cluster_id",
        "transcriptomic_cluster_label",
    }
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Public metadata missing required fields: {sorted(missing)}")
    if metadata["cell_specimen_id"].isna().any() or metadata["cell_specimen_id"].duplicated().any():
        raise ValueError("Public metadata cell_specimen_id is missing or non-unique")
    metadata["cell_specimen_id"] = metadata["cell_specimen_id"].astype(np.int64)
    return metadata


def build_cell_crosswalk(
    data: dict[str, np.ndarray], metadata: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join T and E modalities by released specimen ID, then public metadata.

    Row equality is checked as supporting evidence but is never used as the
    join key. The returned crosswalk retains all released MAT cells, including
    explicit metadata mismatches.
    """
    t_ids = np.asarray(data["T_spec_id_label"], dtype=np.int64)
    e_ids = np.asarray(data["E_spec_id_label"], dtype=np.int64)
    sample_ids = one_dimensional_strings(data["sample_id"])
    n = np.asarray(data["T_dat"]).shape[0]
    if not (len(t_ids) == len(e_ids) == len(sample_ids) == n):
        raise ValueError("Released T/E identifier lengths do not match matrix rows")

    t = pd.DataFrame(
        {
            "canonical_cell_id": t_ids,
            "transcriptomic_mat_row": np.arange(n, dtype=np.int64),
            "mat_internal_sample_id": sample_ids,
            "cluster": one_dimensional_strings(data["cluster"]),
            "cluster_id": np.asarray(data["cluster_id"], dtype=np.int64),
            "cluster_color": one_dimensional_strings(data["cluster_color"]),
            "map_confidence": one_dimensional_strings(data["map_conf"]),
            "T_ispaired": np.asarray(data["T_ispaired"], dtype=float),
        }
    )
    e = pd.DataFrame(
        {
            "canonical_cell_id": e_ids,
            "ephys_mat_row": np.arange(n, dtype=np.int64),
            "E_ispaired": np.asarray(data["E_ispaired"], dtype=float),
        }
    )
    for label, frame in (("T", t), ("E", e)):
        if frame["canonical_cell_id"].duplicated().any():
            raise ValueError(f"{label} specimen identifiers are not unique")
    crosswalk = t.merge(e, on="canonical_cell_id", how="outer", validate="one_to_one", indicator="mat_join")
    crosswalk["mat_modalities_match"] = crosswalk["mat_join"].eq("both")

    if metadata is not None:
        keep = [
            "cell_specimen_id",
            "donor_id",
            "donor_name",
            "biological_sex",
            "age",
            "structure",
            "ephys_session_id",
            "transcriptomics_sample_id",
            "transcriptomics_batch",
            "transcriptomic_cluster_id",
            "transcriptomic_cluster_label",
        ]
        public = metadata[[column for column in keep if column in metadata.columns]].copy()
        public = public.rename(columns={"cell_specimen_id": "canonical_cell_id"})
        crosswalk = crosswalk.merge(
            public,
            on="canonical_cell_id",
            how="left",
            validate="one_to_one",
            indicator="public_metadata_join",
        )
        crosswalk["public_metadata_match"] = crosswalk["public_metadata_join"].eq("both")
        public_label = crosswalk["transcriptomic_cluster_label"]
        crosswalk["public_cluster_label_available"] = public_label.notna()
        crosswalk["cluster_label_agrees"] = pd.Series(pd.NA, index=crosswalk.index, dtype="boolean")
        crosswalk.loc[crosswalk["public_cluster_label_available"], "cluster_label_agrees"] = public_label.eq(
            crosswalk["cluster"]
        )
        crosswalk["mat_subclass"] = crosswalk["cluster"].str.split().str[0]
        crosswalk["public_subclass"] = public_label.str.split().str[0]
        crosswalk["broad_subclass_agrees"] = pd.Series(pd.NA, index=crosswalk.index, dtype="boolean")
        crosswalk.loc[crosswalk["public_cluster_label_available"], "broad_subclass_agrees"] = crosswalk[
            "public_subclass"
        ].eq(crosswalk["mat_subclass"])
        # Primary subclass follows the MAT branch contemporaneous with E_feature.
        crosswalk["subclass"] = crosswalk["mat_subclass"]
    else:
        crosswalk["public_metadata_match"] = False
        crosswalk["public_cluster_label_available"] = False
        crosswalk["mat_subclass"] = crosswalk["cluster"].str.split().str[0]
        crosswalk["public_subclass"] = np.nan
        crosswalk["cluster_label_agrees"] = pd.Series(pd.NA, index=crosswalk.index, dtype="boolean")
        crosswalk["broad_subclass_agrees"] = pd.Series(pd.NA, index=crosswalk.index, dtype="boolean")
        crosswalk["subclass"] = crosswalk["mat_subclass"]

    fine_label_disagreement = (
        crosswalk["public_cluster_label_available"]
        & ~crosswalk["cluster_label_agrees"].fillna(False)
    )
    mismatch = crosswalk.loc[
        (~crosswalk["mat_modalities_match"])
        | (~crosswalk["public_metadata_match"])
        | (~crosswalk["public_cluster_label_available"])
        | fine_label_disagreement
    ].copy()
    mismatch["mismatch_reason"] = ""
    mismatch.loc[~mismatch["mat_modalities_match"], "mismatch_reason"] += "MAT_MODALITY_ID_MISSING;"
    mismatch.loc[~mismatch["public_metadata_match"], "mismatch_reason"] += "PUBLIC_METADATA_ID_MISSING;"
    mismatch.loc[
        mismatch["public_metadata_match"] & ~mismatch["public_cluster_label_available"], "mismatch_reason"
    ] += "PUBLIC_METADATA_CLUSTER_LABEL_MISSING;"
    if "cluster_label_agrees" in mismatch:
        mismatch.loc[
            mismatch["public_cluster_label_available"]
            & ~mismatch["cluster_label_agrees"].fillna(False),
            "mismatch_reason",
        ] += "PUBLIC_VS_MAT_CLUSTER_LABEL_DISAGREEMENT;"
    if "broad_subclass_agrees" in mismatch:
        mismatch.loc[
            mismatch["public_cluster_label_available"]
            & ~mismatch["broad_subclass_agrees"].fillna(False),
            "mismatch_reason",
        ] += "PUBLIC_VS_MAT_BROAD_SUBCLASS_DISAGREEMENT;"
    mismatch["mismatch_reason"] = mismatch["mismatch_reason"].str.rstrip(";")
    return crosswalk, mismatch


def _single_csv_member(archive: tarfile.TarFile) -> tarfile.TarInfo:
    members = [m for m in archive.getmembers() if m.isfile() and m.name.lower().endswith(".csv")]
    if len(members) != 1:
        raise ValueError(f"Expected exactly one CSV member, observed {[m.name for m in members]}")
    member = members[0]
    if Path(member.name).is_absolute() or ".." in Path(member.name).parts:
        raise ValueError(f"Unsafe tar member path: {member.name}")
    return member


def inspect_count_archive(path: Path) -> dict[str, object]:
    with tarfile.open(Path(path), mode="r:*") as archive:
        member = _single_csv_member(archive)
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("Could not open count CSV member")
        header = handle.readline().decode("utf-8-sig").rstrip("\r\n").split(",")
    return {
        "archive_bytes": Path(path).stat().st_size,
        "member_name": member.name,
        "member_bytes": member.size,
        "first_column": header[0],
        "sample_n": len(header) - 1,
        "sample_ids_unique": len(set(header[1:])) == len(header[1:]),
        "sample_ids": header[1:],
    }


def extract_selected_counts(
    archive_path: Path,
    requested_genes: Iterable[str],
    chunksize: int = 256,
) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    """One-pass extraction of requested rows and full per-cell library sizes.

    The CSV is consumed directly from the tar member. Irrelevant gene rows are
    summed into library sizes then discarded; only requested genes are retained.
    """
    requested = tuple(dict.fromkeys(str(gene) for gene in requested_genes))
    requested_set = set(requested)
    selected: list[pd.DataFrame] = []
    library_sizes: pd.Series | None = None
    rows_seen = 0
    duplicate_gene_rows: list[str] = []
    selected_seen: set[str] = set()
    with tarfile.open(Path(archive_path), mode="r:*") as archive:
        member = _single_csv_member(archive)
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("Could not open count CSV member")
        for chunk in pd.read_csv(handle, index_col=0, chunksize=chunksize):
            chunk.index = chunk.index.astype(str)
            if not all(np.issubdtype(dtype, np.number) for dtype in chunk.dtypes):
                raise ValueError("Count matrix contains non-numeric sample columns")
            numeric = chunk.to_numpy(dtype=np.float64, copy=False)
            if (numeric < 0).any():
                raise ValueError("Count matrix contains negative values")
            block_sum = pd.Series(numeric.sum(axis=0), index=chunk.columns, dtype=np.float64)
            library_sizes = block_sum if library_sizes is None else library_sizes.add(block_sum, fill_value=0)
            rows_seen += len(chunk)
            keep = chunk.index.isin(requested_set)
            if keep.any():
                hit = chunk.loc[keep]
                duplicated = selected_seen.intersection(hit.index)
                duplicate_gene_rows.extend(sorted(duplicated))
                selected_seen.update(hit.index)
                selected.append(hit)
    if library_sizes is None:
        raise ValueError("Count matrix was empty")
    counts = pd.concat(selected, axis=0) if selected else pd.DataFrame(columns=library_sizes.index)
    if counts.index.duplicated().any() or duplicate_gene_rows:
        raise ValueError(f"Duplicate requested gene rows: {sorted(set(duplicate_gene_rows))}")
    counts = counts.reindex([gene for gene in requested if gene in counts.index])
    audit = {
        "rows_seen": rows_seen,
        "samples_seen": len(library_sizes),
        "requested_gene_n": len(requested),
        "present_gene_n": len(counts),
        "missing_genes": [gene for gene in requested if gene not in counts.index],
        "tar_member": member.name,
    }
    return counts, library_sizes, audit
