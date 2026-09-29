#!/usr/bin/env python3
"""Audit Allen cplAE-TE MAT/features and build evidence-based ID artifacts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.io import whosmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import (  # noqa: E402
    MAT_FILENAME,
    PUBLIC_METADATA_FILENAME,
    build_cell_crosswalk,
    ephys_frame,
    load_ephys_name_map,
    load_public_metadata,
    load_release_mat,
    one_dimensional_strings,
    sha256,
)


BRANCH_COMMIT = "6c0be92e7a919321a4b877623354ee48d5f75921"
MAT_GIT_BLOB = "6270922577fca6005225df719c4358b7c80ca01b"
UPSTREAM_NOTEBOOK_URL = (
    "https://github.com/AllenInstitute/coupledAE-patchseq/blob/cplAE-TE/"
    "notebooks/data_proc_E.ipynb"
)
UPSTREAM_HELPER_URL = (
    "https://github.com/AllenInstitute/coupledAE-patchseq/blob/cplAE-TE/"
    "cplAE_TE/utils/preproc_helpers.py#L45-L99"
)


UNITS = {
    "ap_1_threshold_v_short_square": "mV",
    "ap_1_peak_v_short_square": "mV",
    "ap_1_upstroke_short_square": "V/s (equivalent mV/ms)",
    "ap_1_downstroke_short_square": "V/s (equivalent mV/ms)",
    "ap_1_upstroke_downstroke_ratio_short_square": "dimensionless",
    "ap_1_width_short_square": "s",
    "ap_1_fast_trough_v_short_square": "mV",
    "short_square_current": "pA",
    "input_resistance": "MOhm",
    "tau": "s",
    "v_baseline": "mV",
    "sag_nearest_minus_100": "dimensionless",
    "sag_measured_at": "mV",
    "rheobase_i": "pA",
    "ap_1_threshold_v_0_long_square": "mV",
    "ap_1_peak_v_0_long_square": "mV",
    "ap_1_upstroke_0_long_square": "V/s (equivalent mV/ms)",
    "ap_1_downstroke_0_long_square": "V/s (equivalent mV/ms)",
    "ap_1_upstroke_downstroke_ratio_0_long_square": "dimensionless",
    "ap_1_width_0_long_square": "s",
    "ap_1_fast_trough_v_0_long_square": "mV",
    "avg_rate_0_long_square": "Hz",
    "latency_0_long_square": "s",
    "stimulus_amplitude_0_long_square": "pA",
}


MEANINGS = {
    "ap_1_threshold_v_short_square": "Voltage threshold of the first short-square action potential.",
    "ap_1_peak_v_short_square": "Peak voltage of the first short-square action potential.",
    "ap_1_upstroke_short_square": "Maximum rising-phase dV/dt of the first short-square action potential.",
    "ap_1_downstroke_short_square": "Maximum-magnitude falling-phase dV/dt of the first short-square action potential.",
    "ap_1_upstroke_downstroke_ratio_short_square": "First short-square AP upstroke divided by downstroke magnitude.",
    "ap_1_width_short_square": "Width of the first short-square action potential.",
    "ap_1_fast_trough_v_short_square": "Fast-trough voltage after the first short-square action potential.",
    "short_square_current": "Current amplitude of the selected short-square sweep.",
    "input_resistance": "Membrane input resistance estimated from subthreshold responses.",
    "tau": "Membrane time constant from subthreshold response fitting.",
    "v_baseline": "Baseline membrane voltage before stimulation.",
    "sag_nearest_minus_100": "Sag response ratio/value from the sweep nearest -100 pA.",
    "sag_measured_at": "Voltage at which sag was measured.",
    "rheobase_i": "Minimum long-square current eliciting an action potential.",
    "ap_1_threshold_v_0_long_square": "Threshold voltage of the first AP in the rheobase/first spiking long-square sweep.",
    "ap_1_peak_v_0_long_square": "Peak voltage of that first long-square action potential.",
    "ap_1_upstroke_0_long_square": "Maximum rising-phase dV/dt of that first long-square action potential.",
    "ap_1_downstroke_0_long_square": "Maximum-magnitude falling-phase dV/dt of that first long-square action potential.",
    "ap_1_upstroke_downstroke_ratio_0_long_square": "That AP's upstroke divided by downstroke magnitude.",
    "ap_1_width_0_long_square": "Width of that first long-square action potential.",
    "ap_1_fast_trough_v_0_long_square": "Fast-trough voltage after that first long-square action potential.",
    "avg_rate_0_long_square": "Average firing rate in the first/rheobase long-square spiking sweep.",
    "latency_0_long_square": "Latency from stimulus onset to first AP in that long-square sweep.",
    "stimulus_amplitude_0_long_square": "Stimulus amplitude of that long-square sweep.",
}


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.data-audit.tmp")
    for attempt in range(20):
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False)
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.2)
    os.replace(temporary, path)


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.data-audit.tmp")
    for attempt in range(20):
        try:
            temporary.write_text(content, encoding="utf-8")
            break
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.2)
    os.replace(temporary, path)


def matrix_summary(array: np.ndarray) -> dict[str, object]:
    values = np.asarray(array)
    numeric = np.issubdtype(values.dtype, np.number)
    result: dict[str, object] = {"shape": list(values.shape), "dtype": str(values.dtype)}
    if numeric:
        result.update(
            {
                "finite_fraction": float(np.isfinite(values).mean()),
                "nan_n": int(np.isnan(values).sum()) if np.issubdtype(values.dtype, np.floating) else 0,
                "min": float(np.nanmin(values)),
                "max": float(np.nanmax(values)),
            }
        )
    return result


def feature_dictionary(data: dict[str, np.ndarray], name_map: dict[str, str]) -> pd.DataFrame:
    pc_names = one_dimensional_strings(data["pc_name"])
    ipfx_names = one_dimensional_strings(data["feature_name"])
    pc_values = np.asarray(data["E_pc_scaled"], dtype=float)
    ipfx_values = np.asarray(data["E_feature"], dtype=float)
    means = np.asarray(data["feature_mean"], dtype=float)
    stds = np.asarray(data["feature_std"], dtype=float)
    rows: list[dict[str, object]] = []
    for index, name in enumerate(pc_names):
        values = pc_values[:, index]
        rows.append(
            {
                "feature_order": index,
                "feature_key": name,
                "display_name": name_map[name],
                "classification": "SPC/WAVEFORM_PC",
                "selected_primary": False,
                "unit": "scaled PC score",
                "meaning": "Sparse principal-component score derived from an electrophysiological waveform/time series.",
                "source": "Allen cplAE-TE E_pc_scaled / data_proc_E.ipynb",
                "missing_n": int(np.isnan(values).sum()),
                "missing_fraction": float(np.isnan(values).mean()),
                "upstream_center": np.nan,
                "upstream_scale": np.nan,
                "boundary_low_n": np.nan,
                "boundary_high_n": np.nan,
                "scaling_status": "Upstream waveform-PC scaling; excluded from primary interpretable feature set.",
            }
        )
    for index, name in enumerate(ipfx_names):
        values = ipfx_values[:, index]
        rows.append(
            {
                "feature_order": len(pc_names) + index,
                "feature_key": name,
                "display_name": name_map[name],
                "classification": "INTRINSIC_IPFX",
                "selected_primary": True,
                "unit": UNITS[name],
                "meaning": MEANINGS[name],
                "source": "Allen IPFX feature table via official cplAE-TE data_proc_E.ipynb",
                "missing_n": int(np.isnan(values).sum()),
                "missing_fraction": float(np.isnan(values).mean()),
                "upstream_center": means[index],
                "upstream_scale": stds[index],
                "boundary_low_n": int(np.isclose(values, -6.0, equal_nan=False).sum()),
                "boundary_high_n": int(np.isclose(values, 6.0, equal_nan=False).sum()),
                "scaling_status": (
                    "Globally standardized upstream using central 1st-99th percentile mean/std, "
                    "then clipped to [-6,6]; affine inverse available but boundary outliers are censored."
                ),
            }
        )
    return pd.DataFrame(rows)


def write_feature_config(dictionary: pd.DataFrame, path: Path) -> None:
    selected = dictionary.loc[dictionary["selected_primary"]].copy()
    entries = []
    for _, row in selected.iterrows():
        entries.append(
            {
                "name": row["feature_key"],
                "display_name": row["display_name"],
                "unit": row["unit"],
                "meaning": row["meaning"],
                "source": row["source"],
                "missing_n": int(row["missing_n"]),
                "missing_fraction": float(row["missing_fraction"]),
                "upstream_center": float(row["upstream_center"]),
                "upstream_scale": float(row["upstream_scale"]),
                "boundary_low_n": int(row["boundary_low_n"]),
                "boundary_high_n": int(row["boundary_high_n"]),
            }
        )
    config = {
        "schema_version": "1.0",
        "feature_set": {
            "name": "allen_cplae_te_intrinsic_ipfx_24",
            "frozen": True,
            "feature_n": len(entries),
            "matrix_key": "E_feature",
            "ordered_as_released": True,
            "primary_excludes": ["SPC/WAVEFORM_PC"],
            "stored_value_policy": (
                "Use inverse value z*feature_std+feature_mean for physical-unit reporting/model input. "
                "Values at z=-6 or z=6 are censored boundaries, not recovered original outliers."
            ),
            "leakage_warning": (
                "Center/scale parameters were estimated upstream on the full 3411-cell release, "
                "not fold-locally. Do not claim upstream preprocessing is leakage-free."
            ),
            "features": entries,
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, yaml.safe_dump(config, sort_keys=False, allow_unicode=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw = root / "data" / "raw"
    tables = root / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    mat_path = raw / MAT_FILENAME
    names_path = raw / "E_names.json"
    metadata_path = raw / PUBLIC_METADATA_FILENAME

    data = load_release_mat(mat_path)
    name_map = load_ephys_name_map(names_path)
    feature_names = one_dimensional_strings(data["feature_name"])
    pc_names = one_dimensional_strings(data["pc_name"])
    combined_names = list(pc_names) + list(feature_names)
    if len(name_map) != len(combined_names) or set(name_map) != set(combined_names):
        raise AssertionError("E_names.json keys do not exactly equal pc_name + feature_name")
    if list(name_map.keys()) != combined_names:
        raise AssertionError("E_names.json insertion order differs from pc_name + feature_name")

    dictionary = feature_dictionary(data, name_map)
    atomic_to_csv(dictionary, tables / "ephys_feature_dictionary.csv")
    write_feature_config(dictionary, root / "config" / "ephys_features.yaml")

    metadata = load_public_metadata(metadata_path) if metadata_path.exists() else None
    crosswalk, mismatches = build_cell_crosswalk(data, metadata)
    atomic_to_csv(crosswalk, tables / "cell_id_crosswalk_all.csv")
    authoritative_crosswalk = crosswalk.loc[
        crosswalk["mat_modalities_match"] & crosswalk["public_metadata_match"]
    ].copy()
    required_crosswalk_fields = [
        "canonical_cell_id",
        "transcriptomics_sample_id",
        "ephys_session_id",
        "donor_id",
    ]
    if authoritative_crosswalk[required_crosswalk_fields].isna().any().any():
        raise AssertionError("Authoritative crosswalk contains null canonical/public identifiers")
    for field in required_crosswalk_fields[:3]:
        if authoritative_crosswalk[field].duplicated().any():
            raise AssertionError(f"Authoritative crosswalk field is non-unique: {field}")
    atomic_to_csv(authoritative_crosswalk, tables / "cell_id_crosswalk.csv")
    atomic_to_csv(mismatches, tables / "cell_id_mismatches.csv")

    t_ids = np.asarray(data["T_spec_id_label"], dtype=np.int64)
    e_ids = np.asarray(data["E_spec_id_label"], dtype=np.int64)
    z = np.asarray(data["E_feature"], dtype=float)
    recovered = ephys_frame(data, raw_units=True).drop(columns="canonical_cell_id").to_numpy()
    restated = (recovered - np.asarray(data["feature_mean"])) / np.asarray(data["feature_std"])
    finite = np.isfinite(z)
    public_match = int(crosswalk["public_metadata_match"].sum())
    public_label_available = (
        int(
            (
                crosswalk["public_metadata_match"]
                & crosswalk["public_cluster_label_available"]
            ).sum()
        )
        if metadata is not None
        else None
    )
    cluster_agree = (
        int(crosswalk["cluster_label_agrees"].fillna(False).sum())
        if metadata is not None
        else None
    )
    broad_subclass_agree = (
        int(crosswalk["broad_subclass_agrees"].fillna(False).sum())
        if metadata is not None
        else None
    )
    broad_disagreement_pairs = (
        crosswalk.loc[
            crosswalk["public_cluster_label_available"]
            & ~crosswalk["broad_subclass_agrees"].fillna(False),
            ["mat_subclass", "public_subclass"],
        ]
        .value_counts()
        .rename("n")
        .reset_index()
        .to_dict(orient="records")
        if metadata is not None
        else []
    )
    whos = {name: {"shape": list(shape), "matlab_class": cls} for name, shape, cls in whosmat(mat_path)}
    audit = {
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "repository": "https://github.com/AllenInstitute/coupledAE-patchseq",
            "branch": "cplAE-TE",
            "branch_commit": BRANCH_COMMIT,
            "mat_git_blob_sha1": MAT_GIT_BLOB,
            "upstream_ephys_notebook": UPSTREAM_NOTEBOOK_URL,
            "upstream_standardization_helper": UPSTREAM_HELPER_URL,
        },
        "file": {
            "path": mat_path.relative_to(root).as_posix(),
            "bytes": mat_path.stat().st_size,
            "sha256": sha256(mat_path),
            "format": "MATLAB 5.0 MAT-file",
        },
        "keys": {key: matrix_summary(value) for key, value in data.items()},
        "whosmat": whos,
        "matrices": {
            "T_dat": {
                **matrix_summary(data["T_dat"]),
                "interpretation": "Official README: log1p(CPM), feature-selected 1,252-gene matrix; not used for six frozen full modules.",
            },
            "E_feature": matrix_summary(data["E_feature"]),
            "E_pc_scaled": matrix_summary(data["E_pc_scaled"]),
            "E_pc_zscored": matrix_summary(data["E_pc_zscored"]),
        },
        "identifiers": {
            "T_spec_id_label_n": len(t_ids),
            "E_spec_id_label_n": len(e_ids),
            "T_unique": int(np.unique(t_ids).size),
            "E_unique": int(np.unique(e_ids).size),
            "released_arrays_same_order": bool(np.array_equal(t_ids, e_ids)),
            "released_sets_equal": bool(set(t_ids) == set(e_ids)),
            "all_T_ispaired_one": bool(np.all(np.asarray(data["T_ispaired"]) == 1)),
            "all_E_ispaired_one": bool(np.all(np.asarray(data["E_ispaired"]) == 1)),
            "join_method": "one-to-one merge on integer specimen ID; never row position",
            "official_code_semantics": (
                "data_proc_E.ipynb states spec_id_label identifies E cells, reindexes every E table "
                "to T_annotations.spec_id_label; data_proc_T.ipynb reindexes T_dat by T_annotations.sample_id."
            ),
            "public_metadata_matched_n": public_match,
            "public_metadata_unmatched_n": int(len(crosswalk) - public_match),
            "matched_donor_n": int(crosswalk.loc[crosswalk["public_metadata_match"], "donor_id"].nunique())
            if metadata is not None
            else None,
            "public_cluster_label_agreement_n": cluster_agree,
            "public_cluster_label_available_n": public_label_available,
            "public_cluster_label_missing_n": public_match - public_label_available
            if public_label_available is not None
            else None,
            "public_cluster_label_disagreement_n": public_label_available - cluster_agree
            if cluster_agree is not None and public_label_available is not None
            else None,
            "broad_subclass_agreement_n": broad_subclass_agree,
            "broad_subclass_disagreement_n": public_label_available - broad_subclass_agree
            if broad_subclass_agree is not None and public_label_available is not None
            else None,
            "broad_subclass_disagreement_pairs": broad_disagreement_pairs,
            "primary_subclass_source": (
                "First token of MAT cluster, because it is contemporaneous with the processed E_feature branch; "
                "public metadata cluster/subclass is retained for sensitivity/audit."
            ),
            "public_metadata_header_repair": (
                "Validated CS[0-9]+ values in mislabeled Tree_first_cl_label and transcriptomic "
                "type strings in trailing unnamed column; renamed to transcriptomic_cluster_id/label."
            )
            if metadata is not None
            else None,
        },
        "labels_and_metadata": {
            "cluster_unique_n": int(np.unique(one_dimensional_strings(data["cluster"])).size),
            "cluster_id_unique_n": int(np.unique(data["cluster_id"]).size),
            "map_confidence_values": sorted(set(one_dimensional_strings(data["map_conf"]))),
            "sample_id_unique_n": int(np.unique(one_dimensional_strings(data["sample_id"])).size),
            "note": "MAT contains no donor/subject IDs; donor_id is added only through the ID-joined public metadata.",
        },
        "features": {
            "combined_feature_n": len(combined_names),
            "waveform_pc_n": len(pc_names),
            "intrinsic_ipfx_n": len(feature_names),
            "pc_order": list(pc_names),
            "intrinsic_ipfx_order": list(feature_names),
            "E_names_exact_key_and_order_match": True,
            "classification_counts": dictionary["classification"].value_counts().to_dict(),
        },
        "upstream_scaling": {
            "E_feature_status": "globally standardized upstream and clipped",
            "formula": "z=(x-mean_central_1_to_99_percentile)/std_central_1_to_99_percentile; clip z to [-6,6]",
            "mean_key": "feature_mean",
            "std_key": "feature_std",
            "parameters_feature_n": len(data["feature_mean"]),
            "inverse_formula": "x_reconstructed=z*feature_std+feature_mean",
            "inverse_roundtrip_max_abs_error": float(np.max(np.abs(restated[finite] - z[finite]))),
            "censoring_warning": "Original values beyond +/-6 SD cannot be recovered exactly from the released MAT.",
            "fold_local_warning": "Upstream center/scale used the full release and was not fold-local.",
            "do_not_double_standardize_globally": True,
        },
    }
    atomic_write_text(
        tables / "mat_audit.json",
        json.dumps(jsonable(audit), indent=2, ensure_ascii=False) + "\n",
    )
    print(
        json.dumps(
            {
                "mat_cells": len(crosswalk),
                "authoritative_crosswalk_cells": len(authoritative_crosswalk),
                "public_metadata_matched": public_match,
                "mismatches": len(mismatches),
                "donors": audit["identifiers"]["matched_donor_n"],
                "intrinsic_ipfx": len(feature_names),
                "waveform_pc": len(pc_names),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
