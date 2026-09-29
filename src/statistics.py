"""Uncertainty and prediction-label permutation inference from saved OOF predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .evaluation import METRICS, regression_metrics


def cluster_bootstrap_metrics(
    frame: pd.DataFrame,
    *,
    group_column: str,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    """Resample complete groups and evaluate saved OOF predictions without refitting."""

    if group_column not in frame:
        raise KeyError(group_column)
    group_values = frame[group_column].astype(str).to_numpy()
    group_codes, groups = pd.factorize(group_values, sort=False)
    if len(groups) < 2:
        raise ValueError("Cluster bootstrap requires at least two groups")
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    base_indices = np.arange(len(frame), dtype=np.int64)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    for replicate in range(n_resamples):
        # A group bootstrap is exactly represented by the sampled multiplicity
        # of each group. Repeating row indices avoids constructing hundreds of
        # tiny DataFrames per replicate while preserving all metric semantics,
        # including rank ties for Spearman correlation.
        sampled_codes = rng.integers(0, len(groups), size=len(groups))
        multiplicity = np.bincount(sampled_codes, minlength=len(groups))
        draw_indices = np.repeat(base_indices, multiplicity[group_codes])
        rows.append(
            {
                "replicate": replicate,
                **regression_metrics(y_true[draw_indices], y_pred[draw_indices]),
            }
        )
    return pd.DataFrame(rows)


def summarize_bootstrap(draws: pd.DataFrame) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in METRICS:
        values = draws[metric].replace([np.inf, -np.inf], np.nan).dropna()
        summary[f"{metric}_ci_low"] = float(values.quantile(0.025)) if len(values) else np.nan
        summary[f"{metric}_ci_high"] = float(values.quantile(0.975)) if len(values) else np.nan
    return summary


def stratified_spearman_permutation(
    frame: pd.DataFrame,
    *,
    strata_column: str,
    n_permutations: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Permute labels within subclass strata; this deliberately does not refit models."""

    rng = np.random.default_rng(seed)
    y_true = frame["y_true"].to_numpy(dtype=float)
    y_pred = frame["y_pred"].to_numpy(dtype=float)
    strata = frame[strata_column].astype(str).to_numpy()
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        raise ValueError("Permutation input contains non-finite values")
    true_rank = rankdata(y_true).astype(float)
    pred_rank = rankdata(y_pred).astype(float)
    true_rank -= true_rank.mean()
    pred_rank -= pred_rank.mean()
    denominator = float(np.linalg.norm(true_rank) * np.linalg.norm(pred_rank))
    if denominator == 0:
        raise ValueError("Spearman permutation requires variable truth and predictions")
    observed = float(np.dot(true_rank, pred_rank) / denominator)
    null = np.empty(n_permutations, dtype=float)
    indices = [np.flatnonzero(strata == value) for value in np.unique(strata)]
    for iteration in range(n_permutations):
        permuted = true_rank.copy()
        for idx in indices:
            permuted[idx] = rng.permutation(permuted[idx])
        null[iteration] = np.dot(permuted, pred_rank) / denominator
    # Two-sided plus-one correction.
    p_value = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (n_permutations + 1)
    return {
        "observed_spearman": observed,
        "p_value_two_sided": float(p_value),
        "n_permutations": int(n_permutations),
        "null_mean": float(np.nanmean(null)),
        "null_sd": float(np.nanstd(null, ddof=1)),
        "test_type": "subclass-stratified OOF prediction-label permutation; no model refitting",
    }


def donor_level_subclass_adjusted_spearman_permutation(
    frame: pd.DataFrame,
    *,
    group_column: str,
    strata_column: str,
    n_permutations: int,
    seed: int,
) -> dict[str, float | int | str]:
    """Donor-unit sensitivity after removing cell-level subclass means.

    Both truth and saved OOF prediction are centered within subclass and then
    averaged once per donor/group. Permuting the donor-level truth residuals
    treats the grouped sampling unit as independent. This is deliberately a
    no-refit sensitivity analysis, not a replacement for the pre-specified
    cell-level subclass-stratified test.
    """

    required = {group_column, strata_column, "y_true", "y_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing donor-permutation columns: {sorted(missing)}")
    local = frame[list(required)].copy()
    local[group_column] = local[group_column].astype(str)
    local[strata_column] = local[strata_column].astype(str)
    if not np.isfinite(local[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise ValueError("Donor permutation input contains non-finite values")
    for column in ("y_true", "y_pred"):
        local[f"{column}_residual"] = local[column] - local.groupby(strata_column)[column].transform("mean")
    donor = (
        local.groupby(group_column, sort=True)
        .agg(
            y_true_residual=("y_true_residual", "mean"),
            y_pred_residual=("y_pred_residual", "mean"),
            n_cells=("y_true", "size"),
        )
        .reset_index()
    )
    if len(donor) < 3:
        raise ValueError("Donor-level permutation requires at least three groups")
    true_rank = rankdata(donor["y_true_residual"].to_numpy(dtype=float)).astype(float)
    pred_rank = rankdata(donor["y_pred_residual"].to_numpy(dtype=float)).astype(float)
    true_rank -= true_rank.mean()
    pred_rank -= pred_rank.mean()
    denominator = float(np.linalg.norm(true_rank) * np.linalg.norm(pred_rank))
    if denominator == 0:
        raise ValueError("Donor-level Spearman permutation requires variable residuals")
    observed = float(np.dot(true_rank, pred_rank) / denominator)
    rng = np.random.default_rng(seed)
    null = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        null[iteration] = np.dot(rng.permutation(true_rank), pred_rank) / denominator
    p_value = (1 + np.count_nonzero(np.abs(null) >= abs(observed))) / (n_permutations + 1)
    return {
        "observed_spearman": observed,
        "p_value_two_sided": float(p_value),
        "n_permutations": int(n_permutations),
        "n_groups": int(len(donor)),
        "n_cells": int(len(local)),
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null, ddof=1)),
        "test_type": (
            "donor-level permutation of subclass-adjusted OOF truth/prediction means; "
            "no model refitting"
        ),
    }
