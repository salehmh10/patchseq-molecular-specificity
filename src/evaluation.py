"""Out-of-fold regression metrics and consistency helpers."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


METRICS = ("r2", "mae", "rmse", "spearman")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_pred)
    if valid.sum() < 2:
        return {metric: math.nan for metric in METRICS}
    truth, prediction = y_true[valid], y_pred[valid]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        rho = spearmanr(truth, prediction).statistic
    return {
        "r2": float(r2_score(truth, prediction)),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "spearman": float(rho) if np.isfinite(rho) else math.nan,
    }


def performance_table(oof: pd.DataFrame) -> pd.DataFrame:
    required = {"module", "model", "analysis", "y_true", "y_pred"}
    missing = required.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF table lacks required columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for keys, frame in oof.groupby(["module", "model", "analysis"], sort=True, dropna=False):
        values = regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
        rows.append({"module": keys[0], "model": keys[1], "analysis": keys[2], "n": len(frame), **values})
    return pd.DataFrame(rows).sort_values(["analysis", "module", "model"]).reset_index(drop=True)


def assert_exact_oof_once(
    oof: pd.DataFrame,
    expected_cells: set[str],
    expected_keys: set[tuple[str, str, str]] | None = None,
) -> None:
    required = {
        "canonical_cell_id",
        "module",
        "model",
        "analysis",
        "fold",
        "y_true",
        "y_pred",
    }
    missing_columns = required.difference(oof.columns)
    if missing_columns:
        raise AssertionError(f"OOF table lacks required columns: {sorted(missing_columns)}")
    observed_keys = {
        tuple(values)
        for values in oof[["module", "model", "analysis"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    }
    if expected_keys is not None and observed_keys != expected_keys:
        missing = sorted(expected_keys - observed_keys)
        extra = sorted(observed_keys - expected_keys)
        raise AssertionError(f"Wrong OOF model grid: missing={missing}, extra={extra}")
    if not np.isfinite(oof[["y_true", "y_pred"]].to_numpy(dtype=float)).all():
        raise AssertionError("OOF truth/prediction values must all be finite")
    for keys, frame in oof.groupby(["module", "model", "analysis"], dropna=False):
        identifiers = frame["canonical_cell_id"].astype(str)
        if identifiers.duplicated().any():
            raise AssertionError(f"Duplicate OOF cell for {keys}")
        observed = set(identifiers)
        if observed != expected_cells:
            missing = len(expected_cells - observed)
            extra = len(observed - expected_cells)
            raise AssertionError(f"Incomplete OOF coverage for {keys}: missing={missing}, extra={extra}")


def fold_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, frame in oof.groupby(["module", "model", "analysis", "fold"], sort=True):
        values = regression_metrics(frame["y_true"].to_numpy(), frame["y_pred"].to_numpy())
        rows.append(
            {
                "module": keys[0],
                "model": keys[1],
                "analysis": keys[2],
                "fold": int(keys[3]),
                "n": len(frame),
                **values,
            }
        )
    return pd.DataFrame(rows)
