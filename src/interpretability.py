"""Held-out-only TreeSHAP extraction and fold-stability summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap


def heldout_tree_shap(
    fitted_pipeline,
    X_test: pd.DataFrame,
    *,
    cell_ids: pd.Series | np.ndarray,
    module: str,
    fold: int,
) -> pd.DataFrame:
    """Compute SHAP values only on an outer fold never used to fit this pipeline."""

    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    estimator = fitted_pipeline.named_steps["estimator"]
    transformed = np.asarray(preprocessor.transform(X_test), dtype=float)
    feature_names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    if transformed.shape[1] != len(feature_names):
        raise AssertionError("SHAP feature-name dimension mismatch")
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(transformed, check_additivity=False)
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.shape != transformed.shape:
        raise AssertionError(f"Unexpected SHAP shape {values.shape}; expected {transformed.shape}")
    wide = pd.DataFrame(values, columns=feature_names)
    wide.insert(0, "fold", int(fold))
    wide.insert(0, "module", module)
    wide.insert(0, "canonical_cell_id", pd.Series(cell_ids).astype(str).to_numpy())
    return wide


def shap_stability(shap_values: pd.DataFrame) -> pd.DataFrame:
    id_columns = {"canonical_cell_id", "module", "fold"}
    features = [column for column in shap_values if column not in id_columns]
    fold_rows: list[dict[str, object]] = []
    for (module, fold), frame in shap_values.groupby(["module", "fold"], sort=True):
        magnitudes = frame[features].abs().mean().sort_values(ascending=False)
        for rank, (feature, value) in enumerate(magnitudes.items(), start=1):
            fold_rows.append(
                {
                    "module": module,
                    "fold": int(fold),
                    "feature": feature,
                    "mean_abs_shap": float(value),
                    "fold_rank": rank,
                    "is_top5": rank <= 5,
                }
            )
    by_fold = pd.DataFrame(fold_rows)
    summary = (
        by_fold.groupby(["module", "feature"], as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            sd_abs_shap=("mean_abs_shap", "std"),
            mean_fold_rank=("fold_rank", "mean"),
            max_fold_rank=("fold_rank", "max"),
            top5_frequency=("is_top5", "mean"),
            folds_observed=("fold", "nunique"),
        )
    )
    summary["stable_top_feature"] = (summary.top5_frequency >= 2 / 3) & (
        summary.folds_observed == shap_values["fold"].nunique()
    )
    summary["overall_rank"] = summary.groupby("module")["mean_abs_shap"].rank(
        ascending=False, method="min"
    )
    return summary.sort_values(["module", "overall_rank", "feature"]).reset_index(drop=True)


def high_correlation_pairs(
    table: pd.DataFrame,
    features: list[str],
    *,
    threshold: float = 0.90,
) -> pd.DataFrame:
    """Return highly correlated frozen predictors for SHAP interpretation audits."""

    missing = set(features).difference(table.columns)
    if missing:
        raise ValueError(f"Correlation input lacks frozen features: {sorted(missing)}")
    if not 0.0 <= threshold < 1.0:
        raise ValueError("Correlation threshold must be in [0, 1)")
    numeric = table[features].apply(pd.to_numeric, errors="coerce")
    spearman = numeric.corr(method="spearman")
    pearson = numeric.corr(method="pearson")
    rows: list[dict[str, object]] = []
    for index, feature_1 in enumerate(features):
        for feature_2 in features[index + 1 :]:
            rho = float(spearman.loc[feature_1, feature_2])
            if not np.isfinite(rho) or abs(rho) <= threshold:
                continue
            pair = numeric[[feature_1, feature_2]].dropna()
            rows.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho),
                    "pearson_r": float(pearson.loc[feature_1, feature_2]),
                    "n_pairwise_complete": int(len(pair)),
                    "exact_duplicate_on_complete_cases": bool(
                        np.array_equal(
                            pair[feature_1].to_numpy(dtype=float),
                            pair[feature_2].to_numpy(dtype=float),
                        )
                    ),
                }
            )
    columns = [
        "feature_1",
        "feature_2",
        "spearman_rho",
        "abs_spearman_rho",
        "pearson_r",
        "n_pairwise_complete",
        "exact_duplicate_on_complete_cases",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(
        ["abs_spearman_rho", "feature_1", "feature_2"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
