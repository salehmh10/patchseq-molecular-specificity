"""Leakage-safe fold-local preprocessing utilities."""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def numeric_pipeline(scale: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median", add_indicator=False))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    return Pipeline(steps)


def make_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    *,
    scale_numeric: bool,
) -> ColumnTransformer:
    transformers = [("numeric", numeric_pipeline(scale_numeric), list(numeric_features))]
    if categorical_features:
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical, list(categorical_features)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
