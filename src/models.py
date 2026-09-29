"""Small, pre-specified model families with group-aware inner selection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from .evaluation import regression_metrics
from .preprocessing import make_preprocessor


@dataclass
class SelectedModel:
    pipeline: Pipeline
    parameters: dict[str, object]
    inner_rmse: float


def _pipeline(
    estimator: object,
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    *,
    scale_numeric: bool,
) -> Pipeline:
    return Pipeline(
        [
            (
                "preprocessor",
                make_preprocessor(
                    numeric_features,
                    categorical_features,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("estimator", estimator),
        ]
    )


def elastic_net_candidates(
    numeric_features: Sequence[str], categorical_features: Sequence[str] = (), *, seed: int = 42
) -> Iterable[tuple[dict[str, object], Pipeline]]:
    for alpha in (0.001, 0.01, 0.1, 1.0):
        for l1_ratio in (0.1, 0.5, 0.9):
            params = {"alpha": alpha, "l1_ratio": l1_ratio}
            estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=seed)
            yield params, _pipeline(
                estimator, numeric_features, categorical_features, scale_numeric=True
            )


def xgboost_candidates(
    numeric_features: Sequence[str], *, seed: int = 42, level: int = 0
) -> Iterable[tuple[dict[str, object], Pipeline]]:
    grid = [
        dict(n_estimators=250, max_depth=2, learning_rate=0.05, reg_lambda=1.0),
        dict(n_estimators=400, max_depth=2, learning_rate=0.03, reg_lambda=3.0),
        dict(n_estimators=250, max_depth=3, learning_rate=0.05, reg_lambda=3.0),
        dict(n_estimators=300, max_depth=4, learning_rate=0.03, reg_lambda=5.0),
    ]
    if level == 1:
        grid = grid[:3]
    elif level >= 2:
        grid = grid[:2]
    for params in grid:
        estimator = XGBRegressor(
            **params,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=seed,
            tree_method="hist",
        )
        yield params, _pipeline(estimator, numeric_features, scale_numeric=False)


def mlp_candidate(
    numeric_features: Sequence[str], *, seed: int = 42, max_iter: int = 300
) -> tuple[dict[str, object], Pipeline]:
    params = {
        "hidden_layer_sizes": (64, 32),
        "alpha": 0.001,
        "seed": seed,
        "max_iter": max_iter,
    }
    estimator = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=0.001,
        # Early stopping is selected explicitly with a donor-disjoint validation
        # split in fit_mlp_on_inner_groups.  sklearn's built-in early_stopping
        # performs a row-wise split and is therefore not suitable here.
        early_stopping=False,
        validation_fraction=0.15,
        max_iter=max_iter,
        random_state=seed,
    )
    return params, _pipeline(estimator, numeric_features, scale_numeric=True)


def fit_mlp_on_inner_groups(
    numeric_features: Sequence[str],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Sequence[object],
    *,
    model_seed: int,
    split_seed: int = 42,
    max_iter: int = 300,
    patience: int = 20,
    min_delta: float = 1e-5,
) -> SelectedModel:
    """Choose the MLP epoch count on one donor-disjoint inner split.

    ``MLPRegressor(early_stopping=True)`` uses a random row split, which can put
    cells from the same donor into both inner partitions.  We instead monitor a
    frozen grouped validation partition with one-epoch ``partial_fit`` calls,
    then refit a fresh pipeline on the complete outer-training partition for the
    selected number of epochs.
    """

    train_idx, validation_idx = group_inner_split(groups, seed=split_seed)
    y_array = np.asarray(y, dtype=float)
    preprocessor = make_preprocessor(numeric_features, scale_numeric=True)
    X_inner = np.asarray(preprocessor.fit_transform(X.iloc[train_idx]), dtype=float)
    X_validation = np.asarray(preprocessor.transform(X.iloc[validation_idx]), dtype=float)
    estimator = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        alpha=0.001,
        early_stopping=False,
        max_iter=1,
        random_state=model_seed,
    )
    best_iteration = 0
    best_rmse = float("inf")
    stale_epochs = 0
    for iteration in range(1, max_iter + 1):
        estimator.partial_fit(X_inner, y_array[train_idx])
        prediction = estimator.predict(X_validation)
        score = regression_metrics(y_array[validation_idx], prediction)["rmse"]
        if score < best_rmse - min_delta:
            best_rmse = score
            best_iteration = iteration
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    if best_iteration < 1:
        raise RuntimeError("MLP inner early stopping did not produce a finite validation score")

    parameters, pipeline = mlp_candidate(
        numeric_features,
        seed=model_seed,
        max_iter=best_iteration,
    )
    pipeline.fit(X, y_array)
    parameters.update(
        {
            "selected_iter": best_iteration,
            "inner_split_seed": split_seed,
            "early_stopping": "group_aware_manual",
        }
    )
    return SelectedModel(pipeline, parameters, best_rmse)


def group_inner_split(groups: Sequence[object], *, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    groups_array = np.asarray(groups).astype(str)
    if np.unique(groups_array).size < 2:
        raise ValueError("Group-aware inner selection requires at least two training groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, validation_idx = next(splitter.split(np.zeros(len(groups_array)), groups=groups_array))
    if set(groups_array[train_idx]).intersection(groups_array[validation_idx]):
        raise AssertionError("Inner group leakage")
    return train_idx, validation_idx


def select_on_inner_groups(
    candidates: Iterable[tuple[dict[str, object], Pipeline]],
    X: pd.DataFrame,
    y: np.ndarray,
    groups: Sequence[object],
    *,
    seed: int = 42,
) -> SelectedModel:
    train_idx, validation_idx = group_inner_split(groups, seed=seed)
    best: SelectedModel | None = None
    for parameters, candidate in candidates:
        fitted = clone(candidate).fit(X.iloc[train_idx], np.asarray(y)[train_idx])
        prediction = fitted.predict(X.iloc[validation_idx])
        score = regression_metrics(np.asarray(y)[validation_idx], prediction)["rmse"]
        if best is None or score < best.inner_rmse:
            best = SelectedModel(candidate, parameters, score)
    if best is None:
        raise RuntimeError("No model candidate was evaluated")
    best.pipeline.fit(X, y)
    return best


def dummy_model(numeric_features: Sequence[str]) -> Pipeline:
    return _pipeline(DummyRegressor(strategy="mean"), numeric_features, scale_numeric=False)


def subclass_ridge_model(subclass_column: str = "subclass") -> Pipeline:
    return _pipeline(Ridge(alpha=1.0), (), (subclass_column,), scale_numeric=False)


def ephys_subclass_ridge_model(
    numeric_features: Sequence[str], subclass_column: str = "subclass"
) -> Pipeline:
    return _pipeline(Ridge(alpha=1.0), numeric_features, (subclass_column,), scale_numeric=True)
