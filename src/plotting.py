"""Publication-oriented figures generated only from saved result artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODULE_ORDER = ["NaV", "Kv", "CaV", "HCN", "GABAA", "iGluR"]


def _save(fig: plt.Figure, output_stem: Path) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    colorbar_label: str,
    center: float | None = None,
) -> plt.Figure:
    width = max(6.5, 0.9 * matrix.shape[1] + 2.5)
    height = max(4.0, 0.48 * matrix.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    values = matrix.to_numpy(dtype=float)
    if center is None:
        image = ax.imshow(values, cmap="viridis", aspect="auto")
    else:
        bound = float(np.nanmax(np.abs(values - center)))
        if not np.isfinite(bound) or bound == 0.0:
            bound = 1.0
        image = ax.imshow(
            values,
            cmap="RdBu_r",
            aspect="auto",
            vmin=center - bound,
            vmax=center + bound,
        )
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.set_title(title, loc="left", fontweight="bold")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label(colorbar_label)
    return fig


def figure_study_design(
    table: pd.DataFrame,
    output_stem: Path,
    *,
    n_features: int | None = None,
) -> None:
    if n_features is None:
        metadata = {
            "canonical_cell_id",
            "group_id",
            "donor_id",
            "subclass",
            "mat_cluster",
            "metadata_cluster",
            "biological_sex",
            "age",
            "structure",
        }
        n_features = sum(column not in metadata and not column.startswith("target_") for column in table)
    fig, ax = plt.subplots(figsize=(12, 3.2), constrained_layout=True)
    ax.axis("off")
    labels = [
        f"Verified paired cells\nN = {len(table):,}",
        f"Intrinsic IPFX\n{n_features} frozen features",
        "Six frozen\nexpression modules",
        "3-fold grouped OOF\nElastic Net · XGBoost · MLP",
        "Baselines · CI · permutation\nheld-out SHAP",
    ]
    xs = np.linspace(0.1, 0.9, len(labels))
    for index, (x, label) in enumerate(zip(xs, labels)):
        ax.text(
            x,
            0.5,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.55", facecolor="#E8F1FA", edgecolor="#235789"),
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(xs[index + 1] - 0.075, 0.5),
                xytext=(x + 0.075, 0.5),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", color="#444444", lw=1.5),
            )
    ax.set_title("Study design and analysis flow", loc="left", fontweight="bold")
    _save(fig, output_stem)


def figure_cohort(table: pd.DataFrame, output_stem: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), constrained_layout=True)
    for axis, module in zip(axes.flat[:6], MODULE_ORDER):
        axis.hist(table[f"target_{module}"], bins=30, color="#3A7CA5", edgecolor="white")
        axis.set_title(module)
        axis.set_xlabel("Module score")
        axis.set_ylabel("Cells")
    subclass_counts = table.subclass.value_counts().sort_values()
    axes.flat[6].barh(subclass_counts.index.astype(str), subclass_counts.values, color="#D17B0F")
    axes.flat[6].set_title("Subclass composition")
    axes.flat[6].set_xlabel("Cells")
    group_sizes = table.groupby("group_id").size()
    axes.flat[7].hist(
        group_sizes,
        bins=min(20, max(3, len(group_sizes))),
        color="#4C956C",
        edgecolor="white",
    )
    axes.flat[7].set_title(f"Group sizes (groups={len(group_sizes):,})")
    axes.flat[7].set_xlabel("Cells per group")
    axes.flat[7].set_ylabel("Groups")
    fig.suptitle("Module distributions and cohort characteristics", fontweight="bold")
    _save(fig, output_stem)


def figure_performance(performance: pd.DataFrame, output_stem: Path) -> None:
    primary = performance.loc[performance.analysis.eq("ephys_only")]
    matrix = primary.pivot(index="module", columns="model", values="r2").reindex(MODULE_ORDER)
    matrix = matrix.reindex(
        columns=[column for column in ("ElasticNet", "XGBoost", "MLP") if column in matrix]
    )
    _save(
        _heatmap(
            matrix,
            title="Held-out performance",
            colorbar_label="OOF R²",
            center=0.0,
        ),
        output_stem,
    )


def figure_observed_predicted(
    oof: pd.DataFrame,
    performance: pd.DataFrame,
    output_stem: Path,
) -> None:
    primary = performance.loc[performance.analysis.eq("ephys_only")].copy()
    best_by_module = (
        primary.sort_values(["module", "r2"], ascending=[True, False])
        .groupby("module")
        .head(1)
    )
    chosen = best_by_module.sort_values("r2", ascending=False).head(3)
    fig, axes = plt.subplots(
        1,
        len(chosen),
        figsize=(4.3 * len(chosen), 4.0),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for axis, row in zip(axes, chosen.itertuples()):
        frame = oof.loc[
            oof.module.eq(row.module)
            & oof.model.eq(row.model)
            & oof.analysis.eq("ephys_only")
        ]
        axis.scatter(frame.y_true, frame.y_pred, s=12, alpha=0.45, edgecolors="none")
        low = float(min(frame.y_true.min(), frame.y_pred.min()))
        high = float(max(frame.y_true.max(), frame.y_pred.max()))
        axis.plot([low, high], [low, high], ls="--", color="black", lw=1)
        axis.set_title(f"{row.module} · {row.model}\nR²={row.r2:.2f}, ρ={row.spearman:.2f}")
        axis.set_xlabel("Observed module score")
        axis.set_ylabel("OOF prediction")
    fig.suptitle("Observed versus out-of-fold predicted scores", fontweight="bold")
    _save(fig, output_stem)


def figure_shap(
    stability: pd.DataFrame,
    output_stem: Path,
    feature_labels: dict[str, str] | None = None,
) -> None:
    matrix = stability.pivot(index="feature", columns="module", values="mean_abs_shap")
    matrix = matrix.reindex(columns=MODULE_ORDER)
    order = matrix.max(axis=1).sort_values(ascending=False).index[:24]
    matrix = matrix.loc[order]
    if feature_labels:
        matrix = matrix.rename(index=feature_labels)
    _save(
        _heatmap(
            matrix,
            title="Held-out XGBoost feature contributions",
            colorbar_label="Mean |SHAP| (module-score units)",
        ),
        output_stem,
    )


def figure_baselines(performance: pd.DataFrame, output_stem: Path) -> None:
    selections = {
        "Dummy": ("Dummy", "dummy"),
        "Subclass": ("SubclassRidge", "subclass_only"),
        "Ephys (Elastic Net)": ("ElasticNet", "ephys_only"),
        "Ephys + subclass": ("EphysSubclassRidge", "ephys_plus_subclass"),
    }
    parts = []
    for label, (model, analysis) in selections.items():
        frame = performance.loc[
            performance.model.eq(model) & performance.analysis.eq(analysis),
            ["module", "r2"],
        ].copy()
        frame["comparison"] = label
        parts.append(frame)
    matrix = (
        pd.concat(parts)
        .pivot(index="module", columns="comparison", values="r2")
        .reindex(MODULE_ORDER)
    )
    matrix = matrix.reindex(columns=list(selections))
    _save(
        _heatmap(
            matrix,
            title="Baseline and confounder comparison",
            colorbar_label="OOF R²",
            center=0.0,
        ),
        output_stem,
    )


def generate_all_figures(root: str | Path) -> list[str]:
    root = Path(root)
    table = pd.read_parquet(root / "data/processed/modeling_table.parquet")
    oof = pd.read_parquet(root / "results/predictions/oof_predictions.parquet")
    performance = pd.read_csv(root / "results/tables/model_performance.csv")
    stability = pd.read_csv(root / "results/tables/shap_stability.csv")
    dictionary = pd.read_csv(root / "results/tables/ephys_feature_dictionary.csv")
    selected = dictionary.selected_primary.astype(str).str.lower().eq("true")
    feature_labels = dict(
        dictionary.loc[selected, ["feature_key", "display_name"]]
        .astype(str)
        .itertuples(index=False, name=None)
    )
    stems = [
        root / "figures" / f"figure_{number}_{name}"
        for number, name in [
            (1, "study_design"),
            (2, "cohort"),
            (3, "performance"),
            (4, "observed_predicted"),
            (5, "shap_heatmap"),
            (6, "baseline_comparison"),
        ]
    ]
    figure_study_design(table, stems[0], n_features=len(feature_labels))
    figure_cohort(table, stems[1])
    figure_performance(performance, stems[2])
    figure_observed_predicted(oof, performance, stems[3])
    figure_shap(stability, stems[4], feature_labels)
    figure_baselines(performance, stems[5])
    return [str(stem.relative_to(root)) for stem in stems]
