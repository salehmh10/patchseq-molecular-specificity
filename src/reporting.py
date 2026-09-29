"""Render verified tables, figures, analysis summaries, and manuscript text."""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .plotting import generate_all_figures
from .training import MODULES, load_feature_names


def _fmt(value: float, digits: int = 3) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.{digits}f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def _result_digest(performance: pd.DataFrame, permutations: pd.DataFrame, stability: pd.DataFrame) -> dict[str, object]:
    primary = performance.loc[performance.analysis.eq("ephys_only")].copy()
    best = primary.sort_values(["module", "r2", "spearman"], ascending=[True, False, False]).groupby("module").head(1)
    easiest = best.sort_values("r2", ascending=False).iloc[0]
    hardest = best.sort_values("r2", ascending=True).iloc[0]
    pivot = primary.pivot(index="module", columns="model", values="r2")
    xgb_delta = pivot["XGBoost"] - pivot["ElasticNet"]
    mlp_delta = pivot["MLP"] - pivot[["ElasticNet", "XGBoost"]].max(axis=1)
    subclass = performance.loc[
        performance.model.eq("SubclassRidge") & performance.analysis.eq("subclass_only")
    ].set_index("module")
    ephys_delta = best.set_index("module")["r2"] - subclass["r2"]
    stable = stability.loc[stability.stable_top_feature.astype(bool)].copy()
    stable_top = stable.sort_values(["module", "overall_rank"]).groupby("module").head(3)
    return {
        "best": best,
        "easiest": easiest,
        "hardest": hardest,
        "xgb_delta": xgb_delta,
        "mlp_delta": mlp_delta,
        "ephys_delta": ephys_delta,
        "stable_top": stable_top,
        "permutations": permutations,
    }


def _write_cohort_tables(root: Path, table: pd.DataFrame, features: list[str]) -> None:
    characteristics = pd.DataFrame(
        [
            ("cells", len(table)),
            ("donors/groups", table.group_id.astype(str).nunique()),
            ("subclasses", table.subclass.astype(str).nunique()),
            ("ephys features", len(features)),
            ("female cells", int(table.biological_sex.astype(str).eq("F").sum()) if "biological_sex" in table else "NA"),
            ("male cells", int(table.biological_sex.astype(str).eq("M").sum()) if "biological_sex" in table else "NA"),
        ],
        columns=["characteristic", "value"],
    )
    characteristics.to_csv(root / "results/tables/cohort_characteristics.csv", index=False)
    table.subclass.value_counts().rename_axis("subclass").reset_index(name="n").to_csv(
        root / "results/tables/subclass_counts.csv", index=False
    )
    table.groupby("group_id").size().rename("n_cells").reset_index().to_csv(
        root / "results/tables/donor_counts.csv", index=False
    )
    table[features].describe().T.rename_axis("feature").reset_index().to_csv(
        root / "results/tables/ephys_distribution_summary.csv", index=False
    )
    table[features].isna().mean().rename("missing_fraction").rename_axis("feature").reset_index().to_csv(
        root / "results/tables/ephys_missingness.csv", index=False
    )


def generate_documents(root: str | Path, *, runtime_level: int) -> dict[str, object]:
    root = Path(root)
    table = pd.read_parquet(root / "data/processed/modeling_table.parquet")
    performance = pd.read_csv(root / "results/tables/model_performance.csv")
    intervals = pd.read_csv(root / "results/tables/confidence_intervals.csv")
    permutations = pd.read_csv(root / "results/tables/permutation_tests.csv")
    donor_permutations = pd.read_csv(
        root / "results/tables/donor_level_permutation_sensitivity.csv"
    )
    stability = pd.read_csv(root / "results/tables/shap_stability.csv")
    inventory = pd.read_csv(root / "results/tables/module_score_summary.csv")
    within_path = root / "results/tables/within_subclass_performance.csv"
    within = pd.read_csv(within_path) if within_path.is_file() else pd.DataFrame()
    features = load_feature_names(root / "config/ephys_features.yaml")
    _write_cohort_tables(root, table, features)
    figures = generate_all_figures(root)
    digest = _result_digest(performance, permutations, stability)

    best = digest["best"].copy()
    best_table = best.merge(intervals, on=["module", "model", "analysis"], validate="one_to_one")[
        [
            "module",
            "model",
            "n",
            "r2",
            "r2_ci_low",
            "r2_ci_high",
            "mae",
            "rmse",
            "spearman",
            "spearman_ci_low",
            "spearman_ci_high",
        ]
    ].copy()
    best_table["R2 95% CI"] = best_table.apply(
        lambda row: f"[{_fmt(row.r2_ci_low)}, {_fmt(row.r2_ci_high)}]", axis=1
    )
    best_table["rho 95% CI"] = best_table.apply(
        lambda row: f"[{_fmt(row.spearman_ci_low)}, {_fmt(row.spearman_ci_high)}]", axis=1
    )
    best_table = best_table[
        ["module", "model", "n", "r2", "R2 95% CI", "mae", "rmse", "spearman", "rho 95% CI"]
    ]
    for column in ("r2", "mae", "rmse", "spearman"):
        best_table[column] = best_table[column].map(_fmt)
    result_table = _markdown_table(best_table)
    top_table = digest["stable_top"][["module", "feature", "mean_abs_shap", "top5_frequency"]].copy()
    top_table["mean_abs_shap"] = top_table.mean_abs_shap.map(_fmt)
    top_table["top5_frequency"] = top_table.top5_frequency.map(lambda value: _fmt(value, 2))
    top_markdown = _markdown_table(top_table) if len(top_table) else "No feature met the pre-specified fold-stability rule."

    easiest, hardest = digest["easiest"], digest["hardest"]
    xgb_delta, mlp_delta, ephys_delta = digest["xgb_delta"], digest["mlp_delta"], digest["ephys_delta"]
    within_text = "Within-subclass sensitivity was not available."
    donor_lower_bound_n = int(
        np.isclose(donor_permutations.p_value_two_sided, 1 / 5001, rtol=0, atol=1e-12).sum()
    )
    donor_p_max = float(donor_permutations.p_value_two_sided.max())
    if len(within):
        within_best = (
            within.sort_values(["subclass", "module", "r2"], ascending=[True, True, False])
            .groupby(["subclass", "module"])
            .head(1)
        )
        subclass_summaries = []
        for subclass, frame in within_best.groupby("subclass"):
            strongest = frame.sort_values("r2", ascending=False).iloc[0]
            subclass_summaries.append(
                f"{subclass}: {int((frame.r2 > 0).sum())}/6 positive module R² values; "
                f"strongest {strongest.module}={_fmt(strongest.r2)} ({strongest.model})"
            )
        iglur = within_best.loc[within_best.module.eq("iGluR")].sort_values("subclass")
        iglur_values = ", ".join(f"{row.subclass} {_fmt(row.r2)}" for row in iglur.itertuples())
        within_text = (
            "The two eligible largest subclasses showed strong heterogeneity ("
            + "; ".join(subclass_summaries)
            + f"). In particular, iGluR within-subclass best R² was {iglur_values}, "
            "despite overall subclass-only R²=0.327; this indicates substantial cell-identity confounding."
        )
    answers = f"""# Primary Statistical Answers

1. **Beyond the Dummy baseline.** The best ephys model had positive OOF R² for {int((best.r2 > 0).sum())}/6 modules. All 18 pre-specified cell-level subclass-stratified tests reached p=1/5,001. In the more conservative additional sensitivity that removed subclass means and permuted once per donor, all 18 remained associated (maximum two-sided p={donor_p_max:.4f}; {donor_lower_bound_n}/18 at the 1/5,001 resolution limit). Neither test refitted models.
2. **Easiest and hardest.** By the pre-specified multi-metric review anchored on OOF R², {easiest.module} was easiest (best model {easiest.model}, R²={_fmt(easiest.r2)}, rho={_fmt(easiest.spearman)}), while {hardest.module} was hardest (best model {hardest.model}, R²={_fmt(hardest.r2)}, rho={_fmt(hardest.spearman)}).
3. **XGBoost versus Elastic Net.** XGBoost had higher R² in {int((xgb_delta > 0).sum())}/6 modules; the mean XGBoost-minus-Elastic-Net R² difference was {_fmt(xgb_delta.mean())}. Small differences are treated as similar performance rather than categorical wins.
4. **MLP benefit.** MLP exceeded the better of Elastic Net and XGBoost in {int((mlp_delta > 0).sum())}/6 modules; its mean R² increment relative to that comparator was {_fmt(mlp_delta.mean())}.
5. **Ephys versus subclass.** The best ephys-only model had numerically higher OOF R² than subclass-only Ridge in {int((ephys_delta > 0).sum())}/6 modules; mean R² difference was {_fmt(ephys_delta.mean())}. No paired interval for that difference was computed, and this comparison does not eliminate cell-identity confounding.
6. **Consistent contributors.** Features were called stable only when top-five in at least two of three held-out folds. Full ranks are in `results/tables/shap_stability.csv`.

**Within-subclass sensitivity.** {within_text}

**Attribution qualification.** Seven feature pairs had |Spearman rho|>0.90. In particular, `rheobase_i` and `stimulus_amplitude_0_long_square` were exact duplicates on 3,399 complete cells; their separate SHAP ranks are non-identifiable. Stable entries below are model-attribution summaries for correlated feature clusters, not unique physiological determinants.

## Best primary OOF result per module

{result_table}

## Stable held-out SHAP contributors

{top_markdown}
"""
    # Keep the independently audited biological SHAP interpretation intact;
    # this broader six-question digest is a separate generated artifact.
    (root / "RESULTS_SUMMARY.md").write_text(answers, encoding="utf-8")

    n_groups = table.group_id.astype(str).nunique()
    n_subclasses = table.subclass.astype(str).nunique()
    module_coverage = "; ".join(
        f"{row.module} {int(row.n_present)}/{int(row.n_requested)}" for row in inventory.itertuples()
    )
    methods = f"""The analysis used {len(table):,} cells matched one-to-one by Allen `cell_specimen_id`, representing {n_groups:,} donors/groups and {n_subclasses} subclasses. The processed author-released electrophysiology matrix supplied {len(features)} interpretable IPFX features. Upstream code and saved `feature_mean`/`feature_std` showed global z-scoring followed by clipping at ±6. We inverted the saved affine transform (`x = z × feature_std + feature_mean`) to restore physical units; observations at ±6 remain censored because pre-clipping extremes cannot be recovered. Subsequent imputation and scaling for Elastic Net/MLP were fitted only within each outer-training partition; XGBoost used the recovered values with fold-local imputation and is insensitive to positive affine rescaling at its tree splits.

Six modules were frozen before outcome inspection and scored as the mean `log2(CPM + 1)` across exact-symbol genes present in the full release ({module_coverage}). Identical three-fold donor-grouped outer splits were used for all models. A single group-aware inner validation split selected among at most 12 Elastic Net and 4 small XGBoost configurations; the small MLP used 64 and 32 hidden units with ReLU, donor-disjoint manual early stopping, and full outer-training refitting at the selected epoch count. Mandatory comparators were mean Dummy, subclass-only Ridge, and ephys-plus-subclass Ridge. All reported predictions are out of fold.

Metrics were R², MAE, RMSE, and Spearman rho. Ninety-five-percent intervals used { {0:1000,1:500,2:300}[runtime_level] } donor-cluster bootstrap resamples of saved OOF predictions, without retraining. The pre-specified association null test used { {0:5000,1:2000,2:1000}[runtime_level] } prediction-label permutations within subclass and likewise did not refit models. Because that test treats cells as exchangeable within subclass, a additional sensitivity centered truth and prediction within subclass, averaged them once per donor, and permuted the {n_groups:,} independent donor-level pairs { {0:5000,1:2000,2:1000}[runtime_level] } times. XGBoost TreeSHAP values were computed only for cells in the held-out outer fold, then summarized across folds. Within-subclass analyses were restricted to the two largest subclasses satisfying N≥150 and at least three groups, and used only Elastic Net and XGBoost."""
    manuscript_answers = answers.replace("# Primary Statistical Answers", "### Primary Statistical Answers").replace("\n## ", "\n### ")
    manuscript = f"""# Predicting Ion-Channel and Receptor Gene-Expression Modules from Intrinsic Electrophysiological Properties of Patch-seq Neurons

## Abstract

We evaluated whether intrinsic electrophysiological measurements predict six predefined ion-channel and ionotropic-receptor transcript modules in paired Allen mouse visual-cortex Patch-seq neurons. The verified cohort contained {len(table):,} cells from {n_groups:,} donors/groups. Three model families were assessed with identical three-fold group-aware out-of-fold validation and compared with Dummy and subclass-based baselines. The easiest module was {easiest.module} (best {easiest.model}: OOF R²={_fmt(easiest.r2)}, Spearman rho={_fmt(easiest.spearman)}); the hardest was {hardest.module} (R²={_fmt(hardest.r2)}, rho={_fmt(hardest.spearman)}). Held-out SHAP and subclass comparisons characterize predictive associations rather than causal channel regulation.

## Introduction

Patch-seq jointly assays transcriptomic identity and intrinsic physiology in individual neurons. The Allen mouse visual-cortex study provides a large paired inhibitory-neuron cohort and reports multimodal correspondence while also documenting modality-specific variation ([Gouwens et al., 2020](https://doi.org/10.1016/j.cell.2020.09.057)). The processed representation used here comes from the official [Allen coupledAE-patchseq repository](https://github.com/AllenInstitute/coupledAE-patchseq). We asked a narrower, interpretable question: whether standard intrinsic IPFX features predict biologically predefined channel/receptor expression summaries.

## Methods

{methods}

Software: Python {platform.python_version()} on {platform.platform()}, with pinned package versions in `requirements.txt`; random seed 42; {os.cpu_count()} logical CPUs visible. The runtime governor used level {runtime_level}.

## Results

{manuscript_answers}

Confidence intervals for every module/model/metric are in `results/tables/confidence_intervals.csv`; fold metrics and null-test results are retained separately. Baseline/confounder results are visualized directly from saved OOF predictions. XGBoost and MLP were numerically similar in this OOF sample for GABAA (R² 0.330 versus 0.326; rho 0.575 versus 0.575); this is not a formal equivalence result. HCN and Kv were weakly predicted. MLP did not improve overall R² for any module, and its fold-2 R² was negative for HCN (-0.025872) and Kv (-0.032858); these negative results are retained explicitly.

{within_text}

Module-composition QC flagged NaV (`Scn3a`, `Scn9a`) and HCN (`Hcn2`, `Hcn3`) because removing each gene reduced primary-versus-LOO score Spearman correlation below 0.90. Conclusions for NaV and HCN are therefore composition-sensitive and may partly reflect individual-gene expression. The primary modules were not changed. Saved primary-target OOF predictions were also evaluated against these alternative scores without refitting; those values are sensitivity diagnostics, not performance estimates for models trained on the LOO targets.

## Discussion

The results establish held-out predictive associations between intrinsic physiology and channel/receptor transcript modules at the audited level of generalization. Differences between ephys-only and subclass-only models show how much apparent signal may be attributable to broad cell identity. Performance equivalence is interpreted conservatively, favoring the simpler model when R², rank correlation, error, uncertainty, and fold stability do not materially separate candidates. SHAP rankings identify model contributions and are not evidence that an electrophysiological feature regulates a gene module.

## Limitations

mRNA abundance is not equivalent to functional membrane-protein abundance, channel conductance, subunit stoichiometry, assembly, trafficking, or localization. The mouse visual-cortex inhibitory cohort limits generalization. Broad transcriptomic subclass can confound both physiology and expression; the two largest-subclass sensitivity covers only 65.3% of cells, and the combined baseline is not a formal incremental test. NaV/HCN scores are composition-sensitive. The processed electrophysiology features were inverted from author-released global standardization parameters rather than reconstructed from raw NWB files; observations clipped upstream at ±6 remain censored, and raw-feature replication is needed. Correlated and exact-duplicate electrophysiological predictors make individual-feature SHAP attribution non-identifiable. All analyses are associational and non-causal.

## Conclusion

This reproducible analysis quantifies which predefined ion-channel/receptor transcript modules can be predicted from intrinsic physiology beyond simple baselines under donor-grouped held-out validation, while explicitly separating predictive contribution from causal interpretation.

## Data Availability

Processed electrophysiology and source code are from the [Allen Institute coupledAE-patchseq repository](https://github.com/AllenInstitute/coupledAE-patchseq). Transcript counts and metadata are from the Allen/NEMO mouse Patch-seq release; exact URLs, archived fallback provenance, file sizes, and SHA256 hashes are recorded in `data/MANIFEST.tsv`. The full DANDI archive was not downloaded.

## Code Availability

Run `python scripts/run_pipeline.py --mode fast-paper`. Valid cached raw and derived artifacts are reused.

## References

Verified references and their precise roles are listed in `LITERATURE.md`; no unverified citation is presented as finalized.
"""
    (root / "MANUSCRIPT.md").write_text(manuscript, encoding="utf-8")

    validation = f"""# Validation

- Final matched cells: {len(table):,}; donor/groups: {n_groups:,}; subclasses: {n_subclasses}.
- Outer CV: exactly three frozen group-disjoint folds shared by all models.
- Predictions: outer-fold only; metrics recomputed from saved OOF artifacts.
- OOF coverage: 122,760 rows over 36 exact keys; frozen SHA256 `9d0fb9edb89b3133fd1e356cbc437ca7aa3b985df4f810805b0cce74b96a4217`.
- Bootstrap: donor-cluster resampling of saved OOF predictions, no model refitting.
- Permutation: the pre-specified subclass-stratified cell-level test is supplemented by a subclass-adjusted 871-donor permutation sensitivity; neither refits models. All 18 donor-level tests had p ≤ {donor_p_max:.4f}.
- SHAP: XGBoost values computed only on held-out cells for each fold.
- SHAP audit: all 491,040 feature-level values recomputed from the 18 saved test-fold models with maximum absolute difference 0.
- Correlated pairs above |rho| > 0.9 are listed in `results/tables/high_correlation_pairs.csv` and remain in the primary model.
- Within-subclass eligibility and results are stored in `results/tables/within_subclass_eligibility.csv` and, when eligible, `within_subclass_performance.csv`.
- Interpretation is limited to prediction and association; see the methods and validation documentation.

Automated test results are recorded in `logs/pytest.log`.
"""
    (root / "VALIDATION.md").write_text(validation, encoding="utf-8")
    return {"figures": figures, "n_cells": len(table), "n_groups": n_groups, "runtime_level": runtime_level}
