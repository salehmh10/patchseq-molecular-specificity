"""Frozen molecular-module loading, scoring, and dominance quality control."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def load_gene_modules(path: str | Path) -> dict[str, list[str]]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    raw_modules = payload.get("modules", payload)
    modules: dict[str, list[str]] = {}
    for name, definition in raw_modules.items():
        genes = definition.get("genes", definition) if isinstance(definition, dict) else definition
        if not isinstance(genes, list) or not genes:
            raise ValueError(f"Module {name!r} has no frozen gene list")
        normalized = [str(gene).strip() for gene in genes]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"Module {name!r} contains duplicate symbols")
        modules[str(name)] = normalized
    expected = {"NaV", "Kv", "CaV", "HCN", "GABAA", "iGluR"}
    if set(modules) != expected:
        raise ValueError(f"Expected modules {sorted(expected)}, found {sorted(modules)}")
    return modules


def score_modules(
    cpm: pd.DataFrame,
    modules: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate mean log2(CPM+1) across genes present in each frozen module."""

    if cpm.index.duplicated().any():
        raise ValueError("CPM table contains duplicate cell IDs")
    if cpm.columns.duplicated().any():
        raise ValueError("CPM table contains duplicate gene symbols")
    if (cpm.select_dtypes(include=[np.number]) < 0).any().any():
        raise ValueError("CPM values must be non-negative")
    transformed = np.log2(cpm.astype(float) + 1.0)
    scores: dict[str, pd.Series] = {}
    inventory: list[dict[str, object]] = []
    available = set(cpm.columns.astype(str))
    for module, requested in modules.items():
        present = [gene for gene in requested if gene in available]
        missing = [gene for gene in requested if gene not in available]
        coverage = len(present) / len(requested)
        if len(present) < 2 or coverage < 0.25:
            raise ValueError(
                f"Frozen-module coverage failed for {module}: {len(present)}/{len(requested)}"
            )
        scores[module] = transformed[present].mean(axis=1)
        inventory.append(
            {
                "module": module,
                "genes_requested": ";".join(requested),
                "genes_present": ";".join(present),
                "genes_missing": ";".join(missing),
                "n_requested": len(requested),
                "n_present": len(present),
                "coverage_fraction": coverage,
                "coverage_warning": coverage < 0.5,
                "score_mean": float(scores[module].mean()),
                "score_sd": float(scores[module].std(ddof=1)),
            }
        )
    score_frame = pd.DataFrame(scores, index=cpm.index)
    score_frame.index.name = "canonical_cell_id"
    return score_frame, pd.DataFrame(inventory)


def module_dominance(
    cpm: pd.DataFrame,
    scores: pd.DataFrame,
    modules: dict[str, list[str]],
) -> pd.DataFrame:
    """Report part-whole and leave-one-out dominance diagnostics without filtering."""

    transformed = np.log2(cpm.astype(float) + 1.0)
    rows: list[dict[str, object]] = []
    for module, requested in modules.items():
        present = [gene for gene in requested if gene in transformed]
        variances = transformed[present].var(ddof=1)
        module_score = scores[module].astype(float)
        module_variance = float(module_score.var(ddof=1))
        median_gene_variance = float(variances.median())
        for gene in present:
            gene_values = transformed[gene]
            remaining = [other for other in present if other != gene]
            loo = transformed[remaining].mean(axis=1) if remaining else pd.Series(np.nan, index=transformed.index)
            full_spearman = gene_values.corr(module_score, method="spearman")
            full_pearson = gene_values.corr(module_score, method="pearson")
            loo_spearman = gene_values.corr(loo, method="spearman") if remaining else np.nan
            loo_pearson = gene_values.corr(loo, method="pearson") if remaining else np.nan
            score_loo_spearman = module_score.corr(loo, method="spearman") if remaining else np.nan
            covariance = float(np.cov(gene_values / len(present), module_score, ddof=1)[0, 1])
            covariance_share = covariance / module_variance if module_variance > 0 else np.nan
            dominance_flag = bool(
                covariance_share >= 0.5
                or (np.isfinite(score_loo_spearman) and score_loo_spearman < 0.9)
                or (
                    np.isfinite(loo_spearman)
                    and abs(loo_spearman) >= 0.9
                    and float(variances[gene]) >= 2 * median_gene_variance
                )
            )
            rows.append(
                {
                    "module": module,
                    "gene": gene,
                    "gene_variance": float(variances[gene]),
                    "nonzero_cell_fraction": float((cpm[gene] > 0).mean()),
                    "gene_module_pearson": float(full_pearson),
                    "gene_module_spearman": float(full_spearman),
                    "gene_loo_pearson": float(loo_pearson) if np.isfinite(loo_pearson) else np.nan,
                    "gene_loo_spearman": float(loo_spearman) if np.isfinite(loo_spearman) else np.nan,
                    "module_loo_spearman": float(score_loo_spearman) if np.isfinite(score_loo_spearman) else np.nan,
                    "covariance_contribution_share": float(covariance_share),
                    "dominance_flag": dominance_flag,
                }
            )
    return pd.DataFrame(rows)
