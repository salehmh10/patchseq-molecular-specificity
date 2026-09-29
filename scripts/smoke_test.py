"""Run a deterministic synthetic end-to-end check without network access."""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models import elastic_net_candidates, xgboost_candidates, group_inner_split
from src.evaluation import regression_metrics
from src.revision_section4 import bh_adjust
from src.biology import score_modules, load_gene_modules

def smoke() -> dict:
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(120, 3)), columns=['e1', 'e2', 'e3'])
    x.loc[::17, 'e2'] = np.nan
    y = 2 * x.e1.to_numpy() + rng.normal(0, 0.1, len(x))
    groups = np.repeat(np.arange(40), 3)
    train, test = group_inner_split(groups, seed=42)
    assert set(groups[train]).isdisjoint(groups[test])
    results = {}
    for name, candidates in [('ElasticNet', elastic_net_candidates(list(x))),
                             ('XGBoost', xgboost_candidates(list(x)))]:
        _, model = next(iter(candidates))
        model.fit(x.iloc[train], y[train])
        prediction = model.predict(x.iloc[test])
        metrics = regression_metrics(y[test], prediction)
        assert np.isfinite(list(metrics.values())).all()
        assert metrics['r2'] > 0.5
        results[name] = metrics['r2']
    modules = load_gene_modules(ROOT / 'config/gene_modules.yaml')
    genes = set().union(*map(set, modules.values())) - {'Scn2a'}
    counts = pd.DataFrame({gene: [0., 1., 3.] for gene in sorted(genes)}, index=['a','b','c'])
    scores, _ = score_modules(counts, modules)
    np.testing.assert_allclose(scores.to_numpy(), np.repeat([[0.], [1.], [2.]], 6, axis=1))
    p = np.array([.003, .014, .05, .196, .259, .899])
    assert (bh_adjust(p) <= .05).sum() == 2
    return {'synthetic_cells': len(x), 'donor_disjoint': True, 'model_r2': results, 'raw_data_downloaded': False}

if __name__ == '__main__':
    print(json.dumps(smoke(), indent=2))
