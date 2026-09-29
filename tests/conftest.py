"""Keep strict local artifact checks opt-in for data-free installations."""
from pathlib import Path
import pytest

def pytest_addoption(parser):
    parser.addoption('--run-integration', action='store_true', default=False,
                     help='Run strict tests requiring complete local reference artifacts')

def pytest_configure(config):
    config.addinivalue_line('markers', 'integration: requires complete local reference data and outputs')

def pytest_collection_modifyitems(config, items):
    for item in items:
        filename = Path(str(item.fspath)).name
        local = (filename in {'test_pipeline.py', 'test_robustness_v2.py'}
                 or item.name.startswith('test_full_')
                 or item.name == 'test_synthetic_frozen_fold_geometry_is_donor_disjoint')
        if local:
            item.add_marker(pytest.mark.integration)
            if not config.getoption('--run-integration'):
                item.add_marker(pytest.mark.skip(reason='Requires local reference artifacts; enable --run-integration'))
