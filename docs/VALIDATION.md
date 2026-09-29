# Validation

Three distinct levels are provided:

1. **Repository validation:** required files, JSON/YAML parsing, citation
   consistency, reference-result schemas, checksum verification, and imports.
2. **Synthetic validation:** deterministic seeds, expression/detection matching,
   balanced partitions, group leakage rejection, preprocessing, regression
   metrics, empirical p-values, six-test BH, and corrupted-cache rejection.
3. **Reference integration validation:** complete local source and output
   artifacts, exact cohort/folds, OOF coverage, target reconstruction,
   reference hashes, model/SHAP geometry, and inference reconstruction.

`python -m pytest -q` runs the first two levels and explicitly skips tests
marked `integration`. `--run-integration` enables the strict artifact tests;
missing inputs then fail. Scientific assertions have not been replaced with
existence-only tests. Administrative document-status assertions are excluded
from the distributable scientific test suite.

The offline smoke command uses real implemented model and analysis utilities
with synthetic data. No network or full-data download occurs in CI.

The publication-package checks do not constitute a fresh full analysis run.
Cross-platform serialization, numerical libraries, and timestamps can affect
bitwise reproduction. Original hashes and current-run provenance are separate
records. The frozen supplementary software inventory describes the original
analysis, not whichever Python environment happens to build the PDF.
