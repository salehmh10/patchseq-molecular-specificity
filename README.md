# Testing Molecular Specificity in Electrophysiology-to-Transcript Prediction

Code and supplementary evidence for **Testing Molecular Specificity in
Electrophysiology-to-Transcript Prediction with Matched Negative Controls**.
The framework tests whether prediction exceeds expression- and
detection-matched gene-set controls in Allen mouse Patch-seq neurons.

## Overview

The study evaluates six ion-channel and receptor transcript modules in 3,410
inhibitory neurons from 871 donor groups. Elastic Net, XGBoost, and an MLP
share donor-disjoint outer folds. The final model comparison uses 23
nonredundant electrophysiological predictors; a 24-feature reference analysis
is retained for provenance and sensitivity analyses.

## Scientific Question

Does electrophysiology predict a named molecular target unusually well
relative to equally sized gene sets matched for expression and detection?
Positive predictive accuracy alone does not establish molecular specificity.

## Main Findings

- GABAA and iGluR retained specificity after 1,000 matched controls per target
  and Benjamini–Hochberg correction across six primary tests.
- Kv had positive held-out R² but did not exceed its matched-control benchmark.
- NaV conclusions depended on the exact gene definition; NaV6 and NaV7 are
  reported as distinct targets.
- Total library count was not predicted better than the pooled mean baseline.
- Results are predictive and associational. They do not establish causal
  mechanisms, protein abundance, or ion-channel conductance.

See [selected results](results/README.md) for the corresponding tables and
[methods](docs/METHODS_OVERVIEW.md) for interpretive limits.

## Repository Structure

```text
src/          Analysis implementations
scripts/      Acquisition, analysis, validation, and reproduction commands
config/       Frozen analytical definitions and model specifications
tests/        Synthetic contracts and local integration tests
data/         Source manifests and acquisition instructions
results/      Selected reference tables and checksums
figures/      Final manuscript panels and supplementary figure index
supplement/   Final PDF, LaTeX source, figures, and machine-readable tables
docs/         Methods, validation, provenance, and reproduction guides
```

## Installation

Use Python 3.12 and an isolated environment. Run commands from the repository
root. Install the recorded analytical versions:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in PowerShell, then run:

```bash
python -m pip install -r requirements.txt
```

## Data Access

Raw data are excluded. [Data instructions](data/README.md) list the five
source files, repository-relative destinations, acquisition URLs, and checksums.
The complete count archive is approximately 445 MB. Acquisition is explicit:

```bash
python scripts/00_download.py --include-transcriptomics
python scripts/verify_checksums.py --scope data
```

Upstream and recorded archival URLs are tried with content validation.
Availability of third-party endpoints may change.

## Quick Start

The following commands run without downloading raw data:

```bash
python scripts/validate_repository.py
python scripts/smoke_test.py
python -m pytest -q
python scripts/verify_checksums.py --scope all
```

The smoke test exercises real preprocessing, models, module scoring, metrics,
and inference with synthetic inputs. The test suite additionally checks matching.
Neither command reproduces the scientific estimates.

## Reproducing the Analyses

After acquiring and validating the data:

```bash
python scripts/reproduce_paper.py --stage primary
python scripts/reproduce_paper.py --stage robustness
python scripts/run_specificity_controls.py --mode full
python scripts/run_model_comparison.py --mode full
python scripts/run_model_comparison.py --mode resume
python scripts/run_extended_controls.py --mode full
```

Alternatively, `python scripts/reproduce_paper.py --stage all` runs those
stages in order. Full runs are computationally intensive. Stage receipts bind
new-run inputs to their checksums; the original reference hashes remain in
`results/manifests/expected_hashes.json`. The release audit covers installation,
synthetic execution, and packaged evidence, not a fresh full model fit.
See [reproducibility](docs/REPRODUCIBILITY.md) for prerequisites, resumption,
reference-artifact tests, and supplementary compilation.

## Expected Outputs

Full runs produce cohort crosswalks, fold assignments, module scores, held-out
predictions, model comparisons, matched-control summaries, figures, and
artifact manifests. Generated data, models, and large prediction tables are
excluded from Git. Selected published reference tables are stored separately
in `results/selected` and remain unchanged by reproduction commands.

## Validation and Tests

CI uses Python 3.12 and runs syntax/import checks, metadata and checksum checks,
synthetic scientific tests, and a data-free smoke test. It does not acquire
Allen data. Strict integration tests require the corresponding complete local
reference artifacts and are explicitly enabled with:

```bash
python -m pytest -q --run-integration -m integration
```

These tests include frozen byte-level expectations; they are distinct from
evaluating numerical agreement in a new environment. See
[validation](docs/VALIDATION.md).

## Supplementary Material

[Supplementary Material](supplement/Supplementary_Material.pdf) contains the
final 12-page supplement. Its LaTeX source, Figures S1–S5, and small
machine-readable tables are included. The full Table S44 is supplied in the
complete supplementary ZIP attached to the candidate release, alongside the
other Tables S1–S50. [Supplement instructions](supplement/README_SUPPLEMENT.md)
describe compilation and checksum verification.

## Citation

Saleh Mohammadhasani; Reza Kazemeynimoghaddam; Amirreza Khadempir; Pedram Hamidirad; Amirreza Dehghan Nayeri. (2026). *Patch-seq Molecular Specificity Analysis*
(version 1.0.0-rc1). https://github.com/salehmh10/patchseq-molecular-specificity

Machine-readable citations are provided in [CITATION.cff](CITATION.cff) and
[CITATION.bib](CITATION.bib). No DOI has been assigned in this repository.

## License

Original repository software is licensed under the [MIT License](LICENSE).
Allen Institute, NeMO, and other third-party data retain their original terms.
Raw data and copyrighted source-paper PDFs are not redistributed. See
[NOTICE](NOTICE) and [DATA_LICENSES.md](DATA_LICENSES.md) for scope; no separate
Creative Commons license is granted for figures or documents.

## Contact

For technical questions, please open a GitHub issue.
