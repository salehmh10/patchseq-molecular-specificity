# Reproducibility

## Environment setup

Use Python 3.12. Create and activate an isolated environment and install
`python -m pip install -r requirements.txt`. Exact direct versions reproduce
the recorded analytical environment; this is not a complete transitive lock.
CI adds `requirements-ci.txt` for CFF validation. No package restructuring or
editable install is required: commands run from the repository root.

## Data placement

The five files in `data/MANIFEST.tsv` belong under `data/raw/`. Run:

```bash
python scripts/00_download.py --include-transcriptomics
python scripts/verify_checksums.py --scope data
```

The count archive is approximately 445 MB. The full experiment also generates
large prediction partitions, models, and caches. Reserve substantially more
disk space than the raw inputs. Full runtime depends on hardware; no fixed
runtime is promised for the 1,000-control extension.

## Smoke test

```bash
python scripts/smoke_test.py
python -m pytest -q
```

These commands use synthetic inputs and packaged metadata. They download no
data and do not fit the full research cohort. The scientific smoke modes of
the full analysis entry points additionally require their upstream artifacts.

## Primary analysis

```bash
python scripts/reproduce_paper.py --stage primary
```

This runs the audited identifier join, dataset construction, full level-0
24-feature training, interpretation, validation, and figure/table generation.
It uses all specified model seeds and inference draws. The command verifies
source checksums before processing and runs the primary integration tests.

```bash
python scripts/reproduce_paper.py --stage robustness
```

This prepares duplicate-removal, repeated grouped-validation, and gene-omission
sensitivity artifacts consumed downstream. Each completed stage writes a
checksum receipt in ignored local storage.

## Matched-control specificity analysis

```bash
python scripts/run_specificity_controls.py --mode full
```

Requires primary and robustness receipts. Uses the existing 200-control
implementation, technical targets, target definitions, count QC, and
fold-local adjustment. `--mode resume` reuses only verified compatible caches.

## Model-comparison analysis

```bash
python scripts/run_model_comparison.py --mode full
python scripts/run_model_comparison.py --mode resume
```

Requires completed specificity outputs. Fits the nonredundant 23-feature
models, paired comparisons, and specificity bridge. The resume run records
the no-refit state required by the control extension.

## 1,000-control extension

```bash
python scripts/run_extended_controls.py --mode full
```

Requires the model-comparison stage and its successful no-refit resume.
The extension retains the original 200 controls and expands each primary
target to 1,000 controls. NaV7 and residualized analyses remain distinct.

For the full sequence, use `python scripts/reproduce_paper.py --stage all`.
The public wrappers bind regenerated inputs to recorded stage receipts, then
call the existing implementations with a generated runtime configuration.
Only provenance hashes, manifest row counts, and checkpoint signatures are
bound to the new run. Cohort definitions, feature counts, seeds, model grids,
matching rules, and scientific assertions are retained. Original reference
hashes remain immutable in `results/manifests/expected_hashes.json`.

A receipt mismatch stops execution. Do not edit a completed input in place or
delete a checksum assertion to resume. Use a fresh checkout for a new analysis.
Legacy implementation identifiers are retained because they are also seed and
cache namespaces; public wrappers provide the descriptive command interface.

## Supplement generation

```bash
python scripts/build_supplement.py
```

This compiles the frozen LaTeX source into `build/supplement/` using three
`pdflatex` passes. A TeX installation with the source's declared packages is
required; compilation is separate from Python CI. It does not recompute
scientific values or overwrite the checked-in PDF. Supplement tables and
figures retain their frozen publication numbering.

## Validation commands

```bash
python -m compileall -q src scripts tests
python scripts/validate_repository.py
python scripts/verify_checksums.py --scope all
python -m pytest -q
```

For complete compatible local reference artifacts:

```bash
python -m pytest -q --run-integration -m integration
```

Reference integration tests retain exact original digest and manifest
assertions. Regenerated artifacts can differ in serialization or run metadata;
the packaged audit does not claim full numerical or bitwise reproduction.
Review numerical results against `results/selected/` separately from checking
within-run integrity. Synthetic CI success alone is not evidence that the full
cohort has been rerun.
