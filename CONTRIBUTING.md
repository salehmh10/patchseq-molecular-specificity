# Contributing

Open an issue describing the problem and the smallest reproducible example.
Use synthetic data for bug reports whenever possible. Do not upload raw data,
credentials, or private files.

Run `python scripts/validate_repository.py`, `python scripts/smoke_test.py`,
and `python -m pytest -q` before submitting a change. Explain any change to
cohort selection, target definitions, matching, model selection, donor splits,
or multiplicity families. Keep reference results immutable and identify new
analyses separately. Include package versions and the relevant configuration.

The integration tests retain strict scientific assertions and are enabled
explicitly with `--run-integration` when their local artifacts are available.
