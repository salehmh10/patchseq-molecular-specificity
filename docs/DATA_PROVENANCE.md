# Data provenance

`data/DATA_PROVENANCE.tsv` preserves the verified acquisition records for
all five required sources. `data/MANIFEST.tsv` is the analysis-compatible
manifest. Records contain official URLs, actual acquisition URLs, available
upstream commit/blob identities, file sizes, SHA-256 values, and verification
timestamps. No raw data are committed.

The Allen processed-release commit is
`6c0be92e7a919321a4b877623354ee48d5f75921`. Retrieval from a moving branch is
accepted only if the recorded content identity matches. Metadata and count
archives include recorded archival fallbacks. Availability and terms remain
controlled by the original providers.

The Supplement's source checksums were validated before curation. Distributed
scientific tables, figure files, PDF, and LaTeX source retain their original
bytes. New package checksums describe the curated package, rather than an
unmodified copy of an earlier distribution. The large complete Table S44 is
available in the supplementary release archive.

`results/manifests/expected_hashes.json` records full reference-analysis
contracts. Runtime receipts identify inputs and outputs of a new reproduction
run without changing those reference records.
