# Data access

Raw Allen/NeMO inputs are excluded because of their size and third-party
distribution terms. Paths below are relative to the repository root.

| File in `data/raw/` | Official source | Approximate size |
|---|---|---:|
| `PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat` | Allen Institute coupledAE-patchseq, `cplAE-TE` processed release | 10.4 MB |
| `E_names.json` | Same Allen repository | 3 KB |
| `E_feature_correspondences.txt` | Same Allen repository | 5 KB |
| `20200711_patchseq_metadata_mouse.csv` | Allen Brain Map mouse Patch-seq metadata | 1.1 MB |
| `20200513_Mouse_PatchSeq_Release_count.v2.csv.tar` | NeMO AIBS Patch-seq transcriptome release, 20200611 | 445 MB |

`MANIFEST.tsv` records exact URLs, byte counts, SHA-256 values, available Git
blob hashes, and the upstream commit. `DATA_PROVENANCE.tsv` preserves the
original acquisition record. These are the expected checksums, rather than
hashes inferred from filenames.

Run `python scripts/00_download.py --include-transcriptomics` to acquire all
five files. Existing valid files are reused. An existing local data directory
may be supplied with `--reuse-dir local/source_data`. Recorded archive URLs
are fallbacks for unavailable original endpoints. Every acquired file must
match its expected byte count and SHA-256 or upstream Git blob hash.

Run `python scripts/verify_checksums.py --scope data` to independently check
all five files against the committed provenance. Acquisition may update the
working manifest; the immutable provenance record remains available.

Preprocessing creates `data/interim/` and `data/processed/`. Neither directory
is versioned. Identity joins, count denominators, gene coverage, and donor-fold
separation are checked by the analysis code. See `DATA_LICENSES.md` before
using or redistributing upstream material.
