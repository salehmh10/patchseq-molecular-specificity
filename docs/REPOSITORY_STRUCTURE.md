# Repository structure

| Directory | Contents |
|---|---|
| `src/` | Executed analytical implementations and portable stage support |
| `scripts/` | Data acquisition, primary pipeline, robustness analyses, public wrappers, smoke and validation tools |
| `config/` | Recorded model, feature, module, and control settings |
| `tests/` | Synthetic tests and opt-in local artifact tests |
| `data/` | Data placement, acquisition records, expected hashes |
| `results/selected/` | Small frozen reference tables |
| `results/manifests/` | Selected-result checksums and full reference digest contracts |
| `figures/manuscript/` | Final manuscript raster panels |
| `figures/supplementary/` | Index pointing to the canonical supplementary graphics |
| `supplement/` | Frozen PDF/source and supporting machine-readable material |
| `docs/` | Scientific and technical documentation |
| `.github/` | CI and contribution templates |

Generated data, predictions, models, logs, runtime configurations, and local
receipts are ignored. Historical implementation names remain where they
encode import, deterministic-seed, or cache compatibility. No internal project
Git history is included.
