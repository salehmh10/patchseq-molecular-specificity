# Supplementary material

The final 12-page `Supplementary_Material.pdf` and matching LaTeX source are
preserved from the validated publication package. Scientific contents were
not recomputed during repository curation.

Tables S1–S50 cover cohort construction, model specifications, prediction,
inference, controls, technical targets, consistency, sensitivity analyses,
attribution, and provenance. Small tables are included in `tables/`. Table S44
is approximately 16 MB and is supplied only in the complete supplementary ZIP
attached to the draft candidate release. `SUBCLASS_ELIGIBILITY.csv` supports
the within-subclass table. Figures S1–S5 have PDF and 300-dpi PNG versions in
`figures/`; some frozen plot layers are raster, so not every PDF is wholly vector.

Run `python scripts/verify_checksums.py --scope supplement` from the repository
root. `CHECKSUMS.sha256` covers every distributed file in this directory except
the checksum list itself. The full archive has a separate internal checksum
list that additionally covers Table S44. The archive also contains a complete
data-free source snapshot so its validation commands can run after extraction.
`REPRODUCTION_COMMANDS.txt` lists offline validation and full-analysis commands;
run them from the extracted repository root.

Run `python scripts/build_supplement.py` with `pdflatex` installed to compile
three passes into `build/supplement/`. The checked-in PDF remains unchanged.
The source reads its frozen figure files directly and does not require the
complete Table S44 CSV for PDF compilation.

The original software inventory is retained in `SOFTWARE_VERSIONS.txt` and
Table S48. Reference test counts in Table S49 describe the frozen analysis;
they are not the repository's current synthetic CI test count.
