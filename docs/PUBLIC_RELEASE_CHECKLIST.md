# Public release checklist

The candidate repository and draft release remain private. The following are
future maintainer actions; they are not performed by the candidate workflow.

- Confirm all five authors, affiliations, and institutional approvals.
- Confirm rights for original code, derived tables, figures, and supplementary
  documents. Review upstream Allen/NeMO terms without assuming MIT covers data.
- Re-run privacy, credential, file-size, dependency, and source-link checks.
- Confirm current CI and fresh-clone checks succeed. Review any full-data
  reproduction results separately from synthetic CI.
- Finalize a version and update CFF, BibTeX, change log, and archive metadata.
- Review the complete draft release and its checksums; do not publish by accident.
- With explicit authorization, change repository visibility to public and
  verify its settings and published contents.
- Connect the repository through the intended Zenodo account and enable the
  repository integration. Verify the creators and affiliations in order.
- Publish the approved tagged GitHub release after enabling archival
  integration, then verify the resulting Zenodo record, assets, and rights.
- Use the DOI actually assigned by Zenodo. Add the real version and/or concept
  DOI to the citation files as appropriate; never substitute an invented DOI.

`.zenodo.json` records the current candidate version, 1.0.0-rc1. Before a final
public release, update all version metadata together to 1.0.0 and create the
approved `v1.0.0` tag on the validated commit. The metadata file does not
activate an integration or assign a DOI. No Zenodo connection is part of the
private candidate.
