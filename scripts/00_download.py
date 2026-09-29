#!/usr/bin/env python3
"""Acquire and validate the Allen cplAE-TE/public Patch-seq inputs.

Existing files are validated against the official Git blob SHA-1 and byte size.
The MAT file can therefore be reused from the workspace-level DATA directory
without downloading it again.  Every run rewrites the manifest atomically.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


BRANCH_COMMIT = "6c0be92e7a919321a4b877623354ee48d5f75921"
REPOSITORY = "https://github.com/AllenInstitute/coupledAE-patchseq"


@dataclass(frozen=True)
class Source:
    name: str
    source_url: str
    size: int
    retrieval_urls: tuple[str, ...]
    git_blob_sha1: str = ""
    expected_sha256: str = ""
    repository: str = ""
    branch_commit: str = ""
    optional_large: bool = False


SOURCES = (
    Source(
        "PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat",
        "https://github.com/AllenInstitute/coupledAE-patchseq/raw/refs/heads/cplAE-TE/data/proc/PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat",
        10_418_681,
        ("https://github.com/AllenInstitute/coupledAE-patchseq/raw/refs/heads/cplAE-TE/data/proc/PS_v5_beta_0-4_pc_scaled_ipfx_eqTE.mat",),
        git_blob_sha1="6270922577fca6005225df719c4358b7c80ca01b",
        repository=REPOSITORY,
        branch_commit=BRANCH_COMMIT,
    ),
    Source(
        "E_names.json",
        "https://raw.githubusercontent.com/AllenInstitute/coupledAE-patchseq/cplAE-TE/data/proc/E_names.json",
        3_036,
        ("https://raw.githubusercontent.com/AllenInstitute/coupledAE-patchseq/cplAE-TE/data/proc/E_names.json",),
        git_blob_sha1="d4132a4eac80ac23a36990c798461e083c6beba9",
        repository=REPOSITORY,
        branch_commit=BRANCH_COMMIT,
    ),
    Source(
        "E_feature_correspondences.txt",
        "https://raw.githubusercontent.com/AllenInstitute/coupledAE-patchseq/cplAE-TE/data/proc/E_feature_correspondences.txt",
        4_752,
        ("https://raw.githubusercontent.com/AllenInstitute/coupledAE-patchseq/cplAE-TE/data/proc/E_feature_correspondences.txt",),
        git_blob_sha1="8e364d71fd02013c6bf0afea290189924c4ea038",
        repository=REPOSITORY,
        branch_commit=BRANCH_COMMIT,
    ),
    Source(
        "20200711_patchseq_metadata_mouse.csv",
        "https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/5e/2a/5e2a5936-61da-4e09-b6da-74ab97ce1b02/20200711_patchseq_metadata_mouse.csv",
        1_065_200,
        (
            "https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/5e/2a/5e2a5936-61da-4e09-b6da-74ab97ce1b02/20200711_patchseq_metadata_mouse.csv",
            "https://web.archive.org/web/20220721064421id_/https://brainmapportal-live-4cc80a57cd6e400d854-f7fdcae.divio-media.net/filer_public/5e/2a/5e2a5936-61da-4e09-b6da-74ab97ce1b02/20200711_patchseq_metadata_mouse.csv",
        ),
        expected_sha256="821de6ce5e69048957a24157c5bfaee0995979cdf3d24f8f5d3a762461fec60d",
    ),
    Source(
        "20200513_Mouse_PatchSeq_Release_count.v2.csv.tar",
        "https://data.nemoarchive.org/other/AIBS/AIBS_patchseq/transcriptome/scell/SMARTseq/processed/analysis/20200611/20200513_Mouse_PatchSeq_Release_count.v2.csv.tar",
        444_999_680,
        (
            "https://data.nemoarchive.org/other/AIBS/AIBS_patchseq/transcriptome/scell/SMARTseq/processed/analysis/20200611/20200513_Mouse_PatchSeq_Release_count.v2.csv.tar",
            "https://web.archive.org/web/20230630092909id_/https://data.nemoarchive.org/other/AIBS/AIBS_patchseq/transcriptome/scell/SMARTseq/processed/analysis/20200611/20200513_Mouse_PatchSeq_Release_count.v2.csv.tar",
        ),
        expected_sha256="c413201dede43d9c75437567d54312e4ffb2fc9952e41526e2ea94bd0aee627b",
        optional_large=True,
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_valid(path: Path, source: Source) -> bool:
    if not path.is_file() or path.stat().st_size != source.size:
        return False
    if source.git_blob_sha1 and git_blob_sha1(path) != source.git_blob_sha1:
        return False
    if source.expected_sha256 and sha256(path) != source.expected_sha256:
        return False
    return True


def download_atomic(source: Source, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{source.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        failures = []
        for url in source.retrieval_urls:
            temporary.unlink(missing_ok=True)
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(request, timeout=600) as response, temporary.open("wb") as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                if not is_valid(temporary, source):
                    raise RuntimeError("size/checksum mismatch")
                os.replace(temporary, destination)
                return url
            except Exception as error:  # continue to an explicitly recorded official archive fallback
                failures.append(f"{url}: {type(error).__name__}: {error}")
        raise RuntimeError(f"All retrieval URLs failed for {source.name}: {' | '.join(failures)}")
    finally:
        temporary.unlink(missing_ok=True)


def acquire(source: Source, raw_dir: Path, reuse_dirs: list[Path]) -> tuple[Path, str, str]:
    destination = raw_dir / source.name
    if is_valid(destination, source):
        return destination, "verified-existing", source.retrieval_urls[-1]
    for directory in reuse_dirs:
        candidate = directory / source.name
        if candidate.resolve() != destination.resolve() and is_valid(candidate, source):
            shutil.copy2(candidate, destination)
            if not is_valid(destination, source):
                raise RuntimeError(f"Copied file failed validation: {source.name}")
            return destination, f"verified-local-copy:{candidate.resolve()}", candidate.resolve().as_uri()
    retrieval_url = download_atomic(source, destination)
    return destination, "downloaded-and-verified", retrieval_url


def write_manifest(rows: list[dict[str, object]], manifest: Path) -> None:
    fields = [
        "relative_path",
        "source_url",
        "acquisition_url",
        "repository",
        "branch_commit",
        "bytes",
        "sha256",
        "git_blob_sha1",
        "acquisition_status",
        "verified_utc",
    ]
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_suffix(manifest.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--reuse-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory to search for already downloaded official files (repeatable).",
    )
    parser.add_argument(
        "--include-transcriptomics",
        action="store_true",
        help="Download the 445 MB public count archive if it is not already present.",
    )
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    raw_dir = project_root / "data" / "raw"
    default_reuse = project_root.parent / "DATA"
    reuse_dirs = [p.resolve() for p in args.reuse_dir]
    if default_reuse.is_dir() and default_reuse.resolve() not in reuse_dirs:
        reuse_dirs.append(default_reuse.resolve())

    rows: list[dict[str, object]] = []
    verified_at = datetime.now(timezone.utc).isoformat()
    for source in SOURCES:
        destination = raw_dir / source.name
        if source.optional_large and not args.include_transcriptomics and not is_valid(destination, source):
            print(f"{source.name}: skipped (pass --include-transcriptomics)")
            continue
        path, status, acquisition_url = acquire(source, raw_dir, reuse_dirs)
        rows.append(
            {
                "relative_path": path.relative_to(project_root).as_posix(),
                "source_url": source.source_url,
                "acquisition_url": acquisition_url,
                "repository": source.repository,
                "branch_commit": source.branch_commit,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
                "git_blob_sha1": git_blob_sha1(path) if source.git_blob_sha1 else "",
                "acquisition_status": status,
                "verified_utc": verified_at,
            }
        )
        print(f"{source.name}: {status}; bytes={path.stat().st_size}; sha256={sha256(path)}")
    write_manifest(rows, project_root / "data" / "MANIFEST.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
