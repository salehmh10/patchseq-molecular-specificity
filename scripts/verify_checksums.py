"""Verify packaged artifacts or locally acquired data without downloading."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

def verify_list(manifest: Path, base: Path) -> int:
    count = 0
    for line in manifest.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = (base / name.lstrip('*')).resolve()
        if not path.is_relative_to(base.resolve()):
            raise ValueError(f'Unsafe checksum path: {name}')
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Checksum mismatch or missing file: {name}')
        count += 1
    if not count:
        raise ValueError('Empty checksum manifest')
    return count

def verify_data(root: Path = ROOT) -> int:
    with (root / 'data/DATA_PROVENANCE.tsv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream, delimiter='\t'))
    for row in rows:
        path = (root / row['relative_path']).resolve()
        if not path.is_relative_to((root / 'data/raw').resolve()):
            raise ValueError('Source destination is outside data/raw')
        if not path.is_file() or path.stat().st_size != int(row['bytes']) or digest(path) != row['sha256']:
            raise ValueError(f"Source checksum mismatch or missing file: {row['relative_path']}")
    if len(rows) != 5:
        raise ValueError('Expected five source records')
    return len(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope', choices=('all', 'selected', 'supplement', 'data'), default='all')
    args = parser.parse_args()
    count = 0
    if args.scope in ('all', 'selected'):
        count += verify_list(ROOT / 'results/manifests/selected_artifact_checksums.sha256', ROOT)
    if args.scope in ('all', 'supplement'):
        count += verify_list(ROOT / 'supplement/CHECKSUMS.sha256', ROOT / 'supplement')
    if args.scope == 'data':
        count += verify_data()
    print(f'Verified {count} files ({args.scope}).')

if __name__ == '__main__':
    main()
