"""Portable orchestration and checksum-bound handoffs between analysis stages.

Reference digests are retained separately. New runs bind their own outputs
only after successful stage execution; scientific configurations are unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    'specificity': ('revision_v3_specificity.yaml', '09_revision_specificity.py', ('primary', 'robustness')),
    'comparison': ('revision_v3_section2.yaml', '10_revision_section2.py', ('primary', 'robustness', 'specificity')),
    'extended': ('revision_v3_section4.yaml', '11_revision_section4.py', ('primary', 'robustness', 'specificity', 'comparison')),
}
OUTPUTS = {
    'primary': ['data/interim', 'data/processed', 'results/tables', 'results/predictions', 'results/shap', 'MANUSCRIPT.md', 'config/ephys_features.yaml', 'config/gene_modules.yaml'],
    'robustness': ['results/robustness_v2', 'logs/robustness_v2'],
    'specificity': ['results/revision_v3/specificity', 'figures/revision_v3/specificity', 'logs/revision_v3/specificity', 'models/revision_v3/specificity'],
    'comparison': ['results/revision_v3/section2', 'figures/revision_v3/section2', 'logs/revision_v3/section2', 'models/revision_v3/section2'],
    'extended': ['results/revision_v3/section4', 'figures/revision_v3/section4', 'logs/revision_v3/section4', 'models/revision_v3/section4'],
}

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def receipt_path(stage: str, root: Path = ROOT) -> Path:
    return root / 'local/receipts' / (stage + '.json')

def write_receipt(stage: str, root: Path = ROOT) -> dict:
    files = []
    for name in OUTPUTS[stage]:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f'Stage output missing: {name}')
        files.extend(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else files.append(path)
    payload = {'stage': stage, 'files': {p.relative_to(root).as_posix(): sha256(p) for p in sorted(set(files))}}
    destination = receipt_path(stage, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    return payload

def verify_receipt(stage: str, root: Path = ROOT) -> dict:
    path = receipt_path(stage, root)
    if not path.is_file():
        raise FileNotFoundError(f'Complete the {stage} stage first; its verified receipt is missing.')
    record = json.loads(path.read_text(encoding='utf-8'))
    if record.get('stage') != stage or not record.get('files'):
        raise ValueError(f'Invalid {stage} stage receipt')
    for name, expected in record['files'].items():
        artifact = (root / name).resolve()
        if not artifact.is_relative_to(root.resolve()) or not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f'{stage} receipt mismatch: {name}')
    return record

def reference_or_run_digest(root: Path, stage: str, path: str, reference: str) -> str:
    """Use a completed local run's recorded digest, otherwise strict reference."""
    if not receipt_path(stage, root).is_file():
        return reference
    return verify_receipt(stage, root)['files'][path]

def _read_manifest(root: Path, path: Path) -> int:
    with path.open(encoding='utf-8', newline='') as stream:
        records = list(csv.DictReader(stream))
    seen = set()
    for row in records:
        name = row.get('relative_path', row.get('path'))
        file = (root / name).resolve()
        if name in seen or not file.is_relative_to(root.resolve()):
            raise ValueError('Duplicate or unsafe artifact-manifest path')
        seen.add(name)
        if not file.is_file() or file.stat().st_size != int(row['bytes']) or sha256(file) != row['sha256']:
            raise ValueError(f'Upstream manifest mismatch: {name}')
    if not records:
        raise ValueError('Empty upstream manifest')
    return len(records)

def prepare_config(stage: str, root: Path = ROOT) -> Path:
    filename, _, dependencies = SPECS[stage]
    verified = {}
    for dependency in dependencies:
        verified.update(verify_receipt(dependency, root)['files'])
    config = yaml.safe_load((root / 'config' / filename).read_text(encoding='utf-8'))
    paths, expected = config['paths'], config['expected']
    aliases = {'old_matched_sets_sha256': 'section1_matched_sets',
               'old_matching_assignments_sha256': 'section1_matching_assignments',
               'old_matching_quality_sha256': 'section1_matching_quality',
               'gene_qc_sha256': 'section1_gene_qc',
               'technical_targets_sha256': 'section1_technical_targets',
               'nav7_target_sha256': 'section1_nav7_target'}
    for key in list(expected):
        if key == 'control_target_sha256':
            expected[key] = {module: verified[(Path(paths['section1_control_targets']) / (module + '.parquet')).as_posix()]
                             for module in expected[key]}
        elif key.endswith('_sha256'):
            path_key = aliases.get(key, key[:-7])
            if path_key not in paths:
                raise ValueError(f'Unmapped provenance field: {key}')
            name = paths[path_key]
            if name not in verified:
                raise ValueError(f'Input not covered by completed stage receipts: {name}')
            expected[key] = verified[name]
    for upstream in ('section1', 'section2'):
        key = upstream + '_manifest'
        if key in paths:
            expected[key + '_rows'] = _read_manifest(root, root / paths[key])
            checkpoint = json.loads((root / paths[upstream + '_checkpoint']).read_text(encoding='utf-8'))
            expected[upstream + '_signature'] = checkpoint['signature']
    destination = root / 'config' / ('runtime_' + stage + '.yaml')
    text = yaml.safe_dump(config, sort_keys=False)
    if destination.exists() and destination.read_text(encoding='utf-8') != text:
        raise ValueError('Runtime inputs changed; use a fresh checkout for a new run.')
    destination.write_text(text, encoding='utf-8', newline='\n')
    return destination

def execute(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script), *args], cwd=ROOT, check=True)

def run_stage(stage: str, mode: str = 'full') -> None:
    if stage == 'primary':
        if receipt_path(stage).exists():
            verify_receipt(stage); print('Primary stage already verified.'); return
        execute('verify_checksums.py', '--scope', 'data')
        for script, args in [('01_audit.py', ()), ('02_build_dataset.py', ()),
                             ('03_train.py', ('--level', '0')), ('04_interpret.py', ()),
                             ('05_validate.py', ('--level', '0')), ('06_report.py', ('--level', '0'))]:
            execute(script, *args)
        subprocess.run([sys.executable, '-m', 'pytest', '-q', '--run-integration', 'tests/test_pipeline.py'], cwd=ROOT, check=True)
    elif stage == 'robustness':
        verify_receipt('primary')
        if receipt_path(stage).exists():
            verify_receipt(stage); print('Robustness stage already verified.'); return
        for script in ('run_robustness.py', '07_robustness_v2_duplicate.py', 'run_gene_omission.py'):
            execute(script)
        verify_receipt('primary')
    else:
        config = prepare_config(stage)
        execute(SPECS[stage][1], '--config', config.relative_to(ROOT).as_posix(), '--mode', mode)
        for upstream in SPECS[stage][2]:
            verify_receipt(upstream)
        if mode == 'smoke':
            return
    write_receipt(stage)

def entrypoint(stage: str) -> None:
    parser = argparse.ArgumentParser(description=f'Run the {stage} analysis using verified upstream outputs.')
    parser.add_argument('--mode', choices=('smoke', 'full', 'resume'), default='full')
    args = parser.parse_args()
    run_stage(stage, args.mode)
