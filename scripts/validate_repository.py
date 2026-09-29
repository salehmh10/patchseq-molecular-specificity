"""Validate schemas, author consistency, imports, and packaged evidence."""
from __future__ import annotations
import importlib
import json
import sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.verify_checksums import verify_list

def validate() -> dict[str, int]:
    required = ['README.md', 'LICENSE', 'NOTICE', 'AUTHORS.md', 'CITATION.cff',
                'CITATION.bib', '.zenodo.json', 'DATA_LICENSES.md', 'requirements.txt',
                'data/MANIFEST.tsv', 'data/DATA_PROVENANCE.tsv',
                'supplement/Supplementary_Material.pdf', '.github/workflows/tests.yml']
    for name in required:
        if not (ROOT / name).is_file():
            raise ValueError(f'Missing required file: {name}')
    parsed = 0
    for path in ROOT.rglob('*'):
        if any(p in {'.git', '.venv', '__pycache__', 'local', 'build', 'logs', 'models'} for p in path.relative_to(ROOT).parts):
            continue
        if path.suffix in ('.yaml', '.yml', '.cff'):
            yaml.safe_load(path.read_text(encoding='utf-8')); parsed += 1
        elif path.suffix == '.json':
            json.loads(path.read_text(encoding='utf-8')); parsed += 1
    cff = yaml.safe_load((ROOT / 'CITATION.cff').read_text(encoding='utf-8'))
    zenodo = json.loads((ROOT / '.zenodo.json').read_text(encoding='utf-8'))
    if cff['cff-version'] != '1.2.0' or cff['license'] != 'MIT':
        raise ValueError('Invalid citation version or code license')
    expected = [('Saleh', 'Mohammadhasani'), ('Reza', 'Kazemeynimoghaddam'),
                ('Amirreza', 'Khadempir'), ('Pedram', 'Hamidirad'), ('Amirreza', 'Dehghan Nayeri')]
    if [(a['given-names'], a['family-names']) for a in cff['authors']] != expected:
        raise ValueError('Author order mismatch')
    positions = {'AUTHORS.md': [], 'README.md': [], 'CITATION.bib': []}
    for author, creator in zip(cff['authors'], zenodo['creators'], strict=True):
        given, family = author['given-names'], author['family-names']
        if creator != {'name': f'{family}, {given}', 'affiliation': author['affiliation']}:
            raise ValueError('Zenodo creator metadata mismatch')
        for name in positions:
            content = (ROOT / name).read_text(encoding='utf-8')
            needle = f'{family}, {given}' if name.endswith('.bib') else f'{given} {family}'
            positions[name].append(content.index(needle))
        if author['affiliation'] not in (ROOT / 'AUTHORS.md').read_text(encoding='utf-8'):
            raise ValueError('Affiliation missing from author table')
    if any(order != sorted(order) for order in positions.values()):
        raise ValueError('Inconsistent metadata author order')
    for path in (ROOT / 'src').glob('*.py'):
        importlib.import_module('src.' + path.stem)
    selected = verify_list(ROOT / 'results/manifests/selected_artifact_checksums.sha256', ROOT)
    supplement = verify_list(ROOT / 'supplement/CHECKSUMS.sha256', ROOT / 'supplement')
    return {'parsed_metadata_files': parsed, 'authors': 5, 'selected_files': selected, 'supplement_files': supplement}

if __name__ == '__main__':
    print(json.dumps(validate(), indent=2))
