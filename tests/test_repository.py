"""Publication artifacts and new-run provenance must fail closed."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts.validate_repository import validate
from scripts.verify_checksums import verify_list
from src.reproduction import verify_receipt, sha256, prepare_config
from src.revision_section4 import bh_adjust

ROOT = Path(__file__).resolve().parents[1]

def test_repository_metadata_and_checksums():
    assert validate()['authors'] == 5

def test_selected_specificity_reconstructs_empirical_p_and_bh():
    for file in ['specificity_bh6.csv', 'residual_specificity_bh6.csv']:
        frame = pd.read_csv(ROOT / 'results/selected' / file)
        assert len(frame) == 6
        p = frame['raw_p'] if 'raw_p' in frame else frame['residual_p']
        np.testing.assert_allclose(p, (frame.extreme_count + 1) / (frame.control_n + 1), atol=1e-15)
        np.testing.assert_allclose(frame.bh_q, bh_adjust(p), atol=1e-15)
        assert frame.family_n.eq(6).all()

def test_receipt_rejects_missing_and_mutated_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        verify_receipt('primary', tmp_path)
    artifact = tmp_path / 'data.csv'
    artifact.write_text('a\n1\n')
    receipts = tmp_path / 'local/receipts'; receipts.mkdir(parents=True)
    (receipts / 'primary.json').write_text(json.dumps({'stage':'primary','files':{'data.csv':sha256(artifact)}}))
    assert verify_receipt('primary', tmp_path)['stage'] == 'primary'
    artifact.write_text('a\n2\n')
    with pytest.raises(ValueError, match='receipt mismatch'):
        verify_receipt('primary', tmp_path)

def test_checksum_verifier_rejects_path_escape(tmp_path):
    manifest = tmp_path / 'CHECKSUMS.sha256'
    manifest.write_text('0' * 64 + '  ../outside.txt\n')
    with pytest.raises(ValueError, match='Unsafe'):
        verify_list(manifest, tmp_path)

def test_runtime_binding_retains_scientific_settings(tmp_path):
    import yaml
    (tmp_path / 'config').mkdir()
    (tmp_path / 'local/receipts').mkdir(parents=True)
    data = tmp_path / 'input.csv'; data.write_text('x\n1\n')
    for stage in ('primary','robustness'):
        (tmp_path / 'local/receipts' / (stage+'.json')).write_text(json.dumps({'stage':stage,'files':{'input.csv':sha256(data)}}))
    config = {'seed':42,'paths':{'primary_oof':'input.csv'},'expected':{'cohort_n':3410,'feature_n':24,'primary_oof_sha256':'old'},'random_controls':{'full_controls_per_module':200}}
    (tmp_path / 'config/revision_v3_specificity.yaml').write_text(yaml.safe_dump(config))
    path = prepare_config('specificity',tmp_path)
    rebound = yaml.safe_load(path.read_text())
    assert rebound['expected']['primary_oof_sha256'] == sha256(data)
    rebound['expected']['primary_oof_sha256'] = 'old'
    assert rebound == config
