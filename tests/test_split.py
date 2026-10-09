import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from ecg_classifier.split import load_config, partitions, validate_patient_split

def test_no_patient_crosses_any_split():
    cfg=load_config(Path(__file__).resolve().parents[1]/'config.yaml')
    p=partitions(cfg)
    train=set(p['train']); val=set(p['val']); test=set(p['test'])
    assert not train & val
    assert not train & test
    assert not val & test
    assert validate_patient_split(set(cfg['ds1']),set(cfg['ds2']),set(cfg['validation']),set(cfg['paced_records']))

def test_shared_subject_records_201_202_cannot_cross_split():
    with pytest.raises(AssertionError,match='patient overlap'):
        validate_patient_split({'201'},{'202'})
