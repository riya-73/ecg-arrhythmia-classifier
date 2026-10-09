"""Record-level patient-independent benchmark split."""
import yaml
from pathlib import Path

# MIT-BIH records 201 and 202 are separate recordings from the same person.
PATIENT_GROUP = {'201':'subject_201_202', '202':'subject_201_202'}

def patient_id(record):
    record=str(record)
    return PATIENT_GROUP.get(record, f'record_{record}')

def load_config(path='config.yaml'):
    return yaml.safe_load(Path(path).read_text())

def validate_patient_split(ds1, ds2, validation=(), paced=()):
    ds1,ds2=set(ds1),set(ds2); validation=set(validation); paced=set(paced)
    overlap={patient_id(r) for r in ds1}&{patient_id(r) for r in ds2}
    assert not overlap, f'DS1/DS2 patient overlap: {overlap}'
    assert validation <= ds1, 'Validation patients must be carved from DS1'
    train=ds1-validation
    overlap={patient_id(r) for r in train}&{patient_id(r) for r in validation}
    assert not overlap, f'DS1 train/validation patient overlap: {overlap}'
    overlap={patient_id(r) for r in validation}&{patient_id(r) for r in ds2}
    assert not overlap, f'Validation/test patient overlap: {overlap}'
    assert not ((ds1|ds2)&paced), 'Paced records must be excluded'
    return True

def partitions(config):
    validate_patient_split(config['ds1'], config['ds2'], config['validation'], config['paced_records'])
    val=set(config['validation'])
    return {'train':[r for r in config['ds1'] if r not in val], 'val':sorted(val), 'test':config['ds2']}
