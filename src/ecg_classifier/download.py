"""Download only split-relevant MIT-BIH files using WFDB's PhysioNet client."""
from pathlib import Path
import wfdb

def download(records_dir='data/raw', records=None):
    path=Path(records_dir); path.mkdir(parents=True,exist_ok=True)
    if records is None:
        from .split import load_config
        c=load_config(); records=sorted(set(c['ds1']+c['ds2'])-set(c['paced_records']))
    files=[f'{r}.{ext}' for r in records for ext in ('hea','dat','atr')]
    wfdb.dl_files('mitdb',str(path),files=files,keep_subdirs=False,overwrite=False)
    missing=[f for f in files if not (path/f).exists() or (path/f).stat().st_size==0]
    if missing: raise RuntimeError(f'Missing downloaded MIT-BIH files: {missing}')
    print(f'WFDB download verified: {len(records)} records, {len(files)} files in {path}',flush=True)
    return path

if __name__ == '__main__':
    from .split import load_config
    cfg=load_config(); download(cfg['records_dir'],sorted(set(cfg['ds1']+cfg['ds2'])-set(cfg['paced_records'])))
