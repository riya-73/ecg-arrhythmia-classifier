"""MIT-BIH MLII filtering and annotation-centered beat extraction."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

NORMAL = {'N'}

def preprocess_record(record, ann, sampling_rate=360, window_samples=252):
    signal = np.asarray(record.p_signal, dtype=np.float32)
    names = list(record.sig_name)
    # MLII is preferred; otherwise choose lead II if it is the best available match.
    choices = [i for i,n in enumerate(names) if n.upper().replace(' ','') in ('MLII','II')]
    if not choices: raise ValueError(f'No MLII/II lead in record: {names}')
    x = signal[:, choices[0]]
    sos = butter(4, [0.5, 40.0], btype='bandpass', fs=sampling_rate, output='sos')
    x = sosfiltfilt(sos, x).astype(np.float32)  # removes baseline drift while retaining QRS
    half = window_samples // 2
    beats=[]; labels=[]; symbols=[]; positions=[]
    for r, sym in zip(ann.sample, ann.symbol):
        if sym == '+' or r-half < 0 or r+(window_samples-half) > len(x): continue
        beat=x[r-half:r+(window_samples-half)].copy()
        # per-beat z normalization with a stable floor
        sd=float(beat.std()); beat=(beat-float(beat.mean())) / max(sd, 1e-6)
        beats.append(beat); labels.append(0 if sym in NORMAL else 1); symbols.append(sym); positions.append(int(r))
    return np.asarray(beats,np.float32), np.asarray(labels,np.int64), np.asarray(symbols), np.asarray(positions,np.int64)
