import numpy as np
from sklearn.ensemble import RandomForestClassifier

def handcrafted(x, positions=None, record_ids=None, sampling_rate=360):
    # Window morphology summaries plus record-local preceding/following RR intervals.
    n=len(x); rr_prev=np.zeros(n,dtype=np.float32); rr_next=np.zeros(n,dtype=np.float32)
    if positions is not None and record_ids is not None:
        ids=np.asarray(record_ids).astype(str); pos=np.asarray(positions)
        for rec in np.unique(ids):
            ix=np.flatnonzero(ids==rec); order=ix[np.argsort(pos[ix])]
            delta=np.diff(pos[order]).astype(np.float32)/sampling_rate
            rr_prev[order[1:]]=delta; rr_next[order[:-1]]=delta
    feats=np.empty((n,12),dtype=np.float32)
    for i,b in enumerate(x):
        q=np.quantile(b,[.05,.25,.5,.75,.95]); d=np.diff(b)
        peak=int(np.argmax(np.abs(b))); width=int(np.sum(np.abs(b)>0.5*np.max(np.abs(b))))
        feats[i]=[*q,float(np.sqrt(np.mean(b*b))),float(np.mean(np.abs(d))),float(np.std(d)),peak/len(b),width/len(b),rr_prev[i],rr_next[i]]
    return feats

def build_baseline(seed=42):
    return RandomForestClassifier(n_estimators=140,min_samples_leaf=2,max_features='sqrt',class_weight='balanced_subsample',n_jobs=-1,random_state=seed)
