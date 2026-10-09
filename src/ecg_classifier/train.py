"""End-to-end MIT-BIH benchmark pipeline."""
from pathlib import Path
import json, random, time, sys
import numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, average_precision_score, accuracy_score)
from sklearn.ensemble import RandomForestClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from .split import load_config, partitions
from .preprocess import preprocess_record
from .model import ECGCNN
from .dataset import BeatDataset
from .baseline import handcrafted, build_baseline
import wfdb

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(4)

def read_record(record_id, cfg):
    base=Path(cfg['records_dir'])/record_id
    rec=wfdb.rdrecord(str(base)); ann=wfdb.rdann(str(base),'atr')
    x,y,sym,pos=preprocess_record(rec,ann,cfg['sampling_rate'],cfg['window_samples'])
    return x,y,sym,pos

def get_data(cfg, parts):
    cache=Path(cfg['processed_path'])
    if cache.exists():
        z=np.load(cache,allow_pickle=False)
        # verify cache contains precisely expected records
        all_ids=set(z['record'].astype(str)); expected=set(sum(parts.values(),[]))
        if all_ids==expected:
            return {k:[(z['X'][z['record']==r],z['y'][z['record']==r],z['symbol'][z['record']==r],z['pos'][z['record']==r],r) for r in v] for k,v in parts.items()}
    data={}
    for split,records in parts.items():
        items=[]
        for r in records:
            print(f'Preprocessing record {r} ({split})',flush=True)
            x,y,sym,pos=read_record(r,cfg); items.append((x,y,sym,pos,r))
        data[split]=items
    flat=[(x,y,s,p,r) for split in data for x,y,s,p,r in data[split]]
    np.savez_compressed(cache,X=np.concatenate([v[0] for v in flat]),y=np.concatenate([v[1] for v in flat]),symbol=np.concatenate([v[2].astype('U4') for v in flat]),pos=np.concatenate([v[3] for v in flat]),record=np.concatenate([np.repeat(v[4],len(v[1])).astype('U3') for v in flat]))
    return data

def flatten(items):
    return np.concatenate([v[0] for v in items]),np.concatenate([v[1] for v in items]),np.concatenate([v[2] for v in items]),np.concatenate([v[3] for v in items]),np.concatenate([np.repeat(v[4],len(v[1])).astype('U3') for v in items])

def score(y,prob):
    pred=prob.argmax(1)
    return {'accuracy':float(accuracy_score(y,pred)),'roc_auc':float(roc_auc_score(y,prob[:,1])),'pr_auc':float(average_precision_score(y,prob[:,1])),'confusion_matrix':confusion_matrix(y,pred,labels=[0,1]).tolist(),'classification_report':classification_report(y,pred,labels=[0,1],target_names=['Normal','Abnormal'],output_dict=True,zero_division=0)}

def plot_curve(hist,out):
    fig,ax=plt.subplots(figsize=(7,4)); ax.plot(hist['train_loss'],label='Train'); ax.plot(hist['val_loss'],label='Validation'); ax.set(xlabel='Epoch',ylabel='Weighted cross-entropy',title='Training curves'); ax.legend(); fig.tight_layout(); fig.savefig(out,dpi=150); plt.close(fig)

def run():
    root=Path.cwd(); cfg=load_config(root/'config.yaml'); parts=partitions(cfg); seed_all(cfg['seed'])
    data=get_data(cfg,parts)
    tx,ty,_,txp,txr=flatten(data['train']); vx,vy,_,_,_=flatten(data['val']); qx,qy,qs,qp,qr=flatten(data['test'])
    print('Counts train/val/test:',len(ty),len(vy),len(qy),'class counts test',np.bincount(qy,minlength=2),flush=True)
    model=ECGCNN(); weights=torch.tensor(len(ty)/(2*np.maximum(np.bincount(ty,minlength=2),1)),dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['train']['learning_rate'],weight_decay=1e-4); sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=1)
    lossfn=torch.nn.CrossEntropyLoss(weight=weights)
    tr=DataLoader(BeatDataset(tx,ty),batch_size=cfg['train']['batch_size'],shuffle=True); va=DataLoader(BeatDataset(vx,vy),batch_size=1024)
    hist={'train_loss':[],'val_loss':[]}; best=float('inf'); stale=0; bestpath=Path('models/cnn.pt')
    for epoch in range(cfg['train']['epochs']):
        model.train(); total=0
        for xb,yb in tr:
            opt.zero_grad(set_to_none=True); loss=lossfn(model(xb),yb); loss.backward(); opt.step(); total+=loss.item()*len(yb)
        model.eval(); vl=0
        with torch.no_grad():
            for xb,yb in va: vl+=lossfn(model(xb),yb).item()*len(yb)
        tl=total/len(ty); vl/=len(vy); hist['train_loss'].append(tl); hist['val_loss'].append(vl); sched.step(vl)
        print(f'Epoch {epoch+1}: train_loss={tl:.5f} val_loss={vl:.5f}',flush=True)
        if vl<best:
            best=vl; stale=0; torch.save(model.state_dict(),bestpath)
        else: stale+=1
        if stale>=cfg['train']['patience']: break
    model.load_state_dict(torch.load(bestpath,map_location='cpu',weights_only=True)); model.eval()
    with torch.no_grad(): cnnprob=torch.softmax(model(torch.from_numpy(qx[:,None,:])),dim=1).numpy()
    # RF is trained only on DS1 train; validation and DS2 are never used for fitting/tuning.
    # Cap RF fitting size deterministically to keep CPU execution practical; CNN uses all beats.
    rng=np.random.default_rng(cfg['seed']); idx=np.arange(len(ty))
    if len(idx)>25000: idx=rng.choice(idx,25000,replace=False)
    train_feats=handcrafted(tx[idx],txp[idx],txr[idx]); test_feats=handcrafted(qx,qp,qr)
    rf=build_baseline(cfg['seed']); rf.fit(train_feats,ty[idx]); rfprob=rf.predict_proba(test_feats)
    # Align probability columns in edge cases
    if rfprob.shape[1]==1:
        pp=np.zeros((len(qy),2)); pp[:,int(rf.classes_[0])]=1; rfprob=pp
    metrics={'metadata':{'data':'PhysioNet MIT-BIH Arrhythmia Database, 360 Hz','lead':'MLII (recorded lead-II fallback only)','window_samples':cfg['window_samples'],'window_seconds':cfg['window_samples']/cfg['sampling_rate'],'seed':cfg['seed'],'records':parts,'counts':{'train':len(ty),'validation':len(vy),'test':len(qy)},'test_class_counts':{'normal':int((qy==0).sum()),'abnormal':int((qy==1).sum())},'rf_train_beats':len(idx)},'cnn':score(qy,cnnprob),'random_forest':score(qy,rfprob)}
    Path('results/metrics.json').write_text(json.dumps(metrics,indent=2))
    plot_curve(hist,'results/figures/loss_curves.png')
    # DS2 confusion matrix only
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,(name,m) in zip(axs,[('CNN',metrics['cnn']),('Random forest',metrics['random_forest'])]):
        cm=np.asarray(m['confusion_matrix']); ax.imshow(cm,cmap='Blues'); ax.set_title(name+' (DS2)'); ax.set_xticks([0,1],['Normal','Abnormal']); ax.set_yticks([0,1],['Normal','Abnormal']); ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
    fig.tight_layout(); fig.savefig('results/figures/confusion_matrices.png',dpi=150); plt.close(fig)
    # Six correctly and six incorrectly classified beats (mixed classes when available).
    pred=cnnprob.argmax(1); good=np.where(pred==qy)[0]; bad=np.where(pred!=qy)[0]
    chosen=np.r_[good[:6],bad[:6]]
    fig,axs=plt.subplots(3,4,figsize=(13,7));
    for j,ax in enumerate(axs.flat):
        if j>=len(chosen): ax.axis('off'); continue
        i=chosen[j]; ax.plot(np.arange(len(qx[i]))/cfg['sampling_rate'],qx[i],lw=1); ax.set_title(f"{'Correct' if pred[i]==qy[i] else 'Wrong'} | y={qy[i]} p={pred[i]} | {qr[i]} {qs[i]}",fontsize=8); ax.set_xlabel('s')
    fig.tight_layout(); fig.savefig('results/figures/example_beats.png',dpi=150); plt.close(fig)
    # Input-gradient saliency for three DS2 examples; never feeds back into training.
    torch.manual_seed(cfg['seed']); ids=np.linspace(0,len(qy)-1,3,dtype=int); fig,axs=plt.subplots(3,1,figsize=(9,7))
    for ax,i in zip(axs,ids):
        inp=torch.tensor(qx[i:i+1,None,:],requires_grad=True); model.zero_grad(); model(inp)[0,int(pred[i])].backward(); sal=inp.grad.detach().abs().numpy().reshape(-1); sal/=max(sal.max(),1e-8)
        ax.plot(np.arange(len(qx[i]))/cfg['sampling_rate'],qx[i],color='#335c81'); ax2=ax.twinx(); ax2.fill_between(np.arange(len(sal))/cfg['sampling_rate'],sal,color='#e76f51',alpha=.35); ax.set_title(f'DS2 beat {i}: true={qy[i]}, predicted={pred[i]}',fontsize=9)
    fig.tight_layout(); fig.savefig('results/figures/saliency.png',dpi=150); plt.close(fig)
    # Persist measured training history for reproducibility.
    Path('results/history.json').write_text(json.dumps(hist,indent=2))
    print('RESULTS_JSON='+json.dumps({k:{'accuracy':v['accuracy'],'roc_auc':v['roc_auc'],'pr_auc':v['pr_auc'],'normal':v['classification_report']['Normal'],'abnormal':v['classification_report']['Abnormal']} for k,v in [('cnn',metrics['cnn']),('random_forest',metrics['random_forest'])]},indent=2),flush=True)
if __name__=='__main__': run()
