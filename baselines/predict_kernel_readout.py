"""Strict deployment inference for frozen classical kernel readouts."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
WINDOW, BINS = 2048, 128

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def _finite(a,name):
    if not isinstance(a,np.ndarray) or a.dtype.kind not in 'fc' or not np.isfinite(a).all(): raise ValueError(f'{name} must be finite real ndarray')

def load_model(path):
    path=Path(path).resolve(strict=True)
    with np.load(path,allow_pickle=False) as z:
        req={'metadata_json','target_mean','target_std','schema_version','method_code','num_classes','alpha'}; miss=req-set(z.files)
        if miss: raise ValueError(f'model missing fields: {sorted(miss)}')
        state={k:z[k].copy() for k in z.files if k!='metadata_json'}
        try: meta=json.loads(str(z['metadata_json'].item()))
        except Exception as e: raise ValueError('invalid metadata_json') from e
    if state['schema_version'].shape!=() or int(state['schema_version'])!=1: raise ValueError('unsupported schema')
    classes=int(state['num_classes']); method=int(state['method_code'])
    if classes<2 or method not in (0,1): raise ValueError('invalid model method/classes')
    if not np.isfinite(state['alpha']) or float(state['alpha']) not in (100.,10.,1.,.1,.01): raise ValueError('invalid fixed alpha')
    for k in ('target_mean','target_std'):
        if state[k].shape!=(BINS,): raise ValueError(f'{k} must have shape [128]')
        _finite(state[k],k)
    if np.any(state['target_std']<=0): raise ValueError('target_std must be positive')
    if method==0:
        if state['coef'].shape!=(BINS,classes) or state['intercept'].shape!=(classes,): raise ValueError('invalid linear state')
    else:
        train=state.get('train_views')
        if not isinstance(train,np.ndarray) or train.ndim!=2 or train.shape[1]!=BINS or len(train)<3 or len(train)%3: raise ValueError('invalid RBF train_views')
        for k,s in (('dual_coef',(len(train),classes)),('kernel_train_mean',(len(train),)),('kernel_grand_mean',()),('class_response_mean',(classes,))):
            if state[k].shape!=s: raise ValueError(f'invalid RBF field {k}')
    for k,v in state.items():
        if isinstance(v,np.ndarray) and v.dtype.kind in 'fc': _finite(v,k)
    cond=meta.get('condition')
    if not isinstance(cond,dict) or not isinstance(cond.get('target'),list) or not isinstance(cond.get('source'),list): raise ValueError('model metadata lacks explicit slots')
    target=cond['target']; source=cond['source']
    if target not in ([0],[1],[2],[0,1,2]) or source!=[0,1,2]: raise ValueError('unsupported model slots')
    expected_method='linear_ridge' if method==0 else 'rbf_krr'
    if meta.get('method') != expected_method or meta.get('num_classes') != classes: raise ValueError('model metadata disagrees with numeric state')
    return {'path':path,'state':state,'metadata':meta,'method':method,'classes':classes,'target_slots':target}

def spectrum(raw):
    _finite(raw,'raw')
    if raw.ndim!=3 or raw.shape[1] not in (1,3) or raw.shape[2]!=WINDOW: raise ValueError('raw must be [N,1|3,2048]')
    x=raw.astype(np.float32,copy=False); mean=x.mean(axis=-1,keepdims=True); std=x.std(axis=-1,keepdims=True)
    # Public task loading first performs per-window z-score normalization;
    # frozen ``log_power_grid`` then centers the normalized waveform once
    # more before the Hann-windowed FFT. Keep both operations here so raw
    # deployment matches the training/evaluation path exactly.
    x=(x-mean)/np.where(std>0,std,1.).astype(np.float32); x=x-x.mean(axis=-1,keepdims=True)
    w=np.hanning(WINDOW).astype(np.float32)
    f=np.fft.rfft(x*w,axis=-1)[...,1:257]; power=(f.real*f.real+f.imag*f.imag).reshape(len(x),x.shape[1],BINS,2).sum(axis=-1)
    rel=power/np.maximum(power.sum(axis=-1,keepdims=True),np.finfo(np.float32).tiny); out=np.log(rel+1e-8).astype(np.float64); _finite(out,'features'); return out

def rbf(a,b,gamma):
    d=a@b.T; d=-2*d; d+=(a*a).sum(1)[:,None]; d+=(b*b).sum(1)[None,:]; np.maximum(d,0,out=d); np.exp(-gamma*d,out=d); return d

def predict(model,values,batch_size=128):
    if not isinstance(batch_size,int) or isinstance(batch_size,bool) or batch_size<1: raise ValueError('batch_size must be positive integer')
    st=model['state']; values=np.asarray(values,dtype=np.float64)
    if values.ndim!=3 or values.shape[1] not in (1,3) or values.shape[2]!=BINS: raise ValueError('features must be [N,1|3,128]')
    _finite(values,'features'); x=(values-st['target_mean'])/st['target_std']; out=[]
    for start in range(0,len(x),batch_size):
        block=x[start:start+batch_size]; flat=block.reshape(-1,BINS)
        if model['method']==0: per=flat@st['coef']+st['intercept']
        else:
            k=rbf(flat,st['train_views'],float(st['gamma'])); k-=k.mean(1,keepdims=True); k-=st['kernel_train_mean'][None,:]; k+=st['kernel_grand_mean']; per=k@st['dual_coef']+st['class_response_mean']
        score=per.reshape(len(block),block.shape[1],model['classes']).mean(1); _finite(score,'scores'); out.append(score)
    scores=np.concatenate(out,axis=0) if out else np.empty((0,model['classes'])); return scores.argmax(1).astype(np.int64),scores

def run(model_path,input_path,input_type,target_slots,output,batch_size):
    model=load_model(model_path); target=[int(x) for x in target_slots]
    if target!=model['target_slots']: raise ValueError(f'target slots {target} do not match model {model["target_slots"]}')
    if len(target) not in (1,3): raise ValueError('target slots must be one slot or [0,1,2]')
    p=Path(input_path).resolve(strict=True); data=np.load(p,allow_pickle=False,mmap_mode='r')
    if not isinstance(data,np.ndarray) or data.ndim!=3 or data.shape[0]<1 or data.shape[1]!=len(target) or data.dtype.kind not in 'fc': raise ValueError('input must be nonempty finite floating array with width matching target slots')
    if input_type=='raw':
        if data.shape[2]!=WINDOW: raise ValueError('raw input must have 2048 samples')
        values=spectrum(np.asarray(data))
    elif input_type=='features':
        if data.shape[2]!=BINS: raise ValueError('feature input must have 128 bins')
        values=np.asarray(data,dtype=np.float64)
    else: raise ValueError('input_type must be raw or features')
    labels,scores=predict(model,values,batch_size); out=Path(output)
    if out.exists() or out.suffix.lower()!='.npz': raise ValueError('output must be a new .npz')
    out.parent.mkdir(parents=True,exist_ok=True); meta={'schema':'kernel_readout_inference_v2','model_sha256':sha(model['path']),'input_sha256':sha(p),'input_type':input_type,'input_shape':list(data.shape),'target_slots':target,'labels_read':False,'test_time_tuning':False,'batch_size':batch_size,'method':'linear_ridge' if model['method']==0 else 'rbf_krr','training_metadata':model['metadata']}
    with out.open('xb') as f: np.savez_compressed(f,predicted_class=labels,scores=scores,metadata_json=np.array(json.dumps(meta,sort_keys=True)))
    return {'predictions':len(labels),'classes':model['classes'],'output':str(out.resolve()),'labels_read':False,'input_type':input_type}

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--input',type=Path,required=True); ap.add_argument('--input-type',choices=('raw','features'),required=True); ap.add_argument('--target-slots',type=int,nargs='+',required=True); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args(); print(json.dumps(run(a.model,a.input,a.input_type,a.target_slots,a.output,a.batch_size),ensure_ascii=False))
if __name__=='__main__': main()
