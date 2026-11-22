"""Training and evaluation functions for the fresh WebTEST 5,000-epoch
GraphSAGE comparison. Source-code provenance is recorded separately."""
from pathlib import Path
import copy, hashlib, json, random
import numpy as np
import pandas as pd
import torch
import g2_core as core
import g2_shared as shared
import webtest_assets as assets

CONDITIONS={'self':'descriptor_self','k10':'descriptor_tanimoto_K10'}
MAX_EPOCHS=5000
BUDGET500=500
METRICS=['RMSE','MAE','R2','Pearson','Spearman']

def require(ok,msg):
    if not ok: raise RuntimeError(msg)

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def save_json(p,obj):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def model_for(b,mode,device):
    return core.ControlledGraphSAGE(**b['architecture'],mode=mode).to(device)

def predict(model,b,x,edges,device):
    model.eval()
    with torch.no_grad():
        out=model(torch.as_tensor(x,dtype=torch.float32,device=device),edges[0].to(device),edges[1].to(device)).cpu().numpy()
    out=out*float(b['target_z']['std'])+float(b['target_z']['mean'])
    require(np.isfinite(out).all(),'Nonfinite predictions')
    return out

def numerical_checks(device):
    x=torch.tensor([[2.],[8.],[0.]],device=device,requires_grad=True)
    ei=torch.tensor([[0,1],[2,2]],device=device)
    w=torch.tensor([.1,.4],device=device)
    a=core.normalized_neighbor_mean(x,ei,w,True,absolute=True)
    r=core.normalized_neighbor_mean(x,ei,w,True)
    require(torch.allclose(a[2],torch.tensor([1.7],device=device)),'Absolute arithmetic failed')
    require(torch.allclose(r[2],torch.tensor([6.8],device=device)),'Normalized arithmetic failed')
    a[2].sum().backward()
    require(torch.allclose(x.grad,torch.tensor([[.05],[.2],[0.]],device=device)),'Gradient failed')
    for mode in ['weighted','absolute_weighted']:
        layer=core.ControlledSAGELayer(1,1).to(device)
        with torch.no_grad():
            layer.lin_self.weight.zero_();layer.lin_self.bias.zero_();layer.lin_neigh.weight.fill_(1.)
        require(torch.allclose(layer(x,ei,w,mode),a if mode=='absolute_weighted' else r),'Layer mode failed')
    for absolute in [False,True]:
        low=core.normalized_neighbor_mean(x,ei,torch.tensor([.2,.2],device=device),True,absolute=absolute)
        high=core.normalized_neighbor_mean(x,ei,torch.tensor([.8,.8],device=device),True,absolute=absolute)
        require(torch.allclose(high,low*(4 if absolute else 1)),'Similarity scaling failed')
        require(torch.equal(high[:2],torch.zeros_like(high[:2])),'Isolation failed')
    print('[PASS] Native aggregation, scaling, gradient and layer checks',flush=True)

def check_edges(edges,ntrain,nquery):
    ei,w=edges
    require(ei.shape[0]==2 and ei.shape[1]==len(w),'Graph shape mismatch')
    require(bool(torch.isfinite(w).all()) and bool(((w>0)&(w<=1)).all()),'Invalid Tanimoto weights')
    require(bool(((ei[0]>=0)&(ei[0]<ntrain)).all()),'Non-training message source')
    require(bool(((ei[1]>=0)&(ei[1]<ntrain+nquery)).all()),'Invalid destination')

def prepare_split(s,X,y):
    idx=assets.load_split(s,len(y))
    require(np.array_equal(np.sort(np.concatenate(list(idx.values()))),np.arange(len(y))),'Split indices not exhaustive/in range')
    tr,va=idx['train'],idx['val']
    train,val=assets.load_precomputed_edges(s,tr,va,'val')
    check_edges(train,len(tr),0);check_edges(val,len(tr),len(va))
    return idx,train,val

def validate_bundle(b,s,m,v,idx,X,y):
    require(b['split_seed']==s and b['model_seed']==m and b['condition']==CONDITIONS[v],'Baseline identity mismatch')
    require(b['relation_mode']==('self' if v=='self' else 'unweighted'),'Unexpected baseline mode')
    require(b['n_total']==len(y) and all(b['n_'+p]==len(idx[p]) for p in idx),'Bundle counts mismatch')
    require(not b.get('test_evaluated',True),'Expected original validation-selected checkpoint')
    pp=shared.fit_descriptor_preprocessor_train_only(X,idx['train'],assets.COMPACT6)
    require(pp['names']==b['preprocessor']['names']==assets.COMPACT6,'Descriptor order differs')
    for key in ['kept_cols','medians','mean','std']:
        require(np.allclose(pp[key],b['preprocessor'][key],rtol=0,atol=1e-10),'Preprocessing differs: '+key)
    yt=y[idx['train']]
    require(np.isclose(yt.mean(),b['target_z']['mean'],rtol=0,atol=1e-10) and np.isclose(yt.std(),b['target_z']['std'],rtol=0,atol=1e-10),'Target scaling differs')

def evaluate(b,mode,idx,X,y,train,val,test,device,tag):
    model=model_for(b,mode,device);model.load_state_dict(b['state_dict'])
    pp=b['preprocessor'];xt=shared.transform_descriptor_rows(X,idx['train'],pp)
    empty=(torch.empty((2,0),dtype=torch.long),torch.empty(0))
    row={**tag,'best_epoch':int(b['best_epoch']),'epochs_ran':b.get('epochs_ran'), 'history_available':bool(b.get('history_available',False))}
    row.update({'architecture_'+k:v for k,v in b['architecture'].items()})
    row.update({'training_'+k:v for k,v in b['optimizer_spec'].items()})
    row['train_curve_consistency_max_abs']=b.get('train_curve_consistency_max_abs')
    row['stop_reason']=b.get('stop_reason','not_recorded')
    row['best_validation_RMSE']=b.get('best_validation_RMSE')
    preds={}
    for part,edges in [('train',train),('val',val),('test',test)]:
        xx=xt if part=='train' else np.concatenate([xt,shared.transform_descriptor_rows(X,idx[part],pp)])
        pred=predict(model,b,xx,empty if mode=='self' else edges,device)
        if part!='train':pred=pred[len(xt):]
        preds[part]=pred
        row['n_'+part]=len(pred)
        row.update({part+'_'+k:v for k,v in core.regression_metrics(y[idx[part]],pred).items()})
    for metric in ['RMSE','MAE']:
        for part in ['val','test']:row[part+'_minus_train_'+metric]=row[part+'_'+metric]-row['train_'+metric]
    return row,preds

def train_one(ref,mode,idx,X,y,train,val,device,run,signature):
    from selection import update,reason
    require(int(ref['optimizer_spec']['epochs_max'])==BUDGET500,'Expected original 500-epoch budget')
    assets.seed_everything(ref['model_seed'])
    b=copy.deepcopy(ref);b['relation_mode']=mode;b['condition']=mode;b['history_available']=True
    b['optimizer_spec']={**ref['optimizer_spec'],'epochs_max':MAX_EPOCHS}
    b['scientific_spec']={**b['scientific_spec'],'message_weighting':mode}
    pp=b['preprocessor'];tr=idx['train'];va=idx['val']
    xt=shared.transform_descriptor_rows(X,tr,pp);xv=np.concatenate([xt,shared.transform_descriptor_rows(X,va,pp)])
    xt=torch.tensor(xt,dtype=torch.float32,device=device);xv=torch.tensor(xv,dtype=torch.float32,device=device)
    yt=torch.tensor(((y[tr]-b['target_z']['mean'])/b['target_z']['std']).astype(np.float32),device=device)
    if mode=='self':
        train=(torch.empty((2,0),dtype=torch.long),torch.empty(0));val=train
    te=tuple(e.to(device) for e in train);ve=tuple(e.to(device) for e in val)
    model=model_for(b,mode,device);sp=b['optimizer_spec']
    opt=torch.optim.Adam(model.parameters(),lr=sp['lr'],weight_decay=sp['weight_decay'])
    best=float('inf');state=None;state500=None;best500=float('inf');epoch500=None;best_epoch=-1;wait=0;history=[];last_epoch=0
    resume=run/'resume.pt'
    if resume.is_file():
        r=load(resume);require(r['signature']==signature and r['mode']==mode,'Resume signature mismatch')
        model.load_state_dict(r['current_model']);opt.load_state_dict(r['optimizer'])
        best=r['best'];state=r['best_state'];best_epoch=r['best_epoch'];wait=r['wait'];history=r['history'];last_epoch=r['epoch']
        state500=r['best500_state'];best500=r['best500'];epoch500=r['epoch500']
        random.setstate(r['python_rng']);np.random.set_state(r['numpy_rng']);torch.set_rng_state(r['torch_rng'])
        if device.type=='cuda':torch.cuda.set_rng_state_all(r['cuda_rng'])
        print(f'[RESUME] {run.name} after epoch {last_epoch}',flush=True)
    def checkpoint(epoch):
        r=dict(signature=signature,mode=mode,epoch=epoch,current_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),
            best=best,best_state=state,best_epoch=best_epoch,wait=wait,history=history,best500_state=state500,best500=best500,epoch500=epoch500,
            python_rng=random.getstate(),numpy_rng=np.random.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None)
        tmp=run/'resume.tmp';torch.save(r,tmp);tmp.replace(resume)
        pd.DataFrame(history).to_csv(run/'history.csv',index=False)
    epoch=last_epoch
    if reason(last_epoch,MAX_EPOCHS,wait,int(sp['patience']))=='running':
        for epoch in range(last_epoch+1,MAX_EPOCHS+1):
            model.train();opt.zero_grad();p=model(xt,*te);loss=torch.nn.functional.mse_loss(p,yt);loss.backward();opt.step()
            model.eval()
            with torch.no_grad():pv=model(xv,*ve).cpu().numpy()*b['target_z']['std']+b['target_z']['mean']
            require(np.isfinite(pv).all() and np.isfinite(loss.item()),'Nonfinite training')
            ertr=pv[:len(tr)]-y[tr];erva=pv[len(tr):]-y[va];rmse=float(np.sqrt(np.mean(erva**2)))
            history.append(dict(epoch=epoch,optimization_loss_z=float(loss.item()),train_RMSE=float(np.sqrt(np.mean(ertr**2))),train_MAE=float(np.mean(abs(ertr))),val_RMSE=rmse,val_MAE=float(np.mean(abs(erva)))))
            best,wait,improved=update(best,wait,rmse)
            if improved:
                best_epoch=epoch;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch<=BUDGET500:
                # Same chosen best within the first 500 epochs, not the epoch-500 weights.
                state500=state;best500=best;epoch500=best_epoch
            status=reason(epoch,MAX_EPOCHS,wait,int(sp['patience']))
            if epoch%10==0 or epoch==1 or status!='running':
                print(f'  epoch={epoch}/{MAX_EPOCHS} val={rmse:.6f} best={best:.6f} best_epoch={best_epoch} wait={wait}',flush=True)
            if epoch%10==0 or status!='running':checkpoint(epoch)
            if status!='running':break
    require(state is not None and state500 is not None,'No selected checkpoint')
    status=reason(epoch,MAX_EPOCHS,wait,int(sp['patience']))
    b['state_dict']=state;b['best_epoch']=best_epoch;b['epochs_ran']=epoch;b['test_evaluated']=False;b['stop_reason']=status;b['best_validation_RMSE']=best
    b500=copy.deepcopy(b);b500['state_dict']=state500;b500['best_epoch']=epoch500;b500['epochs_ran']=min(epoch,BUDGET500);b500['best_validation_RMSE']=best500
    b500['optimizer_spec']={**b['optimizer_spec'],'epochs_max':BUDGET500}
    b500['stop_reason']=status if epoch<=BUDGET500 else 'max_epochs'
    pd.DataFrame(history).to_csv(run/'history.csv',index=False)
    for selected in [b,b500]:
        model.load_state_dict(selected['state_dict'])
        a=predict(model,selected,xt,te,device);c=predict(model,selected,xv,ve,device)[:len(tr)]
        require(np.allclose(a,c,rtol=0,atol=1e-4),'TRAIN predictions affected by query graph')
        selected['train_curve_consistency_max_abs']=float(np.max(abs(a-c)))
    return b,b500

def summarize(df,out):
    cols=[p+'_'+m for p in ['train','val','test'] for m in METRICS]+[p+'_minus_train_'+m for p in ['val','test'] for m in ['RMSE','MAE']]
    sm=df.groupby(['variant','split_seed'],as_index=False)[cols].mean();sm.to_csv(out/'split_mean_metrics.csv',index=False)
    rows=[]
    for v,g in sm.groupby('variant'):
        row=dict(variant=v,n_splits=len(g))
        for c in cols:row[c+'_mean']=g[c].mean();row[c+'_sd']=g[c].std(ddof=1)
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'hierarchical_summary.csv',index=False)
    base=sm[sm.variant.eq('self')].drop(columns='variant')
    d=sm.merge(base,on='split_seed',suffixes=('','_self'),validate='many_to_one')
    for c in cols:d['delta_'+c]=d[c]-d[c+'_self']
    d.to_csv(out/'split_differences_vs_self.csv',index=False)
    sd=df.groupby(['variant','split_seed'])[cols].std(ddof=1).add_suffix('_modelseed_sd').reset_index()
    sd.to_csv(out/'within_split_modelseed_variability.csv',index=False)
