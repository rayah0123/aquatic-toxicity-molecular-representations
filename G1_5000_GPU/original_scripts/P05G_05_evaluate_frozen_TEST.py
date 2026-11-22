#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate frozen G1 bundles on each split's TEST partition. No training occurs."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch_geometric.loader import DataLoader
import P05G_00_g1_core as core

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--graph_asset',required=True); p.add_argument('--y_csv',required=True); p.add_argument('--split_root',required=True); p.add_argument('--out_dir',required=True); p.add_argument('--eval_batch_size',type=int,default=2048); p.add_argument('--device',default='cuda'); return p.parse_args()

def test_idx(path):
    m=pd.read_csv(path,encoding='utf-8-sig'); ic=next((c for c in ['original_row_index','row_idx','index','global_row_index'] if c in m.columns),None)
    if ic is None: raise KeyError(path)
    return np.sort(pd.to_numeric(m.loc[m['split'].astype(str).str.lower().eq('test'),ic],errors='raise').astype(int).to_numpy())

def met(y,p):
    return {'RMSE':float(np.sqrt(mean_squared_error(y,p))),'MAE':float(mean_absolute_error(y,p)),'R2':float(r2_score(y,p)),'Pearson':float(pd.Series(y).corr(pd.Series(p),method='pearson')),'Spearman':float(pd.Series(y).corr(pd.Series(p),method='spearman'))}
@torch.no_grad()
def predict(model,graphs,device,batch_size,mean,std):
    loader=DataLoader(graphs,batch_size=batch_size,shuffle=False); rows=[]; preds=[]; model.eval()
    for b in loader:
        b=b.to(device); rows.append(b.row_idx.view(-1).cpu().numpy()); preds.append(model(b).cpu().numpy()*std+mean)
    return np.concatenate(rows),np.concatenate(preds)

def main():
    a=parse_args(); out=Path(a.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True); man=pd.read_csv(a.manifest,encoding='utf-8-sig')
    obj=torch.load(a.graph_asset,map_location='cpu',weights_only=False); data=obj['data_list']; gm=obj['meta']; yd=pd.read_csv(a.y_csv,encoding='utf-8-sig'); y=pd.to_numeric(yd['y'],errors='raise').to_numpy(float)
    device=torch.device(a.device if str(a.device).startswith('cuda') and torch.cuda.is_available() else 'cpu'); rr=[]; pp=[]
    for r in man.itertuples(index=False):
        bp=Path(r.bundle_path).resolve()
        if core.file_sha256(bp)!=str(r.bundle_sha256): raise RuntimeError(f'Bundle SHA mismatch: {bp}')
        bun=torch.load(bp,map_location='cpu',weights_only=False); core.assert_bundle_compatible(bun,gm); ms=bun['model_spec']
        model=core.G1GraphSAGE(int(ms['in_dim']),int(ms['hidden_dim']),int(ms['num_layers']),float(ms['dropout'])).to(device); model.load_state_dict(bun['model_state_dict'])
        te=test_idx(Path(a.split_root).resolve()/f'split_seed{int(r.split_seed)}'/'split_membership.csv'); tf=bun['target_transform']; rows,p=predict(model,[data[int(i)] for i in te],device,a.eval_batch_size,float(tf['train_mean']),float(tf['train_std']))
        if not np.array_equal(rows,te): raise RuntimeError('TEST row order mismatch')
        mm=met(y[te],p); rr.append({'run_name':r.run_name,'split_seed':int(r.split_seed),'model_seed':int(r.model_seed),'n_test':len(te),**{f'test_{k}':v for k,v in mm.items()}})
        for i,pred in zip(te,p): pp.append({'run_name':r.run_name,'split_seed':int(r.split_seed),'model_seed':int(r.model_seed),'global_row_index':int(i),'y_true':float(y[i]),'y_pred':float(pred)})
    rdf=pd.DataFrame(rr); rdf.to_csv(out/'g1_frozen_test_all_runs.csv',index=False,encoding='utf-8-sig'); pd.DataFrame(pp).to_csv(out/'g1_frozen_test_predictions.csv',index=False,encoding='utf-8-sig')
    mcols=[c for c in rdf.columns if c.startswith('test_')]; sm=rdf.groupby('split_seed',as_index=False)[mcols].mean(numeric_only=True); sm.to_csv(out/'g1_frozen_test_splitmeans.csv',index=False,encoding='utf-8-sig')
    rows=[]
    for c in mcols:
        v=sm[c].dropna(); rows.append({'metric':c,'n_structural_splits':len(v),'splitmean_mean':float(v.mean()) if len(v) else None,'splitmean_std':float(v.std(ddof=1)) if len(v)>1 else None,'splitmean_median':float(v.median()) if len(v) else None})
    pd.DataFrame(rows).to_csv(out/'g1_frozen_test_hierarchical_summary.csv',index=False,encoding='utf-8-sig'); print(f'[DONE] {out}'); print('[NOTE] Frozen TEST evaluated; no training occurred.')
if __name__=='__main__': main()
