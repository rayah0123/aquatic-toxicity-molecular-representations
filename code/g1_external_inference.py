#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""External inference from an explicit model manifest with schema/SHA checks."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
import g1_core as core

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--graph_asset',required=True); p.add_argument('--rowmap_csv',required=True); p.add_argument('--out_dir',required=True); p.add_argument('--eval_batch_size',type=int,default=2048); p.add_argument('--device',default='cuda'); return p.parse_args()
@torch.no_grad()
def pred(model,data,device,batch_size,mean,std):
    loader=DataLoader(data,batch_size=batch_size,shuffle=False); rr=[]; pp=[]; model.eval()
    for b in loader:
        b=b.to(device); rr.append(b.row_idx.view(-1).cpu().numpy()); pp.append(model(b).cpu().numpy()*std+mean)
    return np.concatenate(rr),np.concatenate(pp)

def main():
    a=parse_args(); outd=Path(a.out_dir).resolve(); outd.mkdir(parents=True,exist_ok=True); man=pd.read_csv(a.manifest,encoding='utf-8-sig'); obj=torch.load(a.graph_asset,map_location='cpu',weights_only=False); data=obj['data_list']; gm=obj['meta']; rowmap=pd.read_csv(a.rowmap_csv,encoding='utf-8-sig')
    exp=np.arange(len(data)); got=pd.to_numeric(rowmap['global_row_index'],errors='raise').astype(int).to_numpy()
    if len(rowmap)!=len(data) or not np.array_equal(exp,got): raise RuntimeError('rowmap/graph order mismatch')
    device=torch.device(a.device if str(a.device).startswith('cuda') and torch.cuda.is_available() else 'cpu'); out=rowmap.copy(); cols=[]
    for r in man.itertuples(index=False):
        bp=Path(r.bundle_path).resolve()
        if core.file_sha256(bp)!=str(r.bundle_sha256): raise RuntimeError(f'Bundle SHA mismatch: {bp}')
        bun=torch.load(bp,map_location='cpu',weights_only=False); core.assert_bundle_compatible(bun,gm); ms=bun['model_spec']; model=core.G1GraphSAGE(int(ms['in_dim']),int(ms['hidden_dim']),int(ms['num_layers']),float(ms['dropout'])).to(device); model.load_state_dict(bun['model_state_dict']); tf=bun['target_transform']; rows,p=pred(model,data,device,a.eval_batch_size,float(tf['train_mean']),float(tf['train_std']))
        if not np.array_equal(rows,exp): raise RuntimeError(f'Inference row mismatch: {bp}')
        c=f'g1_pred_split{int(r.split_seed)}_seed{int(r.model_seed)}'; out[c]=p; cols.append(c)
    arr=out[cols].to_numpy(float); out['g1_pred_mean']=arr.mean(1); out['g1_pred_median']=np.median(arr,axis=1); out['g1_pred_std']=arr.std(1); out['g1_pred_min']=arr.min(1); out['g1_pred_max']=arr.max(1); out['g1_n_models']=arr.shape[1]
    op=outd/'g1_external_predictions_all_models_and_aggregate.csv'; out.to_csv(op,index=False,encoding='utf-8-sig'); man[['run_name','split_seed','model_seed','bundle_path','bundle_sha256','graph_schema_hash']].to_csv(outd/'g1_external_inference_model_manifest_used.csv',index=False,encoding='utf-8-sig'); print(f'[DONE] {op}'); print(f'[models] {len(cols)}'); print(f'[schema] {core.graph_schema_hash()}')
if __name__=='__main__': main()
