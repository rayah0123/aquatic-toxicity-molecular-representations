#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run 5 structural splits x model seeds and write an explicit model manifest."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import pandas as pd

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--trainer',default='g1_train_validation.py'); p.add_argument('--graph_asset',required=True); p.add_argument('--y_csv',required=True)
    p.add_argument('--split_root',required=True); p.add_argument('--out_dir',required=True); p.add_argument('--split_seeds',nargs='+',type=int,default=[42,123,777,2025,3407]); p.add_argument('--model_seeds',nargs='+',type=int,default=[42,123,777,2025,3407])
    p.add_argument('--hidden_dim',type=int,default=128); p.add_argument('--num_layers',type=int,default=2); p.add_argument('--dropout',type=float,default=.30); p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--weight_decay',type=float,default=1e-4)
    p.add_argument('--epochs',type=int,default=500); p.add_argument('--patience',type=int,default=50); p.add_argument('--train_batch_size',type=int,default=1024); p.add_argument('--eval_batch_size',type=int,default=2048); p.add_argument('--device',default='cuda'); p.add_argument('--skip_existing',action='store_true'); return p.parse_args()

def main():
    a=parse_args(); out=Path(a.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True); trainer=Path(a.trainer); trainer=trainer if trainer.is_absolute() else (Path(__file__).resolve().parent/trainer).resolve()
    rows=[]; total=len(a.split_seeds)*len(a.model_seeds); done=0
    for ss in a.split_seeds:
        sp=Path(a.split_root).resolve()/f'split_seed{ss}'/'split_membership.csv'
        if not sp.exists(): raise FileNotFoundError(sp)
        for ms in a.model_seeds:
            rn=f'g1_graphsage_split{ss}_seed{ms}'; rd=out/rn; rj=rd/'result.json'
            if a.skip_existing and rj.exists(): res=json.loads(rj.read_text(encoding='utf-8'))
            else:
                rd.mkdir(parents=True,exist_ok=True)
                cmd=[sys.executable,str(trainer),'--graph_asset',str(Path(a.graph_asset).resolve()),'--y_csv',str(Path(a.y_csv).resolve()),'--split_membership',str(sp),'--out_dir',str(rd),'--run_name',rn,'--split_seed',str(ss),'--model_seed',str(ms),'--hidden_dim',str(a.hidden_dim),'--num_layers',str(a.num_layers),'--dropout',str(a.dropout),'--lr',str(a.lr),'--weight_decay',str(a.weight_decay),'--epochs',str(a.epochs),'--patience',str(a.patience),'--train_batch_size',str(a.train_batch_size),'--eval_batch_size',str(a.eval_batch_size),'--device',a.device]
                subprocess.run(cmd,check=True); res=json.loads(rj.read_text(encoding='utf-8'))
            rows.append(res); done+=1; print(f'{int(round(100*done/total))}%',end=' ' if done<total else '\n',flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/'g1_validation_all_runs.csv',index=False,encoding='utf-8-sig')
    metrics=[c for c in df.columns if c.startswith('val_') and pd.api.types.is_numeric_dtype(df[c])]
    sm=df.groupby('split_seed',as_index=False)[metrics].mean(numeric_only=True); sm.to_csv(out/'g1_validation_splitmeans.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for c in metrics:
        v=sm[c].dropna(); summary.append({'metric':c,'n_structural_splits':len(v),'splitmean_mean':float(v.mean()) if len(v) else None,'splitmean_std':float(v.std(ddof=1)) if len(v)>1 else None,'splitmean_median':float(v.median()) if len(v) else None})
    pd.DataFrame(summary).to_csv(out/'g1_validation_hierarchical_summary.csv',index=False,encoding='utf-8-sig')
    man=df[['run_name','split_seed','model_seed','bundle_path','bundle_sha256','graph_schema_hash','best_epoch','val_RMSE','val_MAE','val_R2','val_Pearson','val_Spearman','test_evaluated']].copy(); man['bundle_path']=man['bundle_path'].map(lambda x:str(Path(x).resolve())); man.to_csv(out/'g1_model_manifest.csv',index=False,encoding='utf-8-sig')
    print(f"[DONE] {out/'g1_model_manifest.csv'}"); print('[TEST evaluated] NO')
if __name__=='__main__': main()
