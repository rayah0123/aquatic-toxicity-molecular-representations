#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen Daphnia G2/self bundles -> ADORE external evaluation. No training."""

from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import g2_core as core

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",required=True)
    p.add_argument("--train_x_raw",required=True)
    p.add_argument("--train_rowmap_csv",required=True)
    p.add_argument("--external_x_raw",required=True)
    p.add_argument("--external_rowmap_csv",required=True)
    p.add_argument("--external_table",required=True)
    p.add_argument("--external_y_col",default="target_y")
    p.add_argument("--out_dir",required=True)
    p.add_argument("--device",default="cuda")
    return p.parse_args()

def main():
    a=parse_args(); out=Path(a.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    man=pd.read_csv(a.manifest,encoding="utf-8-sig")
    Xt=np.load(a.train_x_raw); trm=pd.read_csv(a.train_rowmap_csv,encoding="utf-8-sig")
    Xe=np.load(a.external_x_raw); exm=pd.read_csv(a.external_rowmap_csv,encoding="utf-8-sig")
    ext=pd.read_csv(a.external_table,encoding="utf-8-sig",low_memory=False).reset_index(drop=True)
    if Xt.shape!=(316,6) or len(trm)!=316: raise RuntimeError("Unexpected Daphnia G2 training asset.")
    if Xe.shape!=(274,6) or len(exm)!=274 or len(ext)!=274: raise RuntimeError("Unexpected ADORE G2 asset.")
    if core.smiles_hash(exm["SMILES"].astype(str).tolist()) != core.smiles_hash(ext["smiles_exact"].astype(str).tolist()):
        raise RuntimeError("External rowmap/table SMILES mismatch.")
    yext=pd.to_numeric(ext[a.external_y_col],errors="raise").to_numpy(float)
    train_smiles=trm["SMILES"].astype(str).tolist(); ext_smiles=exm["SMILES"].astype(str).tolist()
    all_smiles=train_smiles+ext_smiles
    fps=core.build_count_morgan(all_smiles)
    qglobal=np.arange(len(train_smiles),len(train_smiles)+len(ext_smiles),dtype=int)
    device=torch.device(a.device if str(a.device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    rows=[]; preds=[]; qcs=[]
    for r in man.itertuples(index=False):
        bp=Path(r.bundle_path).resolve()
        if core.file_sha256(bp)!=str(r.bundle_sha256): raise RuntimeError(f"Bundle hash mismatch: {bp}")
        bun=torch.load(bp,map_location="cpu",weights_only=False)
        if bun.get("schema_hash")!=core.schema_hash(): raise RuntimeError("G2 schema mismatch.")
        tr=np.asarray(bun["training"]["train_indices"],dtype=int)
        pp=bun["preprocessor"]
        Xtr=core.transform_rows(Xt,tr,pp)
        Xex=core.transform_rows(Xe,np.arange(len(Xe),dtype=int),pp)
        if r.condition=="g2":
            ei,ew,qc=core.build_inductive_edges(tr,qglobal,fps)
            mode="unweighted"
        elif r.condition=="self":
            ei=torch.empty((2,0),dtype=torch.long); ew=torch.empty((0,),dtype=torch.float32)
            qc={"n_train":len(tr),"n_query":len(Xe),"query_message_edge_n":0,
                "query_neighbor_count_min":0,"query_neighbor_count_mean":0.0,
                "query_neighbor_count_max":0,"query_zero_neighbor_n":len(Xe)}
            mode="self"
        else: raise RuntimeError(r.condition)
        model=core.ControlledGraphSAGE(input_dim=6,
            hidden_dim=int(bun["training"]["hidden_dim"]),
            num_layers=int(bun["training"]["num_layers"]),
            dropout=float(bun["training"]["dropout"]),mode=mode).to(device)
        model.load_state_dict(bun["model_state_dict"]); model.eval()
        Xall=torch.tensor(np.concatenate([Xtr,Xex],axis=0),dtype=torch.float32,device=device)
        with torch.no_grad():
            pall=model(Xall,ei.to(device),ew.to(device)).detach().cpu().numpy()
        loc=np.arange(len(tr),len(tr)+len(Xe),dtype=int)
        tf=bun["target_transform"]; pred=pall[loc]*float(tf["train_std"])+float(tf["train_mean"])
        met=core.regression_metrics(yext,pred)
        rows.append({"run_name":r.run_name,"condition":r.condition,"split_seed":int(r.split_seed),
                     "model_seed":int(r.model_seed),"n_train_total":316,"n_train":len(tr),
                     "n_val":int(r.n_val),"n_test_internal":int(r.n_test),"n_external":274,
                     **{f"external_{k}":v for k,v in met.items()}})
        qcs.append({"run_name":r.run_name,"condition":r.condition,"split_seed":int(r.split_seed),
                    "model_seed":int(r.model_seed),**qc})
        for i,(yt,yp) in enumerate(zip(yext,pred)):
            preds.append({"run_name":r.run_name,"condition":r.condition,"split_seed":int(r.split_seed),
                          "model_seed":int(r.model_seed),"external_row_index":i,
                          "y_true":float(yt),"y_pred":float(yp)})
    rdf=pd.DataFrame(rows)
    metric_cols=["external_RMSE","external_MAE","external_R2","external_Pearson","external_Spearman"]
    sm=rdf.groupby(["condition","split_seed"],as_index=False)[metric_cols].mean()
    counts=rdf.groupby(["condition","split_seed"],as_index=False)[
        ["n_train_total","n_train","n_val","n_test_internal","n_external"]].first()
    sm=counts.merge(sm,on=["condition","split_seed"],how="left")
    outrows=[]
    for cond,g in sm.groupby("condition"):
        z={"condition":cond,"n_external":274,"n_structural_splits":len(g),
           "train_n_min":int(g.n_train.min()),"train_n_max":int(g.n_train.max()),
           "val_n_min":int(g.n_val.min()),"val_n_max":int(g.n_val.max()),
           "internal_test_n_min":int(g.n_test_internal.min()),"internal_test_n_max":int(g.n_test_internal.max())}
        for m in metric_cols:
            z[f"{m}_splitmean_mean"]=float(g[m].mean())
            z[f"{m}_splitmean_std"]=float(g[m].std(ddof=1))
            z[f"{m}_splitmean_median"]=float(g[m].median())
        outrows.append(z)
    h=pd.DataFrame(outrows).sort_values("external_RMSE_splitmean_mean")
    rdf.to_csv(out/"01_g2_external_all_runs.csv",index=False,encoding="utf-8-sig")
    sm.to_csv(out/"02_g2_external_splitmeans.csv",index=False,encoding="utf-8-sig")
    h.to_csv(out/"03_g2_external_hierarchical_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(preds).to_csv(out/"04_g2_external_predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(qcs).to_csv(out/"05_g2_external_graph_qc.csv",index=False,encoding="utf-8-sig")
    print(h.to_string(index=False)); print(f"[DONE] {out}")

if __name__=="__main__": main()
