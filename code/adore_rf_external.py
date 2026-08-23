#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen Daphnia RF -> ADORE external evaluation. No tuning."""

from __future__ import annotations
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from rdkit import RDLogger
import rf_baselines as rfcore

SPLIT_SEEDS = [42,123,777,2025,3407]
MODEL_SEEDS = [42,123,777,2025,3407]

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--train_table",required=True)
    p.add_argument("--external_table",required=True)
    p.add_argument("--split_root",required=True)
    p.add_argument("--out_dir",required=True)
    p.add_argument("--train_smiles_col",default="canonical_SMILES_RDKit")
    p.add_argument("--external_smiles_col",default="smiles_exact")
    p.add_argument("--train_y_col",default="Tox_original")
    p.add_argument("--external_y_col",default="target_y")
    p.add_argument("--disable_rdkit_warnings",action="store_true")
    return p.parse_args()

def summary(run_df, metric_cols):
    sm=(run_df.groupby(["model","split_seed"],as_index=False)[metric_cols]
        .mean(numeric_only=True))
    counts=(run_df.groupby(["model","split_seed"],as_index=False)[
        ["n_train_total","n_train","n_val","n_test_internal","n_external"]].first())
    sm=counts.merge(sm,on=["model","split_seed"],how="left")
    rows=[]
    for model,g in sm.groupby("model"):
        r={"model":model,"n_external":int(g["n_external"].iloc[0]),
           "n_structural_splits":int(len(g)),
           "train_n_min":int(g["n_train"].min()),"train_n_max":int(g["n_train"].max()),
           "val_n_min":int(g["n_val"].min()),"val_n_max":int(g["n_val"].max()),
           "internal_test_n_min":int(g["n_test_internal"].min()),
           "internal_test_n_max":int(g["n_test_internal"].max())}
        for m in metric_cols:
            r[f"{m}_splitmean_mean"]=float(g[m].mean())
            r[f"{m}_splitmean_std"]=float(g[m].std(ddof=1))
            r[f"{m}_splitmean_median"]=float(g[m].median())
        rows.append(r)
    return pd.DataFrame(rows).sort_values("external_RMSE_splitmean_mean"),sm

def main():
    a=parse_args()
    if a.disable_rdkit_warnings: RDLogger.DisableLog("rdApp.*")
    trp,exp,sr,out=map(lambda x:Path(x).resolve(),
                       [a.train_table,a.external_table,a.split_root,a.out_dir])
    out.mkdir(parents=True,exist_ok=True)
    trdf=pd.read_csv(trp,encoding="utf-8-sig",low_memory=False).reset_index(drop=True)
    exdf=pd.read_csv(exp,encoding="utf-8-sig",low_memory=False).reset_index(drop=True)
    if len(trdf)!=316: raise RuntimeError(f"Expected Daphnia train universe N=316, got {len(trdf)}")
    if len(exdf)!=274: raise RuntimeError(f"Expected ADORE external N=274, got {len(exdf)}")
    for c in [a.train_smiles_col,a.train_y_col]:
        if c not in trdf: raise KeyError(c)
    for c in [a.external_smiles_col,a.external_y_col]:
        if c not in exdf: raise KeyError(c)

    tr_sm=trdf[a.train_smiles_col].astype(str).tolist()
    ex_sm=exdf[a.external_smiles_col].astype(str).tolist()
    ytrall=pd.to_numeric(trdf[a.train_y_col],errors="raise").to_numpy(float)
    yext=pd.to_numeric(exdf[a.external_y_col],errors="raise").to_numpy(float)
    if not np.isfinite(ytrall).all() or not np.isfinite(yext).all():
        raise RuntimeError("Non-finite target.")

    print("[RF] Count Morgan train/external",flush=True)
    Xm_tr=rfcore.build_count_morgan(tr_sm,radius=4,n_bits=2048)
    Xm_ex=rfcore.build_count_morgan(ex_sm,radius=4,n_bits=2048)
    print("[RF] RDKit descriptor train/external",flush=True)
    Xd_tr,names,_=rfcore.build_rdkit_descriptor_raw(tr_sm)
    Xd_ex,names_ex,_=rfcore.build_rdkit_descriptor_raw(ex_sm)
    if list(names)!=list(names_ex): raise RuntimeError("Descriptor schema mismatch.")

    rfargs=SimpleNamespace(n_estimators=500,max_depth=None,min_samples_leaf=1,max_features=1.0,n_jobs=-1)
    run_rows=[]; pred_rows=[]; dqc=[]
    for ss in SPLIT_SEEDS:
        idx=rfcore.load_split_indices(sr,ss,len(trdf)); tr,va,te=idx["train"],idx["val"],idx["test"]
        kept,med,kept_names,qc=rfcore.fit_descriptor_preprocessor_train_only(Xd_tr,tr,names)
        Xdt=rfcore.transform_descriptors(Xd_tr,tr,kept,med)
        # transform external with TRAIN-only preprocessing
        Xde=np.asarray(Xd_ex[:,kept],dtype=float).copy()
        bad=~np.isfinite(Xde)
        if bad.any():
            rr,cc=np.where(bad); Xde[rr,cc]=np.asarray(med)[cc]
        if not np.isfinite(Xde).all(): raise RuntimeError("Descriptor external nonfinite remains.")
        dqc.append({"split_seed":ss,**qc,"n_external":len(exdf),
                    "kept_descriptor_names":"|".join(kept_names)})
        for ms in MODEL_SEEDS:
            for model,Xtrain,Xexternal in [
                ("Morgan-RF",Xm_tr[tr],Xm_ex),
                ("Descriptor-RF",Xdt,Xde),
            ]:
                rf=rfcore.make_rf(rfargs,ms); rf.fit(Xtrain,ytrall[tr]); pred=rf.predict(Xexternal)
                met=rfcore.regression_metrics(yext,pred)
                run_rows.append({"model":model,"split_seed":ss,"model_seed":ms,
                                 "n_train_total":len(trdf),"n_train":len(tr),"n_val":len(va),
                                 "n_test_internal":len(te),"n_external":len(exdf),
                                 **{f"external_{k}":v for k,v in met.items()}})
                for i,(yt,yp) in enumerate(zip(yext,pred)):
                    pred_rows.append({"model":model,"split_seed":ss,"model_seed":ms,
                                      "external_row_index":i,"y_true":yt,"y_pred":float(yp)})

    rdf=pd.DataFrame(run_rows)
    metrics=[c for c in rdf.columns if c.startswith("external_") and c not in {"external_row_index"}]
    # restrict actual metric fields
    metrics=[c for c in metrics if c in [
        "external_RMSE","external_MAE","external_R2","external_Pearson","external_Spearman",
        "external_bias_pred_minus_true","external_pred_to_true_sd_ratio",
        "external_calibration_slope_ytrue_on_ypred"]]
    h,sm=summary(rdf,metrics)
    rdf.to_csv(out/"01_rf_external_all_runs.csv",index=False,encoding="utf-8-sig")
    sm.to_csv(out/"02_rf_external_splitmeans.csv",index=False,encoding="utf-8-sig")
    h.to_csv(out/"03_rf_external_hierarchical_summary.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(pred_rows).to_csv(out/"04_rf_external_predictions.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(dqc).to_csv(out/"05_descriptor_trainonly_external_qc.csv",index=False,encoding="utf-8-sig")
    print(h.to_string(index=False))
    print(f"[DONE] {out}")

if __name__=="__main__": main()
