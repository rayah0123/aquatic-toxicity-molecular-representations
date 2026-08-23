#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score external G1 predictions against the ADORE target."""

from __future__ import annotations
import argparse,re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def metrics(y,p):
    return {"RMSE":float(np.sqrt(mean_squared_error(y,p))),
            "MAE":float(mean_absolute_error(y,p)),
            "R2":float(r2_score(y,p)),
            "Pearson":float(pd.Series(y).corr(pd.Series(p),method="pearson")),
            "Spearman":float(pd.Series(y).corr(pd.Series(p),method="spearman"))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--pred_csv",required=True)
    ap.add_argument("--external_table",required=True); ap.add_argument("--out_dir",required=True)
    a=ap.parse_args(); out=Path(a.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(a.pred_csv,encoding="utf-8-sig"); e=pd.read_csv(a.external_table,encoding="utf-8-sig",low_memory=False)
    if len(p)!=274 or len(e)!=274: raise RuntimeError("Expected external N=274.")
    y=pd.to_numeric(e["target_y"],errors="raise").to_numpy(float)
    rows=[]
    rx=re.compile(r"^g1_pred_split(\d+)_seed(\d+)$")
    for c in p.columns:
        m=rx.match(c)
        if not m: continue
        ss,ms=map(int,m.groups()); pr=pd.to_numeric(p[c],errors="raise").to_numpy(float)
        rows.append({"split_seed":ss,"model_seed":ms,"n_external":274,
                     **{f"external_{k}":v for k,v in metrics(y,pr).items()}})
    rdf=pd.DataFrame(rows)
    if len(rdf)!=25: raise RuntimeError(f"Expected 25 G1 predictions, got {len(rdf)}")
    mc=["external_RMSE","external_MAE","external_R2","external_Pearson","external_Spearman"]
    sm=rdf.groupby("split_seed",as_index=False)[mc].mean()
    z={"model":"G1","n_external":274,"n_structural_splits":len(sm)}
    for m in mc:
        z[f"{m}_splitmean_mean"]=float(sm[m].mean()); z[f"{m}_splitmean_std"]=float(sm[m].std(ddof=1))
        z[f"{m}_splitmean_median"]=float(sm[m].median())
    rdf.to_csv(out/"01_g1_external_all_runs.csv",index=False,encoding="utf-8-sig")
    sm.to_csv(out/"02_g1_external_splitmeans.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([z]).to_csv(out/"03_g1_external_hierarchical_summary.csv",index=False,encoding="utf-8-sig")
    print(pd.DataFrame([z]).to_string(index=False)); print(f"[DONE] {out}")

if __name__=="__main__": main()
