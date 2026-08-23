#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run frozen Daphnia models on chemically non-overlapping ADORE N=274."""

from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np,pandas as pd

def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()

def run(x):
    print("\n[RUN]"," ".join(map(str,x)),flush=True); subprocess.run(x,check=True)

def main():
    sd=Path(__file__).resolve().parent; root=sd.parent
    train=root/"03_processed/TEST/TEST_Daphnia_modeling_pool_no_ADORE_overlap.csv"
    ext=root/"03_processed/ADORE/ADORE_Daphnia_48h_LC50_external_chemical_level.csv"
    splits=root/"04_splits/TEST_Daphnia/scaffold_split_murcko_topology_refined"
    g1man=root/"07_results/Daphnia/g1_validation/g1_model_manifest.csv"
    g2man=root/"07_results/Daphnia/g2_validation/g2_model_manifest.csv"
    g1train=root/"05_features/G1_Daphnia"
    g2train=root/"05_features/G2_Daphnia"
    extfeat=root/"05_features/ADORE_Daphnia_external"
    g1ext=extfeat/"G1"; g2ext=extfeat/"G2"
    res=root/"07_results/Daphnia/adore_external"
    for p in [g1ext,g2ext,res]: p.mkdir(parents=True,exist_ok=True)
    req=[train,ext,g1man,g2man,g1train/"g1_y_experimental.csv",
         g2train/"g2_compact6_raw.npy",g2train/"g2_rowmap.csv"]
    for p in req:
        if not p.exists(): raise FileNotFoundError(p)
    td=pd.read_csv(train,encoding="utf-8-sig",low_memory=False)
    ed=pd.read_csv(ext,encoding="utf-8-sig",low_memory=False)
    if len(td)!=316 or len(ed)!=274: raise RuntimeError(f"Unexpected N train={len(td)} ext={len(ed)}")
    if ed["smiles_connectivity"].nunique()!=274: raise RuntimeError("ADORE connectivity is not unique.")
    # verify no cross-dataset connectivity overlap
    tconn=set(td["smiles_connectivity"].dropna().astype(str))
    econn=set(ed["smiles_connectivity"].dropna().astype(str))
    ov=tconn&econn
    if ov: raise RuntimeError(f"Connectivity leakage into external set: {len(ov)}")
    freeze={"stage":"Daphnia ADORE external pre-inference","n_train_universe":316,"n_external":274,
            "connectivity_overlap":0,"tuning_after_external":False,
            "files":{str(p.relative_to(root)):sha(p) for p in req}}
    (res/"00_pre_external_freeze.json").write_text(json.dumps(freeze,indent=2),encoding="utf-8")

    # external G1/G2 raw assets
    run([sys.executable,str(sd/"g1_build_graph_assets.py"),
         "--input_table",str(ext),"--out_dir",str(g1ext),"--smiles_col","smiles_exact",
         "--disable_rdkit_warnings"])
    run([sys.executable,str(sd/"g2_build_assets.py"),
         "--input_table",str(ext),"--out_dir",str(g2ext),"--smiles_col","smiles_exact",
         "--disable_rdkit_warnings"])

    # RF external
    run([sys.executable,str(sd/"adore_rf_external.py"),
         "--train_table",str(train),"--external_table",str(ext),"--split_root",str(splits),
         "--out_dir",str(res/"RF"),"--disable_rdkit_warnings"])

    # G1 external prediction from frozen 25 bundles + metrics
    run([sys.executable,str(sd/"g1_external_inference.py"),
         "--manifest",str(g1man),"--graph_asset",str(g1ext/"g1_graph_assets.pt"),
         "--rowmap_csv",str(g1ext/"g1_graph_rowmap.csv"),"--out_dir",str(res/"G1"),"--device","cuda"])
    run([sys.executable,str(sd/"adore_g1_external.py"),
         "--pred_csv",str(res/"G1/g1_external_predictions_all_models_and_aggregate.csv"),
         "--external_table",str(ext),"--out_dir",str(res/"G1")])

    # G2/self strict-inductive external
    run([sys.executable,str(sd/"adore_g2_external.py"),
         "--manifest",str(g2man),"--train_x_raw",str(g2train/"g2_compact6_raw.npy"),
         "--train_rowmap_csv",str(g2train/"g2_rowmap.csv"),
         "--external_x_raw",str(g2ext/"g2_compact6_raw.npy"),
         "--external_rowmap_csv",str(g2ext/"g2_rowmap.csv"),
         "--external_table",str(ext),"--out_dir",str(res/"G2"),"--device","cuda"])

    print("\n[ADORE EXTERNAL EVALUATION COMPLETE]")
    print(res/"RF/03_rf_external_hierarchical_summary.csv")
    print(res/"G1/03_g1_external_hierarchical_summary.csv")
    print(res/"G2/03_g2_external_hierarchical_summary.csv")
    print("[STATUS] External set evaluated. Do not tune from these results.")

if __name__=="__main__": main()
