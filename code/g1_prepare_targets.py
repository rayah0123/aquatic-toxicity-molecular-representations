#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create a target-only experimental y asset. No graph construction occurs here."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import g1_core as core

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--input_table',required=True); p.add_argument('--out_dir',required=True)
    p.add_argument('--smiles_col',default='canonical_SMILES_RDKit'); p.add_argument('--y_col',default='Tox_original'); return p.parse_args()

def main():
    args=parse_args(); inp=Path(args.input_table).resolve(); out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(inp,encoding='utf-8-sig',low_memory=False).reset_index(drop=True)
    for c in [args.smiles_col,args.y_col]:
        if c not in df.columns: raise KeyError(c)
    y=pd.to_numeric(df[args.y_col],errors='raise').to_numpy(dtype=float)
    if not np.isfinite(y).all(): raise ValueError('Non-finite y values found')
    smiles=df[args.smiles_col].astype(str).tolist()
    yd=pd.DataFrame({'global_row_index':np.arange(len(df),dtype=int),'SMILES':smiles,'y':y})
    yp=out/'g1_y_experimental.csv'; yd.to_csv(yp,index=False,encoding='utf-8-sig')
    meta={'asset_type':'G1_experimental_target','input_table':str(inp),'n_rows':len(yd),'source_y_col':args.y_col,
          'smiles_sequence_sha256':core.smiles_sequence_hash(smiles),'target_file_sha256':core.file_sha256(yp)}
    (out/'g1_y_experimental_meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'[DONE] {yp}')
if __name__=='__main__': main()
