#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build unlabeled G1 graph assets. No y/toxicity values are stored."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import torch
from rdkit import RDLogger
import g1_core as core

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input_table',required=True)
    p.add_argument('--out_dir',required=True)
    p.add_argument('--smiles_col',default='canonical_SMILES_RDKit')
    p.add_argument('--disable_rdkit_warnings',action='store_true')
    return p.parse_args()

def read_table(path:Path):
    if path.suffix.lower()=='.csv':
        return pd.read_csv(path,encoding='utf-8-sig',low_memory=False)
    if path.suffix.lower() in {'.parquet','.pq'}:
        return pd.read_parquet(path)
    raise ValueError(f'Unsupported table: {path}')

def main():
    args=parse_args()
    if args.disable_rdkit_warnings: RDLogger.DisableLog('rdApp.*')
    inp=Path(args.input_table).resolve(); out=Path(args.out_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    df=read_table(inp).reset_index(drop=True)
    if args.smiles_col not in df.columns: raise KeyError(args.smiles_col)
    smiles=df[args.smiles_col].astype(str).tolist()
    outcome_like=[c for c in df.columns if any(t in c.lower() for t in ['tox','lc50','target','label'])]
    data_list=[]; qc=[]; n=len(smiles)
    checkpoints={max(1,n//4),max(1,n//2),max(1,3*n//4),n}
    for i,smi in enumerate(smiles):
        d=core.mol_to_pyg(smi,row_idx=i)
        if d is None: raise RuntimeError(f'Invalid SMILES row={i}: {smi}')
        data_list.append(d); qc.append(core.validate_single_graph(d,row_idx=i,smiles=smi))
        if i+1 in checkpoints:
            print(f'{int(round(100*(i+1)/n))}%',end=' ' if i+1<n else '\n',flush=True)
    schema=core.graph_schema_spec(); sh=core.graph_schema_hash(); smh=core.smiles_sequence_hash(smiles)
    meta={
        'asset_type':'G1_unlabeled_molecular_graphs','input_table':str(inp),'n_rows':len(df),
        'smiles_col':args.smiles_col,'smiles_sequence_sha256':smh,'graph_schema_hash':sh,
        'graph_schema_version':core.SCHEMA_VERSION,'outcome_like_columns_present_but_not_used':outcome_like,
        'contains_y':False,
    }
    asset=out/'g1_graph_assets.pt'
    torch.save({'data_list':data_list,'meta':meta,'graph_schema_spec':schema},asset)
    pd.DataFrame({'global_row_index':range(len(df)),'SMILES':smiles}).to_csv(out/'g1_graph_rowmap.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(qc).to_csv(out/'g1_graph_qc.csv',index=False,encoding='utf-8-sig')
    (out/'g1_graph_feature_spec.json').write_text(json.dumps(schema,indent=2,ensure_ascii=False),encoding='utf-8')
    meta['asset_file_sha256']=core.file_sha256(asset)
    (out/'g1_graph_asset_meta.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),encoding='utf-8')
    print(f'[DONE] {asset}'); print(f'[schema] {sh}'); print('[contains_y] False')
if __name__=='__main__': main()
