#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figure_s1.py

Create a supplementary ECDF of T10 similarity from the structural-neighborhood characterization
query-level structural-neighborhood characterization.

Scientific unit shown
---------------------
Each ECDF uses query-split T10 observations from the structural-neighborhood characterization output.
No model fitting, graph modification, threshold optimization, toxicity label,
or prediction is used.

Expected input
--------------
01_per_query_Top10_similarity.csv
produced by characterize_top10_similarity.py

Run from anywhere inside the project, or specify --input explicitly.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

INPUT_NAME='01_per_query_Top10_similarity.csv'
ORDER=['Daphnia_TEST','ADORE_external','Fathead_TEST','WebTEST_surrogate']
LABELS={
    'Daphnia_TEST': r'$D.$ magna TEST',
    'ADORE_external':'ADORE external validation',
    'Fathead_TEST':'Fathead minnow TEST',
    'WebTEST_surrogate':'WebTEST surrogate',
}
LINESTYLES=['-','--','-.',':']

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,default=None)
    p.add_argument('--outdir',type=Path,default=None)
    p.add_argument('--dpi',type=int,default=600)
    return p.parse_args()

def find_input(explicit):
    if explicit is not None:
        q=explicit.expanduser().resolve()
        if not q.exists(): raise FileNotFoundError(q)
        return q
    here=Path(__file__).resolve().parent
    cwd=Path.cwd().resolve()
    roots=[]
    for r in [cwd,here,here.parent,cwd.parent]:
        if r.exists() and r not in roots: roots.append(r)
    rel=Path('07_results')/'diagnostics'/'top10_similarity_characterization'/INPUT_NAME
    for r in roots:
        q=r/rel
        if q.exists(): return q.resolve()
        q=r/INPUT_NAME
        if q.exists(): return q.resolve()
    found=[]
    for r in roots:
        try: found.extend(r.rglob(INPUT_NAME))
        except (PermissionError,OSError): pass
    found=list(dict.fromkeys(x.resolve() for x in found))
    if len(found)==1: return found[0]
    if not found:
        raise FileNotFoundError(
            f'Could not find {INPUT_NAME}. Run characterize_top10_similarity.py first or provide --input with the exact query-level CSV.'
        )
    raise RuntimeError('Multiple inputs found; use --inpu<local-path>'+'\n'.join(map(str,found)))

def main():
    args=parse_args(); inp=find_input(args.input)
    outdir=args.outdir.expanduser().resolve() if args.outdir else inp.parent/'supplementary_figure_T10_ECDF'
    outdir.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(inp,encoding='utf-8-sig',low_memory=False)
    req={'context','split_seed','n_neighbors','T10'}
    missing=req-set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if set(df['context'].unique())!=set(ORDER):
        raise ValueError(f'Unexpected contexts: {sorted(df.context.unique())}')
    for ctx in ORDER:
        sub=df[df.context.eq(ctx)]
        if sub['split_seed'].nunique()!=5: raise ValueError(f'{ctx}: expected 5 splits')
        if not sub['n_neighbors'].eq(10).all(): raise ValueError(f'{ctx}: not all query observations have K=10')
        if sub['T10'].isna().any(): raise ValueError(f'{ctx}: missing T10')

    mpl.rcParams.update({
        'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],
        'font.size':7.0,'axes.labelsize':7.2,'xtick.labelsize':6.5,'ytick.labelsize':6.5,
        'axes.linewidth':0.6,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'
    })
    fig,ax=plt.subplots(figsize=(140/25.4,88/25.4))
    for ctx,ls in zip(ORDER,LINESTYLES):
        x=np.sort(df.loc[df.context.eq(ctx),'T10'].to_numpy(float))
        y=np.arange(1,len(x)+1,dtype=float)/len(x)
        ax.step(x,y,where='post',linewidth=1.15,linestyle=ls,label=LABELS[ctx])
    ax.set_xlabel('10th-nearest-neighbor Tanimoto similarity (T10)')
    ax.set_ylabel('Empirical cumulative fraction')
    ax.set_xlim(left=0)
    ax.set_ylim(0,1.01)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False,fontsize=6.4,loc='lower right')
    fig.tight_layout(pad=0.8)
    for ext in ['png','pdf','svg']:
        fig.savefig(outdir/f'Figure_Sx_T10_similarity_ECDF.{ext}',dpi=args.dpi if ext=='png' else None,bbox_inches='tight',facecolor='white')
    plt.close(fig)

    cap=(
        'Figure Sx. Distribution of structural similarity to the 10th nearest neighbor across aquatic-toxicity datasets. '
        'Empirical cumulative distribution functions summarize query-split T10 observations from five structural splits for each dataset. '
        'T10 is the exact Count Morgan (radius 4, 2048 dimensions) Tanimoto similarity between a query chemical and its tenth selected training neighbor. '
        'All query observations had ten positive-similarity training neighbors. The distributions are descriptive structural characterizations and were not used to retune K, define a similarity threshold, or weight G2 message passing.'
    )
    (outdir/'Figure_Sx_caption.txt').write_text(cap,encoding='utf-8')
    print(f'Input: {inp}')
    print(f'Output: {outdir}')
    print('QC: 4 datasets x 5 structural splits; all query observations full K=10; PASS')

if __name__=='__main__': main()
