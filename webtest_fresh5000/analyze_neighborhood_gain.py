#!/usr/bin/env python3
"""Recalculate the descriptive chemical-level associations for Supplementary Table S9.

Use the final held-out predictions and the per-query Top-10 neighborhood table.
Positive gain = lower absolute error for unweighted G2. Average errors over
five model seeds within each chemical and split; never pool splits as independent.
"""
from pathlib import Path
import argparse, gzip, io, zipfile
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SEEDS = (42,123,777,2025,3407)
METRICS = ('T1','T10','mean_Top10','median_Top10','spread_T1_T10','n_ge_0p3')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,required=True,help='G2 final result ZIP with test_predictions/ entries')
    p.add_argument('--neighbors',type=Path,required=True,help='P15 01_per_query_Top10_similarity.csv')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args(); args.out.mkdir(parents=True,exist_ok=False)
    nb=pd.read_csv(args.neighbors)
    print('Neighborhood contexts:', sorted(nb.context.unique()))
    # Select only WebTEST query-neighborhood records for this analysis.
    nb=nb[nb.context.astype(str).str.lower().str.contains('webtest')].copy()
    if len(nb)!=89450: raise ValueError('Unexpected WebTEST split-query count')
    rows=[]
    with zipfile.ZipFile(args.predictions) as z:
        for s in SEEDS:
            frames=[]; target=None
            for m in SEEDS:
                def get(v):
                    n=f'test_predictions/budget5000_{v}_split{s}_seed{m}.csv.gz'
                    return pd.read_csv(io.BytesIO(gzip.decompress(z.read(n)))).sort_values('row_index').reset_index(drop=True)
                a,b=get('self'),get('k10')
                assert a.row_index.is_unique and a.row_index.equals(b.row_index) and np.array_equal(a.y_true,b.y_true)
                if target is None:target=a[['row_index','y_true']].copy()
                else:assert target.row_index.equals(a.row_index) and np.array_equal(target.y_true,a.y_true)
                e0=(a.y_true-a.y_pred).to_numpy();e1=(b.y_true-b.y_pred).to_numpy()
                frames.append(np.stack((np.abs(e0)-np.abs(e1), e0**2-e1**2),axis=1))
            gains=np.stack(frames).mean(axis=0);target[['gain_ae','gain_se']]=gains
            q=nb[nb.split_seed==s].rename(columns={'query_row_index':'row_index'})
            j=target.merge(q,on='row_index',validate='one_to_one');assert len(j)==len(target)==len(q)
            for metric in METRICS:
                for typ in ['gain_ae','gain_se']:
                    rho,pval=spearmanr(j[metric],j[typ]);rows.append(dict(split_seed=s,neighborhood_metric=metric,gain_type=typ,spearman_rho=rho,p_descriptive=pval,n_test=len(j),fraction_g2_better=(j.gain_ae>0).mean(),mean_gain_ae=j.gain_ae.mean(),median_gain_ae=j.gain_ae.median()))
    r=pd.DataFrame(rows);r.to_csv(args.out/'R1-5_new5000_spearman_by_split.csv',index=False)
    summary=r.groupby(['neighborhood_metric','gain_type']).agg(rho_mean=('spearman_rho','mean'),rho_sd=('spearman_rho','std'),rho_min=('spearman_rho','min'),rho_max=('spearman_rho','max'),n_splits_positive=('spearman_rho',lambda x:(x>0).sum()),n_splits=('spearman_rho','size'),n_queries_total_across_splits=('n_test','sum')).reset_index()
    summary.to_csv(args.out/'R1-5_new5000_spearman_summary.csv',index=False)
    print(summary.to_string(index=False))
if __name__=='__main__':main()
