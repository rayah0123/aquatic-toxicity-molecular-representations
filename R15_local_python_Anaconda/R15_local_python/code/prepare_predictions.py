from pathlib import Path
import zipfile, io, json, hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--sources',type=Path,required=True,help='Directory containing the four source ZIP archives')
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
args=parser.parse_args()
SRC=args.sources
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)
def read(z,n):
 with zipfile.ZipFile(SRC/z) as f:return pd.read_csv(io.BytesIO(f.read(n)))
a=read('results_for_review.zip','all_evaluation_predictions.csv')
b=read('results_for_review(1).zip','all_evaluation_predictions.csv')
base=pd.concat([a,b],ignore_index=True)
absd=pd.concat([read('absolute_weight_results_for_review.zip',p+'/all_evaluation_predictions.csv') for p in ['daphnia','fathead']],ignore_index=True)
keys=['dataset','split_seed','model_seed','row_index','variant']
shared=base[base.variant.isin(['self','k10'])].merge(absd[absd.variant.isin(['self','k10'])],on=keys,suffixes=('_main','_absolute'),validate='one_to_one')
assert np.allclose(shared.y_true_main,shared.y_true_absolute)
max_reproduction=float((shared.y_pred_main-shared.y_pred_absolute).abs().max())
pred=pd.concat([base,absd[absd.variant.eq('absolute_weighted')]],ignore_index=True)
assert not pred.duplicated(keys).any()
assert pred.groupby(['dataset','split_seed','row_index']).y_true.nunique().max()==1
pred['ae']=(pred.y_true-pred.y_pred).abs()
k=['dataset','split_seed','row_index']
agg=pred.groupby(k+['variant']).agg(ae=('ae','mean'),n_seeds=('model_seed','nunique')).reset_index()
assert agg.n_seeds.eq(5).all()
w=agg.pivot(index=k,columns='variant',values='ae')
assert not w.isna().any().any()
variants=[c for c in w if c!='self']
delta=w[variants].sub(w['self'],axis=0)
# No practical-effect threshold: sign counts describe any numerical difference.
tol=1e-6
sign=np.where(delta.to_numpy() < -tol,-1,np.where(delta.to_numpy()>tol,1,0))
status=pd.Series(np.where((sign==-1).all(axis=1),'gain_all',np.where((sign==1).all(axis=1),'loss_all','mixed_or_tie')),index=w.index)
chem=delta.copy();chem.columns=['delta_ae_'+c for c in chem]
chem['status']=status;chem['n_graph_variants_gain']=(sign==-1).sum(axis=1);chem['ae_self']=w['self'];chem['ae_k10']=w['k10'];chem.reset_index().to_csv(OUT/'chemical_configuration_deltas.csv',index=False)
srows=[]
for (ds,sp),ix in w.groupby(level=[0,1]).groups.items():
 ss=status.loc[ix];d=delta.loc[ix]
 srows.append(dict(dataset=ds,split_seed=int(sp),n=len(ss),n_variants=len(variants),gain_all_n=int(ss.eq('gain_all').sum()),loss_all_n=int(ss.eq('loss_all').sum()),mixed_or_tie_n=int(ss.eq('mixed_or_tie').sum()),gain_all_pct=100*ss.eq('gain_all').mean(),loss_all_pct=100*ss.eq('loss_all').mean(),mixed_or_tie_pct=100*ss.eq('mixed_or_tie').mean(),k10_gain_pct=100*d.k10.lt(-tol).mean()))
summary=pd.DataFrame(srows);summary.to_csv(OUT/'configuration_consistency_by_split.csv',index=False)
# ADORE repeats the same chemicals across all five split models.
ad=delta.loc['ADORE'];assert ad.groupby(level='row_index').size().eq(5).all()
ar=[]
for ri,d in ad.groupby(level='row_index'):
 v=d.to_numpy();ar.append(dict(row_index=int(ri),n_comparisons=v.size,gain_n=int((v < -tol).sum()),loss_n=int((v>tol).sum()),gain_every_configuration_and_split=bool((v < -tol).all()),loss_every_configuration_and_split=bool((v>tol).all())))
adc=pd.DataFrame(ar);adc.to_csv(OUT/'ADORE_across_splits_and_configurations.csv',index=False)
nb=read('neighbor_results.zip','neighbor_results/neighbor_error_by_chemical_split.csv')
nb['sims']=nb.neighbor_similarities.map(json.loads)
nb=nb.merge(w.reset_index(),on=k,validate='one_to_one')
assert np.allclose(nb.delta_ae,nb.k10-nb.self,atol=1e-6)
metrics=read('results_for_review.zip','all_evaluation_metrics.csv')
coverage=[];effect=[];cm=[]
for v,thr in [('t01',.1),('t02',.2),('t03',.3)]:
 nb['retained']=nb.sims.map(lambda xs:sum(s>thr for s in xs))
 for (ds,sp),d in nb.groupby(['dataset','split_seed']):
  m=metrics[(metrics.dataset==ds)&(metrics.split_seed==sp)&(metrics.variant==v)]
  assert len(m)==5
  assert np.allclose(m.query_neighbor_count_mean,d.retained.mean(),atol=1e-9)
  assert m.query_zero_neighbor_n.eq(d.retained.eq(0).sum()).all()
  dv=d[v]-d.k10
  coverage.append(dict(dataset=ds,split_seed=int(sp),variant=v,threshold=thr,n=len(d),mean_retained=d.retained.mean(),zero_neighbor_n=int(d.retained.eq(0).sum()),zero_neighbor_pct=100*d.retained.eq(0).mean(),mean_delta_vs_k10=dv.mean(),mean_delta_vs_self=(d[v]-d.self).mean()))
  for label,mask in [('0',d.retained.eq(0)),('1-4',d.retained.between(1,4)),('5-9',d.retained.between(5,9)),('10',d.retained.eq(10))]:
   dd=d[mask]
   if len(dd):effect.append(dict(dataset=ds,split_seed=int(sp),variant=v,retained_group=label,n=len(dd),mean_delta_vs_k10=(dd[v]-dd.k10).mean(),median_delta_vs_k10=(dd[v]-dd.k10).median(),gain_vs_k10_pct=100*(dd[v]-dd.k10).lt(-tol).mean(),mean_delta_vs_self=(dd[v]-dd.self).mean()))
  for _,r in d.iterrows():cm.append(dict(dataset=ds,split_seed=int(sp),row_index=int(r.row_index),variant=v,threshold=thr,retained_neighbors=int(r.retained),delta_vs_k10=float(r[v]-r.k10),delta_vs_self=float(r[v]-r.self)))
pd.DataFrame(coverage).to_csv(OUT/'cutoff_coverage_and_error_by_split.csv',index=False)
pd.DataFrame(effect).to_csv(OUT/'cutoff_retained_groups_by_split.csv',index=False)
pd.DataFrame(cm).to_csv(OUT/'cutoff_chemical_details.csv',index=False)
# Correlations remain within split; overlapping partitions are not independent studies.
cr=[]
for (ds,sp),d in nb.groupby(['dataset','split_seed']):
 for f in ['nearest_similarity','selected_mean_similarity']:
  for target,y in [('G2_ae',d.k10),('descriptor_only_ae',d.self),('G2_minus_descriptor_ae',d.k10-d.self)]:
   cr.append(dict(dataset=ds,split_seed=int(sp),similarity=f,error_measure=target,n=len(d),spearman_rho=float(spearmanr(d[f],y).statistic)))
pd.DataFrame(cr).to_csv(OUT/'similarity_error_correlations.csv',index=False)
# Match chemical identities using supplied SMILES; do not assign unverified names.
nb[k+['SMILES','y_true']].to_csv(OUT/'chemical_identity_TEST_ADORE.csv',index=False)
meta=dict(analysis_type='Post-review exploratory reanalysis of existing predictions; no training',variants=variants,n_chemical_split_rows=len(w),sign_tolerance=tol,max_baseline_prediction_difference_between_archives=max_reproduction,seed_aggregation='mean absolute error across five model seeds, NOT error of ensemble mean prediction',split_policy='report each split separately; no pooled independent-observation significance tests',cutoff_reconstruction='Top10 neighbors with Tanimoto strictly greater than threshold; verified against exported per-run coverage',cutoff_causal_limit='Each cutoff model was retrained previously. Error changes jointly reflect changed training topology and query neighborhoods; not a controlled removal-only intervention.',sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SRC.glob('*.zip')})
(OUT/'analysis_manifest.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
print('VALIDATION',json.dumps({k:meta[k] for k in ['n_chemical_split_rows','max_baseline_prediction_difference_between_archives','variants']}))
print('CONSISTENCY\n',summary.to_string(index=False))
print('ADORE_all45',adc[['gain_every_configuration_and_split','loss_every_configuration_and_split']].sum().to_dict())
print('CUTOFF\n',pd.DataFrame(coverage).groupby(['dataset','variant']).agg(retained_min=('mean_retained','min'),retained_max=('mean_retained','max'),zero_pct_min=('zero_neighbor_pct','min'),zero_pct_max=('zero_neighbor_pct','max'),delta_mean=('mean_delta_vs_k10','mean'),delta_min=('mean_delta_vs_k10','min'),delta_max=('mean_delta_vs_k10','max')).to_string())
print('CUTOFF_GROUPS\n',pd.DataFrame(effect).groupby(['dataset','variant','retained_group']).agg(n_split_groups=('n','size'),n_min=('n','min'),n_max=('n','max'),delta_min=('mean_delta_vs_k10','min'),delta_max=('mean_delta_vs_k10','max')).to_string())
