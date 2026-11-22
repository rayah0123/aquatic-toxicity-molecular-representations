from pathlib import Path
import argparse,json,zipfile,io,hashlib
import pandas as pd
import numpy as np
from rdkit import Chem,rdBase
p=argparse.ArgumentParser();p.add_argument('--sources',type=Path,required=True);p.add_argument('--prior',type=Path,required=True);p.add_argument('--output',type=Path,default=Path(__file__).parent);args=p.parse_args()
out=args.output;out.mkdir(exist_ok=True,parents=True)
protocol=json.loads((Path(__file__).parent/'protocol.json').read_text(encoding='utf-8'));patterns={k:Chem.MolFromSmarts(v) for k,v in protocol['patterns'].items()};assert all(v is not None for v in patterns.values())
# Verify SMARTS patterns using positive and negative examples.
examples={'aromatic':('c1ccccc1','C1CCCCC1'),'carbon_bound_halogen':('CCCl','[Cl-]'),'phenolic_OH':('Oc1ccccc1','CO'),'aliphatic_alcohol_OH':('CCO','Oc1ccccc1'),'carboxylic_acid_or_carboxylate':('CC(=O)[O-]','CC(=O)OC'),'ester':('CC(=O)OC','CC(=O)O'),'amide':('CC(=O)N','CCN'),'nitro':('C[N+](=O)[O-]','CN'),'phosphorus_containing':('CP(=O)(O)O','CS'),'sulfur_containing':('CS','CP')}
for k,(pos,neg) in examples.items():assert Chem.MolFromSmiles(pos).HasSubstructMatch(patterns[k]) and not Chem.MolFromSmiles(neg).HasSubstructMatch(patterns[k]),k
identity=pd.read_csv(args.prior/'chemical_identity_TEST_ADORE.csv')
assert identity.groupby(['dataset','row_index']).SMILES.nunique().max()==1
u=identity[['dataset','row_index','SMILES']].drop_duplicates().copy(); records=[]
for _,r in u.iterrows():
 m=Chem.MolFromSmiles(r.SMILES)
 rec=r.to_dict();rec['parse_ok']=m is not None
 if m is not None:
  rec['canonical_smiles']=Chem.MolToSmiles(m);rec['n_fragments']=len(Chem.GetMolFrags(m))
  rec.update({k:m.HasSubstructMatch(q) for k,q in patterns.items()})
 records.append(rec)
flags=pd.DataFrame(records);flags.to_csv(out/'chemical_structural_flags.csv',index=False);assert flags.parse_ok.all(),'Invalid structures must be handled before analysis'
# Retain row identifiers used by original evaluation, and audit canonical duplicates explicitly.
dups=flags[flags.duplicated(['dataset','canonical_smiles'],keep=False)];dups.to_csv(out/'canonical_duplicate_audit.csv',index=False)
k=['dataset','split_seed','row_index'];w=pd.read_csv(args.prior/'chemical_configuration_deltas.csv').merge(identity,on=k,validate='one_to_one').merge(flags.drop(columns='SMILES'),on=['dataset','row_index'],validate='many_to_one')
with zipfile.ZipFile(args.sources/'neighbor_results.zip') as z:nb=pd.read_csv(io.BytesIO(z.read('neighbor_results/neighbor_error_by_chemical_split.csv')))
w=w.merge(nb[k+['selected_mean_similarity','nearest_similarity']],on=k,validate='one_to_one')
with zipfile.ZipFile(args.sources/'results_for_review.zip') as z:pred=pd.read_csv(io.BytesIO(z.read('all_evaluation_predictions.csv')))
pred=pred[pred.variant.isin(['self','k10'])].copy();pred['sqerr']=(pred.y_true-pred.y_pred)**2
pred=pred.merge(flags,on=['dataset','row_index'],validate='many_to_one')
rows=[]
for (ds,sp),d in w.groupby(['dataset','split_seed']):
 for group in patterns:
  g=d[d[group]];c=d[~d[group]]
  for col in d.filter(regex='^delta_ae_').columns:
   v=col.removeprefix('delta_ae_');delta=g[col];comp=c[col]
   row=dict(dataset=ds,split_seed=int(sp),group=group,variant=v,n=len(g),n_complement=len(c),eligible=len(g)>=10 and len(c)>=10,mean_delta=delta.mean(),median_delta=delta.median(),gain_pct=100*(delta < -1e-6).mean(),mean_complement_delta=comp.mean(),group_minus_complement_delta=delta.mean()-comp.mean(),mean_self_ae=g.ae_self.mean(),mean_graph_ae=(g.ae_self+delta).mean(),mean_neighbor_similarity=g.selected_mean_similarity.mean())
   if v=='k10':
    pr=pred[(pred.dataset==ds)&(pred.split_seed==sp)&pred[group]]
    rm=pr.groupby(['variant','model_seed']).sqerr.mean().pow(.5).groupby(level='variant').mean()
    row.update(self_rmse_mean=rm.get('self',np.nan),k10_rmse_mean=rm.get('k10',np.nan))
   rows.append(row)
s=pd.DataFrame(rows);s.to_csv(out/'subgroup_results_all_settings_by_split.csv',index=False)
s[s.variant.eq('k10')].to_csv(out/'subgroup_results_original_k10_by_split.csv',index=False)
sums=[]
for (ds,g,v),d in s.groupby(['dataset','group','variant']):
 e=d[d.eligible];n=len(e)
 sums.append(dict(dataset=ds,group=g,variant=v,n_unique_structures=int(flags[(flags.dataset==ds)&flags[g]].canonical_smiles.nunique()),n_min=int(d.n.min()),n_max=int(d.n.max()),eligible_splits=n,summary_eligible=n>=3,mean_delta_across_eligible_splits=e.mean_delta.mean() if n>=3 else np.nan,min_delta=e.mean_delta.min() if n>=3 else np.nan,max_delta=e.mean_delta.max() if n>=3 else np.nan,gain_splits=int((e.mean_delta < -1e-6).sum()),loss_splits=int((e.mean_delta>1e-6).sum()),consistent_gain_all5=n==5 and bool((e.mean_delta < -1e-6).all()),consistent_loss_all5=n==5 and bool((e.mean_delta > 1e-6).all()),mean_group_minus_complement=e.group_minus_complement_delta.mean() if n>=3 else np.nan,median_delta_across_eligible_splits=e.median_delta.mean() if n>=3 else np.nan,gain_pct_min=e.gain_pct.min() if n>=3 else np.nan,gain_pct_max=e.gain_pct.max() if n>=3 else np.nan))
su=pd.DataFrame(sums);su.to_csv(out/'subgroup_summary_all_settings.csv',index=False);su[su.variant.eq('k10')].to_csv(out/'subgroup_summary_original_k10.csv',index=False)
# Overlap matrix counts unique dataset row IDs, not repeated split appearances.
ov=[]
for ds,d in flags.groupby('dataset'):
 for a in patterns:
  for b in patterns:ov.append(dict(dataset=ds,group_a=a,group_b=b,n_overlap=int((d[a]&d[b]).sum())))
pd.DataFrame(ov).to_csv(out/'group_overlap_counts.csv',index=False)
w.to_csv(out/'chemical_errors_with_flags.csv',index=False)
manifest={'rdkit_version':rdBase.rdkitVersion,'n_chemical_split_rows':len(w),'n_distinct_dataset_row_ids':len(flags),'parse_failures':int((~flags.parse_ok).sum()),'canonical_duplicate_rows':len(dups),'n_multifragment_rows':int((flags.n_fragments>1).sum()),'dataset_counts':flags.groupby('dataset').size().to_dict(),'scope':'TEST Daphnia and ADORE only. FHM and WebTEST not analyzed.','training_performed':False,'unit':'original toxicity target scale, -log10(LC50 mol/L)','protocol_sha256':hashlib.sha256((Path(__file__).parent/'protocol.json').read_bytes()).hexdigest(),'inputs':{str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in [args.prior/'chemical_identity_TEST_ADORE.csv',args.prior/'chemical_configuration_deltas.csv',args.sources/'neighbor_results.zip',args.sources/'results_for_review.zip']}}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2));print(su[su.variant.eq('k10')].to_string(index=False))
