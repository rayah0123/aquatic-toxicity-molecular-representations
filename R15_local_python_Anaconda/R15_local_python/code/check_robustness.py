from pathlib import Path
import json
import pandas as pd
from rdkit import Chem
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
p=parser.parse_args().output
w=pd.read_csv(p/'chemical_errors_with_flags.csv');rows=[]
for (ds,sp),d in w.groupby(['dataset','split_seed']):
 for g in ['amide','ester','carboxylic_acid_or_carboxylate','nitro','phosphorus_containing','sulfur_containing']:
  x=d.loc[d[g],'delta_ae_k10'];n=len(x)
  if n>=10:
   loo=(x.sum()-x)/(n-1)
   rows.append(dict(dataset=ds,split_seed=sp,group=g,n=n,mean_delta=x.mean(),leave_one_out_min=loo.min(),leave_one_out_max=loo.max(),sign_survives_every_single_removal=bool((loo<0).all() if x.mean()<0 else (loo>0).all())))
pd.DataFrame(rows).to_csv(p/'leave_one_out_diagnostic.csv',index=False)
# Examine overlap between ester and amide motifs in carbamate structures.
query=Chem.MolFromSmarts('[N]-[C;X3](=[O;X1])-[O;X2]-[#6]')
w['carbamate_motif']=w.SMILES.map(lambda s:Chem.MolFromSmiles(s).HasSubstructMatch(query))
w['amide_without_carbamate']=w.amide & ~w.carbamate_motif
w['ester_without_carbamate']=w.ester & ~w.carbamate_motif
rows=[]
for (ds,sp),d in w.groupby(['dataset','split_seed']):
 for g in ['carbamate_motif','amide_without_carbamate','ester_without_carbamate']:
  x=d[d[g]];a=x.delta_ae_k10
  rows.append(dict(dataset=ds,split_seed=sp,group=g,n=len(x),mean_delta=a.mean(),median_delta=a.median(),eligible=len(x)>=10 and len(d)-len(x)>=10))
r=pd.DataFrame(rows);r.to_csv(p/'carbamate_overlap_diagnostic.csv',index=False)
print(r[r.dataset.eq('ADORE')].to_string(index=False))
