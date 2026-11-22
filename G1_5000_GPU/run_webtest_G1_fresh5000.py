#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train and evaluate G1 on WebTEST with a maximum of 5,000 epochs.

Stages: check, train (25 fresh TRAIN/VAL runs), evaluate (held-out TEST
after successful completion of all runs). The fixed G1 model specification
is used; historical model weights are not loaded.
WebTEST targets are model-derived QSAR consensus predictions, not measured toxicity."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

N = 119276
SEEDS = (42, 123, 777, 2025, 3407)
EPOCHS = 5000
PATIENCE = 50
SCRIPTS = (
    'P05G_00_g1_core.py',
    'P05G_01_build_G1_graph_assets.py',
    'P05G_03_train_G1_graphsage_v1_1.py',
    'P05G_04_run_G1_multisplit_multiseed.py',
    'P05G_05_evaluate_frozen_TEST.py',
)

def require(test, msg):
    if not test:
        raise RuntimeError(msg)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def writej(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(path)

def paths(root, here, out):
    require(root.is_dir(), f'Project root missing: {root}')
    forbidden=[root/'03_processed',root/'04_splits',root/'05_features',root/'07_results',root/'webtest_budget2000',root/'webtest_budget5000',root/'webtest_fresh5000']
    require(out != root and all(out != p and p not in out.parents for p in forbidden),
            'Output directory must NOT overwrite project data, any original results, or G2 study.')
    src = here/'original_scripts'
    dataset = root/'03_processed/WebTEST/WebTEST_Daphnia_consensus_unique_connectivity_frozen.csv'
    split_root = root/'04_splits/WebTEST_Daphnia/scaffold_split_murcko_topology_refined'
    graphdir=root/'05_features/G1_WebTEST_revised'
    graph=graphdir/'g1_graph_assets.pt'
    meta=graphdir/'g1_graph_asset_meta.json'
    rowmap=graphdir/'g1_graph_rowmap.csv'
    required=[dataset,graph,meta,rowmap]+[src/p for p in SCRIPTS]
    required += [split_root/f'split_seed{s}/split_membership.csv' for s in SEEDS]
    missing=[str(p) for p in required if not p.is_file()]
    require(not missing, 'Missing required files:\n'+'\n'.join(missing))
    return dict(dataset=dataset, split_root=split_root, graph=graph, meta=meta,rowmap=rowmap,src=src, required=required)

def base_check(root,here,out, heavy=True):
    import torch
    from rdkit import rdBase
    try:
        import torch_geometric
    except ImportError as e:
        raise RuntimeError('PyTorch Geometric is required in the active GPU environment.') from e
    require(torch.cuda.is_available(), 'CUDA is not available; check conda environment.')
    p=paths(root,here,out)
    graph_meta=js(p['meta'])
    require(graph_meta.get('n_rows')==N and graph_meta.get('contains_y') is False,
            'G1 unlabeled graph asset is not the expected N=119276 cohort')
    require(graph_meta.get('smiles_col')=='SMILES_RDKit','Graph SMILES source differs from frozen WebTEST SMILES_RDKit')
    core_module=p['src']/'P05G_00_g1_core.py'
    sys.path.insert(0,str(p['src']))
    import P05G_00_g1_core as core
    require(graph_meta.get('graph_schema_hash')==core.graph_schema_hash(),
            'Graph schema mismatch. Check RDKit version and original graph creation; DO NOT rebuild silently.')
    if heavy:
        require(sha(p['graph'])==graph_meta.get('asset_file_sha256'),
                'G1 graph .pt SHA256 does not match its frozen metadata.')
    d=pd.read_csv(p['dataset'], encoding='utf-8-sig', low_memory=False, usecols=['SMILES_RDKit','target_neglog10_molar'])
    require(len(d)==N,'WebTEST frozen cohort size mismatch')
    require(d['SMILES_RDKit'].notna().all(),'Null SMILES')
    y=pd.to_numeric(d['target_neglog10_molar'],errors='raise').to_numpy(float)
    require(np.isfinite(y).all(),'Nonfinite targets')
    m=pd.read_csv(p['rowmap'],encoding='utf-8-sig',usecols=['global_row_index','SMILES'])
    require(len(m)==N and np.array_equal(m.global_row_index.to_numpy(),np.arange(N)), 'G1 rowmap index mismatch')
    smiles=d.SMILES_RDKit.astype(str).tolist()
    require(m.SMILES.astype(str).tolist()==smiles, 'G1 rowmap SMILES do not match frozen data order')
    require(core.smiles_sequence_hash(smiles)==graph_meta.get('smiles_sequence_sha256'),'G1 graph SMILES sequence hash mismatch')
    for s in SEEDS:
        sp=p['split_root']/f'split_seed{s}/split_membership.csv'
        sm=pd.read_csv(sp, encoding='utf-8-sig', usecols=lambda c: c in ('split','original_row_index','row_idx','index','global_row_index'))
        col=next((c for c in ('original_row_index','row_idx','index','global_row_index') if c in sm.columns),None)
        require(col is not None and 'split' in sm.columns,f'Bad split CSV {sp}')
        inds=pd.to_numeric(sm[col],errors='raise').to_numpy(dtype=int)
        require(len(inds)==N and len(set(inds.tolist()))==N and inds.min()==0 and inds.max()==N-1,f'Partition coverage failed for split={s}')
        parts=sm['split'].astype(str).str.lower().value_counts().to_dict()
        require(set(parts)=={'train','val','test'} and all(v>0 for v in parts.values()),f'Unexpected split labels: {parts}')
    source_hash={x:sha(p['src']/x) for x in SCRIPTS}
    expected_source_hash=js(here/'original_source_sha256.json')
    require(source_hash==expected_source_hash, 'Original G1 source files changed from the audited ZIP. Abort before training.')
    dataset_hash=sha(p['dataset'])
    split_hash={str(s):sha(p['split_root']/f'split_seed{s}/split_membership.csv') for s in SEEDS}
    freeze={
        'experiment':'WebTEST G1 original G1GraphSAGE 5k-cap fresh training',
        'status':'EXPLORATORY_TRAINING_BUDGET_UPDATE_AFTER_EARLIER_TEST_OBSERVATION',
        'dataset_n':N,'split_seeds':list(SEEDS),'model_seeds':list(SEEDS),
        'max_epochs':EPOCHS,'patience':PATIENCE,'selection_metric':'validation_RMSE_original_scale',
        'architecture':{'atom_features':10,'hidden_dim':128,'num_layers':2,'dropout':0.3,'pool':'global_mean_pool'},
        'optimizer':{'name':'Adam','lr':1e-3,'weight_decay':1e-4},
        'batches':{'train':1024,'validation':2048},
        'fresh_model_init':True,'historical_checkpoint_weights_loaded':False,
        'original_graph_sha256':graph_meta.get('asset_file_sha256'),
        'original_graph_schema_hash':graph_meta.get('graph_schema_hash'),
        'frozen_dataset_sha256':dataset_hash,'splits_sha256':split_hash,
        'source_sha256':source_hash,
        'torch':torch.__version__,'pyg':torch_geometric.__version__,
        'rdkit':rdBase.rdkitVersion, 'CUDA':torch.cuda.get_device_name(0),
        'test_policy':'No Test evaluation until all 25 independent G1 runs completed and reviewed',
    }
    signature=hashlib.sha256(json.dumps(freeze,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()
    freeze=dict(freeze,signature=signature)
    out.mkdir(parents=True,exist_ok=True)
    freeze_path=out/'experiment_freeze.json'
    if freeze_path.is_file():
        require(js(freeze_path)==freeze,'Existing study freeze differs; do not mix environments or input revisions. Use another output directory.')
    else:
        writej(freeze_path,freeze)
    print('[PASS] CUDA:',torch.cuda.get_device_name(0), 'torch=',torch.__version__,'PyG=',torch_geometric.__version__,flush=True)
    print('[PASS] G1 original unlabeled graph and WebTEST row/split matching; N=',N,flush=True)
    print('[PASS] Five existing scaffold splits, frozen source checksums and study config',flush=True)
    print('[PASS] Independent output folder:',out,flush=True)
    return p,freeze,d

def target_asset(d,out,freeze):
    p=out/'g1_target_frozen.csv';meta=out/'g1_target_meta.json'
    if not p.is_file():
        data=pd.DataFrame({'global_row_index':np.arange(N,dtype=np.int64),'SMILES':d.SMILES_RDKit.astype(str),
                           'y':pd.to_numeric(d.target_neglog10_molar,errors='raise').to_numpy(float)})
        tmp=p.with_name(p.name+'.partial')
        data.to_csv(tmp,index=False,encoding='utf-8-sig')
        tmp.replace(p)
    require(p.is_file(),'Target asset creation failed')
    if meta.is_file():
        old=js(meta)
        require(old['experiment_signature']==freeze['signature'] and old['sha256']==sha(p),'Target asset provenance mismatch')
    else:
        # Verify every row, not just aggregate hash
        yd=pd.read_csv(p,encoding='utf-8-sig')
        require(len(yd)==N and np.array_equal(yd.global_row_index.to_numpy(),np.arange(N)),'Target asset row order mismatch')
        require(yd.SMILES.astype(str).tolist()==d.SMILES_RDKit.astype(str).tolist(),'Target asset SMILES changed')
        require(np.allclose(yd.y.to_numpy(float),d.target_neglog10_molar.to_numpy(float),rtol=0,atol=1e-12),'Target values changed')
        writej(meta,{'experiment_signature':freeze['signature'],'sha256':sha(p),'target_definition':'-log10(LC50 [mol/L])','N':N})
    return p

def run_dir(out,s,m):
    return out/'validation'/f'g1_graphsage_split{s}_seed{m}'

def verify_run(out,s,m,freeze,core,need=True):
    import torch
    r=run_dir(out,s,m)
    if not (r/'result.json').is_file():
        if need: raise RuntimeError(f'Incomplete run: {r}')
        return False
    a=js(r/'result.json')
    require(int(a.get('split_seed',-1))==s and int(a.get('model_seed',-1))==m,'Wrong split or model seed in stored run')
    require(a.get('test_evaluated') is False,'TEST evaluated inside validation run')
    bp=Path(a['bundle_path'])
    require(bp.is_file() and bp.resolve()==(r/'model_bundle.pt').resolve(),'Wrong bundle path')
    require(core.file_sha256(bp)==a['bundle_sha256'],'Saved bundle digest mismatch')
    bundle=torch.load(bp,map_location='cpu',weights_only=False)
    spec=bundle['training_spec']
    require(spec['max_epochs']==EPOCHS and spec['patience']==PATIENCE,'Run did not use max5000/patience50')
    require(spec['split_seed']==s and spec['model_seed']==m,'Bundle run identity mismatch')
    require(spec['selection_metric']=='validation_RMSE_original_scale','Unexpected checkpoint criterion')
    require(spec['test_evaluated'] is False,'TEST evaluated in bundle')
    hist=pd.read_csv(r/'history.csv',encoding='utf-8-sig')
    require(len(hist)>=1 and len(hist)<=EPOCHS and hist.epoch.tolist()==list(range(1,len(hist)+1)),'Bad epoch history')
    vals=hist.val_RMSE.to_numpy(float)
    require(np.isfinite(vals).all(),'Nonfinite validation history')
    best=int(np.argmin(vals))+1
    require(best==int(a['best_epoch'])==int(spec['best_epoch']),'Saved best epoch differs from history min')
    require(abs(vals[best-1]-a['val_RMSE'])<2e-6,'Stored best val differs from history')
    require(len(hist)==EPOCHS or len(hist)==best+PATIENCE,'Early stopping did not match patience50')
    return True

def verify_all(out,freeze,require_all=True):
    sys.path.insert(0,str(Path(__file__).resolve().parent/'original_scripts'))
    import P05G_00_g1_core as core
    done=[]
    for s in SEEDS:
        for m in SEEDS:
            if verify_run(out,s,m,freeze,core,need=require_all): done.append((s,m))
    return done

def train(p,freeze,d,out):
    import torch
    require(torch.cuda.is_available(),'CUDA not available')
    target=target_asset(d,out,freeze)
    done=verify_all(out,freeze,require_all=False)
    if (out/'TRAINING_25_COMPLETE.json').exists():
        require(len(done)==25 and js(out/'TRAINING_25_COMPLETE.json')['experiment_signature']==freeze['signature'],'Completion marker invalid')
        print('[PASS] G1 already completed 25/25. Do not retrain.',flush=True)
        return
    print(f'[START] Validated completed G1 runs: {len(done)}/25; NEW model initializations only.',flush=True)
    command=[sys.executable,str(p['src']/'P05G_04_run_G1_multisplit_multiseed.py'),
      '--trainer','P05G_03_train_G1_graphsage_v1_1.py',
      '--graph_asset',str(p['graph']), '--y_csv',str(target), '--split_root',str(p['split_root']),
      '--out_dir',str(out/'validation'),'--split_seeds',*[str(s) for s in SEEDS],
      '--model_seeds',*[str(m) for m in SEEDS],
      '--hidden_dim','128','--num_layers','2','--dropout','0.30','--lr','0.001',
      '--weight_decay','0.0001','--epochs','5000','--patience','50',
      '--train_batch_size','1024','--eval_batch_size','2048','--device','cuda','--skip_existing']
    print('[RUN]', ' '.join(command),flush=True)
    subprocess.run(command,check=True)
    verified=verify_all(out,freeze,require_all=True)
    require(len(verified)==25,'All G1 runs were not verified')
    manifest=out/'validation/g1_model_manifest.csv'
    require(manifest.is_file(),'Model manifest missing')
    mm=pd.read_csv(manifest,encoding='utf-8-sig')
    pairs=set(zip(mm.split_seed.astype(int),mm.model_seed.astype(int)))
    require(len(mm)==25 and pairs=={(s,m) for s in SEEDS for m in SEEDS},'G1 manifest grid incomplete')
    require(not mm.test_evaluated.astype(str).str.lower().isin(('true','1','yes')).any(),'Train manifest indicates test already opened')
    writej(out/'TRAINING_25_COMPLETE.json',{'experiment_signature':freeze['signature'],'n_runs':25,
           'phase':'TRAIN_VALIDATION_ONLY','completed_at_utc':datetime.now(timezone.utc).isoformat(),
           'manifest_sha256':sha(manifest)})
    print('[PASS] G1 25/25 independently trained, best-validation models stored; NO TEST evaluated.',flush=True)

def evaluate(p,freeze,d,out):
    cm=out/'TRAINING_25_COMPLETE.json'
    require(cm.is_file() and js(cm).get('experiment_signature')==freeze['signature'],'No verified 25/25 G1 completion marker; Test is blocked.')
    require(len(verify_all(out,freeze,True))==25,'Not all 25 G1 runs verified')
    man=out/'validation/g1_model_manifest.csv'
    require(sha(man)==js(cm)['manifest_sha256'],'Manifest modified since training completion')
    target=out/'g1_target_frozen.csv'
    require(target.is_file(),'G1 frozen target missing')
    testdir=out/'test'
    require(not (testdir/'g1_frozen_test_all_runs.csv').exists(),'TEST outputs already exist; do not silently overwrite, use the existing results.')
    command=[sys.executable,str(p['src']/'P05G_05_evaluate_frozen_TEST.py'),
       '--manifest',str(man),'--graph_asset',str(p['graph']),'--y_csv',str(target),
       '--split_root',str(p['split_root']),'--out_dir',str(testdir),'--device','cuda']
    print('[TEST UNLOCKED] All 25 runs verified. Evaluating held-out TEST one time.',flush=True)
    subprocess.run(command,check=True)
    te=pd.read_csv(testdir/'g1_frozen_test_all_runs.csv',encoding='utf-8-sig')
    va=pd.read_csv(out/'validation/g1_validation_all_runs.csv',encoding='utf-8-sig')
    require(len(te)==25 and len(va)==25,'Metric results have wrong run counts')
    both=va.merge(te,on=['run_name','split_seed','model_seed'],how='inner',validate='one_to_one')
    require(len(both)==25,'Train/Val/Test run alignment failed')
    both['gap_val_minus_train_RMSE']=both.val_RMSE-both.train_RMSE
    both['gap_test_minus_train_RMSE']=both.test_RMSE-both.train_RMSE
    both.to_csv(out/'all_train_val_test_metrics.csv',index=False,encoding='utf-8-sig')
    cols=[c for c in both.columns if c.startswith(('train_','val_','test_','gap_')) and pd.api.types.is_numeric_dtype(both[c])]
    split=both.groupby('split_seed',as_index=False)[cols].mean(numeric_only=True)
    split.to_csv(out/'splitmean_train_val_test_metrics.csv',index=False,encoding='utf-8-sig')
    s=[]
    for c in cols:
        x=split[c].dropna();s.append({'metric':c,'mean_of_splitmeans':float(x.mean()),'sd_across_splits':float(x.std(ddof=1)),'n_splits':len(x)})
    pd.DataFrame(s).to_csv(out/'summary_train_val_test_metrics.csv',index=False,encoding='utf-8-sig')
    with zipfile.ZipFile(out/'G1_results_for_review.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        files=[out/'experiment_freeze.json',cm,out/'g1_target_meta.json',out/'all_train_val_test_metrics.csv',
               out/'splitmean_train_val_test_metrics.csv',out/'summary_train_val_test_metrics.csv']
        files += list((out/'validation').glob('*.csv'))+list((out/'test').glob('*.csv'))
        files += list((out/'validation').glob('*/history.csv'))+list((out/'validation').glob('*/result.json'))
        for f in files:
            if f.is_file():z.write(f,f.relative_to(out))
    writej(out/'TEST_EVALUATED.json',{'experiment_signature':freeze['signature'],'runs':25,
           'status':'test_scored_once_after_all_validation_runs','completed_at_utc':datetime.now(timezone.utc).isoformat()})
    print('[COMPLETE] G1 validation-selected 5000 cap, Train/Val/Test and curves:',out/'G1_results_for_review.zip',flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--stage',required=True,choices=('check','train','evaluate'))
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'g1_fresh5000_results')
    a=ap.parse_args();root=a.project_root.resolve();here=Path(__file__).resolve().parent;out=a.out.resolve()
    p,freeze,d=base_check(root,here,out,heavy=True)
    if a.stage=='check':
        print('[READY] Next stage: train. No training or TEST evaluation performed.',flush=True)
    elif a.stage=='train':train(p,freeze,d,out)
    elif a.stage=='evaluate':evaluate(p,freeze,d,out)

if __name__=='__main__':main()
