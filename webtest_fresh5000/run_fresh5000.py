#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run WebTEST G2 fresh training with a 5,000-epoch maximum.

Uses configuration-only references for the fixed model specification and
TRAIN-fitted preprocessing; no historical weights or optimizer state are loaded.
Stages: check, train (100 split/seed/condition runs on TRAIN/VAL), evaluate
(held-out TEST only after all 100 runs pass completion checks).
Prior TEST metrics and predictions are not used for training or model selection.
WebTEST targets are model-derived QSAR consensus predictions, not measured toxicity."""
from __future__ import annotations
import argparse, hashlib, json, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import rdBase
import training_engine as engine

SEEDS = (42,123,777,2025,3407)
MODES = ('self','unweighted','weighted','absolute_weighted')
N_EXPECTED = 119276
CAP = 5000
PATIENCE = 50
FILES = ('run_fresh5000.py','training_engine.py','selection.py','g2_core.py','g2_shared.py','webtest_assets.py')

req=engine.require

def stamp():
    return datetime.now(timezone.utc).isoformat()

def readj(p):return json.loads(p.read_text(encoding='utf-8'))

def bundlepath(root,s,m,mode):
    v = 'descriptor_self' if mode=='self' else 'descriptor_tanimoto_K10'
    return root/'07_results/WebTEST_revised/P10D_G2_self_validation/model_specs'/f'split{s}__model{m}__{v}.json'

def load_spec(path):
    """Load the public configuration-only reference; no serialized model is needed."""
    b=readj(path)
    req('state_dict' not in b, 'Reference specification must not contain weights')
    pp=b['preprocessor']
    col_dtype=b.pop('kept_cols_dtype')
    req(col_dtype in ('int32','int64'),'Unexpected column-index dtype')
    for k in ('kept_cols','medians','mean','std'):
        pp[k]=np.asarray(pp[k],dtype=col_dtype if k=='kept_cols' else np.float64)
    return b

def required_paths(root):
    paths=[root/'03_processed/WebTEST/WebTEST_Daphnia_consensus_unique_connectivity_frozen.csv']
    gr=root/'07_results/WebTEST_revised/P10C_exact_Top10_graph_assets'
    sr=root/'04_splits/WebTEST_Daphnia/scaffold_split_murcko_topology_refined'
    for s in SEEDS:
        paths.append(sr/f'split_seed{s}/split_membership.csv')
        paths.extend(gr/f'split_seed{s}'/f for f in ['train_undirected_union_exact_top10.npz','val_source_to_train_exact_top10.npz','test_source_to_train_exact_top10.npz'])
        for m in SEEDS:
            paths.append(bundlepath(root,s,m,'self'))
            paths.append(bundlepath(root,s,m,'unweighted'))
    return paths

def freeze_config(root, out, device):
    paths=required_paths(root)
    paths.extend(Path(__file__).with_name(f) for f in FILES)
    missing=[str(p) for p in paths if not p.is_file()]
    req(not missing,'Missing required frozen assets:\n'+'\n'.join(missing[:20]))
    protected=[root/'03_processed',root/'04_splits',root/'05_features',root/'07_results',root/'webtest_budget2000',root/'webtest_budget5000']
    req(out!=root and all(not (out==p or p in out.parents) for p in protected),'Output directory overlaps protected original data/results')
    sources={str(p):engine.sha(p) for p in paths}
    frozen=dict(
       study='WebTEST fixed design, NEW training from epoch 1, maximum 5000, validation-selected checkpoints',
       prior_test_results_seen=True,
       provenance='Exploratory, training-budget changed after inspection of former TEST results; NOT a new independent external holdout',
       n_chemicals=N_EXPECTED, split_seeds=list(SEEDS), model_seeds=list(SEEDS), modes=list(MODES),
       max_epochs=CAP, budget500=500, patience=PATIENCE, min_delta=1e-12,
       training='Extracted training_engine.train_one() retains the historical function body; engine.MAX_EPOCHS=5000, fresh model and Adam in a NEW output folder. Public source hashes differ from historical launcher/engine.',
       test_policy='No TEST evaluation or predictions in check/train; evaluate requires all 100 completion proofs',
       torch_version=str(torch.__version__),rdkit_version=rdBase.rdkitVersion,numpy_version=np.__version__,
       cuda_device=torch.cuda.get_device_name(0) if device.type=='cuda' else 'NOT_CUDA',
       cuda_device_count=torch.cuda.device_count(), input_sha256=sources)
    sig=hashlib.sha256(json.dumps(frozen,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return sig, frozen

def install_freeze(out,sig,frozen):
    out.mkdir(parents=True,exist_ok=True)
    p=out/'experiment_freeze.json'
    if p.exists():req(readj(p)['signature']==sig,'Experiment frozen configuration changed! Use distinct new --out folder, do not overwrite.')
    else:engine.save_json(p,dict(signature=sig,**frozen))

def prepare(root,out,signature,stage):
    inp=root/'03_processed/WebTEST/WebTEST_Daphnia_consensus_unique_connectivity_frozen.csv'
    data=pd.read_csv(inp)
    req(len(data)==N_EXPECTED,'Unexpected WebTEST dataset row count')
    req({'SMILES_RDKit','target_neglog10_molar'}.issubset(data.columns),'Missing input columns')
    y=pd.to_numeric(data.target_neglog10_molar,errors='raise').to_numpy(dtype=float)
    req(np.isfinite(y).all(),'Non-finite target')
    engine.assets.SPLIT_ROOT=root/'04_splits/WebTEST_Daphnia/scaffold_split_murcko_topology_refined'
    engine.assets.GRAPH_ROOT=root/'07_results/WebTEST_revised/P10C_exact_Top10_graph_assets'
    xfile=out/'compact6_raw.npy';qcfile=out/'compact6_cache.json'
    if xfile.is_file() and qcfile.is_file():
        qc=readj(qcfile)
        req(qc.get('signature')==signature and qc.get('sha256')==engine.sha(xfile),'Input descriptor cache changed')
        X=np.load(xfile)
    else:
        X=engine.assets.build_compact6_raw(data.SMILES_RDKit.astype(str).tolist())
        np.save(xfile,X)
        engine.save_json(qcfile,dict(signature=signature,sha256=engine.sha(xfile)))
    req(X.shape==(N_EXPECTED,6) and np.isfinite(X).all(),'Compact6 feature error')
    return X,y

def runpath(out,mode,s,m):
    return out/'runs'/f'{mode}_split{s}_seed{m}'

def completed(run,sig):
    p=run/'complete.json'
    if not p.is_file():return False
    d=readj(p)
    req(d['signature']==sig,'Changed training signature: '+str(run))
    for name,key in [('model_bundle.pt','bundle_sha256'),('model_bundle_500.pt','bundle500_sha256'),('history.csv','history_sha256')]:
        req((run/name).is_file() and engine.sha(run/name)==d[key],'Corrupted completed run: '+str(run/name))
    req(d['epochs_ran']<=CAP and d['stop_reason'] in ('early_stopping','max_epochs'),'Unexpected completion status')
    h=pd.read_csv(run/'history.csv')
    req(len(h)==d['epochs_ran'] and h.epoch.tolist()==list(range(1,len(h)+1)),'History incomplete: '+str(run))
    from selection import update,reason
    best=float('inf');wait=0;be=0
    for row in h.itertuples():
        best,wait,improved=update(best,wait,float(row.val_RMSE))
        if improved:be=int(row.epoch)
    req(reason(len(h),CAP,wait,PATIENCE)==d['stop_reason'],'Stop status not reproduced')
    req(be==d['best_epoch'] and abs(best-d['best_validation_RMSE'])<1e-8,'Best checkpoint selection mismatch')
    return True

def qc_original_bundle(ref,s,m,mode,idx,X,y):
    v='self' if mode=='self' else 'k10'
    engine.validate_bundle(ref,s,m,v,idx,X,y)
    req(int(ref['optimizer_spec']['epochs_max'])==500,'Baseline epoch limit differs from 500')
    req(int(ref['optimizer_spec']['patience'])==PATIENCE,'Patience differs')
    req(set(ref['architecture']).issuperset({'input_dim','hidden_dim','num_layers','dropout'}),'Unexpected architecture fields')
    req(ref['architecture']['hidden_dim']==128 and abs(ref['architecture']['dropout']-0.3)<1e-10,'Unexpected G2 architecture')

def stage_check(root, out, sig, frozen):
    print('[OK] CUDA:',torch.cuda.get_device_name(0),'torch',torch.__version__,flush=True)
    print('[OK] Frozen asset hashes:',len(frozen['input_sha256']),'signature=',sig[:18],flush=True)
    engine.numerical_checks(torch.device('cuda'))
    for s in SEEDS:
        for m in SEEDS:
            for mode in ('self','unweighted'):
                b=load_spec(bundlepath(root,s,m,mode))
                req(b['split_seed']==s and b['model_seed']==m,'Reference identity mismatch')
                req(not b.get('test_evaluated',True),'Original bundle already test-evaluated')
                req(int(b['optimizer_spec']['epochs_max'])==500,'Original max_epoch is not 500')
                req(int(b['optimizer_spec']['patience'])==50,'Original patience is not 50')
    for s in SEEDS:
        idx=engine.assets.load_split(s,N_EXPECTED)
        req(len(np.concatenate(list(idx.values())))==N_EXPECTED,'Split not complete')
    print('[PASS] 50 original validation-only JSON specifications, 5 frozen splits, and native aggregation checks.',flush=True)
    print('[NOTE] No historical checkpoint weights are distributed or loaded.',flush=True)


def train(root,out,sig,X,y,device):
    engine.MAX_EPOCHS=CAP
    run_count=sum(bool(completed(runpath(out,mode,s,m),sig)) for s in SEEDS for m in SEEDS for mode in MODES)
    print(f'[START] completed={run_count}/100; continuing NEW fresh-study directory only',flush=True)
    for s in SEEDS:
        idx,tr,va=engine.prepare_split(s,X,y)
        for m in SEEDS:
            for mode in MODES:
                run=runpath(out,mode,s,m)
                if completed(run,sig):continue
                run.mkdir(parents=True,exist_ok=True)
                ref=load_spec(bundlepath(root,s,m,mode))
                qc_original_bundle(ref,s,m,mode,idx,X,y)
                # engine.train_one always seed_everything(model_seed) and NEW model_for(...)
                # and never model.load_state_dict(ref['state_dict']). Its own 'resume.pt'
                # can only be a continuation within THIS clean study directory.
                new_run_start=not (run/'resume.pt').exists()
                t0=time.perf_counter()
                print(f'[TRAIN] mode={mode} split={s} seed={m} from_epoch1={new_run_start}',flush=True)
                b,b500=engine.train_one(ref,mode,idx,X,y,tr,va,device,run,sig)
                for name,bb in [('model_bundle.pt',b),('model_bundle_500.pt',b500)]:
                    tmp=run/(name+'.tmp')
                    torch.save(bb,tmp);tmp.replace(run/name)
                engine.save_json(run/'complete.json',dict(signature=sig,epochs_ran=int(b['epochs_ran']),
                    stop_reason=b['stop_reason'],best_epoch=int(b['best_epoch']),best_validation_RMSE=float(b['best_validation_RMSE']),
                    bundle_sha256=engine.sha(run/'model_bundle.pt'),bundle500_sha256=engine.sha(run/'model_bundle_500.pt'),
                    history_sha256=engine.sha(run/'history.csv')))
                row=dict(mode=mode,split_seed=s,model_seed=m,started_utc=stamp(),seconds_this_call=round(time.perf_counter()-t0,2),
                    epochs_ran=b['epochs_ran'],stop_reason=b['stop_reason'],best_epoch=b['best_epoch'])
                timing=out/'completed_training_times.csv'
                pd.DataFrame([row]).to_csv(timing,mode='a',header=not timing.exists(),index=False)
                run_count+=1
                print(f'[DONE {run_count}/100] {mode} split={s} seed={m} epochs={b["epochs_ran"]} best={b["best_epoch"]}',flush=True)
        del tr,va
    req(all(completed(runpath(out,mode,s,m),sig) for s in SEEDS for m in SEEDS for mode in MODES),'Some runs missing')
    engine.save_json(out/'TRAINING_100_COMPLETE.json',dict(signature=sig,all_runs=100,verified_at_utc=stamp(),
        note='Held-out TEST evaluation NOT performed by this script stage.'))
    print('[PASS] 100/100 fresh runs complete. NEXT: run --stage evaluate separately.',flush=True)


def evaluate(root,out,sig,X,y,device):
    req((out/'TRAINING_100_COMPLETE.json').is_file(),'No 100-run training completion marker; do not evaluate TEST')
    req(readj(out/'TRAINING_100_COMPLETE.json')['signature']==sig,'Training signature changed')
    for s in SEEDS:
        for m in SEEDS:
            for mode in MODES: req(completed(runpath(out,mode,s,m),sig),'Missing/inconsistent completed run')
    engine.MAX_EPOCHS=CAP
    rowlist=[];save=out/'test_predictions';save.mkdir(exist_ok=True)
    for s in SEEDS:
        idx,tr,va=engine.prepare_split(s,X,y)
        _,te=engine.assets.load_precomputed_edges(s,idx['train'],idx['test'],'test')
        engine.check_edges(te,len(idx['train']),len(idx['test']))
        for m in SEEDS:
            for mode in MODES:
                run=runpath(out,mode,s,m)
                for budget,name in [(500,'model_bundle_500.pt'),(5000,'model_bundle.pt')]:
                    b=engine.load(run/name)
                    label='k10' if mode=='unweighted' else mode
                    vals,preds=engine.evaluate(b,mode,idx,X,y,tr,va,te,device,
                        dict(variant=label,budget=budget,split_seed=s,model_seed=m))
                    rowlist.append(vals)
                    pd.DataFrame(dict(row_index=idx['test'],y_true=y[idx['test']],y_pred=preds['test'])).to_csv(
                        save/f'budget{budget}_{label}_split{s}_seed{m}.csv.gz',index=False)
                print(f'[TEST] {mode} split={s} modelseed={m}',flush=True)
    metrics=pd.DataFrame(rowlist)
    req(len(metrics)==200 and not metrics.duplicated(['budget','variant','split_seed','model_seed']).any(),'Invalid metric grid')
    metrics.to_csv(out/'all_checkpoint_metrics.csv',index=False)
    for budget in [500,5000]:
        sub=out/f'budget{budget}';sub.mkdir(exist_ok=True)
        engine.summarize(metrics[metrics.budget.eq(budget)],sub)
    paired=metrics[metrics.budget.eq(5000)].merge(metrics[metrics.budget.eq(500)],on=['variant','split_seed','model_seed'],suffixes=('_5000','_500'),validate='one_to_one')
    for part in ['train','val','test']:
        for met in ['RMSE','MAE']:
            paired[f'delta_{part}_{met}']=paired[f'{part}_{met}_5000']-paired[f'{part}_{met}_500']
    paired.to_csv(out/'paired_5000_vs_500.csv',index=False)
    paired.groupby(['variant','split_seed'],as_index=False)[[c for c in paired.columns if c.startswith('delta_')]].mean().to_csv(out/'paired_5000_vs_500_splitmeans.csv',index=False)
    metrics[metrics.budget.eq(5000)].groupby(['variant','stop_reason']).size().rename('n_runs').reset_index().to_csv(out/'stopping_summary.csv',index=False)
    with zipfile.ZipFile(out/'results_for_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        fs=[out/'experiment_freeze.json',out/'TRAINING_100_COMPLETE.json']
        fs+=sorted(out.glob('*.csv'))+sorted(out.glob('budget*/*.csv'))+sorted(out.glob('runs/*/history.csv'))
        fs+=sorted(out.glob('runs/*/complete.json'))+sorted(save.glob('*.csv.gz'))
        for p in fs:z.write(p,p.relative_to(out))
    print('[COMPLETE] 100 fresh runs, 500 vs 5000, train/val/TEST and curves:',out/'results_for_review.zip',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parent/'fresh5000_results')
    ap.add_argument('--stage',required=True,choices=['check','train','evaluate'])
    a=ap.parse_args();root=a.project_root.resolve();out=a.out.resolve()
    req(torch.cuda.is_available(),'CUDA required; run in the original GPU environment')
    dev=torch.device('cuda')
    sig,config=freeze_config(root,out,dev)
    engine.assets.SPLIT_ROOT=root/'04_splits/WebTEST_Daphnia/scaffold_split_murcko_topology_refined'
    engine.assets.GRAPH_ROOT=root/'07_results/WebTEST_revised/P10C_exact_Top10_graph_assets'
    install_freeze(out,sig,config)
    if a.stage=='check':
        stage_check(root,out,sig,config)
        return
    X,y=prepare(root,out,sig,a.stage)
    if a.stage=='train':train(root,out,sig,X,y,dev)
    elif a.stage=='evaluate':evaluate(root,out,sig,X,y,dev)

if __name__=='__main__':
    main()
