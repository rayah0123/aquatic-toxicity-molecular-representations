"""Analyze chemical-level and structural-subgroup results from saved predictions.
Requires the companion code/ directory and project-relative evaluation records.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib,io,json,os,subprocess,sys,traceback,zipfile
from datetime import datetime

DEFAULT_PROJECT=Path(__file__).resolve().parents[2]
PRED_COLUMNS={'dataset','variant','split_seed','model_seed','row_index','y_true','y_pred'}
NEIGHBOR_COLUMNS={'dataset','split_seed','row_index','SMILES','y_true','delta_ae','neighbor_similarities','selected_mean_similarity','nearest_similarity'}
METRIC_COLUMNS={'dataset','variant','split_seed','query_neighbor_count_mean','query_zero_neighbor_n'}

class InputProblem(Exception):pass

def check_csv(data,columns,label):
 reader=csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
 missing=columns-set(reader.fieldnames or [])
 if missing:raise InputProblem(f'{label}: missing columns: {sorted(missing)}')
 if next(reader,None) is None:raise InputProblem(f'{label}: file has no data rows')
 return data

def choose(candidates,label,columns):
 if not candidates:raise InputProblem(f'{label}: no existing result file found')
 hashes={hashlib.sha256(data).hexdigest() for _,data in candidates}
 if len(hashes)>1:
  raise InputProblem(label+': different candidate files found; no file was chosen.\n'+'\n'.join('  '+name for name,_ in candidates))
 name,data=sorted(candidates,key=lambda x:x[0])[0]
 check_csv(data,columns,name)
 return {'source':name,'equivalent_sources':[s for s,_ in candidates],'sha256':hashlib.sha256(data).hexdigest(),'data':data}

def archive_member(path,member):
 with zipfile.ZipFile(path) as z:
  options=[n for n in z.namelist() if n==member or n.endswith('/'+member)]
  if len(options)!=1:raise InputProblem(f'{path}: expected one entry {member}, found {len(options)}')
  return str(path)+'::'+options[0],z.read(options[0])

def from_result_dir(directory,filename,columns):
 direct=directory/filename
 if direct.is_file():return choose([(str(direct),direct.read_bytes())],str(direct),columns)
 archive=directory/'results_for_review.zip'
 if archive.is_file():return choose([archive_member(archive,filename)],str(archive),columns)
 raise InputProblem(f'Expected existing CSV or results_for_review.zip in: {directory}\n  CSV: {filename}')

def project_files(project,runner_root):
 # Restrict source discovery to the specified project directory without following symbolic links.
 skip={'output','input','__pycache__','.git','.venv','venv','node_modules','envs'}
 found=[]
 for base,dirs,files in os.walk(project,followlinks=False):
  dirs[:]=[d for d in dirs if d not in skip and not d.lower().startswith('r15_local_python') and not d.lower().startswith('run_') and (Path(base)/d).resolve()!=runner_root.resolve()]
  for n in files:
   low=n.lower()
   if n=='neighbor_error_by_chemical_split.csv' or (low.endswith('.zip') and ('neighbor' in low or ('neighbor' in str(base).lower() and low=='results_for_review.zip'))):found.append(Path(base)/n)
 return found

def resolve_inputs(project,runner_root):
 resolved={}
 ddir=project/'revision_parallel'/'sensitivity_results'
 fdir=project/'fathead_sensitivity'/'fathead_results'
 resolved['daphnia_predictions']=from_result_dir(ddir,'all_evaluation_predictions.csv',PRED_COLUMNS)
 resolved['daphnia_metrics']=from_result_dir(ddir,'all_evaluation_metrics.csv',METRIC_COLUMNS)
 resolved['fathead_predictions']=from_result_dir(fdir,'all_evaluation_predictions.csv',PRED_COLUMNS)
 adir=project/'absolute_weight_run';bundle=adir/'absolute_weight_results_for_review.zip'
 if bundle.is_file():
  for species in ['daphnia','fathead']:
   resolved['absolute_'+species]=choose([archive_member(bundle,species+'/all_evaluation_predictions.csv')],'absolute '+species,PRED_COLUMNS)
 else:
  candidates={'daphnia':[],'fathead':[]}
  if adir.is_dir():
   for f in sorted(adir.rglob('all_evaluation_predictions.csv')):
    data=f.read_bytes();check_csv(data,PRED_COLUMNS,str(f))
    rows=csv.DictReader(io.StringIO(data.decode('utf-8-sig')))
    datasets=set();variants=set()
    for r in rows:datasets.add(r['dataset']);variants.add(r['variant'])
    if 'absolute_weighted' not in variants:continue
    if datasets=={'TEST','ADORE'}:candidates['daphnia'].append((str(f),data))
    elif datasets=={'FATHEAD'}:candidates['fathead'].append((str(f),data))
  for species in ['daphnia','fathead']:resolved['absolute_'+species]=choose(candidates[species],f'absolute {species} under {adir}',PRED_COLUMNS)
 found=project_files(project,runner_root)
 raw=[(str(f),f.read_bytes()) for f in found if f.name=='neighbor_error_by_chemical_split.csv']
 if raw:resolved['neighbors']=choose(raw,'neighbor CSV',NEIGHBOR_COLUMNS)
 else:
  zipped=[]
  for f in found:
   if f.suffix.lower()!='.zip':continue
   with zipfile.ZipFile(f) as z:
    members=[n for n in z.namelist() if n.split('/')[-1]=='neighbor_error_by_chemical_split.csv']
    for n in members:zipped.append((str(f)+'::'+n,z.read(n)))
  resolved['neighbors']=choose(zipped,'neighbor_error_by_chemical_split.csv within '+str(project),NEIGHBOR_COLUMNS)
 return resolved

def stage_inputs(resolved,directory):
 # Prepare verified source tables for the subgroup analysis.
 directory.mkdir(parents=True)
 spec={'results_for_review.zip':{'all_evaluation_predictions.csv':'daphnia_predictions','all_evaluation_metrics.csv':'daphnia_metrics'},'results_for_review(1).zip':{'all_evaluation_predictions.csv':'fathead_predictions'},'absolute_weight_results_for_review.zip':{'daphnia/all_evaluation_predictions.csv':'absolute_daphnia','fathead/all_evaluation_predictions.csv':'absolute_fathead'},'neighbor_results.zip':{'neighbor_results/neighbor_error_by_chemical_split.csv':'neighbors'}}
 for filename,entries in spec.items():
  with zipfile.ZipFile(directory/filename,'w',zipfile.ZIP_DEFLATED) as z:
   for member,key in entries.items():z.writestr(member,resolved[key]['data'])

def main():
 root=Path(__file__).resolve().parent
 default=DEFAULT_PROJECT
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--project-root',type=Path,default=default)
 parser.add_argument('--check-inputs',action='store_true',help='Locate and validate input headers only; do not analyze data')
 args=parser.parse_args();project=args.project_root.resolve()
 print('Python: '+sys.executable,flush=True);print('Project: '+str(project),flush=True)
 if not project.is_dir():raise InputProblem('Project folder not found: '+str(project))
 required=['prepare_predictions.py','analyze_classes.py','check_robustness.py','protocol.json']
 missing=[n for n in required if not (root/'code'/n).is_file()]
 if missing:raise InputProblem('Required companion code files missing: '+', '.join(missing))
 print('Reading existing project results...',flush=True)
 resolved=resolve_inputs(project,root)
 for key,item in resolved.items():print('  '+key+': '+item['source'],flush=True)
 print('INPUT CHECK PASSED: six source tables found and CSV headers checked.',flush=True)
 if args.check_inputs:
  print('No analysis performed. To analyze, run: python run_analysis.py');return 0
 packages={}
 for name in ['numpy','pandas','scipy','rdkit']:
  try:module=importlib.import_module(name);packages[name]=getattr(module,'__version__','not reported')
  except ImportError as e:raise InputProblem(f'Cannot import {name} in the current Python environment: {e}\nNo environment or package was changed. Share this message.') from e
 out=root/'output'/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f');out.mkdir(parents=True)
 provenance={key:{k:v for k,v in entry.items() if k!='data'} for key,entry in resolved.items()}
 (out/'resolved_sources.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2),encoding='utf-8')
 (out/'environment.json').write_text(json.dumps({'python':sys.version,'executable':sys.executable,'packages':packages,'project_root':str(project),'training_performed':False,'runner_version':'2-project-paths'},indent=2),encoding='utf-8')
 stage_inputs(resolved,out/'_source_adapter')
 src=out/'_source_adapter'
 steps=[('1/3 Rebuild chemical-level errors',[str(root/'code/prepare_predictions.py'),'--sources',str(src),'--output',str(out/'prepared')]),('2/3 Calculate structural subgroup summaries',[str(root/'code/analyze_classes.py'),'--sources',str(src),'--prior',str(out/'prepared'),'--output',str(out/'classes')]),('3/3 Check influence and motif overlap',[str(root/'code/check_robustness.py'),'--output',str(out/'classes')])]
 env=os.environ.copy();env['PYTHONUTF8']='1';env['PYTHONIOENCODING']='utf-8'
 with (out/'run_log.txt').open('w',encoding='utf-8') as log:
  for label,cmd in steps:
   print(label,flush=True);log.write('\n'+label+'\n');log.flush()
   proc=subprocess.run([sys.executable,*cmd],env=env,stdout=log,stderr=subprocess.STDOUT)
   if proc.returncode:
    print('STOPPED. Share this log: '+str(out/'run_log.txt'));return proc.returncode
 (out/'SUCCESS.txt').write_text('All stages completed. No models trained. Scientific interpretation pending.\n',encoding='utf-8')
 target=out/'results_for_review_R15.zip'
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(out.rglob('*')):
   if f.is_file() and f!=target and '_source_adapter' not in f.parts:z.write(f,f.relative_to(out))
  for f in sorted((root/'code').glob('*')):
   if f.is_file():z.write(f,'analysis_code/'+f.name)
  z.write(Path(__file__),'analysis_code/run_analysis.py')
 print('\nCOMPLETE. Share this result file:\n'+str(target));return 0

if __name__=='__main__':
 try:sys.exit(main())
 except InputProblem as e:print('\nINPUT / ENVIRONMENT CHECK STOPPED:\n'+str(e));sys.exit(2)
 except Exception:traceback.print_exc();print('\nAnalysis did not complete. Share the error shown above.');sys.exit(1)
