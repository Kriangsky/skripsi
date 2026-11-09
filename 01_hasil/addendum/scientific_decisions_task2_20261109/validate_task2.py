"""Validate proposals and execute a safe notebook copy, preserving historical outputs."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback

import nbformat
from nbclient import NotebookClient

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
NOTEBOOK=ROOT/'06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb'
sys.path.insert(0,str(ROOT/'05_kode/src'))
import confirmatory_evaluation as evaluation


def main():
    record=dict(started_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],tests=[])
    if (OUT/'validation.json').exists():
        raise FileExistsError('Preserve completed validation; use new evidence for follow-up')
    current=nbformat.read(NOTEBOOK,4)
    before=nbformat.from_dict(evaluation.read(OUT/'notebook_before.json'))
    historical={c.id:c for c in before.cells if c.cell_type=='code'}
    current.metadata.bisindo.update(task=2,completed_tasks=[0,1,2],actions_enabled=False,
                                    scientific_decisions='PROPOSED_NOT_APPROVED',final_readiness='BLOCKED')
    executed=nbformat.reads(nbformat.writes(current),4)
    try:
        print('Executing all safe notebook cells in project kernel; persisting TASK2 outputs only',flush=True)
        NotebookClient(executed,timeout=60,startup_timeout=30,kernel_name='thesis_cuthbert',
                       resources={'metadata':{'path':str(NOTEBOOK.parent)}}).execute()
        by_id={c.id:c for c in executed.cells}
        for cell in current.cells:
            if any(t.startswith('task2-') for t in cell.metadata.get('tags',[])):
                cell.outputs=by_id[cell.id].outputs
                cell.execution_count=by_id[cell.id].execution_count
                assert cell.execution_count is not None and cell.outputs
            if cell.id in historical:
                for key in ['source','outputs','execution_count']:
                    assert cell[key]==historical[cell.id][key]
        nbformat.validate(current)
        with NOTEBOOK.open('w',encoding='utf8',newline='\n') as stream:
            nbformat.write(current,stream)
        record.update(notebook_execution='PASS',historical_code_cells_preserved=len(historical),new_task2_code_cells=2)
    except Exception:
        record.update(notebook_execution='FAIL',notebook_error=traceback.format_exc())
    with (OUT/'notebook_execution.json').open('x',encoding='utf8') as stream:
        json.dump([dict(id=c.id,execution_count=c.execution_count,outputs=c.outputs)
                   for c in executed.cells if c.cell_type=='code'],stream,indent=2,ensure_ascii=False)
    print('Notebook',record['notebook_execution'],flush=True)
    env=dict(os.environ,PYTHONPATH=str(ROOT/'05_kode/src'),
             TEMP=str(Path(tempfile.gettempdir()).resolve()),TMP=str(Path(tempfile.gettempdir()).resolve()))
    for pattern in ['test_task2_configuration.py','test_final_control_notebook.py',
                    'test_confirmatory_statistics.py','test_interference_metrics.py']:
        command=[sys.executable,'-m','unittest','discover','-s',str(ROOT/'05_kode/tests'),'-p',pattern,'-v']
        start=datetime.now(timezone.utc).isoformat()
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
        log=pattern.replace('.py','.log')
        with (OUT/log).open('x',encoding='utf8') as stream:
            stream.write(result.stdout+result.stderr)
        record['tests'].append(dict(command=command,started_utc=start,exit_code=result.returncode,log=log))
        print(pattern,result.returncode,(result.stdout+result.stderr)[-130:],flush=True)
    baseline=evaluation.read(OUT/'protected_before.json')
    allowed={str(NOTEBOOK.relative_to(ROOT)),str(Path('07_dokumen/TRACKING_BISINDO.md'))}
    record['protected_files_checked']=len(baseline)
    record['unexpected_protected_drift']=[name for name,digest in baseline.items()
        if name not in allowed and evaluation.sha(ROOT/name)!=digest]
    record['tracker_append_only']=(ROOT/'07_dokumen/TRACKING_BISINDO.md').read_text(encoding='utf8').startswith(
        evaluation.read(OUT/'tracker_before.json')['text'])
    mismatches=[]
    for seed in range(42,47):
        path=ROOT/f'02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed{seed}/research_config.json'
        for name,digest in evaluation.read(path)['source_sha256'].items():
            if evaluation.sha(ROOT/'05_kode/src'/name)!=digest:
                mismatches.append(dict(seed=seed,source=name))
    record['historical_project_source_mismatches']=mismatches
    candidate=evaluation.read(OUT/'candidate_config.json')
    record['initial_checkpoint_hash_matches_pin']=evaluation.sha(ROOT/candidate['initial_checkpoint']['path'])==candidate['initial_checkpoint']['sha256']
    commands=[[sys.executable,str(ROOT/'05_kode/check_runtime.py'),str(ROOT)],['git','diff','--check']]
    for index,command in enumerate(commands):
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
        log=f'final_check_{index}.log'
        with (OUT/log).open('x',encoding='utf8') as stream:
            stream.write(result.stdout+result.stderr)
        record['tests'].append(dict(command=command,exit_code=result.returncode,log=log))
    result=subprocess.run(['git','status','--short','--untracked-files=all'],cwd=ROOT,capture_output=True,text=True)
    record['workspace_git_status_includes_preexisting_task1_changes']=result.stdout.splitlines()
    record['changed_file_inventory_task2']=[str(NOTEBOOK.relative_to(ROOT)),
        '07_dokumen/TRACKING_BISINDO.md','05_kode/tests/test_task2_configuration.py']+[
        str(p.relative_to(ROOT)) for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    record['actions']=dict(training=0,model_forward=0,kaggle_inference=0,external_scoring=0,freeze=0,
                           data_or_split_changes=0,protocol_amendment=0,task3_started=False)
    record['acceptance']=(record['notebook_execution']=='PASS' and not record['unexpected_protected_drift']
        and not mismatches and record['initial_checkpoint_hash_matches_pin'] and record['tracker_append_only']
        and all(t['exit_code']==0 for t in record['tests']))
    record['task2_status']='PASS' if record['acceptance'] else 'FAIL'
    record['final_readiness']='BLOCKED'
    record['completed_utc']=datetime.now(timezone.utc).isoformat()
    record['notebook_sha256']=evaluation.sha(NOTEBOOK)
    record['artifacts_sha256']={p.name:evaluation.sha(p) for p in OUT.iterdir() if p.is_file()}
    evaluation.write_new(OUT/'validation.json',record)
    print(json.dumps({k:record[k] for k in ['task2_status','final_readiness','protected_files_checked',
        'unexpected_protected_drift','historical_project_source_mismatches','acceptance']}),flush=True)
    return 0 if record['acceptance'] else 1


if __name__=='__main__':
    raise SystemExit(main())
