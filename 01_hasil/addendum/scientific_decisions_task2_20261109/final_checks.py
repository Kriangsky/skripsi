"""Close TASK2 after documentation/serialization; preserve prior validation records."""
from datetime import datetime, timezone
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import subprocess
import sys

import nbformat

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'05_kode/src'))
import confirmatory_evaluation as evaluation


def main():
    if (OUT/'final_verification.json').exists():
        raise FileExistsError('Final evidence exists')
    record=dict(started_utc=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],checks=[])
    notebook='06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb'
    tracker='07_dokumen/TRACKING_BISINDO.md'
    # Retain HEAD line bytes when unchanged; added lines use LF under this repo's * -text attributes.
    for name in [notebook,tracker]:
        path=ROOT/name
        head=subprocess.check_output(['git','show',f'HEAD:{name}'],cwd=ROOT).splitlines(keepends=True)
        current=path.read_bytes().replace(b'\r\n',b'\n').splitlines(keepends=True)
        result=[]
        matcher=SequenceMatcher(None,[x.rstrip(b'\r\n') for x in head],
                                [x.rstrip(b'\r\n') for x in current],autojunk=False)
        for tag,a,b,c,d in matcher.get_opcodes():
            result.extend(head[a:b] if tag=='equal' else current[c:d])
        path.write_bytes(b''.join(result))
    old=nbformat.from_dict(evaluation.read(OUT/'notebook_before.json'))
    nb=nbformat.read(ROOT/notebook,4);nbformat.validate(nb)
    lookup={c.id:c for c in nb.cells}
    count=0
    for cell in old.cells:
        if cell.cell_type=='code':
            count+=1
            for key in ['source','outputs','execution_count']:
                assert cell[key]==lookup[cell.id][key]
    new=[c for c in nb.cells if any(t.startswith('task2-') for t in c.metadata.get('tags',[]))]
    assert len(new)==2 and all(c.execution_count and c.outputs for c in new)
    assert not any(o.output_type=='error' for c in new for o in c.outputs)
    prior=evaluation.read(OUT/'validation.json')
    assert prior['acceptance'] and prior['task2_status']=='PASS'
    assert all(c['exit_code']==0 for c in prior['tests'])
    roots=ROOT/'02_exp/log_sesi/exploratory_development_stage1b_user_20261103_v1_session'
    approvals=[]
    for seed in range(42,47):
        approval=evaluation.read(roots/f'exploratory_approval_seed{seed}.json')
        assert approval['scope']=='stage1b' and approval['training']==dict(seed=seed,epochs=30,imgsz=224)
        assert approval['approval_date']=='2026-11-03'
        approvals.append(dict(seed=seed,sha256=evaluation.sha(roots/f'exploratory_approval_seed{seed}.json')))
    import pandas as pd
    fits=pd.read_csv(ROOT/'01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_all_fits.csv')
    full=fits[fits.kind.eq('full')]
    assert len(full)==10 and full.status.eq('completed').all()
    assert set(zip(full.seed,full.model))=={(s,a) for s in range(42,47) for a in ['A','B']}
    baseline=evaluation.read(OUT/'protected_before.json')
    drift=[name for name,digest in baseline.items()
           if Path(name).as_posix() not in {notebook,tracker} and evaluation.sha(ROOT/name)!=digest]
    assert not drift,drift
    assert (ROOT/tracker).read_text(encoding='utf8').startswith(evaluation.read(OUT/'tracker_before.json')['text'])
    env=dict(os.environ,PYTHONPATH=str(ROOT/'05_kode/src'))
    commands=[
        [sys.executable,'-m','unittest','discover','-s','05_kode/tests','-p','test_task2_configuration.py','-v'],
        [sys.executable,'-m','unittest','discover','-s','05_kode/tests','-p','test_final_control_notebook.py','-v'],
        [sys.executable,'05_kode/check_runtime.py',str(ROOT)],['git','diff','--check']]
    for index,command in enumerate(commands):
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
        log=f'completion_check_{index}.log'
        with (OUT/log).open('x',encoding='utf8') as stream:
            stream.write(result.stdout+result.stderr)
        record['checks'].append(dict(command=command,exit_code=result.returncode,log=log))
        assert result.returncode==0,result.stdout+result.stderr
    for p in list(OUT.glob('*.py'))+[ROOT/'05_kode/tests/test_task2_configuration.py']:
        compile(p.read_text(encoding='utf8'),str(p),'exec')
    record.update(task2_status='PASS',final_readiness='BLOCKED',historical_code_cells_preserved=count,
        task2_outputs_executed=2,protected_files_checked=len(baseline),unexpected_protected_drift=drift,
        seed_approval_records_verified=approvals,all_ten_full_fits_retained=True,tracker_append_only=True,
        scientific_decisions_approved=False,task3_started=False,
        no_training_scoring_or_freeze=True,no_data_or_split_modification=True,
        prior_validation='validation.json; 25/25 tests; full safe notebook execution PASS',
        notebook_sha256=evaluation.sha(ROOT/notebook),tracker_sha256=evaluation.sha(ROOT/tracker),
        completed_utc=datetime.now(timezone.utc).isoformat())
    record['changed_file_inventory_task2']=[notebook,tracker,'05_kode/tests/test_task2_configuration.py']+[
        str(p.relative_to(ROOT)) for p in sorted(OUT.rglob('*')) if p.is_file() and '__pycache__' not in p.parts]+[
        str((OUT/'final_verification.json').relative_to(ROOT))]
    record['artifacts_sha256']={str(p.relative_to(ROOT)):evaluation.sha(p) for p in sorted(OUT.rglob('*'))
                              if p.is_file() and '__pycache__' not in p.parts}
    record['new_test_sha256']=evaluation.sha(ROOT/'05_kode/tests/test_task2_configuration.py')
    evaluation.write_new(OUT/'final_verification.json',record)
    print('TASK2 PASS; 25/25 original targeted tests; 11/11 completion rechecks; protected drift0; STOP before TASK3')


if __name__=='__main__':
    main()
