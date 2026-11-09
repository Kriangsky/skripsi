"""Verify final documentation pass without replacing scientific or historical outputs."""
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import nbformat

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'05_kode/src'))
import confirmatory_evaluation as evaluation


def main():
    if (OUT/'final_verification.json').exists():
        raise FileExistsError('Final evidence exists; refusing overwrite')
    record=dict(command=[sys.executable,*sys.argv],started_utc=datetime.now(timezone.utc).isoformat(),checks=[])
    path=ROOT/'06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb'
    notebook=nbformat.read(path,4)
    nbformat.validate(notebook)
    previous=evaluation.read(OUT/'notebook_outputs_before.json')
    old_cells=[c for c in notebook.cells if c.id in previous]
    assert len(old_cells)==12
    for cell in old_cells:
        assert dict(source=cell.source,outputs=cell.outputs,execution_count=cell.execution_count)==previous[cell.id]
    findings=next(c for c in notebook.cells if 'task1-findings' in c.metadata.get('tags',[]))
    assert findings.source in (OUT/'notebook_execution_verified.log').read_text(encoding='utf8')
    assert findings.execution_count is not None and findings.outputs
    assert not any(o.output_type=='error' for o in findings.outputs)
    prior=evaluation.read(OUT/'validation.json')
    assert prior['acceptance'] and all(t['exit_code']==0 for t in prior['tests'])
    audit=evaluation.read(OUT/'audit.json')
    assert audit['technical_audit_passed'] and audit['observed']['physical_verified_files']==2800
    assert len(audit['internal_pairing'])==5 and audit['across_seed_manifest_identity']
    assert not audit['video_end_hash_mismatches'] and not audit['metadata_end_hash_mismatches']
    protected=evaluation.read(OUT/'protected_before.json')
    drift=[name for name,digest in protected.items()
           if name!=str(path.relative_to(ROOT)) and evaluation.sha(ROOT/name)!=digest]
    assert not drift
    env=dict(os.environ,PYTHONPATH=str(ROOT/'05_kode/src'))
    commands=[
        [sys.executable,'-m','unittest','discover','-s','05_kode/tests','-p','test_final_control_notebook.py','-v'],
        [sys.executable,'05_kode/check_runtime.py',str(ROOT)],
        ['git','diff','--check'],
    ]
    for index,command in enumerate(commands):
        process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=60)
        with (OUT/f'final_check_{index}.log').open('x',encoding='utf8') as stream:
            stream.write(process.stdout+process.stderr)
        record['checks'].append(dict(command=command,exit_code=process.returncode,log=f'final_check_{index}.log'))
        assert process.returncode==0,process.stdout+process.stderr
    record.update(status='PASS TASK 1 audit; final external readiness BLOCKED',previous_code_cells_preserved=12,
                  new_task1_code_cell_executed=True,unexpected_protected_drift=drift,
                  notebook_sha256=evaluation.sha(path),completed_utc=datetime.now(timezone.utc).isoformat(),
                  documentation_only_since_kernel_validation=True,training=False,external_scoring=False,task2_started=False)
    record['artifacts_sha256']={p.name:evaluation.sha(p) for p in OUT.iterdir() if p.is_file()}
    record['source_sha256']={str(p.relative_to(ROOT)):evaluation.sha(p) for p in
                           [ROOT/'05_kode/src/task1_data_audit.py',ROOT/'05_kode/tests/test_task1_data_audit.py',path,ROOT/'07_dokumen/TRACKING_BISINDO.md']}
    evaluation.write_new(OUT/'final_verification.json',record)
    print(record['status'])
    print('Preserved 12 original code sources/outputs/counts; current notebook/schema/compatibility/diff checks passed')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
