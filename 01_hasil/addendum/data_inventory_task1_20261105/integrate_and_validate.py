"""Append verified TASK 1 findings, preserving every existing executed code cell output."""
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

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NOTEBOOK = ROOT/'06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb'
sys.path.insert(0,str(ROOT/'05_kode/src'))
import confirmatory_evaluation as evaluation


def main():
    if (OUT/'validation.json').exists():
        raise FileExistsError('Preserve completed validation; use a new record for subsequent work')
    record=dict(command=[sys.executable,*sys.argv],started_utc=datetime.now(timezone.utc).isoformat(),tests=[])
    notebook=nbformat.read(NOTEBOOK,4)
    if any('task1-findings' in c.metadata.get('tags',[]) for c in notebook.cells):
        raise ValueError('TASK 1 integration already exists; no duplicate insertion')
    previous={c.id:dict(source=c.source,outputs=c.outputs,execution_count=c.execution_count)
              for c in notebook.cells if c.cell_type=='code'}
    if (OUT/'notebook_outputs_before.json').exists():
        assert evaluation.read(OUT/'notebook_outputs_before.json') == previous
    else:
        evaluation.write_new(OUT/'notebook_outputs_before.json',previous)
    index=next(i for i,c in enumerate(notebook.cells) if c.cell_type=='markdown' and c.source.startswith('## 3.'))
    notebook.cells[index].source += '\n\n**TASK 1:** technical audit VERIFIED; final readiness BLOCKED. Historical TASK 0 outputs below remain unchanged. Current findings follow in a separate cell.'
    insert=index+2
    description=nbformat.v4.new_markdown_cell('''### TASK 1 — hasil audit file aktual

**Status: VERIFIED technical audit / BLOCKED final readiness.** Input: immutable `audit.json` dan `provenance.json` hasil audit read-only.
Output: expected/observed counts, pairing lima seed, exact duplicate findings dan provenance evidence table.
Penerimaan: file physical/hashes diperiksa; unknown dinyatakan UNVERIFIED; tidak training, scoring, selection, atau perubahan split.
Risiko: exact hashes tidak membuktikan signer/semantic independence; near-duplicate dan all-frame raw video comparison belum dilakukan.
Sel ini hanya membaca evidence; tidak menjalankan ulang auditor, model, atau frame extraction.

[Laporan TASK 1](../../01_hasil/addendum/data_inventory_task1_20261105/audit_report.md) ·
[Audit JSON](../../01_hasil/addendum/data_inventory_task1_20261105/audit.json) ·
[Provenance](../../01_hasil/addendum/data_inventory_task1_20261105/provenance.json).
Outputs sel lama adalah catatan eksekusi TASK 0, tidak diganti dengan hasil eksekusi baru.
''')
    findings=nbformat.v4.new_code_cell('''TASK1_DIRECTORY = ROOT/'01_hasil/addendum/data_inventory_task1_20261105'
TASK1_AUDIT = evaluation.read(TASK1_DIRECTORY/'audit.json')
TASK1_PROVENANCE = evaluation.read(TASK1_DIRECTORY/'provenance.json')
TASK1_DUPLICATES = evaluation.read(TASK1_DIRECTORY/'duplicate_impact.json')
assert TASK1_AUDIT['schema_version'] == 1 and TASK1_AUDIT['task'] == 1
assert TASK1_AUDIT['technical_audit_passed'] is True
assert TASK1_AUDIT['class_mapping'] == ALL_CLASSES
assert [row['seed'] for row in TASK1_AUDIT['internal_pairing']] == SEEDS
print(json.dumps({'task': 1, 'status': 'VERIFIED read-only audit; external readiness BLOCKED',
                  'audit_completed_utc': TASK1_AUDIT['completed_utc'],
                  'expected': TASK1_AUDIT['expected'], 'observed': TASK1_AUDIT['observed'],
                  'class_split_counts': TASK1_AUDIT['class_split_counts'],
                  'pairing': TASK1_AUDIT['internal_pairing'],
                  'frame_provenance': TASK1_AUDIT['frame_provenance'],
                  'actual_video_duplicate_groups': len(TASK1_AUDIT['duplicates']['actual_video_groups']),
                  'decoded_rgb_duplicate_groups': len(TASK1_DUPLICATES['groups']),
                  'cross_split_groups': TASK1_DUPLICATES['cross_split_groups'],
                  'duplicate_interpretation': TASK1_DUPLICATES['interpretation'],
                  'near_duplicate_status': TASK1_AUDIT['duplicates']['near_duplicates'],
                  'final_manifests': TASK1_AUDIT['final_manifest_paths'],
                  'provenance': TASK1_PROVENANCE['evidence_table'],
                  'audit_sha256': evaluation.sha(TASK1_DIRECTORY/'audit.json'),
                  'provenance_sha256': evaluation.sha(TASK1_DIRECTORY/'provenance.json')}, indent=2))
''',metadata={'tags':['task1-findings']})
    notebook.cells[insert:insert]=[description,findings]
    notebook.metadata.bisindo.update(task=1,completed_tasks=[0,1],actions_enabled=False)
    nbformat.validate(notebook)
    # Execute a separate in-memory copy. Persist only the newly added cell outputs.
    executed=nbformat.reads(nbformat.writes(notebook),as_version=4)
    try:
        print('RUN complete safe notebook in project kernel; preserve saved TASK 0 outputs',flush=True)
        NotebookClient(executed,timeout=120,startup_timeout=30,kernel_name='thesis_cuthbert',
                       resources={'metadata':{'path':str(NOTEBOOK.parent)}}).execute()
        actual=next(c for c in executed.cells if 'task1-findings' in c.metadata.get('tags',[]))
        findings.outputs=actual.outputs
        findings.execution_count=actual.execution_count
        record['notebook_execution']='passed'
    except Exception:
        record['notebook_execution']='failed'
        record['notebook_error']=traceback.format_exc()
    with (OUT/'notebook_execution_verified.log').open('x',encoding='utf8') as stream:
        for cell in executed.cells:
            if cell.cell_type=='code':
                stream.write(f'CELL {cell.execution_count}\n{cell.source}\n')
                for output in cell.outputs:
                    stream.write(output.get('text','\n'.join(output.get('traceback',[])))+'\n')
    if record['notebook_execution']=='passed':
        for cell in notebook.cells:
            if cell.id in previous:
                assert dict(source=cell.source,outputs=cell.outputs,execution_count=cell.execution_count)==previous[cell.id]
        nbformat.write(notebook,NOTEBOOK)
        record['previous_code_cells_preserved']=len(previous)
        record['new_task1_cells_executed']=1
    env=dict(os.environ,PYTHONPATH=str(ROOT/'05_kode/src'),
             TEMP=str(Path(tempfile.gettempdir()).resolve()),TMP=str(Path(tempfile.gettempdir()).resolve()))
    for pattern in ['test_task1_data_audit.py','test_final_control_notebook.py','test_frame_candidates.py',
                    'test_interference_metrics.py','test_confirmatory_statistics.py']:
        command=[sys.executable,'-m','unittest','discover','-s',str(ROOT/'05_kode/tests'),'-p',pattern,'-v']
        start=datetime.now(timezone.utc).isoformat()
        process=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
        log=pattern.replace('.py','.log')
        with (OUT/log).open('x',encoding='utf8') as stream:
            stream.write(process.stdout+process.stderr)
        record['tests'].append(dict(command=command,started_utc=start,completed_utc=datetime.now(timezone.utc).isoformat(),exit_code=process.returncode,log=log))
        print(pattern,process.returncode,(process.stdout+process.stderr)[-150:],flush=True)
    baseline=evaluation.read(OUT/'protected_before.json')
    record['unexpected_protected_drift']=[name for name,digest in baseline.items()
                                          if name!=str(NOTEBOOK.relative_to(ROOT)) and evaluation.sha(ROOT/name)!=digest]
    record['audit_source_still_matches']=all(evaluation.sha(ROOT/'05_kode/src'/name)==digest
                                             for name,digest in evaluation.read(OUT/'audit.json')['source_sha256'].items())
    record['acceptance']=(record['notebook_execution']=='passed' and not record['unexpected_protected_drift']
                          and record['audit_source_still_matches'] and all(t['exit_code']==0 for t in record['tests']))
    record['completed_utc']=datetime.now(timezone.utc).isoformat()
    record['notebook_sha256']=evaluation.sha(NOTEBOOK)
    record['artifacts_sha256']={p.name:evaluation.sha(p) for p in OUT.iterdir() if p.is_file()}
    evaluation.write_new(OUT/'validation.json',record)
    print(json.dumps({k:record[k] for k in ['notebook_execution','unexpected_protected_drift','acceptance']}),flush=True)
    return 0 if record['acceptance'] else 1


if __name__=='__main__':
    raise SystemExit(main())
