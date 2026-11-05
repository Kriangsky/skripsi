"""Actual TASK 0 validation recipe. New evidence directory; never runs model actions."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
NOTEBOOK = ROOT/'06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    if (OUT/'validation.json').exists():
        raise FileExistsError('Preserve previous validation evidence; use a new approved record path')
    record = {'started_utc': now(), 'command': [sys.executable, str(Path(__file__).resolve())],
              'cwd': str(ROOT), 'model_fits': 0, 'external_scoring': False, 'freeze': False,
              'notebook_before_sha256': sha(NOTEBOOK), 'tests': []}
    notebook = nbformat.read(NOTEBOOK, 4)
    nbformat.validate(notebook)
    try:
        print('RUN notebook kernel; no model actions', flush=True)
        NotebookClient(notebook, timeout=120, startup_timeout=30, kernel_name='thesis_cuthbert',
                       resources={'metadata': {'path': str(NOTEBOOK.parent)}}).execute()
        record['notebook_execution'] = 'passed'
    except Exception:
        record['notebook_execution'] = 'failed'
        record['notebook_error'] = traceback.format_exc()
    finally:
        # Preserve real kernel outputs, including any failure; never fabricate outputs/counts.
        nbformat.write(notebook, NOTEBOOK)
        record['notebook_completed_utc'] = now()
        record['notebook_after_sha256'] = sha(NOTEBOOK)
        (OUT/'notebook_execution.log').write_text('\n\n'.join(
            f"CELL {cell.execution_count}\n{cell.source}\n" + '\n'.join(
                output.get('text', '\n'.join(output.get('traceback', [])))
                for output in cell.get('outputs', []))
            for cell in notebook.cells if cell.cell_type == 'code'), encoding='utf8')
    env = dict(os.environ, PYTHONPATH=str(ROOT/'05_kode/src'))
    checks = [
        ('notebook_contract', ROOT, ['-m', 'unittest', 'discover', '-s', '05_kode/tests', '-p', 'test_final_control_notebook.py', '-v']),
        ('interference_synthetic', ROOT, ['-m', 'unittest', 'discover', '-s', '05_kode/tests', '-p', 'test_interference_metrics.py', '-v']),
        ('statistics_synthetic', ROOT, ['-m', 'unittest', 'discover', '-s', '05_kode/tests', '-p', 'test_confirmatory_statistics.py', '-v']),
        ('existing_regression', ROOT/'.compat_runtime', ['-m', 'unittest', 'discover', '-v']),
    ]
    for name, cwd, args in checks:
        started = now()
        command = [sys.executable, *args]
        print('RUN', name, flush=True)
        process = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True)
        output = process.stdout + process.stderr
        (OUT/f'{name}.log').write_text(output, encoding='utf8')
        record['tests'].append({'name': name, 'command': command, 'cwd': str(cwd),
                                'started_utc': started, 'completed_utc': now(),
                                'exit_code': process.returncode, 'log': f'{name}.log'})
        print(name, 'exit', process.returncode, output[-250:], flush=True)
    baseline = json.loads((OUT/'baseline.json').read_text(encoding='utf8'))
    record['baseline_drift'] = [name for name, digest in baseline['sha256'].items()
                                if name != '06_notebook/README.md' and sha(ROOT/name) != digest]
    record['intentional_baseline_edits'] = ['06_notebook/README.md']
    record['git'] = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT,
                                    capture_output=True, text=True, check=True).stdout
    record['completed_utc'] = now()
    record['artifacts_sha256'] = {str(p.relative_to(ROOT)): sha(p) for p in OUT.iterdir() if p.is_file()}
    record['new_notebook_test_sha256'] = sha(ROOT/'05_kode/tests/test_final_control_notebook.py')
    # Existing regression failure is disclosed separately; it never becomes a green full-suite claim.
    record['task0_acceptance'] = (record['notebook_execution'] == 'passed'
                                  and not record['baseline_drift']
                                  and all(x['exit_code'] == 0 for x in record['tests'][:3]))
    with (OUT/'validation.json').open('x', encoding='utf8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps({k: record[k] for k in ['notebook_execution', 'baseline_drift', 'task0_acceptance']}), flush=True)
    return 0 if record['task0_acceptance'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
