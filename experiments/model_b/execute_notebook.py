"""Jupyter execution driver. All training/evaluation remains inside notebook cells."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import nbformat
from nbclient import NotebookClient

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['smoke', 'official'], required=True)
args = parser.parse_args()
os.environ['MODEL_B_MODE'] = args.mode
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['YOLO_CONFIG_DIR'] = str(HERE / 'outputs/ultralytics_config')
kernel_root = HERE / 'outputs/jupyter'
kernel = kernel_root / 'kernels/bisindo-model-b'
kernel.mkdir(parents=True, exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps(dict(argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                                                  display_name='BISINDO Model B', language='python')), encoding='utf8')
os.environ['JUPYTER_PATH'] = str(kernel_root)
os.environ['JUPYTER_RUNTIME_DIR'] = str(HERE / 'outputs/jupyter/runtime')
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(exist_ok=True)
source = HERE / 'Model_B_YOLO26_8Class_Training.ipynb'
reports = HERE / 'reports'
reports.mkdir(exist_ok=True)
destination = source if args.mode == 'official' else reports / 'Model_B_smoke_executed.ipynb'
nb = nbformat.read(source, as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code':
        cell.outputs = []
        cell.execution_count = None
client = NotebookClient(nb, timeout=None, kernel_name='bisindo-model-b', resources={'metadata': {'path': str(ROOT)}})
def save():
    # Windows readers can briefly lock the displayed notebook; replace a fully written sibling.
    temporary = destination.with_suffix('.saving.ipynb')
    for attempt in range(5):
        try:
            nbformat.write(nb, temporary)
            temporary.replace(destination)
            return
        except OSError:
            if attempt == 4: raise
            time.sleep(0.2)
def persist(cell, cell_index, **kwargs):
    save()
    print(f'Cell {cell_index} finished; notebook saved.', flush=True)
client.on_cell_executed = persist
try:
    client.execute()
finally:
    save()
latest = reports / ('latest_smoke_run.json' if args.mode == 'smoke' else 'latest_official_run.json')
run = Path(json.loads(latest.read_text())['run_directory'])
shutil.copyfile(destination, run / 'reproducibility/executed_notebook.ipynb')
print('Executed notebook:', destination, flush=True)
