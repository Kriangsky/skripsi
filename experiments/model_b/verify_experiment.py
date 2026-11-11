"""Read-only acceptance checks; writes one verification report, never trains."""
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import nbformat
import numpy as np
import pandas as pd
import model_b_support as b

base = Path(__file__).resolve().parent
latest = json.loads((base / 'reports/latest_official_run.json').read_text())
run = Path(latest['run_directory'])
manifest = json.loads((run / 'reproducibility/run_manifest.json').read_text())
assert manifest['status'] == 'COMPLETE'
notebook = nbformat.read(base / 'Model_B_YOLO26_8Class_Training.ipynb', as_version=4)
nbformat.validate(notebook)
codes = [c for c in notebook.cells if c.cell_type == 'code']
assert [c.execution_count for c in codes] == list(range(1, 17))
assert not any(o.output_type == 'error' for c in codes for o in c.outputs)
assert sum('image/png' in o.get('data', {}) for c in codes for o in c.outputs) >= 3
assert any('model.train(' in c.source for c in codes)
assert any('best_model.predict(' in c.source for c in codes)
assert (run / 'reproducibility/executed_notebook.ipynb').read_bytes() == (base / 'Model_B_YOLO26_8Class_Training.ipynb').read_bytes()
rows = pd.read_csv(run / 'reproducibility/dataset_manifest.csv')
assert len(rows) == 2400 and rows.sample_id.nunique() == 2400
assert rows.groupby(['class_name', 'split']).size().value_counts().to_dict() == {240: 8, 60: 8}
assert set(rows.source) == {'train'}
assert b.sha(run / 'reproducibility/dataset_manifest.csv') == manifest['dataset_manifest_sha256']
for relative, digest in json.loads((run / 'reproducibility/protected_model_a_hashes.json').read_text()).items():
    assert b.sha(b.ROOT / relative) == digest, relative
for relative, digest in manifest['dataset_audit']['source_hashes'].items():
    assert b.sha(b.ROOT / relative) == digest, relative
trials = json.loads((run / 'trial_history.json').read_text())
assert len(trials) == manifest['completed_trials'] == 8
for trial in trials:
    assert trial['status'] == 'COMPLETE'
    assert 1 <= trial['completed_epochs'] <= trial['configuration']['epochs'] == 100
    assert trial['configuration']['patience'] == 20 and not trial['configuration']['resume']
    assert trial['pretrained_sha256'] == b.a.CONFIG['checkpoint_sha256']
    history = pd.read_csv(run / 'training_history' / (trial['trial'] + '.csv'))
    values = history['metrics/macro_f1'].to_numpy()
    assert len(history) == trial['completed_epochs']
    assert float(values.max()) == trial['macro_f1']
    assert int(np.flatnonzero(values == values.max())[-1]) + 1 == trial['best_epoch']
    assert b.sha(trial['checkpoint']) == trial['checkpoint_sha256']
selected = sorted(trials, key=lambda t: (-t['macro_f1'], t['trial']))[0]
assert selected['trial'] == manifest['selected_trial']['trial']
assert b.sha(run / 'weights/best.pt') == selected['checkpoint_sha256']
for name, digest in manifest['checkpoint_hashes'].items(): assert b.sha(run / 'weights' / name) == digest
pred = pd.read_csv(run / 'predictions/validation_predictions.csv')
assert len(pred) == pred.sample_id.nunique() == 480
assert set(pred.sample_id) == set(rows[rows.split.eq('val')].sample_id)
probs = pred[['p_' + name for name in b.CLASSES]].to_numpy()
assert np.isfinite(probs).all() and (probs >= 0).all() and np.allclose(probs.sum(1), 1, atol=1e-4)
assert np.array_equal(probs.argmax(1), pred.predicted_canonical_id.to_numpy())
for labels, data, key in [(range(8), pred, 'final_metrics8'), (range(6), pred[pred.true_canonical_id.lt(6)], 'final_metrics6')]:
    metrics, _, _ = b.metrics(data.true_canonical_id, data.predicted_canonical_id, labels=labels)
    for metric, value in metrics.items(): assert np.isclose(value, manifest[key][metric])
paired = pd.read_csv(run / 'predictions/paired_model_a_model_b.csv')
assert len(paired) == paired.sample_id.nunique() == 360
assert paired.true_canonical_id_a.eq(paired.true_canonical_id_b).all()
report = dict(status='PASS', run=str(run), executed_code_cells=16, completed_trials=8,
              training_samples=1920, validation_samples=480, shared_samples_exact=1800,
              paired_validation_samples=360, selected_trial=selected['trial'],
              metrics8=manifest['final_metrics8'], metrics6=manifest['final_metrics6'],
              model_a_evidence_unchanged=True)
b.write_json(base / 'reports/acceptance_verification.json', report)
print(json.dumps(report, indent=2))
