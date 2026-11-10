"""Read saved artifacts only. Preserve originals; never start a kernel or import training code."""
import ast
import base64
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import re

import nbformat
from nbconvert import HTMLExporter

WORK = Path(__file__).resolve().parent
ROOT = WORK.parents[1]
REVISION = WORK / 'reports/automatic_selection_revision'
OUT = WORK / 'reports/notebook_output_audit'
SOURCE = WORK / 'Model_A_YOLO26_6Class_Training.ipynb'
EXECUTED = REVISION / 'Model_A_completed_results_executed.ipynb'
REVISED = WORK / 'Model_A_YOLO26_6Class_Completed_Evidence.ipynb'


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf8'))


def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig', newline='')))


def relative(path):
    return path.relative_to(ROOT).as_posix()


def link(path):
    return f'[{relative(path)}](<{path.relative_to(WORK).as_posix()}>)'


def table(records, columns=None):
    columns = columns or list(records[0])
    escape = lambda value: str(value).replace('|', '\\|').replace('\n', '<br>')
    return ('| ' + ' | '.join(columns) + ' |\n| ' + ' | '.join(['---'] * len(columns)) + ' |\n' +
            ''.join('| ' + ' | '.join(escape(row[key]) for key in columns) + ' |\n' for row in records))


def statistics(path):
    notebook = nbformat.read(path, as_version=4)
    code = [c for c in notebook.cells if c.cell_type == 'code']
    return dict(path=relative(path), sha256=sha(path), total_cells=len(notebook.cells), code_cells=len(code),
                markdown_cells=sum(c.cell_type == 'markdown' for c in notebook.cells),
                executed_code_cells=sum(c.execution_count is not None for c in code),
                output_blocks=sum(len(c.outputs) for c in code),
                error_outputs=sum(o.output_type == 'error' for c in code for o in c.outputs),
                cells=[dict(index=i, id=c.id, execution_count=c.get('execution_count'),
                            output_blocks=len(c.get('outputs', [])), type=c.cell_type)
                       for i, c in enumerate(notebook.cells)])


def main():
    assert not REVISED.exists() and not OUT.exists(), 'Preserve prior audit outputs; choose a new revision instead of overwriting'
    run = Path(read_json(WORK / 'reports/latest_official_run.json')['run_directory']).resolve()
    assert run.is_relative_to(WORK / 'outputs/official_runs')
    source = nbformat.read(SOURCE, as_version=4)
    executed = nbformat.read(EXECUTED, as_version=4)
    manifest = read_json(run / 'run_manifest.json')
    metrics = read_json(run / 'validation_metrics.json')
    assert manifest['status'] == 'COMPLETE' and manifest['completed_epochs'] == 65 and manifest['best_epoch'] == 46
    assert manifest['early_stopper_best_epoch'] == 45
    assert sha(SOURCE) == manifest['notebook_sha256'] == sha(run / 'source_snapshot' / SOURCE.name)
    assert [(c.id, c.cell_type, c.source) for c in source.cells] == [(c.id, c.cell_type, c.source) for c in executed.cells]
    assert all(c.execution_count is None and not c.outputs for c in source.cells if c.cell_type == 'code')
    assert [c.execution_count for c in executed.cells if c.cell_type == 'code'] == list(range(1, 16))
    assert not any(o.output_type == 'error' for c in executed.cells if c.cell_type == 'code' for o in c.outputs)
    for index, name in [(24, 'validation_metrics.json'), (28, 'run_manifest.json')]:
        displays = [o.data['text/plain'] for o in executed.cells[index].outputs if 'text/plain' in o.get('data', {})]
        assert len(displays) == 1 and ast.literal_eval(displays[0]) == read_json(run / name)
    assert manifest['metrics'] == {key: metrics[key] for key in manifest['metrics']}
    for name, digest in manifest['final_checkpoints'].items():
        assert sha(run / name) == digest, f'Checkpoint differs: {name}'
    assert any(base64.b64decode(o.data['image/png']) == (run / 'confusion_matrix.png').read_bytes()
               for o in executed.cells[26].outputs if 'image/png' in o.get('data', {}))
    history = rows(run / 'training_history.csv')
    native_history = rows(run / 'fit/results.csv')
    assert len(history) == len(native_history) == 65 and history[-1]['epoch'] == '65'
    assert [row['epoch'] for row in history] == [row['epoch'] for row in native_history]
    for key in manifest['metrics']:
        assert float(rows(run / 'validation_metrics.csv')[0][key]) == manifest['metrics'][key]
    assert len(rows(run / 'validation_predictions.csv')) == metrics['samples'] == 360
    per_class = rows(run / 'per_class_metrics.csv')
    for row, saved in zip(per_class, metrics['per_class']):
        assert row['class_name'] == saved['class_name']
        assert all(float(row[k]) == saved[k] for k in ['precision', 'recall', 'f1', 'support'])
    log = (run / 'training.log').read_text(encoding='utf8')
    assert '65 epochs completed' in log and 'last 20 epochs' in log and 'epoch 45' in log
    guarded = [SOURCE, *REVISION.glob('*.ipynb'),
               REVISION / 'official_training_execution.log', REVISION / 'notebook_completed_verification.json',
               WORK / 'reports/latest_official_run.json', *[p for p in run.rglob('*') if p.is_file()]]
    protected = {relative(p): sha(p) for p in guarded}
    OUT.mkdir()
    inspections = [statistics(p) for p in [SOURCE, EXECUTED, REVISION / 'Model_A_pretraining_executed.ipynb']]
    (OUT / 'original_notebook_inventory.json').write_text(json.dumps(inspections, indent=2), encoding='utf8')
    new_cells, presentation_sources = [], []

    def markdown(text):
        new_cells.append(nbformat.v4.new_markdown_cell(text))

    def artifact(path, title, kind='text', columns=None, limit=None):
        presentation_sources.append(dict(path=relative(path), sha256=sha(path), display=title))
        heading = f'## {title}\n\n**Reconstructed artifact display — not an original execution trace.**\n\nSource: {link(path)}\n\nSHA256: `{sha(path)}`.\n\n'
        if kind == 'image':
            name = path.name
            cell = nbformat.v4.new_markdown_cell(heading + f'![{title}](attachment:{name})')
            cell.attachments = {name: {'image/png': base64.b64encode(path.read_bytes()).decode('ascii')}}
            new_cells.append(cell)
        elif kind == 'csv':
            data = rows(path)
            note = f'Preview: first {min(limit, len(data))} of {len(data)} rows, original order; selected columns.\n\n' if limit else ''
            markdown(heading + note + table(data[:limit] if limit else data, columns))
        else:
            text = path.read_text(encoding='utf8')
            if kind == 'json':
                text = json.dumps(json.loads(text), indent=2)
            if kind == 'log':
                text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
                heading += 'Full saved LOGGER text below; ANSI display controls removed only. Original log unchanged.\n\n'
            markdown(heading + f'```{("json" if kind == "json" else "text")}\n{text}\n```')

    markdown(f'''# Completed Model A — preserved evidence and artifact presentation

**Evidence copy. Open to read saved outputs; do not Run All for this audit. No training or kernel execution was performed to create this copy.**

Run: `{run.name}`. Completed **65 epochs**, early stopping patience20. Saved best checkpoint **epoch46**, last tied maximum; first strict maximum **epoch45**, which controls patience.

| Saved validation metric | Saved value |
|---|---:|
|Top1 accuracy|{metrics['top1_accuracy']}|
|Macro precision|{metrics['macro_precision']}|
|Macro recall|{metrics['macro_recall']}|
|Macro-F1|{metrics['macro_f1']}|
|Validation samples|{metrics['samples']}|

Values copied from the existing metrics JSON, not recomputed. Best checkpoint: {link(run / 'fit/weights/best.pt')}. SHA256 `{sha(run / 'fit/weights/best.pt')}`.

**Why reopening the source showed no results:** the source notebook is an unexecuted template (15 code cells, null counts, zero outputs). The 31-cell count includes16 Markdown cells. A separate completed-results notebook saved15 executed code cells and33 output blocks, including authentic post-training result displays. Its training flags were False; the actual fit ran externally and its original console output resides in log files.

**Evidence types below:** Part A contains static artifact displays reconstructed during this audit, each labelled with source and current SHA256. Part B preserves all31 cells of the saved completed-results notebook unchanged, including15 execution counts, execution timing metadata and33 output blocks. Those are original saved **post-training replay traces**, not a Jupyter trace of the65-epoch fit.

Original source: {link(SOURCE)}. Original executed evidence: {link(EXECUTED)}. Neither was overwritten. The revised copy's extra Markdown displays have no synthetic execution counts or outputs.

Limits: original in-notebook training-cell/IOPub traces cannot be restored because the fit did not run in a notebook cell. The external training logs, history, checkpoint and evaluation artifacts exist. Training-time interactive progress and kernel state cannot be recreated as historical notebook execution. New checksums describe current preserved evidence, not independently timestamped creation receipts.

Scientific limits remain: video-derived labels with unresolved historical semantic concerns; validation is not signer-independent; no external-generalization or real-time claim.

# Part A — reconstructed displays of authentic saved artifacts''')
    artifact(run / 'run_manifest.json', 'Run identity, stopping and checkpoint hashes', 'json')
    artifact(run / 'training.log', 'Saved training log', 'log')
    markdown('The separate full redirected console stream, including batch progress, remains at ' +
             link(REVISION / 'official_training_execution.log') + '. It was not inserted into a synthetic code-cell stream.')
    artifact(run / 'fit/results.png', 'Existing Ultralytics training-curves image', 'image')
    artifact(run / 'training_history.csv', 'Complete saved 65-epoch training history', 'csv')
    artifact(run / 'validation_metrics.json', 'Saved internal-validation metrics', 'json')
    artifact(run / 'per_class_metrics.csv', 'Saved per-class precision, recall, F1 and support', 'csv')
    artifact(run / 'confusion_matrix.png', 'Saved canonical-order confusion matrix', 'image')
    artifact(run / 'confusion_matrix.csv', 'Saved canonical-order confusion values', 'csv')
    artifact(run / 'validation_errors.csv', 'All saved validation error records', 'csv',
             ['sample_id', 'video', 'true_class', 'predicted_class', 'model_output_index', 'confidence', 'split'])
    artifact(run / 'validation_predictions.csv', 'Saved prediction CSV preview', 'csv',
             ['sample_id', 'true_class', 'true_canonical_id', 'predicted_class', 'predicted_canonical_id', 'model_output_index', 'confidence', 'split'], 10)
    artifact(run / 'label_mappings.json', 'Canonical and actual output-index mappings', 'json')
    artifact(run / 'fit/actual_training_config.json', 'Actual saved training configuration', 'json')
    artifact(run / 'fit/six_class_architecture.txt', 'Actual saved six-class architecture')
    markdown('# Part B — unchanged original post-training replay evidence\n\nAll31 original cells follow byte-equivalently as notebook cell objects. Existing execution counts and timestamps refer to that saved replay, not this audit. The source contains pipeline code; view its saved outputs without executing it.')
    offset = len(new_cells)
    revised = copy.deepcopy(executed)
    revised.cells = new_cells + copy.deepcopy(executed.cells)
    revised.metadata['model_a_evidence_audit'] = dict(original_executed_notebook=relative(EXECUTED),
        original_executed_sha256=sha(EXECUTED), original_source_sha256=sha(SOURCE),
        preserved_cells_start_index=offset, reconstructed_display_cells=offset, audit_kernel_executed=False,
        training_rerun=False, metrics_recalculated=False)
    nbformat.validate(revised)
    nbformat.write(revised, REVISED)
    reopened = nbformat.read(REVISED, as_version=4)
    assert [dict(c) for c in reopened.cells[offset:]] == [dict(c) for c in executed.cells]
    assert all(c.cell_type == 'markdown' and 'execution_count' not in c and 'outputs' not in c for c in reopened.cells[:offset])
    assert sum(len(c.get('outputs', [])) for c in reopened.cells) == 33
    html, _ = HTMLExporter().from_notebook_node(reopened)
    assert 'Reconstructed artifact display' in html and 'Part B' in html
    (OUT / 'Model_A_completed_evidence.html').write_text(html, encoding='utf8')
    assert all(sha(ROOT / name) == digest for name, digest in protected.items()), 'Original evidence changed'
    audit = dict(status='PASS', cause='Wrong notebook opened: clean source template versus separately saved executed evidence; actual training ran externally',
        original_notebooks=inspections, run_directory=relative(run), checkpoint_epoch=46, completed_epochs=65,
        checkpoint_sha256=sha(run / 'fit/weights/best.pt'), original_cells_preserved=31,
        genuine_saved_code_execution_counts=list(range(1, 16)), genuine_saved_output_blocks=33,
        saved_metric_and_run_manifest_displays_match_artifacts=True, embedded_confusion_image_matches_artifact=True,
        original_input_hashes_before_and_after=protected, reconstructed_artifact_displays=presentation_sources,
        revised_notebook=relative(REVISED), revised_notebook_sha256=sha(REVISED),
        html_export_sha256=sha(OUT / 'Model_A_completed_evidence.html'), metrics_recalculated=False,
        training_or_kernel_execution=False, original_files_unchanged=True,
        unavailable=['Original notebook training-cell execution count, IOPub progress and kernel state: never captured because the fit ran externally'],
        verification_limit='New artifact digests record current bytes; prediction files lacked separate original per-file digests. Existing run metadata, saved replay outputs, checkpoint hash and prior verification records link the evidence; no historical timestamps invented.')
    (OUT / 'AUDIT_MANIFEST.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf8')
    (OUT / 'AUDIT_REPORT.md').write_text(f'''# Model A notebook-output audit

Problem: opening the clean source notebook shows no saved results.
Cause: source has31 total cells, including16 Markdown and15 code cells; all source code counts are null and outputs empty. The executed evidence was saved separately. It has15 executed code cells (counts1–15),33 output blocks and no errors. All original source cell IDs/types/text match. Training ran externally, not in that notebook replay.

Fix: created `{relative(REVISED)}`. Part A explicitly labels static displays reconstructed from existing artifacts. Part B preserves all31 saved completed-results cells unchanged, including counts, timestamps and outputs. Original source and executed notebooks preserved. No kernel, training, frame generation, split edits, hyperparameter changes or metric recalculation.

Verified: source notebook SHA matches the official run snapshot; best.pt SHA matches the run manifest; completed65 epochs, selected checkpoint46, first strict maximum45; embedded metrics/run-manifest outputs exactly match saved JSON; embedded confusion PNG matches saved PNG byte-for-byte. History, per-class CSV, metric CSV and360 prediction records are present. All protected original files remain byte-identical. All restored image attachments use the saved image bytes. Notebook schema, save/reopen cell equality and static HTML conversion passed.

Missing: no required log/history/metric/confusion/checkpoint/prediction artifact is missing. Original in-notebook training execution counts/IOPub streams and kernel state cannot be recovered: the65-epoch fit ran outside Jupyter. Existing external log text is available and labelled as an artifact display, never an original notebook stream.

Evidence limitation: newly generated SHA receipts establish current preservation, not original creation times. Validation prediction files did not have separate contemporaneous per-file hashes; existing run metadata, checkpoint hashes, the saved replay displays and prior verification records provide the available linkage. No metric was recomputed during this audit.

Open the revised notebook to view outputs without running cells, or open `Model_A_completed_evidence.html` in this audit directory. `AUDIT_MANIFEST.json` records every input digest, cell inventory, preserved-cell offset and presentation source.
''', encoding='utf8')
    print(json.dumps(dict(status='PASS', revised_notebook=str(REVISED), preserved_cells=31,
                          preserved_code_execution_counts=15, preserved_output_blocks=33,
                          original_files_unchanged=True, training_rerun=False, metrics_recalculated=False)))


if __name__ == '__main__':
    main()
