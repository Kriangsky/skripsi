"""Author notebook cells; never runs training. Execute the resulting notebook with Jupyter."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).resolve().parent
cells = []
def section(title, prose, code):
    cells.extend([nbf.v4.new_markdown_cell(f'## {title}\n\n{prose}'), nbf.v4.new_code_cell(code.strip())])

section('01 — Experiment Overview', '''Independent standard YOLO26n eight-class baseline. No Model A weights, Model C changes, or external Kaggle evaluation.
Shared six-class frames and frozen splits match Model A exactly. Internal validation selects hyperparameters; it is neither an independent test nor signer-independent.
Eight predefined seeded random trials, at most 100 epochs/trial, patience 20. Rank by eight-class validation macro-F1 (five-decimal framework fitness), then lowest trial ID. Within a trial, last tied epoch wins.
Model A has one documented baseline: 65 epochs, accuracy-selected epoch 46, macro-F1 0.9944666391. Model A requires independent hyperparameter optimization before the final optimized Model A vs Model B comparison.
New-class historical selections cover only 160/600 videos. Preserve these; for the other 440 reuse existing cached candidates with Model A's exact technical rank. No extraction or split generation. Labels remain semantically unassessed.
Training runs only in section 08. Reopening does not run cells. A per-kernel guard prevents a second launch. Each fresh Run All creates a unique run; no stale checkpoint reuse.''', r'''
import os, sys, json, time, shutil, hashlib, platform, subprocess, itertools, random, logging
from pathlib import Path
from datetime import datetime, timezone
sys.dont_write_bytecode = True
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'experiments/model_a/model_a.py').exists())
BASE = ROOT / 'experiments/model_b'
os.environ['YOLO_CONFIG_DIR'] = str(BASE / 'outputs/ultralytics_config')
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
sys.path.insert(0, str(BASE))
import model_b_support as b
import numpy as np
import pandas as pd
import torch, ultralytics
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import display
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
def now(): return datetime.now(timezone.utc).isoformat()
def protected_files():
    return {str(p.relative_to(ROOT)): b.sha(p) for p in (ROOT / 'experiments/model_a').rglob('*')
            if p.is_file() and '__pycache__' not in p.parts}
PROTECTED = protected_files()
print('Model B notebook initialized. Model A evidence hashes recorded:', len(PROTECTED))
''')

section('02 — Environment Verification', 'Pinned to the audited installed Ultralytics implementation. Source hashes record local trainer/validator behavior.', r'''
import importlib.metadata as metadata, inspect
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
from ultralytics.engine.trainer import BaseTrainer
ENV = dict(python=sys.version, executable=sys.executable, kernel='ipykernel', platform=platform.platform(),
           packages={p: metadata.version(p) for p in ['torch','torchvision','ultralytics','numpy','pandas','Pillow','matplotlib','nbformat','nbclient','ipykernel']},
           cuda_available=torch.cuda.is_available(), cuda=torch.version.cuda,
           gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None,
           framework_sources={str(inspect.getfile(c)): b.sha(inspect.getfile(c)) for c in [ClassificationTrainer, ClassificationValidator, BaseTrainer]})
assert ultralytics.__version__ == b.a.CONFIG['ultralytics_version'], 'Re-audit changed framework before running'
display(ENV)
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
''')

section('03 — Reproducibility Configuration', '''Independent seeded random search without replacement. SGD learning rates: 0.003, 0.01, 0.03; AdamW: 0.0001, 0.0003, 0.001. Weight decay: 0.0001, 0.0005; batch: 16, 32. Scheduler stays Model A's linear schedule (lrf 0.01). Optimizer-specific LR ranges avoid an arbitrary shared range. Seed 42 for search and each fit. No augmentation. Full-frame RGB letterbox 224, /255, mean 0/std 1.
MODE defaults to official. Smoke uses two existing samples/class/split, one trial and one epoch; smoke scores are technical checks only.''', r'''
MODE = os.environ.get('MODEL_B_MODE', 'official')
assert MODE in {'smoke', 'official'}
SEED = 42
SETTINGS = dict(b.a.CONFIG['training'])
SPACE = [dict(optimizer=opt, lr0=lr, weight_decay=wd, batch=batch)
         for opt, rates in [('SGD', [0.003, 0.01, 0.03]), ('AdamW', [0.0001, 0.0003, 0.001])]
         for lr, wd, batch in itertools.product(rates, [0.0001, 0.0005], [16, 32])]
TRIALS = random.Random(SEED).sample(SPACE, 8)
if MODE == 'smoke':
    TRIALS = TRIALS[:1]
    SETTINGS.update(epochs=1, patience=1)
for candidate in TRIALS: get_cfg(overrides=dict(SETTINGS, **candidate))
EXPERIMENT_ID = 'model_b_' + MODE + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
OUT = BASE / 'outputs' / ('smoke_test' if MODE == 'smoke' else 'official_runs') / EXPERIMENT_ID
OUT.mkdir(parents=True, exist_ok=False)
for name in ['metrics','predictions','figures','training_history','reproducibility','weights']:
    (OUT / name).mkdir()
for name in ['config','manifests','reports']: (BASE / name).mkdir(exist_ok=True)
CONFIG = dict(experiment_id=EXPERIMENT_ID, mode=MODE, base=SETTINGS, candidates=TRIALS, search_space=SPACE,
              seed=SEED, planned_trials=len(TRIALS), primary_objective='eight_class_validation_macro_f1',
              epoch_ties='last at five decimal fitness', trial_ties='lowest trial ID', device=DEVICE)
b.write_json(OUT / 'reproducibility/config.json', CONFIG)
b.write_json(BASE / 'config/search_protocol.json', dict(base=b.a.CONFIG['training'], search_space=SPACE, trials=8, seed=SEED))
display(CONFIG)
display(pd.DataFrame(TRIALS).rename_axis('trial_zero_based'))
''')

section('04 — Dataset Audit', 'Verify all 2,400 video hashes, selected-frame hashes, cached provenance, class mapping, locked assignments and cross-split byte/pixel leakage. Compare all 1,800 original samples to the actual Model A official run manifest. No external images are loaded.', r'''
ROWS, AUDIT = b.audit_dataset(verify_files=True)
FULL_ROWS = ROWS.copy()
display(ROWS.groupby(['class_name','split']).size().unstack().reindex(b.CLASSES))
display(AUDIT)
for relative, digest in AUDIT['source_hashes'].items(): assert b.sha(ROOT / relative) == digest
FULL_ROWS.to_csv(OUT / 'reproducibility/full_selected_frames.csv', index=False)
frozen_hash = b.sha(OUT / 'reproducibility/full_selected_frames.csv')
frozen_copy = BASE / 'manifests' / (frozen_hash + '.csv')
if frozen_copy.exists(): assert b.sha(frozen_copy) == frozen_hash
else: shutil.copyfile(OUT / 'reproducibility/full_selected_frames.csv', frozen_copy)
AUDIT['selected_frame_manifest_sha256'] = frozen_hash
AUDIT['split_hashes'] = {}
for split in ['train','val']:
    split_path = OUT / 'reproducibility' / (split + '_locked_assignments.csv')
    FULL_ROWS[FULL_ROWS.split.eq(split)][['sample_id','video','class_id','split']].to_csv(split_path,index=False)
    AUDIT['split_hashes'][split] = b.sha(split_path)
if MODE == 'smoke':
    ROWS = ROWS.groupby(['class_name','split'], sort=True).head(2).copy()
DATASET = OUT / 'dataset'
for row in ROWS.itertuples():
    target = DATASET / row.split / row.class_name / (row.sample_id + '.jpg')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(b.a.safe_path(b.a.CACHE, row.source_frame_path), target)
ROWS.to_csv(OUT / 'reproducibility/dataset_manifest.csv', index=False)
DATA_HASH = b.sha(OUT / 'reproducibility/dataset_manifest.csv')
b.write_json(OUT / 'reproducibility/dataset_audit.json', AUDIT)
print('Training:', sum(ROWS.split.eq('train')), 'Validation:', sum(ROWS.split.eq('val')), 'Manifest SHA256:', DATA_HASH)
''')

section('05 — Preprocessing Verification', 'Identical imported Model A transform and predictor. No center crop, random crop, flip, color or geometric augmentation. Full-frame context retained where present. Technical readability does not establish gesture-label validity.', r'''
fig, axes = plt.subplots(8, 2, figsize=(10, 24), constrained_layout=True)
for i, name in enumerate(b.CLASSES):
    row = ROWS[ROWS.class_name.eq(name)].sort_values('sample_id').iloc[0]
    with Image.open(b.a.safe_path(b.a.CACHE, row.source_frame_path)) as image:
        original = image.convert('RGB')
        transformed = b.transform(SETTINGS['imgsz'])(original)
        assert torch.equal(transformed, b.a.transform(SETTINGS['imgsz'])(original))
        axes[i,0].imshow(original)
        axes[i,1].imshow(transformed.permute(1,2,0).numpy())
    axes[i,0].set_title(name + ': ' + row.sample_id)
    axes[i,1].set_title('RGB letterbox 224; no augmentation')
    for ax in axes[i]: ax.axis('off')
fig.savefig(OUT / 'figures/preprocessing.png', dpi=100)
plt.show()
''')

section('06 — Model Initialization', 'Read only the official asset used by Model A; verify its pinned SHA-256 and 1,000-class pretrained head. Every trial initializes afresh from this file.', r'''
PRETRAINED = ROOT / 'experiments/model_a/outputs/pretrained/yolo26n-cls.pt'
assert PRETRAINED.is_file(), 'Official pretrained asset missing; obtain the recorded upstream asset before running'
PRETRAINED_HASH = b.sha(PRETRAINED)
assert PRETRAINED_HASH == b.a.CONFIG['checkpoint_sha256']
model = YOLO(str(PRETRAINED), task='classify')
assert model.task == 'classify' and len(model.names) == 1000
display(dict(source=b.a.CONFIG['checkpoint_url'], path=str(PRETRAINED), sha256=PRETRAINED_HASH))
''')

section('07 — Architecture Summary', 'Standard classification head resized by Ultralytics to eight outputs; no architectural innovation. This preview is discarded; each trial reloads the official asset.', r'''
from ultralytics.nn.tasks import ClassificationModel
ClassificationModel.reshape_outputs(model.model, 8)
model.model.names = dict(enumerate(sorted(b.CLASSES)))
model.model.eval()
with torch.inference_mode():
    forward = model.model(torch.zeros(1,3,SETTINGS['imgsz'],SETTINGS['imgsz']))
probabilities = forward[0] if isinstance(forward, tuple) else forward
assert tuple(probabilities.shape) == (1,8)
assert model.model.model[-1].linear.out_features == 8
print(model.model)
print('Parameters:', sum(p.numel() for p in model.model.parameters()), 'Output shape:', tuple(probabilities.shape))
display(pd.DataFrame(dict(output_index=range(8), class_name=[model.names[i] for i in range(8)], canonical_id=b.mapping(model.names))))
del model
''')

section('08 — Training Execution', '''Direct notebook API trials. Custom validator returns eight-class macro-F1 as fitness, controlling both best.pt and early stopping. Installed framework rounds training validation values to five decimals and replaces best.pt on ties. Upstream default instead averages top-1 and top-5 accuracy; Model A overrides to top-1.
Full logs go to each trial's training.log. Exact configurations, epoch histories, optimizer steps, timing and failed trials persist. Re-running this cell in the same kernel raises instead of launching twice.''', r'''
assert not globals().get('_MODEL_B_LAUNCHED', False), 'Training already launched in this kernel; inspect existing outputs'
_MODEL_B_LAUNCHED = True
from contextlib import redirect_stdout, redirect_stderr
from ultralytics.utils import LOGGER
STARTED = now()
TRIAL_HISTORY = []
b.write_json(OUT / 'reproducibility/run_manifest.json', dict(status='RUNNING', started=STARTED, config=CONFIG))
for trial_index, candidate in enumerate(TRIALS, 1):
    trial_id = f'trial_{trial_index:02d}'
    trial_dir = OUT / trial_id
    trial_dir.mkdir(exist_ok=False)
    config = dict(SETTINGS, **candidate)
    b.a.seeds(SEED)
    assert b.sha(PRETRAINED) == PRETRAINED_HASH
    model = YOLO(str(PRETRAINED), task='classify')
    assert len(model.names) == 1000
    trace = []
    def epoch_record(trainer):
        csv = Path(trainer.save_dir) / 'results.csv'
        count = len(pd.read_csv(csv)) if csv.exists() else 0
        if count > len(trace):
            trace.append(dict(epoch=count, fitness=float(trainer.fitness), optimizer_steps=trainer.actual_optimizer_steps,
                              learning_rates=[g['lr'] for g in trainer.optimizer.param_groups]))
            b.write_json(trial_dir / 'epoch_trace.json', trace)
    model.add_callback('on_fit_epoch_end', epoch_record)
    record = dict(trial=trial_id, **candidate, configuration=config, status='RUNNING', started=now(),
                  pretrained_sha256=PRETRAINED_HASH, dataset_manifest_sha256=DATA_HASH)
    b.write_json(trial_dir / 'trial_manifest.json', record)
    tick = time.perf_counter()
    handlers = list(LOGGER.handlers)
    log_handler = logging.FileHandler(trial_dir / 'training.log', encoding='utf8')
    for handler in handlers: LOGGER.removeHandler(handler)
    LOGGER.addHandler(log_handler)
    try:
        with (trial_dir / 'console.log').open('w', encoding='utf8') as console, redirect_stdout(console), redirect_stderr(console):
            results = model.train(trainer=b.ModelBTrainer, data=str(DATASET), project=str(trial_dir), name='fit',
                                  exist_ok=False, device=DEVICE, verbose=False, **config)
        fit = Path(model.trainer.save_dir)
        assert fit == trial_dir / 'fit'
        history = pd.read_csv(fit / 'results.csv')
        history.columns = history.columns.str.strip()
        values = history['metrics/macro_f1'].to_numpy()
        best_epoch = int(np.flatnonzero(values == values.max())[-1]) + 1
        best = fit / 'weights/best.pt'
        assert best.is_file() and (fit / 'weights/last.pt').is_file()
        record.update(status='COMPLETE', completed_epochs=len(history), best_epoch=best_epoch,
                      macro_f1=float(values.max()), duration_seconds=time.perf_counter()-tick,
                      optimizer_steps=model.trainer.actual_optimizer_steps, completed=now(),
                      stopping_reason='maximum_epochs' if len(history) == config['epochs'] else 'early_stopping',
                      stopper_best_epoch=int(model.trainer.stopper.best_epoch), checkpoint=str(best),
                      checkpoint_sha256=b.sha(best), selection='macro-F1 rounded 5 decimals; last epoch tie')
        history.to_csv(OUT / 'training_history' / (trial_id + '.csv'), index=False)
    except Exception as error:
        record.update(status='FAILED', error=f'{type(error).__name__}: {error}', completed=now(), duration_seconds=time.perf_counter()-tick)
        b.write_json(trial_dir / 'trial_manifest.json', record)
        TRIAL_HISTORY.append(record)
        b.write_json(OUT / 'trial_history.json', TRIAL_HISTORY)
        b.write_json(OUT / 'reproducibility/run_manifest.json', dict(status='FAILED', error=record['error'], started=STARTED, config=CONFIG))
        raise
    finally:
        LOGGER.removeHandler(log_handler)
        log_handler.close()
        for handler in handlers: LOGGER.addHandler(handler)
    b.write_json(trial_dir / 'trial_manifest.json', record)
    TRIAL_HISTORY.append(record)
    b.write_json(OUT / 'trial_history.json', TRIAL_HISTORY)
    pd.DataFrame(TRIAL_HISTORY).drop(columns=['configuration']).to_csv(OUT / 'trial_history.csv', index=False)
    print(trial_id, 'complete:', record['completed_epochs'], 'epochs; macro-F1', record['macro_f1'], flush=True)
    del model
    torch.cuda.empty_cache()
COMPARISON = pd.DataFrame(TRIAL_HISTORY).sort_values(['macro_f1','trial'], ascending=[False,True])
SELECTED = COMPARISON.iloc[0].to_dict()
BEST = Path(SELECTED['checkpoint'])
display(COMPARISON[['trial','optimizer','lr0','weight_decay','batch','best_epoch','completed_epochs','macro_f1','stopping_reason','duration_seconds','optimizer_steps']])
display(SELECTED)
''')

section('09 — Training History', 'Actual selected-run curves and completed exposure. Epoch numbers come from results.csv, never assumed from the maximum budget.', r'''
HISTORY = pd.read_csv(OUT / 'training_history' / (SELECTED['trial'] + '.csv'))
display(HISTORY.tail())
columns = [c for c in HISTORY if 'loss' in c or c.startswith('metrics/')]
axes = HISTORY.plot(x='epoch', y=columns, subplots=True, figsize=(10,3*len(columns)), grid=True)
plt.tight_layout()
plt.gcf().savefig(OUT / 'figures/selected_training_curves.png', dpi=120)
plt.show()
print('Best epoch:', SELECTED['best_epoch'], 'Completed:', SELECTED['completed_epochs'], 'Duration seconds:', SELECTED['duration_seconds'])
''')

section('10 — Internal Validation', 'Reload selected best checkpoint and evaluate with the exact Model A predictor transform. Preserve all eight probabilities. No novel-class masking.', r'''
best_model = YOLO(str(BEST), task='classify')
OUTPUT_MAPPING = b.mapping(best_model.names)
assert best_model.model.model[-1].linear.out_features == 8
validation = ROWS[ROWS.split.eq('val')].sort_values('sample_id')
records = []
batch = int(SELECTED['batch'])
for offset in range(0, len(validation), batch):
    part = validation.iloc[offset:offset+batch]
    paths = [str(b.a.safe_path(b.a.CACHE, r.source_frame_path)) for r in part.itertuples()]
    predictions = best_model.predict(paths, predictor=b.Predictor, imgsz=SETTINGS['imgsz'], batch=batch,
                                     device=DEVICE, verbose=False, save=False)
    assert len(predictions) == len(part)
    for row, prediction in zip(part.itertuples(), predictions):
        probs = prediction.probs.data.cpu().numpy()
        assert probs.shape == (8,) and np.isfinite(probs).all() and (probs >= 0).all() and np.isclose(probs.sum(),1,atol=1e-4)
        canonical = np.zeros(8)
        canonical[OUTPUT_MAPPING] = probs
        index = int(probs.argmax())
        predicted_id = OUTPUT_MAPPING[index]
        record = dict(experiment_id=EXPERIMENT_ID, model_name='Model B', source_video_id=row.sample_id,
                      sample_id=row.sample_id, video=row.video, frame_id=row.candidate_id, frame_index=int(row.frame_index),
                      frame_path=row.source_frame_path, frame_sha256=row.source_frame_sha256, split=row.split,
                      true_class=row.class_name, true_canonical_id=int(row.class_id), model_output_index=index,
                      predicted_class=b.CLASSES[predicted_id], predicted_canonical_id=predicted_id,
                      confidence=float(probs[index]), correct=predicted_id == int(row.class_id),
                      probabilities_model_order=json.dumps(probs.tolist()), probabilities_canonical_order=json.dumps(canonical.tolist()))
        record.update({f'p_{name}':float(canonical[i]) for i,name in enumerate(b.CLASSES)})
        records.append(record)
PREDICTIONS = pd.DataFrame(records)
assert len(PREDICTIONS) == len(validation) and not PREDICTIONS.sample_id.duplicated().any()
PREDICTIONS.to_csv(OUT / 'predictions/validation_predictions.csv', index=False)
display(PREDICTIONS.head())
''')

section('11 — Eight-Class Evaluation', 'Macro averages include all eight labels; zero denominators map to zero. Matrix rows are truth; columns are full eight-way predictions. FP16 saved checkpoints can differ slightly from in-training FP32 EMA fitness; both values are retained.', r'''
METRICS8, PER_CLASS8, MATRIX8 = b.metrics(PREDICTIONS.true_canonical_id, PREDICTIONS.predicted_canonical_id)
display(pd.DataFrame([METRICS8]))
display(PER_CLASS8)
print('Selection fitness (rounded, training EMA):', SELECTED['macro_f1'], 'Reloaded checkpoint macro-F1:', METRICS8['macro_f1'])
pd.DataFrame([METRICS8]).to_csv(OUT / 'metrics/eight_class_metrics.csv', index=False)
PER_CLASS8.to_csv(OUT / 'metrics/eight_class_per_class.csv', index=False)
b.write_json(OUT / 'metrics/eight_class_metrics.json', dict(**METRICS8, per_class=PER_CLASS8.to_dict('records')))
''')

section('12 — Original Six-Class Evaluation', 'Restrict ground-truth samples to the six originals, retain predictions into halo/tidur as errors. Six-label macro-F1 counts intrusions as false negatives. This score and Model A share the validation sample set only in official mode.', r'''
SIX = PREDICTIONS[PREDICTIONS.true_canonical_id.lt(6)].copy()
METRICS6, PER_CLASS6, MATRIX6 = b.metrics(SIX.true_canonical_id, SIX.predicted_canonical_id, labels=range(6))
display(pd.DataFrame([METRICS6]))
display(PER_CLASS6)
SIX.to_csv(OUT / 'predictions/original_six_predictions.csv', index=False)
b.write_json(OUT / 'metrics/original_six_metrics.json', dict(**METRICS6, per_class=PER_CLASS6.to_dict('records')))
PER_CLASS6.to_csv(OUT / 'metrics/original_six_per_class.csv', index=False)
''')

section('13 — Per-Class Analysis', 'Original-to-new intrusion and reverse new-to-original errors. Visible full confusion matrix; no detection mAP.', r'''
INTRUSIONS = pd.crosstab(SIX.true_class, SIX.predicted_class).reindex(index=b.CLASSES[:6], columns=b.CLASSES[6:], fill_value=0).fillna(0).astype(int)
INTRUSIONS['support'] = SIX.groupby('true_class').size().reindex(b.CLASSES[:6])
INTRUSIONS['intrusion_rate'] = INTRUSIONS[['halo','tidur']].sum(axis=1) / INTRUSIONS.support
NOVEL = PREDICTIONS[PREDICTIONS.true_canonical_id.ge(6)]
INTRUSION_SUMMARY = dict(predicted_halo=int(SIX.predicted_class.eq('halo').sum()), predicted_tidur=int(SIX.predicted_class.eq('tidur').sum()),
                         original_to_new_rate=float(SIX.predicted_canonical_id.ge(6).mean()),
                         new_to_original_count=int(NOVEL.predicted_canonical_id.lt(6).sum()),
                         new_to_original_rate=float(NOVEL.predicted_canonical_id.lt(6).mean()))
display(INTRUSIONS)
display(INTRUSION_SUMMARY)
INTRUSIONS.to_csv(OUT / 'metrics/intrusions_by_original_class.csv')
b.write_json(OUT / 'metrics/intrusion_summary.json', INTRUSION_SUMMARY)
pd.DataFrame(MATRIX8,index=b.CLASSES,columns=b.CLASSES).to_csv(OUT / 'metrics/confusion_matrix.csv')
fig, ax = plt.subplots(figsize=(9,8), constrained_layout=True)
ax.imshow(MATRIX8, cmap='Blues')
ax.set(xticks=range(8),yticks=range(8),xticklabels=b.CLASSES,yticklabels=b.CLASSES,xlabel='Predicted',ylabel='True',title='Model B internal validation — canonical order')
plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
for i,j in itertools.product(range(8),repeat=2): ax.text(j,i,str(MATRIX8[i,j]),ha='center',va='center')
fig.savefig(OUT / 'figures/confusion_matrix.png',dpi=150)
plt.show()
''')

section('14 — Model A Compatibility Checks', 'Exact sample/frame/split equality precedes any paired comparison. Model A remains an untuned, accuracy-selected baseline. This is an exploratory internal-validation comparison, not an optimized-model fairness claim.', r'''
A_RUN = Path(AUDIT['model_a_run'])
A_ROWS = b.read_csv(A_RUN / 'dataset_manifest.csv')
A_PRED = pd.read_csv(A_RUN / 'validation_predictions.csv')
cols = ['sample_id','video','class_id','class_name','split','candidate_id','frame_index','source_frame_path','source_frame_sha256']
assert FULL_ROWS[FULL_ROWS.class_name.isin(b.CLASSES[:6])][cols].sort_values('sample_id').reset_index(drop=True).equals(A_ROWS[cols].sort_values('sample_id').reset_index(drop=True))
PAIRED = SIX.merge(A_PRED[['sample_id','true_canonical_id','predicted_canonical_id']],on='sample_id',suffixes=('_b','_a'),validate='one_to_one')
assert len(PAIRED) == len(SIX) and PAIRED.true_canonical_id_b.eq(PAIRED.true_canonical_id_a).all()
if MODE == 'official': assert set(SIX.sample_id) == set(A_PRED.sample_id)
PAIRED.to_csv(OUT / 'predictions/paired_model_a_model_b.csv',index=False)
A_MANIFEST = json.loads((A_RUN / 'run_manifest.json').read_text())
display(pd.DataFrame([dict(model='A untuned baseline',samples=360,macro_f1=A_MANIFEST['metrics']['macro_f1']),
                      dict(model='B common six ' + MODE,samples=len(SIX),macro_f1=METRICS6['macro_f1'])]))
print('Exact shared source/frame/split matches: 1800. Paired evaluated samples:',len(PAIRED))
''')

section('15 — Artifact Export', 'Preserve original trial checkpoints plus copies of selected best/last. Record checkpoint, manifest and code hashes, full environment, actual exposure and deviations. Model A evidence must remain byte-identical.', r'''
for name in ['best.pt','last.pt']: shutil.copyfile(BEST.parent / name, OUT / 'weights' / name)
MAPPING = dict(canonical=dict(enumerate(b.CLASSES)),model_output=best_model.names,output_to_canonical=OUTPUT_MAPPING)
b.write_json(OUT / 'reproducibility/label_mapping.json',MAPPING)
b.write_json(OUT / 'reproducibility/environment.json',ENV)
try: GIT = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
except subprocess.CalledProcessError: GIT = None
assert protected_files() == PROTECTED, 'Model A evidence changed during execution'
for relative,digest in AUDIT['source_hashes'].items(): assert b.sha(ROOT / relative) == digest
b.write_json(OUT / 'reproducibility/protected_model_a_hashes.json',PROTECTED)
for filename in ['model_b_support.py','build_notebook.py','test_model_b.py','Model_B_YOLO26_8Class_Training.ipynb']:
    shutil.copyfile(BASE / filename,OUT / 'reproducibility' / filename)
MANIFEST = dict(status='TECHNICAL_SMOKE_COMPLETE' if MODE == 'smoke' else 'COMPLETE', experiment_id=EXPERIMENT_ID,
                started=STARTED,completed=now(),architecture='yolo26n-cls',pretrained_source=b.a.CONFIG['checkpoint_url'],
                pretrained_path=str(PRETRAINED),pretrained_sha256=PRETRAINED_HASH,config=CONFIG,
                completed_trials=len(TRIAL_HISTORY),selected_trial=SELECTED,final_metrics8=METRICS8,final_metrics6=METRICS6,
                environment=ENV,dataset_audit=AUDIT,dataset_manifest_sha256=DATA_HASH,class_mapping=MAPPING,git_commit=GIT,
                checkpoint_hashes={p.name:b.sha(p) for p in (OUT / 'weights').glob('*.pt')},
                total_duration_seconds=sum(r['duration_seconds'] for r in TRIAL_HISTORY),
                total_epochs=sum(r['completed_epochs'] for r in TRIAL_HISTORY),
                total_optimizer_steps=sum(r['optimizer_steps'] for r in TRIAL_HISTORY),
                source_code_hashes={name:b.sha(BASE / name) for name in ['model_b_support.py','build_notebook.py','test_model_b.py']},
                limitations=['Model A untuned; independent optimization required','Validation reused for tuning; no independent generalization estimate',
                             'Not signer independent; semantic labels unassessed','Vocabulary, sample count and optimized settings differ',
                             '440 new-class frames selected from existing cache; 160 historical selections preserved',
                             'Framework rounds selection fitness to five decimals; saved FP16 checkpoint may differ from training EMA'])
b.write_json(OUT / 'reproducibility/run_manifest.json',MANIFEST)
b.write_json(BASE / 'reports' / ('latest_smoke_run.json' if MODE == 'smoke' else 'latest_official_run.json'),dict(run_directory=str(OUT),status=MANIFEST['status']))
display(dict(output=str(OUT),status=MANIFEST['status'],completed_trials=len(TRIAL_HISTORY),checkpoint_hashes=MANIFEST['checkpoint_hashes']))
''')

section('16 — Final Experiment Summary', 'Executed values only. Save this notebook with outputs. Smoke results must never be reported as the official experiment. The selected configuration is already represented by its best trial checkpoint; no extra validation-driven retraining.', r'''
display(pd.DataFrame([dict(status=MANIFEST['status'],selected_trial=SELECTED['trial'],trials_completed=len(TRIAL_HISTORY),
                           selected_epochs=SELECTED['completed_epochs'],best_epoch=SELECTED['best_epoch'],
                           eight_class_macro_f1=METRICS8['macro_f1'],eight_class_accuracy=METRICS8['top1_accuracy'],
                           original_six_macro_f1=METRICS6['macro_f1'],original_six_accuracy=METRICS6['top1_accuracy'],
                           intrusion_rate=INTRUSION_SUMMARY['original_to_new_rate'],output=str(OUT))]))
display(SELECTED['configuration'])
print('\n'.join(MANIFEST['limitations']))
print('Model A requires independent hyperparameter optimization before the final optimized Model A vs Model B comparison.')
print('Training completed inside notebook cells. Persist execution counts and outputs; do not clear outputs.')
''')

nb = nbf.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(display_name='BISINDO Python', language='python', name='python3'),language_info=dict(name='python',version='3.11.5')))
nbf.validate(nb)
nbf.write(nb,HERE / 'Model_B_YOLO26_8Class_Training.ipynb')
print('Notebook authored:',len(cells),'cells; unexecuted.')
