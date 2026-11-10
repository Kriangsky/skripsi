"""Model A only. No historical training imports, automatic fitting or split generation."""
from __future__ import annotations

from collections import Counter
from copy import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import inspect
import io
import json
import logging
import os
from pathlib import Path
import random
import shutil
import sys
from contextlib import redirect_stdout
from urllib.request import urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ.setdefault('YOLO_CONFIG_DIR', str(HERE / 'outputs/ultralytics_config'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision import transforms as T
import ultralytics
from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator
from ultralytics.models.yolo.classify.predict import ClassificationPredictor

CLASSES = ['saya', 'kamu', 'senang', 'bingung', 'marah', 'apa_kabar']
ALL_CLASSES = CLASSES + ['halo', 'tidur']
LOCK = ROOT / '02_exp/arsip_pipeline/experiment_yolo26_v3/split_lock.csv'
CACHE = LOCK.parent
SELECTION = ROOT / '02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/manifest.csv'
PILOT = ROOT / '02_exp/arsip_pipeline/experiment_yolo26_pilot/pilot_manifest_private.csv'
CONFIG = json.loads((HERE / 'config/model_a.json').read_text())
DECISIONS = HERE / 'manifests/frame_decisions.csv'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n', encoding='utf8')


def safe_path(base, relative):
    path = (Path(base) / relative).resolve()
    require(path.is_relative_to(Path(base).resolve()), f'Path escapes source: {relative}')
    return path


def seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def environment():
    require(ultralytics.__version__ == CONFIG['ultralytics_version'],
            'Ultralytics version changed; re-audit transforms and trainer before fitting')
    from ultralytics.data.augment import classify_transforms, classify_augmentations
    from ultralytics.cfg import get_cfg
    get_cfg(overrides=CONFIG['training'])  # Reject unsupported arguments before any fit.
    return dict(python=sys.version, executable=sys.executable,
                packages={p: metadata.version(p) for p in ['ultralytics', 'torch', 'torchvision',
                           'numpy', 'pandas', 'Pillow', 'matplotlib', 'nbformat', 'nbclient']},
                cuda_available=torch.cuda.is_available(), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                default_validation_transform=str(classify_transforms(224)),
                upstream_transform_source_sha256=sha(inspect.getfile(classify_transforms)),
                default_training_has_random_crop='RandomResizedCrop' in inspect.getsource(classify_augmentations),
                preprocessing='RGB; full-frame bilinear resize + centered gray114 padding; tensor /255; mean0/std1; no augmentation')


def verify_inventory(lock):
    for field in ['sample_id', 'video', 'video_sha256']:
        require(not lock[field].eq('').any() and not lock[field].duplicated().any(), f'Duplicate/empty {field}')
    for row in lock.itertuples():
        parts = row.video.replace('\\', '/').split('/')
        require(parts[0] == row.source and (row.source == 'test' or parts[1] == row.class_name) and
                not any(p in ['', '.', '..'] for p in parts), 'Video path/label mismatch')
        require(row.class_name in ALL_CLASSES and str(ALL_CLASSES.index(row.class_name)) == row.class_id,
                'Canonical class mapping mismatch')
        require(row.source in ['train', 'test'] and row.split in ['train', 'val', 'test'] and
                (row.source == 'test') == (row.split == 'test'), 'External/internal split mismatch')
        require(row.sample_id == hashlib.sha256(row.video.encode()).hexdigest()[:20], 'Sample/video ID mismatch')
    internal = lock[lock.source.eq('train')]
    require(len(internal) == 2400 and internal.class_name.value_counts().to_dict() == dict.fromkeys(ALL_CLASSES, 300),
            'Expected 2400 internal videos, 300 per class')
    six = internal[internal.class_name.isin(CLASSES)].copy()
    require(len(six) == 1800, 'Expected 1800 six-class source videos')
    require(not set(six[six.split.eq('train')].video_sha256) & set(six[six.split.eq('val')].video_sha256),
            'Train/validation source-content leakage')
    return six


def review_concerns():
    """Reviewer IDs refer to a different pilot namespace; join through video_id."""
    pilot = read_csv(PILOT).set_index('sample_id')
    records = []
    for path in sorted((ROOT / '08_review/respons').glob('reviewer_*_final.json')):
        data = json.loads(path.read_text(encoding='utf8'))
        for response in data['responses']:
            require(response['sample_id'] in pilot.index, 'Unknown reviewer pilot ID')
            row = pilot.loc[response['sample_id']]
            if response['decision'] != row.class_name or response.get('reasons'):
                records.append(dict(sample_id=row.video_id, video=row.video,
                                    reviewer=data['reviewer_id'], decision=response['decision'],
                                    reasons=json.dumps(response.get('reasons', [])),
                                    scope='Historical pilot frame; does not adjudicate another selected frame'))
    return pd.DataFrame(records)


def audit(hash_videos=True):
    """Read internal videos only; external rows are used solely for identity rejection."""
    lock = read_csv(LOCK)
    six = verify_inventory(lock)
    frozen = read_csv(CACHE / 'inventory.csv')
    require(lock.sort_values('sample_id').reset_index(drop=True).equals(
        frozen.sort_values('sample_id').reset_index(drop=True)), 'Inventory/split lock disagree')
    raw = ROOT / '04_dataset/train'
    physical = {p.relative_to(raw).as_posix() for p in raw.rglob('*')
                if p.is_file() and p.suffix.lower() in ['.mov', '.mp4', '.avi', '.mkv']}
    expected = {x.removeprefix('train/') for x in lock[lock.source.eq('train')].video}
    require(physical == expected, 'Internal physical video inventory differs from frozen source')
    for row in six.itertuples():
        path = safe_path(ROOT / '04_dataset', row.video)
        require(path.is_file(), f'Missing internal video: {row.video}')
        if hash_videos:
            require(sha(path) == row.video_sha256, f'Video hash mismatch: {row.video}')
    prior = read_csv(SELECTION)
    prior = prior[prior.class_name.isin(CLASSES)]
    require(not prior.sample_id.duplicated().any(), 'Duplicate historical selection')
    by_id = six.set_index('sample_id')
    for row in prior.itertuples():
        require(row.sample_id in by_id.index, 'Selection outside internal six-class inventory')
        require(all(str(getattr(row, k)) == by_id.loc[row.sample_id, k]
                    for k in ['video', 'class_id', 'class_name', 'split', 'video_sha256', 'source']),
                'Historical selection identity mismatch')
    candidates = read_csv(CACHE / 'candidates.csv').set_index('candidate_id')
    require(not candidates.index.duplicated().any(), 'Duplicate candidate IDs')
    for row in prior.itertuples():
        require(row.candidate_id in candidates.index, 'Missing selected candidate')
        c = candidates.loc[row.candidate_id]
        require(c.sample_id == row.sample_id and c.frame_index == row.frame_index and
                c.frame_path == row.source_frame_path and c.frame_sha256 == row.source_frame_sha256,
                'Selected candidate provenance mismatch')
        require(sha(safe_path(CACHE, c.frame_path)) == c.frame_sha256, 'Selected-frame hash mismatch')
    concerns = review_concerns()
    reports = HERE / 'reports'
    reports.mkdir(exist_ok=True)
    concerns.to_csv(reports / 'historical_review_concerns.csv', index=False)
    six.to_csv(HERE / 'manifests/source_inventory.csv', index=False)
    template = six.copy()
    selection = prior.set_index('sample_id')
    for field in ['candidate_id', 'frame_index', 'source_frame_path', 'source_frame_sha256', 'quality_status', 'quality_flags']:
        template[field] = template.sample_id.map(selection[field]).fillna('')
    template['semantic_status'] = np.where(template.candidate_id.ne(''), 'not_assessed', 'selection_missing')
    template['historical_review_concern'] = template.sample_id.isin(set(concerns.sample_id)).astype(str)
    template['disposition'] = 'pending'
    template['decision_note'] = ''
    template['reviewer'] = ''
    template['reviewed_candidate_id'] = ''
    template.to_csv(HERE / 'manifests/frame_decisions_template.csv', index=False)
    if not DECISIONS.exists():
        shutil.copyfile(HERE / 'manifests/frame_decisions_template.csv', DECISIONS)
    counts = six.groupby(['class_name', 'split']).size().unstack(fill_value=0).reindex(CLASSES)
    counts.to_csv(reports / 'split_counts.csv')
    protected = [LOCK, CACHE / 'inventory.csv', CACHE / 'annotations.csv', CACHE / 'candidate_config.json',
                 CACHE / 'candidates.csv', SELECTION, PILOT, *sorted((ROOT / '08_review/respons').glob('reviewer_*_final.json'))]
    state = dict(status='BLOCKED_PENDING_FRAME_DECISIONS', source_videos=len(six), counts=counts.to_dict(),
                 existing_selected_frames=len(prior), missing_selected_frames=len(six)-len(prior),
                 raw_video_hashes_verified=hash_videos, video_leakage=False, external_samples_used=0,
                 reviewed_v3_frames=int(read_csv(CACHE / 'annotations.csv').frame_review_status.eq('approved').sum()),
                 historical_concern_videos=int(six.sample_id.isin(set(concerns.sample_id)).sum()),
                 protected_input_hashes={str(p.relative_to(ROOT)): sha(p) for p in protected},
                 note='480 existing six-class selections are unassessed; no new frame-selection policy applied. '
                      'Official preparation rechecks all decisions; this audit does not certify semantic validity.')
    write_json(reports / 'dataset_audit.json', state)
    return state, template


class LetterboxRGB:
    """PIL RGB in, uint8 square out; never crops or mirrors source context."""
    def __init__(self, size=224):
        self.size = size

    def __call__(self, image):
        image = image.convert('RGB')
        w, h = image.size
        scale = self.size / max(w, h)
        resized = image.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (self.size, self.size), (114, 114, 114))
        canvas.paste(resized, ((self.size-resized.width)//2, (self.size-resized.height)//2))
        return canvas


def transform(size=224):
    return T.Compose([LetterboxRGB(size), T.ToTensor(), T.Normalize((0., 0., 0.), (1., 1., 1.))])


def sanity_plot(rows, size=224):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(6, 2, figsize=(10, 20), constrained_layout=True)
    for i, name in enumerate(CLASSES):
        row = rows[rows.class_name.eq(name) & rows.source_frame_path.ne('')].sort_values('sample_id').iloc[0]
        with Image.open(safe_path(CACHE, row.source_frame_path)) as image:
            axes[i, 0].imshow(image.convert('RGB'))
            axes[i, 1].imshow(LetterboxRGB(size)(image))
        axes[i, 0].set_title(f'{name}: {row.sample_id} (unassessed)')
        axes[i, 1].set_title(f'RGB letterbox {size}; no crop/flip')
        for ax in axes[i]:
            ax.axis('off')
    fig.savefig(HERE / 'reports/preprocessing_sanity.png', dpi=110)
    return fig


class ModelAValidator(ClassificationValidator):
    def build_dataset(self, img_path):
        dataset = super().build_dataset(img_path)
        dataset.torch_transforms = transform(self.args.imgsz)
        return dataset

    def get_stats(self):
        stats = super().get_stats()
        stats['fitness'] = stats['metrics/accuracy_top1']
        return stats


class ModelATrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        dataset.torch_transforms = transform(self.args.imgsz)
        expected = sorted(str(p) for p in Path(img_path).glob('*/*.jpg'))
        require(sorted(str(x[0]) for x in dataset.samples) == expected, 'Loader silently dropped/added images')
        require(set(dataset.base.classes) == set(CLASSES), 'Unexpected class folders')
        mapping = {str(name): int(index) for index, name in self.data['names'].items()}
        require(all(mapping[Path(sample[0]).parent.name] == int(sample[1]) for sample in dataset.samples),
                'Loader output indices differ from model names')
        return dataset

    def _build_train_pipeline(self):
        require(self.batch_size == 16, 'Automatic minibatch change forbidden; change config explicitly')
        return super()._build_train_pipeline()

    def get_validator(self):
        return ModelAValidator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def _setup_train(self):
        super()._setup_train()
        require(self.batch_size == 16 and isinstance(self.optimizer, torch.optim.SGD),
                'Unexpected batch/optimizer; abort rather than silently change configuration')
        mapping = label_mapping(self.data['names'])
        write_json(Path(self.save_dir) / 'training_label_mappings.json',
                   dict(canonical=dict(enumerate(CLASSES)), model_output=self.data['names'],
                        model_output_to_canonical=dict(enumerate(mapping))))
        write_json(Path(self.save_dir) / 'actual_training_config.json',
                   dict(args=vars(self.args), batch=self.batch_size, optimizer=str(self.optimizer),
                        accumulation=int(self.accumulate),
                        train_samples=len(self.train_loader.dataset), val_samples=len(self.test_loader.dataset),
                        full_train_batches=len(self.train_loader.dataset)//self.batch_size,
                        last_train_batch=min(self.batch_size, len(self.train_loader.dataset) % self.batch_size or self.batch_size),
                        names=self.data['names'], scheduler='linear lr0 -> lr0*lrf after warmup',
                        optimizer_groups=[{k: v for k, v in group.items() if k != 'params'}
                                          for group in self.optimizer.param_groups]))
        with (Path(self.save_dir) / 'six_class_architecture.txt').open('w', encoding='utf8') as stream:
            stream.write(str(self.model))


class ModelAPredictor(ClassificationPredictor):
    def setup_source(self, source):
        super().setup_source(source)
        self.transforms = transform(self.imgsz[0])
        self.host_transforms = None
        self.device_transform = None


def pretrained():
    """Fresh upstream asset only; never search historical checkpoint directories."""
    path = HERE / 'outputs/pretrained/yolo26n-cls.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix('.download')
        require(not partial.exists(), f'Incomplete download preserved at {partial}; inspect before retry')
        with urlopen(CONFIG['checkpoint_url'], timeout=60) as response, partial.open('xb') as stream:
            shutil.copyfileobj(response, stream)
        require(sha(partial) == CONFIG['checkpoint_sha256'], 'Upstream asset hash differs; download preserved for review')
        partial.rename(path)
    require(sha(path) == CONFIG['checkpoint_sha256'], 'Official pretrained checkpoint hash mismatch')
    model = YOLO(str(path), task='classify')
    require(model.task == 'classify' and len(model.names) == 1000, 'Expected ImageNet classification checkpoint')
    model.model.eval()
    with torch.inference_mode():
        result = model.model(torch.zeros(1, 3, CONFIG['training']['imgsz'], CONFIG['training']['imgsz']))
    probabilities = result[0] if isinstance(result, tuple) else result
    require(tuple(probabilities.shape) == (1, 1000) and torch.isfinite(probabilities).all(), 'Pretrained forward failed')
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.info(verbose=True)
    (HERE / 'reports/pretrained_architecture.txt').write_text(str(model.model) + '\n' + buffer.getvalue(), encoding='utf8')
    write_json(HERE / 'reports/pretrained_verification.json',
               dict(url=CONFIG['checkpoint_url'], sha256=sha(path), task=model.task, output_shape=list(probabilities.shape)))
    return path, model


def validate_decisions(decisions, inventory, candidates):
    require(not decisions.sample_id.duplicated().any() and set(decisions.sample_id) == set(inventory.sample_id),
            'Decisions must account for every one of the 1800 source videos exactly once')
    indexed = inventory.set_index('sample_id')
    for row in decisions.itertuples():
        original = indexed.loc[row.sample_id]
        require(all(str(getattr(row, k)) == original[k] for k in
                    ['video', 'source', 'class_id', 'class_name', 'split', 'video_sha256']), 'Decision changed frozen identity')
        require(row.disposition in ['include', 'exclude'], f'Unresolved frame decision: {row.sample_id}')
        require(bool(row.reviewer.strip()) and bool(row.decision_note.strip()), 'Explicit reviewer and decision note required')
        if row.disposition == 'exclude':
            continue
        require(row.semantic_status == 'valid_for_target' and row.reviewed_candidate_id == row.candidate_id,
                'Included frames require candidate-bound semantic review; known ambiguity cannot be silently included')
        require(row.candidate_id in candidates.index, 'Unknown selected candidate')
        c = candidates.loc[row.candidate_id]
        require(c.sample_id == row.sample_id and all(getattr(row, k) == c[target] for k, target in
                [('frame_index', 'frame_index'), ('source_frame_path', 'frame_path'), ('source_frame_sha256', 'frame_sha256')]),
                'Frame provenance mismatch')
    return decisions[decisions.disposition.eq('include')].copy()


def prepare(smoke=False):
    """Immutable content-addressed dataset; verify every existing copy on rerun."""
    state, template = audit(hash_videos=True)
    decisions = read_csv(DECISIONS)
    if smoke:
        # shortcut: technical subset cannot establish full-data readiness; replace only after frame adjudication.
        rows = template[template.candidate_id.ne('') & template.historical_review_concern.eq('False')]
        rows = rows.sort_values('sample_id').groupby(['split', 'class_name'], sort=True).head(2).copy()
        require(len(rows) == 24, 'Smoke requires two existing train/val selections per class')
    else:
        rows = validate_decisions(decisions, read_csv(HERE / 'manifests/source_inventory.csv'),
                                  read_csv(CACHE / 'candidates.csv').set_index('candidate_id'))
        for split in ['train', 'val']:
            require(set(rows[rows.split.eq(split)].class_name) == set(CLASSES), 'Missing class/split after explicit exclusions')
    require(set(rows.source) == {'train'} and set(rows.split) == {'train', 'val'}, 'External/unknown sample rejected')
    require(not rows.sample_id.duplicated().any() and not rows.video_sha256.duplicated().any(), 'Duplicate video identity')
    seen_bytes, seen_pixels = {}, {}
    for row in rows.itertuples():
        path = safe_path(CACHE, row.source_frame_path)
        require(sha(path) == row.source_frame_sha256, f'Selected bytes changed: {row.sample_id}')
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
            digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
        for store, key in [(seen_bytes, row.source_frame_sha256), (seen_pixels, digest)]:
            require(key not in store or store[key] == row.split, 'Cross-split selected-frame content leakage')
            store[key] = row.split
    payload = dict(rows=rows.to_dict('records'), inputs=state['protected_input_hashes'],
                   decisions=sha(DECISIONS) if not smoke else 'technical_only', mode='smoke' if smoke else 'official')
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    folder = HERE / 'datasets' / ('smoke' if smoke else 'official') / fingerprint
    folder.mkdir(parents=True, exist_ok=True)
    expected = set()
    for row in rows.itertuples():
        target = folder / row.split / row.class_name / f'{row.sample_id}.jpg'
        target.parent.mkdir(parents=True, exist_ok=True)
        expected.add(target.relative_to(folder).as_posix())
        if target.exists():
            require(sha(target) == row.source_frame_sha256, 'Existing prepared copy changed; no overwrite')
        else:
            shutil.copyfile(safe_path(CACHE, row.source_frame_path), target)
    require({p.relative_to(folder).as_posix() for p in folder.glob('*/*/*.jpg')} == expected,
            'Unexpected prepared files; no automatic cleanup')
    rows.to_csv(folder / 'dataset_manifest.csv', index=False)
    write_json(folder / 'dataset_fingerprint.json', dict(fingerprint=fingerprint, **payload))
    exclusions = decisions[decisions.disposition.eq('exclude')]
    if not smoke:
        exclusions.to_csv(folder / 'explicit_exclusions.csv', index=False)
    return folder, rows, fingerprint


def label_mapping(names):
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    require(set(names) == set(range(6)) and set(names.values()) == set(CLASSES), 'Model output mapping differs from six classes')
    return [CLASSES.index(names[i]) for i in range(6)]


def classification_metrics(truth, predicted):
    """Rows=true, columns=predicted; zero denominator -> 0; macro averages all six classes."""
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    require(len(truth) > 0 and truth.shape == predicted.shape and truth.ndim == 1, 'Invalid prediction lengths')
    require(np.isin(truth, range(6)).all() and np.isin(predicted, range(6)).all(), 'Invalid canonical IDs')
    matrix = np.zeros((6, 6), dtype=int)
    np.add.at(matrix, (truth.astype(int), predicted.astype(int)), 1)
    tp = matrix.diagonal()
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(6), where=matrix.sum(0) != 0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(6), where=matrix.sum(1) != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(6), where=(precision+recall) != 0)
    per_class = pd.DataFrame(dict(class_name=CLASSES, canonical_id=range(6), precision=precision,
                                 recall=recall, f1=f1, support=matrix.sum(1)))
    return dict(top1_accuracy=float(tp.sum()/matrix.sum()), macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()), macro_f1=float(f1.mean()), samples=int(matrix.sum())), per_class, matrix


def evaluate(checkpoint, rows, out, size=224, batch=16, device='cpu'):
    """Explicit predictor enforces the training transform after checkpoint reload."""
    model = YOLO(str(checkpoint), task='classify')
    mapping = label_mapping(model.names)
    validation = rows[rows.split.eq('val')].sort_values('sample_id')
    records = []
    for start in range(0, len(validation), batch):
        part = validation.iloc[start:start+batch]
        paths = [str(safe_path(CACHE, r.source_frame_path)) for r in part.itertuples()]
        results = model.predict(paths, predictor=ModelAPredictor, imgsz=size, device=device, verbose=False, save=False)
        require(len(results) == len(part), 'Prediction coverage differs')
        for row, result in zip(part.itertuples(), results):
            probabilities = result.probs.data.cpu().numpy()
            require(probabilities.shape == (6,) and np.isfinite(probabilities).all() and
                    (probabilities >= 0).all() and np.isclose(probabilities.sum(), 1, atol=1e-4), 'Invalid six-class probabilities')
            index = int(probabilities.argmax())
            canonical_probabilities = np.zeros(6)
            canonical_probabilities[mapping] = probabilities
            record = dict(sample_id=row.sample_id, video=row.video, true_class=row.class_name,
                          true_canonical_id=int(row.class_id), predicted_class=CLASSES[mapping[index]],
                          predicted_canonical_id=mapping[index], model_output_index=index,
                          confidence=float(probabilities[index]), split='val',
                          probabilities_model_order=json.dumps(probabilities.tolist()),
                          probabilities_canonical_order=json.dumps(canonical_probabilities.tolist()))
            record.update({f'p_{name}': float(canonical_probabilities[i]) for i, name in enumerate(CLASSES)})
            records.append(record)
    predictions = pd.DataFrame(records)
    require(len(predictions) == len(validation), 'Incomplete validation predictions')
    metrics, per_class, matrix = classification_metrics(predictions.true_canonical_id, predictions.predicted_canonical_id)
    out = Path(out)
    predictions.to_csv(out / 'validation_predictions.csv', index=False)
    predictions[predictions.true_class.ne(predictions.predicted_class)].to_csv(out / 'validation_errors.csv', index=False)
    per_class.to_csv(out / 'per_class_metrics.csv', index=False)
    pd.DataFrame([metrics]).to_csv(out / 'validation_metrics.csv', index=False)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(out / 'confusion_matrix.csv')
    write_json(out / 'validation_metrics.json', dict(**metrics, per_class=per_class.to_dict('records')))
    write_json(out / 'label_mappings.json', dict(canonical=dict(enumerate(CLASSES)), model_output=model.names,
                                               model_output_to_canonical=dict(enumerate(mapping))))
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(matrix, cmap='Blues')
    ax.set(xticks=range(6), yticks=range(6), xticklabels=CLASSES, yticklabels=CLASSES,
           xlabel='Predicted', ylabel='True', title='Internal validation (canonical order)')
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center')
    fig.savefig(out / 'confusion_matrix.png', dpi=150)
    plt.close(fig)
    return metrics


def run(smoke=False):
    environment_record = environment()
    seeds(CONFIG['training']['seed'])
    dataset, rows, fingerprint = prepare(smoke=smoke)
    checkpoint, model = pretrained()
    settings = dict(CONFIG['training'])
    if smoke:
        settings.update(epochs=1, patience=1)
    mode = 'smoke_test' if smoke else 'official_runs'
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    name = f'model_a_seed{settings["seed"]}_{fingerprint[:12]}_{timestamp}'
    out = HERE / 'outputs' / mode / name
    out.mkdir(parents=True, exist_ok=False)
    device = 0 if torch.cuda.is_available() else 'cpu'
    write_json(out / 'run_manifest.json', dict(status='STARTED', mode=mode, dataset_fingerprint=fingerprint,
               checkpoint_source=CONFIG['checkpoint_url'], pretrained_sha256=sha(checkpoint),
               dataset_path=str(dataset), environment=environment_record, configuration=settings,
               code_sha256=sha(__file__), config_sha256=sha(HERE / 'config/model_a.json'),
               notebook_sha256=sha(HERE / 'Model_A_YOLO26_6Class_Training.ipynb')))
    snapshot = out / 'source_snapshot'
    snapshot.mkdir()
    for source in [Path(__file__), HERE / 'config/model_a.json', HERE / 'Model_A_YOLO26_6Class_Training.ipynb']:
        shutil.copyfile(source, snapshot / source.name)
    write_json(out / 'environment_packages.json',
               {d.metadata['Name']: d.version for d in metadata.distributions() if d.metadata['Name']})
    shutil.copyfile(dataset / 'dataset_manifest.csv', out / 'dataset_manifest.csv')
    shutil.copyfile(dataset / 'dataset_fingerprint.json', out / 'dataset_fingerprint.json')
    if (dataset / 'explicit_exclusions.csv').exists():
        shutil.copyfile(dataset / 'explicit_exclusions.csv', out / 'explicit_exclusions.csv')
    trace = []
    def record_epoch(trainer):
        # Final-validation callback can repeat epoch+1; only actual CSV rows count.
        csv_path = Path(trainer.save_dir) / 'results.csv'
        completed = len(read_csv(csv_path)) if csv_path.exists() else 0
        if completed and len(trace) < completed:
            trace.append(dict(epoch=completed, fitness=float(trainer.fitness),
                              best_fitness=float(trainer.best_fitness), batch=int(trainer.batch_size),
                              learning_rates=[g['lr'] for g in trainer.optimizer.param_groups]))
            write_json(out / 'epoch_trace.json', trace)
    model.add_callback('on_fit_epoch_end', record_epoch)
    from ultralytics.utils import LOGGER
    log_handler = logging.FileHandler(out / 'training.log', encoding='utf8')
    LOGGER.addHandler(log_handler)
    try:
        model.train(trainer=ModelATrainer, data=str(dataset), project=str(out), name='fit',
                    exist_ok=False, device=device, **settings)
        fit = Path(model.trainer.save_dir)
        require(fit == out / 'fit', 'Framework changed run directory unexpectedly')
        best = fit / 'weights/best.pt'
        require(best.is_file(), 'Best internal-validation checkpoint not produced')
        history = pd.read_csv(fit / 'results.csv')
        history.columns = history.columns.str.strip()
        history.to_csv(out / 'training_history.csv', index=False)
        # Framework replaces best.pt on tied fitness, so report the last maximum.
        values = history['metrics/accuracy_top1'].to_numpy()
        best_epoch = int(np.flatnonzero(values == values.max())[-1]) + 1
        metrics = evaluate(best, rows, out, settings['imgsz'], settings['batch'], device)
        manifest = json.loads((out / 'run_manifest.json').read_text())
        manifest.update(status='COMPLETE', completed_epochs=len(history), best_epoch=best_epoch,
                        best_selection='highest internal-validation top1; last tie',
                        stopping_reason='maximum_epochs' if len(history) == settings['epochs'] else
                            'early_stop_or_framework_stop; inspect fit/training log and epoch trace',
                        early_stopper_best_epoch=getattr(model.trainer.stopper, 'best_epoch', None),
                        final_checkpoints={str(p.relative_to(out)): sha(p) for p in (fit / 'weights').glob('*.pt')},
                        metrics=metrics, scientific_results='TECHNICAL_SMOKE_ONLY' if smoke else 'MODEL_A_INTERNAL_VALIDATION')
        write_json(out / 'run_manifest.json', manifest)
    except Exception as error:
        manifest = json.loads((out / 'run_manifest.json').read_text())
        manifest.update(status='FAILED', error=f'{type(error).__name__}: {error}')
        write_json(out / 'run_manifest.json', manifest)
        raise
    finally:
        LOGGER.removeHandler(log_handler)
        log_handler.close()
    return out
