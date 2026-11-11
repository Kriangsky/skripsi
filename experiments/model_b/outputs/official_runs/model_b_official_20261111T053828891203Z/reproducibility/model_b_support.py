"""Data/protocol helpers only. Training, prediction and experiment selection live in the notebook."""
from copy import copy
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
os.environ.setdefault('YOLO_CONFIG_DIR', str(HERE / 'outputs/ultralytics_config'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(ROOT / 'experiments/model_a'))
import model_a as a
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics.models.yolo.classify import ClassificationTrainer, ClassificationValidator

CLASSES = a.ALL_CLASSES
sha, read_csv, write_json, require = a.sha, a.read_csv, a.write_json, a.require
transform, LetterboxRGB, Predictor = a.transform, a.LetterboxRGB, a.ModelAPredictor


def mapping(names):
    names = dict(enumerate(names)) if isinstance(names, list) else {int(k): v for k, v in names.items()}
    require(set(names) == set(range(8)) and set(names.values()) == set(CLASSES), 'Expected all eight output classes')
    return [CLASSES.index(names[i]) for i in range(8)]


def metrics(truth, predicted, labels=range(8)):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    require(truth.ndim == 1 and len(truth) > 0 and truth.shape == predicted.shape, 'Invalid prediction lengths')
    require(np.isin(truth, range(8)).all() and np.isin(predicted, range(8)).all(), 'Invalid canonical IDs')
    matrix = np.zeros((8, 8), dtype=int)
    np.add.at(matrix, (truth.astype(int), predicted.astype(int)), 1)
    tp = matrix.diagonal()
    precision = np.divide(tp, matrix.sum(0), out=np.zeros(8), where=matrix.sum(0) != 0)
    recall = np.divide(tp, matrix.sum(1), out=np.zeros(8), where=matrix.sum(1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(8), where=precision + recall != 0)
    ix = list(labels)
    per_class = pd.DataFrame(dict(class_name=CLASSES, canonical_id=range(8), precision=precision,
                                 recall=recall, f1=f1, support=matrix.sum(1))).iloc[ix]
    return dict(top1_accuracy=float(tp.sum() / matrix.sum()), macro_precision=float(precision[ix].mean()),
                macro_recall=float(recall[ix].mean()), macro_f1=float(f1[ix].mean()), samples=len(truth)), per_class, matrix


def audit_dataset(verify_files=True):
    lock = read_csv(a.LOCK)
    a.verify_inventory(lock)
    require(lock.equals(read_csv(a.CACHE / 'inventory.csv')), 'Inventory differs from split lock')
    latest = __import__('json').loads((a.HERE / 'reports/latest_official_run.json').read_text())
    official = Path(latest['run_directory'])
    original = read_csv(official / 'dataset_manifest.csv')
    extra = read_csv(a.SELECTION)
    extra = extra[extra.source.eq('train') & extra.class_name.isin(CLASSES[6:])]
    # Reuse existing new-class selections; fill uncovered videos from cached frames with Model A's ranking.
    from frame_selection import rank
    catalogue = read_csv(a.CACHE / 'candidates.csv')
    missing = lock[lock.source.eq('train') & lock.class_name.isin(CLASSES[6:]) & ~lock.sample_id.isin(extra.sample_id)]
    additions = []
    for source in missing.to_dict('records'):
        choices = catalogue[catalogue.sample_id.eq(source['sample_id'])].to_dict('records')
        require(bool(choices), f'No cached frames for {source["sample_id"]}')
        for c in choices:
            c.update(quality_flags=json.loads(c['quality_flags']), focus=float(c['focus']),
                     num_hands=int(c['num_hands']), frame_index=int(c['frame_index']))
        frame = min(choices, key=rank)
        additions.append(dict(source, candidate_id=frame['candidate_id'], frame_index=str(frame['frame_index']),
                              source_frame_path=frame['frame_path'], source_frame_sha256=frame['frame_sha256']))
    extra = pd.concat([extra, pd.DataFrame(additions)], ignore_index=True)
    columns = ['video', 'sample_id', 'source', 'class_name', 'class_id', 'video_sha256', 'split',
               'candidate_id', 'frame_index', 'source_frame_path', 'source_frame_sha256']
    rows = pd.concat([original[columns], extra[columns]], ignore_index=True).sort_values('sample_id').reset_index(drop=True)
    require(len(rows) == 2400 and set(rows.source) == {'train'}, 'Incomplete/internal-only dataset required')
    for field in ['video', 'sample_id', 'video_sha256', 'candidate_id']:
        require(not rows[field].eq('').any() and not rows[field].duplicated().any(), f'Duplicate/empty {field}')
    require(rows.groupby(['class_name', 'split']).size().to_dict() ==
            {(name, split): n for name in CLASSES for split, n in [('train', 240), ('val', 60)]}, 'Frozen class counts differ')
    frozen = lock.set_index('sample_id', drop=False)
    candidates = read_csv(a.CACHE / 'candidates.csv').set_index('candidate_id')
    for row in rows.itertuples():
        require(row.sample_id in frozen.index, 'Unknown locked sample')
        source = frozen.loc[row.sample_id]
        require(all(getattr(row, k) == source[k] for k in columns[:7]), 'Locked identity/split changed')
        require(row.candidate_id in candidates.index, 'Unknown cached frame')
        frame = candidates.loc[row.candidate_id]
        require(frame.sample_id == row.sample_id and frame.frame_index == row.frame_index and
                frame.frame_path == row.source_frame_path and frame.frame_sha256 == row.source_frame_sha256,
                'Cached frame provenance differs')
    shared = rows[rows.class_name.isin(CLASSES[:6])].set_index('sample_id').sort_index()
    require(shared.equals(original[columns].set_index('sample_id').sort_index()), 'Model A shared frames/splits differ')
    seen_bytes, seen_pixels = {}, {}
    if verify_files:
        for row in rows.itertuples():
            path = a.safe_path(a.CACHE, row.source_frame_path)
            require(sha(path) == row.source_frame_sha256, f'Frame hash changed: {row.sample_id}')
            raw = a.safe_path(ROOT / '04_dataset', row.video)
            require(sha(raw) == row.video_sha256, f'Video hash changed: {row.sample_id}')
            with Image.open(path) as image:
                pixels = np.asarray(image.convert('RGB'))
            pixel_hash = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
            for seen, digest in [(seen_bytes, row.source_frame_sha256), (seen_pixels, pixel_hash)]:
                require(digest not in seen or seen[digest] == row.split, 'Cross-split frame leakage')
                seen[digest] = row.split
    sources = [a.LOCK, a.CACHE / 'inventory.csv', a.CACHE / 'candidates.csv', a.SELECTION,
               official / 'dataset_manifest.csv', a.AUTO_SELECTION, a.HERE / 'config/model_a.json']
    return rows, dict(shared_samples_exact=len(shared), class_counts=rows.groupby(['class_name', 'split']).size().to_string(),
                      source_hashes={str(p.relative_to(ROOT)): sha(p) for p in sources},
                      model_a_run=str(official), files_verified=verify_files,
                      new_frame_source=str(a.SELECTION), frame_regeneration=False,
                      existing_new_class_selections=600-len(additions), new_selections_from_existing_cache=len(additions),
                      signer_independent=False, external_data_used=False)


class MacroF1Validator(ClassificationValidator):
    def build_dataset(self, img_path):
        dataset = super().build_dataset(img_path)
        dataset.torch_transforms = transform(self.args.imgsz)
        return dataset

    def get_stats(self):
        stats = super().get_stats()
        truth = torch.cat(self.targets).numpy().reshape(-1)
        predicted = torch.cat(self.pred).numpy()[:, 0]
        result, _, _ = metrics(truth, predicted)
        stats['metrics/macro_f1'] = result['macro_f1']
        stats['fitness'] = result['macro_f1']
        return stats


class ModelBTrainer(ClassificationTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        dataset = super().build_dataset(img_path, mode, batch)
        dataset.torch_transforms = transform(self.args.imgsz)
        require(sorted(str(x[0]) for x in dataset.samples) == sorted(str(p) for p in Path(img_path).glob('*/*.jpg')),
                'Loader changed image coverage')
        require(set(dataset.base.classes) == set(CLASSES), 'Unexpected class folders')
        names = {name: int(i) for i, name in self.data['names'].items()}
        require(all(names[Path(x[0]).parent.name] == int(x[1]) for x in dataset.samples), 'Loader label mismatch')
        return dataset

    def get_validator(self):
        return MacroF1Validator(self.test_loader, self.save_dir, args=copy(self.args), _callbacks=self.callbacks)

    def optimizer_step(self):
        super().optimizer_step()
        self.actual_optimizer_steps += 1

    def _setup_train(self):
        self.actual_optimizer_steps = 0
        super()._setup_train()
        require(self.batch_size == self.args.batch, 'Automatic batch change forbidden')
        require(type(self.optimizer).__name__ == self.args.optimizer, 'Optimizer silently changed')
        require(self.model.model[-1].linear.out_features == 8, 'Expected eight logits')
        write_json(Path(self.save_dir) / 'actual_training_config.json', dict(
            args=vars(self.args), optimizer=str(self.optimizer), accumulation=int(self.accumulate),
            optimizer_groups=[{k: v for k, v in g.items() if k != 'params'} for g in self.optimizer.param_groups],
            scheduler='cosine' if self.args.cos_lr else 'linear', train_samples=len(self.train_loader.dataset),
            val_samples=len(self.test_loader.dataset), output_to_canonical=mapping(self.model.names),
            preprocessing=str(transform(self.args.imgsz)), selection='validation macro-F1; framework rounds to 5 decimals; last tied epoch'))
        (Path(self.save_dir) / 'architecture.txt').write_text(str(self.model), encoding='utf8')
