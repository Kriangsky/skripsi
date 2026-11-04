"""Local-only controlled COCO -> 6 / COCO -> 8 pilot, with fail-closed quality gates."""
import argparse
from datetime import datetime, timezone
import importlib.metadata as meta
import json
import logging
import os
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np
import pandas as pd
import yaml

import bisindo_pipeline as b
import frame_candidates as f
import pilot_study as human_pilot

CLASSES = b.CLASSES
BASE = b.BASE
TARGETS = {'train': 60, 'val': 20}
REJECT_REASONS = ['preparation_pose', 'resting_pose', 'transition_pose', 'ambiguous',
                  'temporal_ambiguity', 'clipped_hands', 'motion_blur', 'wrong_gesture', 'other']
PILOT_CONFIG = dict(epochs=30, imgsz=640, batch=16, seed=42, deterministic=True,
    optimizer='SGD', lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005,
    warmup_epochs=3., warmup_momentum=.8, warmup_bias_lr=.1, cos_lr=False, patience=0,
    nbs=16, device=0, workers=0, amp=False, compile=False, pretrained=True, resume=False,
    single_cls=False, fraction=1., cache=False, rect=False, multi_scale=0.,
    degrees=0., translate=0., scale=0., shear=0., perspective=0., fliplr=0., flipud=0.,
    hsv_h=0., hsv_s=0., hsv_v=0., bgr=0., mosaic=0., mixup=0., cutmix=0.,
    copy_paste=0., close_mosaic=0, erasing=0., auto_augment=None,
    val=True, save=True, plots=True, exist_ok=False, verbose=True)
INFERENCE = dict(mode='end_to_end_nms_free', fixed_confidence=.25, matching_iou=.5,
                 ap_confidence=.001, iou=.7, max_det=300, augment=False, classes=None,
                 agnostic_nms=False, imgsz=640, device=0, half=False)
LOG = logging.getLogger(__name__)


def csv(path):
    return pd.read_csv(path, keep_default_na=False)


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_json(path, default=None):
    return json.loads(Path(path).read_text(encoding='utf-8-sig')) if Path(path).exists() else default


def select_samples(inventory, reviews, human_ids):
    """Fixed order, no replacement. Every inventory row remains accounted for."""
    required = {'sample_id', 'video', 'source', 'split', 'class_name', 'class_id', 'video_sha256'}
    if not required.issubset(inventory):
        raise ValueError('Incomplete inventory')
    for field in ['sample_id', 'video', 'video_sha256']:
        if inventory[field].duplicated().any():
            raise ValueError(f'Duplicate {field}')
    for row in inventory.itertuples():
        if row.class_name not in CLASSES or str(row.class_id) != str(CLASSES.index(row.class_name)):
            raise ValueError('Class mapping changed')
        if row.source not in ['train', 'test'] or row.split not in ['train', 'val', 'test'] or (row.source == 'test') != (row.split == 'test'):
            raise ValueError('Source/split mismatch')
        parts = str(row.video).replace('\\', '/').split('/')
        if parts[0] != row.source or any(part in ['.', '..'] for part in parts):
            raise ValueError('Source/path mismatch or traversal')
    eligible = inventory[inventory.source.eq('train') & inventory.split.isin(TARGETS)]
    selected, counts, blockers = [], [], []
    for cls in CLASSES:
        for split, target in TARGETS.items():
            pool = eligible[eligible.class_name.eq(cls) & eligible.split.eq(split)].sort_values('sample_id').sample(frac=1, random_state=42)
            valid = pool[pool.sample_id.map(lambda sid: reviews.get(sid, {}).get('status') == 'positive')]
            if split == 'val':
                valid = valid[~valid.sample_id.isin(human_ids)]
            chosen = valid.head(target)
            selected.extend(chosen.to_dict('records'))
            reviewed_n = sum(reviews.get(sid, {}).get('status') in ['positive', 'excluded'] for sid in pool.sample_id)
            rejected_n = sum(reviews.get(sid, {}).get('status') == 'excluded' for sid in pool.sample_id)
            counts.append(dict(class_name=cls, split=split, inventory_n=len(pool), target=target,
                available_positive=len(valid), selected_n=len(chosen),
                reviewed_n=reviewed_n, reviewed_rejected_n=rejected_n,
                rejection_rate=rejected_n/reviewed_n if reviewed_n else None,
                unreviewed_n=sum(reviews.get(sid, {}).get('status') not in ['positive','excluded'] for sid in pool.sample_id)))
            if len(chosen) < target:
                blockers.append(f'{cls}/{split}: {len(valid)} verified positives available; target {target}')
    selected = pd.DataFrame(selected, columns=inventory.columns)
    chosen_ids = set(selected.sample_id)
    excluded = []
    for row in inventory.to_dict('records'):
        if row['sample_id'] in chosen_ids:
            continue
        review = reviews.get(row['sample_id'], {})
        if row['source'] != 'train':
            reason = 'kaggle_external_not_allowed'
        elif row['split'] == 'val' and row['sample_id'] in human_ids:
            reason = 'human_pilot_development'
        elif review.get('status') == 'excluded':
            reason = review.get('reason', 'explicit_human_rejection')
        elif review.get('status') == 'positive':
            reason = 'capacity_not_selected'
        else:
            reason = 'needs_semantic_review'
        excluded.append(dict(row, reason=reason, review_note=review.get('note', ''),
                             explicitly_rejected=review.get('status') == 'excluded'))
    return selected, pd.DataFrame(counts), pd.DataFrame(excluded), blockers


def metadata_issues(selected, metadata):
    if not {'video', 'signer', 'session'}.issubset(metadata):
        return ['Participant metadata requires video, signer, session']
    if metadata.video.duplicated().any():
        return ['Duplicate video in participant metadata']
    lookup = metadata.set_index('video')
    errors, groups = [], {}
    for row in selected.itertuples():
        if row.video not in lookup.index:
            errors.append(f'{row.sample_id}: signer/session metadata missing')
            continue
        values = lookup.loc[row.video]
        if any(not str(values[k]).strip() for k in ['signer', 'session']):
            errors.append(f'{row.sample_id}: signer/session metadata incomplete')
            continue
        if str(getattr(row, 'signer', '')).strip() and str(row.signer) != str(values.signer):
            errors.append(f'{row.sample_id}: signer conflicts with frozen inventory')
        for field in ['signer', 'session']:
            groups.setdefault((field, str(values[field]).strip()), set()).add(row.split)
    for (field, value), splits in groups.items():
        if len(splits) > 1:
            errors.append(f'{field} overlap train/validation: {value}')
    return errors


def label_line(class_id, box):
    if isinstance(class_id, bool) or int(class_id) != class_id or not 0 <= int(class_id) < 8:
        raise ValueError('Invalid class ID')
    values = [f'{value:.8f}' for value in b.yolo_box(box)]
    if any(float(value) <= 0 for value in values[2:]):
        raise ValueError('Box rounds to zero at label precision')
    return str(int(class_id)) + ' ' + ' '.join(values) + '\n'


def validate_positive(decision, candidate):
    if any(decision.get(k) is not True for k in ['gesture_verified', 'box_verified', 'all_required_hands_visible']):
        raise ValueError('Each positive needs semantic, relevant-hands and box verification')
    if decision.get('semantic_status') != 'positive' or decision.get('reason', ''):
        raise ValueError('Preparation/resting/transition/ambiguous/clipped frames cannot be positives')
    box = decision.get('box')
    if not isinstance(box, list) or len(box) != 4 or any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in box):
        raise ValueError('Numeric reviewed box required')
    b.yolo_box(box)
    if candidate['quality_flags'] and not str(decision.get('quality_override_reason', '')).strip():
        raise ValueError('Technical flags require explicit human justification; actual clipped required hands must be rejected')
    if decision.get('candidate_id') != candidate['candidate_id']:
        raise ValueError('Candidate ID mismatch')
    if decision['candidate_id'] in decision.get('rejected_candidates', {}):
        raise ValueError('Selected candidate remains rejected')


def local_yolo(work):
    """No HUB, account, hosted inference or paid API. Keep all settings inside this pilot."""
    config = Path(work).resolve() / 'ultralytics_config'
    config.mkdir(parents=True, exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(config)
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['WANDB_MODE'] = 'disabled'
    from ultralytics import YOLO
    from ultralytics.utils import SETTINGS, USER_CONFIG_DIR
    if not Path(USER_CONFIG_DIR).resolve().is_relative_to(config):
        raise RuntimeError('Ultralytics already imported with different settings directory; run pilot CLI in a fresh process')
    disable = ['sync', 'hub', 'wandb', 'mlflow', 'neptune', 'comet', 'clearml', 'tensorboard', 'dvc']
    SETTINGS.update({key: False for key in disable if key in SETTINGS})
    return YOLO


def checkpoint_info(source):
    info = read_json(Path(source) / 'checkpoint.json')
    if not info or info.get('filename') != 'yolo26n.pt' or info.get('repo_id') != 'Ultralytics/YOLO26' or info.get('revision') != b.HF_REVISION:
        raise ValueError('Existing pinned official COCO YOLO26n checkpoint required')
    path = Path(info['path'])
    if not path.is_file() or b.sha(path) != info['sha256']:
        raise ValueError('Local pretrained checkpoint missing or changed; no automatic download')
    return info


def set_inference_mode(model):
    head = model.model.model[-1]
    if not model.model.yaml.get('end2end') or getattr(head, 'one2one_cv2', None) is None:
        raise ValueError('YOLO26 end-to-end architecture/weights required')
    # Older checkpoint heads lack the newer _end2end runtime attribute.
    model.model.end2end = True
    if not model.model.end2end:
        raise ValueError('Cannot activate end-to-end inference mode')


def probe_environment(source, work):
    result = dict(ready=False, blockers=[], local_only=True, paid_api=False, executable=sys.executable)
    try:
        YOLO = local_yolo(work)
        import torch
        from types import SimpleNamespace
        from ultralytics.cfg import get_cfg
        from ultralytics.engine.trainer import BaseTrainer
        info = checkpoint_info(source)
        args = get_cfg(overrides=PILOT_CONFIG)
        model = YOLO(info['path'])
        set_inference_mode(model)
        if model.task != 'detect' or len(model.names) != 80 or model.names[0] != 'person' or not model.model.end2end:
            raise ValueError('Expected 80-class COCO YOLO26 end-to-end detector')
        optimizer = BaseTrainer.build_optimizer(SimpleNamespace(), model.model, name=args.optimizer,
            lr=args.lr0, momentum=args.momentum, decay=args.weight_decay)
        if type(optimizer).__name__ != 'SGD':
            raise ValueError('Actual optimizer differs from requested SGD')
        optimizer_groups = [{k: group.get(k) for k in ['param_group', 'lr', 'momentum', 'weight_decay', 'nesterov']}
                            for group in optimizer.param_groups]
        if any(group['lr'] != args.lr0 or group['momentum'] != args.momentum or group['nesterov'] is not True for group in optimizer_groups):
            raise ValueError('Actual SGD parameter-group hyperparameters differ from controlled configuration')
        model.model.eval()
        with torch.inference_mode():
            output = model.model(torch.zeros(1, 3, 64, 64))
        prediction = output[0] if isinstance(output, tuple) else output
        if prediction.ndim != 3 or prediction.shape[-1] != 6:
            raise ValueError('YOLO26 dry forward did not return end-to-end box/conf/class output')
        result.update(checkpoint=info, optimizer=type(optimizer).__name__,
            optimizer_groups=optimizer_groups,
            optimizer_param_groups=len(optimizer.param_groups), end2end=bool(model.model.end2end),
            dry_forward_shape=list(prediction.shape), dry_forward_input='synthetic zeros, 1x3x64x64; no dataset images',
            cuda_available=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            cuda_runtime=torch.version.cuda, python=sys.version,
            packages={key: meta.version(key) for key in ['ultralytics', 'torch', 'torchvision', 'numpy', 'pandas', 'mediapipe', 'PyYAML']})
        if not torch.cuda.is_available():
            result['blockers'].append('CUDA GPU unavailable; CPU training forbidden')
        del optimizer, model
        result['ready'] = not result['blockers']
    except Exception as exc:
        result['blockers'].append(f'Local API/environment: {type(exc).__name__}: {exc}')
    dump(Path(work) / 'reports/environment_probe.json', result)
    return result


def source_snapshot(source):
    return {name: b.sha(Path(source) / name) for name in
            ['split_lock.csv', 'candidates.csv', 'annotations.csv', 'candidate_config.json', 'checkpoint.json']}


def human_development_ids(root, inventory):
    path = Path(root) / 'experiment_yolo26_pilot/pilot_manifest_private.csv'
    if not path.exists():
        raise ValueError('Frozen human pilot manifest missing; cannot exclude development samples from validation')
    pilot = csv(path)
    return set(pilot.video_id) | set(inventory.loc[inventory.video_sha256.isin(pilot.video_sha256), 'sample_id'])


def participant_metadata(root, work):
    paths = [Path(work) / 'participant_metadata.csv', Path(root) / 'signer_metadata.csv']
    for path in paths:
        if path.exists():
            return pd.read_csv(path, dtype=str, keep_default_na=False), str(path), b.sha(path)
    return pd.DataFrame(columns=['video', 'signer', 'session']), None, None


def review_queue(inventory, reviews, human_ids):
    chosen = []
    for cls in CLASSES:
        for split, target in TARGETS.items():
            pool = inventory[inventory.class_name.eq(cls) & inventory.split.eq(split)].sort_values('sample_id').sample(frac=1, random_state=42)
            if split == 'val':
                pool = pool[~pool.sample_id.isin(human_ids)]
            known = pool[pool.sample_id.map(lambda sid: reviews.get(sid, {}).get('status') == 'positive')]
            pending = pool[pool.sample_id.map(lambda sid: reviews.get(sid, {}).get('status') not in ['positive', 'excluded'])]
            chosen.extend(known.head(target).sample_id)
            chosen.extend(pending.head(max(0, target-len(known))).sample_id)
    return inventory[inventory.sample_id.isin(chosen)].sort_values('sample_id')


def render_review(root, source, work, queue, catalog, reviews):
    """Reuse the existing per-candidate canvas/hand-union editor; no mass approval."""
    records, grouped = [], {}
    for candidate in catalog.values():
        grouped.setdefault(candidate['sample_id'], []).append(candidate)
    for row in queue.to_dict('records'):
        options = []
        for candidate in grouped.get(row['sample_id'], []):
            c = {k: v for k, v in candidate.items() if k != 'landmarks'}
            c['frame_path'] = b.safe_path(source, c['frame_path']).as_uri() if c['frame_path'] else ''
            c['hand_bounds'] = [[*np.min(hand, axis=0).tolist(), *np.max(hand, axis=0).tolist()]
                                for hand in candidate['landmarks'] if hand]
            options.append(c)
        eligible = [c for c in options if c['quality_status'] == 'eligible']
        prior = reviews.get(row['sample_id'], {})
        records.append(dict(sample_id=row['sample_id'], video=row['video'], class_name=row['class_name'],
            class_id=int(row['class_id']), source='train', split=row['split'],
            recommended=max(eligible, key=lambda c: (c['focus'], -c['frame_index']))['candidate_id'] if eligible else '',
            selected=prior.get('candidate_id', ''), selected_box=prior.get('box'), extended=True,
            rejection_reason=prior.get('reason', 'gesture_not_yet_verified'),
            rejected_candidates=prior.get('rejected_candidates', {}), quality_override_reason=prior.get('quality_override_reason', ''),
            review_note=prior.get('note', ''), candidates=options,
            box_verified=prior.get('box_verified', False), all_required_hands_visible=prior.get('all_required_hands_visible', False),
            semantic_status=prior.get('semantic_status', ''),
            contact_sheet_uri=(Path(source) / 'contact_sheets' / f"{row['sample_id']}.jpg").resolve().as_uri()))
    payload = dict(annotations_sha256=b.sha(work / 'review_queue.csv'),
                   candidates_sha256=b.sha(source / 'candidates.csv'), records=records)
    template = (Path(__file__).parent / 'frame_review_template.html').read_text(encoding='utf-8')
    changes = {
        "sheet.src='contact_sheets/'+r.sample_id+'.jpg'": "sheet.src=r.contact_sheet_uri",
        '<option value="test">External test</option>': '',
        '<option value="other">Lainnya': '<option value="ambiguous">Ambiguous</option><option value="temporal_ambiguity">Temporal ambiguity</option><option value="other">Lainnya',
        '<p id="editorMessage"': '<p><label><input id="handsVerified" type="checkbox"> Semua tangan yang diperlukan terlihat utuh dan masuk box</label><label><input id="boxVerified" type="checkbox"> Saya memeriksa box pada gambar ini</label></p><p id="editorMessage"',
        "function open(r){": "function open(r){$('handsVerified').checked=false;$('boxVerified').checked=false;",
        "function setActive(candidate){": "function setActive(candidate){$('handsVerified').checked=false;$('boxVerified').checked=false;",
        "if(!active||!image||!valid(box)||!$('verified').checked)": "if(!active||!image||!valid(box)||!$('verified').checked||!$('handsVerified').checked||!$('boxVerified').checked)",
        "Object.assign(s,{candidate_id:active.candidate_id,box:box.slice(),gesture_verified:true,": "Object.assign(s,{candidate_id:active.candidate_id,box:box.slice(),gesture_verified:true,box_verified:true,all_required_hands_visible:true,semantic_status:'positive',",
        "reason,note:reason,request_more:more": "semantic_status:'excluded',reason:$('rejectReason').value,note:reason,request_more:false",
        "!s.gesture_verified||s.rejected_candidates": "!s.gesture_verified||s.box_verified!==true||s.all_required_hands_visible!==true||s.semantic_status!=='positive'||s.rejected_candidates",
        "output.push({sample_id:r.sample_id,class_id:r.class_id,...s})": "output.push({sample_id:r.sample_id,class_id:r.class_id,...s,timestamp:new Date().toISOString()})",
        "gesture_verified:Boolean(r.selected),quality_override_reason:": "gesture_verified:Boolean(r.selected),box_verified:r.box_verified===true,all_required_hands_visible:r.all_required_hands_visible===true,semantic_status:r.semantic_status||'',quality_override_reason:",
        "kind:'frame_selection'": "kind:'training_pilot_review'",
        "'bisindo_frame_review.json'": "'training_pilot_review.json'",
    }
    for old, new in changes.items():
        if old not in template:
            raise ValueError(f'Existing review template changed: missing {old}')
        template = template.replace(old, new)
    start, end = template.index("$('bulk').onclick="), template.index("$('export').onclick=")
    template = template[:start] + "$('bulk').hidden=true;$('bulkConfirm').disabled=true;$('more').disabled=true;\n" + template[end:]
    template = template.replace('<h1>BISINDO — review kandidat frame, bukan auto-label gesture</h1>',
        '<h1>Controlled training pilot — internal train/validation saja</h1><p>Rekomendasi fokus hanya proposal teknis. '
        'Positive wajib diverifikasi satu per satu, termasuk seluruh tangan relevan dan box. '
        'Jika frasa tidak dapat diwakili frame, pilih temporal_ambiguity lalu tandai needs_review. '
        'Tidak ada auto-label, bulk approval, atau ekstraksi kandidat baru. Ekspor lalu impor lewat training_pilot.py.</p>')
    html = template.replace('__DATASET_JSON__', json.dumps(payload, allow_nan=False).replace('<', '\\u003c'))
    (work / 'review/frame_review.html').write_text(html, encoding='utf-8')
    # The existing video URL is relative to the work folder; this HTML lives in review/.
    html = html.replace("'../'+r.video", json.dumps(Path(root).resolve().as_uri() + '/') + '+r.video')
    (work / 'review/frame_review.html').write_text(html, encoding='utf-8')


def validate_reviewed_sources(root, source, selected, reviews, catalog):
    errors = []
    frame_hashes = [reviews.get(sid, {}).get('frame_sha256', '') for sid in selected.sample_id]
    if len(frame_hashes) != len(set(frame_hashes)):
        errors.append('Duplicate selected frame content; no oversampling or duplicated positives allowed')
    for row in selected.to_dict('records'):
        try:
            d = reviews[row['sample_id']]
            c = catalog[d['candidate_id']]
            validate_positive(d, c)
            if not str(d.get('reviewer', '')).strip() or datetime.fromisoformat(d.get('timestamp', '').replace('Z', '+00:00')).utcoffset() is None:
                raise ValueError('Named reviewer and timezone timestamp required')
            if c['sample_id'] != row['sample_id'] or b.sha(b.safe_path(root, row['video'])) != row['video_sha256']:
                raise ValueError('Video/candidate identity or hash changed')
            image_path = b.safe_path(source, c['frame_path'])
            if b.sha(image_path) != c['frame_sha256'] or d['frame_sha256'] != c['frame_sha256']:
                raise ValueError('Selected image changed since review')
            image = cv2.imread(str(image_path))
            if image is None or image.shape[:2] != (c['height'], c['width']):
                raise ValueError('Selected image invalid')
            label_line(int(row['class_id']), d['box'])
        except (KeyError, ValueError, OSError) as exc:
            errors.append(f"{row['sample_id']}: {exc}")
    return errors


def release_datasets(work, source, selected, reviews, catalog, metadata, metadata_sha):
    records = []
    people = metadata.set_index('video')
    for row in selected.to_dict('records'):
        d = reviews[row['sample_id']]
        c = catalog[d['candidate_id']]
        records.append(dict(row, signer=str(people.loc[row['video'], 'signer']),
            session=str(people.loc[row['video'], 'session']), candidate_id=d['candidate_id'],
            frame_sha256=c['frame_sha256'], box=d['box'], quality_flags=c['quality_flags'],
            reviewer=d['reviewer'], review_timestamp=d['timestamp'],
            quality_override_reason=d.get('quality_override_reason', '')))
    fingerprint = human_pilot.digest(dict(records=records, metadata_sha256=metadata_sha))
    release_path = work / 'datasets/release.json'
    previous = read_json(release_path)
    if previous:
        if previous['fingerprint'] != fingerprint:
            raise ValueError('Released dataset changed; use a new work directory, never overwrite')
        validate_release(work)
        return previous
    manifest_path = work / 'datasets/manifest.csv'
    pd.DataFrame(records).assign(box=lambda d: d.box.map(json.dumps), quality_flags=lambda d: d.quality_flags.map(json.dumps)).to_csv(manifest_path, index=False)
    for name, names in [('A', BASE), ('B', CLASSES)]:
        folder = work / 'datasets' / name
        for split in TARGETS:
            for kind in ['images', 'labels']:
                (folder / kind / split).mkdir(parents=True, exist_ok=True)
        for row in records:
            if row['class_name'] not in names:
                continue
            relative = Path(row['split']) / row['sample_id']
            shutil.copy2(b.safe_path(source, catalog[row['candidate_id']]['frame_path']), folder / 'images' / relative.with_suffix('.jpg'))
            (folder / 'labels' / relative.with_suffix('.txt')).write_text(label_line(row['class_id'], row['box']), encoding='utf-8')
        data = dict(path=str(folder.resolve()), train='images/train', val='images/val', names=names)
        (folder / 'pilot.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    common = work / 'datasets/B/common6_val.txt'
    common.write_text(''.join(str((work / 'datasets/B/images/val' / f"{r['sample_id']}.jpg").resolve())+'\n'
        for r in records if r['split'] == 'val' and r['class_name'] in BASE), encoding='utf-8')
    cfg = yaml.safe_load((work / 'datasets/B/pilot.yaml').read_text())
    cfg['val'] = str(common.resolve())
    (work / 'datasets/B/common6.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
    from training_pilot_eval import contact_sheets
    sheets = contact_sheets(work, records)
    release = dict(fingerprint=fingerprint, manifest_sha256=b.sha(manifest_path),
        files={str(p.relative_to(work)): b.sha(p) for p in (work / 'datasets').rglob('*') if p.is_file() and p != release_path},
        contact_sheets={str(p.relative_to(work)): b.sha(p) for p in sheets}, metadata_sha256=metadata_sha)
    dump(release_path, release)
    validate_release(work)
    return release


def validate_release(work):
    work = Path(work)
    release = read_json(work / 'datasets/release.json')
    if not release:
        raise ValueError('Dataset not released')
    for relative, expected in {**release['files'], **release['contact_sheets']}.items():
        if b.sha(b.safe_path(work, relative)) != expected:
            raise ValueError(f'Released file changed: {relative}')
    manifest = csv(work / 'datasets/manifest.csv')
    if b.sha(work / 'datasets/manifest.csv') != release['manifest_sha256']:
        raise ValueError('Released manifest changed')
    if len(manifest) != 640 or manifest.sample_id.duplicated().any() or not manifest.source.eq('train').all():
        raise ValueError('Invalid release count/source/identity')
    for key in ['sample_id', 'video_sha256', 'frame_sha256', 'signer', 'session']:
        grouped = manifest.groupby(key).split.nunique()
        if grouped.gt(1).any():
            raise ValueError(f'Release train/validation leakage: {key}')
    for name, names in [('A', BASE), ('B', CLASSES)]:
        folder = work / 'datasets' / name
        cfg = yaml.safe_load((folder / 'pilot.yaml').read_text())
        if cfg != dict(path=str(folder.resolve()), train='images/train', val='images/val', names=names):
            raise ValueError('Dataset YAML mismatch')
        wanted = manifest[manifest.class_name.isin(names)]
        for split, target in TARGETS.items():
            rows = wanted[wanted.split.eq(split)]
            if not rows.groupby('class_name').size().reindex(names, fill_value=0).eq(target).all():
                raise ValueError('Unbalanced/insufficient released dataset')
            expected_ids = set(rows.sample_id)
            for kind, ext in [('images', '.jpg'), ('labels', '.txt')]:
                if {p.stem for p in (folder / kind / split).glob('*'+ext)} != expected_ids:
                    raise ValueError('Unexpected/missing dataset image or label')
        for row in wanted.itertuples():
            if str(row.class_id) != str(CLASSES.index(row.class_name)):
                raise ValueError('Released class mapping mismatch')
            path = folder / 'labels' / row.split / f'{row.sample_id}.txt'
            if path.read_text() != label_line(int(row.class_id), json.loads(row.box)):
                raise ValueError('Released label differs from reviewed box/class')
            image = folder / 'images' / row.split / f'{row.sample_id}.jpg'
            if b.sha(image) != row.frame_sha256:
                raise ValueError('Released image differs from reviewed candidate')
            if name == 'A':
                for kind, ext in [('images', 'jpg'), ('labels', 'txt')]:
                    relative = Path(kind) / row.split / f'{row.sample_id}.{ext}'
                    if b.sha(folder / relative) != b.sha(work / 'datasets/B' / relative):
                        raise ValueError('Shared A/B image or annotation differs')
    common_cfg = yaml.safe_load((work / 'datasets/B/common6.yaml').read_text())
    if common_cfg['names'] != CLASSES or 'classes' in common_cfg:
        raise ValueError('B common6 must retain all eight prediction classes')
    common_paths = (work / 'datasets/B/common6_val.txt').read_text().splitlines()
    expected = {str((work / 'datasets/B/images/val' / f'{r.sample_id}.jpg').resolve())
                for r in manifest[manifest.split.eq('val') & manifest.class_name.isin(BASE)].itertuples()}
    if len(common_paths) != 120 or set(common_paths) != expected:
        raise ValueError('Shared validation list mismatch')
    return release


def prepare(root, source, work, probe=True):
    root, source, work = Path(root).resolve(), Path(source).resolve(), Path(work).resolve()
    protected = [root / x for x in ['train', 'test', 'experiment_yolo26_v2', 'experiment_yolo26_v3', 'experiment_yolo26_pilot']]
    if work == root or source.is_relative_to(work) or any(work.is_relative_to(p) for p in protected):
        raise ValueError('Output must be a separate training-pilot directory')
    for folder in ['reports', 'review', 'datasets/A', 'datasets/B', 'runs', 'evaluation/error_examples']:
        (work / folder).mkdir(parents=True, exist_ok=True)
    snapshot = source_snapshot(source)
    old_snapshot = read_json(work / 'reports/source_snapshot.json')
    if old_snapshot and old_snapshot != snapshot:
        raise ValueError('Source v3 changed; refusing to mix annotation/candidate snapshots')
    dump(work / 'reports/source_snapshot.json', snapshot)
    inventory = csv(source / 'split_lock.csv')
    config = read_json(source / 'candidate_config.json')
    if config['split_lock_sha256'] != snapshot['split_lock.csv'] or config['quality_heuristics'] != f.QUALITY:
        raise ValueError('Frozen split/technical recipe mismatch')
    human_ids = human_development_ids(root, inventory)
    human_digest = b.sha(root / 'experiment_yolo26_pilot/pilot_manifest_private.csv')
    human_lock_path = work / 'reports/human_pilot_snapshot.json'
    if human_lock_path.exists() and read_json(human_lock_path)['manifest_sha256'] != human_digest:
        raise ValueError('Frozen human-pilot development manifest changed')
    if not human_lock_path.exists():
        dump(human_lock_path, dict(manifest_sha256=human_digest))
    inventory = inventory[inventory.source.eq('train')].copy()
    reviews = read_json(work / 'reviews.json', {})
    if (work / 'reviews.json').exists() and read_json(work / 'reports/review_state.json', {}).get('reviews_sha256') != b.sha(work / 'reviews.json'):
        raise ValueError('Review store changed outside validated import')
    selected, counts, excluded, blockers = select_samples(inventory, reviews, human_ids)
    metadata, metadata_path, metadata_sha = participant_metadata(root, work)
    queue = review_queue(inventory, reviews, human_ids)
    queue.to_csv(work / 'review_queue.csv', index=False)
    if not (work / 'reports/metadata_template.csv').exists():
        queue[['video']].assign(signer='', session='').to_csv(work / 'reports/metadata_template.csv', index=False)
    metadata_errors = metadata_issues(queue if len(selected) != 640 else selected, metadata)
    blockers.extend(metadata_errors)
    catalog = f.catalog(source)
    grouped = {}
    for candidate in catalog.values():
        grouped.setdefault(candidate['sample_id'], []).append(candidate)
    frame_errors = validate_reviewed_sources(root, source, selected, reviews, catalog)
    blockers.extend(frame_errors)
    for row in counts.to_dict('records'):
        ids = set(inventory.loc[inventory.class_name.eq(row['class_name']) & inventory.split.eq(row['split']), 'sample_id'])
        counts.loc[counts.class_name.eq(row['class_name']) & counts.split.eq(row['split']), 'videos_with_technical_candidate'] = sum(
            any(c['quality_status'] == 'eligible' for c in grouped.get(sid, [])) for sid in ids)
    counts.to_csv(work / 'reports/dataset_quality_report.csv', index=False)
    excluded.to_csv(work / 'reports/excluded_samples.csv', index=False)
    inventory.to_csv(work / 'reports/internal_inventory.csv', index=False)
    diagnostics = []
    for row in inventory[inventory.class_name.isin(['apa_kabar', 'halo'])].itertuples():
        d = reviews.get(row.sample_id, {})
        options = grouped.get(row.sample_id, [])
        reason = d.get('reason', '')
        diagnostics.append(dict(sample_id=row.sample_id, class_name=row.class_name, split=row.split,
            review_status=d.get('status', 'pending'), reason=reason,
            preparation_or_transition=reason in ['preparation_pose', 'resting_pose', 'transition_pose'],
            temporal_ambiguity=reason == 'temporal_ambiguity',
            technical_candidate_n=sum(c['quality_status'] == 'eligible' for c in options),
            quality_flags=json.dumps(sorted({flag for c in options for flag in c['quality_flags']})),
            rejected_candidates=json.dumps(d.get('rejected_candidates', {})), note=d.get('note', '')))
    pd.DataFrame(diagnostics).to_csv(work / 'reports/apa_kabar_halo_annotation_diagnostics.csv', index=False)
    render_review(root, source, work, queue, catalog, reviews)
    environment = probe_environment(source, work) if probe else dict(ready=False, blockers=['Environment API probe not run'])
    blockers.extend(environment['blockers'])
    release = None
    if not blockers:
        release = release_datasets(work, source, selected, reviews, catalog, metadata, metadata_sha)
        audit = read_json(work / 'reports/visual_audit.json', {})
        if audit.get('fingerprint') != release['fingerprint'] or audit.get('contact_sheets') != release['contact_sheets'] or not audit.get('reviewer') or audit.get('confirmed') is not True:
            blockers.append('Human visual batch audit pending: inspect reports/contact_sheets and approve-audit explicitly')
    elif not (work / 'datasets/release.json').exists():
        blockers.append('Dataset YAML/release not built: valid positives and independent signer/session metadata required first')
    experiment = dict(kind='preliminary_diagnostic_not_final_thesis', local_only=True, paid_api=False,
        classes=CLASSES, base=BASE, targets=TARGETS, train=PILOT_CONFIG, inference=INFERENCE,
        source_snapshot=snapshot, environment=environment, metadata_path=metadata_path, metadata_sha256=metadata_sha,
        human_pilot_manifest_sha256=human_digest,
        human_development_exclusion='All frozen human-pilot video IDs/content hashes excluded from validation; no human-pilot answers consumed',
        release_fingerprint=release['fingerprint'] if release else None)
    dump(work / 'reports/experiment_config.json', experiment)
    ready = not blockers
    state = dict(ready=ready, blockers=blockers, verified_positive_n=sum(d.get('status') == 'positive' for d in reviews.values()),
        explicitly_rejected_n=sum(d.get('status') == 'excluded' for d in reviews.values()), selected_n=len(selected),
        queued_review_n=len(queue), metadata_missing_or_overlap_n=len(metadata_errors), environment_ready=environment['ready'])
    dump(work / 'reports/readiness.json', state)
    lines = ['# Controlled training pilot readiness', '', '**READY**' if ready else '**BLOCKED — training not started by preparation**', '',
        f'Internal inventory: {len(inventory)} videos. Verified positives: {state["verified_positive_n"]}. '
        f'Explicit rejections: {state["explicitly_rejected_n"]}. Selected: {len(selected)}/640. Review queue: {len(queue)}.',
        'Unreviewed samples are unresolved, not human rejections. No oversampling, split changes, external-test use, or auto-positive labels.', '',
        '| Class | Split | Target | Positive available | Selected | Rejected | Unreviewed |', '|---|---|---:|---:|---:|---:|---:|']
    lines.extend(f'| {r.class_name} | {r.split} | {r.target} | {r.available_positive} | {r.selected_n} | {r.reviewed_rejected_n} | {r.unreviewed_n} |' for r in counts.itertuples())
    lines += ['', '## Blockers', *['- '+x for x in blockers], '',
        'Signer and globally unique recording-session IDs must be supplied by the researcher; video IDs are not signer identities.',
        'Frozen splits cannot be repaired by moving videos. If metadata exposes overlap, select an independent subset or report infeasibility.',
        'Review uses the same technical proposal policy across classes; sharpness is not semantic validity.',
        'All augmentation parameters that could alter gesture meaning are disabled; only resize/letterbox preprocessing remains.',
        'API is local Ultralytics with local pinned COCO weights. No paid API, HUB, cloud training, or CPU fallback.']
    (work / 'reports/pilot_readiness_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if not (work / 'runs/trained_models.json').exists():
        (work / 'pilot_conclusion.md').write_text('# Controlled training pilot — NOT TRAINED\n\n'
            f'Verified positive videos={state["verified_positive_n"]}; explicit rejections={state["explicitly_rejected_n"]}; '
            f'unreviewed={len(inventory)-state["verified_positive_n"]-state["explicitly_rejected_n"]}.\n\n'
            'Models A/B have not completed training. Shared-six-class comparison and apa_kabar/halo model diagnostics are unavailable. '
            'Dataset final-readiness and temporal causes are not established. Do not infer a model advantage or a need for LSTM/TCN.\n\n'
            'See reports/pilot_readiness_report.md for gates. No training metrics are fabricated.\n', encoding='utf-8')
        dump(work / 'evaluation/status.json', dict(status='NOT_TRAINED', metrics_available=False))
    LOG.info('Readiness=%s; selected=%d/640; blockers=%d', ready, len(selected), len(blockers))
    return state


def import_review(root, source, work, path):
    root, source, work = Path(root).resolve(), Path(source).resolve(), Path(work).resolve()
    if (work / 'datasets/release.json').exists():
        raise ValueError('Dataset already frozen; review edits require a new work directory')
    if source_snapshot(source) != read_json(work / 'reports/source_snapshot.json'):
        raise ValueError('Source snapshot changed')
    payload = read_json(path)
    if payload.get('schema') != 1 or payload.get('kind') != 'training_pilot_review':
        raise ValueError('Expected controlled training pilot review, not human blind review or v3 export')
    if payload.get('annotations_sha256') != b.sha(work / 'review_queue.csv') or payload.get('candidates_sha256') != b.sha(source / 'candidates.csv'):
        raise ValueError('Stale review queue/catalog')
    reviewer = payload.get('reviewer')
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError('Named human reviewer required')
    queue = csv(work / 'review_queue.csv').set_index('sample_id')
    records = payload.get('records', [])
    ids = [r.get('sample_id') for r in records]
    if len(ids) != len(set(ids)) or set(ids) != set(queue.index):
        raise ValueError('Every queued video must occur exactly once')
    catalog = f.catalog(source)
    full_catalog = csv(source / 'candidates.csv')
    reviews = read_json(work / 'reviews.json', {})
    if (work / 'reviews.json').exists() and read_json(work / 'reports/review_state.json', {}).get('reviews_sha256') != b.sha(work / 'reviews.json'):
        raise ValueError('Review store changed outside validated import')
    changes = {}
    for d in records:
        sid = d['sample_id']
        row = queue.loc[sid]
        if d.get('class_id') != int(row.class_id) or d.get('request_more'):
            raise ValueError('Class changes and source re-extraction are forbidden')
        if not isinstance(d.get('note', ''), str):
            raise ValueError('Review note must be text')
        rejected = d.get('rejected_candidates', {})
        if not isinstance(rejected, dict) or any(cid not in catalog or catalog[cid]['sample_id'] != sid or not isinstance(reason, str) or not reason.strip() for cid, reason in rejected.items()):
            raise ValueError('Invalid per-candidate rejection')
        common = dict(d, reviewer=reviewer.strip())
        if d.get('candidate_id'):
            c = catalog.get(d['candidate_id'])
            if not c or c['sample_id'] != sid:
                raise ValueError('Candidate does not belong to video')
            validate_positive(d, c)
            stamp = datetime.fromisoformat(d.get('timestamp', '').replace('Z', '+00:00'))
            if stamp.utcoffset() is None:
                raise ValueError('Timezone timestamp required')
            checked, _ = human_pilot.inspect_candidates(root, source, row.to_dict() | {'sample_id': sid},
                full_catalog[full_catalog.sample_id.eq(sid)], read_json(source / 'candidate_config.json'))
            checked_candidate = next(x for x in checked if x['candidate_id'] == c['candidate_id'])
            if not checked_candidate['readable']:
                raise ValueError('Candidate image unreadable or corrupt')
            validate_positive(d, checked_candidate)
            changes[sid] = dict(common, status='positive', frame_sha256=c['frame_sha256'])
        elif d.get('semantic_status') == 'excluded' and d.get('reason') in REJECT_REASONS:
            if d['reason'] == 'other' and not d.get('note', '').strip():
                raise ValueError('Other rejection needs a note')
            changes[sid] = dict(common, status='excluded')
        elif d.get('semantic_status') == 'excluded':
            raise ValueError('Explicit whole-video rejection requires a supported reason')
        elif rejected:
            changes[sid] = dict(common, status='pending')
        # Untouched rows stay pending; never silently become rejects or positives.
    history = work / 'review/imports'
    history.mkdir(exist_ok=True)
    archive = history / f'{b.sha(path)}.json'
    if archive.exists():
        raise ValueError('This review file was already imported')
    shutil.copy2(path, archive)
    reviews.update(changes)
    dump(work / 'reviews.json', reviews)
    dump(work / 'reports/review_state.json', dict(reviews_sha256=b.sha(work / 'reviews.json'), last_import_sha256=b.sha(path)))
    LOG.info('Imported %d explicit video decisions; source and human pilot untouched', len(changes))


def approve_audit(work, reviewer, confirmed):
    if not reviewer or not reviewer.strip() or not confirmed:
        raise ValueError('Named human reviewer and explicit --confirmed required after viewing every batch sheet')
    work = Path(work)
    release = validate_release(work)
    dump(work / 'reports/visual_audit.json', dict(reviewer=reviewer.strip(), confirmed=True,
        fingerprint=release['fingerprint'], contact_sheets=release['contact_sheets'], timestamp=datetime.now(timezone.utc).isoformat()))


def train(root, source, work):
    state = prepare(root, source, work)
    if not state['ready']:
        raise RuntimeError('Training blocked:\n'+'\n'.join(state['blockers']))
    work, source = Path(work).resolve(), Path(source).resolve()
    release = validate_release(work)
    for name in ['A', 'B']:
        if (work / 'runs' / f'model_{name}').exists():
            raise ValueError('Run exists; refusing overwrite or sequential/resumed fine-tuning')
    YOLO = local_yolo(work)
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable; CPU training forbidden')
    checkpoint = checkpoint_info(source)
    b.record_environment(work)
    trained = []
    for name in ['A', 'B']:
        if b.sha(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('Initial COCO checkpoint changed')
        class ControlledTrainer(DetectionTrainer):
            actual_steps = 0

            def _setup_train(self):
                super()._setup_train()
                if self.batch_size != 16 or type(self.optimizer).__name__ != 'SGD' or not self.model.end2end:
                    raise RuntimeError('Actual batch/optimizer/inference mode drift; OOM batch reduction forbidden')

            def optimizer_step(self):
                if self.batch_size != 16:
                    raise RuntimeError('Automatic batch-size change forbidden')
                super().optimizer_step()
                self.actual_steps += 1

        model = YOLO(checkpoint['path'])
        set_inference_mode(model)
        if len(model.names) != 80 or not model.model.end2end:
            raise ValueError('Each run must start from identical COCO end-to-end weights')
        def epoch_log(trainer):
            dump(Path(trainer.save_dir) / 'optimizer_steps.json', dict(actual_optimizer_steps=trainer.actual_steps,
                epoch=int(trainer.epoch)+1, batches_per_epoch=len(trainer.train_loader), batch=trainer.batch_size,
                optimizer=type(trainer.optimizer).__name__, nbs=trainer.args.nbs))
        model.add_callback('on_fit_epoch_end', epoch_log)
        try:
            model.train(trainer=ControlledTrainer, data=str(work / 'datasets' / name / 'pilot.yaml'),
                        project=str(work / 'runs'), name=f'model_{name}', **PILOT_CONFIG)
        finally:
            trainer = getattr(model, 'trainer', None)
            if trainer and hasattr(trainer, 'actual_steps'):
                dump(work / 'runs' / f'model_{name}_execution.json', dict(
                    actual_optimizer_steps=trainer.actual_steps, initial_checkpoint_sha256=checkpoint['sha256'],
                    actual_args=vars(trainer.args), release_fingerprint=release['fingerprint']))
        best = Path(model.trainer.save_dir) / 'weights/best.pt'
        if not best.is_file() or int(model.trainer.epoch)+1 != 30:
            raise RuntimeError('Training did not complete all 30 pilot epochs')
        trained.append(dict(model=name, path=str(best), sha256=b.sha(best),
            initial_checkpoint_sha256=checkpoint['sha256'], actual_optimizer_steps=model.trainer.actual_steps))
        del model
        torch.cuda.empty_cache()
    dump(work / 'runs/trained_models.json', dict(models=trained, release_fingerprint=release['fingerprint']))
    from training_pilot_eval import evaluate
    evaluate(work)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'import-review', 'approve-audit', 'train', 'evaluate'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--work', type=Path)
    parser.add_argument('--file', type=Path)
    parser.add_argument('--reviewer')
    parser.add_argument('--confirmed', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    source = args.source or root / 'experiment_yolo26_v3'
    work = args.work or root / 'experiment_yolo26_training_pilot'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.command == 'prepare':
        prepare(root, source, work)
    elif args.command == 'import-review':
        if not args.file:
            parser.error('--file required')
        import_review(root, source, work, args.file)
        prepare(root, source, work)
    elif args.command == 'approve-audit':
        approve_audit(work, args.reviewer, args.confirmed)
    elif args.command == 'train':
        train(root, source, work)
    else:
        from training_pilot_eval import evaluate
        evaluate(work)


if __name__ == '__main__':
    main()
