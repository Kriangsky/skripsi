"""Preparation and QC for independent COCO -> 6 / COCO -> 8 experiments.

No training is invoked by this module or its command-line entry point.
"""
from pathlib import Path
from collections import Counter
import hashlib
import html
import importlib.metadata as package_metadata
from functools import lru_cache
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.request

import cv2
import numpy as np
import pandas as pd
import yaml

os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).resolve().parent / 'experiment_yolo26_v2' / 'ultralytics_config'))

CLASSES = ['saya', 'kamu', 'senang', 'bingung', 'marah', 'apa_kabar', 'halo', 'tidur']
BASE = CLASSES[:6]
HF_REVISION = '070ac3c51435984d992ba92eb3f234daf4e4500d'
HAND_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
HAND_SHA256 = 'fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1'
SEED = 42
PADDING = 0.15  # per side, relative to union box width/height
EXT = {'.mov', '.mp4', '.avi', '.mkv'}
FRAME_PRIMARY=[.20,.30,.40,.50,.60,.70,.80]
FRAME_FALLBACK=[.05,.10,.15,.85,.90,.95]
FRAME_QUALITY=dict(border_margin=.01,min_hand_side_px=16,min_hand_area_px=256,min_focus=60.)

@lru_cache(maxsize=1)
def annotation_recipe():
    return hashlib.sha256(json.dumps(dict(model=HAND_SHA256,padding=PADDING,num_hands=2,
        detection=.5,presence=.5,mode='IMAGE',frame='human_selected_middle_candidates',format='saved_jpeg_union_v3',
        primary=FRAME_PRIMARY,fallback=FRAME_FALLBACK,
        quality=dict(border_margin=FRAME_QUALITY['border_margin'],min_side_px=FRAME_QUALITY['min_hand_side_px'],
                     min_area_px=FRAME_QUALITY['min_hand_area_px'],min_focus=FRAME_QUALITY['min_focus']),
        runtime=dict(mediapipe=package_metadata.version('mediapipe'),protobuf=package_metadata.version('protobuf'),
                     numpy=np.__version__,opencv=cv2.__version__)),sort_keys=True).encode()).hexdigest()

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')

def sid(video):
    return hashlib.sha256(video.encode()).hexdigest()[:20]

def safe_path(root, relative):
    p = (Path(root) / relative).resolve()
    if not p.is_relative_to(Path(root).resolve()) or not p.is_file():
        raise ValueError(f'Invalid file path: {relative}')
    return p

def split_rows(df, metadata=None, seed=SEED):
    df = df.copy()
    df['split'] = np.where(df.source.eq('test'), 'test', '')
    df['signer'] = ''
    internal = df.source.eq('train')
    mode = 'video_stratified_80_20_no_signer_metadata'
    if metadata is not None:
        if metadata.video.duplicated().any():
            raise ValueError('Duplicate video in signer metadata')
        mapping = metadata.set_index('video').signer.to_dict()
        df.loc[internal, 'signer'] = df.loc[internal, 'video'].map(mapping).fillna('')
        if df.loc[internal, 'signer'].astype(str).str.strip().eq('').any():
            raise ValueError('Signer metadata incomplete; supply all internal videos or remove optional metadata file')
        groups = sorted(df.loc[internal, 'signer'].unique())
        if len(groups) < 2:
            raise ValueError('At least two signers needed for disjoint split')
        rng = np.random.default_rng(seed)
        best = None
        # Select signer groups globally, not independently per class.
        for _ in range(500):
            shuffled = rng.permutation(groups)
            for n in range(1, len(groups)):
                val = internal & df.signer.isin(shuffled[:n])
                ratios = df.loc[val].class_name.value_counts().reindex(CLASSES, fill_value=0) / df.loc[internal].class_name.value_counts().reindex(CLASSES)
                if ((ratios > 0) & (ratios < 1)).all():
                    score = float(np.square(ratios - .2).sum())
                    if best is None or score < best[0]:
                        best = (score, set(shuffled[:n]))
        if best is None:
            raise ValueError('Cannot obtain signer-disjoint split containing every class in train and val')
        df.loc[internal, 'split'] = 'train'
        df.loc[internal & df.signer.isin(best[1]), 'split'] = 'val'
        mode = 'signer_disjoint_approximate_80_20'
    else:
        for cls in CLASSES:
            indices = df[internal & df.class_name.eq(cls)].sort_values('video').sample(frac=1, random_state=seed).index
            df.loc[indices, 'split'] = 'train'
            df.loc[indices[:round(len(indices)*.2)], 'split'] = 'val'
    return df, mode

def inventory(root, work):
    root, work = Path(root).resolve(), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(root / 'test_ground_truth.csv', dtype=str, keep_default_na=False)
    if truth.filename.duplicated().any() or not set(truth.ground_truth).issubset(CLASSES):
        raise ValueError('Test CSV has duplicate filenames or unknown classes')
    lookup = truth.set_index('filename').ground_truth.to_dict()
    rows = []
    for source in ['train', 'test']:
        for p in sorted((root / source).rglob('*')):
            if p.suffix.lower() not in EXT:
                continue
            cls = p.parent.name if source == 'train' else lookup.get(p.stem)
            if cls not in CLASSES:
                raise ValueError(f'Unknown class for {p}')
            relative = p.relative_to(root).as_posix()
            rows.append(dict(video=relative, sample_id=sid(relative), source=source, class_name=cls,
                             class_id=CLASSES.index(cls), video_sha256=sha(p)))
    df = pd.DataFrame(rows)
    counts = df.groupby(['class_name', 'source']).size().unstack(fill_value=0).reindex(CLASSES)
    counts.to_csv(work / 'inventory_counts.csv')
    if counts.isna().any().any() or not (counts[['train','test']].to_numpy() == [300,50]).all():
        raise ValueError(f'Expected 300 internal + 50 external per class:\n{counts}')
    actual_test = {Path(v).stem for v in df.loc[df.source.eq('test'), 'video']}
    if actual_test != set(truth.filename):
        raise ValueError('Test CSV and test files do not match exactly')
    metadata_path = root / 'signer_metadata.csv'
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False) if metadata_path.exists() else None
    lock_path=work/'split_lock.csv'
    if lock_path.exists():
        locked=pd.read_csv(lock_path,keep_default_na=False).set_index('sample_id')
        if locked.index.duplicated().any() or set(locked.index)!=set(df.sample_id):
            raise ValueError('Source inventory changed; frozen split must be reviewed explicitly')
        for r in df.itertuples():
            original=locked.loc[r.sample_id]
            for field in ['video','class_name','class_id','source','video_sha256']:
                if str(getattr(r,field))!=str(original[field]):raise ValueError(f'Frozen inventory mismatch: {r.video}, {field}')
        df['split']=df.sample_id.map(locked['split'])
        df['signer']=df.sample_id.map(locked['signer']).fillna('')
        if not df.split.isin(['train','val','test']).all() or not (df.source.eq('test')==df.split.eq('test')).all():
            raise ValueError('Invalid frozen split')
        mode='frozen_existing_split'
    else:
        df, mode = split_rows(df, metadata)
    save_json(work / 'preparation_config.json', dict(classes=CLASSES, split_mode=mode, seed=SEED,
        frame_rule='reviewed middle candidates' if lock_path.exists() else 'legacy midpoint', padding=PADDING, num_hands=2,
        min_hand_detection_confidence=.5, min_hand_presence_confidence=.5,
        hand_model_sha256=HAND_SHA256,
        test_csv_sha256=sha(root / 'test_ground_truth.csv'),
        signer_metadata_sha256=sha(metadata_path) if metadata_path.exists() else None))
    df.to_csv(work / 'inventory.csv', index=False)
    return df

def midpoint(video):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f'Cannot open {video}')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count <= 0:
            while cap.read()[0]:
                count += 1
            cap.release()
            cap = cv2.VideoCapture(str(video))
        if count <= 0:
            raise ValueError(f'No readable frame: {video}')
        index = count // 2
        for _ in range(index+1):
            ok, image = cap.read()
            if not ok or image is None:
                raise ValueError(f'Cannot decode midpoint: {video}')
        return index, image
    finally:
        cap.release()

def union_box(landmarks, padding=PADDING):
    points = np.asarray([[lm.x, lm.y] for hand in landmarks for lm in hand], dtype=float)
    if points.size == 0:
        return None
    if not np.isfinite(points).all():
        raise ValueError('Non-finite landmarks')
    lo, hi = points.min(axis=0), points.max(axis=0)
    pad = (hi-lo)*padding
    lo, hi = np.clip(lo-pad, 0, 1), np.clip(hi+pad, 0, 1)
    if not (hi > lo).all():
        raise ValueError('Degenerate box')
    return [float(x) for x in [lo[0],lo[1],hi[0],hi[1]]]

def yolo_box(box):
    x1,y1,x2,y2 = map(float, box)
    if not np.isfinite(box).all() or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f'Invalid normalized box: {box}')
    return [(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1]

def review_digest(row):
    return hashlib.sha256(json.dumps(dict(frame=row['frame_sha256'],class_id=int(row['class_id']),
        box=[float(row[k]) for k in ['x1','y1','x2','y2']]),sort_keys=True).encode()).hexdigest()

def get_hand_asset(root, work):
    target = Path(work) / 'assets' / 'hand_landmarker.task'
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        previous = Path(root) / 'experiment_yolo26/assets/hand_landmarker.task'
        if previous.exists() and sha(previous) == HAND_SHA256:
            shutil.copy2(previous, target)
        else:
            tmp = target.with_suffix('.download')
            urllib.request.urlretrieve(HAND_URL, tmp)
            tmp.replace(target)
    if sha(target) != HAND_SHA256:
        raise ValueError('MediaPipe model checksum mismatch')
    return target

def prepare(root, work):
    """Propose several frames; explicit human selection is required for positives."""
    from frame_candidates import prepare as prepare_candidates
    return prepare_candidates(root,work)

def render_report(work):
    work = Path(work).resolve()
    if (work/'candidate_config.json').exists():
        from frame_candidates import render_report as render_candidates
        return render_candidates(work)
    df = pd.read_csv(work/'annotations.csv', keep_default_na=False)
    cards = []
    for r in df.to_dict('records'):
        image_path = work/'frames'/f"{r['sample_id']}.jpg"
        image = cv2.imread(str(image_path)) if image_path.exists() else None
        label = work/'candidate_labels'/f"{r['sample_id']}.txt"
        valid = False
        if image is not None:
            try:
                box = [float(r[k]) for k in ['x1','y1','x2','y2']]
                norm = yolo_box(box)
                valid = True
                h,w = image.shape[:2]
                cv2.rectangle(image, (round(box[0]*w),round(box[1]*h)), (min(w-1,round(box[2]*w)),min(h-1,round(box[3]*h))), (0,255,0), 3)
                label.write_text(str(r['class_id'])+' '+' '.join(f'{v:.8f}' for v in norm)+'\n')
            except (ValueError, TypeError):
                pass
            cv2.putText(image, f"{r['class_name']} {r['auto_status']} / {r['review_status']}", (10,30), cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255),2)
            if not cv2.imwrite(str(work/'overlays'/f"{r['sample_id']}.jpg"), image):
                raise RuntimeError('Cannot save overlay')
        if not valid and label.exists():
            label.unlink()
        e = lambda v: html.escape(str(v), quote=True)
        cards.append(f'<article data-class="{e(r["class_name"])}" data-status="{e(r["auto_status"])}"><a href="overlays/{e(r["sample_id"])}.jpg"><img loading="lazy" src="overlays/{e(r["sample_id"])}.jpg" alt="{e(r["video"])}"></a><p>{e(r["sample_id"])}<br>{e(r["video"])}<br>{e(r["split"])} · hands={r["num_hands"]} · {e(r["auto_status"])} · review={e(r["review_status"])}</p></article>')
    document = '''<!doctype html><meta charset="utf-8"><title>BISINDO annotation QC</title>
<style>body{font:16px system-ui;margin:24px;background:#eee}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}article{background:white;padding:12px;overflow-wrap:anywhere}img{width:100%}button,select{font:inherit;padding:8px}</style>
<h1>BISINDO — all annotation candidates</h1><p>Green box = candidate, not verified ground truth. Click image for full resolution. Inspect both hands, gesture, padding, and class. Correct/review using notebook review_sample(). Refresh after saving.</p>
<label>Class <select id="cls"><option value="">All</option>''' + ''.join(f'<option>{c}</option>' for c in CLASSES) + '''</select></label>
<label>Status <select id="status"><option value="">All</option><option>detected</option><option>no_hands</option><option>error</option></select></label><p id="total"></p><main>''' + ''.join(cards) + '''</main><script>
function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{const yes=(!cls.value||a.dataset.class===cls.value)&&(!status.value||a.dataset.status===status.value);a.hidden=!yes;if(yes)n++});document.getElementById('total').textContent=n+' samples'}
const cls=document.getElementById('cls'),status=document.getElementById('status');cls.onchange=status.onchange=filter;filter();</script>'''
    (work/'annotation_gallery.html').write_text(document, encoding='utf-8')
    df.groupby(['source','class_name','split','auto_status','review_status']).size().rename('samples').to_csv(work/'annotation_counts.csv')
    df[df.auto_status.ne('detected')].to_csv(work/'failed_annotations.csv', index=False)
    save_json(work/'annotation_summary.json', dict(total=len(df), detection_status=dict(Counter(df.auto_status)),
        review_status=dict(Counter(df.review_status)), training_allowed=False,
        note='Training requires validate_dataset() with reviewed, hash-bound boxes; automatic detections are candidates.'))
    render_review_gallery(work)

def render_review_gallery(work):
    """Offline quick scan/editor, with explicit human bulk approval export."""
    work=Path(work).resolve()
    df=pd.read_csv(work/'annotations.csv',keep_default_na=False)
    records=[]
    for r in df.to_dict('records'):
        try:
            box=[float(r[k]) for k in ['x1','y1','x2','y2']]
            yolo_box(box)
        except (ValueError,TypeError):
            box=None
        records.append(dict(sample_id=r['sample_id'],video=r['video'],class_name=r['class_name'],
            class_id=int(r['class_id']),source=r['source'],split=r['split'],num_hands=int(r['num_hands']),
            auto_status=r['auto_status'],frame_sha256=r['frame_sha256'],box=box))
    payload=json.dumps(dict(annotations_sha256=sha(work/'annotations.csv'),records=records),allow_nan=False)
    # Prevent embedded text from terminating the JSON script element.
    payload=payload.replace('<','\\u003c')
    template=(Path(__file__).parent/'annotation_review_template.html').read_text(encoding='utf-8')
    target=work/'annotation_review.html'
    target.write_text(template.replace('__DATASET_JSON__',payload),encoding='utf-8')
    return target

def import_bulk_review(work, review_json):
    """Apply one explicit human approval export, atomically, with rollback CSV."""
    work=Path(work).resolve()
    path=work/'annotations.csv'
    payload=json.loads(Path(review_json).read_text(encoding='utf-8'))
    fingerprint=sha(path)
    if payload.get('schema')!=1 or payload.get('kind')!='approval' or payload.get('all_images_reviewed') is not True:
        raise ValueError('Approval export required; a draft does not approve annotations')
    if payload.get('annotations_sha256')!=fingerprint:
        raise ValueError('Stale review: annotations changed after HTML was generated; regenerate and review current snapshot')
    reviewer=payload.get('reviewer')
    if not isinstance(reviewer,str) or not reviewer.strip():
        raise ValueError('Reviewer name required')
    df=pd.read_csv(path,keep_default_na=False)
    rows=payload.get('records',[])
    if not isinstance(rows,list):
        raise ValueError('Invalid records')
    ids=[r.get('sample_id') for r in rows]
    if len(ids)!=len(set(ids)) or set(ids)!=set(df.sample_id):
        raise ValueError('Review must contain every sample exactly once')
    lookup={r['sample_id']:r for r in rows}
    for index,row in df.iterrows():
        edited=lookup[row.sample_id]
        if edited.get('class_id')!=int(row.class_id) or edited.get('frame_sha256')!=row.frame_sha256:
            raise ValueError(f'{row.sample_id}: class/frame mismatch')
        if sha(work/'frames'/f'{row.sample_id}.jpg')!=row.frame_sha256:
            raise ValueError(f'{row.sample_id}: frame file changed')
        box=edited.get('box')
        if not isinstance(box,list) or len(box)!=4 or any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in box):
            raise ValueError(f'{row.sample_id}: invalid box')
        yolo_box(box)
        note=edited.get('note','')
        if not isinstance(note,str):
            raise ValueError('Review note must be text')
        original=pd.to_numeric(row[['x1','y1','x2','y2']],errors='coerce').to_numpy(dtype=float)
        corrected=not np.allclose(box,original,rtol=0,atol=1e-12,equal_nan=False)
        df.loc[index,['x1','y1','x2','y2']]=box
        df.loc[index,['review_status','reviewer','review_note','review_frame_sha256']]=[
            'corrected' if corrected or row.review_status=='corrected' else 'approved',reviewer.strip(),note,row.frame_sha256]
        df.loc[index,'review_box_sha256']=review_digest(df.loc[index])
    backup=work/f'annotations_before_bulk_{fingerprint[:16]}.csv'
    if not backup.exists():
        shutil.copy2(path,backup)
    temporary=work/'annotations.bulk.tmp'
    df.to_csv(temporary,index=False)
    temporary.replace(path)
    save_json(work/f'bulk_review_{fingerprint[:16]}.json',payload)
    save_json(work/'qc_approval.json',dict(approved=False,reason='Bulk review imported; run validation gates'))
    render_report(work)
    return df

def review_sample(work, sample_id, reviewer, box=None, note='', reject=False):
    """box: normalized x1,y1,x2,y2, or None to approve existing candidate.

    Human must actually inspect the frame before calling this function.
    Rejection never silently excludes a sample from the experiment.
    """
    work = Path(work)
    if not reviewer.strip():
        raise ValueError('Reviewer name required')
    df = pd.read_csv(work/'annotations.csv', keep_default_na=False)
    mask = df.sample_id.eq(sample_id)
    if mask.sum() != 1:
        raise ValueError('Unknown sample_id')
    row = df.loc[mask].iloc[0]
    if reject:
        if not note.strip():
            raise ValueError('Reason required for rejection')
        df.loc[mask,'review_status'] = 'rejected'
    else:
        bounds = box if box is not None else [float(row[k]) for k in ['x1','y1','x2','y2']]
        yolo_box(bounds)
        df.loc[mask,['x1','y1','x2','y2']] = bounds
        df.loc[mask,'review_status'] = 'corrected' if box is not None else 'approved'
    frame = work/'frames'/f'{sample_id}.jpg'
    if sha(frame) != row.frame_sha256:
        raise ValueError('Frame changed; re-run prepare before reviewing')
    df.loc[mask,['reviewer','review_note','review_frame_sha256']] = [reviewer,note,row.frame_sha256]
    df.loc[mask,'review_box_sha256'] = review_digest(df.loc[mask].iloc[0]) if not reject else ''
    df.to_csv(work/'annotations.csv', index=False)
    save_json(work/'qc_approval.json', dict(approved=False, reason='Annotation changed; run QC again'))

def validation_issues(root, work, df):
    root,work = Path(root).resolve(),Path(work).resolve()
    errors = []
    if (work/'candidate_config.json').exists():
        from frame_candidates import selection_issues
        errors.extend(selection_issues(work,df))
    expected = inventory(root, work)
    if set(df.sample_id) != set(expected.sample_id) or df.sample_id.duplicated().any():
        errors.append('Manifest sample IDs do not match source inventory')
    wanted = expected.set_index('sample_id')
    content_splits, frame_splits = {}, {}
    for r in df.to_dict('records'):
        key = r['sample_id']
        if key not in wanted.index:
            continue
        original = wanted.loc[key]
        if r.get('annotation_recipe_sha256') != annotation_recipe():
            errors.append(f'{key}: outdated annotation recipe')
        for field in ['video','source','class_name','class_id','split','signer','video_sha256']:
            if str(r[field]) != str(original[field]):
                errors.append(f'{key}: stale/modified {field}')
        if r['class_name'] not in CLASSES or r['class_id'] != CLASSES.index(r['class_name']):
            errors.append(f'{key}: class mapping mismatch')
        if r['review_status'] not in ['approved','corrected'] or not str(r['reviewer']).strip():
            errors.append(f'{key}: human QC pending/rejected')
        image_path = work/'frames'/f'{key}.jpg'
        if not image_path.exists():
            errors.append(f'{key}: missing frame')
            continue
        digest = sha(image_path)
        if digest != r['frame_sha256'] or digest != r['review_frame_sha256']:
            errors.append(f'{key}: frame/review hash mismatch')
        image = cv2.imread(str(image_path))
        if image is None or image.shape[:2] != (r['height'],r['width']):
            errors.append(f'{key}: frame dimensions invalid')
        try:
            yolo_box([float(r[k]) for k in ['x1','y1','x2','y2']])
            if r.get('review_box_sha256') != review_digest(r):
                errors.append(f'{key}: box/class changed since review')
        except (ValueError,TypeError):
            errors.append(f'{key}: invalid box')
        for mapping, digest_key in [(content_splits,r['video_sha256']),(frame_splits,digest)]:
            if digest_key in mapping and mapping[digest_key] != r['split']:
                errors.append(f'{key}: exact duplicate content crosses splits')
            mapping[digest_key] = r['split']
    internal = df[df.source.eq('train')]
    if internal.signer.astype(str).str.strip().ne('').any():
        if set(internal[internal.split.eq('train')].signer) & set(internal[internal.split.eq('val')].signer):
            errors.append('Signer overlap between train and validation')
    return errors

def validate_dataset(root, work):
    work = Path(work)
    df = pd.read_csv(work/'annotations.csv', keep_default_na=False)
    errors = validation_issues(root,work,df)
    report = dict(passed=not errors, error_count=len(errors), errors=errors,
                  annotations_sha256=sha(work/'annotations.csv'), counts=df.groupby(['class_name','split']).size().to_dict().__str__())
    save_json(work/'validation_report.json',report)
    save_json(work/'qc_approval.json', dict(approved=not errors, annotations_sha256=report['annotations_sha256']))
    if errors:
        raise RuntimeError(f'QC not passed: {len(errors)} findings. See {work / "validation_report.json"}. Training blocked.')
    return df

def build_datasets(root, work):
    work = Path(work).resolve()
    df = validate_dataset(root,work)
    fingerprint = sha(work/'annotations.csv')
    data = work/'datasets'/fingerprint[:16]
    data.mkdir(parents=True,exist_ok=True)
    for name, classes in [('A',BASE),('B',CLASSES)]:
        folder = data/name
        for r in df[df.class_name.isin(classes)].to_dict('records'):
            for kind in ['images','labels']:
                (folder/kind/r['split']).mkdir(parents=True,exist_ok=True)
            shutil.copy2(work/'frames'/f"{r['sample_id']}.jpg", folder/'images'/r['split']/f"{r['sample_id']}.jpg")
            vals = yolo_box([float(r[k]) for k in ['x1','y1','x2','y2']])
            (folder/'labels'/r['split']/f"{r['sample_id']}.txt").write_text(f"{r['class_id']} "+' '.join(f'{x:.8f}' for x in vals)+'\n')
        common = folder/'test_common.txt'
        common.write_text(''.join(str(folder/'images/test'/f'{r.sample_id}.jpg')+'\n' for r in df[df.split.eq('test') & df.class_name.isin(BASE)].itertuples()))
        cfg = dict(path=str(folder),train='images/train',val='images/val',test=str(common),names=classes)
        (folder/'common.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        if name=='B':
            cfg['test']='images/test'
            (folder/'full.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    # Check shared images and labels are byte-identical in the released datasets.
    for r in df[df.class_name.isin(BASE)].itertuples():
        for kind,ext in [('images','jpg'),('labels','txt')]:
            relative = Path(kind)/r.split/f'{r.sample_id}.{ext}'
            if sha(data/'A'/relative) != sha(data/'B'/relative):
                raise RuntimeError('Shared A/B sample differs')
    df.to_csv(data/'manifest.csv',index=False)
    save_json(data/'release.json',dict(annotations_sha256=fingerprint, class_mapping=dict(enumerate(CLASSES))))
    return data

def official_checkpoint(work):
    from huggingface_hub import hf_hub_download
    work = Path(work).resolve()
    lock_path = work/'checkpoint.json'
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        path = Path(lock['path'])
        if path.exists() and sha(path)==lock['sha256'] and lock['revision']==HF_REVISION:
            return path
    path = Path(hf_hub_download(repo_id='Ultralytics/YOLO26', filename='yolo26n.pt',
                revision=HF_REVISION, cache_dir=str(work/'hf_cache')))
    save_json(lock_path,dict(repo_id='Ultralytics/YOLO26',filename='yolo26n.pt',revision=HF_REVISION,path=str(path),sha256=sha(path)))
    return path

def record_environment(work):
    import importlib.metadata as meta
    import torch
    work = Path(work)
    packages = ['ultralytics','ultralytics-thop','ultralytics-platform','mediapipe','protobuf','numpy','pandas','opencv-python','opencv-contrib-python','opencv-python-headless','huggingface-hub','torch','torchvision','PyYAML','Pillow','ipywidgets','ipykernel','nbformat','matplotlib']
    versions = {}
    packages += ['pyzmq', 'jupyter-client', 'tornado']
    for p in packages:
        try:
            versions[p] = meta.version(p)
        except meta.PackageNotFoundError:
            pass
    record = dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages=versions,
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    save_json(work/'environment.json',record)
    (work/'requirements-resolved.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    return record

def train_independent(root, work, seeds=(42,)):
    """Explicit call only, with fresh validation and a content-addressed release."""
    import torch
    from ultralytics import YOLO
    work = Path(work).resolve()
    import importlib.metadata as meta
    environment_path = work/'environment.json'
    if not environment_path.exists():
        raise RuntimeError('Record preparation environment before training')
    locked = json.loads(environment_path.read_text())
    for package,version in locked['packages'].items():
        if meta.version(package) != version:
            raise RuntimeError(f'Dependency drift: {package}; use recorded environment')
    data = build_datasets(root,work)
    checkpoint = official_checkpoint(work)
    environment = record_environment(work)
    args = dict(imgsz=640,batch=16,epochs=100,optimizer='MuSGD',lr0=.01,lrf=.01,momentum=.937,
        weight_decay=.0005,warmup_epochs=3.,cos_lr=True,patience=0,deterministic=True,
        device=0 if torch.cuda.is_available() else 'cpu',workers=0,amp=False,plots=True,
        degrees=15.,translate=.1,scale=.2,shear=0.,perspective=0.,fliplr=0.,flipud=0.,
        hsv_h=.015,hsv_s=.4,hsv_v=.4,mosaic=0.,mixup=0.,cutmix=0.,close_mosaic=0)
    save_json(work/'training_plan.json',dict(args=args,seeds=list(seeds),dataset=str(data),checkpoint_sha256=sha(checkpoint),environment=environment))
    best = {}
    for seed in seeds:
        for name in ['A','B']:
            run_name=f'{data.name}_{name}_seed{seed}'
            if (work/'runs'/run_name).exists():
                raise RuntimeError(f'Run exists; refusing to overwrite: {run_name}')
            model=YOLO(str(checkpoint))
            if model.task!='detect' or len(model.names)!=80 or model.names[0]!='person':
                raise ValueError('Expected COCO detection checkpoint')
            model.train(data=str(data/name/'common.yaml'),seed=seed,project=str(work/'runs'),name=run_name,**args)
            best[name,seed]=Path(model.trainer.save_dir)/'weights/best.pt'
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    save_json(work/'trained_models.json',dict(dataset=str(data),models=[dict(model=n,seed=s,path=str(p)) for (n,s),p in best.items()]))
    return data,best

if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    work = root/'experiment_yolo26_v3'
    prepare(root,work)
    record_environment(work)
    print((work/'annotation_summary.json').read_text(),flush=True)
