"""Audit local BISINDO inputs without modifying sources or annotations."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import sys
import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiment_yolo26'
OUT.mkdir(exist_ok=True)
EXT = {'.mp4', '.mov', '.avi', '.mkv'}
truth = pd.read_csv(ROOT / 'test_ground_truth.csv', dtype=str)
lookup = dict(zip(truth.filename, truth.ground_truth))
videos = sorted(p for source in ('train', 'test') for p in (ROOT / source).rglob('*') if p.suffix.lower() in EXT)

def inspect(p):
    source = p.relative_to(ROOT).parts[0]
    row = dict(video=p.relative_to(ROOT).as_posix(), source=source,
               class_name=p.parent.name if source == 'train' else lookup.get(p.stem),
               bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise ValueError('cannot open video')
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        row.update(frame_count=count, fps=cap.get(cv2.CAP_PROP_FPS))
        if count <= 0:
            raise ValueError('frame count unavailable')
        for _ in range(count // 2 + 1):
            ok, image = cap.read()
            if not ok:
                raise ValueError('decode failed before midpoint')
        row.update(midpoint=count // 2, width=image.shape[1], height=image.shape[0],
                   frame_sha256=hashlib.sha256(image.tobytes()).hexdigest(), error=None)
    except Exception as exc:
        row['error'] = str(exc)
    finally:
        cap.release()
    return row

results = []
with ThreadPoolExecutor(max_workers=4) as pool:
    futures = [pool.submit(inspect, p) for p in videos]
    for future in as_completed(futures):
        results.append(future.result())
        if len(results) % 200 == 0:
            print(f'Audited {len(results)}/{len(videos)}', flush=True)
results.sort(key=lambda r:r['video'])
groups = defaultdict(list)
frame_groups = defaultdict(list)
for r in results:
    groups[r['sha256']].append(r['video'])
    if r.get('frame_sha256'):
        frame_groups[r['frame_sha256']].append(r['video'])
report = dict(python=sys.executable, total_videos=len(videos),
    counts=dict(Counter(f"{r['source']}/{r['class_name']}" for r in results)),
    test_csv_rows=len(truth), duplicate_csv_names=truth[truth.filename.duplicated()].to_dict('records'),
    test_missing_labels=[r['video'] for r in results if r['class_name'] is None],
    csv_without_video=sorted(set(truth.filename)-{p.stem for p in videos if p.parent.name=='test'}),
    decode_errors=[r for r in results if r['error']],
    duplicate_video_groups=[v for v in groups.values() if len(v)>1],
    duplicate_midpoint_groups=[v for v in frame_groups.values() if len(v)>1],
    dependencies={name:bool(importlib.util.find_spec(name)) for name in ['ultralytics','torch','mediapipe','pandas','cv2','yaml','nbformat']},
    checkpoint_files=[str(p.relative_to(ROOT)) for p in ROOT.rglob('*.pt')],
    manifest=None)
manifest = OUT / 'manifest.csv'
if manifest.exists():
    df = pd.read_csv(manifest, keep_default_na=False)
    active = df['include'].astype(str).str.lower().eq('true')
    coords = df[['x1','y1','x2','y2']].apply(pd.to_numeric, errors='coerce')
    report['manifest'] = dict(rows=len(df), included=int(active.sum()),
        reviewed=int((active & df.reviewed.astype(str).str.lower().eq('true')).sum()),
        boxes_numeric=int((active & coords.notna().all(axis=1)).sum()),
        frames_chosen=int((active & pd.to_numeric(df.frame_index,errors='coerce').ge(0)).sum()),
        splits=df.loc[active].groupby(['class_name','split']).size().to_dict().__str__(),
        existing_frame_images=len(list((OUT / 'frames').glob('*.jpg'))))
import torch
report['torch'] = dict(version=torch.__version__, cuda=torch.cuda.is_available(), gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
(OUT / 'dataset_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(OUT / 'video_audit.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2), flush=True)
