"""Read-only TASK 1 audit. Writes only a fresh evidence directory, never source data."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback

import cv2
import pandas as pd
from PIL import Image

import bisindo_pipeline as pipeline
import classification_pilot as classification
import confirmatory_evaluation as evaluation
import interference_metrics as interference

ROOT = Path(__file__).resolve().parents[2]
V3 = Path('02_exp/arsip_pipeline/experiment_yolo26_v3')
IDENTITY = ['sample_id', 'video', 'source', 'split', 'class_id', 'class_name', 'video_sha256']


def validate_identity(frame):
    required = set(IDENTITY + ['signer'])
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing inventory columns: {required-set(frame.columns)}')
    for key in ['sample_id', 'video']:
        if frame[key].duplicated().any() or frame[key].astype(str).str.strip().eq('').any():
            raise ValueError(f'Duplicate/empty {key}')
    for row in frame.itertuples():
        if row.class_name not in pipeline.CLASSES or row.class_id != pipeline.CLASSES.index(row.class_name):
            raise ValueError('Class mapping mismatch')
        if row.source not in ['train', 'test'] or row.split not in ['train', 'val', 'test']:
            raise ValueError('Unknown source/split')
        if (row.source == 'test') != (row.split == 'test'):
            raise ValueError('Internal/external split mismatch')
        parts = row.video.replace('\\', '/').split('/')
        if parts[0] != row.source or any(p in ['.', '..', ''] for p in parts):
            raise ValueError('Unsafe/mismatched video path')
        if row.sample_id != pipeline.sid(row.video):
            raise ValueError('Sample ID does not bind video path')
        if len(row.video_sha256) != 64 or any(c not in '0123456789abcdef' for c in row.video_sha256):
            raise ValueError('Invalid stored video SHA256')


def duplicate_groups(rows, field):
    groups = defaultdict(list)
    for row in rows:
        if row.get(field):
            groups[row[field]].append(row)
    return [dict(digest=digest, rows=group, cross_split=len({r['split'] for r in group}) > 1)
            for digest, group in groups.items()
            if len({r['sample_id'] for r in group}) > 1]


def inspect_file(job):
    path = Path(job['path'])
    result = dict(job)
    if not path.is_file():
        return dict(result, status='UNAVAILABLE', error='Source file missing')
    before = path.stat()
    try:
        result.update(bytes=before.st_size, sha256=pipeline.sha(path))
        if job['kind'] != 'video':
            # Bound individual decode work; do not decode videos or run any model.
            with Image.open(path) as header:
                width, height = header.size
            if width*height > 20_000_000 or before.st_size > 100_000_000:
                raise ValueError('Image exceeds bounded audit decode limits')
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cached image decode failed')
            result.update(width=image.shape[1], height=image.shape[0],
                          pixel_sha_bgr=classification.pixel_sha(image),
                          pixel_sha_rgb=classification.pixel_sha(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Input changed during audit read')
        result['status'] = 'VERIFIED' if result['sha256'] == job['expected_sha256'] else 'HASH_MISMATCH'
    except (OSError, ValueError, cv2.error) as error:
        result.update(status='UNVERIFIED', error=str(error))
    return result


def run(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    if not out.is_relative_to(root/'01_hasil/addendum'):
        raise ValueError('Audit evidence must be under existing 01_hasil/addendum structure')
    out.mkdir(exist_ok=False)
    inputs, failures = {}, []
    def read_csv(relative, required=()):
        path = pipeline.safe_path(root, str(relative))
        inputs[str(path.relative_to(root))] = pipeline.sha(path)
        frame = pd.read_csv(path, keep_default_na=False)
        if not set(required).issubset(frame.columns):
            raise ValueError(f'Missing columns: {relative}')
        return frame
    def read_json(relative):
        path = pipeline.safe_path(root, str(relative))
        inputs[str(path.relative_to(root))] = pipeline.sha(path)
        return evaluation.read(path)
    def check(condition, message):
        if not condition:
            failures.append(message)
    started = datetime.now(timezone.utc).isoformat()
    protected = [*root.joinpath('06_notebook').rglob('*.ipynb'), *root.joinpath('02_exp').rglob('*.pt')]
    protected += [p for folder in ['01_hasil','03_insight_exp'] for p in (root/folder).rglob('*')
                  if p.is_file() and not p.is_relative_to(out)]
    protected_hashes = {str(p.relative_to(root)):pipeline.sha(p) for p in protected}
    evaluation.write_new(out/'protected_before.json',protected_hashes)
    lock = read_csv(V3/'split_lock.csv')
    validate_identity(lock)
    inventory = read_csv(V3/'inventory.csv')
    validate_identity(inventory)
    check(lock.sort_values('sample_id').reset_index(drop=True).equals(
          inventory.sort_values('sample_id').reset_index(drop=True)), 'v3 inventory differs from split lock')
    truth = read_csv('08_review/ground_truth/test_ground_truth.csv', ['filename', 'ground_truth', 'original_filename'])
    check(not truth.filename.duplicated().any(), 'Duplicate neutral test filenames')
    lookup = truth.set_index('filename').ground_truth.to_dict()
    check(set(lookup.values()) == set(pipeline.CLASSES), 'External truth class coverage differs')
    for row in lock.itertuples():
        if row.source == 'test':
            check(lookup.get(Path(row.video).stem) == row.class_name, f'External GT mismatch {row.sample_id}')
    original_labels = truth.original_filename.str.removeprefix('BISINDO_').str.rsplit('_', n=1).str[0].str.lower().str.replace(' ', '_')
    check(original_labels.equals(truth.ground_truth), 'Original filename label disagrees with GT CSV')
    physical = {str(p.relative_to(root/'04_dataset')).replace('\\', '/')
                for source in ['train', 'test'] for p in (root/'04_dataset'/source).rglob('*')
                if p.is_file() and p.suffix.lower() in pipeline.EXT}
    expected_paths = set(lock.video)
    missing, extra = sorted(expected_paths-physical), sorted(physical-expected_paths)
    jobs = [dict(path=str(root/'04_dataset'/row.video), kind='video', sample_id=row.sample_id,
                 source=row.source, split=row.split, expected_sha256=row.video_sha256)
            for row in lock.itertuples()]
    catalog = read_csv(V3/'candidates.csv', ['sample_id','candidate_id','frame_path','frame_sha256','frame_index','width','height'])
    check(not catalog.candidate_id.duplicated().any(), 'Duplicate candidate IDs')
    check(set(catalog.sample_id) == set(lock.sample_id), 'Candidate coverage differs from inventory')
    config = read_json(V3/'candidate_config.json')
    check(config['split_lock_sha256'] == inputs[str(V3/'split_lock.csv')], 'Candidate config split hash drift')
    by_id = lock.set_index('sample_id').to_dict('index')
    catalog_groups = {sid: group for sid, group in catalog.groupby('sample_id')}
    for row in lock.itertuples():
        cache = read_json(V3/'candidate_records'/f'{row.sample_id}.json')
        check(all(cache[key] == getattr(row, key) for key in ['sample_id','video','video_sha256']), f'Cache source mismatch {row.sample_id}')
        check(cache['recipe'] == config['annotation_recipe_sha256'], f'Cache recipe mismatch {row.sample_id}')
        cached = {x['candidate_id']: x for x in cache['candidates']}
        group = catalog_groups[row.sample_id]
        check(len(cached) == len(cache['candidates']) and set(cached) == set(group.candidate_id), f'Cache catalog IDs mismatch {row.sample_id}')
        for item in group.itertuples():
            for key in ['frame_index','frame_path','frame_sha256','width','height']:
                check(cached[item.candidate_id][key] == getattr(item,key), f'Cache field mismatch {item.candidate_id}/{key}')
    expected_frames = set(catalog.frame_path)
    existing_frames = {p.relative_to(root/V3).as_posix() for p in (root/V3/'candidate_frames').glob('*.jpg')}
    check(expected_frames == existing_frames, 'Uncatalogued or missing v3 cached candidate image')
    for row in catalog.itertuples():
        identity = by_id[row.sample_id]
        jobs.append(dict(path=str(root/V3/row.frame_path), kind='candidate_v3', sample_id=row.sample_id,
                         source=identity['source'], split=identity['split'], expected_sha256=row.frame_sha256,
                         expected_width=row.width, expected_height=row.height))
    annotations = read_csv(V3/'annotations.csv')
    validate_identity(annotations)
    check(annotations[IDENTITY].equals(lock[IDENTITY]), 'v3 annotation identity differs from locked split')
    old = read_csv('02_exp/arsip_pipeline/experiment_yolo26_v2/annotations.csv')
    validate_identity(old)
    check(old[IDENTITY].equals(lock[IDENTITY]), 'v2 identity differs from v3 lock')
    for row in old[old.source.eq('test')].itertuples():
        jobs.append(dict(path=str(root/'02_exp/arsip_pipeline/experiment_yolo26_v2/frames'/f'{row.sample_id}.jpg'),
                         kind='external_v2_cached', sample_id=row.sample_id, source=row.source, split=row.split,
                         expected_sha256=row.frame_sha256))
    plan = read_json('07_dokumen/protokol/analysis_plan.json')
    manifests, pairs, reference = {}, [], None
    for seed in plan['training_seeds']:
        work = evaluation.resolve(root, plan['development_runs'][str(seed)])
        manifest = read_csv(work.relative_to(root)/'manifest.csv')
        validate_identity(manifest)
        check(manifest.source.eq('train').all(), f'External sample in seed {seed} development manifest')
        target = lock.set_index('sample_id').loc[manifest.sample_id].reset_index()
        check(manifest[IDENTITY].reset_index(drop=True).equals(target[IDENTITY]), f'Locked identity mismatch seed {seed}')
        manifests[seed] = manifest
        compare = manifest[IDENTITY+['candidate_id','frame_index','source_frame_sha256','image_sha256']].sort_values('sample_id').reset_index(drop=True)
        if reference is None:
            reference = compare
        check(compare.equals(reference), f'Across-seed sample/frame identity mismatch seed {seed}')
        frames = evaluation.enriched_predictions(work, seed)
        interference.align_predictions(frames['A'], frames['B'], range(6))
        read_csv(work.relative_to(root)/'evaluation/predictions.csv')
        common = manifest[manifest.class_id.lt(6)]
        pairs.append(dict(seed=seed, common6_val_n=len(frames['A']), B_full8_val_n=len(frames['B']),
                          shared_train_n=int(common.split.eq('train').sum()), shared_val_n=int(common.split.eq('val').sum()),
                          identical_common6_samples_gt_video_and_image_hashes=True))
        candidate_lookup = catalog.set_index('candidate_id')
        for row in manifest.itertuples():
            c = candidate_lookup.loc[row.candidate_id]
            check(c.sample_id == row.sample_id and c.frame_index == row.frame_index and c.frame_sha256 == row.source_frame_sha256,
                  f'Selected provenance mismatch {seed}/{row.sample_id}')
            for name, classes in [('A',6),('B',8)]:
                if row.class_id >= classes:
                    continue
                jobs.append(dict(path=str(work/'datasets'/name/row.split/f'{row.class_id:02d}_{row.class_name}'/f'{row.sample_id}.png'),
                                 kind='development_png', sample_id=row.sample_id, source=row.source, split=row.split,
                                 expected_sha256=row.image_sha256, expected_pixel_sha_bgr=row.processed_pixel_sha256))
            jobs.append(dict(path=str(work/'selected_frames'/f'{row.sample_id}.jpg'), kind='development_selected',
                             sample_id=row.sample_id, source=row.source, split=row.split, expected_sha256=row.source_frame_sha256,
                             expected_pixel_sha_bgr=row.pixel_sha256))
    if any(not Path(job['path']).resolve().is_relative_to(root) for job in jobs):
        raise ValueError('Audit input path escapes repository')
    print(f'Audit jobs: {len(jobs)}; workers=4; no raw-video decoding or model loading', flush=True)
    rows = []
    with (out/'file_integrity.jsonl').open('x', encoding='utf8') as log, ThreadPoolExecutor(max_workers=4) as pool:
        for index, item in enumerate(pool.map(inspect_file, jobs),1):
            for key, expected in [('pixel_sha_bgr','expected_pixel_sha_bgr'),('width','expected_width'),('height','expected_height')]:
                if expected in item and item.get(key) != item[expected]:
                    item['status'] = 'CONTENT_MISMATCH'
            rows.append(item)
            log.write(json.dumps(item)+'\n')
            if index % 5000 == 0:
                print(f'Checked {index}/{len(jobs)}', flush=True)
    bad = [row for row in rows if row['status'] != 'VERIFIED']
    videos = [row for row in rows if row['kind'] == 'video']
    candidates = [row for row in rows if row['kind'] == 'candidate_v3']
    actual_groups = duplicate_groups(videos, 'sha256')
    frame_groups = duplicate_groups(rows, 'pixel_sha_rgb')
    candidate_byte_groups = duplicate_groups(candidates, 'sha256')
    metadata_groups = duplicate_groups(lock.rename(columns={'video_sha256':'stored_sha'}).to_dict('records'),'stored_sha')
    def counts(frame):
        return frame.groupby(['source','split','class_name']).size().rename('n').reset_index().to_dict('records')
    summary = dict(schema_version=1, task=1, started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                   invocation=[sys.executable,*sys.argv], git=evaluation.git_state(root),
                   scope='read-only actual video bytes, existing cached image bytes/decoded RGB, internal predictions only',
                   raw_video_decode_performed=False, frame_extraction_or_reselection=False, training=False, external_model_predictions_accessed=False,
                   expected=dict(internal=2400,external=400,internal_train=1920,internal_val=480,external_common6=300,external_full8=400),
                   observed=dict(inventory_rows=len(lock),internal=int(lock.source.eq('train').sum()),external=int(lock.source.eq('test').sum()),
                                 internal_train=int(lock.split.eq('train').sum()), internal_val=int(lock.split.eq('val').sum()),
                                 external_common6=int((lock.source.eq('test') & lock.class_id.lt(6)).sum()),
                                 external_full8=int(lock.source.eq('test').sum()),physical_video_files=len(physical),
                                 physical_verified_files=sum(row['status']=='VERIFIED' for row in videos)),
                   class_mapping=list(pipeline.CLASSES),class_split_counts=counts(lock),
                   missing_source_paths=missing,extra_source_paths=extra,integrity_failures=bad,identity_failures=failures,
                   internal_pairing=pairs, across_seed_manifest_identity=not any('Across-seed' in x for x in failures),
                   candidate_count=len(catalog),candidate_cache_records=len(catalog_groups),
                   scanned_file_counts=dict(Counter(row['kind'] for row in rows)),
                   frame_provenance=dict(v3_selected_positive_frames=int(pd.to_numeric(annotations.frame_index).ge(0).sum()),
                                         v3_review_status=annotations.groupby(['source','review_status']).size().rename('n').reset_index().to_dict('records'),
                                         development_semantic_status={str(k):int(v) for k,v in manifests[42].semantic_status.value_counts().items()},
                                         development_technical_status={str(k):int(v) for k,v in manifests[42].quality_status.value_counts().items()},
                                         candidate_config=config),
                   external_signer_nonempty=int(lock.loc[lock.source.eq('test'),'signer'].str.strip().ne('').sum()),
                   internal_signer_nonempty=int(lock.loc[lock.source.eq('train'),'signer'].str.strip().ne('').sum()),
                   final_manifest_paths={key:plan['artifacts'][key] for key in ['external_manifest','internal_manifest']},
                   neutral_gt_csv_rows=len(truth), neutral_mapping_status='VERIFIED within local CSV/files; original acquisition labels UNVERIFIED',
                   duplicates=dict(metadata_only_groups=metadata_groups,actual_video_groups=actual_groups,
                                   actual_candidate_byte_groups=candidate_byte_groups, decoded_rgb_distinct_video_groups=frame_groups,
                                   near_duplicates='UNVERIFIED: not evaluated',
                                   scope='All accessible v3 cached candidates, v2 external cached images, five-seed selected JPEG and training/validation PNG; no all-frame video comparison'),
                   inputs_sha256=inputs,source_sha256={p.name:pipeline.sha(p) for p in [Path(__file__),Path(pipeline.__file__),Path(classification.__file__),Path(evaluation.__file__)]})
    for key,value in summary['expected'].items():
        check(summary['observed'][key] == value, f'Expected/observed count mismatch {key}')
    check(all(item['n']==(240 if item['split']=='train' else 60 if item['split']=='val' else 50)
              for item in summary['class_split_counts']), 'Per-class split count mismatch')
    summary['technical_audit_passed'] = not (bad or failures or missing or extra)
    # Rehash source videos at the end; changes during this audit cannot pass silently.
    summary['video_end_hash_mismatches'] = [row['path'] for row in videos
                                          if row.get('sha256') and pipeline.sha(row['path']) != row['sha256']]
    summary['metadata_end_hash_mismatches'] = [name for name,digest in inputs.items() if pipeline.sha(root/name) != digest]
    summary['protected_end_hash_mismatches'] = [name for name,digest in protected_hashes.items() if pipeline.sha(root/name) != digest]
    summary['technical_audit_passed'] &= not (summary['video_end_hash_mismatches'] or summary['metadata_end_hash_mismatches'] or summary['protected_end_hash_mismatches'])
    summary['completed_utc'] = datetime.now(timezone.utc).isoformat()
    evaluation.write_new(out/'audit.json',summary)
    print(json.dumps({key:summary[key] for key in ['observed','scanned_file_counts','technical_audit_passed']},indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError('Refusing to overwrite prior audit evidence')
    record=dict(command=[sys.executable,*sys.argv],cwd=str(Path.cwd()),started_utc=datetime.now(timezone.utc).isoformat())
    try:
        result=run(ROOT,args.out)
        code=0 if result['technical_audit_passed'] else 1
        record.update(status='completed',exit_code=code)
    except Exception:
        record.update(status='failed',exit_code=1,error=traceback.format_exc())
        raise
    finally:
        record['completed_utc']=datetime.now(timezone.utc).isoformat()
        if args.out.is_dir():
            evaluation.write_new(args.out/'execution.json',record)
    return code


if __name__=='__main__':
    raise SystemExit(main())
