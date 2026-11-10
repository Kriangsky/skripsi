"""Deterministic technical selection from INTERNAL cached candidates only."""
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

import model_a as a

POLICY = a.HERE / 'config/frame_selection_policy_v1.json'
MANIFEST = a.HERE / 'manifests/automatic_frame_selection_v1.csv'
REPORT = a.HERE / 'reports/automatic_selection_revision'


def rank(candidate):
    flags = candidate['quality_flags']
    return (bool(flags), candidate['num_hands'] == 0, len(flags),
            -candidate['focus'], candidate['frame_index'], candidate['candidate_id'])


def inspect_video(job):
    row, catalogue, recipe = job
    # Reuse only the audited technical helper; no historical training module imported.
    from frame_candidates import assess_hands
    record_path = a.CACHE / 'candidate_records' / f'{row["sample_id"]}.json'
    cache = json.loads(record_path.read_text(encoding='utf8'))
    a.require(all(cache[k] == row[k] for k in ['sample_id', 'video', 'video_sha256']) and
              cache['recipe'] == recipe, f'Candidate cache provenance mismatch: {row["sample_id"]}')
    cached = {c['candidate_id']: c for c in cache['candidates']}
    a.require(len(cached) == len(cache['candidates']) and set(cached) == set(catalogue), 'Cache/catalogue ID mismatch')
    candidates, checks = [], []
    for candidate_id in sorted(cached):
        c = dict(cached[candidate_id])
        original = catalogue[candidate_id]
        for field in ['sample_id', 'frame_index', 'frame_path', 'frame_sha256', 'num_hands', 'width', 'height']:
            expected = row['sample_id'] if field == 'sample_id' else c[field]
            a.require(str(original[field]) == str(expected), f'Cache/catalogue mismatch: {candidate_id}/{field}')
        a.require(json.loads(original['quality_flags']) == c['quality_flags'] and
                  json.loads(original['landmarks']) == c['landmarks'] and
                  math.isclose(float(original['focus']), float(c['focus']), rel_tol=1e-12, abs_tol=1e-12),
                  f'Quality provenance mismatch: {candidate_id}')
        path = a.safe_path(a.CACHE, c['frame_path'])
        a.require(path.is_file() and a.sha(path) == c['frame_sha256'], f'Candidate missing/hash mismatch: {candidate_id}')
        image = cv2.imread(str(path))
        a.require(image is not None and image.shape[:2] == (c['height'], c['width']), f'Candidate decode/dimensions failure: {candidate_id}')
        hands = [[SimpleNamespace(x=x, y=y) for x, y in hand] for hand in c['landmarks']]
        _, focus, flags = assess_hands(image, hands)
        a.require(math.isfinite(focus) and flags == c['quality_flags'] and
                  math.isclose(focus, float(c['focus']), rel_tol=1e-10, abs_tol=1e-10),
                  f'Recomputed technical score mismatch: {candidate_id}')
        c.update(focus=focus, quality_flags=flags, readable=True)
        candidates.append(c)
        checks.append(dict(sample_id=row['sample_id'], candidate_id=candidate_id,
                           image_sha256=c['frame_sha256'], width=c['width'], height=c['height'],
                           focus=focus, num_hands=c['num_hands'], quality_flags=json.dumps(flags), status='VERIFIED'))
    a.require(bool(candidates), f'No technically usable candidate: {row["sample_id"]}')
    selected = min(candidates, key=rank)
    result = dict(row, candidate_id=selected['candidate_id'], frame_index=selected['frame_index'],
                  source_frame_path=selected['frame_path'], source_frame_sha256=selected['frame_sha256'],
                  frame_width=selected['width'], frame_height=selected['height'], frame_count=selected['frame_count'],
                  num_hands=selected['num_hands'], focus=selected['focus'],
                  quality_flags=json.dumps(selected['quality_flags']),
                  quality_status='flagged' if selected['quality_flags'] else 'eligible',
                  fallback=bool(selected['quality_flags']),
                  fallback_reason='no_eligible_candidate; ranked readable fallback' if selected['quality_flags'] else '',
                  semantic_status='not_assessed', label_provenance='frozen_video_metadata_semantics_unverified',
                  selection_policy='model_a_cached_full_frame_v1', candidate_count=len(candidates),
                  candidate_record_path=record_path.relative_to(a.ROOT).as_posix(),
                  candidate_record_sha256=a.sha(record_path), candidate_record_mtime_ns=record_path.stat().st_mtime_ns,
                  extraction_timestamp='NOT_RECORDED_IN_SOURCE_CACHE',
                  source_frame_mtime_ns=a.safe_path(a.CACHE, selected['frame_path']).stat().st_mtime_ns)
    return result, checks


def build():
    REPORT.mkdir(parents=True, exist_ok=True)
    policy = json.loads(POLICY.read_text())
    config = json.loads((a.CACHE / 'candidate_config.json').read_text())
    a.require(policy['quality_heuristics'] == config['quality_heuristics'], 'Technical thresholds changed')
    a.require(config['split_lock_sha256'] == a.sha(a.LOCK), 'Frozen split hash changed')
    sys.path.insert(0, str(a.ROOT / '05_kode/src'))
    import frame_candidates
    a.require(frame_candidates.QUALITY == policy['quality_heuristics'], 'Technical helper thresholds changed')
    cv2.setNumThreads(1)
    inventory = a.verify_inventory(a.read_csv(a.LOCK))
    ids = set(inventory.sample_id)
    catalogue = a.read_csv(a.CACHE / 'candidates.csv')
    catalogue = catalogue[catalogue.sample_id.isin(ids)]
    a.require(set(catalogue.sample_id) == ids and not catalogue.candidate_id.duplicated().any(), 'Internal candidate coverage/IDs differ')
    groups = {sid: group.set_index('candidate_id', drop=False).to_dict('index') for sid, group in catalogue.groupby('sample_id')}
    prior = a.read_csv(a.SELECTION).set_index('sample_id')
    rows, checks, failures = [], [], []
    jobs = [(row, groups[row['sample_id']], config['annotation_recipe_sha256'])
            for row in inventory.sort_values('sample_id').to_dict('records')]
    def checked(job):
        try:
            return inspect_video(job), None
        except Exception as error:
            return None, dict(sample_id=job[0]['sample_id'], video=job[0]['video'], error=f'{type(error).__name__}: {error}')
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (result, error) in enumerate(pool.map(checked, jobs), 1):
            if error:
                failures.append(error)
            else:
                row, evidence = result
                old = prior.loc[row['sample_id']] if row['sample_id'] in prior.index else None
                row['prior_candidate_id'] = old.candidate_id if old is not None else ''
                row['prior_frame_index'] = old.frame_index if old is not None else ''
                row['prior_selection_manifest'] = str(a.SELECTION.relative_to(a.ROOT)) if old is not None else ''
                row['selection_origin'] = 'reused_existing_selection' if old is not None and old.candidate_id == row['candidate_id'] else 'new_selection_from_existing_cache'
                rows.append(row)
                checks.extend(evidence)
            if i % 100 == 0:
                print(f'Internal candidates verified: {i}/1800 videos', flush=True)
    a.write_json(REPORT / 'technical_failures.json', dict(count=len(failures), failures=failures))
    a.require(not failures, f'{len(failures)} technical failures; exact IDs in {REPORT / "technical_failures.json"}')
    frame = pd.DataFrame(rows)
    frame['historical_review_concern'] = frame.sample_id.isin(set(a.review_concerns().sample_id)).astype(str)
    if MANIFEST.exists():
        previous = a.read_csv(MANIFEST)
        a.require(previous.to_csv(index=False) == frame.astype(str).to_csv(index=False), 'Automatic selection release changed; previous manifest preserved')
    else:
        frame.to_csv(MANIFEST, index=False)
    pd.DataFrame(checks).to_csv(REPORT / 'candidate_integrity.csv', index=False)
    concerns = a.review_concerns()
    pilot = a.read_csv(a.PILOT.parent / 'pilot_selection_report.csv')
    lookup = frame.set_index('sample_id')
    pilot_lookup = pilot.set_index('video_id')
    for index, response in concerns.iterrows():
        if response.sample_id not in lookup.index:
            continue
        selected = lookup.loc[response.sample_id]
        old = pilot_lookup.loc[response.sample_id]
        concerns.loc[index, 'current_candidate_id'] = selected.candidate_id
        concerns.loc[index, 'pilot_candidate_id'] = old.candidate_id
        concerns.loc[index, 'same_candidate_as_pilot'] = str(selected.candidate_id == old.candidate_id)
        concerns.loc[index, 'current_technical_status'] = 'READABLE_INTEGRITY_VERIFIED'
        concerns.loc[index, 'current_quality_flags'] = selected.quality_flags
        concerns.loc[index, 'adjudication'] = 'Retained original video label; historical semantic concern is not proof of technically unusable image'
    concerns = concerns[concerns.sample_id.isin(ids)]
    concerns.to_csv(REPORT / 'flagged_video_review.csv', index=False)
    summary = dict(status='PASS_TECHNICAL_ONLY', selected_frames=len(frame), candidates_verified=len(checks),
                   frames_extracted=0, selected_cached_frames_reused=len(frame),
                   prior_selections_reused=int(frame.selection_origin.eq('reused_existing_selection').sum()),
                   new_selections=int(frame.selection_origin.eq('new_selection_from_existing_cache').sum()),
                   fallback_count=int(frame.fallback.sum()),
                   quality_flags=dict(Counter(f for value in frame.quality_flags for f in json.loads(value))),
                   selected_hand_counts={str(k): int(v) for k, v in frame.num_hands.value_counts().items()},
                   historical_concern_videos=int(frame.historical_review_concern.eq('True').sum()),
                   reviewer_decisions=dict(Counter(concerns.decision)),
                   technical_failures=failures, policy_sha256=a.sha(POLICY), manifest_sha256=a.sha(MANIFEST),
                   provenance_sources={str(p.relative_to(a.ROOT)): a.sha(p) for p in
                       [a.LOCK, a.CACHE / 'candidate_config.json', a.CACHE / 'candidates.csv', a.SELECTION,
                        a.PILOT, a.PILOT.parent / 'pilot_selection_report.csv',
                        a.ROOT / '05_kode/src/frame_candidates.py', a.ROOT / '05_kode/src/bisindo_pipeline.py']},
                   limitation='Technical scores do not establish semantic validity or independent signers. Cache has no extraction timestamps; filesystem mtimes preserved as observations only.')
    a.write_json(REPORT / 'selection_summary.json', summary)
    return frame


def contact_sheets(frame):
    from PIL import Image, ImageDraw
    folder = REPORT / 'contact_sheets'
    folder.mkdir(exist_ok=True)
    evidence = []
    # Include each class/split, flagged and eligible frames, first stable IDs; no score-driven sampling.
    for (name, split), group in frame.groupby(['class_name', 'split']):
        selected = pd.concat([g.sort_values('sample_id').head(6) for _, g in group.groupby('quality_status')])
        canvas = Image.new('RGB', (4*280, 3*240), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(selected.itertuples()):
            with Image.open(a.safe_path(a.CACHE, row.source_frame_path)) as image:
                tile = a.LetterboxRGB(210)(image)
            x, y = (i % 4)*280, (i // 4)*240
            canvas.paste(tile, (x, y))
            draw.text((x, y+210), f'{row.sample_id} {row.quality_status}', fill='black')
            draw.text((x, y+225), f'{name}/{split} hands={row.num_hands}', fill='black')
            evidence.append(dict(sample_id=row.sample_id, class_name=name, split=split, quality_status=row.quality_status))
        canvas.save(folder / f'{name}_{split}.jpg')
    pd.DataFrame(evidence).to_csv(REPORT / 'visual_audit_samples.csv', index=False)
    flagged = frame[frame.historical_review_concern.eq('True')].sort_values('sample_id')
    for start in range(0, len(flagged), 25):
        canvas = Image.new('RGB', (5*250, 5*245), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(flagged.iloc[start:start+25].itertuples()):
            with Image.open(a.safe_path(a.CACHE, row.source_frame_path)) as image:
                canvas.paste(a.LetterboxRGB(210)(image), ((i % 5)*250, (i // 5)*245))
            draw.text(((i % 5)*250, (i // 5)*245+212), row.sample_id, fill='black')
            draw.text(((i % 5)*250, (i // 5)*245+227), f'{row.class_name} {row.split} hands={row.num_hands}', fill='black')
        canvas.save(folder / f'historical_concerns_{start//25+1}.jpg')
    return folder


if __name__ == '__main__':
    selected = build()
    contact_sheets(selected)
    print((REPORT / 'selection_summary.json').read_text())
