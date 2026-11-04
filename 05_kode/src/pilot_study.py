"""Frozen single-frame feasibility pilot. Never trains, relabels, or writes to v3."""
import argparse
import base64
from collections import Counter
from datetime import datetime
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import secrets
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

import bisindo_pipeline as b
import frame_candidates as f

CLASSES = b.CLASSES
CATEGORIES = CLASSES + ['ambiguous', 'invalid']
REASONS = ['preparation', 'transition', 'hand_clipped', 'blur', 'temporal_insufficient', 'other']
REVIEWERS = ['reviewer_1', 'reviewer_2']
RULE = 'v1: readable; eligible first; fewer technical flags; focus descending; frame index ascending'
LOG = logging.getLogger(__name__)


def read_csv(path):
    return pd.read_csv(path, keep_default_na=False)


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sample_videos(inventory, per_class=20, seed=42):
    """Round-robin known signers, then unknown records; never invent signer identities."""
    if per_class != 20 or seed != 42:
        raise ValueError('Primary pilot requires 20 per class and seed 42')
    required = {'sample_id', 'video', 'source', 'split', 'class_name', 'class_id', 'video_sha256', 'signer'}
    if not required.issubset(inventory.columns):
        raise ValueError(f'Missing inventory columns: {required - set(inventory.columns)}')
    for field in ['sample_id', 'video', 'video_sha256']:
        if inventory[field].duplicated().any() or inventory[field].astype(str).str.strip().eq('').any():
            raise ValueError(f'Duplicate/empty {field}; leakage or ambiguous identity')
    for row in inventory.itertuples():
        if row.class_name not in CLASSES or str(row.class_id) != str(CLASSES.index(row.class_name)):
            raise ValueError('Invalid class mapping')
        if row.source not in ['train', 'test'] or row.split not in ['train', 'val', 'test'] or (row.source == 'test') != (row.split == 'test'):
            raise ValueError('Invalid frozen source/split')
        if not str(row.video).replace('\\', '/').startswith(row.source + '/'):
            raise ValueError('Source/path mismatch')
        if any(part in ['.', '..'] for part in str(row.video).replace('\\', '/').split('/')):
            raise ValueError('Source path traversal')
    pool = inventory[inventory.source.eq('train') & inventory.split.eq('train')]
    rng = random.Random(seed)
    chosen = []
    for cls in CLASSES:
        rows = pool[pool.class_name.eq(cls)].sort_values('sample_id').to_dict('records')
        if len(rows) < per_class:
            raise ValueError(f'{cls}: training pool {len(rows)} < {per_class}')
        known, unknown = {}, []
        for row in rows:
            signer = str(row['signer']).strip()
            if signer:
                known.setdefault(signer, []).append(row)
            else:
                unknown.append(row)
        groups = [known[k] for k in sorted(known)]
        for group in groups:
            rng.shuffle(group)
        rng.shuffle(groups)
        rng.shuffle(unknown)
        # Unknown metadata is one residual pool, never a set of fictional signers.
        if unknown:
            groups.append(unknown)
        count = 0
        while count < per_class:
            for group in groups:
                if group and count < per_class:
                    chosen.append(group.pop())
                    count += 1
    return pd.DataFrame(chosen)


def choose_candidate(candidates):
    readable = [c for c in candidates if c['readable']]
    return min(readable, key=lambda c: (bool(c['quality_flags']), len(c['quality_flags']),
        -c['focus'], c['frame_index'], c['candidate_id'])) if readable else None


def inspect_candidates(root, source, record, catalog, config):
    """Validate cached provenance; reuse v3 technical checks, without another detector run."""
    path = b.safe_path(root, record['video'])
    if b.sha(path) != record['video_sha256']:
        raise ValueError(f"Source video changed: {record['sample_id']}")
    cache_path = source / 'candidate_records' / f"{record['sample_id']}.json"
    if not cache_path.resolve().is_relative_to((source / 'candidate_records').resolve()):
        raise ValueError('Invalid cache path')
    if not cache_path.exists():
        return [], 'missing_candidate_cache'
    cache = json.loads(cache_path.read_text(encoding='utf-8'))
    if (cache['sample_id'] != record['sample_id'] or cache['video'] != record['video'] or
            cache['video_sha256'] != record['video_sha256'] or cache['recipe'] != config['annotation_recipe_sha256']):
        raise ValueError('Stale/mismatched cache provenance')
    cached = {c['candidate_id']: c for c in cache['candidates']}
    if len(cached) != len(cache['candidates']) or set(catalog.candidate_id) != set(cached):
        raise ValueError('Candidate CSV/cache identity mismatch')
    candidates = []
    errors = []
    for row in catalog.to_dict('records'):
        c = cached[row['candidate_id']]
        for field in ['frame_index', 'frame_path', 'frame_sha256', 'quality_status', 'num_hands']:
            if str(c[field]) != str(row[field]):
                raise ValueError(f'Candidate CSV/cache mismatch: {field}')
        if (json.loads(row['quality_flags']) != c['quality_flags'] or
                json.loads(row['landmarks']) != c['landmarks'] or not math.isclose(float(row['focus']), float(c['focus']), rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Candidate quality/cache mismatch')
        result = dict(c, readable=False)
        try:
            frame = b.safe_path(source, c['frame_path'])
            if b.sha(frame) != c['frame_sha256']:
                raise ValueError('candidate_image_hash_mismatch')
            image = cv2.imread(str(frame))
            if image is None:
                raise ValueError('candidate_decode_failed')
            if image.shape[:2] != (c['height'], c['width']):
                raise ValueError('candidate_dimensions_mismatch')
            hands = [[SimpleNamespace(x=x, y=y) for x, y in hand] for hand in c['landmarks']]
            _, focus, flags = f.assess_hands(image, hands)
            if not math.isfinite(focus):
                raise ValueError('nonfinite_focus')
            result.update(readable=True, focus=focus, quality_flags=flags,
                          quality_status='eligible' if not flags else 'flagged')
        except (ValueError, OSError, cv2.error, TypeError) as exc:
            result.update(quality_flags=[str(exc)], quality_status='missing')
            errors.append(f"{c['candidate_id']}: {exc}")
        candidates.append(result)
    return candidates, '; '.join(errors) or cache.get('error', '')


def prepare(root, source, work):
    root, source, work = Path(root).resolve(), Path(source).resolve(), Path(work).resolve()
    if work.exists():
        raise ValueError('Pilot folder already exists; refusing to replace frozen frames or responses')
    if work.is_relative_to(source) or work.is_relative_to(root / 'train') or work.is_relative_to(root / 'test'):
        raise ValueError('Output must be separate from sources')
    config = json.loads((source / 'candidate_config.json').read_text())
    if config['split_lock_sha256'] != b.sha(source / 'split_lock.csv') or config['quality_heuristics'] != f.QUALITY:
        raise ValueError('Frozen split or technical thresholds changed')
    inventory = read_csv(source / 'split_lock.csv')
    signer_path = root / 'signer_metadata.csv'
    signer_digest = None
    if signer_path.exists():
        metadata = pd.read_csv(signer_path, dtype=str, keep_default_na=False)
        if not {'video', 'signer'}.issubset(metadata) or metadata.video.duplicated().any() or not set(metadata.video).issubset(set(inventory.video)):
            raise ValueError('Invalid signer metadata')
        mapping = metadata.set_index('video').signer.to_dict()
        for index, row in inventory.iterrows():
            supplied = mapping.get(row.video, '').strip()
            if row.signer and supplied and row.signer != supplied:
                raise ValueError('Signer metadata conflicts with frozen inventory')
            if supplied:
                inventory.loc[index, 'signer'] = supplied
        signer_digest = b.sha(signer_path)
    sampled = sample_videos(inventory)
    catalog = read_csv(source / 'candidates.csv')
    if catalog.candidate_id.duplicated().any() or not set(catalog.sample_id).issubset(set(inventory.sample_id)):
        raise ValueError('Duplicate or unknown candidate identity')
    records, reports = [], []
    # Public seed must not allow reconstruction of class-block allocation from IDs.
    for row in sampled.to_dict('records'):
        candidates, error = inspect_candidates(root, source, row, catalog[catalog.sample_id.eq(row['sample_id'])], config)
        chosen = choose_candidate(candidates)
        opaque = secrets.token_hex(16)
        records.append(dict(row, video_id=row['sample_id'], sample_id=opaque, seed=42,
            split_lock_sha256=b.sha(source / 'split_lock.csv'), source_metadata_sha256=signer_digest or 'unavailable'))
        reports.append(dict(sample_id=opaque, video_id=row['sample_id'],
            candidate_id=chosen['candidate_id'] if chosen else '', frame_index=chosen['frame_index'] if chosen else -1,
            frame_sha256=chosen['frame_sha256'] if chosen else '',
            selected_path=f'selected_frames/{opaque}.png' if chosen else '',
            selection_status=chosen['quality_status'] if chosen else 'missing',
            focus=chosen['focus'] if chosen else 0., num_hands=chosen['num_hands'] if chosen else 0,
            roi_box=json.dumps(chosen['box'] if chosen else None),
            quality_flags=json.dumps(chosen['quality_flags'] if chosen else ['no_inspectable_frame']),
            candidate_count=len(candidates), unreadable_candidate_count=sum(not c['readable'] for c in candidates),
            reason=RULE if chosen else 'Selection failure retained in denominator', cache_error=error))
    # Validation finishes before any output is written.
    work.mkdir()
    for folder in ['selected_frames', 'review', 'responses', 'analysis']:
        (work / folder).mkdir()
    manifest = pd.DataFrame(records)
    by_id = catalog.set_index('candidate_id').to_dict('index')
    for report in reports:
        if report['selected_path']:
            original = by_id[report['candidate_id']]
            image = cv2.imread(str(b.safe_path(source, original['frame_path'])))
            if image is None or not cv2.imwrite(str(work / report['selected_path']), image):
                raise ValueError('Cannot write selected frame')
            # PNG strips source metadata without another lossy JPEG compression.
            report['review_frame_sha256'] = b.sha(work / report['selected_path'])
        else:
            report['review_frame_sha256'] = ''
    selection = pd.DataFrame(reports)
    manifest.to_csv(work / 'pilot_manifest_private.csv', index=False)
    selection.to_csv(work / 'pilot_selection_report.csv', index=False)
    study_id = digest(dict(manifest=b.sha(work / 'pilot_manifest_private.csv'),
                          selection=b.sha(work / 'pilot_selection_report.csv'), rule=RULE))
    protocol = dict(schema=1, study_id=study_id, seed=42, per_class=20, rule=RULE, classes=CLASSES,
        categories=CATEGORIES, reasons=REASONS, reviewers=REVIEWERS,
        split_lock_sha256=b.sha(source / 'split_lock.csv'), candidates_sha256=b.sha(source / 'candidates.csv'),
        candidate_config_sha256=b.sha(source / 'candidate_config.json'), signer_metadata_sha256=signer_digest,
        manifest_sha256=b.sha(work / 'pilot_manifest_private.csv'), selection_sha256=b.sha(work / 'pilot_selection_report.csv'),
        limitations=['Technical heuristics are not calibrated semantic validity checks.',
            'Signer-independent inference unavailable when signer metadata is missing; video != signer.',
            'Exploratory training-pool pilot; not an external-test benchmark.',
            'Reviewer independence requires separate people/devices and no answer sharing.',
            'Image appearance may itself reveal source cues; opaque files cannot eliminate visual confounding.'],
        known_signer_samples=int(manifest.signer.astype(str).str.strip().ne('').sum()))
    public = [dict(sample_id=r['sample_id'], image=('data:image/png;base64,' +
        base64.b64encode((work / r['selected_path']).read_bytes()).decode()) if r['selected_path'] else '') for r in reports]
    # Seeded reviewer permutations must start from opaque ID order, not class blocks.
    public.sort(key=lambda r: r['sample_id'])
    for number, reviewer in enumerate(REVIEWERS, 1):
        shuffled = list(public)
        random.Random(42 + number).shuffle(shuffled)
        (work / 'review' / f'{reviewer}.html').write_text(review_html(study_id, reviewer, shuffled), encoding='utf-8')
    protocol['review_html_sha256'] = {r: b.sha(work / 'review' / f'{r}.html') for r in REVIEWERS}
    b.save_json(work / 'pilot_protocol_private.json', protocol)
    (work / 'preparation.log').write_text(
        f'Samples={len(manifest)}; seed=42; per_class=20\n' +
        selection.selection_status.value_counts().to_string() + '\n' +
        f'Known signer samples={protocol["known_signer_samples"]}\n', encoding='utf-8')
    analyze(work)
    LOG.info('Prepared %d samples; eligible=%d flagged=%d missing=%d', len(manifest),
        selection.selection_status.eq('eligible').sum(), selection.selection_status.eq('flagged').sum(), selection.selection_status.eq('missing').sum())
    return manifest, selection


def review_html(study_id, reviewer, samples):
    payload = json.dumps(dict(study_id=study_id, reviewer_id=reviewer, samples=samples), ensure_ascii=True).replace('<', '\\u003c')
    return '''<!doctype html><html lang="id"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Pilot BISINDO</title>
<style>body{font:18px system-ui;max-width:1000px;margin:24px auto;padding:16px;background:#f5f5f0;color:#172923}img{max-width:100%;max-height:60vh;display:block;margin:auto}button,select{font:inherit;padding:10px;margin:6px}fieldset{margin:14px 0}label{display:inline-block;margin:10px}#status{white-space:pre-wrap}</style>
<h1>Pilot frame tunggal BISINDO</h1>
<p>Nilai mandiri. Jangan berbagi jawaban atau membuka video sumber. Satu frame tetap per sampel.
Pilih kelas hanya bila dapat dibedakan; ambiguous = tidak dapat dibedakan; invalid = gesture belum terlihat.
Frame tidak tersedia dihitung gagal seleksi dan harus dinilai invalid. Tidak ada umpan balik jawaban benar.</p>
<p>Browser menyimpan draft bila tersedia. Ekspor draft sebagai cadangan. Ekspor final memerlukan seluruh sampel.</p>
<div id="status" role="status"></div><p id="counter"></p><p id="sample"></p>
<img id="frame" alt="Frame untuk dinilai"><p id="missing" hidden>Frame tidak tersedia (kegagalan seleksi).</p>
<label for="decision">Keputusan</label><select id="decision"><option value="">Pilih keputusan</option></select>
<fieldset id="reasons"><legend>Alasan (boleh lebih dari satu)</legend></fieldset>
<button id="prev">Sebelumnya</button><button id="next">Berikutnya</button>
<button id="draft">Ekspor draft</button><button id="export">Ekspor final</button>
<label>Impor draft sendiri <input id="import" type="file" accept="application/json"></label>
<script>
const DATA=__PAYLOAD__;
const categories=__CATEGORIES__, reasons=__REASONS__;
const el=id=>document.getElementById(id), key='bisindo-pilot:'+DATA.study_id+':'+DATA.reviewer_id;
let answers={}, index=0;
function valid(p,complete){
 if(p.schema!==1||p.study_id!==DATA.study_id||p.reviewer_id!==DATA.reviewer_id||!Array.isArray(p.responses))throw Error('Identitas studi/reviewer salah');
 let seen=new Set();
 for(const r of p.responses){if(!DATA.samples.some(s=>s.sample_id===r.sample_id)||seen.has(r.sample_id)||!categories.includes(r.decision)||!Array.isArray(r.reasons)||r.reasons.some(x=>!reasons.includes(x))||new Set(r.reasons).size!==r.reasons.length||!r.timestamp||!Number.isFinite(Date.parse(r.timestamp)))throw Error('Respons tidak valid');seen.add(r.sample_id);}
 if(complete&&seen.size!==DATA.samples.length)throw Error('Semua sampel wajib dijawab');
 for(const s of DATA.samples){const r=p.responses.find(r=>r.sample_id===s.sample_id);if(r&&!s.image&&r.decision!=='invalid')throw Error('Frame tidak tersedia wajib invalid');}
 return Object.fromEntries(p.responses.map(r=>[r.sample_id,r]));
}
function payload(){return {schema:1,study_id:DATA.study_id,reviewer_id:DATA.reviewer_id,responses:Object.values(answers)}}
function persist(){try{localStorage.setItem(key,JSON.stringify(payload()))}catch(e){el('status').textContent='Penyimpanan browser tidak tersedia. Ekspor draft.'}}
function save(){const s=DATA.samples[index], decision=el('decision').value;if(!decision)return;answers[s.sample_id]={sample_id:s.sample_id,decision,reasons:[...document.querySelectorAll('#reasons input:checked')].map(x=>x.value),timestamp:new Date().toISOString()};persist();}
function render(){const s=DATA.samples[index], a=answers[s.sample_id];el('counter').textContent=DATA.reviewer_id+' | '+(index+1)+' / '+DATA.samples.length+' | dijawab '+Object.keys(answers).length;el('sample').textContent=s.sample_id;el('frame').hidden=!s.image;if(s.image)el('frame').src=s.image;else el('frame').removeAttribute('src');el('missing').hidden=!!s.image;el('decision').value=a?a.decision:'';for(const o of el('decision').options)o.disabled=!s.image&&o.value!==''&&o.value!=='invalid';for(const c of document.querySelectorAll('#reasons input'))c.checked=!!a&&a.reasons.includes(c.value);el('prev').disabled=index===0;el('next').disabled=index===DATA.samples.length-1;}
for(const c of categories){const o=document.createElement('option');o.value=c;o.textContent=c;el('decision').append(o)}
for(const r of reasons){const l=document.createElement('label'),c=document.createElement('input');c.type='checkbox';c.value=r;c.onchange=save;l.append(c,document.createTextNode(r));el('reasons').append(l)}
el('decision').onchange=()=>{const id=DATA.samples[index].sample_id;if(!el('decision').value){delete answers[id];persist()}else save();};
el('prev').onclick=()=>{index--;render()};el('next').onclick=()=>{index++;render()};
function download(final){try{valid(payload(),final);const blob=new Blob([JSON.stringify(payload(),null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=DATA.reviewer_id+(final?'_final':'_draft')+'.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}catch(e){el('status').textContent=e.message}}
el('draft').onclick=()=>download(false);el('export').onclick=()=>download(true);
el('import').onchange=async e=>{try{answers=valid(JSON.parse(await e.target.files[0].text()),false);persist();render();el('status').textContent='Draft diimpor'}catch(err){el('status').textContent=err.message}};
try{const stored=localStorage.getItem(key);if(stored)answers=valid(JSON.parse(stored),false)}catch(e){el('status').textContent='Draft tidak dipulihkan: '+e.message}
render();
</script></html>'''.replace('__PAYLOAD__', payload).replace('__CATEGORIES__', json.dumps(CATEGORIES)).replace('__REASONS__', json.dumps(REASONS))


def validate_responses(payload, study_id, reviewer, sample_ids):
    if (not isinstance(payload, dict) or payload.get('schema') != 1 or payload.get('study_id') != study_id or
            payload.get('reviewer_id') != reviewer or reviewer not in REVIEWERS):
        raise ValueError('Stale study or incorrect reviewer identity')
    responses = payload.get('responses')
    if not isinstance(responses, list):
        raise ValueError('Responses must be a list')
    seen = set()
    for row in responses:
        if not isinstance(row, dict) or set(row) != {'sample_id', 'decision', 'reasons', 'timestamp'}:
            raise ValueError('Invalid response fields')
        sid = row['sample_id']
        if not isinstance(sid, str) or sid not in sample_ids or sid in seen:
            raise ValueError('Duplicate or unknown sample')
        seen.add(sid)
        if not isinstance(row['decision'], str) or row['decision'] not in CATEGORIES:
            raise ValueError('Invalid decision')
        reasons = row['reasons']
        if not isinstance(reasons, list) or any(not isinstance(r, str) or r not in REASONS for r in reasons) or len(set(reasons)) != len(reasons):
            raise ValueError('Invalid reasons')
        try:
            stamp = datetime.fromisoformat(row['timestamp'].replace('Z', '+00:00'))
            if stamp.utcoffset() is None:
                raise ValueError('Timezone required')
        except (ValueError, TypeError, AttributeError) as exc:
            raise ValueError('Invalid timestamp') from exc
    if seen != set(sample_ids):
        raise ValueError('Incomplete responses; all pilot samples required')
    return responses


def validate_frozen(work):
    work = Path(work)
    protocol = json.loads((work / 'pilot_protocol_private.json').read_text())
    if protocol['classes'] != CLASSES or protocol['categories'] != CATEGORIES or protocol['rule'] != RULE:
        raise ValueError('Pilot protocol differs from implemented protocol')
    for filename, field in [('pilot_manifest_private.csv', 'manifest_sha256'), ('pilot_selection_report.csv', 'selection_sha256')]:
        if b.sha(work / filename) != protocol[field]:
            raise ValueError(f'Frozen pilot modified: {filename}')
    expected_id = digest(dict(manifest=protocol['manifest_sha256'], selection=protocol['selection_sha256'], rule=RULE))
    if protocol['study_id'] != expected_id:
        raise ValueError('Pilot study ID mismatch')
    for reviewer in REVIEWERS:
        if b.sha(work / 'review' / f'{reviewer}.html') != protocol['review_html_sha256'][reviewer]:
            raise ValueError('Frozen reviewer HTML modified')
    for row in read_csv(work / 'pilot_selection_report.csv').itertuples():
        if row.selected_path and b.sha(b.safe_path(work, row.selected_path)) != row.review_frame_sha256:
            raise ValueError('Frozen review image modified')
    return protocol


def import_review(work, response_file, reviewer):
    work = Path(work)
    protocol = validate_frozen(work)
    payload = json.loads(Path(response_file).read_text(encoding='utf-8-sig'))
    selection = read_csv(work / 'pilot_selection_report.csv')
    responses = validate_responses(payload, protocol['study_id'], reviewer, set(selection.sample_id))
    missing = set(selection.loc[selection.selection_status.eq('missing'), 'sample_id'])
    if any(r['sample_id'] in missing and r['decision'] != 'invalid' for r in responses):
        raise ValueError('Missing image must be invalid; cannot be recognized')
    target = work / 'responses' / f'{reviewer}.json'
    if target.exists():
        raise ValueError('Reviewer response already imported; refusing overwrite')
    with target.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    LOG.info('Imported complete independent response: %s (%d)', reviewer, len(responses))


def wilson(successes, n):
    if not n:
        return None, None
    z = 1.959963984540054
    phat = successes / n
    denominator = 1 + z*z/n
    center = (phat + z*z/(2*n))/denominator
    half = z*math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/denominator
    return center-half, center+half


def calculate_metrics(manifest, selection, response_sets):
    if manifest.sample_id.duplicated().any() or selection.sample_id.duplicated().any() or set(manifest.sample_id) != set(selection.sample_id):
        raise ValueError('Manifest/selection IDs do not match')
    n = len(manifest)
    if not n or len(response_sets) != 2:
        raise ValueError('Two reviewers and nonempty manifest required')
    truth = manifest.set_index('sample_id').class_name.to_dict()
    statuses = selection.set_index('sample_id').selection_status.to_dict()
    summaries, per_class, matrices, predictions = [], [], [], []
    for reviewer, responses in zip(REVIEWERS, response_sets):
        if len(responses) != n or {r['sample_id'] for r in responses} != set(truth):
            raise ValueError('Response set differs from pilot')
        decisions = {r['sample_id']: r['decision'] for r in responses}
        reasons_by_id = {r['sample_id']: r['reasons'] for r in responses}
        predictions.append(decisions)
        matrix = [[0]*len(CATEGORIES) for _ in CLASSES]
        for sid, actual in truth.items():
            matrix[CLASSES.index(actual)][CATEGORIES.index(decisions[sid])] += 1
        matrices.append(matrix)
        correct = sum(decisions[sid] == actual for sid, actual in truth.items())
        identifiable = sum(d in CLASSES for d in decisions.values())
        missing = sum(s == 'missing' for s in statuses.values())
        counts = Counter(decisions.values())
        summary = dict(reviewer_id=reviewer, n=n, correct_n=correct, accuracy_all=correct/n,
            identifiable_n=identifiable, identifiable_coverage=identifiable/n,
            accuracy_identifiable=correct/identifiable if identifiable else None,
            unidentifiable_n=n-identifiable, ambiguous_n=counts['ambiguous'], invalid_n=counts['invalid'],
            ambiguous_proportion=counts['ambiguous']/n, invalid_proportion=counts['invalid']/n,
            technical_coverage=sum(s == 'eligible' for s in statuses.values())/n,
            inspectable_coverage=(n-missing)/n, technical_failure_n=sum(s != 'eligible' for s in statuses.values()),
            selection_missing_n=missing,
            semantic_failure_readable_n=sum(statuses[sid] != 'missing' and d in ['ambiguous', 'invalid'] for sid, d in decisions.items()),
            wrong_class_readable_n=sum(statuses[sid] != 'missing' and d in CLASSES and d != truth[sid] for sid, d in decisions.items()))
        summary['unidentifiable_eligible_n'] = sum(statuses[sid] == 'eligible' and d not in CLASSES for sid, d in decisions.items())
        summary['wrong_class_eligible_n'] = sum(statuses[sid] == 'eligible' and d in CLASSES and d != truth[sid] for sid, d in decisions.items())
        for reason in REASONS:
            count = sum(reason in r['reasons'] for r in responses)
            summary[reason + '_n'] = count
            summary[reason + '_proportion'] = count/n
        summary['accuracy_all_ci95_low'], summary['accuracy_all_ci95_high'] = wilson(correct, n)
        summary['accuracy_identifiable_ci95_low'], summary['accuracy_identifiable_ci95_high'] = wilson(correct, identifiable)
        summaries.append(summary)
        for cls in CLASSES:
            ids = [sid for sid in truth if truth[sid] == cls]
            nc = len(ids)
            success = sum(decisions[sid] == cls for sid in ids)
            ident = sum(decisions[sid] in CLASSES for sid in ids)
            low, high = wilson(success, nc)
            per_class.append(dict(reviewer_id=reviewer, class_name=cls, n=nc, correct_n=success,
                accuracy_all=success/nc if nc else None, identifiable_n=ident,
                identifiable_coverage=ident/nc if nc else None,
                accuracy_identifiable=success/ident if ident else None,
                unidentifiable_n=nc-ident, technical_eligible_n=sum(statuses[sid] == 'eligible' for sid in ids),
                technical_flagged_n=sum(statuses[sid] == 'flagged' for sid in ids),
                selection_missing_n=sum(statuses[sid] == 'missing' for sid in ids),
                ambiguous_n=sum(decisions[sid] == 'ambiguous' for sid in ids),
                invalid_n=sum(decisions[sid] == 'invalid' for sid in ids),
                unidentifiable_eligible_n=sum(statuses[sid] == 'eligible' and decisions[sid] not in CLASSES for sid in ids),
                wrong_class_eligible_n=sum(statuses[sid] == 'eligible' and decisions[sid] in CLASSES and decisions[sid] != cls for sid in ids),
                temporal_insufficient_n=sum('temporal_insufficient' in reasons_by_id[sid] for sid in ids),
                ci95_low=low, ci95_high=high))
    a, c = predictions
    agreement = sum(a[sid] == c[sid] for sid in truth)/n
    ca, cc = Counter(a.values()), Counter(c.values())
    expected = sum(ca[k]*cc[k] for k in CATEGORIES)/(n*n)
    pairs = []
    for i, first in enumerate(CLASSES):
        for j in range(i+1, len(CLASSES)):
            second = CLASSES[j]
            count = sum(m[i][j] + m[j][i] for m in matrices)
            if count:
                pairs.append(dict(class_a=first, class_b=second, pooled_errors=count,
                                  frequent=count >= 2))
    agreement_result = dict(n=n, categories=CATEGORIES,
        category_definition='Eight exact class choices plus ambiguous and invalid as two distinct nominal categories. All samples retained, including missing images.',
        agreement=agreement, expected_agreement=expected,
        cohens_kappa=(agreement-expected)/(1-expected) if expected < 1 else None,
        undefined_kappa_reason='Identical constant ratings; expected agreement=1' if expected == 1 else None,
        frequent_confusion_definition='At least 2 pooled directional errors across both reviewers; descriptive, not a significance threshold.',
        confused_pairs=sorted(pairs, key=lambda x: -x['pooled_errors']),
        frequent_confused_pair_n=sum(x['frequent'] for x in pairs),
        frequent_confused_class_n=len({x[k] for x in pairs if x['frequent'] for k in ['class_a', 'class_b']}),
        unidentifiable_either_n=sum(a[sid] not in CLASSES or c[sid] not in CLASSES for sid in truth),
        unidentifiable_both_n=sum(a[sid] not in CLASSES and c[sid] not in CLASSES for sid in truth))
    return summaries, per_class, agreement_result, matrices


def analyze(work):
    work = Path(work)
    manifest = read_csv(work / 'pilot_manifest_private.csv')
    selection = read_csv(work / 'pilot_selection_report.csv')
    out = work / 'analysis'
    out.mkdir(exist_ok=True)
    files = [work / 'responses' / f'{r}.json' for r in REVIEWERS]
    if not all(path.exists() for path in files):
        eligible = int(selection.selection_status.eq('eligible').sum())
        flagged = int(selection.selection_status.eq('flagged').sum())
        missing_n = int(selection.selection_status.eq('missing').sum())
        known_signers = int(manifest.signer.astype(str).str.strip().ne('').sum()) if 'signer' in manifest else 0
        pd.DataFrame([dict(status='PENDING', samples=len(manifest),
            eligible=int(selection.selection_status.eq('eligible').sum()),
            flagged=int(selection.selection_status.eq('flagged').sum()),
            missing=int(selection.selection_status.eq('missing').sum()))]).to_csv(out / 'pilot_summary.csv', index=False)
        manifest.groupby('class_name').size().rename('n').reindex(CLASSES, fill_value=0).to_csv(out / 'per_class_results.csv')
        b.save_json(out / 'inter_reviewer_agreement.json', dict(status='PENDING', categories=CATEGORIES))
        (out / 'pilot_report.md').write_text('# Pilot single-frame BISINDO — PENDING\n\n'
            f'{len(manifest)} sampel tetap. Respons lengkap dua reviewer belum diimpor. '
            'Accuracy, confusion matrix, agreement, dan keputusan arsitektur belum tersedia.\n\n'
            f'Teknis: eligible={eligible}, flagged={flagged}, missing={missing_n}. '
            f'Signer diketahui={known_signers}/{len(manifest)}; video berbeda tidak menjamin signer berbeda.\n\n'
            'Semua sampel tetap dalam denominator; flagged tidak membuktikan gesture invalid. '
            'Coverage teknis bukan accuracy gesture. Reviewer harus bekerja independen. '
            'Pilot eksploratif pada training pool; bukan benchmark test eksternal.\n\n'
            'Tidak ada bukti empiris untuk keputusan kelayakan, publikasi, atau pergantian ke LSTM/TCN.\n', encoding='utf-8')
        return
    protocol = validate_frozen(work)
    responses = [validate_responses(json.loads(path.read_text()), protocol['study_id'], reviewer, set(manifest.sample_id))
                 for path, reviewer in zip(files, REVIEWERS)]
    missing = set(selection.loc[selection.selection_status.eq('missing'), 'sample_id'])
    if any(r['sample_id'] in missing and r['decision'] != 'invalid' for group in responses for r in group):
        raise ValueError('Missing image must be invalid')
    summary, per_class, agreement, matrices = calculate_metrics(manifest, selection, responses)
    pd.DataFrame(summary).to_csv(out / 'pilot_summary.csv', index=False)
    pd.DataFrame(per_class).to_csv(out / 'per_class_results.csv', index=False)
    b.save_json(out / 'inter_reviewer_agreement.json', agreement)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(19, 7), constrained_layout=True)
    for ax, reviewer, matrix in zip(axes, REVIEWERS, matrices):
        ax.imshow(matrix, cmap='Blues', vmin=0)
        ax.set(xticks=range(len(CATEGORIES)), xticklabels=CATEGORIES, yticks=range(len(CLASSES)),
               yticklabels=CLASSES, xlabel='Keputusan reviewer', ylabel='Kelas sumber', title=reviewer)
        ax.tick_params(axis='x', rotation=60)
        for y, row in enumerate(matrix):
            for x, value in enumerate(row):
                ax.text(x, y, str(value), ha='center', va='center')
        pd.DataFrame(matrix, index=CLASSES, columns=CATEGORIES).to_csv(out / f'{reviewer}_confusion.csv')
    fig.savefig(out / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    lines = ['# Pilot single-frame BISINDO — exploratory', '',
        f'N={len(manifest)}; 20 per kelas; internal train saja; seed 42. Frame otomatis dibekukan sebelum review.',
        'Technical coverage = frame lolos semua heuristik / semua sampel. Inspectable coverage mencakup frame flagged.',
        'Ambiguous dan invalid selalu gagal pada accuracy_all. Subset identifiable berarti reviewer memilih salah satu kelas, bukan berarti jawabannya benar.', '',
        '| Reviewer | Benar / semua | Accuracy semua (95% Wilson) | Identifiable n / semua | Accuracy subset |',
        '|---|---|---|---|---|']
    for row in summary:
        subset = f"{row['accuracy_identifiable']:.3f}" if row['accuracy_identifiable'] is not None else 'undefined'
        lines.append(f"| {row['reviewer_id']} | {row['correct_n']} / {row['n']} | {row['accuracy_all']:.3f} ({row['accuracy_all_ci95_low']:.3f}–{row['accuracy_all_ci95_high']:.3f}) | {row['identifiable_n']} / {row['n']} | {subset} |")
    lines += ['', f"Agreement={agreement['agreement']:.3f}; Cohen kappa={agreement['cohens_kappa']}. Kategori: {', '.join(CATEGORIES)}.",
        f"Technical coverage={summary[0]['technical_coverage']:.3f}; inspectable coverage={summary[0]['inspectable_coverage']:.3f}.",
        f"Tidak teridentifikasi oleh salah satu reviewer: {agreement['unidentifiable_either_n']}; oleh keduanya: {agreement['unidentifiable_both_n']}.",
        f"Pasangan sering tertukar: {agreement['frequent_confused_pair_n']}; kelas terlibat: {agreement['frequent_confused_class_n']}. Definisi: >=2 kesalahan pooled; bukan uji signifikansi.", '',
        '## Kelas sulit dan sumber kesalahan', '']
    for cls in CLASSES:
        rows = [r for r in per_class if r['class_name'] == cls]
        rates = ', '.join(f"{r['reviewer_id']}={r['accuracy_all']:.2f}" for r in rows)
        lines.append(f'- {cls}: accuracy semua {rates}; tidak teridentifikasi pooled={sum(r["unidentifiable_n"] for r in rows)}; '
            f'teknis eligible/flagged/missing={rows[0]["technical_eligible_n"]}/{rows[0]["technical_flagged_n"]}/{rows[0]["selection_missing_n"]}; '
            f'tidak teridentifikasi pada eligible pooled={sum(r["unidentifiable_eligible_n"] for r in rows)}; '
            f'alasan temporal pooled={sum(r["temporal_insufficient_n"] for r in rows)}.')
    for row in summary:
        lines.append(f"- {row['reviewer_id']}: missing={row['selection_missing_n']}, ambiguous/invalid pada frame terbaca={row['semantic_failure_readable_n']}, salah kelas pada frame terbaca={row['wrong_class_readable_n']}, alasan temporal={row['temporal_insufficient_n']}, preparation={row['preparation_n']}, transition={row['transition_n']}.")
    lines += ['', '## Implikasi arsitektur',
        'Nilai semua kelas dan agreement bersama coverage; accuracy subset saja mengandung selection bias. '
        'Tidak ada threshold kelayakan yang dipraregistrasikan, sehingga laporan ini tidak memberi keputusan lulus/gagal otomatis.',
        'Kegagalan seleksi teknis dipisahkan dari jawaban semantik pada gambar terbaca. Jawaban temporal_insufficient adalah pendapat reviewer, '
        'bukan bukti kausal bahwa temporal model diperlukan. Pilot ini tidak membandingkan video dengan frame secara terkontrol.',
        'Jika kelas tertentu tetap sulit pada frame teknis baik dan kedua reviewer konsisten menyebut informasi temporal kurang, '
        'pertimbangkan studi temporal LSTM/TCN berikutnya. Jangan mengganti arsitektur atau memilih ulang frame pilot dari respons ini.',
        'Kelayakan YOLO26 single-frame harus diputuskan setelah meninjau per-class accuracy, confusion, alasan, dan konsistensi; '
        'hasil manusia tidak mengukur performa YOLO26. Tidak ada klaim kelayakan publikasi Model A.', '',
        '## Keterbatasan',
        'Wilson 95% mengasumsikan observasi independen. Korelasi signer/video dapat membuat interval terlalu sempit; '
        'tanpa metadata signer tidak dapat dihitung interval cluster yang sah. N=20 per kelas kecil. '
        'Pilot eksploratif pada training pool, bukan benchmark Kaggle external test. '
        'Tidak ada training, full-dataset review, atau tuning test set.']
    lines.extend('- ' + s for s in protocol['limitations'])
    (out / 'pilot_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'import', 'analyze'])
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--source', type=Path, default=None)
    parser.add_argument('--work', type=Path, default=None)
    parser.add_argument('--reviewer', choices=REVIEWERS)
    parser.add_argument('--file', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    work = args.work or root / 'experiment_yolo26_pilot'
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    if args.command == 'prepare':
        prepare(root, args.source or root / 'experiment_yolo26_v3', work)
    elif args.command == 'import':
        if not args.reviewer or not args.file:
            parser.error('import requires --reviewer and --file')
        import_review(work, args.file, args.reviewer)
    else:
        analyze(work)


if __name__ == '__main__':
    main()
