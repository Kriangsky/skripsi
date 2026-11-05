"""Explicit development / freeze / external-score commands. Import never scores or trains."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

import interference_metrics as m
import confirmatory_statistics as stats

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / '07_dokumen/protokol'
DEFAULT_PLAN = PROTOCOL / 'analysis_plan.json'
INITIAL_SHA = '0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def source_hashes():
    files = [*Path(__file__).parent.glob('*.py'), *ROOT.joinpath('05_kode/tests').glob('test_*.py')]
    return {str(p.resolve()):sha(p) for p in files}


def validation_record(out):
    """Run synthetic suites only; no model inference, training or external data."""
    if Path(out).exists(): raise FileExistsError('Validation record exists')
    before = source_hashes()
    results = []
    for pattern in ['test_interference_metrics.py','test_confirmatory_statistics.py']:
        command = [sys.executable,'-m','unittest','discover','-s',str(ROOT/'05_kode/tests'),'-p',pattern,'-v']
        import os
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).parent))
        process = subprocess.run(command,capture_output=True,text=True,env=env,cwd=ROOT)
        results.append(dict(command=command,exit_code=process.returncode,output=process.stdout+process.stderr))
        if process.returncode: raise RuntimeError(results[-1]['output'])
    if before != source_hashes(): raise ValueError('Source/test drift during validation')
    write_new(out,dict(status='passed',external_scoring_performed=False,source_sha256=before,
                       timestamp_utc=datetime.now(timezone.utc).isoformat(),tests=results))
    return out


def resolve(root, value):
    if not isinstance(value, str) or not value.strip(): raise ValueError('Unresolved artifact path')
    return (root / value).resolve()


def enriched_predictions(work, seed):
    manifest = pd.read_csv(work/'manifest.csv', keep_default_na=False)
    predictions = pd.read_csv(work/'evaluation/predictions.csv', keep_default_na=False)
    if not manifest.source.eq('train').all() or not set(manifest.split).issubset({'train', 'val'}):
        raise ValueError('Development analysis must use internal data only')
    identity = ['sample_id', 'video', 'video_sha256', 'image_sha256']
    for key in ['sample_id', 'video', 'video_sha256']:
        if manifest[key].duplicated().any(): raise ValueError('Duplicate manifest identity')
    expected = manifest[manifest.split.eq('val')]
    if set(predictions.model) != {'A', 'B'}: raise ValueError('A and B predictions required')
    frames = {}
    for name, n in [('A', 6), ('B', 8)]:
        part = predictions[predictions.model.eq(name)]
        target = expected[expected.class_id.lt(n)]
        if part.sample_id.duplicated().any() or set(part.sample_id) != set(target.sample_id):
            raise ValueError('Missing or duplicate development predictions')
        frame = target[identity+['class_id', 'class_name']].merge(part, on='sample_id', validate='one_to_one')
        if not frame.class_id.eq(frame.true_class_id).all() or not frame.class_name.eq(frame.true_class).all():
            raise ValueError('Prediction GT does not match manifest')
        for row in frame.itertuples():
            image = work/'datasets'/name/'val'/f'{row.class_id:02d}_{m.ALL_CLASSES[row.class_id]}'/f'{row.sample_id}.png'
            if sha(image) != row.image_sha256: raise ValueError('Development image bytes changed')
        frames[name] = frame.assign(seed=seed).sort_values('sample_id').reset_index(drop=True)
    return frames


def export_reports(frames, plan, out, tier):
    seeds = plan['training_seeds']
    if set(frames) != set(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('All frozen training seeds required exactly once')
    if tier not in ['development', 'confirmatory']: raise ValueError('Unknown evidence tier')
    aligned, reference, full_reference = [], None, None
    primary, interference, intrusion, decomposition, full8 = [], [], [], [], []
    full_predictions = []
    for seed in seeds:
        a, b = frames[seed]['A'], frames[seed]['B']
        y, pa, pb = m.align_predictions(a, b, range(6))
        common = a.sort_values('sample_id')[['sample_id','video','video_sha256','image_sha256','true_class_id']].reset_index(drop=True)
        if reference is not None and not common.equals(reference): raise ValueError('Videos differ across training seeds')
        reference = common
        aligned.append((y, pa.argmax(1), pb.argmax(1)))
        result = m.interference(y, pa, pb)
        primary.append(dict(seed=seed, n=len(y), macro_f1_A=result['A']['macro_f1'], macro_f1_B=result['B']['macro_f1'],
                            delta=result['B']['macro_f1']-result['A']['macro_f1'], evidence_tier=tier))
        interference.extend(dict(seed=seed, **r) for r in result['classes'])
        intrusion.extend(dict(seed=seed, **r) for r in result['intrusion'])
        decomposition.append(dict(seed=seed, **result['decomposition']))
        b = b.sort_values('sample_id').reset_index(drop=True)
        full_id = b[['sample_id','video','video_sha256','image_sha256','true_class_id']]
        if full_reference is not None and not full_id.equals(full_reference): raise ValueError('Full8 videos differ across seeds')
        full_reference = full_id
        scores = m.probabilities([json.loads(x) for x in b.probabilities], 8, len(b))
        full_y, full_p = b.true_class_id.to_numpy(), scores.argmax(1)
        full_predictions.append(full_p)
        r8 = m.metrics(full_y, full_p, range(8))
        full8.extend(dict(seed=seed, accuracy=r8['accuracy'], macro_f1=r8['macro_f1'], **r) for r in r8['classes'])
    r = stats.bootstrap(aligned[0][0], np.array([x[1] for x in aligned]), np.array([x[2] for x in aligned]),
                        repetitions=plan['bootstrap_replicates'], seed=plan['bootstrap_seed'])
    bounds = stats.interval(r['hierarchical'])
    intervals = []
    for method in ['hierarchical', 'video_only', 'seed_only']:
        intervals.append(dict(endpoint='delta_macro_f1', method=method, estimate=r['estimate'],
                              n_videos=r['n_videos'], n_seeds=r['n_seeds'], **stats.interval(r[method])))
    simultaneous = stats.simultaneous_recall_intervals(r['recall_delta'])
    for k in range(6):
        for key in ['recall_A', 'recall_B', 'recall_delta']:
            value_key = 'delta_recall' if key == 'recall_delta' else key
            estimate = float(np.mean([x[value_key] for x in interference if x['class_id'] == k]))
            intervals.append(dict(endpoint=f'{key}_{m.BASE[k]}', method='hierarchical', estimate=estimate,
                                  n_videos=int((aligned[0][0] == k).sum()), n_seeds=len(seeds),
                                  **stats.interval(r[key][:, k])))
        row = dict(seed='mean', class_id=k, class_name=m.BASE[k], n=int((aligned[0][0] == k).sum()))
        for field in ['recall_A', 'recall_B', 'delta_recall']:
            row[field] = float(np.mean([x[field] for x in interference if x['class_id'] == k and x['seed'] != 'mean']))
        for key, field in [('recall_A','recall_A'), ('recall_B','recall_B'), ('recall_delta','delta_recall')]:
            ci = stats.interval(r[key][:, k]); row[field+'_ci95_low'] = ci['ci95_low']; row[field+'_ci95_high'] = ci['ci95_high']
        row.update(delta_simultaneous95_low=simultaneous[k][0], delta_simultaneous95_high=simultaneous[k][1])
        interference.append(row)
    for name, rows in [('intrusion', intrusion), ('decomposition', decomposition)]:
        if name == 'intrusion':
            for cls in ['overall', *m.BASE]:
                subset = [x for x in rows if x['class_name'] == cls]
                rows.append(dict(seed='mean', class_name=cls, n=subset[0]['n'],
                                 **{key:float(np.mean([x[key] for x in subset])) for key in ['overall','halo','tidur']}))
        else:
            rows.append(dict(seed='mean', **{key:float(np.mean([x[key] for x in rows])) for key in rows[0] if key!='seed'}))
    full = np.array(full_predictions)
    h2_draws = stats.bootstrap(full_y, full, full, repetitions=plan['bootstrap_replicates'],
                              seed=plan['bootstrap_seed'], labels=tuple(range(8)))
    h2_bounds = {m.ALL_CLASSES[k]:stats.interval(h2_draws['recall_B'][:, k])['lower_one_sided95'] for k in [6, 7]}
    h2 = 'not tested: development evidence' if tier == 'development' else (
        'H2 demonstrated' if all(x > plan['h2_threshold'] for x in h2_bounds.values()) else 'H2 not demonstrated')
    for k in [6, 7]:
        intervals.append(dict(endpoint=f'H2_recall_{m.ALL_CLASSES[k]}', method='hierarchical',
                              estimate=float(np.mean(full[:,full_y == k] == k)), n_videos=int((full_y == k).sum()),
                              n_seeds=len(seeds), **stats.interval(h2_draws['recall_B'][:, k])))
    decision = ('not tested: development evidence' if tier == 'development' else
                'not tested: approved estimation-only endpoint' if plan['ni_status']=='ESTIMATION_ONLY_APPROVED' else
                stats.non_inferiority(bounds['lower_one_sided95'], plan['ni_margin'], plan['ni_justification']))
    primary.append(dict(seed='mean', n=r['n_videos'], macro_f1_A=float(np.mean([x['macro_f1_A'] for x in primary])),
                        macro_f1_B=float(np.mean([x['macro_f1_B'] for x in primary])), delta=r['estimate'],
                        evidence_tier=tier, **bounds, ni_decision=decision))
    for k in range(8):
        subset = [x for x in full8 if x['class_id']==k]
        full8.append(dict(seed='mean',class_id=k,class_name=m.ALL_CLASSES[k],n=subset[0]['n'],
                          **{key:float(np.mean([x[key] for x in subset])) for key in ['accuracy','macro_f1','precision','recall','f1']}))
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    for filename, rows in [('common6_primary_results',primary), ('class_interference',interference),
                           ('intrusion_rates',intrusion), ('accuracy_decomposition',decomposition),
                           ('full8_results',full8), ('bootstrap_intervals',intervals)]:
        pd.DataFrame(rows).to_csv(out/(filename+'.csv'), index=False)
    for seed, b in zip(seeds, full_predictions):
        matrix = m.metrics(full_y,b,range(8))['confusion_matrix']
        pd.DataFrame(matrix,index=m.ALL_CLASSES,columns=m.ALL_CLASSES).to_csv(out/f'confusion_B_full8_seed{seed}.csv')
    sensitivity = stats.design_effect_sensitivity(r['estimate'],r['hierarchical'])
    for row in sensitivity: row['assumed_effective_video_n'] = r['n_videos']/row['design_effect']
    write_new(out/'analysis_details.json', dict(evidence_tier=tier, training_seeds=seeds, n_videos=r['n_videos'],
        n_full8_videos=len(full_y), bootstrap_seed=r['bootstrap_seed'], repetitions=r['repetitions'],
        seed_delta_sd=r['seed_delta_sd'], ni_status=plan['ni_status'], ni_decision=decision, h2=h2,
        h2_one_sided_lower95=h2_bounds, design_effect_sensitivity=sensitivity))
    report = (f'# {tier} evaluation\n\nMean paired common6 delta macro-F1: {r["estimate"]:.8f}. '
              f'Hierarchical percentile CI95% [{bounds["ci95_low"]:.8f}, {bounds["ci95_high"]:.8f}]; '
              f'one-sided lower95% {bounds["lower_one_sided95"]:.8f}.\n\n'
              f'N={r["n_videos"]} common6 videos; {len(full_y)} full8 videos; all seeds {seeds}. '
              f'{r["repetitions"]} bootstrap replicates, RNG {r["bootstrap_seed"]}. NI: {decision}. '
              f'Margin status: {plan["ni_status"]}. H2: {h2}; lower bounds {h2_bounds}.\n\n'
              'Sampling: resample videos within GT strata, same draw for A/B and all seeds; independently '
              'resample paired training seeds. Recompute F1 before averaging seed differences. '
              'Video-only intervals condition on observed seeds; seed-only intervals condition on observed videos. '
              'Five seeds give limited training-variability information. Seeds are not independent test datasets.\n\n'
              'Class-interference mean rows contain marginal recall and difference CIs, plus Bonferroni '
              'simultaneous nominal 95% intervals over six differences (approximate bootstrap coverage). '
              'No class is declared most affected from rank alone. Full8 is a separate task. '
              'Restricted argmax and accuracy decomposition are diagnostic, without causal attribution.\n\n'
              f'Design-effect normal-SE stress tests: {json.dumps(sensitivity)}. '
              'Multiplying SE by sqrt(DE) is an assumption, not signer-cluster resampling. '
              'Unknown signer metadata prevents signer-population inference; one external signer would not support it. '
              'No measured efficiency endpoint is available from this analysis.\n')
    (out/'external_evaluation_report.md').write_text(report, encoding='utf-8')
    return out


def development(plan_path, out):
    plan = read(plan_path)
    frames = {seed:enriched_predictions(resolve(ROOT, plan['development_runs'][str(seed)]), seed)
              for seed in plan['training_seeds']}
    result = export_reports(frames, plan, out, 'development')
    inputs = {str(Path(plan_path).resolve()):sha(plan_path)}
    for seed in plan['training_seeds']:
        work = resolve(ROOT,plan['development_runs'][str(seed)])
        for relative in ['manifest.csv','evaluation/predictions.csv','research_config.json','runs/trained_models.json']:
            inputs[str(work/relative)] = sha(work/relative)
    write_new(Path(out)/'input_provenance.json',dict(inputs_sha256=inputs,
        source_sha256={str(p.resolve()):sha(p) for p in Path(__file__).parent.glob('*.py')},
        timestamp_utc=datetime.now(timezone.utc).isoformat(),git=git_state(ROOT)))
    return result


def validate_plan(plan, root):
    """Validate final decisions and their evidence BEFORE touching external image/model files."""
    if plan.get('status') != 'final' or plan.get('task') != 'classify' or plan.get('model_variant') != 'yolo26n-cls':
        raise ValueError('Final Classification configuration unresolved')
    if plan.get('class_mapping') != list(m.ALL_CLASSES) or plan.get('training_seeds') != [42,43,44,45,46]:
        raise ValueError('Class mapping / prespecified seed list changed')
    if plan.get('bootstrap_replicates') != 10000 or plan.get('bootstrap_seed') != stats.BOOTSTRAP_SEED:
        raise ValueError('Statistical configuration changed')
    if plan.get('h2_threshold') != .80 or plan.get('h2_status') != 'preserved_compatible':
        raise ValueError('H2 unresolved or changed')
    if plan.get('ni_status') == 'NI_MARGIN_JUSTIFIED':
        stats.non_inferiority(0.,plan.get('ni_margin'),plan.get('ni_justification'))
        if plan.get('ni_margin') is None: raise ValueError('NI margin unresolved')
    elif not (plan.get('ni_status') == 'ESTIMATION_ONLY_APPROVED' and plan.get('ni_margin') is None):
        raise ValueError('NI_MARGIN_UNRESOLVED')
    required = ['addendum', 'experiment_config', 'initial_checkpoint', 'dataset_manifest', 'split_mapping',
                'external_manifest', 'internal_manifest', 'environment', 'scientific_decision',
                'provenance_review', 'exposure_review', 'validation_record']
    paths = {key:resolve(root,plan.get('artifacts',{}).get(key)) for key in required}
    if any(not p.is_file() for p in paths.values()): raise ValueError('Missing required freeze artifact')
    scientific = read(paths['scientific_decision'])
    if (scientific.get('status') != 'approved' or not scientific.get('reviewer') or not scientific.get('timestamp')
            or scientific.get('plan_sha256') != sha(resolve(root, plan['plan_path']))
            or scientific.get('addendum_sha256') != sha(paths['addendum'])
            or scientific.get('unresolved_conflicts') != []):
        raise ValueError('Scientific decision must approve exact final plan bytes')
    provenance = read(paths['provenance_review'])
    if (provenance.get('status') != 'verified' or provenance.get('different_from_internal_signers') is not True
            or not provenance.get('evidence') or not provenance.get('reviewer')):
        raise ValueError('External signer provenance unresolved')
    exposure = read(paths['exposure_review'])
    if (exposure.get('status') != 'reviewed' or exposure.get('used_for_tuning') is not False
            or exposure.get('prior_scoring') is not False or not exposure.get('history') or not exposure.get('reviewer')):
        raise ValueError('External tuning/scoring history disqualifies untouched confirmation or is unresolved')
    validation = read(paths['validation_record'])
    if validation.get('status') != 'passed' or validation.get('external_scoring_performed') is not False:
        raise ValueError('Internal/synthetic validation missing')
    sources = source_hashes()
    if validation.get('source_sha256') != sources: raise ValueError('Validation record does not cover current source/tests')
    if (len(validation.get('tests',[]))!=2 or any(x.get('exit_code')!=0 for x in validation['tests'])):
        raise ValueError('Missing successful synthetic test executions')
    config = read(paths['experiment_config'])
    if (config.get('status') != 'final' or config.get('task') != 'classify' or config.get('checkpoint_rule') != 'last.pt'
            or type(config.get('imgsz')) is not int or config['imgsz'] < 1
            or type(config.get('epochs')) is not int or config['epochs'] < 1
            or config.get('preprocessing') != 'frozen_letterbox_RGB_255_gray114_no_crop_no_flip'):
        raise ValueError('Final experiment configuration unresolved')
    if sha(paths['initial_checkpoint']) != INITIAL_SHA: raise ValueError('Initial checkpoint changed')
    checkpoints = plan.get('checkpoints', [])
    if len(checkpoints) != 10 or {(x.get('seed'),x.get('model')) for x in checkpoints} != {(s,n) for s in plan['training_seeds'] for n in ['A','B']}:
        raise ValueError('Ten final paired checkpoints required; no winner seed')
    for item in checkpoints:
        p = resolve(root,item.get('path'))
        if (item.get('tier') != 'final' or p.name != 'last.pt' or sha(p) != item.get('sha256')
                or item.get('initial_checkpoint_sha256') != INITIAL_SHA or item.get('independent_initialization') is not True
                or item.get('experiment_config_sha256') != sha(paths['experiment_config'])):
            raise ValueError('Final checkpoint metadata/hash mismatch')
        paths[f"checkpoint_{item['model']}_{item['seed']}"] = p
    return paths, sources


def git_state(root):
    def run(*args):
        p = subprocess.run(['git','-C',str(root),*args],capture_output=True,text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return dict(commit=run('rev-parse','HEAD'), working_tree=run('status','--porcelain'))


def checked_manifests(paths, root, config):
    """Technical file checks, without opening model predictions or scoring labels."""
    files, manifests = {}, {}
    from PIL import Image
    for kind in ['internal', 'external']:
        frame = pd.read_csv(paths[kind+'_manifest'],keep_default_na=False)
        required = {'sample_id','video','video_sha256','image_path','image_sha256','class_id','class_name'}
        if not required.issubset(frame.columns): raise ValueError('Incomplete final frame manifest')
        if frame.empty: raise ValueError('Empty frame manifest')
        for key in ['sample_id','video','video_sha256','image_path']:
            if frame[key].duplicated().any() or frame[key].astype(str).str.strip().eq('').any():
                raise ValueError('Duplicate or missing video/frame identity')
        pixel_hashes = []
        for row in frame.to_dict('records'):
            cid = row['class_id']
            if not isinstance(cid,int) or not 0 <= cid < 8 or row['class_name'] != m.ALL_CLASSES[cid]:
                raise ValueError('Manifest class mapping mismatch')
            for field, checksum in [('video','video_sha256'),('image_path','image_sha256')]:
                p = resolve(root,row[field]); actual = sha(p)
                if actual != row[checksum]: raise ValueError('Video/image file hash mismatch')
                files[str(p)] = actual
            with Image.open(resolve(root,row['image_path'])) as im:
                if im.size != (config['imgsz'],config['imgsz']) or im.mode != 'RGB':
                    raise ValueError('Frozen preprocessed RGB frame dimensions mismatch')
                pixel_hashes.append(hashlib.sha256(im.tobytes()).hexdigest())
        if len(set(pixel_hashes)) != len(pixel_hashes): raise ValueError('Duplicate decoded frame pixels')
        frame['verified_pixel_sha256'] = pixel_hashes
        manifests[kind] = frame
    internal, external = manifests['internal'], manifests['external']
    for key in ['sample_id','video_sha256','verified_pixel_sha256']:
        if set(internal[key]) & set(external[key]): raise ValueError('Internal/external duplicate leakage')
    if external.class_id.value_counts().to_dict() != {k:50 for k in range(8)}:
        raise ValueError('External full8/common6 counts differ from 400/300, 50/class')
    split = pd.read_csv(paths['split_mapping'],keep_default_na=False)
    if split.sample_id.duplicated().any(): raise ValueError('Duplicate split mapping IDs')
    for name, frame in manifests.items():
        joined = frame.merge(split[['sample_id','split','source','class_id','video_sha256']],on='sample_id',
                             suffixes=('','_split'),validate='one_to_one')
        if (len(joined)!=len(frame) or not joined.class_id.eq(joined.class_id_split).all()
                or not joined.video_sha256.eq(joined.video_sha256_split).all()):
            raise ValueError('Final manifest differs from frozen split')
        if name=='external' and not (joined.source.eq('test') & joined.split.eq('test')).all():
            raise ValueError('External source/split mismatch')
        if name=='internal' and not (joined.source.eq('train') & joined.split.isin(['train','val'])).all():
            raise ValueError('Internal source/split mismatch')
    return files


def freeze(plan_path, out, root=ROOT):
    plan_path, out = Path(plan_path).resolve(), Path(out).resolve()
    plan = read(plan_path)
    if resolve(root,plan['plan_path']) != plan_path: raise ValueError('Plan path mismatch')
    paths, sources = validate_plan(plan,root)
    files = checked_manifests(paths,root,read(paths['experiment_config']))
    files.update({str(p):sha(p) for p in paths.values()}); files.update(sources)
    files[str(plan_path)] = sha(plan_path)
    record = dict(status='frozen', timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  plan_path=str(plan_path), plan_sha256=sha(plan_path), files_sha256=files,
                  training_seeds=plan['training_seeds'], class_mapping=plan['class_mapping'],
                  model_variant=plan['model_variant'], task=plan['task'], ni_margin=plan['ni_margin'],
                  ni_justification=plan['ni_justification'], git=git_state(root),
                  duplicate_check='actual video bytes and decoded frozen RGB frame pixels checked',
                  checkpoint_hashes={k:sha(p) for k,p in paths.items() if k.startswith('checkpoint_')})
    write_new(out,record)
    return record


def verify_freeze(freeze_path, approval_path):
    record, approval = read(freeze_path), read(approval_path)
    if record.get('status') != 'frozen' or not record.get('files_sha256'):
        raise ValueError('Invalid final freeze')
    if (approval.get('action') != 'final_external_scoring' or approval.get('approved') is not True
            or not approval.get('reviewer') or not approval.get('timestamp')
            or approval.get('freeze_sha256') != sha(freeze_path)):
        raise PermissionError('Explicit reviewer approval must bind exact final freeze hash')
    for path, checksum in record['files_sha256'].items():
        if sha(path) != checksum: raise ValueError(f'Frozen artifact changed: {path}')
    if sha(record['plan_path']) != record['plan_sha256']: raise ValueError('Plan drift')
    return record


def score(freeze_path, approval_path, out):
    freeze_path, out = Path(freeze_path).resolve(), Path(out).resolve()
    record = verify_freeze(freeze_path,approval_path)
    plan = read(record['plan_path'])
    paths, sources = validate_plan(plan, ROOT)
    current_files = checked_manifests(paths,ROOT,read(paths['experiment_config']))
    expected = {**current_files, **{str(p):sha(p) for p in paths.values()}, **sources, record['plan_path']:sha(record['plan_path'])}
    if expected != record['files_sha256']: raise ValueError('Freeze coverage incomplete')
    if out.exists(): raise FileExistsError('Scoring output exists; no overwrite')
    # Claim the whole batch BEFORE inference. Failed batches require a documented deviation, not a silent retry.
    receipt = freeze_path.with_name(freeze_path.name+'.scoring_started.json')
    write_new(receipt,dict(freeze_sha256=sha(freeze_path),approval_sha256=sha(approval_path),
                           timestamp_utc=datetime.now(timezone.utc).isoformat(),out=str(out)))
    from PIL import Image
    import torch
    from ultralytics import YOLO
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; no CPU fallback; record deviation')
    manifest = pd.read_csv(paths['external_manifest'],keep_default_na=False).sort_values('sample_id')
    frames = {seed:{} for seed in plan['training_seeds']}
    for item in plan['checkpoints']:
        name, seed = item['model'], item['seed']; n = 6 if name=='A' else 8
        model = YOLO(str(resolve(ROOT,item['path'])),task='classify')
        if model.task != 'classify' or [model.names[k] for k in range(n)] != [f'{k:02d}_{c}' for k,c in enumerate(m.ALL_CLASSES[:n])]:
            raise ValueError('Checkpoint task/class mapping mismatch; record deviation')
        model.model.to('cuda').float().eval()
        rows = []
        for row in manifest[manifest.class_id.lt(n)].to_dict('records'):
            with Image.open(resolve(ROOT,row['image_path'])) as image:
                pixels = np.asarray(image).copy()
            tensor = torch.from_numpy(pixels.transpose(2,0,1).copy()).float().unsqueeze(0).to('cuda')/255.
            with torch.inference_mode(): output = model.model(tensor)
            values = (output[0] if isinstance(output,tuple) else output).detach().cpu().numpy()
            scores = m.probabilities(values,n,1)[0]
            rows.append(dict(seed=seed,model=name,sample_id=row['sample_id'],video=row['video'],
                             video_sha256=row['video_sha256'],image_sha256=row['image_sha256'],
                             true_class_id=row['class_id'],true_class=row['class_name'],
                             probabilities=json.dumps(scores.tolist()),predicted_class_id=int(scores.argmax())))
        frames[seed][name] = pd.DataFrame(rows)
        del model; torch.cuda.empty_cache()
    result = export_reports(frames,plan,out,'confirmatory')
    pd.concat([f for pair in frames.values() for f in pair.values()],ignore_index=True).to_csv(out/'predictions.csv',index=False)
    write_new(out/'scoring_provenance.json',dict(freeze_sha256=sha(freeze_path),approval_sha256=sha(approval_path),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),git=git_state(ROOT),deviations=[]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['validate','development','freeze','score'])
    parser.add_argument('--plan',type=Path,default=DEFAULT_PLAN)
    parser.add_argument('--freeze',type=Path)
    parser.add_argument('--approval',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='validate': result=validation_record(args.out)
    elif args.command=='development': result=development(args.plan,args.out)
    elif args.command=='freeze': result=freeze(args.plan,args.out)
    else:
        if not args.freeze or not args.approval: parser.error('score requires --freeze and --approval')
        result=score(args.freeze,args.approval,args.out)
    print(str(result))


if __name__=='__main__': main()
