"""Exploratory single-frame classification. Weak video labels; technical gates only."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import logging
import os
from pathlib import Path
import shutil

import cv2
import numpy as np
import pandas as pd

import bisindo_pipeline as b
import pilot_study as frames
import training_pilot as detection

CLASSES, BASE = b.CLASSES, b.BASE
TARGETS = dict(train=60, val=20)
SIZE = 224
CONFIG = dict(detection.PILOT_CONFIG, task='classify', imgsz=SIZE, dropout=0.)
LOG = logging.getLogger(__name__)
AUDIT_DECISIONS = ['not_assessed', 'valid_for_target', 'preparation', 'resting', 'transition',
                   'ambiguous', 'temporal_ambiguity', 'wrong_label', 'clipped', 'blur']
WARNINGS = [
    'Exploratory pilot, not final thesis performance; one seed and 30 epochs.',
    'Labels come from frozen video metadata, not verified per-frame semantics; dataset is NOT fully verified.',
    'Technical flags are retained; MediaPipe hand detection is not gesture classification.',
    'Signer/session independence is unverified where metadata is absent; no signer-independent claim.',
    'Static-frame/label errors, background and signer cues may explain performance; no causal temporal claim.',
    'Classification and detection were not compared on a matched trained benchmark; architecture is not automatically replaced.',
]
dump = detection.dump
read_json = detection.read_json
csv = detection.csv


def folder_names(n):
    # ImageFolder sorts alphabetically; numeric prefixes preserve the original shared IDs.
    return [f'{i:02d}_{name}' for i, name in enumerate(CLASSES[:n])]


def sample_videos(inventory, development):
    for field in ['sample_id', 'video', 'video_sha256']:
        if inventory[field].duplicated().any() or inventory[field].astype(str).str.strip().eq('').any():
            raise ValueError(f'Duplicate/empty {field}')
    for row in inventory.itertuples():
        if row.class_name not in CLASSES or str(row.class_id) != str(CLASSES.index(row.class_name)):
            raise ValueError('Class mapping mismatch')
        if row.source not in ['train', 'test'] or row.split not in ['train', 'val', 'test'] or (row.source == 'test') != (row.split == 'test'):
            raise ValueError('Frozen source/split mismatch')
        parts = str(row.video).replace('\\', '/').split('/')
        if parts[0] != row.source or any(x in ['.', '..'] for x in parts):
            raise ValueError('Source/path mismatch')
    pool = inventory[inventory.source.eq('train')]
    result = []
    for cls in CLASSES:
        for split, count in TARGETS.items():
            group = pool[pool.class_name.eq(cls) & pool.split.eq(split)]
            if split == 'val':
                group = group[~group.sample_id.isin(development)]
            group = group.sort_values('sample_id').sample(frac=1, random_state=42)
            if len(group) < count:
                raise ValueError(f'{cls}/{split}: {len(group)} videos < target {count}; no oversampling')
            result.extend(group.head(count).to_dict('records'))
    return pd.DataFrame(result)


def letterbox(image):
    if image is None or image.ndim != 3 or min(image.shape[:2]) <= 0:
        raise ValueError('Invalid image')
    h, w = image.shape[:2]
    scale = min(SIZE/w, SIZE/h)
    width, height = max(1, round(w*scale)), max(1, round(h*scale))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    result = np.full((SIZE, SIZE, 3), 114, dtype=np.uint8)
    left, top = (SIZE-width)//2, (SIZE-height)//2
    result[top:top+height, left:left+width] = resized
    return result


def pixel_sha(image):
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def audit_sample(manifest):
    selected = []
    for cls in CLASSES:
        group = manifest[manifest.class_name.eq(cls)]
        primary = []
        for split, count in [('train', 10), ('val', 5)]:
            part = group[group.split.eq(split)].sort_values('sample_id').sample(frac=1, random_state=42).head(count)
            primary.extend(part.assign(audit_group='stratified_random').to_dict('records'))
        ids = {r['sample_id'] for r in primary}
        flagged = group[group.quality_status.eq('flagged') & ~group.sample_id.isin(ids)]
        extra = flagged.sort_values('sample_id').sample(frac=1, random_state=43).head(5)
        selected.extend(primary + extra.assign(audit_group='additional_flagged').to_dict('records'))
    return pd.DataFrame(selected)


def metrics(truth, predictions, output_classes, gt_classes):
    if len(truth) != len(predictions) or not truth:
        raise ValueError('Matched nonempty predictions required')
    matrix = np.zeros((output_classes, output_classes), dtype=int)
    for actual, predicted in zip(truth, predictions):
        if not 0 <= actual < gt_classes or not 0 <= predicted < output_classes:
            raise ValueError('Class ID outside evaluation mapping')
        matrix[actual, predicted] += 1
    per_class = []
    for cid in range(output_classes):
        tp = int(matrix[cid, cid])
        fp = int(matrix[:, cid].sum())-tp
        fn = int(matrix[cid].sum())-tp
        precision = tp/(tp+fp) if tp+fp else 0.
        recall = tp/(tp+fn) if tp+fn else 0.
        per_class.append(dict(class_id=cid, class_name=CLASSES[cid], n=int(matrix[cid].sum()),
            tp=tp, fp=fp, fn=fn, precision=precision, recall=recall,
            f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.))
    correct = int(np.trace(matrix))
    result = dict(n=len(truth), correct_n=correct, accuracy=correct/len(truth),
        macro_f1=float(np.mean([r['f1'] for r in per_class[:gt_classes]])),
        macro_f1_definition=f'Mean F1 over fixed ground-truth classes 0..{gt_classes-1}; extra predicted classes remain errors/FN.',
        extra_class_predictions_n=sum(p >= gt_classes for p in predictions))
    return result, per_class, matrix.tolist()


def audit_ui(work, audit):
    records = [dict(sample_id=r.sample_id, target=r.class_name, split=r.split,
        image='../selected_frames/'+r.sample_id+'.jpg', flags=json.loads(r.quality_flags), group=r.audit_group)
        for r in audit.itertuples()]
    payload = json.dumps(records).replace('<', '\\u003c')
    html = '''<!doctype html><html lang="id"><meta charset="utf-8"><title>Classification quality audit</title>
<style>body{font:17px system-ui;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:20px}article{border:1px solid #aaa;padding:10px}img{width:100%}select,input,button{font:inherit;padding:8px;margin:4px}header{position:sticky;top:0;background:white;padding:12px}</style>
<header><h1>Exploratory classification quality audit</h1><p>15 sampel acak per kelas + maksimal 5 tambahan flagged. Label berasal dari video, bukan bukti gesture valid. Jangan mengganti frame atau menghapus sampel sulit. Audit semantik belum lengkap boleh menjadi warning.</p><input id="reviewer" placeholder="Nama reviewer" aria-label="Reviewer"><button id="export">Ekspor audit</button><p id="message"></p></header><main id="grid"></main>
<script>const rows=__ROWS__,decisions=__DECISIONS__,state={};
const key='classification_audit___FINGERPRINT__';
try{Object.assign(state,JSON.parse(localStorage.getItem(key)||'{}'))}catch(e){}
for(const r of rows){const a=document.createElement('article'),h=document.createElement('h2'),p=document.createElement('p'),img=document.createElement('img'),s=document.createElement('select'),note=document.createElement('input');h.textContent=r.target+' / '+r.split+' / '+r.sample_id;p.textContent=r.group+' | '+r.flags.join(', ');img.src=r.image;img.loading='lazy';img.alt=r.sample_id;for(const d of decisions){const o=document.createElement('option');o.value=o.textContent=d;s.append(o)}s.value=state[r.sample_id]?.decision||'not_assessed';s.setAttribute('aria-label','Penilaian '+r.sample_id);note.placeholder='Catatan';note.value=state[r.sample_id]?.note||'';const save=()=>{state[r.sample_id]={sample_id:r.sample_id,decision:s.value,note:note.value,timestamp:new Date().toISOString()};try{localStorage.setItem(key,JSON.stringify(state))}catch(e){document.getElementById('message').textContent='Ekspor JSON untuk cadangan; storage tidak tersedia'}};s.onchange=note.oninput=save;a.append(h,p,img,s,note);document.getElementById('grid').append(a)}
document.getElementById('export').onclick=()=>{const reviewer=document.getElementById('reviewer').value.trim();if(!reviewer){document.getElementById('message').textContent='Isi reviewer';return}const result={schema:1,kind:'classification_quality_audit',fingerprint:'__FINGERPRINT__',reviewer,records:rows.map(r=>state[r.sample_id]||{sample_id:r.sample_id,decision:'not_assessed',note:'',timestamp:new Date().toISOString()})};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='classification_quality_audit.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
</script></html>'''.replace('__ROWS__', payload).replace('__DECISIONS__', json.dumps(AUDIT_DECISIONS)).replace('__FINGERPRINT__', b.sha(work/'manifest.csv'))
    (work/'audit/review.html').write_text(html, encoding='utf-8')
    sheets = []
    for cls in CLASSES:
        tiles = []
        for r in audit[audit.class_name.eq(cls)].itertuples():
            image = cv2.imread(str(work/'selected_frames'/f'{r.sample_id}.jpg'))
            h, w = image.shape[:2]; scale=min(320/w, 180/h)
            thumb=cv2.resize(image, (max(1,round(w*scale)),max(1,round(h*scale))))
            tile=np.full((240,320,3),240,dtype=np.uint8)
            tile[:thumb.shape[0],:thumb.shape[1]]=thumb
            cv2.putText(tile, f'{cls} / {r.split} / {r.quality_status}', (4,202), cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),1)
            cv2.putText(tile,r.sample_id,(4,220),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),1)
            tiles.append(tile)
        while len(tiles)%4:tiles.append(np.full_like(tiles[0],240))
        path=work/'audit'/f'{cls}.jpg'
        if not cv2.imwrite(str(path),np.vstack([np.hstack(tiles[i:i+4]) for i in range(0,len(tiles),4)])):
            raise ValueError('Cannot save audit sheet')
        sheets.append(path)
    return sheets


def prepare(root, source, work):
    root, source, work = Path(root).resolve(), Path(source).resolve(), Path(work).resolve()
    protected = [root/x for x in ['train','test','experiment_yolo26_v2','experiment_yolo26_v3','experiment_yolo26_pilot','experiment_yolo26_training_pilot']]
    if work == root or source.is_relative_to(work) or any(work.is_relative_to(p) for p in protected):
        raise ValueError('Classification output must be separate from all existing experiments')
    for folder in ['reports','selected_frames','datasets','audit','runs','evaluation/error_examples']:
        (work/folder).mkdir(parents=True,exist_ok=True)
    if (work/'release.json').exists():
        validate_release(work)
        return read_json(work/'reports/readiness.json')
    inventory=csv(source/'split_lock.csv')
    development=detection.human_development_ids(root,inventory)
    selected=sample_videos(inventory,development)
    snapshot={name:b.sha(source/name) for name in ['split_lock.csv','candidates.csv','candidate_config.json']}
    snapshot['human_pilot_manifest']=b.sha(root/'experiment_yolo26_pilot/pilot_manifest_private.csv')
    dump(work/'reports/source_snapshot.json',snapshot)
    config=read_json(source/'candidate_config.json')
    if config['split_lock_sha256'] != snapshot['split_lock.csv'] or config['quality_heuristics'] != frames.f.QUALITY:
        raise ValueError('Frozen split/selection recipe mismatch')
    catalog=csv(source/'candidates.csv')
    if catalog.candidate_id.duplicated().any():raise ValueError('Duplicate candidate identity')
    groups={key:group for key,group in catalog.groupby('sample_id',sort=False)}
    records,blockers=[],[]
    external_ids=set(inventory.loc[inventory.source.eq('test'),'sample_id'])
    external_frame_hashes=set(catalog.loc[catalog.sample_id.isin(external_ids),'frame_sha256'])- {''}
    for number,row in enumerate(selected.to_dict('records'),1):
        record=dict(row,label_provenance='frozen_video_metadata_semantics_unverified',semantic_status='not_assessed')
        try:
            candidates,error=frames.inspect_candidates(root,source,row,groups.get(row['sample_id'],catalog.iloc[:0]),config)
            candidate=frames.choose_candidate(candidates)
            if candidate is None:raise ValueError('No inspectable automatically selected frame')
            if any(not c['readable'] for c in candidates):
                raise ValueError('Corrupt/unreadable candidate(s) detected: '+error)
            if candidate['frame_sha256'] in external_frame_hashes:
                raise ValueError('Selected frame content duplicates cached external data')
            path=b.safe_path(source,candidate['frame_path'])
            image=cv2.imread(str(path))
            processed=letterbox(image)
            raw=work/'selected_frames'/f"{row['sample_id']}.jpg"
            shutil.copy2(path,raw)
            folder=work/'datasets/B'/row['split']/folder_names(8)[int(row['class_id'])]
            folder.mkdir(parents=True,exist_ok=True)
            target=folder/f"{row['sample_id']}.png"
            if not cv2.imwrite(str(target),processed):raise ValueError('Cannot write classification image')
            if row['class_name'] in BASE:
                shared=work/'datasets/A'/target.relative_to(work/'datasets/B')
                shared.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(target,shared)
            record.update(candidate_id=candidate['candidate_id'],frame_index=candidate['frame_index'],
                source_frame_sha256=candidate['frame_sha256'],source_frame_path=candidate['frame_path'],
                pixel_sha256=pixel_sha(image),processed_pixel_sha256=pixel_sha(processed),
                image_sha256=b.sha(target),quality_status=candidate['quality_status'],
                quality_flags=json.dumps(candidate['quality_flags']),focus=candidate['focus'],
                frame_width=image.shape[1],frame_height=image.shape[0],selection_reason=frames.RULE,
                cache_error=error,selection_failure='')
        except (ValueError,OSError,KeyError) as exc:
            record.update(selection_failure=str(exc),quality_status='failed',quality_flags=json.dumps(['selection_failure']))
            blockers.append(f"{row['sample_id']}: {exc}")
        records.append(record)
        if number%40==0:print(f'Classification automatic frames: {number}/{len(selected)}',flush=True)
    manifest=pd.DataFrame(records)
    manifest.to_csv(work/'manifest.csv',index=False)
    for field in ['pixel_sha256','processed_pixel_sha256']:
        if field in manifest and manifest.loc[manifest.quality_status.ne('failed'),field].duplicated().any():
            blockers.append(f'Duplicate selected image content: {field}; no replacement to improve accuracy')
    known_signers=manifest.signer.astype(str).str.strip() if 'signer' in manifest else pd.Series('',index=manifest.index)
    known=manifest[known_signers.ne('')]
    if set(known.loc[known.split.eq('train'),'signer'])&set(known.loc[known.split.eq('val'),'signer']):
        blockers.append('Known signer overlap across frozen splits detected')
    selected_ids=set(selected.sample_id)
    inventory[inventory.source.eq('train') & ~inventory.sample_id.isin(selected_ids)].assign(
        exclusion_reason='not_sampled_fixed_capacity_no_quality_filter').to_csv(work/'reports/not_sampled_internal.csv',index=False)
    dump(work/'reports/readiness.json',dict(ready=not blockers,blockers=blockers,warnings=WARNINGS,
        samples=len(manifest),technical_flagged_n=int(manifest.quality_status.eq('flagged').sum()),
        selection_failed_n=int(manifest.quality_status.eq('failed').sum()),known_signer_samples=len(known),
        semantic_assessed_n=0,known_ambiguous_n=0,semantic_unknown_n=len(manifest)))
    manifest.groupby(['class_name','split','quality_status']).size().rename('n').to_csv(work/'reports/frame_quality_counts.csv')
    experiment=dict(task='classify',classification_pretrained='ImageNet, not COCO detection',
        class_mapping=dict(enumerate(CLASSES)),folder_mapping=folder_names(8),targets=TARGETS,
        training=CONFIG,preprocessing='Full-frame aspect-preserving letterbox to 224x224, gray 114; RGB /255; no crop/flip/augmentation',
        evaluation_checkpoint='last.pt after 30 epochs, same selection rule for both models',
        warnings=WARNINGS,source_snapshot=snapshot,local_only=True,paid_api=False)
    dump(work/'reports/experiment_config.json',experiment)
    if not blockers:
        audit=audit_sample(manifest)
        audit[['sample_id','class_name','split','audit_group','quality_status','quality_flags']].assign(
            visual_technical_review='pending',semantic_decision='not_assessed').to_csv(work/'audit/audit_samples.csv',index=False)
        audit_ui(work,audit)
        files={str(p.relative_to(work)):b.sha(p) for folder in ['selected_frames','datasets','audit']
               for p in (work/folder).rglob('*') if p.is_file()}
        files['manifest.csv']=b.sha(work/'manifest.csv')
        dump(work/'release.json',dict(fingerprint=frames.digest(dict(manifest=files['manifest.csv'],config=experiment)),
            files=files,source_snapshot=snapshot))
        validate_release(work)
    lines=['# Classification pilot readiness','', 'TECHNICAL READY WITH WARNINGS' if not blockers else 'BLOCKED',
        f'Samples={len(manifest)}; flagged={int(manifest.quality_status.eq("flagged").sum())}; failed={len(blockers)}.',
        'No manual semantic audit or missing signer/session metadata blocks this exploratory pilot. '
        'Technical integrity, exact duplicate content, class mapping and detected leakage remain hard gates.',
        '',*['- '+warning for warning in WARNINGS], '',*['BLOCKER: '+error for error in blockers]]
    (work/'reports/pilot_readiness_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (work/'pilot_conclusion.md').write_text('# Classification pilot — NOT TRAINED\n\nNo empirical classification comparison yet.\n',encoding='utf-8')
    return read_json(work/'reports/readiness.json')


def validate_release(work):
    work=Path(work)
    release=read_json(work/'release.json')
    if not release:raise ValueError('Classification release missing')
    for relative,expected in release['files'].items():
        if b.sha(b.safe_path(work,relative)) != expected:raise ValueError('Frozen artifact changed: '+relative)
    manifest=csv(work/'manifest.csv')
    if len(manifest)!=640 or manifest.sample_id.duplicated().any() or not manifest.source.eq('train').all():
        raise ValueError('Manifest identities/count/source mismatch')
    for field in ['video_sha256','pixel_sha256','processed_pixel_sha256']:
        if manifest[field].duplicated().any():raise ValueError('Duplicate content: '+field)
    for name,n in [('A',6),('B',8)]:
        folder=work/'datasets'/name
        for split,target in TARGETS.items():
            classes=sorted(p.name for p in (folder/split).iterdir() if p.is_dir())
            if classes != folder_names(n):raise ValueError('ImageFolder class IDs changed')
            expected=manifest[manifest.split.eq(split)&manifest.class_id.lt(n)]
            actual={p.stem for p in (folder/split).rglob('*.png')}
            if actual!=set(expected.sample_id) or len(list((folder/split).rglob('*.png')))!=target*n:
                raise ValueError('Missing/extra dataset images')
            for row in expected.itertuples():
                if row.class_name!=CLASSES[int(row.class_id)]:raise ValueError('Class mapping changed')
                image=folder/split/folder_names(n)[int(row.class_id)]/f'{row.sample_id}.png'
                decoded=cv2.imread(str(image))
                if decoded is None or decoded.shape!=(SIZE,SIZE,3) or b.sha(image)!=row.image_sha256:
                    raise ValueError('Invalid/changed classification image')
                if name=='A' and b.sha(image)!=b.sha(work/'datasets/B'/image.relative_to(folder)):
                    raise ValueError('Shared A/B images differ')
    return release


def probe(work):
    work=Path(work)
    result=dict(ready=False,blockers=[])
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        Y=detection.local_yolo(work)
        import torch
        from types import SimpleNamespace
        from ultralytics.cfg import get_cfg
        from ultralytics.engine.trainer import BaseTrainer
        checkpoint=read_json(work/'checkpoint.json')
        if not checkpoint or checkpoint['filename']!='yolo26n-cls.pt' or b.sha(checkpoint['path'])!=checkpoint['sha256'] or checkpoint['sha256']!=checkpoint['upstream_lfs_sha256']:
            raise ValueError('Official pinned classification checkpoint missing or corrupt')
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU training fallback')
        get_cfg(overrides=CONFIG)
        model=Y(checkpoint['path'])
        if model.task!='classify' or len(model.names)!=1000 or model.model.yaml.get('scale')!='n':
            raise ValueError('Expected pretrained YOLO26n ImageNet classifier')
        optimizer=BaseTrainer.build_optimizer(SimpleNamespace(),model.model,name='SGD',lr=.01,momentum=.937,decay=.0005)
        groups=[{k:g.get(k) for k in ['lr','momentum','nesterov','weight_decay']} for g in optimizer.param_groups]
        if type(optimizer).__name__!='SGD' or any(g['lr']!=.01 or g['momentum']!=.937 for g in groups):
            raise ValueError('Optimizer API/configuration mismatch')
        model.model.to('cuda').eval()
        with torch.inference_mode():output=model.model(torch.zeros(1,3,SIZE,SIZE,device='cuda'))
        probabilities=output[0] if isinstance(output,tuple) else output
        if tuple(probabilities.shape)!=(1,1000) or not torch.isfinite(probabilities).all():
            raise ValueError('Classification forward API failed')
        result.update(ready=True,checkpoint=checkpoint,task=model.task,dry_forward=list(probabilities.shape),
            cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),optimizer='SGD',optimizer_groups=groups,
            packages={p:metadata.version(p) for p in ['ultralytics','torch','torchvision','numpy','pandas']})
        del output,probabilities,model,optimizer
        torch.cuda.empty_cache()
    except Exception as exc:result['blockers'].append(f'{type(exc).__name__}: {exc}')
    dump(work/'reports/environment.json',result)
    return result


def completed_epochs(trainer):
    # Ultralytics emits a final fit callback after validation with a temporary epoch+1.
    path=Path(trainer.save_dir)/'results.csv'
    return len(csv(path)) if path.exists() else 0


def train(work, source, root):
    work,source,root=Path(work).resolve(),Path(source).resolve(),Path(root).resolve()
    state=read_json(work/'reports/readiness.json')
    if not state or not state['ready']:raise RuntimeError('Basic technical gates failed; see readiness report')
    release=validate_release(work)
    for file,expected in release['source_snapshot'].items():
        path=(root/'experiment_yolo26_pilot/pilot_manifest_private.csv') if file=='human_pilot_manifest' else source/file
        if b.sha(path)!=expected:raise ValueError('Frozen source changed: '+file)
    if any((work/'runs'/f'model_{name}').exists() for name in ['A','B']):
        raise ValueError('Classification run already exists; no overwrite/resume')
    environment=probe(work)
    if not environment['ready']:raise RuntimeError('Environment blocked: '+'; '.join(environment['blockers']))
    Y=detection.local_yolo(work)
    import torch
    import torchvision.transforms as T
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from ultralytics.utils import LOGGER
    logger_file=logging.FileHandler(work/'reports/training.log',encoding='utf-8')
    LOGGER.addHandler(logger_file)
    checkpoint=environment['checkpoint']
    trained=[]
    for name,n in [('A',6),('B',8)]:
        if b.sha(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('Initial classification checkpoint changed')
        class ControlledClassificationTrainer(ClassificationTrainer):
            actual_steps=0

            def build_dataset(self,img_path,mode='train',batch=None):
                dataset=super().build_dataset(img_path,mode,batch)
                dataset.torch_transforms=T.Compose([T.Resize((SIZE,SIZE)),T.ToTensor(),T.Normalize((0.,0.,0.),(1.,1.,1.))])
                expected=TARGETS['train' if mode=='train' else 'val']*n
                if len(dataset)!=expected:raise RuntimeError('Classification dataset silently lost images')
                return dataset

            def _build_train_pipeline(self):
                if self.batch_size!=16:raise RuntimeError('Automatic batch change forbidden; controlled batch=16')
                return super()._build_train_pipeline()

            def _setup_train(self):
                super()._setup_train()
                if self.batch_size!=16 or type(self.optimizer).__name__!='SGD' or list(self.data['names'].values())!=folder_names(n):
                    raise RuntimeError('Actual batch, optimizer or class mapping drift')

            def optimizer_step(self):
                if self.batch_size!=16:raise RuntimeError('Batch drift')
                super().optimizer_step();self.actual_steps+=1

        model=Y(checkpoint['path'])
        if model.task!='classify' or len(model.names)!=1000:raise ValueError('Each model must start from identical ImageNet checkpoint')
        def log_epoch(trainer):
            dump(Path(trainer.save_dir)/'optimizer_steps.json',dict(model=name,epoch=completed_epochs(trainer),
                framework_callback_epoch=int(trainer.epoch)+1,
                actual_optimizer_steps=trainer.actual_steps,batches_per_epoch=len(trainer.train_loader),batch=trainer.batch_size,
                learning_rates=[g['lr'] for g in trainer.optimizer.param_groups]))
        model.add_callback('on_fit_epoch_end',log_epoch)
        try:
            model.train(trainer=ControlledClassificationTrainer,data=str(work/'datasets'/name),
                project=str(work/'runs'),name=f'model_{name}',**CONFIG)
        finally:
            trainer=getattr(model,'trainer',None)
            if trainer is not None and hasattr(trainer,'args'):
                dump(work/'runs'/f'model_{name}_execution.json',dict(initial_checkpoint_sha256=checkpoint['sha256'],
                    actual_optimizer_steps=trainer.actual_steps,actual_args=vars(trainer.args)))
        final=Path(model.trainer.save_dir)/'weights/last.pt'
        if not final.exists() or completed_epochs(model.trainer)!=30:raise RuntimeError('Full 30 pilot epochs not completed')
        trained.append(dict(model=name,path=str(final),sha256=b.sha(final),epoch=30,
            initial_checkpoint_sha256=checkpoint['sha256'],actual_optimizer_steps=model.trainer.actual_steps))
        del model
        torch.cuda.empty_cache()
    dump(work/'runs/trained_models.json',dict(models=trained,release_fingerprint=release['fingerprint']))
    LOGGER.removeHandler(logger_file);logger_file.close()
    evaluate(work)


def plot_matrix(matrix,names,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,8),constrained_layout=True)
    ax.imshow(matrix,cmap='Blues',vmin=0)
    ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(len(names)),yticklabels=names,
        xlabel='Predicted class',ylabel='Video-metadata class',title=title)
    ax.tick_params(axis='x',rotation=60)
    for y,row in enumerate(matrix):
        for x,count in enumerate(row):ax.text(x,y,str(count),ha='center',va='center')
    fig.savefig(path,dpi=160);plt.close(fig)


def evaluate(work):
    work=Path(work).resolve()
    release=validate_release(work)
    trained=read_json(work/'runs/trained_models.json')
    if not trained or trained['release_fingerprint']!=release['fingerprint']:raise ValueError('Two completed independent models required')
    models={r['model']:r for r in trained['models']}
    if set(models)!={'A','B'} or models['A']['initial_checkpoint_sha256']!=models['B']['initial_checkpoint_sha256']:
        raise ValueError('Initial classification checkpoint hashes differ')
    out=work/'evaluation'
    if (out/'evaluation_started.json').exists():raise ValueError('Evaluation exists; refusing overwrite')
    dump(out/'evaluation_started.json',dict(release_fingerprint=release['fingerprint']))
    Y=detection.local_yolo(work)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    manifest=csv(work/'manifest.csv')
    summaries,classes,predictions,diagnostics=[],[],[],[]
    for name,n in [('A',6),('B',8)]:
        record=models[name]
        if b.sha(record['path'])!=record['sha256']:raise ValueError('Trained checkpoint changed')
        model=Y(record['path'])
        if model.task!='classify' or list(model.names.values())!=folder_names(n):raise ValueError('Classifier class mapping changed')
        model.model.to('cuda').eval()
        rows=manifest[manifest.split.eq('val')&manifest.class_id.lt(n)].sort_values('sample_id')
        model_rows=[]
        records=rows.to_dict('records')
        for start in range(0,len(records),16):
            batch=records[start:start+16]
            tensors=[]
            for row in batch:
                image=cv2.imread(str(work/'datasets'/name/'val'/folder_names(n)[int(row['class_id'])]/f"{row['sample_id']}.png"))
                tensors.append(torch.from_numpy(cv2.cvtColor(image,cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).float()/255.)
            with torch.inference_mode():output=model.model(torch.stack(tensors).to('cuda'))
            probability=(output[0] if isinstance(output,tuple) else output).cpu().numpy()
            if probability.shape!=(len(batch),n) or not np.isfinite(probability).all():raise ValueError('Prediction shape/finite check failed')
            for row,scores in zip(batch,probability):
                predicted=int(np.argmax(scores))
                result=dict(model=name,sample_id=row['sample_id'],true_class_id=int(row['class_id']),
                    true_class=row['class_name'],predicted_class_id=predicted,predicted_class=CLASSES[predicted],
                    confidence=float(scores[predicted]),correct=predicted==int(row['class_id']),
                    probabilities=json.dumps([float(v) for v in scores]),quality_flags=row['quality_flags'],quality_status=row['quality_status'])
                model_rows.append(result);predictions.append(result)
                if row['class_name'] in ['apa_kabar','halo']:
                    diagnostics.append(result)
                    if not result['correct']:
                        original=cv2.imread(str(work/'selected_frames'/f"{row['sample_id']}.jpg"))
                        scale=min(1.,960/original.shape[1]);original=cv2.resize(original,(round(original.shape[1]*scale),round(original.shape[0]*scale)))
                        cv2.putText(original,f"GT {row['class_name']} -> {CLASSES[predicted]} {scores[predicted]:.3f}",(10,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,255),2)
                        if not cv2.imwrite(str(out/'error_examples'/f"{name}_{row['sample_id']}.jpg"),original):raise ValueError('Cannot save error example')
        for scope,k in [('common6',6)]+([('full8',8)] if n==8 else []):
            subset=[r for r in model_rows if r['true_class_id']<k]
            aggregate,per_class,matrix=metrics([r['true_class_id'] for r in subset],[r['predicted_class_id'] for r in subset],n,k)
            summaries.append(dict(model=name,scope=scope,**aggregate))
            classes.extend(dict(model=name,scope=scope,**r) for r in per_class)
            pd.DataFrame(matrix,index=CLASSES[:n],columns=CLASSES[:n]).to_csv(out/f'confusion_{name}_{scope}.csv')
            plot_matrix(matrix,CLASSES[:n],out/f'confusion_{name}_{scope}.png',f'Model {name}: {scope}, epoch 30')
        del model
        torch.cuda.empty_cache()
    pd.DataFrame(summaries).to_csv(out/'comparison_metrics.csv',index=False)
    pd.DataFrame(classes).to_csv(out/'per_class_metrics.csv',index=False)
    pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'apa_kabar_halo_diagnostics.csv',index=False)
    dump(out/'evaluation_complete.json',dict(models=models,release_fingerprint=release['fingerprint'],timestamp=datetime.now(timezone.utc).isoformat()))
    conclusion(work,summaries,diagnostics,models)


def conclusion(work,summaries,diagnostics,models):
    state=read_json(work/'reports/readiness.json')
    audit=read_json(work/'audit/imported_audit.json',{})
    ratings=audit.get('records',[])
    assessed=sum(r['decision']!='not_assessed' for r in ratings)
    ambiguous=sum(r['decision'] in ['ambiguous','temporal_ambiguity'] for r in ratings)
    lines=['# Exploratory YOLO26 classification pilot','',
        'Two independent ImageNet -> BISINDO runs; 30 epochs; seed 42. Epoch-30 last.pt used for both comparisons.',
        '| Model | Evaluation | N | Accuracy | Macro-F1 | Extra-class errors |', '|---|---|---:|---:|---:|---:|']
    lines += [f"| {r['model']} | {r['scope']} | {r['n']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | {r['extra_class_predictions_n']} |" for r in summaries]
    lines += ['', 'Macro-F1 common6 uses the same six GT classes for both models. B keeps all eight output scores; no removal/renormalization of halo/tidur predictions.',
        f"Technical flagged frames retained: {state['technical_flagged_n']}/640. Semantic audit assessed={assessed}; known ambiguous/temporal ambiguity={ambiguous}; unknown={640-assessed}. Unknown does not mean valid.",
        f"Actual SGD steps: A={models['A']['actual_optimizer_steps']}, B={models['B']['actual_optimizer_steps']}. More data also yields more updates at equal epochs.", '', '## apa_kabar / halo']
    visual=read_json(work/'audit/technical_visual_summary.json',{})
    if visual:
        lines += [f"Technical visual audit: {visual['viewed_n']} contact-sheet thumbnails viewed ({visual['random_stratified_n']} stratified random + {visual['additional_flagged_n']} additional flagged). "
            f"Unadjudicated concerns: {visual['suspected_nonrepresentative_n']} possible preparation/resting frames; {visual['suspected_cross_class_pose_overlap_n']} cross-class pose-overlap examples. "
            'These are technical observations, not expert semantic verification. No frame or label was replaced.']
    diagnostic=pd.DataFrame(diagnostics)
    for (model,cls),group in diagnostic.groupby(['model','true_class']):
        wrong=group[~group.correct].predicted_class.value_counts().to_dict()
        lines.append(f'- {model}/{cls}: {int(group.correct.sum())}/{len(group)} correct; wrong classes={wrong}.')
    a=next(r for r in summaries if r['model']=='A' and r['scope']=='common6')
    common_b=next(r for r in summaries if r['model']=='B' and r['scope']=='common6')
    full=next(r for r in summaries if r['scope']=='full8')
    lines += ['', '## Interpretation and next step',
        f"B minus A common6: accuracy={common_b['accuracy']-a['accuracy']:+.4f}; macro-F1={common_b['macro_f1']-a['macro_f1']:+.4f}. One seed is descriptive, not proof of superiority.",
        f"Full-eight accuracy={full['accuracy']:.4f}, compared with a balanced top-1 chance reference of 0.125 (not a significance test)."]
    if full['accuracy']>.125 and full['macro_f1']>.125:
        lines.append('Results provide exploratory evidence to continue investigating classification, especially with verified semantic audits, signer metadata and multiple seeds. They do not establish thesis-methodology replacement.')
    else:
        lines.append('This run gives limited evidence for useful classification. Inspect metadata/frame quality and learning curves before further classification trials; detection remains an available localization approach, not a demonstrated better model.')
    lines += ['Detection was not trained on a matched release, so these numbers cannot justify a causal accuracy advantage over detection or automatically returning to it. '
        'Classification labels the whole image; detection additionally localizes hands. Choose based on the thesis task and a later matched comparison.',
        'No temporal comparator was trained. Errors may reflect bad video-to-frame labels, technical selection, visual ambiguity, signer/background cues or limited training.',
        'Clothing/room cues recur in the audited image groups. High video-label accuracy does not establish that the model learned linguistically valid gestures. '
        'This is closed-set classification; background/unknown-gesture rejection, real-time latency and signer-independent generalization were not tested.',
        'Before final experiments: assess semantic errors and label validity, collect participant metadata, freeze configuration, use multiple seeds and only then run external Kaggle testing.', '',
        '## Limitations',*['- '+w for w in WARNINGS]]
    (work/'pilot_conclusion.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def import_audit(work,path):
    work=Path(work)
    payload=read_json(path)
    audit=csv(work/'audit/audit_samples.csv')
    if payload.get('schema')!=1 or payload.get('kind')!='classification_quality_audit' or payload.get('fingerprint')!=b.sha(work/'manifest.csv') or not str(payload.get('reviewer','')).strip():
        raise ValueError('Wrong/stale audit identity')
    records=payload.get('records',[])
    ids=[r.get('sample_id') for r in records]
    if len(ids)!=len(set(ids)) or set(ids)!=set(audit.sample_id):raise ValueError('Audit sample IDs do not match')
    for r in records:
        if r.get('decision') not in AUDIT_DECISIONS or not isinstance(r.get('note'),str):raise ValueError('Invalid audit decision')
        if datetime.fromisoformat(r['timestamp'].replace('Z','+00:00')).utcoffset() is None:raise ValueError('Audit timestamp timezone required')
    target=work/'audit/imported_audit.json'
    if target.exists():raise ValueError('Audit already imported; no overwrite')
    dump(target,payload)
    dump(work/'reports/semantic_audit_summary.json',dict(assessed_n=sum(r['decision']!='not_assessed' for r in records),
        known_ambiguous_n=sum(r['decision'] in ['ambiguous','temporal_ambiguity'] for r in records),
        unknown_n=640-sum(r['decision']!='not_assessed' for r in records),
        diagnostic_only=True,frame_or_label_changes=0))
    # Audit is diagnostic only; frozen images and weak labels are never reselected.
    if (work/'evaluation/evaluation_complete.json').exists():
        results=csv(work/'evaluation/comparison_metrics.csv').to_dict('records')
        diagnostic=csv(work/'evaluation/apa_kabar_halo_diagnostics.csv').to_dict('records')
        models={r['model']:r for r in read_json(work/'runs/trained_models.json')['models']}
        conclusion(work,results,diagnostic,models)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','probe','train','evaluate','import-audit'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--work',type=Path)
    parser.add_argument('--file',type=Path)
    args=parser.parse_args()
    root=args.root.resolve();source=args.source or root/'experiment_yolo26_v3';work=args.work or root/'experiment_yolo26_classification_pilot'
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    if args.command=='prepare':prepare(root,source,work)
    elif args.command=='probe':print(json.dumps(probe(work),indent=2))
    elif args.command=='train':train(work,source,root)
    elif args.command=='evaluate':evaluate(work)
    elif args.command=='import-audit':
        if not args.file:parser.error('--file required')
        import_audit(work,args.file)


if __name__=='__main__':main()
