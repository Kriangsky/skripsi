"""exploratory/development: deterministic curves, cached hand masks, frozen linear probes.

No training runs on import. Production fits require scoped researcher approval.
"""
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from PIL import Image

import classification_research as r
from analysis_paired import markdown_table


def fresh_output(work,out):
    work,out=Path(work).resolve(),Path(out).resolve()
    if out==work or out.is_relative_to(work) or work.is_relative_to(out):
        raise ValueError('Output must be separate from the source run')
    protected=['train','test','experiment_yolo26','experiment_yolo26_v2','experiment_yolo26_v3',
               'experiment_yolo26_pilot','experiment_yolo26_training_pilot',
               'experiment_yolo26_classification_pilot','experiment_yolo26_classification_notebook_20261102']
    if any(out.is_relative_to(work.parent/name) for name in protected):raise ValueError('Protected source/run output path')
    out.mkdir(parents=True,exist_ok=False)
    return out


def image_loader(folder,size,shuffle=False,seed=20261103):
    import torch
    from torchvision.datasets import ImageFolder
    from torchvision.transforms import Compose,Resize,ToTensor
    dataset=ImageFolder(str(folder),transform=Compose([Resize((size,size)),ToTensor()]))
    return torch.utils.data.DataLoader(dataset,batch_size=16,shuffle=shuffle,num_workers=0,
                                      generator=torch.Generator().manual_seed(seed))


def measure_classifier(model,loader,device):
    """Eval-only diagnostic of actual training weights; restore every mode and RNG."""
    import torch
    modes=[(module,module.training) for module in model.modules()]
    numpy_state=np.random.get_state();python_state=random.getstate()
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    correct=n=0;loss=0.
    try:
        with torch.random.fork_rng(devices=devices),torch.inference_mode():
            model.eval()
            for images,labels in loader:
                images,labels=images.to(device),labels.to(device)
                output=model(images)
                logits=output[1] if isinstance(output,tuple) else output
                if logits.ndim!=2 or not torch.isfinite(logits).all():raise ValueError('Invalid classifier output')
                correct+=int((logits.argmax(1)==labels).sum());n+=len(labels)
                loss+=float(torch.nn.functional.cross_entropy(logits,labels,reduction='sum'))
    finally:
        for module,training in modes:module.training=training
        np.random.set_state(numpy_state);random.setstate(python_state)
    if not n:raise ValueError('Empty training diagnostic set')
    return dict(n=n,correct=correct,top1=correct/n,loss=loss/n)


def log_train_epoch(trainer,folder,size,model_name):
    import torch
    # Loader construction and iteration use a private generator, never the training sampler.
    loader=image_loader(folder,size)
    expected=r.base.folder_names(6 if model_name=='A' else 8)
    if loader.dataset.classes!=expected:raise ValueError('Train diagnostic mapping drift')
    started=time.perf_counter()
    record=measure_classifier(trainer.model,loader,trainer.device)
    path=Path(trainer.save_dir)/'exploratory_train_curve.csv'
    epoch=int(trainer.epoch)+1
    existing=pd.read_csv(path).to_dict('records') if path.exists() else []
    if any(row['epoch']>=epoch for row in existing):raise ValueError('Repeated train epoch diagnostic')
    record.update(epoch=epoch,model=model_name,weights='actual_training_model_not_EMA',
                  scope='A6' if model_name=='A' else 'B8',seconds=time.perf_counter()-started,
                  status='exploratory/development',augmentation=False)
    pd.DataFrame(existing+[record]).to_csv(path,index=False)


def mask_hands(image,hands):
    image=np.asarray(image)
    if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<1:
        raise ValueError('Expected nonempty uint8 RGB image')
    h,w=image.shape[:2];mask=np.zeros((h,w),np.uint8);masked=image.copy()
    record=dict(stratum='landmarks_unavailable',coverage=0.,roi_xyxy=None)
    if not hands:return masked,mask,record
    if any(np.asarray(hand).shape!=(21,2) or not np.isfinite(hand).all() for hand in hands):
        raise ValueError('Invalid cached hand landmarks')
    points=np.concatenate(hands);lo=points.min(axis=0);hi=points.max(axis=0)
    padding=(hi-lo)*.15
    x1,y1=np.clip(np.floor((lo-padding)*[w,h]),[0,0],[w,h]).astype(int)
    x2,y2=np.clip(np.ceil((hi+padding)*[w,h]),[0,0],[w,h]).astype(int)
    if x2<=x1 or y2<=y1:raise ValueError('Degenerate landmark union ROI')
    mask[y1:y2,x1:x2]=255;masked[mask>0]=114
    record.update(stratum='landmarks_available',coverage=float((mask>0).mean()),roi_xyxy=[int(v) for v in [x1,y1,x2,y2]])
    return masked,mask,record


def prepare_masks(work,source,out,size=224):
    work,source=Path(work).resolve(),Path(source).resolve()
    manifest=pd.read_csv(work/'manifest.csv',keep_default_na=False)
    cache=pd.read_csv(source/'candidates.csv',keep_default_na=False)
    if manifest.sample_id.duplicated().any() or manifest.video.duplicated().any() or cache.candidate_id.duplicated().any():
        raise ValueError('Duplicate identity in mask inputs')
    joined=manifest.merge(cache[['candidate_id','sample_id','frame_sha256','landmarks']],
                          on=['sample_id','candidate_id'],validate='one_to_one',how='left',indicator=True)
    if not joined['_merge'].eq('both').all():raise ValueError('Missing landmark cache record')
    out=fresh_output(work,out)
    (out/'exploratory_masks').mkdir();(out/'exploratory_images').mkdir()
    core=r.configured_core(r.Settings(imgsz=size));rows=[]
    for row in joined.itertuples():
        raw=work/'selected_frames'/f'{row.sample_id}.jpg'
        if r.base.b.sha(raw)!=row.source_frame_sha256 or row.frame_sha256!=row.source_frame_sha256:
            raise ValueError('Mask source/cache hash mismatch')
        with Image.open(raw) as opened:image=np.asarray(opened.convert('RGB'))
        masked,mask,record=mask_hands(image,json.loads(row.landmarks))
        square=core.letterbox(masked)
        mask_path=out/'exploratory_masks'/f'{row.sample_id}.png'
        image_path=out/'exploratory_images'/f'{row.sample_id}.png'
        Image.fromarray(mask).save(mask_path);Image.fromarray(square).save(image_path)
        rows.append(dict(sample_id=row.sample_id,video=row.video,video_sha256=row.video_sha256,
            split=row.split,class_id=row.class_id,class_name=row.class_name,**record,
            status='exploratory/development',source_sha256=row.source_frame_sha256,
            mask_sha256=r.base.b.sha(mask_path),image_sha256=r.base.b.sha(image_path)))
    result=pd.DataFrame(rows);result.to_csv(out/'exploratory_mask_coverage.csv',index=False)
    r.base.dump(out/'exploratory_mask_provenance.json',dict(imgsz=size,padding=.15,fill=114,
        manifest_sha256=r.base.b.sha(work/'manifest.csv'),cache_sha256=r.base.b.sha(source/'candidates.csv'),
        coverage_sha256=r.base.b.sha(out/'exploratory_mask_coverage.csv'),
        interpretation='Hand masked context-only; face/body may remain. No landmarks retained unchanged as separate stratum.'))
    return result


def load_frozen_model(work,name,out):
    import torch
    work=Path(work)
    if name not in ['A','B']:raise ValueError('Expected model A/B')
    record=r.base.read_json(work/'runs/trained_models.json')
    if not record or {x['model'] for x in record['models']}!={'A','B'}:raise ValueError('Two trained model records required')
    item=next(x for x in record['models'] if x['model']==name)
    release=r.base.read_json(work/'release.json')
    config=r.base.read_json(work/'reports/experiment_config.json')
    if not release or record.get('release_fingerprint')!=release['fingerprint'] or item.get('epoch')!=config['training']['epochs']:
        raise ValueError('Backbone release/epoch mismatch')
    if Path(item['path']).name!='last.pt' or r.base.b.sha(item['path'])!=item['sha256']:
        raise ValueError('Fixed last.pt hash mismatch')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU fallback')
    Y=r.base.detection.local_yolo(Path(out))
    wrapper=Y(item['path'])
    if wrapper.task!='classify' or list(wrapper.names.values())!=r.base.folder_names(6 if name=='A' else 8):
        raise ValueError('Backbone classification mapping mismatch')
    model=wrapper.model.to('cuda').eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    linear=model.model[-1].linear
    if not isinstance(linear,torch.nn.Linear):raise RuntimeError('Unsupported classification head API')
    return model,linear,item


def extract_features(model,linear,loader,device):
    import torch
    model.eval()
    for parameter in model.parameters():parameter.requires_grad_(False)
    captured=[];features=[];labels=[]
    hook=linear.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().cpu()))
    try:
        with torch.inference_mode():
            for images,y in loader:
                captured.clear();model(images.to(device))
                if len(captured)!=1 or captured[0].ndim!=2 or not torch.isfinite(captured[0]).all():
                    raise RuntimeError('Unsupported backbone embedding API')
                features.append(captured[0].clone());labels.append(y.clone())
    finally:hook.remove()
    if not features:raise ValueError('Empty feature dataset')
    return torch.cat(features),torch.cat(labels)


def fit_head(features,labels,seed,device='cuda'):
    """Fixed 30 epochs on train features only. Caller enforces production authorization."""
    import torch
    if features.ndim!=2 or len(labels)!=len(features) or set(labels.tolist())!=set(range(6)):
        raise ValueError('Six-class old-train features required')
    if not torch.isfinite(features).all():raise ValueError('Nonfinite features')
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        head=torch.nn.Linear(features.shape[1],6).to(device)
        optimizer=torch.optim.SGD(head.parameters(),lr=.01,momentum=.937,weight_decay=.0005)
        loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(features,labels),batch_size=16,
            shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
        logs=[];steps=0
        for epoch in range(1,31):
            head.train();total=0.
            for x,y in loader:
                optimizer.zero_grad(set_to_none=True)
                loss=torch.nn.functional.cross_entropy(head(x.to(device)),y.to(device))
                loss.backward();optimizer.step();steps+=1;total+=float(loss.detach())*len(y)
            head.eval()
            with torch.inference_mode():
                correct=int((head(features.to(device)).argmax(1).cpu()==labels).sum())
            logs.append(dict(epoch=epoch,train_loss=total/len(labels),train_top1=correct/len(labels),
                             actual_optimizer_steps=steps,status='exploratory/development'))
        return head,logs


def fit_linear_probe(work,name,out,approval=None):
    if approval is None:raise PermissionError('Explicit Stage1b approval required for every head fit')
    import torch
    work=Path(work).resolve();settings=r.load_settings(work)
    r.authorize_training(work.parent,work,settings,approval);r.validate_release(work,settings)
    record=r.base.read_json(approval)
    if record.get('linear_probes') is not True:raise PermissionError('Approval must explicitly include linear_probes')
    out=fresh_output(work,out);started=time.perf_counter()
    model,linear,item=load_frozen_model(work,name,out)
    # B ImageFolder has eight classes; use only old training labels, without changing source data.
    loader=image_loader(work/'datasets/A/train',settings.imgsz)
    if loader.dataset.classes!=r.base.folder_names(6) or len(loader.dataset)!=360:raise ValueError('Probe old-train mapping/count drift')
    features,labels=extract_features(model,linear,loader,'cuda')
    head,logs=fit_head(features,labels,settings.seed)
    torch.save(dict(state_dict={k:v.cpu() for k,v in head.state_dict().items()},input_features=features.shape[1],
                    classes=r.base.BASE,epoch=30,seed=settings.seed,backbone_sha256=item['sha256']),out/'exploratory_last_head.pt')
    pd.DataFrame(logs).to_csv(out/'exploratory_head_curve.csv',index=False)
    # Validation is read only after all training epochs; no selection or early stopping.
    validation=image_loader(work/'datasets/A/val',settings.imgsz)
    vx,vy=extract_features(model,linear,validation,'cuda')
    with torch.inference_mode():pred=head(vx.to('cuda')).argmax(1).cpu().tolist()
    metrics,details,_=r.base.metrics(vy.tolist(),pred,6,6)
    pd.DataFrame(dict(sample_id=[Path(p).stem for p,_ in validation.dataset.samples],true_class_id=vy.tolist(),
                      predicted_class_id=pred)).to_csv(out/'exploratory_probe_predictions.csv',index=False)
    r.base.dump(out/'exploratory_probe_summary.json',dict(status='exploratory/development',model=name,
        seed=settings.seed,epoch=30,backbone_eval=True,backbone_sha256=item['sha256'],
        optimizer=dict(name='SGD',lr=.01,momentum=.937,weight_decay=.0005,batch=16),
        train_n=len(labels),validation_used_for_selection=False,seconds=time.perf_counter()-started,**metrics))
    del model,head;torch.cuda.empty_cache()
    return metrics


def evaluate_masked(work,masks,out):
    import torch
    work,masks=Path(work).resolve(),Path(masks).resolve()
    settings=r.load_settings(work);r.validate_release(work,settings)
    coverage=pd.read_csv(masks/'exploratory_mask_coverage.csv')
    manifest=pd.read_csv(work/'manifest.csv')
    provenance=r.base.read_json(masks/'exploratory_mask_provenance.json')
    if (not provenance or provenance.get('coverage_sha256')!=r.base.b.sha(masks/'exploratory_mask_coverage.csv')
        or provenance.get('manifest_sha256')!=r.base.b.sha(work/'manifest.csv')
        or provenance.get('imgsz')!=settings.imgsz):raise ValueError('Mask coverage/provenance drift')
    if coverage.sample_id.duplicated().any() or set(coverage.sample_id)!=set(manifest.sample_id):raise ValueError('Mask identities differ')
    fields=['split','class_id','class_name','video','video_sha256']
    check=manifest[['sample_id','source_frame_sha256',*fields]].merge(coverage,on='sample_id',validate='one_to_one')
    if any(not check[field+'_x'].eq(check[field+'_y']).all() for field in fields) or not check.source_frame_sha256.eq(check.source_sha256).all():
        raise ValueError('Mask GT/split/video/source mismatch')
    out=fresh_output(work,out);predictions=[];summaries=[]
    for name,n in [('A',6),('B',8)]:
        model,_,item=load_frozen_model(work,name,out)
        rows=coverage[coverage.split.eq('val')&coverage.class_id.lt(n)].sort_values('sample_id')
        records=rows.to_dict('records')
        for start in range(0,len(records),16):
            batch=records[start:start+16];tensors=[]
            for row in batch:
                path=masks/'exploratory_images'/f"{row['sample_id']}.png"
                if r.base.b.sha(path)!=row['image_sha256']:raise ValueError('Masked image hash drift')
                with Image.open(path) as opened:image=np.asarray(opened.convert('RGB')).copy()
                if image.shape!=(settings.imgsz,settings.imgsz,3):raise ValueError('Mask resolution drift')
                tensors.append(torch.from_numpy(image.transpose(2,0,1).copy()).float()/255.)
            with torch.inference_mode():output=model(torch.stack(tensors).to('cuda'))
            scores=(output[0] if isinstance(output,tuple) else output).cpu().numpy()
            if scores.shape!=(len(batch),n) or not np.isfinite(scores).all():raise ValueError('Masked classifier output drift')
            for row,p in zip(batch,scores):
                predictions.append(dict(model=name,sample_id=row['sample_id'],true_class_id=row['class_id'],
                    predicted_class_id=int(p.argmax()),stratum=row['stratum'],coverage=row['coverage']))
        del model;torch.cuda.empty_cache()
    frame=pd.DataFrame(predictions)
    for (name,stratum),part in frame.groupby(['model','stratum']):
        for scope,k in [('common6',6)]+([('full8',8)] if name=='B' else []):
            subset=part[part.true_class_id.lt(k)]
            if len(subset):
                stats,_,_=r.base.metrics(subset.true_class_id.tolist(),subset.predicted_class_id.tolist(),6 if name=='A' else 8,k)
                summaries.append(dict(model=name,stratum=stratum,scope=scope,balanced_task_chance=1/(6 if name=='A' else 8),**stats))
    for name,part in frame.groupby('model'):
        for scope,k in [('common6',6)]+([('full8',8)] if name=='B' else []):
            subset=part[part.true_class_id.lt(k)]
            stats,_,_=r.base.metrics(subset.true_class_id.tolist(),subset.predicted_class_id.tolist(),6 if name=='A' else 8,k)
            summaries.append(dict(model=name,stratum='all_including_unavailable',scope=scope,balanced_task_chance=1/(6 if name=='A' else 8),**stats))
    frame.to_csv(out/'exploratory_masked_predictions.csv',index=False)
    pd.DataFrame(summaries).to_csv(out/'exploratory_masked_metrics.csv',index=False)
    (out/'exploratory_masked_report.md').write_text('# exploratory/development — hand masked\n\n'+markdown_table(pd.DataFrame(summaries))+
        '\n\nChance is a uniform-output reference, not an empirical null for imbalanced strata. Missing landmarks remain in overall denominator. '
        'Face/body may remain; distribution shift and incomplete hand landmarks prevent a pure-background causal claim.\n',encoding='utf8')
    return pd.DataFrame(summaries)
