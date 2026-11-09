"""exploratory/development parameterized Classification. Legacy CLI preserved; no resume/overwrite."""
import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import importlib.util
import json
import logging
from pathlib import Path

import classification_pilot as base
from analysis_paired import markdown_table

PINNED_SHA = '0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81'


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    epochs: int = 30
    imgsz: int = 224

    def __post_init__(self):
        if type(self.seed) is not int or self.seed not in [42,43,44,45,46]:
            raise ValueError('Training seed must be 42..46; sampling seed stays 42')
        if type(self.epochs) is not int or not 1 <= self.epochs <= 1000:
            raise ValueError('epochs must be an integer in 1..1000; training still requires scoped approval')
        if type(self.imgsz) is not int or self.imgsz not in [224,320,448]:
            raise ValueError('imgsz must be 224,320,448')


def configured_core(settings):
    # Private module namespace reuses legacy helpers without mutating original module globals.
    spec=importlib.util.spec_from_file_location('_bisindo_research_core',Path(base.__file__))
    core=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    core.SIZE=settings.imgsz
    core.CONFIG=dict(base.CONFIG,**asdict(settings))
    plot=core.plot_matrix
    core.plot_matrix=lambda matrix,names,path,title:plot(matrix,names,path,title.replace('epoch 30',f'epoch {settings.epochs}'))

    def conclusion(work,summaries,diagnostics,models):
        rows=core.pd.DataFrame(summaries)
        state=core.read_json(Path(work)/'reports/readiness.json')
        text=(f'# FACT — exploratory Classification, seed {settings.seed}\n\n'
              f'{settings.epochs} epochs; input {settings.imgsz}; fixed last.pt; independent initialization.\n\n'
              +markdown_table(rows)+'\n\n'+
              'Common6 retains all B outputs. Full8 is a separate task. No claim of verified semantics or signer independence.\n'
              +f'Technical flags retained: {state["technical_flagged_n"]}. '+
              f'Actual steps: { {k:v["actual_optimizer_steps"] for k,v in models.items()} }.\n'
              +'No external evaluation, application benchmark or measured inference cost in this report. '
              +'One seed is descriptive; no confirmatory conclusion.\n')
        (Path(work)/'pilot_conclusion.md').write_text(text,encoding='utf8')
    core.conclusion=conclusion
    return core


def new_work(root,work):
    root,work=Path(root).resolve(),Path(work).resolve()
    protected={'train','test','experiment_yolo26','experiment_yolo26_v2','experiment_yolo26_v3',
        'experiment_yolo26_pilot','experiment_yolo26_training_pilot','experiment_yolo26_classification_pilot',
        'experiment_yolo26_classification_notebook_20261102'}
    if work.parent!=root or work.name in protected or work.exists():
        raise ValueError('Use a new direct-child work directory, outside all prior experiments')
    return work


def settings_record(settings):
    return dict(schema=1,training=asdict(settings),sampling_seed=42,
        source_sha256={name:base.b.sha(Path(__file__).with_name(name)) for name in
                       ['classification_research.py','classification_diagnostics.py','classification_workflow.py',
                        'classification_pilot.py','training_pilot.py','pilot_study.py','frame_candidates.py','bisindo_pipeline.py','analysis_paired.py']})


def prepare(root,source,work,settings=Settings(),checkpoint=None):
    root,source=Path(root).resolve(),Path(source).resolve()
    work=new_work(root,work)
    checkpoint=Path(checkpoint).resolve() if checkpoint else root/'experiment_yolo26_classification_pilot/checkpoints/yolo26n-cls.pt'
    if not checkpoint.is_file() or checkpoint.name!='yolo26n-cls.pt' or base.b.sha(checkpoint)!=PINNED_SHA:
        raise ValueError('Pinned pretrained local Classification checkpoint required; no automatic substitution')
    work.mkdir()
    record=settings_record(settings)
    record['settings_fingerprint']=base.frames.digest(record)
    base.dump(work/'research_config.json',record)
    core=configured_core(settings)
    result=core.prepare(root,source,work)
    base.dump(work/'checkpoint.json',dict(filename='yolo26n-cls.pt',path=str(checkpoint),sha256=PINNED_SHA,upstream_lfs_sha256=PINNED_SHA))
    # Legacy prose describes fixed 224/30; replace only this NEW run's config and fingerprint.
    config=base.read_json(work/'reports/experiment_config.json')
    config['preprocessing']=f'Full-frame aspect-preserving letterbox to {settings.imgsz}x{settings.imgsz}, gray114; RGB/255; no crop/flip'
    config['evaluation_checkpoint']=f'last.pt after {settings.epochs} epochs, same for A/B'
    base.dump(work/'reports/experiment_config.json',config)
    if result['ready']:
        release=base.read_json(work/'release.json')
        release['fingerprint']=base.frames.digest(dict(manifest=base.b.sha(work/'manifest.csv'),config=config))
        release['research_config_sha256']=base.b.sha(work/'research_config.json')
        base.dump(work/'release.json',release)
    return result


def load_settings(work,overrides=None):
    work=Path(work)
    record=base.read_json(work/'research_config.json')
    if not record:raise ValueError('Parameterized run settings missing; do not reinterpret legacy completed runs')
    digest=record['settings_fingerprint']
    if base.frames.digest({k:v for k,v in record.items() if k!='settings_fingerprint'})!=digest:
        raise ValueError('Settings fingerprint changed')
    settings=Settings(**record['training'])
    for name,value in (overrides or {}).items():
        if value is not None and value!=getattr(settings,name):raise ValueError('CLI conflicts with locked prepared settings: '+name)
    for name,expected in record['source_sha256'].items():
        if base.b.sha(Path(__file__).with_name(name))!=expected:raise ValueError('Prepared source changed: '+name)
    return settings


def validate_release(work,settings):
    work=Path(work)
    core=configured_core(settings)
    release=core.validate_release(work)
    config=base.read_json(work/'reports/experiment_config.json')
    if release.get('research_config_sha256')!=base.b.sha(work/'research_config.json'):
        raise ValueError('Prepared settings hash changed')
    if config['training']!=core.CONFIG or release['fingerprint']!=base.frames.digest(dict(manifest=base.b.sha(work/'manifest.csv'),config=config)):
        raise ValueError('Config/release mismatch')
    return release


def probe(work,overrides=None):
    settings=load_settings(work,overrides)
    validate_release(work,settings)
    return configured_core(settings).probe(Path(work))


def authorize_training(root,work,settings,approval):
    if approval is None:raise PermissionError('No training approval. Stage 1a authorizes zero fits.')
    record=base.read_json(approval)
    if not record or record.get('kind')!='bisindo_training_approval' or record.get('scope')!='stage1b' or not record.get('approved_by'):
        raise PermissionError('Explicit Stage 1b approval record required')
    if Path(work).resolve().name not in record.get('work_names',[]) or record.get('training')!=asdict(settings):
        raise PermissionError('Approval does not cover this work/settings')
    files=record.get('protocol_sha256',{})
    # Researcher decision 2026-11-03: exploratory Stage1b needs scoped approval; addendum gates Stage2/3.
    allowed={'PREREGISTRATION.md','PREREGISTRATION_ADDENDUM_01.md'}
    if 'PREREGISTRATION.md' not in files or not set(files)<=allowed:
        raise PermissionError('Candidate protocol hash required; this entry point authorizes Stage1b only')
    for name,expected in files.items():
        path=Path(root)/name
        if not path.is_file() or base.b.sha(path)!=expected:raise PermissionError('Protocol lock mismatch')
    return base.b.sha(approval)


def evaluate(work,overrides=None):
    work=Path(work).resolve()
    settings=load_settings(work,overrides)
    if (work/'evaluation/evaluation_started.json').exists():raise ValueError('Evaluation exists; no overwrite/resume')
    if any((work/'evaluation'/name).exists() for name in ['predictions.csv','comparison_metrics.csv','evaluation_complete.json']):
        raise ValueError('Partial/completed evaluation exists; no overwrite/resume')
    validate_release(work,settings)
    trained=base.read_json(work/'runs/trained_models.json')
    if not trained or len(trained.get('models',[]))!=2 or any(r.get('epoch')!=settings.epochs for r in trained['models']):
        raise ValueError('Both fixed-epoch checkpoints required')
    configured_core(settings).evaluate(work)


def train(root,source,work,overrides=None,approval=None):
    root,source,work=Path(root).resolve(),Path(source).resolve(),Path(work).resolve()
    settings=load_settings(work,overrides)
    approval_sha=authorize_training(root,work,settings,approval)
    if any((work/'runs'/f'model_{name}').exists() for name in ['A','B']):
        raise ValueError('Classification run exists; no overwrite/resume')
    state=base.read_json(work/'reports/readiness.json')
    if not state or not state['ready']:raise RuntimeError('Technical gates failed')
    release=validate_release(work,settings)
    for file,expected in release['source_snapshot'].items():
        path=root/'experiment_yolo26_pilot/pilot_manifest_private.csv' if file=='human_pilot_manifest' else source/file
        if base.b.sha(path)!=expected:raise ValueError('Frozen source changed: '+file)
    core=configured_core(settings)
    env=core.probe(work)
    if not env['ready']:raise RuntimeError('Environment blocked: '+'; '.join(env['blockers']))
    Y=core.detection.local_yolo(work)
    import torch
    import torchvision.transforms as T
    from ultralytics.models.yolo.classify import ClassificationTrainer
    from ultralytics.utils import LOGGER
    handle=logging.FileHandler(work/'reports/training.log',encoding='utf8');LOGGER.addHandler(handle)
    trained=[];checkpoint=env['checkpoint']
    base.dump(work/'reports/training_authorization.json',dict(approval_sha256=approval_sha,settings=asdict(settings),started_utc=datetime.now(timezone.utc).isoformat()))
    try:
        for name,n in [('A',6),('B',8)]:
            if base.b.sha(checkpoint['path'])!=PINNED_SHA:raise ValueError('Initial checkpoint changed')
            class ControlledTrainer(ClassificationTrainer):
                actual_steps=0

                def build_dataset(self,img_path,mode='train',batch=None):
                    data=super().build_dataset(img_path,mode,batch)
                    data.torch_transforms=T.Compose([T.Resize((settings.imgsz,settings.imgsz)),T.ToTensor(),T.Normalize((0.,0.,0.),(1.,1.,1.))])
                    if len(data)!=core.TARGETS['train' if mode=='train' else 'val']*n:raise RuntimeError('Dataset lost images')
                    return data

                def _build_train_pipeline(self):
                    if self.batch_size!=16:raise RuntimeError('Automatic batch change forbidden')
                    return super()._build_train_pipeline()

                def _setup_train(self):
                    super()._setup_train()
                    if self.batch_size!=16 or type(self.optimizer).__name__!='SGD' or list(self.data['names'].values())!=core.folder_names(n):
                        raise RuntimeError('Batch/optimizer/mapping drift')

                def optimizer_step(self):
                    if self.batch_size!=16:raise RuntimeError('Batch drift')
                    super().optimizer_step();self.actual_steps+=1

            def log_epoch(trainer):
                base.dump(Path(trainer.save_dir)/'optimizer_steps.json',dict(model=name,seed=settings.seed,
                    epoch=core.completed_epochs(trainer),actual_optimizer_steps=trainer.actual_steps,
                    batches_per_epoch=len(trainer.train_loader),learning_rates=[g['lr'] for g in trainer.optimizer.param_groups]))
            def train_curve(trainer):
                from classification_diagnostics import log_train_epoch
                log_train_epoch(trainer,work/'datasets'/name/'train',settings.imgsz,name)
            model=Y(checkpoint['path'])
            if model.task!='classify' or len(model.names)!=1000:raise ValueError('Independent pretrained Classification initialization required')
            model.add_callback('on_fit_epoch_end',log_epoch)
            model.add_callback('on_train_epoch_end',train_curve)
            try:
                model.train(trainer=ControlledTrainer,data=str(work/'datasets'/name),project=str(work/'runs'),name=f'model_{name}',**core.CONFIG)
            finally:
                trainer=getattr(model,'trainer',None)
                if trainer is not None and hasattr(trainer,'args'):
                    base.dump(work/'runs'/f'model_{name}_execution.json',dict(actual_args=vars(trainer.args),
                        actual_optimizer_steps=trainer.actual_steps,initial_checkpoint_sha256=PINNED_SHA,
                        release_fingerprint=release['fingerprint'],settings=asdict(settings)))
            final=Path(model.trainer.save_dir)/'weights/last.pt'
            epochs=base.pd.read_csv(Path(model.trainer.save_dir)/'results.csv')['epoch'].tolist()
            if not final.is_file() or epochs!=list(range(1,settings.epochs+1)):raise RuntimeError('Fixed epochs incomplete')
            trained.append(dict(model=name,path=str(final),sha256=base.b.sha(final),epoch=settings.epochs,
                seed=settings.seed,initial_checkpoint_sha256=PINNED_SHA,actual_optimizer_steps=model.trainer.actual_steps))
            del model;torch.cuda.empty_cache()
        base.dump(work/'runs/trained_models.json',dict(models=trained,release_fingerprint=release['fingerprint']))
    finally:
        LOGGER.removeHandler(handle);handle.close()
    evaluate(work)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','probe','train','evaluate'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--source',type=Path)
    parser.add_argument('--work','--work-dir',dest='work',type=Path,required=True)
    for name in ['seed','epochs','imgsz']:parser.add_argument('--'+name,type=int)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--approval',type=Path)
    args=parser.parse_args(argv)
    overrides={name:getattr(args,name) for name in ['seed','epochs','imgsz']}
    root=args.root.resolve();source=args.source or root/'experiment_yolo26_v3'
    if args.command=='prepare':prepare(root,source,args.work,Settings(**{k:v for k,v in overrides.items() if v is not None}),args.checkpoint)
    elif args.command=='probe':print(json.dumps(probe(args.work,overrides),indent=2))
    elif args.command=='evaluate':evaluate(args.work,overrides)
    elif args.command=='train':train(root,source,args.work,overrides,args.approval)


if __name__=='__main__':
    main()
