"""Read local historical metadata/weights on CPU; never train, infer, prepare or score."""
from datetime import datetime, timezone
import importlib.metadata as metadata
import json
import os
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'05_kode/src'))
import classification_research as research
import nbformat
import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(name, value):
    if (OUT/name).exists():
        assert read(OUT/name)==value, 'Existing evidence differs: '+name
        return
    with (OUT/name).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, default=str)


def main():
    os.environ['YOLO_CONFIG_DIR'] = str(OUT/'ultralytics_settings')
    os.environ['YOLO_OFFLINE'] = 'true'
    protected = set()
    for directory in ['05_kode', '07_dokumen/protokol', '06_notebook']:
        protected.update(p for p in (ROOT/directory).rglob('*') if p.is_file()
                         and '__pycache__' not in p.parts and p.suffix in {'.py','.json','.md','.sha256','.ipynb','.csv'})
    for old in ['data_inventory_task1_20261105', 'scientific_decisions_task2_20261105']:
        for name in read(ROOT/'01_hasil/addendum'/old/'protected_before.json'):
            path = ROOT/name
            if path.is_file():
                protected.add(path)
    for p in (ROOT/'02_exp/tahap1b/full_runs').rglob('*'):
        if p.is_file() and not {'datasets','selected_frames','ultralytics_settings'}.intersection(p.parts):
            protected.add(p)
    protected.add(ROOT/'02_exp/arsip_pipeline/experiment_yolo26_v3/split_lock.csv')
    protected.add(ROOT/'02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/checkpoints/yolo26n-cls.pt')
    write('protected_before.json', {str(p.relative_to(ROOT)):research.base.b.sha(p) for p in sorted(protected)})
    nb = nbformat.read(ROOT/'06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb',4)
    write('notebook_before.json', nb)
    write('tracker_before.json', {'text':(ROOT/'07_dokumen/TRACKING_BISINDO.md').read_text(encoding='utf8')})
    import torch
    import yaml
    import pandas as pd
    import ultralytics
    package = Path(ultralytics.__file__).parent
    framework = ['engine/trainer.py','models/yolo/classify/train.py','data/dataset.py',
                 'data/augment.py','data/build.py','utils/torch_utils.py','cfg/default.yaml']
    defaults = yaml.safe_load((package/'cfg/default.yaml').read_text(encoding='utf8'))
    records = []
    image = Image.fromarray(np.random.default_rng(0).integers(0,256,(224,224,3),dtype=np.uint8))
    for seed in [42,43,44,45,46]:
        work=ROOT/f'02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed{seed}'
        config=read(work/'reports/experiment_config.json')
        environment=read(work/'reports/environment.json')
        training_log=re.sub(r'\x1b\[[0-9;]*m','',(work/'reports/training.log').read_text(encoding='utf8'))
        trained=read(work/'runs/trained_models.json')
        assert config['training']==dict(research.base.CONFIG,seed=seed)
        assert environment['checkpoint']['sha256']==research.PINNED_SHA
        for arm,n in [('A',6),('B',8)]:
            run=work/f'runs/model_{arm}'
            execution=read(work/f'runs/model_{arm}_execution.json')
            args=yaml.safe_load((run/'args.yaml').read_text(encoding='utf8'))
            for key,value in config['training'].items():
                assert str(execution['actual_args'][key])==str(value), (seed,arm,key)
                assert str(args[key])==str(value), (seed,arm,key)
            counter=read(run/'optimizer_steps.json')
            table=pd.read_csv(run/'results.csv');table.columns=table.columns.str.strip()
            assert table.epoch.tolist()==list(range(1,31))
            steps=690 if arm=='A' else 900
            assert execution['actual_optimizer_steps']==counter['actual_optimizer_steps']==steps
            assert counter['epoch']==30 and counter['batches_per_epoch']==(23 if arm=='A' else 30)
            lr=table[['lr/pg0','lr/pg1','lr/pg2']].to_numpy()
            assert np.allclose(lr[-1],counter['learning_rates'])
            # Verify epoch-end LR against installed scheduler/warmup; this is reconstruction, not a batch trace.
            batches=counter['batches_per_epoch'];expected=[]
            for epoch in range(30):
                target=.01*((1-epoch/30)*.99+.01)
                ni=(epoch+1)*batches-1
                expected.append([float(np.interp(ni,[0,3*batches],[start,target]))
                                 if ni<=3*batches else target for start in [0.,0.,.1]])
            assert np.allclose(lr,expected,atol=6e-8), (seed,arm,'scheduler')
            path=run/'weights/last.pt'
            item=next(x for x in trained['models'] if x['model']==arm)
            assert research.base.b.sha(path)==item['sha256']
            assert item['epoch']==30 and item['initial_checkpoint_sha256']==research.PINNED_SHA
            ckpt=torch.load(path,map_location='cpu',weights_only=False)
            model=ckpt.get('ema') or ckpt.get('model')
            transform=model.transforms
            names=[type(t).__name__ for t in transform.transforms]
            assert names==['Resize','ToTensor','Normalize']
            first=transform(image);second=transform(image)
            reference=torch.from_numpy(np.asarray(image).copy().transpose(2,0,1)).float()/255
            assert torch.equal(first,second) and torch.equal(first,reference)
            assert len(model.names)==n and model.yaml['scale']=='n'
            assert ckpt['version']==environment['packages']['ultralytics']=='8.4.197'
            assert len(ckpt['train_results']['epoch'])==30
            assert ckpt['train_args']['seed']==seed
            assert 'optimizer: SGD' in training_log
            records.append(dict(seed=seed,arm=arm,work=str(work.relative_to(ROOT)),
                configured=config['training'],framework_defaults={k:defaults[k] for k in config['training'] if k in defaults},
                actual_args=execution['actual_args'],args_yaml_sha256=research.base.b.sha(run/'args.yaml'),
                initial_checkpoint_sha256=execution['initial_checkpoint_sha256'],
                checkpoint_sha256=item['sha256'],checkpoint_version=ckpt['version'],checkpoint_date=ckpt['date'],
                checkpoint_epoch=ckpt['epoch'],checkpoint_optimizer_is_none=ckpt['optimizer'] is None,
                checkpoint_result_epochs=ckpt['train_results']['epoch'],checkpoint_names=model.names,
                checkpoint_model_yaml=model.yaml,checkpoint_transform=str(transform),
                synthetic_transform_exact_rgb_div255=True,transform_repeats_identical=True,
                steps=steps,batches_per_epoch=batches,epoch30_learning_rates=lr[-1].tolist(),
                all_epoch_lr_matches_local_source_reconstruction=True,warmup_iterations_reconstructed=3*batches,
                environment=environment))
            del model,ckpt
    write('effective_configuration.json',dict(status='VERIFIED_WITH_LIMITS',created_utc=datetime.now(timezone.utc).isoformat(),
        packages={k:metadata.version(k) for k in ['ultralytics','torch','torchvision','numpy','pandas']},
        framework_source_sha256={f:research.base.b.sha(package/f) for f in framework},
        historical_framework_source_hash_available=False,
        evidence_limits=['No historical per-batch transform/RNG trace; training transform supported by locked trainer source plus checkpoint validation transform.',
                        'Local framework reconstruction matches all recorded epoch-end LRs; installed source hashes do not establish historical byte identity.',
                        'Stripped checkpoints contain epoch=-1 and optimizer=None; completion comes from 30 results rows, counters and trained_models, not epoch=-1.',
                        'No new model forward, optimizer step or inference. Deterministic settings are controls, not proof of bitwise reproducibility.'],
        runs=records))
    print('PASS: ten local checkpoints/args/counters/curves; CPU transform checks; no model execution')


if __name__=='__main__':
    main()
