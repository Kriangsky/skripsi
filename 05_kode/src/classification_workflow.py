"""exploratory/development notebook functions. No fits without an explicit flag and approval files."""
from pathlib import Path
import time

import numpy as np
import pandas as pd

import analysis_paired as paired
import classification_research as research
import classification_diagnostics as diagnostics
import power_precision

SEEDS=tuple(range(42,47))


def seed_works(root,tag):
    if Path(tag).name!=tag or 'exploratory' not in tag.lower():raise ValueError('Use an exploratory-labelled simple work tag')
    return {seed:Path(root).resolve()/f'{tag}_seed{seed}' for seed in SEEDS}


def prepare_stage1b(root,tag):
    works=seed_works(root,tag)
    for path in works.values():research.new_work(root,path)
    return {seed:research.prepare(root,Path(root)/'experiment_yolo26_v3',path,research.Settings(seed,30,224))
            for seed,path in works.items()}


def train_stage1b(root,tag,approvals,run_training=False):
    if not run_training:return 'SKIPPED: RUN_TRAINING=False; zero full/head fits'
    works=seed_works(root,tag)
    # Validate all five scoped approvals before the first GPU/fit call.
    for seed,path in works.items():
        settings=research.load_settings(path)
        if settings!=research.Settings(seed,30,224):raise ValueError('Stage1b settings must be 224/30 seeds42..46')
        research.authorize_training(root,path,settings,approvals.get(seed))
        if research.base.read_json(approvals[seed]).get('linear_probes') is not True:
            raise PermissionError('Stage1b approval must explicitly include ten linear probe fits')
        if any((path/'runs'/f'model_{name}').exists() for name in ['A','B']):raise ValueError('Run exists; no resume/overwrite')
        for name in ['A','B']:
            if (Path(root)/f'{tag}_probe_{name}_seed{seed}').exists():raise ValueError('Probe exists; no resume/overwrite')
    for seed,path in works.items():
        research.train(root,Path(root)/'experiment_yolo26_v3',path,approval=approvals[seed])
        for name in ['A','B']:
            diagnostics.fit_linear_probe(path,name,Path(root)/f'{tag}_probe_{name}_seed{seed}',approval=approvals[seed])
    return 'exploratory/development: ten full fits and ten fixed last-head fits completed'


def existing_analysis(work,out,repetitions=10000):
    record=paired.analyze(work,out,repetitions)
    # Preserve the legacy analyzer's filenames/API; label new notebook outputs explicitly.
    for path in Path(out).iterdir():
        if path.is_file():path.rename(path.with_name('exploratory_'+path.name))
    report=Path(out)/'exploratory_paired_analysis.md'
    report.write_text(report.read_text(encoding='utf8').replace('# FACT — existing pilot paired analysis',
        '# exploratory/development — FACT: existing pilot paired analysis'),encoding='utf8')
    return record


def plot_curves(work,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=diagnostics.fresh_output(work,out);joined=[]
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for column,name in enumerate(['A','B']):
        folder=Path(work)/'runs'/f'model_{name}'
        validation=pd.read_csv(folder/'results.csv');validation.columns=validation.columns.str.strip()
        curve=folder/'exploratory_train_curve.csv'
        if curve.exists():
            training=pd.read_csv(curve)
            if training.epoch.tolist()!=validation.epoch.tolist():raise ValueError('Incomplete per-epoch train diagnostics')
            data=validation.merge(training,on='epoch',validate='one_to_one')
            axes[0,column].plot(data.epoch,data.top1,label='train actual weights; no augmentation')
            axes[1,column].plot(data.epoch,data.loss,label='train diagnostic CE; actual weights')
        else:
            data=validation
            axes[0,column].text(.1,.2,'Historical train top-1 unavailable',transform=axes[0,column].transAxes)
        accuracy=next((x for x in data if x in ['metrics/accuracy_top1','metrics/accuracy_top1(B)']),None)
        val_loss=next((x for x in data if x.startswith('val/') and 'loss' in x),None)
        if accuracy:axes[0,column].plot(data.epoch,data[accuracy],label='validation EMA (framework)')
        if val_loss:axes[1,column].plot(data.epoch,data[val_loss],label='validation loss (framework)')
        axes[0,column].set_title(f'exploratory: {name}, {6 if name=="A" else 8} classes')
        for axis in axes[:,column]:axis.set_xlabel('epoch');axis.legend()
        joined.append(data.assign(model=name,status='exploratory/development'))
    fig.tight_layout();fig.savefig(out/'exploratory_learning_curves.png',dpi=150);plt.close(fig)
    pd.concat(joined,ignore_index=True).to_csv(out/'exploratory_learning_curves.csv',index=False)
    return out


def summarize_stage1b(root,tag,out,repetitions=10000):
    works=seed_works(root,tag);pairs=[paired.load_pairs(path) for path in works.values()]
    reference=pairs[0][['sample_id','video','video_sha256','class_id']]
    for frame in pairs[1:]:
        if not frame[reference.columns].equals(reference):raise ValueError('Different paired videos across seeds')
    out=diagnostics.fresh_output(works[42],out)
    rng=np.random.default_rng(paired.RNG_SEED);y=reference.class_id.to_numpy()
    weights=np.zeros((repetitions,len(y)),dtype=np.int32)
    for cid in range(6):
        ids=np.flatnonzero(y==cid)
        weights[:,ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)),size=repetitions)
    deltas=[];rows=[]
    for seed,frame in zip(SEEDS,pairs):
        values=[];draws=[]
        for name,n in [('A',6),('B',8)]:
            p=frame[f'predicted_class_id_{name}'].to_numpy()
            stats,_,_=research.base.metrics(y.tolist(),p.tolist(),n,6)
            rows.append(dict(seed=seed,replication=seed==42,model=name,scope='common6',**stats))
            values.append(stats['macro_f1'])
            draws.append(paired.f1_from_matrix((weights@np.eye(48)[y*8+p]).reshape(-1,6,8)))
        deltas.append(draws[1]-draws[0])
        rows[-1]['paired_delta_B_minus_A']=values[1]-values[0]
    deltas=np.asarray(deltas)
    seed_counts=rng.multinomial(5,[.2]*5,size=repetitions)
    hierarchical=(deltas.T*seed_counts).sum(1)/5
    video_only=deltas.mean(0)
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory_all_seeds_common6.csv',index=False)
    summary=table.groupby('model').macro_f1.agg(['mean','std']).reset_index()
    mean_delta=float(table[table.model.eq('B')].paired_delta_B_minus_A.mean())
    point_deltas=table[table.model.eq('B')].paired_delta_B_minus_A.to_numpy()
    lower=float(np.quantile(hierarchical,.05))
    stop=bool(summary['std'].ge(abs(mean_delta)).any() or lower<=-.02<=float(np.quantile(hierarchical,.95)))
    a_only=sum(int((p.correct_A & ~p.correct_B).sum()) for p in pairs)
    b_only=sum(int((~p.correct_A & p.correct_B).sum()) for p in pairs)
    record=dict(status='exploratory/development',training_seeds=list(SEEDS),seed42_is_replication=True,
        video_n=len(y),mean_delta=mean_delta,ci95=np.quantile(hierarchical,[.025,.975]).tolist(),
        lower_one_sided95=lower,video_only_ci95=np.quantile(video_only,[.025,.975]).tolist(),
        paired_delta_seed_sd=float(point_deltas.std(ddof=1)),
        seed_only_ci95=np.quantile(seed_counts@point_deltas/5,[.025,.975]).tolist(),
        stop_cannot_decide=stop,proceed_to_stage2=False,
        pooled_discordant_seed_video_counts=dict(a_only=a_only,b_only=b_only),
        pooled_mcnemar_descriptive_only=paired.mcnemar_exact(a_only,b_only),
        warning='Pooled McNemar p is descriptive only: seed-video trials share videos, independence is false. No inferential pooled claim.',
        seed_variability=summary.to_dict('records'))
    research.base.dump(out/'exploratory_multiseed_summary.json',record)
    (out/'exploratory_multiseed_report.md').write_text('# exploratory/development — all five seeds\n\n'+paired.markdown_table(summary)+
        '\n\n'+str(record)+'\n\nFACT: all seed42..46 retained; seed42 replication. HYPOTHESIS: context/overfit explanations require diagnostics. '
        'PROPOSAL: request researcher decision; never auto-launch Stage2.\n',encoding='utf8')
    return record


def diagnose_stage1b(root,tag):
    works=seed_works(root,tag)
    mask_dir=Path(root)/f'{tag}_exploratory_masks'
    diagnostics.prepare_masks(works[42],Path(root)/'experiment_yolo26_v3',mask_dir)
    for seed,path in works.items():
        diagnostics.evaluate_masked(path,mask_dir,Path(root)/f'{tag}_masked_eval_seed{seed}')
        existing_analysis(path,Path(root)/f'{tag}_paired_seed{seed}')
        plot_curves(path,Path(root)/f'{tag}_curves_seed{seed}')
    return summarize_stage1b(root,tag,Path(root)/f'{tag}_exploratory_multiseed')


def inference_smoke(work,out):
    """Measured inference/embedding cost only. Zero backward calls, zero optimizer steps."""
    import torch
    work=Path(work)
    if (work/'research_config.json').exists():
        settings=research.load_settings(work);research.validate_release(work,settings)
        size=settings.imgsz
    else:
        research.base.validate_release(work)
        size=research.base.read_json(work/'reports/experiment_config.json')['training']['imgsz']
    out=diagnostics.fresh_output(work,out);rows=[]
    for name in ['A','B']:
        model,linear,item=diagnostics.load_frozen_model(work,name,out)
        loader=diagnostics.image_loader(Path(work)/'datasets/A/train',size)
        images,labels=next(iter(loader));images=images.to('cuda')
        with torch.inference_mode():model(images)  # one untimed warmup
        torch.cuda.synchronize();started=time.perf_counter()
        with torch.inference_mode():
            for _ in range(10):model(images)
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        rows.append(dict(model=name,imgsz=size,batch=16,repeats=10,seconds=seconds,seconds_per_batch=seconds/10,
            weights_sha256=item['sha256'],model_fits=0,optimizer_steps=0,status='exploratory/development',
            scope='resident FP32 GPU batch forward only; excludes decode, IO, backward, optimizer, head fitting'))
        del model;torch.cuda.empty_cache()
    table=pd.DataFrame(rows);table.to_csv(out/'exploratory_inference_smoke.csv',index=False)
    timing=[]
    for name in ['A','B']:
        path=Path(work)/'runs'/f'model_{name}'/'results.csv'
        data=pd.read_csv(path);data.columns=data.columns.str.strip()
        timing.append(dict(model=name,epochs=len(data),recorded_training_seconds=float(data.time.iloc[-1]),
                           source_sha256=research.base.b.sha(path)))
    research.base.dump(out/'exploratory_timing_limits.json',dict(historical=timing,
        five_pairs_historical_training_seconds=5*sum(x['recorded_training_seconds'] for x in timing),
        head_fit_seconds=None,new_instrumented_training_seconds=None,
        explanation='No training smoke permitted. Inference smoke cannot estimate backward/SGD or head-fitting duration. '
                    'A measured Stage1b total must wait for explicitly approved training smoke; historical training is only a baseline.'))
    return table
