"""exploratory/development Stage1b runner: fresh process for every Ultralytics work directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
PYTHON=ROOT/'.venv-bisindo/Scripts/python.exe'
TAG='exploratory_development_stage1b_user_20261103_v1'
SESSION=ROOT/'exploratory_development_stage1b_agent_execution_20261103_v2'
OLD_SESSION=ROOT/f'{TAG}_session'


def work(seed):return ROOT/f'{TAG}_seed{seed}'
def approval(seed):return OLD_SESSION/f'exploratory_approval_seed{seed}.json'
def probe_out(seed,model):
    suffix='_attempt2' if (seed,model)==(42,'A') else ''
    return ROOT/f'{TAG}_probe_{model}_seed{seed}{suffix}'


def worker(action,seed,model):
    import classification_research as r
    import classification_diagnostics as d
    import classification_workflow as w
    if action=='prepare':r.prepare(ROOT,ROOT/'experiment_yolo26_v3',work(seed),r.Settings(seed,30,224))
    elif action=='train':r.train(ROOT,ROOT/'experiment_yolo26_v3',work(seed),approval=approval(seed))
    elif action=='probe':d.fit_linear_probe(work(seed),model,probe_out(seed,model),approval=approval(seed))
    elif action=='masks':d.prepare_masks(work(42),ROOT/'experiment_yolo26_v3',ROOT/f'{TAG}_exploratory_masks')
    elif action=='masked':d.evaluate_masked(work(seed),ROOT/f'{TAG}_exploratory_masks',ROOT/f'{TAG}_masked_eval_seed{seed}')
    elif action=='paired':w.existing_analysis(work(seed),ROOT/f'{TAG}_paired_seed{seed}')
    elif action=='curves':w.plot_curves(work(seed),ROOT/f'{TAG}_curves_seed{seed}')
    elif action=='summary':w.summarize_stage1b(ROOT,TAG,ROOT/f'{TAG}_exploratory_multiseed')
    else:raise ValueError(action)


def load_table():
    import pandas as pd
    paths=sorted(SESSION.glob('exploratory_fit_table_*.csv'))
    return pd.read_csv(paths[-1],keep_default_na=False).to_dict('records')


def save(rows,event):
    import pandas as pd
    event=dict(label='exploratory/development',**event)
    with (SESSION/'exploratory_events.jsonl').open('a',encoding='utf8') as stream:
        stream.write(json.dumps(event)+'\n')
    version=len(list(SESSION.glob('exploratory_fit_table_*.csv')))
    with (SESSION/f'exploratory_fit_table_{version:03d}.csv').open('x',encoding='utf8',newline='') as stream:
        pd.DataFrame(rows).to_csv(stream,index=False)


def entry(rows,seed,kind,model):
    return next(row for row in rows if (row['seed'],row['kind'],row['model'])==(seed,kind,model))


def job(rows,action,seed=42,model='A'):
    path=SESSION/f'exploratory_{action}_seed{seed}_{model}.log'
    print(f'START {action} seed{seed} {model}',flush=True)
    started=time.perf_counter()
    with path.open('x',encoding='utf8') as stream:
        result=subprocess.run([str(PYTHON),'-u',str(Path(__file__).resolve()),'--worker',action,
                               '--seed',str(seed),'--model',model],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    duration=time.perf_counter()-started
    event=dict(action=action,seed=seed,model=model,seconds=duration,returncode=result.returncode,log=path.name)
    save(rows,event)
    print(f'END {action} seed{seed} {model}: {duration:.3f}s rc={result.returncode}',flush=True)
    if result.returncode:
        tail=path.read_text(encoding='utf8',errors='replace')[-3500:]
        raise RuntimeError(f'Hard stop; log={path}\n{tail}')
    return duration


def import_full(rows,seed):
    import pandas as pd
    import classification_research as r
    settings=r.load_settings(work(seed));r.validate_release(work(seed),settings)
    trained=r.base.read_json(work(seed)/'runs/trained_models.json')
    assert {item['model'] for item in trained['models']}=={'A','B'}
    metrics=pd.read_csv(work(seed)/'evaluation/comparison_metrics.csv')
    for model in ['A','B']:
        item=next(item for item in trained['models'] if item['model']==model)
        assert item['epoch']==30 and Path(item['path']).name=='last.pt'
        assert r.base.b.sha(item['path'])==item['sha256']
        curve=pd.read_csv(work(seed)/'runs'/f'model_{model}'/'results.csv')
        curve.columns=curve.columns.str.strip()
        assert curve.epoch.tolist()==list(range(1,31))
        selected=metrics[(metrics.model==model)&(metrics.scope=='common6')].iloc[0]
        row=entry(rows,seed,'full',model)
        row.update(status='completed',seconds=float(curve.time.iloc[-1]),macro_f1=float(selected.macro_f1),accuracy=float(selected.accuracy))
        if model=='B':
            extra=metrics[(metrics.model==model)&(metrics.scope=='full8')].iloc[0]
            row.update(full8_macro_f1=float(extra.macro_f1),full8_accuracy=float(extra.accuracy))


def probes(rows,seed):
    for model in ['A','B']:
        row=entry(rows,seed,'linear_probe',model)
        if row['status']!='not_started':raise RuntimeError('Probe attempted; no resume')
        row['status']='running';save(rows,dict(action='probe_started',seed=seed,model=model))
        try:
            seconds=job(rows,'probe',seed,model)
            result=json.loads((probe_out(seed,model)/'exploratory_probe_summary.json').read_text())
            row.update(status='completed',seconds=seconds,macro_f1=result['macro_f1'],accuracy=result['accuracy'])
            save(rows,dict(action='probe_completed',seed=seed,model=model))
        except BaseException as exc:
            row.update(status='failed',error=str(exc));save(rows,dict(action='hard_stop',seed=seed,model=model,error=str(exc)))
            raise


def first():
    SESSION.mkdir(exist_ok=False)
    rows=[dict(kind=kind,model=model,seed=seed,replication=seed==42,status='not_started',seconds='',
               macro_f1='',accuracy='',full8_macro_f1='',full8_accuracy='',error='')
          for seed in range(42,47) for kind in ['full','linear_probe'] for model in ['A','B']]
    old_events=[json.loads(line) for line in (OLD_SESSION/'exploratory_events.jsonl').read_text().splitlines()]
    failures=[event for event in old_events if event['event']=='hard_stop']
    # Preserve the failed attempt; it failed before model load/optimizer and contains no files.
    failed=ROOT/f'{TAG}_probe_A_seed42'
    assert failed.is_dir() and not any(path.is_file() for path in failed.rglob('*'))
    with (SESSION/'exploratory_previous_failures.json').open('x',encoding='utf8') as stream:json.dump(failures,stream,indent=2)
    entry(rows,42,'linear_probe','A')['error']='Attempt1: settings-directory conflict before model loading/optimization; preserved empty output. Attempt2 in new directory.'
    import_full(rows,42)
    save(rows,dict(action='existing_seed42_full_runs_verified_and_reused',new_full_fits=0))
    probes(rows,42)
    prior=next(event for event in old_events if event['event']=='phase_timing')
    timings=dict(prior_prepare_seconds=prior['prepare_seconds'],prior_pair_train_evaluation_seconds=prior['pair_train_and_evaluation_seconds'],
                 prior_active_phase_including_failure_seconds=prior['total_wall_seconds'],
                 A_recorded_training_seconds=entry(rows,42,'full','A')['seconds'],B_recorded_training_seconds=entry(rows,42,'full','B')['seconds'],
                 probe_A_fresh_process_seconds=entry(rows,42,'linear_probe','A')['seconds'],probe_B_fresh_process_seconds=entry(rows,42,'linear_probe','B')['seconds'])
    with (SESSION/'exploratory_first_phase_timing.json').open('x',encoding='utf8') as stream:json.dump(timings,stream,indent=2)
    print(json.dumps(timings),flush=True)


def remaining():
    rows=load_table()
    assert all(entry(rows,42,kind,model)['status']=='completed' for kind in ['full','linear_probe'] for model in ['A','B'])
    for seed in range(43,47):
        if work(seed).exists():raise RuntimeError('Existing work; no resume/overwrite')
        job(rows,'prepare',seed)
        for model in ['A','B']:entry(rows,seed,'full',model)['status']='running'
        save(rows,dict(action='pair_started',seed=seed))
        try:
            job(rows,'train',seed)
            import_full(rows,seed);save(rows,dict(action='pair_completed',seed=seed))
            probes(rows,seed)
        except BaseException as exc:
            for row in rows:
                if row['seed']==seed and row['status']=='running':row.update(status='failed',error=str(exc))
            save(rows,dict(action='hard_stop',seed=seed,error=str(exc)));raise


def diagnose():
    rows=load_table();assert len(rows)==20 and all(row['status']=='completed' for row in rows)
    job(rows,'masks')
    for seed in range(42,47):
        for action in ['masked','paired','curves']:job(rows,action,seed)
    job(rows,'summary')


def report():
    import pandas as pd
    import classification_research as r
    import classification_workflow as w
    rows=load_table();table=pd.DataFrame(rows)
    assert len(table)==20 and table.status.eq('completed').all()
    metrics=[];per_class=[];masked=[];head=[];curves=[];paired=[]
    for seed in range(42,47):
        metrics.append(pd.read_csv(work(seed)/'evaluation/comparison_metrics.csv').assign(seed=seed))
        per_class.append(pd.read_csv(work(seed)/'evaluation/per_class_metrics.csv').assign(seed=seed))
        masked.append(pd.read_csv(ROOT/f'{TAG}_masked_eval_seed{seed}'/'exploratory_masked_metrics.csv').assign(seed=seed))
        paired.append(pd.read_csv(ROOT/f'{TAG}_paired_seed{seed}'/'exploratory_paired_ci.csv').assign(seed=seed))
        for model in ['A','B']:
            item=json.loads((probe_out(seed,model)/'exploratory_probe_summary.json').read_text())
            head.append(dict(seed=seed,model=model,macro_f1=item['macro_f1'],accuracy=item['accuracy'],seconds=item['seconds']))
            training=pd.read_csv(work(seed)/'runs'/f'model_{model}'/'exploratory_train_curve.csv')
            validation=pd.read_csv(work(seed)/'runs'/f'model_{model}'/'results.csv')
            validation.columns=validation.columns.str.strip()
            assert training.epoch.tolist()==list(range(1,31))
            curves.append(dict(seed=seed,model=model,epoch30_train_top1_actual_weights=training.top1.iloc[-1],
                               epoch30_val_top1_EMA=validation['metrics/accuracy_top1'].iloc[-1]))
    metrics=pd.concat(metrics,ignore_index=True)
    per_class=pd.concat(per_class,ignore_index=True)
    masked=pd.concat(masked,ignore_index=True)
    paired=pd.concat(paired,ignore_index=True)
    head=pd.DataFrame(head);curves=pd.DataFrame(curves)
    summary=metrics.groupby(['model','scope']).agg(n_seeds=('seed','size'),macro_f1_mean=('macro_f1','mean'),
        macro_f1_sd=('macro_f1','std'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std')).reset_index()
    probes_summary=head.groupby('model').agg(n_seeds=('seed','size'),macro_f1_mean=('macro_f1','mean'),
        macro_f1_sd=('macro_f1','std'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std')).reset_index()
    masked_summary=masked[masked.stratum.eq('all_including_unavailable')].groupby(['model','scope']).agg(
        n_seeds=('seed','size'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
        chance=('balanced_task_chance','first')).reset_index()
    masked_summary['accuracy_minus_chance']=masked_summary.accuracy_mean-masked_summary.chance
    new_classes=per_class[(per_class.model=='B')&(per_class.scope=='full8')&(per_class.class_id>=6)]
    delta=paired[paired.metric.eq('macro_f1')].copy()
    multiseed=json.loads((ROOT/f'{TAG}_exploratory_multiseed'/'exploratory_multiseed_summary.json').read_text())
    events=[json.loads(line) for line in (SESSION/'exploratory_events.jsonl').read_text().splitlines()]
    failures=json.loads((SESSION/'exploratory_previous_failures.json').read_text())
    failures.append(dict(event='runner_preflight_failed_before_any_fit',error='Empty-directory assertion counted an empty settings subdirectory as a file; corrected check in v2; v1 session preserved.'))
    artifacts={'all_fits':table,'all_metrics':metrics,'model_summary':summary,'paired_per_seed':delta,
               'linear_probes':head,'probe_summary':probes_summary,'masking_all_seeds':masked,
               'masking_summary':masked_summary,'new_class_metrics':new_classes,'epoch30_curves':curves}
    for name,frame in artifacts.items():
        with (SESSION/f'exploratory_{name}.csv').open('x',encoding='utf8',newline='') as stream:frame.to_csv(stream,index=False)
    with (SESSION/'exploratory_failures.json').open('x',encoding='utf8') as stream:json.dump(failures,stream,indent=2)
    first=json.loads((SESSION/'exploratory_first_phase_timing.json').read_text())
    timing=pd.DataFrame([event for event in events if 'seconds' in event])
    body=('# exploratory/development — Tahap1b completed\n\n## FACT\n\n'
        'Five paired seeds42–46 retained; seed42 replication. Ten full fits total: two researcher-completed seed42 '
        'fits reused without retraining, eight additional full fits by agent. Ten completed new head fits. '
        'A failed zero-optimization probe attempt and preflight failure remain in failure history. '
        'Fresh processes fix per-work Ultralytics settings isolation without changing locked source. '
        'No overwrite/resume, external scoring, Kaggle tuning, threshold change or Stage2.\n\n'
        '## FACT — model mean ± SD (ddof1), internal validation only\n\n'+w.paired.markdown_table(summary)
        +'\n\n## FACT — paired delta and conditional per-seed CI\n\n'+w.paired.markdown_table(delta)
        +'\n\n## FACT — shared-video hierarchical bootstrap10000, RNG20261103\n\n```json\n'
        +json.dumps(multiseed,indent=2)+'\n```\n\n'
        'Pooled McNemar is descriptive only: repeated seed-video observations are dependent. '
        'Per-seed exact McNemar in paired reports remains separate.\n\n'
        '## FACT — frozen eval backbone, six-class last linear head\n\n'+w.paired.markdown_table(probes_summary)
        +'\n\n## FACT — hand masked vs uniform-output chance\n\n'+w.paired.markdown_table(masked_summary)
        +'\n\nAll video denominators retained, unavailable-landmark strata retained when present. '
        'Face/body may remain; imbalance/shift prevents a pure-background causal claim.\n\n'
        '## FACT — epoch30 curves\n\n'+w.paired.markdown_table(curves)
        +'\n\nTraining diagnostic uses actual weights; framework validation uses EMA. '
        'Their gap is descriptive, not an identically weighted-model comparison. Full per-epoch plots are saved for all seeds.\n\n'
        '## FACT — first-phase recorded timing\n\n```json\n'+json.dumps(first,indent=2)+'\n```\n\n'
        'Recorded A/B training excludes prepare and other pipeline phases. Probe child wall times include startup/extraction/evaluation. '
        'Prior active phase includes its failed attempt; interactive gap excluded. No estimated times substituted.\n\n'
        '## FACT — process phase timing\n\n'+w.paired.markdown_table(timing)
        +'\n\n## FACT — failures, retained\n\n```json\n'+json.dumps(failures,indent=2)+'\n```\n\n'
        '## HYPOTHESIS\n\nContext cues, weak labels and overfit are possible explanations. '
        'Masking/probes do not identify a causal linguistic mechanism. Signer/session independence unverified.\n\n'
        '## PROPOSAL\n\nSTOP after Stage1b. Candidate delta/H2 unchanged; internal pilot remains exploratory. '
        'If seed SD >=abs(mean paired delta), or interval crosses -delta, pilot cannot decide; do not chase seeds. '
        'Stage2 requires researcher decision, addendum locking delta/H2/domain, validated folds/grid/budget, and separate approval.\n')
    with (SESSION/'exploratory_stage1b_report.md').open('x',encoding='utf8') as stream:stream.write(body)
    provenance=dict(status='exploratory/development',source_sha256=r.settings_record(r.Settings())['source_sha256'],
                    runner_sha256=r.base.b.sha(__file__),candidate_sha256=r.base.b.sha(ROOT/'PREREGISTRATION.md'),
                    total_completed_full_fits=10,agent_new_full_fits=8,completed_head_fits=10,external_scores=0,
                    execution_authorization='Researcher 2026-11-03: kamu yang run aja; complete approved Stage1b, no Stage2.',
                    stop_before_stage2=True)
    assert provenance['candidate_sha256']==(ROOT/'PREREGISTRATION.sha256').read_text().split()[0]
    with (SESSION/'exploratory_provenance.json').open('x',encoding='utf8') as stream:json.dump(provenance,stream,indent=2)
    import nbformat
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Tahap1b results — exploratory/development\nRead-only results. Training executed in isolated worker processes; original failed notebook preserved.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nfrom IPython.display import display, Markdown\nreport=Path.cwd()/"+repr(SESSION.name)+"/'exploratory_stage1b_report.md'\ndisplay(Markdown(report.read_text(encoding='utf8')))\n")])
    nb.metadata['kernelspec']={'display_name':'.venv-bisindo','name':'python3','language':'python'}
    with (ROOT/'BISINDO_Classification_Stage1b_Results_Exploratory.ipynb').open('x',encoding='utf8') as stream:nbformat.write(nb,stream)
    print(summary.to_string(index=False),flush=True)
    print(json.dumps(multiseed),flush=True)
    print('REPORT',SESSION/'exploratory_stage1b_report.md',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',nargs='?',choices=['first','remaining','diagnostics','report'])
    parser.add_argument('--worker',choices=['prepare','train','probe','masks','masked','paired','curves','summary'])
    parser.add_argument('--seed',type=int,choices=range(42,47),default=42)
    parser.add_argument('--model',choices=['A','B'],default='A')
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:
        pids=[int(subprocess.check_output([str(PYTHON),'-c','import os; print(os.getpid())'],text=True)) for _ in range(2)]
        assert pids[0]!=pids[1]
        assert probe_out(42,'A')!=ROOT/f'{TAG}_probe_A_seed42'
        assert not SESSION.exists()
        print('PASS: fresh worker processes, new failed-probe output, no session mutation')
    elif args.worker:worker(args.worker,args.seed,args.model)
    elif args.phase=='first':first()
    elif args.phase=='remaining':remaining()
    elif args.phase=='diagnostics':diagnose()
    elif args.phase=='report':report()
    else:parser.error('Select a phase')


if __name__=='__main__':main()
