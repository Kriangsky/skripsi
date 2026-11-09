"""Read saved evidence only; export consultation package. No ML imports or execution."""
from pathlib import Path
import csv, hashlib, io, json, math, shutil, statistics, subprocess, zipfile
from datetime import datetime, timezone

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
V = 'VERIFIED_FROM_ARTIFACT'
R = 'REPORTED_NOT_INDEPENDENTLY_VERIFIED'
S = '01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2'
D = '01_hasil/addendum/development_interference_20261105'
H = '02_exp/arsip_pipeline/experiment_yolo26_pilot'
T1 = '01_hasil/addendum/data_inventory_task1_20261105'
T2 = '01_hasil/addendum/scientific_decisions_task2_20261109'
OLD = '03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_exploratory_multiseed/exploratory_multiseed_summary.json'
sources = {}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()
def track(s):
    p=ROOT/s
    if s not in sources: sources[s]=(sha(p),p.stat().st_size)
    return p
def read(s): return track(s).read_text(encoding='utf-8-sig')
def js(s): return json.loads(read(s))
def rows(s): return list(csv.DictReader(io.StringIO(read(s))))
def write(s,t): (OUT/s).write_text(t,encoding='utf-8')
def dump(s,d): write(s,json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def csvout(s,cols,data):
    with (OUT/s).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(data)
def table(rs, cols):
    return '\n'.join(['| '+' | '.join(cols)+' |','| '+' | '.join(['---']*len(cols))+' |']+['| '+' | '.join(str(r.get(c,'')) for c in cols)+' |' for r in rs])+'\n'
def near(a,b): assert math.isclose(float(a),float(b),abs_tol=1e-12), (a,b)

# Preserve source/code/protocol/notebook fingerprints without rescanning raw datasets.
protected={}
for dirname in ['05_kode/src','05_kode/tests','06_notebook','07_dokumen/protokol']:
    for p in (ROOT/dirname).rglob('*'):
        if p.is_file() and p.suffix in {'.py','.ipynb','.json','.md','.csv','.sha256'}:
            protected[p.relative_to(ROOT).as_posix()]=sha(p)
git_before=subprocess.run(['git','diff','--binary'],cwd=ROOT,capture_output=True,check=True).stdout

audit=js(T1+'/audit.json'); primary=rows(D+'/common6_primary_results.csv'); mean=primary[-1]
details=js(D+'/analysis_details.json'); old=js(OLD); effective=js(T2+'/effective_configuration.json')
human=rows(H+'/analysis/pilot_summary.csv'); humanclass=rows(H+'/analysis/per_class_results.csv'); agreement=js(H+'/analysis/inter_reviewer_agreement.json')
for i,summary in enumerate(human,1):
    cm=rows(H+f'/analysis/reviewer_{i}_confusion.csv')
    n=sum(sum(int(v) for k,v in r.items() if k) for r in cm)
    correct=sum(int(r[r['']]) for r in cm)
    assert n==160 and correct==int(summary['correct_n'])
    near(correct/n,summary['accuracy_all'])
# Agreement recomputed from two anonymized response exports, never copy private manifests.
responses=[{r['sample_id']:r['decision'] for r in js(H+f'/responses/reviewer_{i}.json')['responses']} for i in (1,2)]
assert set(responses[0])==set(responses[1]) and len(responses[0])==160
obs=sum(responses[0][k]==responses[1][k] for k in responses[0])/160
expected=sum(sum(v==c for v in responses[0].values())*sum(v==c for v in responses[1].values()) for c in agreement['categories'])/160**2
near(obs,agreement['agreement']);near((obs-expected)/(1-expected),agreement['cohens_kappa'])

inventory=[]; metrics=[]; ten=[]; checkpoints=[]
def inv(id,stage,task='',model='',seed='',data='',split='',n='',status='VERIFIED',checkpoint='',metric='',config='',evidence='',notes=''):
    inventory.append(dict(zip(['experiment_id','stage','task_type','model','seed','data_source','split','sample_count','run_status','checkpoint_path','metrics_path','config_path','evidence_path','notes'],[id,stage,task,model,seed,data,split,n,status,checkpoint,metric,config,evidence,notes])))
def metric(exp,model,seed,split,name,value,n,source,status=V):
    metrics.append(dict(experiment=exp,model=model,seed=seed,split=split,metric=name,value=value,denominator=n,source_file=source,evidence_status=status))
for r in human:
    for k in ['correct_n','accuracy_all','ambiguous_n','invalid_n','wrong_class_readable_n','technical_coverage','accuracy_all_ci95_low','accuracy_all_ci95_high']:
        metric('human_single_frame',r['reviewer_id'],42,'DEVELOPMENT_internal_train',k,r[k],160,H+'/analysis/pilot_summary.csv')
for r in humanclass:
    metric('human_single_frame',r['reviewer_id'],42,'DEVELOPMENT_internal_train',r['class_name']+'_accuracy',r['accuracy_all'],20,H+'/analysis/per_class_results.csv')
for k in ['agreement','cohens_kappa']:
    metric('human_single_frame','two_reviewers',42,'DEVELOPMENT_internal_train',k,agreement[k],160,H+'/analysis/inter_reviewer_agreement.json')
inv('human_single_frame','DEVELOPMENT','human review','two reviewer exports',42,'internal','train',160,metric=H+'/analysis/pilot_summary.csv',evidence=H+'/analysis/pilot_report.md',notes='20/class; expertise and independent persons not authenticated')

fits=rows(S+'/exploratory_all_fits.csv'); allmetric=rows(S+'/exploratory_all_metrics.csv')
for seed in range(42,47):
    work=f'02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed{seed}'
    comparison=rows(work+'/evaluation/comparison_metrics.csv'); pc=rows(work+'/evaluation/per_class_metrics.csv'); predictions=rows(work+'/evaluation/predictions.csv')
    models=js(work+'/runs/trained_models.json')['models']; cfg=js(work+'/reports/experiment_config.json')
    manifest=rows(work+'/manifest.csv')
    for arm in ['A','B']:
        rec=next(r for r in models if r['model']==arm)
        ck=work+f'/runs/model_{arm}/weights/last.pt'; digest=sha(track(ck));assert digest==rec['sha256']
        execution=js(work+f'/runs/model_{arm}_execution.json'); args=execution['actual_args']
        assert args['task']=='classify' and args['seed']==seed and args['epochs']==30 and args['optimizer']=='SGD' and args['imgsz']==224 and args['batch']==16
        assert execution['actual_optimizer_steps']==(690 if arm=='A' else 900)
        curve=rows(work+f'/runs/model_{arm}/results.csv'); assert len(curve)==30
        for r in comparison:
            if r['model']!=arm: continue
            k=6 if r['scope']=='common6' else 8
            subset=[p for p in predictions if p['model']==arm and int(p['true_class_id'])<k]
            assert len(subset)==int(r['n'])
            correct=sum(p['true_class_id']==p['predicted_class_id'] for p in subset)
            near(correct/len(subset),r['accuracy'])
            f1s=[]
            for c in range(k):
                tp=sum(int(p['true_class_id'])==c and int(p['predicted_class_id'])==c for p in subset)
                fp=sum(int(p['true_class_id'])!=c and int(p['predicted_class_id'])==c for p in subset)
                fn=sum(int(p['true_class_id'])==c and int(p['predicted_class_id'])!=c for p in subset)
                saved=next(x for x in pc if x['model']==arm and x['scope']==r['scope'] and int(x['class_id'])==c)
                assert (tp,fp,fn)==tuple(int(saved[x]) for x in ['tp','fp','fn'])
                near(2*tp/(2*tp+fp+fn),saved['f1']);f1s.append(float(saved['f1']))
            near(statistics.mean(f1s),r['macro_f1'])
            for key in ['accuracy','macro_f1','correct_n','extra_class_predictions_n']:
                metric('stage1b',arm,seed,'DEVELOPMENT_'+r['scope'],key,r[key],r['n'],work+'/evaluation/comparison_metrics.csv')
            # No separate new measurement: arithmetic means of saved per-class metrics.
            cls=[x for x in pc if x['model']==arm and x['scope']==r['scope'] and int(x['class_id'])<k]
            near(statistics.mean(float(x['recall']) for x in cls),r['accuracy'])
            for key in ['precision','recall']:
                metric('stage1b',arm,seed,'DEVELOPMENT_'+r['scope'],'macro_'+key,statistics.mean(float(x[key]) for x in cls),r['n'],work+'/evaluation/per_class_metrics.csv')
            for x in cls:
                for key in ['precision','recall','f1','tp','fp','fn']:
                    metric('stage1b',arm,seed,'DEVELOPMENT_'+r['scope'],x['class_name']+'_'+key,x[key],x['n'],work+'/evaluation/per_class_metrics.csv')
        common=next(r for r in comparison if r['model']==arm and r['scope']=='common6')
        full=next((r for r in comparison if r['model']==arm and r['scope']=='full8'),{})
        pcs=[r for r in pc if r['model']==arm and r['scope']=='common6' and int(r['class_id'])<6]
        fit=next(r for r in fits if r['kind']=='full' and r['model']==arm and int(r['seed'])==seed)
        er=next(r for r in effective['runs'] if r['seed']==seed and r['arm']==arm)
        assert er['checkpoint_sha256']==digest and er['checkpoint_result_epochs']==list(range(1,31))
        ten.append(dict(model=arm,seed=seed,train=360 if arm=='A' else 480,val=120 if arm=='A' else 160,steps=execution['actual_optimizer_steps'],seconds=fit['seconds'],accuracy=common['accuracy'],macro_precision=statistics.mean(float(x['precision']) for x in pcs),macro_recall=statistics.mean(float(x['recall']) for x in pcs),macro_F1=common['macro_f1'],full8_F1=full.get('macro_f1','N/A'),source=work+'/evaluation/comparison_metrics.csv'))
        checkpoints.append(dict(model=arm,seed=seed,path=ck,sha256=digest,initial_sha256=execution['initial_checkpoint_sha256'],epoch_metadata=er['checkpoint_epoch'],epochs_evidenced=30,optimizer_none=er['checkpoint_optimizer_is_none'],metadata_source=T2+'/effective_configuration.json',metadata_status=R,hash_status=V))
        inv(f'stage1b_{arm}_{seed}','DEVELOPMENT','classification',arm,seed,'internal','train/val',f'{360 if arm=="A" else 480}/{120 if arm=="A" else 160}',checkpoint=ck,metric=work+'/evaluation/comparison_metrics.csv',config=work+'/reports/experiment_config.json',evidence=work+'/runs/trained_models.json',notes='Fresh independent ImageNet init; last30 EMA reported; seed42 reused, not extra independent seed')
        metric('stage1b',arm,seed,'DEVELOPMENT_train','optimizer_steps',execution['actual_optimizer_steps'],360 if arm=='A' else 480,work+f'/runs/model_{arm}_execution.json')
        metric('stage1b',arm,seed,'DEVELOPMENT_train','recorded_training_seconds',fit['seconds'],30,S+'/exploratory_all_fits.csv')
    # paired identity from saved prediction IDs and GT; image-byte pairing remains Task1 historical evidence.
    a={p['sample_id']:p['true_class_id'] for p in predictions if p['model']=='A'}
    b={p['sample_id']:p['true_class_id'] for p in predictions if p['model']=='B' and int(p['true_class_id'])<6}
    assert a==b and len(a)==120
    pr=next(r for r in primary if r['seed']==str(seed))
    near(float(pr['macro_f1_B'])-float(pr['macro_f1_A']),pr['delta'])
near(statistics.mean(float(r['delta']) for r in primary[:-1]),mean['delta']);near(statistics.stdev(float(r['delta']) for r in primary[:-1]),details['seed_delta_sd'])
assert len(ten)==10
dump('CHECKPOINT_HASHES.json',checkpoints)
csvout('STAGE1B_TEN_RUNS.csv',list(ten[0]),ten)

for r in primary:
    for key in ['macro_f1_A','macro_f1_B','delta','ci95_low','ci95_high','lower_one_sided95','upper_one_sided95','bootstrap_se']:
        if r[key]!='': metric('paired_AB','B_minus_A',r['seed'],'DEVELOPMENT_common6',key,r[key],120,D+'/common6_primary_results.csv')
metric('paired_AB','B_minus_A','42-46','DEVELOPMENT_common6','seed_delta_sd',details['seed_delta_sd'],'5 paired seeds;120 shared videos',D+'/analysis_details.json')
for k in ['mean_delta','lower_one_sided95','paired_delta_seed_sd']:
    metric('paired_AB_historical','B_minus_A','42-46','DEVELOPMENT_common6',k,old[k],120,OLD)
for k in ['ci95','video_only_ci95','seed_only_ci95']:
    for j,bound in enumerate(['low','high']):metric('paired_AB_historical','B_minus_A','42-46','DEVELOPMENT_common6',k+'_'+bound,old[k][j],120,OLD)

interference=rows(D+'/class_interference.csv'); intrusion=rows(D+'/intrusion_rates.csv'); decomposition=rows(D+'/accuracy_decomposition.csv')
for filename,rs in [('class_interference.csv',interference),('intrusion_rates.csv',intrusion),('accuracy_decomposition.csv',decomposition),('full8_results.csv',rows(D+'/full8_results.csv')),('bootstrap_intervals.csv',rows(D+'/bootstrap_intervals.csv'))]:
    for r in rs:
        for key,val in r.items():
            if key in ['seed','class_id','class_name','n','model','scope','metric','evidence_tier'] or val=='':continue
            try: float(val)
            except (ValueError,TypeError):continue
            if key in ['n_videos','n_seeds']: continue
            identity=r.get('class_name',r.get('endpoint',r.get('metric','')))
            if r.get('method'): identity+='_'+r['method']
            denom=r.get('n',r.get('n_videos',120))
            if filename=='full8_results.csv' and key in ['accuracy','macro_f1']: denom=160
            full8=filename=='full8_results.csv' or r.get('endpoint','').startswith('H2_')
            metric('interference_'+filename[:-4],r.get('model','B' if full8 else 'B_minus_A'),r.get('seed','42-46'),'DEVELOPMENT_'+('full8' if full8 else 'common6'),identity+'_'+key,val,denom,D+'/'+filename)

for filename in ['exploratory_model_summary.csv','exploratory_masking_summary.csv','exploratory_masking_all_seeds.csv','exploratory_probe_summary.csv','exploratory_linear_probes.csv','exploratory_new_class_metrics.csv','exploratory_epoch30_curves.csv']:
    for r in rows(S+'/'+filename):
        for key,val in r.items():
            if key in ['seed','model','scope','n','n_seeds','class_id','class_name','stratum'] or not val:continue
            try:float(val)
            except ValueError:continue
            metric(filename[:-4],r.get('model',''),r.get('seed','mean_42-46'),'DEVELOPMENT_'+r.get('scope','common6'),r.get('class_name',r.get('stratum',''))+'_'+key,val,r.get('n','120 shared videos;5 seeds' if r.get('scope')!='full8' else '160 shared videos;5 seeds'),S+'/'+filename)
for seed in range(42,47):
    for arm in ['A','B']:
        probe=next(p for p in (ROOT/'02_exp/tahap1b/linear_probes').glob(f'*probe_{arm}_seed{seed}*/exploratory_probe_summary.json'))
        pr=probe.relative_to(ROOT).as_posix();p=js(pr)
        inv(f'probe_{arm}_{seed}','DEVELOPMENT','frozen-backbone six-class head',arm,seed,'internal','common6 train/val','360/120',checkpoint=next(x['path'] for x in checkpoints if x['model']==arm and x['seed']==seed),metric=pr,evidence=pr,notes='30 head epochs; saved summary read; no inference replay; backbone eval mode')
    masked=f'03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed{seed}'
    inv(f'hand_mask_{seed}','DEVELOPMENT','saved masking diagnostic','A/B',seed,'internal','common6/full8','120/160',metric=masked+'/exploratory_masked_metrics.csv',evidence=masked+'/exploratory_masked_report.md',notes='Existing trained last.pt; face/body remain; not background-only causal experiment')
    read(masked+'/exploratory_masked_metrics.csv');read(masked+'/exploratory_masked_report.md')
for name in ['experiment_yolo26_classification_pilot','experiment_yolo26_classification_notebook_20261102']:
    work='02_exp/arsip_pipeline/'+name
    rs=rows(work+'/evaluation/comparison_metrics.csv')
    for r in rs:
        for key in ['accuracy','macro_f1']:metric(name,r['model'],42,'DEVELOPMENT_'+r['scope'],key,r[key],r['n'],work+'/evaluation/comparison_metrics.csv')
    inv(name,'DEVELOPMENT','classification','A/B',42,'internal','common6/full8','120/160',metric=work+'/evaluation/comparison_metrics.csv',evidence=work+'/pilot_conclusion.md',notes='Historical seed42 pilot/reproduction; same metrics; not extra seeds')
    read(work+'/pilot_conclusion.md')
inv('detection_training_pilot','HISTORICAL','detection','YOLO26',data='internal',status='ATTEMPTED',evidence='02_exp/arsip_pipeline/experiment_yolo26_training_pilot/pilot_conclusion.md',notes='Preparation/readiness attempted; NOT TRAINED, 0 verified positives; no completed detection comparison')
read(inventory[-1]['evidence_path'])
inv('paired_AB_statistics','DEVELOPMENT','saved paired analysis','A/B','42-46','internal','common6',120,metric=D+'/common6_primary_results.csv',evidence=D+'/analysis_details.json',notes='Existing 10000 bootstrap draws summarized; no bootstrap rerun')
power='03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power'
for name in ['power_precision_report.md','h1_effect_sensitivity.md','h1_power_scenarios.csv','h2_power_scenarios.csv','simulation_provenance.json']:read(power+'/'+name)
inv('stage1a_power_simulation','PLANNING_SIMULATION','assumption-based Monte Carlo',seed=20261103,split='hypothetical external',n='300 common6;50/new class',status='COMPLETED',metric=power+'/h1_power_scenarios.csv',evidence=power+'/power_precision_report.md',notes='10000/scenario; simulated power, never measured external performance; not copied into observed results table')
for id,task,n in [('final_AB_five_seeds','classification','A1440/360;B1920/480'),('external_Kaggle','classification','300/400'),('CV_grid_105fits','historical search plan',''),('factorial_H3','historical factorial plan',''),('ROI_only','ROI-only comparator','')]:
    inv(id,'FINAL_PLANNED' if id!='ROI_only' else 'UNESTABLISHED',task,'YOLO26n-cls' if id!='ROI_only' else '',data='external Kaggle' if id=='external_Kaggle' else 'internal',n=n,status='PLANNED',config=T2+'/candidate_config.json' if id=='final_AB_five_seeds' else '',evidence='07_dokumen/protokol/analysis_plan.json',notes='BLOCKED; not executed' if id!='ROI_only' else 'UNKNOWN: no completed ROI-only result located in scoped current records; no score claimed')

brief_speech='''Penelitian awal saya membahas perluasan training set untuk pengenalan frasa BISINDO real-time. Namun eksperimen yang tersedia menggunakan satu frame per video dan menghasilkan satu label gambar. Karena itu saya mengusulkan fokus frame-based Classification, bukan klaim pemahaman gerakan temporal atau real-time. Pertanyaan utamanya: apakah perluasan dari enam menjadi delapan kelas mengubah kemampuan mengenali enam kelas awal, dan kelas mana yang terdampak? Dataset lokal mencakup 2.400 video internal dan 400 eksternal; audit terdahulu mencatat kecocokan file dan hash, tetapi validitas semantik serta identitas signer belum lengkap. Pilot dua reviewer pada 160 frame memperoleh 80,625 dan 86,25 persen; apa_kabar paling sulit, hanya 45 dan 70 persen. Sepuluh run development YOLO26n Classification telah selesai pada lima pasangan seed. Mean macro-F1 common6 adalah 0,93990 untuk A dan 0,93117 untuk B; selisih minus 0,00873 dengan interval 95 persen minus 0,03022 sampai plus 0,00761. Arah penurunan belum pasti dan ini bukan bukti kesetaraan. Masking tangan masih menyisakan akurasi tinggi, sehingga shortcut perlu menjadi keterbatasan. Efek yang diukur mencakup tambahan vocabulary, data, dan langkah optimisasi sekaligus. Training final dan scoring eksternal belum dilakukan. Saya meminta persetujuan perubahan metode, judul dan pertanyaan, analisis estimasi versus non-inferiority, serta cakupan final dan syarat provenance sebelum melanjutkan.'''
brief='''# Brief konsultasi pembimbing — 9 November 2026

**Status: hasil DEVELOPMENT tersedia; eksperimen final dan evaluasi eksternal BLOCKED.**

**Judul lama:** *Effect of Training Set Expansion on YOLOv26 Performance for Real-Time BISINDO Phrase Recognition*.

**Usulan judul:** *Impact of Vocabulary Expansion on Frame-Based BISINDO Recognition: A Controlled YOLO26 Study*.

## Arah dan tujuan

Satu frame/video dengan satu label gambar lebih sesuai dengan Classification. Detection memerlukan endpoint lokalisasi; pengenalan temporal memerlukan urutan. Belum ada pembuktian Classification lebih unggul daripada Detection, dan perubahan metode belum mendapat bukti persetujuan akademik.

RQ: Apakah ekspansi vocabulary enam menjadi delapan kelas mengubah performa frame-level pada enam kelas awal, dan kelas mana yang paling terdampak? A dilatih independen pada saya, kamu, senang, bingung, marah, apa_kabar; B menambahkan halo dan tidur. Efek merupakan paket vocabulary **dan data/steps/head**, bukan efek kausal jumlah kelas saja.

## Data dan hasil yang sudah tersedia

| Bukti | Temuan | Batas |
| --- | --- | --- |
| TASK1 audit.json | 2.400 internal, 400 eksternal; train/val 1.920/480; 300/400 eksternal common6/full8 | Receipt audit dibaca; 45.250 checks tidak diulang |
| Human pilot, internal train | 160 frame, dua ekspor reviewer: 129/160 (80,625%) dan 138/160 (86,25%); kappa 0,82516 | Bukan adjudikasi linguistik bersertifikat |
| apa_kabar manusia | 9/20 dan 14/20 benar (45%/70%) | Sampel berbeda dari validation model |
| Stage1b, seluruh lima seed | A common6 macro-F1 0,93990 ± 0,01158; B 0,93117 ± 0,00548 | SD antarseed; 120 video yang sama |
| Paired Δ B−A | −0,0087293; CI95% [−0,0302228; +0,0076121]; SD Δ 0,0158984 | Development, bukan konfirmatori/eksternal |
| B full8 | Mean accuracy 0,9375; macro-F1 0,93619 | 160 video internal |
| Interferensi | Recall saya −3pp, marah −3pp, senang −1pp; tiga kelas lain 0 | Tidak menetapkan kelas paling terdampak secara inferensial |
| Masking tangan | Accuracy A common6 0,9200; B common6 0,91667 | Wajah/tubuh dapat tersisa; bukan bukti background-only |

Sumber ringkas: `VERIFIED_EXPERIMENT_RESULTS.md` §2–9 dan `EVIDENCE_MANIFEST.json`. Akurasi/F1 Stage1b dicek ulang secara aritmetika dari prediksi tersimpan; CI hanya dicocokkan dengan output statistik yang tersimpan. Tidak ada scoring baru. CI historis lain [−0,0301015; +0,0076679] juga dipertahankan dalam laporan.

## Status dan keputusan pembimbing

TASK0–2 menyelesaikan infrastruktur/audit/usulan, **bukan skripsi final**. Controller notebook tersedia. Final frame manifests masih null; semantic review, signer/session, lisensi dan versi/arsip Kaggle serta exposure attestation belum lengkap. Dataset eksternal pernah dipreprocess. NI margin belum memiliki justifikasi; H1 estimasi masih usulan. Full historical regression pernah 91/92; bukan seluruh tes hijau.

Mohon lima keputusan: (1) Classification frame-based menggantikan Detection; (2) judul/RQ dan batas klaim; (3) estimasi Δ+CI atau NI dengan margin substantif, mempertahankan H2 asli; (4) final A/B lima seed pada full locked data, equal epochs dan nasib CV/grid/faktorial lama; (5) persyaratan semantic review, provenance/exposure dan kelayakan klaim eksternal. Semua perlu catatan persetujuan nyata sebelum pelaksanaan.

## Narasi siap ucap ±90 detik

'''+brief_speech+'\n'
write('SUPERVISOR_PROGRESS_BRIEF.md',brief)

report='''# Verified experiment results — consultation 2026-11-09

## 1. Executive scientific summary

Snapshot lokal read-only; hanya menulis folder ekspor baru. Semua hasil model terukur di sini adalah **DEVELOPMENT internal**, bukan final/external. Sepuluh full Stage1b fits dan sepuluh frozen-backbone head fits tersedia. A common6 mean macro-F1=0,9398997391; B=0,9311704349. Mean paired Δ=−0,008729304229932344; CI95% terbaru [−0,030222818611814216; +0,007612139862988273]. CI mencakup nol: tidak menetapkan superiority, equivalence, non-inferiority, atau kepastian penurunan. Menambah halo/tidur masih relevan untuk meneliti trade-off kelas, bukan sudah terbukti memperbaiki diskriminasi.

**Evidence levels:** VERIFIED_FROM_ARTIFACT = saved numerical output dibaca dan cross-check terbatas (prediksi/metrik dihitung aritmetika; CI tidak direplikasi). REPORTED_NOT_INDEPENDENTLY_VERIFIED = laporan/receipt dibaca tetapi operasi aslinya tidak diulang. PLANNED_NOT_EXECUTED = rencana tanpa hasil. BLOCKED = prasyarat belum terpenuhi. UNKNOWN = bukti kurang. Hash membuktikan identitas byte, bukan kebenaran semantik atau execution history lengkap. Historical PASS tidak otomatis verifikasi independen.

## 2. Data inventory and provenance

'''
report+=table(audit['class_split_counts'],['source','split','class_name','n'])
report+='''
TASK1 `audit.json: observed` mencatat 2.400 internal, 400 external, train/val1.920/480, external common6/full8=300/400, 2.800 physical verified videos. Status angka receipt: VERIFIED_FROM_ARTIFACT sebagai isi saved audit; physical integrity saat ini **REPORTED_NOT_INDEPENDENTLY_VERIFIED** karena 45.250 checks tidak diulang. `class_split_counts` di atas berasal langsung JSON; per kelas internal300/train240/val60/external50. Split terkunci dengan SHA256 `03eb34d91c6764d14052d2d8f99f6996d5b54602d50d533b47b9246ad32f706e` menurut receipt. Lihat `audit_report.md` §1–5 dan `duplicate_impact.json`.

Stored audit: 0 duplicate video-byte groups; tiga pasangan candidate JPEG/RGB identik f7/f10/f14 pada dua video senang_223/224, keduanya TRAIN. Development hanya memakai224/f48, bukan frame duplikat. Tidak ada cross-split decoded identity pada scope audit. Tidak membuktikan tidak ada near-duplicate, overlap frame lain, signer leakage. Tidak menghapus/memindah data.

Kaggle citation lokal menunjuk BISINDO40 Kata mp4, bukan acquisition receipt. Versi, hash arsip asal, log unduh, subset/rename provenance, lisensi belum terikat ke400 file. Signer/session internal0/2400 dan external0/400 nonempty. Preprocessing eksternal v2/v3 sudah terjadi; kelengkapan riwayat scoring/tuning belum diattest. Jangan menyebut untouched atau signer-independent. `provenance.json` dan laporan TASK1 menyimpan bukti dokumenter, bukan autentikasi asal dataset.

## 3. Preprocessing and frame-selection history

TASK1 `frame_provenance`: 2.800 cache records, 33.250 kandidat; v3 positive-selected0, seluruh2.800 needs_review. Primary fractions20–80%, fallback5/10/15/85/90/95%. Pemilih teknis `pilot_study.choose_candidate` mempertimbangkan readable/flags/focus/frame index/candidate ID, bukan output model; tidak membuktikan representasi gesture valid. Development640 frame: semantic not_assessed640, flagged497, eligible143. Semua retained.

Full-frame aspect-preserving letterbox224×224, padding gray114, BGR→RGB, RGB/255; tidak crop/flip. Override ControlledTrainer mengganti Compose Classification default. Task2 effective configuration mencatat Resize224/ToTensor/identity Normalize; tidak ada augmentasi stokastik. `augment=False` saja tidak cukup sebagai bukti. Config/execution/source dibaca; transform per batch historis dan hash framework historis tidak tersedia. Tidak mereplay transform atau model dalam snapshot.

Human pilot berbeda: 160 internal training-pool frame, seed42, 20/kelas, technical eligible43/160 (26,875%), inspectable160/160. Ini bukan pemilihan ulang frame berdasarkan hasil review. Full-data final manifests internal/external masih null (`analysis_plan.json`, audit). Development manifests tidak dipromosikan final.

## 4. Human pilot

'''
report+=table(human,['reviewer_id','n','correct_n','accuracy_all','ambiguous_n','invalid_n','wrong_class_readable_n','accuracy_all_ci95_low','accuracy_all_ci95_high'])
report+='\nPer kelas, denominator20 setiap reviewer; source `'+H+'/analysis/per_class_results.csv`, row reviewer_id/class_name, columns correct_n/accuracy_all.\n\n'
report+=table(humanclass,['reviewer_id','class_name','n','correct_n','accuracy_all'])
report+='''
Agreement135/160=0,84375; expected agreement0,106328125; Cohen kappa0,8251595419179998, dihitung ulang dari paired sample IDs dua responses JSON dan cocok `inter_reviewer_agreement.json` keys. Confusion CSV diagonal/sum cocok129/160 dan138/160. Tidak teridentifikasi either31/both12; pooled apa_kabar↔halo errors7. Response exports dan private mapping dihash lokal; hanya hasil agregat disalin, tanpa private mapping atau identitas signer.

Reviewer1: salah kelas4, ambiguous27, invalid0; reviewer2 salah kelas6, ambiguous12, invalid4. Akurasi subset identifiable tinggi tidak menggantikan denominator semua160 karena selection bias. `apa_kabar` paling rendah untuk reviewer1 dan termasuk terendah reviewer2, bukan bukti temporal causation. Jumlah orang, keahlian linguistik dan independensi reviewer tidak diautentikasi dari ekspor. Tidak ada expert-certified full-label review. Wilson CI tercatat, tidak direplikasi; korelasi signer dapat mempersempit CI secara semu.

## 5. Classification training pilot

Historical `experiment_yolo26_classification_pilot` dan notebook reproduction20261102: A accuracy112/120=0,9333333/F1=0,9317756; B common6=111/120=0,925/F1=0,9300419; B full8=150/160=0,9375/F1=0,9354641. Sumber masing-masing `evaluation/comparison_metrics.csv`, rows model/scope. Nilai cocok saved rows; jangan menghitung reproduction sebagai seed independen. `pilot_conclusion.md` mencatat apa_kabar20/20 kedua model, halo19/20 (satu→apa_kabar), technical thumbnails160 tetapi expert-verified0/semantic-unassessed640. Human pilot dan model memakai pool/split berbeda; gap bukan bukti model memahami bahasa lebih baik.

Detection pilot `experiment_yolo26_training_pilot/pilot_conclusion.md`: NOT TRAINED, verified positives0/unreviewed2400; tidak ada hasil pembanding detection yang completed. Classification dipilih karena target satu label/frame, bukan hasil benchmark keunggulan terhadap detection/temporal.

## 6. Stage1b five-seed experiments

Seluruh sepuluh run berikut; common6 N120=20/kelas pada validation yang dipasangkan, B full8 N160. A train360/B480=60/kelas. Lima sampling manifests memakai sampling seed42, training seeds42–46; bukan lima dataset independen. Dua seed42 reused, delapan fits tambahan. Tidak cherry-pick seed.

YOLO26n-cls ImageNet, init SHA256 `0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`; fresh per arm, bukan B meneruskan A. 224²,30 epochs,batch=nbs16,SGD lr0.01,momentum.937,Nesterov true,decay.0005,linear schedule lrf.01,warmup3 epochs,patience0,resume false,workers0,amp false,deterministic true. A23 batches/epoch=690 steps; B30=900. Equal epochs tidak menyamakan compute. Epoch30 LR tercatat.00043, bukan.0001. Checkpoint endpoint last.pt EMA30 menurut records, bukan best.pt. Semua sepuluh bytes last.pt dihash ulang dan cocok trained_models/effective_configuration. Metadata checkpoint tidak di-unpickle ulang; `CHECKPOINT_HASHES.json` membedakan verifikasi byte dan metadata historis. Stripped epoch−1/optimizer None bukan nol training; saved curve memiliki30 rows. Initial checkpoint pin sama pada seluruh records; metadata ImageNet/upstream bukan autentikasi independen baru.

Tabel: accuracy/precision/recall/F1 pada common6; full8_F1 hanya B. Macro precision/recall adalah rata-rata aritmetika saved per-class CSV (derivative eksplisit), bukan metrik training framework. Seconds adalah recorded fit duration, tidak termasuk prepare/inference; bukan benchmark efisiensi baru.

'''
report+=table(ten,['model','seed','train','val','steps','seconds','accuracy','macro_precision','macro_recall','macro_F1','full8_F1'])
report+='\nLokasi lengkap tiap row pada `STAGE1B_TEN_RUNS.csv: source`; source row model/scope, columns n/accuracy/macro_f1. Precision/recall: sibling `per_class_metrics.csv`, rows model/scope/class_id. Steps: sibling `runs/model_*_execution.json:actual_optimizer_steps`; duration: `'+S+'/exploratory_all_fits.csv`, kind=full/model/seed. Hash setiap checkpoint:\n\n'
report+=table(checkpoints,['model','seed','sha256'])
report+='''
Per-class precision/recall/F1/TP/FP/FN seluruh seed tersedia di RESULTS_SUMMARY.csv dan original per_class CSV; confusion matrices A common6, B common6/full8 disalin untuk semua seed (bukan satu seed terbaik). Saved predictions dibaca dan N/correct/per-class TP/FP/FN/F1 dicek ulang tanpa model inference. Framework epoch validation dapat memakai best/EMA yang berbeda dari fixed last endpoint; jangan mengganti hasil endpoint dengan best score.

## 7. Paired A/B statistics

'''
report+=table(primary,['seed','n','macro_f1_A','macro_f1_B','delta','ci95_low','ci95_high'])
report+='''
Source `01_hasil/addendum/development_interference_20261105/common6_primary_results.csv`, rows seed42–46/mean, columns di atas; VERIFIED_FROM_ARTIFACT untuk saved estimates/CI, bukan independen bootstrap replication. Mean delta dan SD aritmetika diperiksa ulang. `analysis_details.json: seed_delta_sd`=0,01589842145086347 (ddof1). `exploratory_model_summary.csv`: A F1 SD0,011577652345406817; B0,0054778690453032575; B full8 mean F1=0,9361928436021125/SD0,006182415991676225. Accuracy means A0,9416667/B common6 0,93/B full8 0,9375. Five observed seeds memberi informasi variasi terbatas.

**Dua CI historis berbeda, keduanya dipertahankan:** summary Stage1b `'''+OLD+'''` key ci95=[−0,03010145130566371; +0,007667945291901918], lower one-sided95=−0,02558950457581278. Output development20261105 row mean=[−0,030222818611814216; +0,007612139862988273], lower=−0,02533392899552548. Mean dan SD Δ sama. Dugaan penyebab: implementasi berbeda dapat menghasilkan realisasi bootstrap berbeda walau10.000/RNG20261103 sama; seed RNG tidak menjamin indeks/draws sama di implementasi berbeda. Tidak dilakukan replay untuk mengisolasi penyebab numerik setiap selisih; jangan menyebut salah satunya koreksi pasti. CI20261105 dipakai sebagai ringkasan analysis engine terkini, yang lama tetap dapat diperiksa.

`external_evaluation_report.md` bernama external tetapi header development dan provenance menyatakan internal. Bootstrap yang didokumentasikan: paired video resampling dalam strata GT, draw video yang sama untuk A/B dan semua seeds, paired training seeds diresample independen; F1 dihitung ulang setiap replicate sebelum mean Δ. N120 unique videos,5 seed pairs; **bukan N600 independen**.10000 reps, RNG20261103. `confirmatory_statistics.py` lines30–50 membangun video indices lalu seed multinomial weights. Tidak dieksekusi kembali.

Historical video-only CI=[−0,02017397590776844;−0,0000416927246196] dan seed-only CI=[−0,02245936208819712;+0,002236875163160068], conditional sensitivities, bukan pengganti hierarchical. Pooled McNemar counts A-only8/B-only1, p0,0390625 hanya deskriptif karena repeated videos, tidak inferensial. Stage1a seed42 McNemar exact p1, discordance1/120; sumber `01_hasil/tahap1a/.../exploratory_paired/exploratory_analysis.json` disertakan. Histori Stage1a “only seed42 exists” berlaku saat itu, bukan kontradiksi lima-seed sesudahnya.

## 8. I1/I2/I3 interference results

I1: recall mean per GT class (20 unique videos/class,5 paired seeds); file class_interference.csv rows mean. CI nominal95 dan Bonferroni simultaneous nominal95 approximate coverage, bukan ranking causal.

'''
report+=table([r for r in interference if r['seed']=='mean'],['class_name','n','recall_A','recall_B','delta_recall','delta_recall_ci95_low','delta_recall_ci95_high','delta_simultaneous95_low','delta_simultaneous95_high'])
report+='\nI2: intrusion ke halo/tidur pada GT common6. `intrusion_rates.csv`, rows mean:\n\n'
report+=table([r for r in intrusion if r['seed']=='mean'],['class_name','n','overall','halo','tidur'])
report+='\nMean overall0,0066667; setara4 predictions/600 repeated seed-video evaluations, bukan600 video independen. Pada tiap seed counts2,2,0,0,0 (cek file rows overall); saya→tidur dan marah→halo. Apa_kabar intrusion0 pada model validation ini, walau human training-pool ambiguity tinggi. Tidak menganggap human confusion pola model universal.\n\nI3: identitas gap accuracy A−B = restricted_gap + intrusion_gap. Restricted argmax B pada6 output hanya diagnosis, tidak menggantikan primary full8 argmax. File accuracy_decomposition.csv:\n\n'
report+=table(decomposition,['seed','accuracy_A','accuracy_B','accuracy_B_restricted','gap_A_minus_B','restricted_gap','intrusion_gap'])
report+='''
Mean gap0,0116667, restricted_gap0,0116667, intrusion_gap0. Raw intrusion tidak sama dengan kehilangan akurasi akibat opsi baru: pada contoh intrusion, restricted old-class prediction juga dapat salah. Ini dekomposisi output, bukan bukti causal representation benefit. Full8 new-class recall halo mean0,93, tidur0,99 (20 videos/class shared across5 seeds), `exploratory_new_class_metrics.csv`; development lower95 halo0,83/tidur0,96 menurut analysis_details. H2 **not tested: development evidence**; point/lower bounds ini bukan kelulusan external H2.

## 9. Additional existing diagnostics

**Hand masking:** existing landmark-based padded box fill114,padding0,15,224² (`exploratory_mask_provenance.json`). Existing last.pt A/B ×5 seeds diuji pada120 common6/160 full8; mean accuracy A0,92±0,0315128; B common6 0,9166667±0,0131762; B full8 0,89125±0,0205396. Saved all-seeds rows dan per-seed reports disertakan. Mean0,92 memiliki bukti CSV, bukan angka tanpa dokumen. All-including-unavailable denominator dipertahankan; duplicate strata tidak dihitung dua kali. Wajah/tubuh/context tersisa, mask tidak sempurna/distribution shift; tidak membuktikan pure background causation. Mendukung kehati-hatian validitas konstruk revised RQ, bukan membuktikan efek vocabulary.

**Frozen-backbone representation probe:** sepuluh completed new six-class linear heads, train360/val120,30epochs,backbone eval,SGD.01/.937/.0005,batch16,source corresponding last.pt A/B tiap seed. Mean A F1=0,9334267350/accuracy0,935; B F1=0,9259280640/accuracy0,9283333. Sumber summary JSON tiap probe dan exploratory_linear_probes.csv/probe_summary.csv. Validation_used_for_selection=false menurut receipt. B backbone sudah melihat lebih banyak kelas/data, probe bukan isolasi causal/geometri embedding. Failed settings conflict seed42 A berhenti sebelum optimization; attempt2 selesai. Runner preflight0fit disimpan. Tidak mengecualikan seed karena kegagalan teknis.

**Learning curves:** semua epoch30 actual-weight train top1=1,0; framework EMA val berbeda (A0,93333–0,95833, B0,93125–0,94375). `exploratory_epoch30_curves.csv` dan original results30 rows. Gap bukan evaluasi model berbobot identik dan tidak membuktikan sebab overfit. Figures kurva existing dipertahankan, tanpa gambar sampel/contact sheets.

**Stage1a precision/power simulation:** existing completed10.000/scenario, bukan data eksternal. Asumsi external300/50 per new class,5seeds; simulated H1 trueΔ0,discordance.02,seedSD0 memberi NI pass probability.7886; seedSD.02 menjadi.5013. H2 binomial reference true accuracy.9, required45/50, probability one class.616123/both-if-independent.379608. Sumber power_precision_report.md tables dan scenario CSV. Ini hasil simulasi berbasis asumsi, REPORTED_NOT_INDEPENDENTLY_VERIFIED per kalkulasi; tidak dicampur sebagai observed model results di RESULTS_SUMMARY.csv. Historical report “no seed variance measured” berlaku sebelum Stage1b. Tidak menjadi bukti power final memadai.

**ROI-only/frame-selection quality:** tidak ditemukan completed matched ROI-only numerical comparator atau eksperimen causal membandingkan recipe frame selection pada current records yang diprioritaskan. Yang tersedia technical coverage/flags dan human pilot; bukan effect estimate selection policy. UNKNOWN di luar scoped discovery; tidak mengklaim pemeriksaan setiap arsip. No temporal/detection matched completed comparator. Rencana CV/grid105fits/factorial H3 belum memiliki completed hasil di bukti saat ini; bukan failed hypothesis.

## 10. Development versus external evidence

Seluruh metrics model dan pilot manusia berasal internal. External400 adalah inventory, **bukan jumlah scored predictions**. Tidak ada skor external final dalam paket. Receipt internal-only dan absence final scoring receipt di audit terdahulu bukan bukti exhaustive tidak pernah ada exposure di luar repo. Stage1b reused validation tetap development, meski paired statistics memakai nama confirmatory module. Tidak mengganti label hasil menjadi final.

## 11. Threats to validity

Frame-level weak labels, semantic validity belum dinilai; single-frame tidak memodelkan gerakan. Signer/session missing menghalangi cluster CI dan klaim generalisasi signer. Source/background cues, near-duplicates dan context leakage belum dieliminasi. Balanced small validation20/class dan5seeds membatasi presisi; degenerate CI kelas perfect hanya menggambarkan support empiris. Equal epochs mengubah data volume,head,optimizer steps; estimand combined vocabulary-and-data expansion, bukan efek kausal class count. Pilot42 diketahui sebelum penetapan seed list final; alasan lima seed/angka42 tidak didokumentasikan; tidak menjanjikan prospective preregistration terhadap semua pilot. No augmentation bukan bukti robustness. Checkpoint hashes tidak memvalidasi bahasa/lisensi. Historical framework/source/per-batch RNG trace tidak lengkap; deterministic flags bukan bitwise reproducibility guarantee.

## 12. Missing experiments and unresolved decisions

Final full-data A1440 train/360val dan B1920/480,10runs,belum dilakukan. Planned2700/3600steps bukan measured. Final manifests null,freeze absent; provenance/exposure dan academic approvals belum lengkap. Detection→Classification adalah perubahan substantif. Judul Addendum01 “Effect of Vocabulary Expansion on YOLO26 Classification for BISINDO Phrase Recognition” berbeda dari draft frame-based yang diajukan; tidak disamakan dengan approval pembimbing.

H1 NI candidateδ.02 **NI_MARGIN_UNRESOLVED**; rekomendasi estimasi Δ+two-sided CI belum approved. H2 asli accuracy conditional GT halo DAN tidur masing-masing≥.80, proposed confirmatory lower one-sided95 **> .80** pada keduanya; point target saja tidak cukup. Jangan mengubah threshold/margin setelah melihat external. Ide discrimination benefit eksploratif termotivasi pilot, bukan hipotesis konfirmatori baru atau hasil proven. CV/grid/factorial lama perlu keputusan scope eksplisit bila dilewati. TASK3 tidak dijalankan.

TASK0 controller `06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb` tersedia; historical completion.json/new tests17, notebook12cells, legacy91/92 (satu `test_notebook_has_module_calls_and_training_false`, assertion1 is not None), full_regression_green=false. TASK1 saved report27/27; TASK2 saved report25/25,13 prior code cells preserved,2026 protected files drift0. Logs/JSON dibaca sebagai **historical receipts**, tes tidak diulang dalam snapshot dan tidak mengubah notebook outputs. TASK0–2 PASS audit/infrastructure tidak berarti scientific experiment completed. Partial Task2 directory20261105 bukan report lengkap;20261109 diprioritaskan.

## 13. Recommendations for final training and paper

Minta approval aktual untuk task/title/RQ, H1 estimasi vs NI substantif, final full-data five-seed equal-epoch scope/nasibCV-H3, dan semantic/provenance/exposure gates. Pertahankan H2 asli dan disclosure pilot. Selesaikan manifests/provenance/cluster/semantic decision sebelum prospective release/freeze; jangan backdate. Laporkan semua seed, fixed last endpoint,paired common6 dengan B eight-way argmax, per-class uncertainty dan diagnostics terpisah. Jelaskan kemungkinan manfaat diskriminasi sebagai mekanisme eksploratif; klaim akhir harus sesuai data eksternal yang benar-benar diperoleh dan independensinya. Tidak ada rekomendasi automatic training atau readiness conference submission.

## 14. Complete evidence references

Seluruh lokasi sumber relatif repository, bukan lokasi historis Windows di dalam records yang sudah direorganisasi. Dalam ZIP, salinan byte-identik berada di `EVIDENCE/<source-relative-path>` dan gambar di `FIGURES/<source-relative-path>`. `EVIDENCE_MANIFEST.json` mengikat SHA256,size,category,purpose,status; checkpoint hanya pointer/hash, tidak disalin. `RESULTS_SUMMARY.csv` memberi source_file per metric; row selector memakai model/seed/split atau reviewer/class dan nama kolom metric. Tabel detailed run precision/recall dihitung sebagai mean saved per-class metrics, diberi label derivative. Hash awal seluruh source yang dibaca dan protected source/notebook/protocol diperiksa sesudah ekspor. Tidak ada fresh video/file-integrity scan45.250; CI,model metadata dan historical tests tidak dieksekusi ulang.

'''

# Compact, explicit source selection; no raw media, private signer records, or full notebooks.
for dirname,names in [(T1,['audit_report.md','audit.json','duplicate_impact.json','provenance.json','manifest_catalog.json','final_verification.json']), (T2,['report.md','effective_configuration.json','scientific_decisions.json','seed_provenance.json','candidate_config.json','validation.json','final_verification.json','test_task2_configuration.log','test_confirmatory_statistics.log','test_interference_metrics.log','test_final_control_notebook.log']), ('01_hasil/addendum/final_control_task0',['completion.json','validation.json','existing_regression_canonical_temp.log']), ('07_dokumen/protokol',['analysis_plan.json','reproducibility_manifest.json','PREREGISTRATION.md','PREREGISTRATION_ADDENDUM_01.md','HYPOTHESES_ENDPOINTS_ADDENDUM.md','external_scoring_readiness.md'])]:
    for name in names:read(dirname+'/'+name)
read('07_dokumen/TRACKING_BISINDO.md')
read(T1+'/validation.json');read(T1+'/test_task1_data_audit.log')
read(S+'/exploratory_stage1b_report.md');read(S+'/exploratory_failures.json');read(S+'/exploratory_final_verification.json');read(S+'/exploratory_first_phase_timing.json');read(S+'/exploratory_provenance.json');read(S+'/exploratory_paired_per_seed.csv')
read(D+'/external_evaluation_report.md');read(D+'/input_provenance.json')
mask='02_exp/tahap1b/mask_inputs/exploratory_development_stage1b_user_20261103_v1_exploratory_masks/exploratory_mask_provenance.json';read(mask)
read('01_hasil/tahap1a/exploratory_development_stage1a_20261103/exploratory_paired/exploratory_analysis.json')
for name in ['classification_pilot.py','classification_research.py','classification_diagnostics.py','pilot_study.py','frame_candidates.py','confirmatory_statistics.py','interference_metrics.py','analysis_paired.py']:
    read('05_kode/src/'+name)
for seed in range(42,47):
    work=f'02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed{seed}'
    for arm in ['A','B']:read(work+f'/runs/model_{arm}/args.yaml')
    for name in ['confusion_A_common6.csv','confusion_B_common6.csv','confusion_B_full8.csv','apa_kabar_halo_diagnostics.csv']:read(work+'/evaluation/'+name)
    for name in ['confusion_A_common6.png','confusion_B_common6.png','confusion_B_full8.png']:
        track(work+'/evaluation/'+name)
    for arm in ['A','B']:track(work+f'/runs/model_{arm}/results.png')
track(H+'/analysis/confusion_matrix.png');read(H+'/analysis/pilot_report.md');read(H+'/analysis/verification.json')

# Include all recorded metrics and selected aggregates, with explicit location/evidence status.
csvout('EXPERIMENT_INVENTORY.csv',list(inventory[0]),inventory)
csvout('RESULTS_SUMMARY.csv',list(metrics[0]),metrics)
refs=[]
for s,(digest,size) in sorted(sources.items()):
    if s.endswith('.pt') or '/responses/' in s or s.endswith('manifest.csv'):
        exported=None;category='CHECKPOINT_POINTER' if s.endswith('.pt') else 'PRIVATE_OR_DETAILED_POINTER'
    else:
        category='EXISTING_FIGURE' if Path(s).suffix=='.png' else 'SOURCE_EVIDENCE'
        exported=('FIGURES/' if category=='EXISTING_FIGURE' else 'EVIDENCE/')+s
        dest=OUT/exported;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/s,dest)
    status=R if ('report' in s.lower() or 'validation' in s or 'completion' in s or 'effective_configuration' in s) else V
    if s.endswith('.pt'):status=V
    purpose='Identitas byte checkpoint; model besar tidak disalin' if s.endswith('.pt') else 'Verifikasi klaim lokal; interpretasi/verification limits pada report section terkait'
    refs.append(dict(source_path=s,exported_path=exported,sha256=digest,size_bytes=size,artifact_category=category,scientific_purpose=purpose,evidence_status=status))
    refs_line=f'- `{s}` — '+('pointer-only' if exported is None else f'`{exported}`')+'; '+status
    report+=refs_line+'\n'
write('VERIFIED_EXPERIMENT_RESULTS.md',report)
dump('EVIDENCE_MANIFEST.json',dict(schema_version=1,date_local='2026-11-09',timezone='Asia/Bangkok',verification_scope='Saved outputs cross-check; no training/inference/bootstrap rerun; metadata/tests/audit execution historical only',artifacts=refs))

# Fresh self-checks: schemas, exact source/copy hashes, preservation, ZIP CRC and membership.
required=['SUPERVISOR_PROGRESS_BRIEF.md','VERIFIED_EXPERIMENT_RESULTS.md','EXPERIMENT_INVENTORY.csv','RESULTS_SUMMARY.csv','EVIDENCE_MANIFEST.json','FIGURES']
assert all((OUT/f).exists() for f in required)
for f in required[:2]: assert len((OUT/f).read_text(encoding='utf-8'))>500
for f,expected_cols in [('EXPERIMENT_INVENTORY.csv',list(inventory[0])),('RESULTS_SUMMARY.csv',list(metrics[0]))]:
    with (OUT/f).open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream);assert reader.fieldnames==expected_cols
        data=list(reader);assert data and all(None not in r for r in data)
for p in OUT.rglob('*.json'):json.loads(p.read_text(encoding='utf-8-sig'))
for r in refs:
    assert sha(ROOT/r['source_path'])==r['sha256']
    if r['exported_path']:assert sha(OUT/r['exported_path'])==r['sha256']
assert all(sha(ROOT/s)==digest for s,digest in protected.items())
assert subprocess.run(['git','diff','--binary'],cwd=ROOT,capture_output=True,check=True).stdout==git_before
validation=dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),output_directory=str(OUT),source_artifacts=len(refs),exported_evidence=sum(r['exported_path'] is not None for r in refs),protected_code_notebook_protocol_files=len(protected),protected_drift=0,source_hash_drift=0,stage1b_full_runs=10,checkpoint_hashes_verified=10,stage1b_prediction_metrics_arithmetic='PASS',paired_mean_and_sd='PASS',human_accuracy_confusion_agreement_kappa='PASS',csv_schemas='PASS',json_syntax='PASS',markdown_nonempty='PASS',git_tracked_diff_preserved=True,development_labels='PASS',external_final_scores_included=False,training=False,inference=False,bootstrap_rerun=False,full_dataset_scan_rerun=False,historical_tests_rerun=False,limitations='Preservation covers read evidence plus protected code/notebooks/protocol; no exhaustive raw data rehash. Model metadata/historical audit and tests were not replayed.')
dump('VALIDATION.json',validation)
# Hash package outputs separately: manifest intentionally does not hash itself recursively.
hashes={p.relative_to(OUT).as_posix():sha(p) for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='PACKAGE_SHA256.json'}
dump('PACKAGE_SHA256.json',hashes)
z=OUT/'BISINDO_CONSULTATION_EVIDENCE.zip'
assert not z.exists(), 'Do not overwrite prior package'
with zipfile.ZipFile(z,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
    for p in sorted(OUT.rglob('*')):
        if p.is_file() and p!=z:archive.write(p,p.relative_to(OUT).as_posix())
with zipfile.ZipFile(z) as archive:
    assert archive.testzip() is None
    for name,digest in hashes.items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest
    assert all(name in archive.namelist() for name in required[:-1])
    assert not any(name.lower().endswith(('.pt','.mp4','.ipynb')) for name in archive.namelist())
print(json.dumps(dict(status='PASS',out=str(OUT),zip_bytes=z.stat().st_size,zip_sha256=sha(z),sources=len(refs),metrics=len(metrics),inventory=len(inventory),figures=len(list((OUT/'FIGURES').rglob('*.png')))),indent=2))
