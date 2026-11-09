"""Assemble proposed JSON and append read-only notebook cells; no scientific execution."""
import copy
import json
from pathlib import Path
import sys

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
sys.path.insert(0,str(ROOT/'05_kode/src'))
import classification_research as research
import nbformat


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf8'))


def write(name,value):
    with (OUT/name).open('x',encoding='utf8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2)


def main():
    audit=read('effective_configuration.json')
    training=copy.deepcopy(audit['runs'][0]['configured']);training.pop('seed')
    candidate=dict(schema_version=1,kind='CANDIDATE',status='PROPOSED_NOT_APPROVED',
        training_authorized=False,freeze_authorized=False,external_scoring_authorized=False,
        approval=None,protocol_amended=False,task='classify',model_variant='yolo26n-cls',
        initial_checkpoint=dict(path='02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/checkpoints/yolo26n-cls.pt',
                                sha256=research.PINNED_SHA,pretrained_domain='ImageNet'),
        class_mapping=list(research.base.CLASSES),training_seeds=[42,43,44,45,46],sampling_seed=42,
        training=training,independent_initialization=True,paired_seed_per_arm=True,
        preprocessing=dict(policy='classification_pilot.letterbox',imgsz=224,padding=114,
            preserve_aspect_ratio=True,color='RGB',tensor_scale='uint8 / 255',normalize_mean=[0,0,0],normalize_std=[1,1,1],
            dataset_transforms=['Resize(224,224)','ToTensor','Normalize(identity)'],
            train_val_external_identical=True,crop=False,horizontal_flip=False,vertical_flip=False,
            randaugment=False,random_erasing=False),
        data=dict(policy='full locked internal training split; validation remains separate',
            split_lock='02_exp/arsip_pipeline/experiment_yolo26_v3/split_lock.csv',
            split_lock_sha256=research.base.b.sha(ROOT/'02_exp/arsip_pipeline/experiment_yolo26_v3/split_lock.csv'),
            targets=dict(train=240,val=60),internal_manifest=None,external_manifest=None,
            one_frame_per_video=True,frame_policy='existing prediction-blind automatic selection; no semantic certification implied',
            common6_ids_pixels_gt_identical=True,validation_used_for_development=True,
            technical_failure_policy='stop and report; no silent exclusion/replacement'),
        arms=dict(A=dict(class_ids=list(range(6)),planned_train=1440,planned_val=360,
                         planned_batches_per_epoch=90,planned_optimizer_steps=2700,planned_warmup_iterations=270),
                  B=dict(class_ids=list(range(8)),planned_train=1920,planned_val=480,
                         planned_batches_per_epoch=120,planned_optimizer_steps=3600,planned_warmup_iterations=360)),
        budget=dict(primary='equal epochs',step_matching=False,actual_counts_require_runtime_verification=True,
                    confounding='vocabulary + additional new-class examples + optimizer steps; not isolated class-count causality'),
        checkpoint_selection=dict(file='last.pt',completed_epochs=30,weights='framework EMA final snapshot',
            best_checkpoint_selection=False,early_stopping=False,best_seed_selection=False,resume=False),
        framework=dict(ultralytics='8.4.197',torch='2.14.0+cu126',torchvision='0.29.0+cu126',
            scheduler='linear LambdaLR, epoch index 0..29; warmup interpolation',
            optimizer='SGD',nesterov=True,accumulation=1,planned_epoch30_lr=0.00043,
            deterministic_limit='warn_only controls; no cross-device/version bitwise guarantee'),
        decisions=dict(h1_recommendation='Option B: estimation-focused paired mean macro-F1 difference',
            h1_status='PENDING_EXPLICIT_APPROVAL_AND_VERSIONED_ADDENDUM',ni_margin=None,
            h2_status='PRESERVED_UNCHANGED',h2_threshold=.80,
            h2_rule='both per-GT new-class mean recalls have hierarchical one-sided 95% lower bounds strictly above 0.80; intersection-union',
            academic_approval='NOT_EVIDENCED',title_status='PENDING_ACADEMIC_APPROVAL'),
        deviations_from_stage1b=[
            dict(change='60/20 to 240/60 videos per class within unchanged locked splits',
                 consequence='A steps 690 to planned2700; B900 to planned3600; validation reused, not independent'),
            dict(change='single fixed baseline recipe without previously proposed CV/grid/factorial or step-matched fits',
                 consequence='no tuned-optimality or H3 factorial claim; requires explicit protocol scope decision; no such fits authorized')],
        readiness=dict(status='BLOCKED',existing_api='stage1b only, TARGETS60/20; candidate is not directly executable',
            dependencies=['approve recipe/full-data and omitted CV/factorial scope',
                'approve H1 Option B and versioned prospective addendum; H2 unchanged',
                'document academic Classification/title/research-question approval',
                'resolve duplicate-cluster/provenance/license/signer/exposure/semantic evidence',
                'register final manifests; validate final-only implementation and runtime transform/step logging',
                'explicit LANJUT TASK 3; no automatic continuation']))
    write('candidate_config.json',candidate)
    protocol=(ROOT/'07_dokumen/protokol/PREREGISTRATION.md').read_text(encoding='utf8')
    h2=protocol[protocol.index('- H2 kemampuan baru:'):protocol.index('\n- H3 sekunder:')]
    write('scientific_decisions.json',dict(status='PROPOSED_NOT_APPROVED',recommended_h1='B',approved_h1=None,
        protocol_amended=False,h1_options=dict(A='NI only after substantive outcome-independent delta justification and approval',
                                             B='mean paired macro-F1 delta with two-sided95% hierarchical CI; no NI success claim'),
        h2_verbatim=h2,h2_threshold=.80,h2_success='both lower one-sided95% bounds >0.80; equality fails',
        academic_approval_evidence=None,signatures=None,finalization_blocked=True,task3_authorized=False))
    write('seed_provenance.json',dict(status='PARTIAL_PROVENANCE',seeds=[42,43,44,45,46],
        earliest_local_suggestion=dict(path='07_dokumen/handoff/BISINDO_PROJECT_HANDOFF_FOR_CLAUDE_2026-11-02.md',
            section='22',wording='Multi-seed A/B, misalnya 42,43,44,45,46 pada sampling dan split yang tetap.',
            document_date='2026-11-02',git_presence='8b3bf45, commit 2026-11-03T12:13:54+07:00',
            limit='document date is not independently verified creation/approval timestamp; suggestion follows already reported pilot'),
        formal_candidate=dict(path='07_dokumen/protokol/PREREGISTRATION.md',document_date='2026-11-03',
                              section='2',pilot42_already_observed=True),
        run_approvals=dict(path='02_exp/log_sesi/exploratory_development_stage1b_user_20261103_v1_session/exploratory_approval_seed{42..46}.json',
                           recorded_date='2026-11-03',scope='stage1b; researcher chat referenced, not independently authenticated'),
        excluded_registered_seeds=[],all_ten_full_runs_accounted=True,
        earlier_seed42_replicates='legacy pilot and notebook reproduction; not additional independent seeds',
        failures='probe A42 before optimization; runner preflight zero fits; multi-work settings conflict retained',
        rationale_for_42_or_consecutive_numbers=None,rationale_for_exactly_five_seeds=None,
        result_influence='No evidence of dropping registered42..46; result-independent original selection NOT established: pilot42 known before proposal.',
        exact_selection_time=None,complete_omitted_attempt_attestation=False,
        scope='local handoff/protocol/addenda/approval/events/all_fits/failure records and git snapshots; no private chat/account attestation'))
    nbpath=ROOT/'06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb'
    nb=nbformat.read(nbpath,4)
    nb.cells[0].source=nb.cells[0].source.replace('TASK 2–8 belum diotorisasi.',
        'TASK 2 diotorisasi untuk audit/usulan saja; TASK 3–8 belum diotorisasi.')
    heading=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## 6.'))
    new=[nbformat.v4.new_markdown_cell('''### TASK 2 — effective recipe dan seed (2026-11-09)

**Status: audit tersedia / kandidat PROPOSED_NOT_APPROVED / final readiness BLOCKED.**
Checkpoint metadata + source override + args/log/counters/results mendukung recipe efektif.
Tidak ada historical batch-level transform trace; rekonstruksi source framework lokal bukan replay training.
TASK 0/1 outputs di atas adalah bukti historis dan dipertahankan.
'''),nbformat.v4.new_code_cell('''TASK2_DIRECTORY = ROOT/'01_hasil/addendum/scientific_decisions_task2_20261109'
TASK2_AUDIT = evaluation.read(TASK2_DIRECTORY/'effective_configuration.json')
TASK2_CANDIDATE = evaluation.read(TASK2_DIRECTORY/'candidate_config.json')
TASK2_SEEDS = evaluation.read(TASK2_DIRECTORY/'seed_provenance.json')
assert TASK2_CANDIDATE['status'] == 'PROPOSED_NOT_APPROVED'
assert TASK2_CANDIDATE['training_authorized'] is False
display(pd.DataFrame([{'seed': row['seed'], 'arm': row['arm'],
                      'epochs': row['configured']['epochs'], 'batch': row['configured']['batch'],
                      'steps': row['steps'], 'epoch30_lr': row['epoch30_learning_rates'][0],
                      'transforms': row['checkpoint_transform']}
                     for row in TASK2_AUDIT['runs']]))
print({'candidate_status': TASK2_CANDIDATE['status'], 'training': TASK2_CANDIDATE['training'],
       'planned_arms': TASK2_CANDIDATE['arms'], 'deviations': TASK2_CANDIDATE['deviations_from_stage1b'],
       'seed_provenance': TASK2_SEEDS, 'evidence_limits': TASK2_AUDIT['evidence_limits']})
''',metadata={'tags':['task2-configuration']})]
    nb.cells[heading:heading]=new
    heading=next(i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and c.source.startswith('## 9.'))
    nb.cells[heading:heading]=[nbformat.v4.new_markdown_cell('''### TASK 2 — keputusan ilmiah menunggu persetujuan

Rekomendasi H1: **Option B**, estimasi paired mean Δmacro-F1 dan two-sided95% hierarchical CI.
Ini usulan, bukan amendemen. Option A memerlukan justifikasi substantif δ yang belum tersedia.
H2 verbatim pada sel historis tetap berlaku. Approval akademik Classification/judul/pertanyaan riset belum terbukti.
Duplikasi frame7/10/14 senang223/224 adalah tiga pasangan frame dari dua video locked train;
bukan tiga pasangan video. Development hanya memakai224/frame48. Content-cluster review tetap pending.
Kaggle provenance/license, signer/session/exposure attestation, final manifests dan semantic labels belum lengkap.
Laporan lengkap: `01_hasil/addendum/scientific_decisions_task2_20261109/report.md`.
'''),nbformat.v4.new_code_cell('''TASK2_DECISIONS = evaluation.read(TASK2_DIRECTORY/'scientific_decisions.json')
assert TASK2_DECISIONS['protocol_amended'] is False
assert TASK2_DECISIONS['task3_authorized'] is False
print(TASK2_DECISIONS)
print({'final_readiness': TASK2_CANDIDATE['readiness'],
       'TASK3': 'NOT_AUTHORIZED', 'training_scoring_freeze': 'DISABLED'})
''',metadata={'tags':['task2-decisions']})]
    nb.cells.append(nbformat.v4.new_markdown_cell('''### Status terkini setelah TASK 2

TASK 2 audit/usulan selesai; verifikasi aktual pada evidence validation.json dan tracker existing.
Kandidat **PROPOSED_NOT_APPROVED**; protokol tidak diamendemen. TASK 3 belum diotorisasi.
Output lama memuat status pada waktu eksekusi historis; tidak ditulis ulang.
MENUNGGU PERSETUJUAN: LANJUT TASK 3
'''))
    nbformat.validate(nb)
    with nbpath.open('w',encoding='utf8',newline='\n') as stream:
        nbformat.write(nb,stream)
    print('Candidate and read-only TASK2 cells created; no protocol amendment')


if __name__=='__main__':
    main()
