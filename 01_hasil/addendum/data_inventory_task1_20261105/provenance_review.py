"""Record local provenance evidence and unresolved claims; no attestation or inference."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'05_kode/src'))
import confirmatory_evaluation as evaluation

OUT = Path(__file__).resolve().parent
inputs = {}


def read_json(path):
    inputs[str(path.relative_to(ROOT))] = evaluation.sha(path)
    return evaluation.read(path)


def read_csv(path):
    inputs[str(path.relative_to(ROOT))] = evaluation.sha(path)
    return pd.read_csv(path,keep_default_na=False)


def main():
    started = datetime.now(timezone.utc).isoformat()
    audit = read_json(OUT/'audit.json')
    historical = ROOT/'07_dokumen/draft_paper/Effect of Training Set Expansion on YOLOv26 Performance for Real-Time BISINDO Phrase Recognition.md'
    inputs[str(historical.relative_to(ROOT))] = evaluation.sha(historical)
    passages = [dict(path=str(historical.relative_to(ROOT)),line=i,text=line)
                for i,line in enumerate(historical.read_text(encoding='utf8').splitlines(),1)
                if len(line)<4000 and ('kaggle' in line.lower() or 'Blurry or transitional' in line)]
    receipts = []
    for path in sorted((ROOT/'02_exp').rglob('evaluation_complete.json')):
        record = read_json(path)
        manifest = read_csv(path.parent.parent/'manifest.csv')
        receipts.append(dict(path=str(path.relative_to(ROOT)),keys=list(record),
                             manifest_sources={str(k):int(v) for k,v in manifest.source.value_counts().items()}))
    scoring_receipts = [str(p.relative_to(ROOT)) for folder in ['01_hasil','02_exp','07_dokumen']
                        for p in (ROOT/folder).rglob('*.scoring_started.json')]
    pilot = ROOT/'02_exp/arsip_pipeline/experiment_yolo26_pilot'
    manifest = read_csv(pilot/'pilot_manifest_private.csv')
    reviews = []
    for path in sorted((pilot/'responses').glob('reviewer_*.json')):
        record = read_json(path)
        responses = record['responses']
        ids = [item['sample_id'] for item in responses]
        assert len(ids) == len(set(ids)) and set(ids) == set(manifest.sample_id)
        reviews.append(dict(path=str(path.relative_to(ROOT)),reviewer=record['reviewer_id'],
                            n=len(ids),decision_counts=dict(Counter(x['decision'] for x in responses)),
                            original_export_matches=evaluation.sha(path)==evaluation.sha(ROOT/'08_review/respons'/f"{record['reviewer_id']}_final.json")))
    technical = read_json(ROOT/'02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/audit/technical_visual_summary.json')
    old = read_csv(ROOT/'02_exp/arsip_pipeline/experiment_yolo26_v2/annotations.csv')
    scan_command = ['rg','--files','-g','*licen*','-g','*LICEN*','-g','*metadata*','-g','*kaggle*','-g','*.zip','-g','*.7z']
    scan = subprocess.run(scan_command,cwd=ROOT,capture_output=True,text=True,check=False)
    if scan.returncode not in [0,1]:
        raise RuntimeError(scan.stderr)
    tables = [
        dict(topic='Dataset source',status='PARTIALLY VERIFIED',evidence='Historical draft reference [26] cites Anggi Yohanes Pardede, BISINDO 40 Kata mp4, Kaggle; URL in source passages.',
             limitation='Citation is local documentary evidence; no archive/version-to-local-byte binding verified.'),
        dict(topic='Version/acquisition',status='UNVERIFIED',evidence='Draft lists access date Jun 10 2026; preparation records bind local hashes.',
             limitation='Draft date is not an acquisition receipt. Need dataset version ID, download/acquisition log, original archive hash, subset/rename mapping provenance.'),
        dict(topic='License',status='UNVERIFIED',evidence='Scoped filename search recorded; no dataset license/version receipt found.',
             limitation='Need license text tied to acquired version, author attribution and reuse conditions. No license inferred from public availability.'),
        dict(topic='Signer/session',status='UNVERIFIED',evidence=f"External nonempty signer={audit['external_signer_nonempty']}/400; internal={audit['internal_signer_nonempty']}/2400.",
             limitation='Need trustworthy source records or reviewer-backed provenance; hashes do not establish signer independence.'),
        dict(topic='Historical v2/v3 exposure',status='VERIFIED preprocessing',evidence='400 external v2 annotation/frame records; 400 external v3 candidate-cache records and actual cached image byte/RGB checks in audit.json/file_integrity.jsonl.',
             limitation='Previously opened/preprocessed; not untouched. Draft describes filtering blurry/transitional frames, without complete underlying acquisition/selection audit trail.'),
        dict(topic='External scoring/tuning history',status='UNVERIFIED completeness',evidence=f'{len(receipts)} discovered completed Classification evaluation receipts have internal-only manifests; {len(scoring_receipts)} final scoring-start receipts found in scoped search.',
             limitation='Absence in these records is not proof of absence elsewhere. Owner/reviewer must review historical external inference, score viewing, frame/threshold/model/checkpoint/hyperparameter selection; do not generate attestation.'),
        dict(topic='Neutral filename GT',status='VERIFIED local consistency',evidence='400 unique neutral filename/GT/original-filename CSV rows map to physical external files and locked class IDs; original filename class tokens match CSV.',
             limitation='Original archive labels and semantic gesture correctness unverified. Mapping CSV alone is not authoritative acquisition provenance.'),
        dict(topic='Semantic/ambiguity review',status='PARTIAL, internal only',evidence='Two reviewer response exports cover 160 training-pool pilot samples; original exports match; counts below. Classification technical contact-sheet review has expert_verified_n=0 and semantic_unassessed_n=640.',
             limitation='No complete external semantic adjudication. Review responses are not relabelled into final manifests; reviewer expertise/independence not established by hashes.'),
    ]
    result = dict(task=1,started_utc=started,completed_utc=datetime.now(timezone.utc).isoformat(),
                  command=[sys.executable,*sys.argv],evidence_table=tables,historical_draft_passages=passages,
                  completed_evaluation_receipts=receipts,final_scoring_receipt_paths=scoring_receipts,
                  scoped_filename_search=dict(command=scan_command,exit_code=scan.returncode,output=scan.stdout,
                                             limitation='rg obeys ignore rules; scoped repo search, not proof that no off-repo/ignored records exist'),
                  human_pilot=dict(n=len(manifest),sources=manifest.source.value_counts().to_dict(),
                                   splits=manifest.split.value_counts().to_dict(),reviews=reviews),
                  classification_technical_review=technical,
                  v2_external_review_status={str(k):int(v) for k,v in old.loc[old.source.eq('test'),'review_status'].value_counts().items()},
                  input_sha256=inputs,attestation_generated=False,external_inference=False)
    evaluation.write_new(OUT/'provenance.json',result)
    print('Local provenance evidence recorded; acquisition/license/signer/exposure completeness remain UNVERIFIED')


if __name__=='__main__':
    main()
