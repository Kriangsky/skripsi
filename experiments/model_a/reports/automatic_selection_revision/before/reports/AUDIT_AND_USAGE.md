# Model A audit and execution — 2026-11-10, Asia/Bangkok

Model A workflow implemented. Official training **NOT RUN**; official metrics and
checkpoints **NOT AVAILABLE**. A separate technical smoke fit passed. No Model B,
Model C, LoRA, dual head, calibration, tuning or external evaluation implemented.

## Repository audit

- Repository originally clean. Sections 01 results, 02 experiments, 03 analysis,
  04 raw datasets, 05 code/tests, 06 notebooks, 07 protocols, 08 review and
  09 environment. Hidden root aliases and `.compat_runtime` use junctions and
  hard links. Editing a historical hard-linked source would alter its evidence.
- Preprocessing lives in `05_kode/src/frame_candidates.py`, `pilot_study.py`,
  `classification_pilot.py` and `bisindo_pipeline.py`. Candidate extraction and
  technical ranking are coupled to frozen manifests and legacy experiments.
  Old preparation helpers also sample only 60 train/20 validation per class,
  import detection/pilot code, default to old checkpoint locations, and may
  train A/B together. They are unsuitable entry points for this new baseline.
- Reuse: frozen video identities/splits, cached full-context candidate JPEGs,
  existing selected-candidate identities, reviewer exports and source provenance.
  New helper does not import old training code, weights, metrics or search results.
- Actual interpreter: `.venv-bisindo/Scripts/python.exe`, Python 3.11.5;
  Ultralytics 8.4.197; PyTorch 2.14.0+cu126; torchvision 0.29.0+cu126;
  NumPy 1.26.4; pandas 2.2.3. CUDA12.6 available, RTX4070 Laptop GPU, 8GB.
  System Python is a different environment and lacks Ultralytics.

## Historical archive

`old hypothesa/ARCHIVE_MANIFEST.md` and `archive_receipts.json` record exact paths
and file SHA256 values. 24 directories / 1,923 files archived; 447 tracked files
moved with `git mv`. Renames staged; no commit created. Original directory paths
are compatibility junctions. No historical file deleted or checkpoint overwritten.

Moved: six experiment notebooks, two result notebooks, stage1a/stage1b results,
historical analysis folders, linear probes, and old classification/detection-pilot
run/evaluation subdirectories. Nine protocol documents copied as archived evidence;
originals retained for historical hash-bound callers. Protected source hashes cover
2,424 raw/internal/source-definition files. Existing runtime hard-link check passes.

Intentionally retained: raw datasets, all split locks/source inventories, source
and selected-frame manifests, candidate caches, reviewer responses, original
protocols, old source/tests/launchers, preparation notebooks, organization and
data-audit/addendum evidence, environment and compatibility runtime. Historical
experiment containers mix required provenance with old outputs; they remain while
their run/evaluation subdirectories route to the archive. Failure/session logs in
`02_exp/kegagalan` and `02_exp/log_sesi`, consultation/draft/handoff/planning
documents remain for continuity of the historical record. They are not new Model A
evidence. Old pretrained assets remain untouched and are never read by Model A.

The first archive request was rejected by automatic approval review because its
scope lacked a verified dry run. The exact 24-directory, per-file-hash dry run and
rollback plan were then checked; the reviewed execution was approved and completed.

## Dataset and split evidence

Frozen source: `02_exp/arsip_pipeline/experiment_yolo26_v3/split_lock.csv`.
Inventory has 2,400 internal videos plus400 historical external metadata rows.
No external video/image is loaded by this workflow. No split regenerated.

| Research ID | Class | Internal | Train | Validation | Existing selected frames |
|---|---|---:|---:|---:|---:|
|0|saya|300|240|60|80|
|1|kamu|300|240|60|80|
|2|senang|300|240|60|80|
|3|bingung|300|240|60|80|
|4|marah|300|240|60|80|
|5|apa_kabar|300|240|60|80|
|Total||1800|1440|360|480|

Unique sample IDs, source paths and stored video hashes; labels match internal
folders and canonical IDs; frozen inventory equals split lock. All1,800 internal
six-class videos were physically located and SHA256 verified. The physical full
internal inventory also matches its2,400 frozen paths. No exact source-byte
train/validation overlap found. No signer-independence claim.

**Blocker:** no complete selected-frame manifest exists. Historical selections
cover480/1,800; missing1,320. All2,800 v3 frame annotations are needs_review, with
zero approved selected positives. Historical six-class selections remain
semantically unassessed. Reviewer pilot responses flag concerns on49 six-class
source videos. Their IDs use a different namespace; joins go through pilot
`video_id`, preserving decisions/reasons and scope. A pilot-frame decision does
not automatically adjudicate a different selected frame.

`manifests/frame_decisions.csv` is created once and never overwritten on rerun.
It contains all1,800 source rows,480 reused selections and explicit pending
decisions. `frame_decisions_template.csv` is a regenerated reference, not the
researcher's decision file. `historical_review_concerns.csv` preserves concerns.
No new frame selection, exclusion or relabeling has been performed.

To release the official dataset, a researcher must complete frame selection under
a documented approved policy and account for every source video:

1. Bind selected candidate/frame/index/hash to existing `candidates.csv`.
2. Set disposition to include or exclude; record reviewer and decision_note.
3. Included frame: semantic_status=valid_for_target and
   reviewed_candidate_id=candidate_id. Resolve known concerns explicitly.
4. Exclude only for a documented methodological reason; never remove difficulty
   or adjust labels to improve results. Export all exclusions and coverage.

No blanket completion script is supplied. Official preparation fails before
model initialization on unresolved decisions. Completed releases are immutable,
content-addressed dataset folders with original decisions/source/split fingerprints.
Future Model B comparisons must preserve that exact Model A release.

Exact source-video leakage checks pass. Selected JPEG-byte and decoded-RGB checks
pass on the prepared technical subset and run again on every future official
release. Full1,800-frame leakage verification cannot finish before those frames
exist. Near-duplicate/video-segment overlap remains unverified. Earlier audit
reported senang223/224 shared candidate frames within locked train; no split or
video has been removed. Kaggle is reserved, not used for fitting/validation/tuning.

## Preprocessing and labels

Installed default training uses RandomResizedCrop; default validation uses
Resize+CenterCrop. The custom trainer/validator/predictor replace those transforms
without forking Ultralytics. RGB conversion, full-frame bilinear aspect-preserving
resize, centered gray114 padding, tensor/255, mean0/std1. No random augmentation.
Initial224² input. Same transform on training, validation and explicit standalone
inference. Sanity figure visually inspected across all six classes; no added crop.
Source-frame clipping, blur and semantic validity remain separate limitations.

Actual output order confirmed by smoke: apa_kabar, bingung, kamu, marah, saya,
senang. Output-to-canonical IDs: `[5,3,1,4,0,2]`. Both maps exported and validated.

The official upstream v8.4.0 `yolo26n-cls.pt` was downloaded independently to the
new workflow, not copied from a historical experiment. SHA256:
`0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`.
Loading, classification task,1,000 ImageNet outputs and finite forward pass passed.
Native Windows download was used because sandbox DNS and Python certificate
validation initially failed. TLS verification was not disabled. The notebook now
uses the verified local asset; an absent asset requires network access or a manual
download from the documented official URL with that exact hash.

## Final initial configuration

Official100 epochs maximum, patience20,224², minibatch16, seed42, pretrained=True,
SGD, lr0=.01, lrf=.01, momentum=.937, weight_decay=.0005, linear scheduler,
3 warmup epochs, warmup_momentum=.8, warmup_bias_lr=.1, nominal batch64 with
framework accumulation/scaling. Dropout0; AMP=False/FP32; workers0; GPU0 if
available, CPU otherwise. Deterministic execution enabled with warnings for
unsupported deterministic operations. All semantic augmentation disabled.

Best checkpoint: maximum internal-validation top1, last tie. A small validator
fitness override avoids the installed default average of top1/top5. No tuning.
Every run uses a unique UTC microsecond timestamp, seed and dataset fingerprint.
Actual optimizer groups, accumulation, sample/batch counts, scheduler, names,
epoch LR trace and completed/best epochs are exported. Any unavailable stop detail
is labelled; the full training log is retained.

## Tests actually executed

-10 focused unittest tests PASS (`reports/unit_tests.log`): identity/content
  duplicate rejection, wrong labels/external split rejection, missing/ambiguous
  frame decisions, candidate-bound review, mapping, known-answer macro-F1,
  zero-support classes, path containment, deterministic RGB letterbox and
  notebook schema/syntax/disabled flags.
-31 notebook cells executed in order in the exact venv: PASS; both fitting flags
  False (`notebook_execution_verified.json`). Ordered Python-cell execution also
  passed. An initial sandbox kernel attempt stalled and was interrupted; the
  verified local-kernel execution used the exact interpreter and passed.
-Official checkpoint download/hash/load/forward: PASS.
-Separate24-image technical smoke (12 train/12 validation, two/class/split),
  one epoch on GPU: PASS. Training, best.pt/last.pt, six-output architecture,
  alphabetical mapping, custom validation, checkpoint deserialization, standalone
  predictions, six-class metrics and confusion/error exports all completed.
  Nominal minibatch16; only12 training samples, so the smoke's observed batch is12.
  These are **not official Model A results**. Two smoke runs were preserved,
  including a rerun after adding mapping assertions and source snapshots.
-Archive file hashes, protected input hashes and compatibility runtime checked.
  Machine-readable final gate: `reports/final_verification.json`.

## Files and outputs

- New `Model_A_YOLO26_6Class_Training.ipynb`, `model_a.py`,
  `test_model_a.py`, `archive_history.py`, `config/model_a.json`.
- New manifests, audit/review/split/verification reports and sanity image.
- Separate `outputs/smoke_test/` and `outputs/official_runs/` (official remains
  empty). Official weights are retained for later authorized sequential expansion.
- Every new completed run exports predictions, canonical and output-order
  probabilities, validation metrics JSON/CSV, per-class support, confusion PNG/CSV,
  errors, history, actual configuration, architecture, environment/package versions,
  source/config/notebook snapshots, seed, dataset release/fingerprints and
  pretrained/final checkpoint hashes.
- Root README points here; `.gitignore` excludes historical compatibility paths
  and the local executed notebook, which embeds participant previews.

## Open and run

From PowerShell:

```powershell
Set-Location -LiteralPath 'C:\Users\Ricardo Cuthbert\Downloads\skrpsi bwabwa'
$env:PATH = "$PWD\.venv-bisindo\Scripts;$env:PATH"
& .\.venv-bisindo\Scripts\python.exe -m jupyter lab experiments/model_a/Model_A_YOLO26_6Class_Training.ipynb
```

Run all cells with defaults for audit/sanity/checkpoint verification only.
Confirm section2 shows `.venv-bisindo/Scripts/python.exe` and Ultralytics8.4.197.
For a separate technical fit, set only `RUN_SMOKE_TEST=True`, restart kernel, run
all. For official Model A, first complete the frame decisions, then set only
`RUN_TRAINING=True`, restart kernel, run all. Do not set both flags. Editing
`config/model_a.json` changes the recorded initial settings; restart after edits.

Standalone evaluation: import `model_a` from this folder and call
`evaluate(best_checkpoint, frozen_rows, new_output_directory, size=224,
batch=16, device=0)`. For prediction use `ModelAPredictor` explicitly. Keep the
module importable because checkpoints serialize the transform. Default unqualified
`YOLO.predict` is not this workflow's inference contract.

No full100-epoch fit, external evaluation, full semantic adjudication,
near-duplicate study or signer-independence verification performed. No accuracy,
temporal translation or real-time performance claim is justified by the smoke.
