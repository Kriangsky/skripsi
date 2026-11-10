# Model A: automatic frame release and official training

Researcher instruction revised on 2026-11-10 (Asia/Bangkok): full manual semantic
approval is not required for this exploratory baseline. Technical integrity,
complete coverage, traceability and frozen splits remain mandatory. The previous
implementation and reports are preserved in `before/`. Historical experiment
archives, raw videos, source manifests and locked assignments remain untouched.

## Audit and policy

The 1,320 missing selections came from the historical development pilot's
60-training/20-validation-per-class subsampling, not missing source videos or
missing candidate frames. There are 21,390 existing cached candidate JPEGs for
all 1,800 original-six internal videos. No new extraction or detector run needed.

Declared policy: `config/frame_selection_policy_v1.json`, applied identically to
every class and both splits. First verify candidate identity, source/cache/catalogue
mapping, SHA256, dimensions and decoding. Recompute existing hand-ROI quality scores
from cached landmarks with the audited `frame_candidates.assess_hands` helper.
Among usable candidates rank by:

1. No technical flags first.
2. Detected hands preferred.
3. Fewer unique technical flags.
4. Higher minimum hand-ROI Laplacian-focus score.
5. Lower frame index, then candidate ID.

No class-specific threshold or prediction/validation-score input. Thresholds stay
at the original border margin .01, minimum hand side16px, area256px², focus60.
Scores are uncalibrated technical heuristics, not semantic classifiers.

This is the old deterministic ranking plus an explicit hand-visibility preference.
All480 existing choices still match. Only previously unselected
`train/kamu/kamu_265.mp4` differs from the old rank: frame52 with a detected hand
instead of frame70 without one. This change preceded any official fitting or
validation scoring. All1,800 use the same policy, with no special-case override.

If no candidate clears all technical thresholds, retain the best-ranked readable,
integrity-verified fallback and its flags. Do not exclude/relabel a video. A source
or cache integrity failure blocks the release and records its exact sample ID.

Manifest: `manifests/automatic_frame_selection_v1.csv`. It preserves IDs, labels,
split, video/candidate/image hashes, frame index/count, quality scores and flags,
policy, fallback reason, prior candidate/index/manifest, record path/hash and
original filesystem mtimes. Cache records do **not** contain extraction timestamps;
`NOT_RECORDED_IN_SOURCE_CACHE` is explicit. File mtimes are observations, not
invented extraction dates. Historical manifests and `frame_decisions.csv` remain
intact; the latter is no longer a training gate.

## Release and technical QA

| Class | Train | Validation | Total |
|---|---:|---:|---:|
|saya|240|60|300|
|kamu|240|60|300|
|senang|240|60|300|
|bingung|240|60|300|
|marah|240|60|300|
|apa_kabar|240|60|300|
|Total|1440|360|1800|

- New frames extracted:0. Cached frames reused:1,800.
- Previous selections retained:480. Newly completed selections:1,320.
- Candidate images verified:21,390. Technical failures:0.
- Selected images missing/corrupt:0. Duplicate selected JPEG hashes:0.
- One selected frame per unique source video. No train/validation video identity,
  exact source-byte or selected RGB-pixel overlap; no split regenerated.
- Eligible selections:352. Flagged readable fallbacks:1,448.
- Selected quality flags: possible_hand_blur1,433; hand_near_border_or_clipped31.
  Flags overlap. Detected hands:1 hand in1,277 frames;2 in523;0 in0.
- RGB full-frame letterbox224, no crop/flip; all original frame context retained.
- Actual custom loader verified all1,800 files without dropping any. Initial
  batches have shape16×3×224×224; actual output order confirmed alphabetical.
- No Kaggle image, video, prediction or score loaded. Candidate processing is
  restricted to the1,800 internal IDs. Frozen metadata supplies identity checks.

`selection_summary.json`, `technical_failures.json`, `candidate_integrity.csv`,
the immutable dataset's `quality_assurance.json` and `data_loading_verification.json`
contain machine-readable evidence. Dataset fingerprint:
`a58633d28d06dd212249fd103b1d714b4de93dbe10b91f2039eb41ff0011346c`.

## Visual and historical-concern audit

Twelve contact sheets cover all six classes and both splits, stratified by
eligible/flagged status. The24-stratum overview was inspected, along with both
separate sheets covering all49 historical-concern videos. Selected thumbnails
preserve source orientation and boundaries. Some source images show hand motion
blur, partial hands near source borders, variable illumination and different
upper-body scales/backgrounds. Letterboxing adds no cropping and cannot repair
source defects. This inspection is technical, not BISINDO semantic certification.

The49 concern videos correspond to74 reviewer response records. All74 records
refer to the same candidate currently selected for that video, not a different
frame. Their actual decisions/reasons remain in `flagged_video_review.csv`.
They include37 ambiguous decisions,2 invalid decisions,1 alternate `halo` decision,
and34 original-target decisions carrying concern reasons. These are response counts,
not counts of unique videos and not final adjudication.

Two reviewer-invalid current frames:

- `07ac31399af843ea1152`, senang_083.mov, frame12: preparation/transition/blur.
- `5a12428a3344d508dbe8`, apa_kabar_009.mov, frame80:
  preparation/transition/hand_clipped.

Alternate-label concern: `7184904cf99bcc877b13`, apa_kabar_081.mov, frame58;
reviewer chose halo with preparation reason. Original labels remain unchanged.
All49 current images are technically readable and hash-verified. Technical
usability does not negate the reviewers' semantic concerns. They remain in this
complete exploratory video-labelled baseline under the revised instruction.

## Notebook and training

Updated the existing notebook; retained its16 sections. Replaced the manual gate
with complete automatic selection, source/candidate identity, technical file,
frozen split and quality-flag checks. Added full-release preparation, data-loading
preflight and current completed-run display. Fitting flags remain False by default
to prevent accidental repeated training. This task explicitly authorized the
official run, launched through `model_a.run(smoke=False)` after QA passed.

Configuration unchanged: official ImageNet `yolo26n-cls.pt`, six classes,224²,
batch16, seed42, maximum100 epochs, patience20, SGD, lr0=.01, lrf=.01, momentum.937,
weight_decay.0005, linear scheduler,3 warmup epochs, nominal batch64 with framework
accumulation, FP32, workers0, deterministic execution where supported, GPU0.
No augmentation or tuning. Highest internal-validation top1 selects best.pt;
last tied maximum retained. Training log and per-epoch checkpoint saves retained.

Training status and actual metrics: see `training_results.json` after completion.
No metric is asserted by this pretraining audit. Best/last checkpoint paths,
actual completed/best epochs, stop reason, full predictions, metrics, confusion
matrix, architecture, environment, source snapshots and hashes are in that run.
Source notebooks/configuration/scripts are frozen in its `source_snapshot/`.

## Verification and execution

Eleven focused tests passed, including class-blind ranking, candidate-bound
integrity, complete coverage, semantic-review independence, mapping, macro-F1,
no-crop preprocessing and notebook flags/syntax. All31 updated notebook cells
executed in the existing venv before training, including full data preparation
and actual loader verification; training flags were False during that check.

From repository root, inspect the notebook with:

```powershell
$env:PATH = "$PWD\.venv-bisindo\Scripts;$env:PATH"
& .\.venv-bisindo\Scripts\python.exe -m jupyter lab experiments/model_a/Model_A_YOLO26_6Class_Training.ipynb
```

Default execution displays the latest completed official run without fitting
again. Set only RUN_TRAINING=True for a new official run, or only
RUN_SMOKE_TEST=True for an isolated smoke run. No resume/tuning is silently applied.

Scientific limits: frame semantics remain partially reviewed and unresolved;
internal validation is not signer-independent; backgrounds/signers may overlap;
hashes do not rule out near-duplicates or temporal overlap. This is an exploratory
frame classifier, not temporal translation or measured real-time recognition.
Internal validation was historically reused during development. No external
generalization or confirmatory performance claim follows from this run.

## Completed official results

Training completed with early stopping after **65 epochs** (patience20), not an
interruption. First strict maximum at epoch45; best.pt is epoch46, the last tied
maximum under the documented Ultralytics checkpoint rule. No resume is needed.

| Metric | Actual best-checkpoint result |
|---|---:|
|Top1 accuracy|0.9944444444 (358/360)|
|Macro precision|0.9946236559|
|Macro recall|0.9944444444|
|Macro-F1|0.9944666391|

Per-class metrics (precision/recall/F1; support60 each):

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
|saya|1.00000000|1.00000000|1.00000000|60|
|kamu|1.00000000|1.00000000|1.00000000|60|
|senang|1.00000000|1.00000000|1.00000000|60|
|bingung|1.00000000|0.98333333|0.99159664|60|
|marah|1.00000000|0.98333333|0.99159664|60|
|apa_kabar|0.96774194|1.00000000|0.98360656|60|

Best checkpoint: `experiments\model_a\outputs\official_runs\model_a_seed42_a58633d28d06_20261110T153001638887Z\fit\weights\best.pt`.

SHA256: `0beb069c365311fc2935526a040657f915b60cafa27e860fd4655b30d5c2a922`.

Two validation errors: marah_216.mp4 and bingung_282.mp4 predicted apa_kabar. Their complete confidence/probability records are in validation_errors.csv. No frame/label adjustment was made after observing these errors.

All required prediction, per-class, confusion PNG/CSV, history, configuration, architecture, environment, source-snapshot and checkpoint-hash exports are present in this run. `final_verification.json` records independent metric/mapping/hash checks and final notebook execution.

Legacy `dataset_audit.json` fields existing_selected_frames/missing_selected_frames refer to historical pilot coverage (480/1320); selected_frame_records and the current release QA give the complete 1800-frame coverage with zero missing selections. Historical semantic decision templates remain provenance, not active gates.
