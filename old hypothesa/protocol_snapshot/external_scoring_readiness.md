# External scoring readiness — BLOCKED

Audit date: 2026-11-05, Asia/Bangkok. No external inference, external metric inspection, model training,
checkpoint selection or final freeze was performed. New files are additive; historical source remains intact.
The supplied task authorizes this implementation, not final scoring.

## Audit findings and conflicts

| Question | Evidence and conclusion |
|---|---|
| Main task | `PREREGISTRATION.md`, Addendum01, Stage1b experiment configs, classification notebook and run records: **YOLO26n Classification**. Legacy Detection code/docstrings remain historical; Detection has no completed trained benchmark. |
| Metric compatibility | Fixed-six macro-F1 with B's full eight-output argmax matches Classification. Novel-class predictions remain old-class false negatives. Full8 is a separate endpoint. |
| Completed training seeds | **42,43,44,45,46**, A/B for every seed. All ten `last.pt` byte hashes match recorded hashes. Seed42 pilot/notebook replications do not add independent seeds. |
| Initial checkpoint | Identical recorded initial SHA `0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`; actual pinned `yolo26n-cls.pt` bytes also hashed in reproducibility inventory. Independent initialization recorded in protocol/training source; no A→B chain. |
| Paired predictions | Every seed contains A120/B160 internal predictions; common6 has identical 120 IDs/video hashes/frame hashes and GT. New development command checks actual validation PNG bytes for each arm before analysis. Across-seed identity also checked. |
| Experiment controls | Existing pilot: 224px,30 epochs,batch16,SGD,no augmentation,last.pt. A690/B900 actual steps. Equal epochs do not mean equal data count or optimization steps. |
| Inventory counts | Read actual frozen `split_lock.csv`: internal train1920/val480; external400=50×8; common6 subset300. External signer fields empty400/400. Counts concern inventory, not a finalized frame-release manifest. |
| Duplicate evidence | Stored inventory hashes have no repeated groups, a metadata-only observation. **No fresh raw-video duplicate scan was run in this audit.** Future freeze checks actual video bytes plus decoded frozen RGB frames; no near-duplicate/semantic identity guarantee. |
| Kaggle history | Protocol/Addendum01/tracker/paper acknowledge preparation in v2/v3. Stage1b provenance records external_scores=0. No classification score-driven tuning found in audited records. This is not proof of all historical use; explicit owner/reviewer review is missing. |
| Untouched confirmation | **Not established.** Previously preprocessed, no documented signer independence, no complete exposure attestation, no final freeze. Do not call it never opened or signer-independent. |
| H1 / margin conflict | Historical δ=.02 is preserved in parent. Substantive scientific justification was not found. New plan marks `NI_MARGIN_UNRESOLVED`, active margin=null; no NI test. Researcher must provide a justification independent of observed outcomes, or approve an explicitly estimation-only endpoint. |
| H2 | Exact original wording copied into X.4. Conditional accuracy is per-class recall; threshold .80 each, both one-sided lower95% bounds strictly above .80. Compatible with Classification. No confirmatory decision on internal results. |
| Title / protocol status | Latest user title used only in new addendum/plan. Historical manuscript title, candidate margins and approvals remain unchanged. This is a blocked draft, not an approved replacement preregistration or final Addendum02. |
| Latest manuscript | `draft_paper/Effect of Vocabulary Expansion on YOLO26 Classification for BISINDO Phrase Recognition.md` still has final config/evidence-domain placeholders and old one-seed narrative. It is a historical skeleton, not a publication-ready report of new confirmation. |

No hypothesis wording was invented. H1 testing is suspended under the user's newer substantive-justification
requirement; .02 was not silently revised. H3/factorial proposals remain historical and outside this implementation.

## Implemented files and verification

- `HYPOTHESES_ENDPOINTS_ADDENDUM.md`: X.1–X.7, draft, unresolved scientific choices explicit.
- `analysis_plan.json`: fixed mapping/seeds/statistics, development run paths, unfilled final artifact fields.
- `reproducibility_manifest.json`: observed audit hashes/checkpoints/counts/Git state, **not final freeze**.
- `05_kode/src/interference_metrics.py`: strict alignment, fixed-label metrics, I1/I2/I3.
- `05_kode/src/confirmatory_statistics.py`: paired stratified video/hierarchical bootstrap, intervals, NI boundary,
  Bonferroni six-class intervals and design-effect1/1.5/2 stress tests.
- `05_kode/src/confirmatory_evaluation.py`: explicit validate/development/freeze/score CLI; no notebook side effects.
- `05_kode/tests/test_interference_metrics.py`, `test_confirmatory_statistics.py`: **14 tests passed**.
- `07_dokumen/rencana/confirmatory_addendum_implementation.md`: scope/implementation decisions.
- `01_hasil/addendum/development_interference_20261105/`: actual internal-only result CSVs, confusion matrices,
  report, analysis metadata and input/source hashes. Every report explicitly says development.
- `01_hasil/addendum/synthetic_validation_20261105.json`: successful source-bound synthetic test executions.
- `01_hasil/addendum/existing_tests_20261105.log`: existing suite **91 passed, 1 failed out of92**.

Existing failure: `DevelopmentTests.test_notebook_has_module_calls_and_training_false` expects empty outputs
and `execution_count=None`; committed historical notebook has execution counts1–5. Notebook JSON equals HEAD.
No existing source/test/notebook was changed. Do not clear historical output merely to make this check green.
This stale expectation needs a separately scoped correction before claiming the entire suite passes.

Internal validation: common6 N120, B full8 N160, all five seeds,10,000 bootstrap replicates/RNG20261103.
Mean paired Δmacro-F1 **−0.0087293042**; hierarchical CI95% **[−0.0302228186,0.0076121399]**;
lower one-sided95% **−0.0253339290**. NI/H2 are **not tested as confirmatory**.
Small CI differences from legacy Stage1b are Monte Carlo differences: this implementation draws video indices
with `Generator.choice` before drawing seed weights; legacy code used multinomial video counts. Both use
paired GT-stratified sampling, but consume RNG differently. Point estimates match. No result was selected by CI.

## Outstanding gates

1. Resolve H1 substantive margin versus approved estimation-only decision; finalize/addendum review, including
   study title/evidence domain and deviations from historical candidate.
2. Final selected experiment configuration and all ten final checkpoints do not exist as approved final artifacts.
   Existing Stage1b checkpoints are development only; they are not promoted automatically.
3. Final external one-frame-per-video manifest and complete final internal train/val manifest are absent.
4. Document external dataset version/acquisition, signer provenance and difference from internal signers.
   If unavailable, revise evidence claims/design with researcher approval; do not override gates with guesses.
5. Owner/reviewer must attest complete Kaggle use history, including model/checkpoint/hyperparameter/frame/threshold
   selection. Any score-guided prior use disqualifies the untouched-confirmatory route implemented here.
6. Complete actual-file integrity/leakage checks at freeze; document source frame selection and preprocessing
   provenance. Hashes alone do not verify semantic labels or signer identities.
7. Freeze environment/hardware/software and final evidence artifacts; run current source-bound synthetic validation.
8. Correct or explicitly document the stale historical notebook test before representing full regression as green.
9. Obtain a separate explicit final-scoring approval that binds the exact freeze SHA-256.

## Final artifact schema

All paths in the plan resolve against project root. The draft intentionally uses null for missing artifacts.
`validate_plan` requires status=`final`, task=`classify`, variant=`yolo26n-cls`, exact mapping/seeds/statistical
settings and preserved H2. NI status must be `NI_MARGIN_JUSTIFIED` with numeric margin and substantive text,
or explicitly `ESTIMATION_ONLY_APPROVED` with null margin. Historical .02 is not a default active margin.

`artifacts` must identify addendum, final experiment config, initial checkpoint, dataset manifest, split mapping,
final external/internal frame manifests, environment record, scientific decision, provenance review, exposure
review and validation record. `checkpoints` must have ten records with:

```json
{
  "seed": 42,
  "model": "A",
  "path": "PATH_TO_FINAL_RUN/last.pt",
  "sha256": "ACTUAL_CHECKPOINT_SHA256",
  "tier": "final",
  "initial_checkpoint_sha256": "0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81",
  "independent_initialization": true,
  "experiment_config_sha256": "ACTUAL_FINAL_CONFIG_SHA256"
}
```

The final experiment config must include all actual training controls and `status: final`, `task: classify`,
`checkpoint_rule: last.pt`, positive integer `imgsz`/`epochs`, and
`preprocessing: frozen_letterbox_RGB_255_gray114_no_crop_no_flip`. Preprocessed frame files must already be RGB,
square at frozen resolution; score uses RGB/255 tensors directly, preserving training's letterbox policy.
Do not apply Ultralytics' default classification crop. Source selection/preprocessing provenance is an audited
scientific responsibility; image hashes bind files, not correctness of that provenance.

Each final frame manifest needs `sample_id,video,video_sha256,image_path,image_sha256,class_id,class_name`.
`video` and `image_path` resolve against project root; hashes are actual file SHA-256. External manifest must match
the frozen source=test/split=test IDs and50/class. Internal manifest covers all internal training and validation
frames; split mapping must match IDs/GT/video hashes. No partial external denominator is accepted silently.
Freeze checks complete files and duplicate RGB pixels, before model access. It does not decode raw videos or
detect near-duplicates; existing pipeline corruption/selection QA and provenance remain necessary.

Evidence JSON requirements:

- Scientific decision: `status: approved`, reviewer,timestamp, `plan_sha256`, `addendum_sha256`,
  `unresolved_conflicts: []`. Reviewer approves substantive H1/design decisions before scoring, not from scores.
- Provenance review: `status: verified`, `different_from_internal_signers: true`, nonempty evidence list, reviewer.
  These are evidence-backed reviewer attestations, not machine-inferred identity.
- Exposure review: `status: reviewed`, `used_for_tuning: false`, `prior_scoring: false`, history list, reviewer.
  Historical preprocessing must remain in the history; this review must not claim it never happened.
- Validation record: generated by `validate`, binds current source/tests and successful synthetic executions.
- Environment: actual software versions, hardware and dependency records; hash-bound at freeze.

## Commands

Run from project root. Commands never start training. Existing artifact directories cannot be overwritten.

```powershell
# Safe now: synthetic tests, new record path required on each validated source version.
& .venv-bisindo/Scripts/python.exe 05_kode/src/confirmatory_evaluation.py validate --out 01_hasil/addendum/validation_next.json

# Safe now: existing internal predictions only, new output directory required.
& .venv-bisindo/Scripts/python.exe 05_kode/src/confirmatory_evaluation.py development --out 01_hasil/addendum/development_next

# AFTER scientific decisions/final artifacts are complete. Current draft is rejected.
& .venv-bisindo/Scripts/python.exe 05_kode/src/confirmatory_evaluation.py freeze --plan 07_dokumen/protokol/analysis_plan.json --out 07_dokumen/protokol/final_freeze.json

# Obtain actual freeze hash for the reviewer's separately supplied approval.
Get-FileHash -Algorithm SHA256 -LiteralPath 07_dokumen/protokol/final_freeze.json

# AFTER explicit reviewer/owner approval of that exact freeze. Not run during this task.
& .venv-bisindo/Scripts/python.exe 05_kode/src/confirmatory_evaluation.py score --freeze 07_dokumen/protokol/final_freeze.json --approval 07_dokumen/protokol/final_scoring_approval.json --out 01_hasil/external_final
```

Approval schema (reviewer supplies real values; this example grants no approval):

```json
{
  "action": "final_external_scoring",
  "approved": true,
  "reviewer": "REQUIRED_ACTUAL_REVIEWER",
  "timestamp": "REQUIRED_ACTUAL_APPROVAL_TIME",
  "freeze_sha256": "REQUIRED_ACTUAL_FREEZE_SHA256"
}
```

`score` revalidates freeze coverage/current files, plan/evidence and counts, then creates
`final_freeze.json.scoring_started.json` exclusively before inference. One whole batch covers all ten models.
CUDA required; no CPU fallback. Receipt remains after failure; no automatic retry. Any failed/partial batch
needs documented deviation/review, not deleting the receipt. Models/checkpoints are trusted local research
artifacts. Final hardware inference path remains unexecuted and requires final-run integration verification.

Final output names follow the requested CSV/report names. Raw predictions and scoring provenance are added for
reproducibility. No fictional external results or final freeze exist. Efficiency benchmarking and manuscript
rewriting were not performed; neither is required to validate this internal statistical implementation.
