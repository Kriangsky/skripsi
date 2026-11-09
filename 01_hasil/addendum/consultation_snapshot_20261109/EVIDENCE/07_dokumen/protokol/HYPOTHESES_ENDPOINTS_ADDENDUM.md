# Hypotheses and Endpoints — Addendum v1 (DRAFT / BLOCKED)

Date: 2026-11-05, Asia/Bangkok. Current requested title:
**Impact of Vocabulary Expansion on Frame-Based BISINDO Recognition: A Controlled YOLO26 Study**.
Question: Does expanding the training vocabulary from six to eight BISINDO classes change frame-level
recognition performance on the original six classes, and which classes are most affected?

Source specification: the researcher's supplied integration task. No separate complete Addendum v1 text was
attached beyond that specification. This document follows its X.1–X.7 structure and records unresolved decisions.
Parent `PREREGISTRATION.md` and `PREREGISTRATION_ADDENDUM_01.md` remain historical, unchanged records.
This draft is neither a final freeze nor retrospective preregistration of the observed pilot.
Conflicts and implementation schemas: `external_scoring_readiness.md`. Machine plan: `analysis_plan.json`.

## X.1 Evidence tiers

**Development:** all existing internal training/validation, pilot, diagnostic work and proposed reused-data CV.
Validation signer identity cannot be independently verified: never call it signer-independent.
New analysis of these predictions is exploratory. It provides no confirmatory H1/H2 decision.

**Candidate confirmatory:** one separately approved, frozen Kaggle final scoring batch across all registered seeds.
Frozen split metadata confirms 400 external videos, 50 per class; common6 is its 300-video subset, not 700 total.
Final frame selection/coverage still requires verification. Kaggle was historically preprocessed in v2/v3;
it has not been proved entirely untouched. No final classification scores or tuning use were found in audited
records. Absence of records does not establish absence of use. Owner/reviewer exposure review remains required.

All 400 external signer fields are empty. Dataset version, signer identity/count, acquisition provenance and
independence from internal signers are unresolved. Do not invent them. Until documented, do not finalize a
signer-independent confirmatory claim. Domain differences may include signer, camera, background and lighting;
this design cannot isolate their causal contributions. One external signer would support only conditional
performance on that signer/domain, not the population of BISINDO signers.

## X.2 Notation

BASE=(saya,kamu,senang,bingung,marah,apa_kabar), IDs 0–5.
EXTRA=(halo,tidur), IDs 6–7. ALL_CLASSES=BASE+EXTRA.
A: six-class Classification; B: independently initialized eight-class Classification.
Both start from identical `yolo26n-cls.pt` bytes (ImageNet classification), SHA-256
`0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`.
Detection is historical/blocked work, not the primary task. No silent task/checkpoint substitution.

For video i and training seed s, y_i is the inherited frame GT; p_Ais=argmax over six outputs,
p_Bis=argmax over all eight. q_Bis=argmax restricted to six original outputs is diagnostic only.
One selected frame per video; identical common6 videos, pixels, GT and preprocessing across models and seeds.
F1 is averaged over the fixed six original labels; novel predictions on old GT remain false negatives.
Δ_s=macro-F1_B,s−macro-F1_A,s; primary Δ is their arithmetic mean over seeds [42,43,44,45,46].
Do not pool seed×video rows as independent test samples. Report per-seed results and variability.

## X.3 Operational definition of interference

**I1:** Δr_k=Recall_B(k)−Recall_A(k), k in BASE, using identical common6 videos.
Report both recalls, delta, marginal 95% intervals and simultaneous intervals for the six deltas.
Rank alone cannot establish which class is most affected.

**I2:** proportion of original-class samples with full B prediction halo or tidur.
Report overall and per-original-class rates, each with halo/tidur components. Keep full eight-output argmax.

**I3:** Acc_A−Acc_B=(Acc_A−Acc_B_restricted)+(Acc_B_restricted−Acc_B).
The identity is tested on synthetic arrays. All accuracies use the same common6 denominator.
Restricted prediction is diagnostic, never the primary prediction. The first term is not causal evidence
that vocabulary expansion caused degradation: total training examples and optimization steps also differ.
Observed pilot equal-epoch budgets were A=690 and B=900 optimizer steps.

## X.4 Hypotheses

**H1:** conditional on a scientifically justified margin δ fixed before external scoring,
H0: Δ≤−δ; H1: Δ>−δ, one-sided α=.05. Success requires lower one-sided95% bound strictly greater than −δ.
Equality fails. Failure means **non-inferiority not demonstrated**, not automatic inferiority.

**NI_MARGIN_UNRESOLVED.** Historical candidate/Addendum01 use δ=.02, but audited documents do not give a
substantive justification for acceptable performance loss. This draft does not replace .02 with another value.
No NI test or success decision is allowed now. Report paired Δ and estimation intervals descriptively.
Do not use pilot, external results or success-favouring values to justify δ. A defensible margin or an explicit
approved estimation-only scientific decision must resolve this conflict before finalization.

**H2 — verbatim from PREREGISTRATION.md, section 1:**

> - H2 kemampuan baru: accuracy B pada GT halo **dan** GT tidur masing-masing ≥ **0.80**.
>   Ambang 80% per kelas disetujui peneliti pada 2026-11-03, sebelum run baru.
>   Laporkan point estimate dan CI per kelas. Aturan konfirmatori yang diusulkan: lower one-sided 95% CI
>   masing-masing kelas > 0.80 (intersection-union: kedua komponen harus lolos).
>   Point estimate ≥ 0.80 saja disebut memenuhi target deskriptif, bukan lulus pengujian konfirmatori.

Compatibility: accuracy conditional on each GT class is Classification recall, not precision or pooled
new-class accuracy. Retain mean across the five seeds and hierarchical bounds. Both component tests must pass;
intersection-union does not require Bonferroni between the two components. Development results remain descriptive.
Historical H3/factorial experiments are outside this integration; no new H2/H3 is invented or silently removed
from historical records, and no factorial confirmatory claim is generated here.

## X.5 Endpoints

Primary candidate external common6: mean paired Δmacro-F1 with uncertainty; NI conditional on resolved δ.
Supporting common6: accuracy, I1–I3, per-class metrics and confusion counts, all seeds retained.
B full8: accuracy, macro-F1, per-class recall/precision/F1, confusion matrices and sample counts.
Common6 and full8 differ in GT and prediction tasks; their macro-F1 values are not interchangeable comparisons.
Planned 300/400 counts are verified inventory counts, not yet final scored denominators.

Efficiency is descriptive only, if actually measured: parameters; FLOPs with method and input resolution;
batch1 inference cost with hardware, software versions, warm-up, median and IQR. Follow Addendum01 measured-cost
protocol if used (FP32,100 warmups,1,000 inputs×5 repeats, CUDA synchronization; preprocessing/transfer/forward/
argmax included, decode separately). No new benchmark is run here. Training wall-time is not inference latency.
Forward-pass or inference measurements alone do not establish real-time gesture recognition.

## X.6 Analysis

Strict one-to-one ID/video alignment, GT, class mapping, content/frame identities, complete coverage, all seeds.
Reject duplicate video IDs/content identities, missing predictions/classes, malformed or missing probabilities,
and disagreement between stored predictions and full argmax. Stable sample ordering precedes resampling.

10,000 paired GT-stratified video bootstrap replicates, RNG20261103. Resample each GT stratum to original size.
Use the same sampled videos for A/B and for every training seed. Independently resample five matched seed pairs
with replacement; compute F1 anew for each model/seed/replicate, then average paired differences across sampled
seed pairs. Percentile two-sided95% bounds are quantiles .025/.975; lower one-sided95% is quantile .05.
Also report video-only conditional intervals, seed-only intervals and SD of paired seed effects (ddof1).
Five seeds yield limited information about training variability; repeated video predictions do not increase N.
Video sample sizes are 300 common6 and 400 full8 only when final manifest/coverage is verified.

I1 marginal intervals accompany Bonferroni simultaneous intervals using quantiles .05/(2×6) and
1−.05/(2×6) for six recall deltas. This controls multiplicity at the nominal bootstrap level, with approximate
coverage; do not infer a unique worst class from overlapping intervals. Zero-error/bootstrap boundary cases
may underestimate uncertainty because unseen errors cannot be invented. H2 retains its intersection-union rule.

Sensitivity: multiply bootstrap SE by sqrt(DE) for DE=1,1.5,2 and display normal-approximation intervals and N/DE.
These are assumptive stress tests, not replacement signer-cluster bootstrap or measured effective sample sizes.
No signer-population claim without appropriate signer sampling and metadata. Report provenance limitations.

## X.7 Gates and stopping rules

Stop finalization while critical conflicts/placeholders remain. Existing evidence currently blocks final scoring.
Before final freeze: resolve H1 margin/domain decision; approve final config/addendum; identify all ten final
last.pt checkpoints; preserve seeds and independent identical initialization; lock manifests, splits, mapping,
frame policy/pixels; complete provenance/exposure and technical leakage checks; validate source/tests internally.
No duplicate-free claim based solely on stored metadata: freeze rehashes actual video bytes and decodes frozen
RGB frame pixels. Semantic/near-duplicate and signer identity require separate evidence.

Freeze records SHA-256 of addendum, config, initial/final checkpoints, inventory, split, final manifests,
source/tests, environment/evidence records and plan, with model/task/mapping/seeds/margin/justification,
UTC timestamp, Git commit and working-tree status. No final freeze from this draft.

External scoring is a separate explicit CLI command; it requires reviewer/owner approval bound to exact freeze
hash, rechecks all hashes/gates before inference, and claims a one-batch receipt before model access.
No notebook imports or development commands launch scoring. Failed batches require a documented deviation and
review; never silently retry or delete the receipt. After results: no margin change, best-seed selection, added
training seeds, checkpoint/hyperparameter/frame/threshold tuning. Record every deviation.

No Kaggle scoring or new training is authorized by this implementation task. Actual final scoring still needs
the explicit approval specified above. No fictional external metrics or final scoring artifacts are generated.
