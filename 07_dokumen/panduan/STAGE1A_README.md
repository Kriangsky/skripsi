# Stage 1a — analysis and implementation, zero training

Date: 2026-11-03, Asia/Bangkok. Protocol candidate v1 + Addendum 01. New manuscript is a conference paper skeleton.
No demo/webcam/application deliverable. Timing, if measured later, is inference cost only.

## FACT — existing artifacts

Pilot A/B were trained independently for seed42, 30 epochs, with fixed last.pt. Seed42 reproduction is not a new seed.
New reports are read-only analyses of that existing pilot; no external test images/scores were opened.

- Paired correctness: both correct111, A-only1, B-only0, both wrong8; exact two-sided McNemar p=1.
- Common6 macro-F1 d(B−A)=−0.001733726. Paired stratified-video bootstrap 95% CI
  [−0.025679348, +0.015664552]; one-sided95% lower −0.020841948.
- This exploratory CI crosses the NI boundary −0.02. It does not confirm H1.
- Correctness discordance1/120 differs from predicted-label disagreement6/120. Union errors9.
- Observed training seeds: [42]; observed between-seed variance: unavailable.

Files: `experiment_yolo26_classification_diag_v1/reports/paired/`
(`paired_errors.csv`, `paired_ci.csv`, `accuracy_wilson.csv`, `per_class_metrics.csv`, `analysis.json`, `paired_analysis.md`).
Wilson is for accuracy; macro-F1 intervals use paired video bootstrap, not Wilson.

## SIMULATION — external planning, not measured performance

Full scenario CSV/assumptions: `experiment_yolo26_classification_diag_v1/reports/power/`.
10,000 Monte Carlo repetitions/scenario, RNG20261103. Video n300 (50/GT class), five paired seeds;
no pooling into1,500 independent videos. Modelled true macro-F1 deltas −.02/−.01/0/+.01,
correctness discordance .02/.05/.10, paired seed SD0/.005/.01/.02.
B's extra outputs are retained. Oracle delta-method intervals are an approximation, not exact power of the
final hierarchical-bootstrap CI. Unknown signer/session clustering and external shift are not measured here.

At true macro-F1 delta0 and assumed paired seed SD1 pp:

| Correctness discordance | Approx. 95% CI width | Approx. H1 pass probability |
|---|---:|---:|
| 2% | 3.63 pp | 69.70% |
| 5% | 5.23 pp | 44.34% |
| 10% | 7.14 pp | 29.52% |

H2 binomial reference (one classifier/perfectly correlated seed errors): n50/class and threshold.80,
one-sided95% Clopper–Pearson requires at least45/50 correct. At true accuracy .90 per class,
pass probability .6161 per class; both independent classes .3796. At .95, these become .9622 and .9259.
This reference does not replace the preregistered five-seed hierarchical interval. Unknown cross-class dependence
is bounded in CSV; the both-class product assumes independence. Do not treat five seeds as n250/class.

## PROPOSAL — decisions remain with researcher

300 external common6 videos do not guarantee adequate NI precision. Keep a prospectively declared external
confirmation test and accept inconclusive outcomes, collect more independent external videos under a revised
protocol, or make internal CV the main **development evidence** while keeping external corroboration separate.
Reused/tuned CV is not an untouched confirmatory domain. Nested separation is needed for stronger interpretation
of selection plus testing; increasing sample count alone does not remove selection bias.

Delta=.02 and H2=.80 are unchanged. Researcher must decide delta, H2 threshold and H1 evidence domain;
record any decisions in Addendum 02 before prospective training. No choice is made by these simulations.

## Implemented parameterized API

New module: `classification_research.py`. Old `classification_pilot.py` commands remain unchanged.
The new module supports `prepare`, `probe`, `train`, `evaluate`, with implemented `--seed`, `--epochs`,
`--imgsz`, `--work` / `--work-dir`, `--root`, `--source`; `--checkpoint` for prepare, `--approval` for train.

Preparation requires a new direct-child work directory, pinned local checkpoint and the existing internal data/cache.
Seed/epochs/imgsz are locked in `research_config.json`, incorporated in release identity, verified on later commands,
and propagated to letterbox, probe, transforms, actual training args, final-epoch checks, evaluator and report labels.
Sampling remains seed42,60/20 per class. Original source module globals are never mutated.
Partial/completed prepare dirs are not overwritten. Train/evaluate reject overwrite/resume.
Training requires a scoped researcher approval record and verified candidate/Addendum01/Addendum02 hashes.
There is no training authorization file supplied by this revision.

Inspect actual options without running a model:

```powershell
.\.venv-bisindo\Scripts\python.exe classification_research.py --help
```

Analysis reproduction requires **new output paths**; existing report folders deliberately reject overwrite:

```powershell
.\.venv-bisindo\Scripts\python.exe analysis_paired.py --work experiment_yolo26_classification_pilot --out NEW_PAIRED_OUTPUT --repetitions 10000
.\.venv-bisindo\Scripts\python.exe power_precision.py --work experiment_yolo26_classification_pilot --out NEW_POWER_OUTPUT --repetitions 10000
```

## Validation and remaining work

Unit tests exercise probability integrity, GT/image-hash matching, paired resampling, exact tests,
F1 gradient/calibration, H2 binomial calculation, no-overwrite, parameter propagation, release fingerprints,
approval-before-GPU, and fixed last-epoch records. Fake model fits in unit tests do not perform optimization.
No live GPU training or smoke fit is run. New scripts use already installed numpy/pandas/scipy and project helpers;
no additional dependency was installed.

Stage1b proposal: 10 full fits (A/B ×seeds42..46) plus10 frozen-backbone head fits if approved together.
Recorded pilot training times: A37.4169s, B47.5833s; five baseline pairs total425.001s before preparation/IO/probe
overhead. Engineering wall-time budget remains15–60min including heads/QA, not a measured end-to-end benchmark.
Masking/curve diagnostics add no full model fits, estimate5–30min. No phase starts automatically.
