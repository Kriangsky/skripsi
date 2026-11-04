# Effect of Vocabulary Expansion on YOLO26 Classification for BISINDO Phrase Recognition

**Conference full-paper skeleton — 2026-11-03.** Research manuscript, not an application deliverable.
Prospective protocol: candidate v1 + Addendum 01. Subsequent design decisions remain pending.
Sections distinguish artifact facts, planning assumptions and proposed experiments. Original manuscript is unchanged.

## Abstract

Vocabulary expansion adds recognition capabilities while potentially changing performance on previously supported
classes. We study independent fine-tuning of YOLO26n Classification for six BISINDO classes (A) and eight classes (B),
adding halo and tidur. The primary question is the cost on shared-six macro-F1 relative to the benefit of new-class
recognition. An exploratory single-seed pilot used video-derived frame labels and identical shared-class images.
A and B obtained shared-six macro-F1 of 0.9318 and 0.9300; B's eight-class accuracy was 93.75% on a separate scope.
These observations do not establish non-inferiority. A prospective design specifies a 0.02 non-inferiority margin,
an 0.80 new-class accuracy threshold with uncertainty, fixed final checkpoints and five training seeds.
**Confirmatory findings: BELUM ADA.** Weak frame labels and unavailable signer/session metadata constrain interpretation.

Keywords: BISINDO; vocabulary expansion; image classification; YOLO26; non-inferiority; paired evaluation.

## Introduction

Expanding a recognition vocabulary introduces new decisions without necessarily providing additional examples for
existing classes. An expanded model therefore should be evaluated both for preservation of prior capabilities and
for recognition of added classes. These are separate questions; eight-class aggregate accuracy cannot directly
answer the shared-six comparison.

We ask: what shared-class performance cost accompanies expansion from six to eight classes, and is that cost
acceptable relative to recognizing halo and tidur? The intended contribution is a controlled empirical paper with
paired evaluation, disclosed data limitations, and uncertainty-aware reporting. No demonstration or product is planned.

H1 concerns non-inferiority of B−A common6 macro-F1 with margin delta=0.02. H2 concerns accuracy above 0.80 for
each new class with lower one-sided 95% CI strictly above threshold. H3 is secondary and restricted to a factorial
contrast that also increases examples of old classes; it cannot isolate an effect due only to added vocabulary.
Domain/precision decisions require Addendum 02 before prospective training.

## Related Work

### Six-phrase BISINDO recognition

Swasono, Himamunanto and Maedjaja (MALCOM 2025) report a YOLO11 object-detection study with the six phrases
used here. Section 2.1 describes an 80:20 training/validation division of images and a separate test collection
from four participants excluded from training. Section 2.2 describes framing video into image sequences.
The 80:20 description alone does not establish video- or signer-grouping of training/validation images; this is
a limit of what can be inferred from that description, not a finding of leakage. Our classification task, data,
split and endpoints differ, so their detection scores are not numerical comparators. [Original paper, §§2.1–2.2,
pp.1063–1064](https://journal.irpi.or.id/index.php/malcom/article/download/2130/1036).

### Sign recognition beyond a single image

Kautsar, Hariono and Akmal compare LSTM and 1D CNN + Transformer approaches to word-level BISINDO recognition.
Their sequence-oriented work motivates distinguishing a static-frame classification task from temporal recognition.
It does not establish that temporal information explains errors in our dataset. No sequence comparator is trained
in the present study. [Authors' preprint, v2](https://arxiv.org/abs/2409.01975).

### Evaluation contribution

The present design measures a vocabulary-expansion trade-off on identical shared-class inputs, rather than a
cross-paper ranking. No comparison of classification versus detection is claimed from unmatched experiments.

## Dataset and Protocol

### Artifact facts

Internal inventory contains 2,400 videos (300/class); Kaggle external inventory contains 400 (50/class).
Frozen internal split has 240 training and 60 validation videos/class. Class IDs are fixed:
0 saya, 1 kamu, 2 senang, 3 bingung, 4 marah, 5 apa_kabar, 6 halo, 7 tidur.
One video contributes at most one automatically selected frame. Selection uses the existing candidate policy,
without prediction confidence, validation score or reviewer answers. Labels are inherited from video metadata.
They are not expert-certified frame-level gesture labels.

### Prospective protocol

Candidate v1 and Addendum 01 specify seeds 42–46, paired evaluation and immutable data/configuration records.
Technical integrity gates remain mandatory. Missing signer/session metadata and incomplete semantic review must
be disclosed. Five-fold internal development CV is proposed in a new fold file; the original split remains unchanged.
Technical container metadata can supply a session proxy only if usable; it does not establish signer independence.

Kaggle was processed during earlier preparation but has not supplied Classification model-selection scores.
The final scoring plan allows one frozen evaluation batch, without subsequent tuning. External common6 consists
of 300 videos and is a subset of full8's 400, not 700 unique observations.
**Final H1 evidence-domain decision: BELUM ADA.** Reused internal CV remains development evidence.

## Method

Model A and Model B initialize independently from identical ImageNet-pretrained `yolo26n-cls.pt` weights.
They do not undergo sequential A→B fine-tuning. Shared-six examples and class IDs remain identical.
The exploratory preprocessing preserves the complete frame using aspect-preserving letterbox to 224×224,
padding114, RGB/255, without flip or crop. Prospective resolution selection is restricted to the preregistered grid;
**selected final configuration: BELUM ADA**.

Exploratory training used 30 epochs, batch16, seed42, SGD lr0 .01, lrf .01, momentum .937, weight decay .0005,
warmup3, deterministic execution and no augmentation. Both official endpoints use `last.pt` at epoch30.
Future equal-epoch comparisons retain a fixed checkpoint rule for all models/seeds. Equal-step analyses are
labelled supplements and do not replace the primary estimand.

The prospective factorial separates vocabulary and old-class sample count: A_n=(old n), B_n=(old n,new n),
C=(old 2n), D=(old 2n,new n). Subsets are nested; validation common6 is identical. A D−A_n difference changes
both factors, so auxiliary contrasts are required to interpret it. These arms have not been trained.

## Experiments

### PILOT EKSPLORATIF — completed, one training seed

Per-class capacity was 60 training and 20 validation videos. A used 360/120 train/val images; B used 480/160.
All 497 technically flagged frames were retained; 143 were technically eligible. These categories do not certify semantics.
Thirty-one selected training videos overlapped the human feasibility study; validation overlap was zero according
to the existing manifest audit. No reviewer answers were used for reselection.

### HASIL KONFIRMATORI — BELUM ADA

Proposed: five paired seeds, one shared configuration chosen without favouring B, full-data training, fixed-epoch
last checkpoints, and one frozen external scoring batch. All seed/config/failure records must be reported.
Training remains unauthorized at this stage. Final configuration, fits, H1/H2 decisions and external scores: **BELUM ADA**.

### Inference cost — planned, unmeasured

**BELUM ADA.** Proposed measurement: batch1 FP32, 100 warmups, CUDA synchronization, 1,000 timed inputs ×5 repeats,
preprocess+host/device transfer+inference+argmax, mean/SD/median/p95, hardware and software versions.
Decode-inclusive cost is a separate endpoint. Timing results would describe inference cost only.

## Results

### Artifact facts: exploratory pilot

| Model | Scope | Correct/N | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| A | common6 | 112/120 | 0.933333 | 0.931776 |
| B | common6 | 111/120 | 0.925000 | 0.930042 |
| B | full8 | 150/160 | 0.937500 | 0.935464 |

Source: `experiment_yolo26_classification_pilot/evaluation/comparison_metrics.csv`.
B's extra-class predictions on common6 remain errors. Full8 is a separate evaluation scope.
Actual optimizer steps were A690 and B900; equal epochs did not imply equal updates.

### Artifact-derived post-hoc paired analysis

Among 120 common6 videos: both correct111, A-only correct1, B-only correct0, both wrong8.
Exact two-sided McNemar p=1.0 concerns correctness, not macro-F1 equivalence.
Paired stratified-video bootstrap (10,000 draws, RNG20261103) gives d_F1=−0.0017337,
two-sided95% interval [−0.0256793,0.0156646], one-sided95% lower bound −0.0208419.
This exploratory interval crosses the −0.02 NI boundary; it does not confirm H1.
Source: `experiment_yolo26_classification_diag_v1/reports/paired/`.

### Confirmatory results

| Endpoint | Estimate | Interval | Decision |
|---|---|---|---|
| H1 common6 macro-F1 difference | BELUM ADA | BELUM ADA | BELUM ADA |
| H2 halo accuracy | BELUM ADA | BELUM ADA | BELUM ADA |
| H2 tidur accuracy | BELUM ADA | BELUM ADA | BELUM ADA |
| H3 factorial D−A_n | BELUM ADA | BELUM ADA | BELUM ADA |
| Inference cost | BELUM ADA | BELUM ADA | BELUM ADA |

## Analysis

### Facts

The observed correctness difference comes from one additional B error; predicted labels differ on six videos.
Both counts matter because macro-F1 also depends on false-positive allocation. The nine-video union of errors
is preserved in the paired report. No seed-to-seed variance has been measured; reproducing seed42 is not another seed.

### Planning simulations, not performance observations

The power report models common6 n300 and hypothetical correctness discordance .02/.05/.10, paired seed SD
0/.005/.01/.02, and true macro-F1 effects −.02/−.01/0/+.01. It retains all B outputs and uses an explicitly
approximate oracle delta-method CI; it is not calibrated power of the final hierarchical-bootstrap procedure.
At true d=0 and assumed paired seed SD .01, estimated NI pass probabilities are .6970, .4434 and .2952 for
discordance .02/.05/.10. These are conditional simulations, not Kaggle measurements.

An exact-binomial reference for one class with n50 requires at least45 correct to place the one-sided95%
Clopper–Pearson lower bound above .80. At true accuracy .90, per-class pass probability is .6161;
two independent classes would both pass with probability .3796. This reference is not the five-seed primary
hierarchical CI, and seeds are not pooled into n250/class.

### Hypotheses and proposals

Additional vocabulary may change representation, head competition, or context reliance. These mechanisms remain
unresolved. Multi-seed training, masking and linear probes are proposed diagnostics, not findings.
The external sample size may leave H1/H2 inconclusive. Maintaining the external confirmation plan, collecting
more independent external data, or making CV the main **development** evidence have different inferential costs.
The researcher must decide before prospective training; this manuscript does not change delta or H2 threshold.

## Limitations

Weak video labels may not identify the gesture present in a selected frame. Expert semantic verification of the
Classification release is incomplete. Signer/session metadata are unavailable; independence cannot be established.
The task is closed-set and always predicts a registered class; unknown gestures are not benchmarked.
Single frames may be ambiguous, particularly apa_kabar/halo as indicated by the human feasibility study;
there is no temporal comparator and no causal attribution to missing motion information.
Internal validation has already informed diagnosis, and CV tuning reuses internal data. Bootstrap intervals
at video level do not account for unidentified signer/session clusters. Power simulations assume distributions
that may not transport to Kaggle. Only one training seed is currently observed.
Detection was not trained on a matched release; its annotation gates remained blocked. Detection is future work.

## Conclusion

The existing exploratory pilot documents a vocabulary-expansion trade-off on paired shared-class images and
additional recognition capability on halo/tidur under weak video labels. It does not establish a confirmatory
non-inferiority result. **Final conclusions: BELUM ADA**, pending protocol decisions and authorized experiments.

## References

1. H. Y. A. Swasono, A. R. Himamunanto, F. Maedjaja. MALCOM 5(3), 1061–1073, 2025.
   DOI [10.57152/malcom.v5i3.2130](https://doi.org/10.57152/malcom.v5i3.2130). Protocol observations above
   refer specifically to §§2.1–2.2; no numerical cross-paper ranking is attempted.
2. M. D. Kautsar, A. M. Hariono, R. Akmal. Attention vs LSTM: Improving Word-level BISINDO Recognition.
   [arXiv:2409.01975, v2](https://arxiv.org/abs/2409.01975), revised 2025. Preprint; not a task-matched benchmark.
