# exploratory/development — Tahap1b completed

## FACT

Five paired seeds42–46 retained; seed42 replication. Ten full fits total: two researcher-completed seed42 fits reused without retraining, eight additional full fits by agent. Ten completed new head fits. A failed zero-optimization probe attempt and preflight failure remain in failure history. Fresh processes fix per-work Ultralytics settings isolation without changing locked source. No overwrite/resume, external scoring, Kaggle tuning, threshold change or Stage2.

## FACT — model mean ± SD (ddof1), internal validation only

| model | scope | n_seeds | macro_f1_mean | macro_f1_sd | accuracy_mean | accuracy_sd |
| --- | --- | --- | --- | --- | --- | --- |
| A | common6 | 5 | 0.9398997391307502 | 0.011577652345406817 | 0.9416666666666667 | 0.011785113019775863 |
| B | common6 | 5 | 0.9311704349008177 | 0.0054778690453032575 | 0.93 | 0.00745355992499929 |
| B | full8 | 5 | 0.9361928436021125 | 0.006182415991676225 | 0.9375 | 0.006249999999999978 |

## FACT — paired delta and conditional per-seed CI

| metric | A | B | delta_B_minus_A | ci95_low | ci95_high | lower_one_sided95 | bootstrap_sd | seed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.931775604492564 | 0.930041878346292 | -0.0017337261462722 | -0.0256793478574532 | 0.015664552249918 | -0.0208419481262939 | 0.0106703516007811 | 42 |
| macro_f1 | 0.9326249749869776 | 0.9308942013767204 | -0.001730773610257 | -0.0244461513758111 | 0.014827119478282 | -0.0198541610906894 | 0.0101589445394435 | 43 |
| macro_f1 | 0.9475002122060946 | 0.9319470937339606 | -0.015553118472134 | -0.0399593946105572 | 0.0 | -0.0329719963866307 | 0.0105485107100318 | 44 |
| macro_f1 | 0.956626061277224 | 0.9238073337649324 | -0.0328187275122917 | -0.06763508573527 | -0.0068862817502104 | -0.0604997936732416 | 0.0162180741450911 | 45 |
| macro_f1 | 0.9309718426908904 | 0.9391616672821838 | 0.0081898245912933 | -0.000787902685423 | 0.0255823006505526 | -0.0005414324756356 | 0.0079747748530038 | 46 |

## FACT — shared-video hierarchical bootstrap10000, RNG20261103

```json
{
  "status": "exploratory/development",
  "training_seeds": [
    42,
    43,
    44,
    45,
    46
  ],
  "seed42_is_replication": true,
  "video_n": 120,
  "mean_delta": -0.008729304229932344,
  "ci95": [
    -0.03010145130566371,
    0.007667945291901918
  ],
  "lower_one_sided95": -0.02558950457581278,
  "video_only_ci95": [
    -0.02017397590776844,
    -4.169272461960138e-05
  ],
  "paired_delta_seed_sd": 0.01589842145086347,
  "seed_only_ci95": [
    -0.02245936208819712,
    0.002236875163160068
  ],
  "stop_cannot_decide": true,
  "proceed_to_stage2": false,
  "pooled_discordant_seed_video_counts": {
    "a_only": 8,
    "b_only": 1
  },
  "pooled_mcnemar_descriptive_only": 0.0390625,
  "warning": "Pooled McNemar p is descriptive only: seed-video trials share videos, independence is false. No inferential pooled claim.",
  "seed_variability": [
    {
      "model": "A",
      "mean": 0.9398997391307502,
      "std": 0.011577652345406817
    },
    {
      "model": "B",
      "mean": 0.9311704349008177,
      "std": 0.00547786904530329
    }
  ]
}
```

Pooled McNemar is descriptive only: repeated seed-video observations are dependent. Per-seed exact McNemar in paired reports remains separate.

## FACT — frozen eval backbone, six-class last linear head

| model | n_seeds | macro_f1_mean | macro_f1_sd | accuracy_mean | accuracy_sd |
| --- | --- | --- | --- | --- | --- |
| A | 5 | 0.9334267350112755 | 0.004827821969261903 | 0.9349999999999999 | 0.0037267799624996394 |
| B | 5 | 0.9259280639876248 | 0.009164240362929659 | 0.9283333333333333 | 0.007453559924999291 |

## FACT — hand masked vs uniform-output chance

| model | scope | n_seeds | accuracy_mean | accuracy_sd | chance | accuracy_minus_chance |
| --- | --- | --- | --- | --- | --- | --- |
| A | common6 | 5 | 0.9199999999999999 | 0.031512784001981736 | 0.1666666666666666 | 0.7533333333333333 |
| B | common6 | 5 | 0.9166666666666666 | 0.013176156917368235 | 0.125 | 0.7916666666666666 |
| B | full8 | 5 | 0.89125 | 0.020539595906443733 | 0.125 | 0.76625 |

All video denominators retained, unavailable-landmark strata retained when present. Face/body may remain; imbalance/shift prevents a pure-background causal claim.

## FACT — epoch30 curves

| seed | model | epoch30_train_top1_actual_weights | epoch30_val_top1_EMA |
| --- | --- | --- | --- |
| 42 | A | 1.0 | 0.93333 |
| 42 | B | 1.0 | 0.9375 |
| 43 | A | 1.0 | 0.93333 |
| 43 | B | 1.0 | 0.93125 |
| 44 | A | 1.0 | 0.95 |
| 44 | B | 1.0 | 0.94375 |
| 45 | A | 1.0 | 0.95833 |
| 45 | B | 1.0 | 0.93125 |
| 46 | A | 1.0 | 0.93333 |
| 46 | B | 1.0 | 0.94375 |

Training diagnostic uses actual weights; framework validation uses EMA. Their gap is descriptive, not an identically weighted-model comparison. Full per-epoch plots are saved for all seeds.

## FACT — first-phase recorded timing

```json
{
  "prior_prepare_seconds": 71.91250240000045,
  "prior_pair_train_evaluation_seconds": 139.92973619999975,
  "prior_active_phase_including_failure_seconds": 215.36696099999972,
  "A_recorded_training_seconds": 53.1245,
  "B_recorded_training_seconds": 67.186,
  "probe_A_fresh_process_seconds": 14.137825099999645,
  "probe_B_fresh_process_seconds": 14.666812699999355
}
```

Recorded A/B training excludes prepare and other pipeline phases. Probe child wall times include startup/extraction/evaluation. Prior active phase includes its failed attempt; interactive gap excluded. No estimated times substituted.

## FACT — process phase timing

| label | action | seed | model | seconds | returncode | log |
| --- | --- | --- | --- | --- | --- | --- |
| exploratory/development | probe | 42 | A | 14.137825099999645 | 0 | exploratory_probe_seed42_A.log |
| exploratory/development | probe | 42 | B | 14.666812699999355 | 0 | exploratory_probe_seed42_B.log |
| exploratory/development | prepare | 43 | A | 71.3357447999997 | 0 | exploratory_prepare_seed43_A.log |
| exploratory/development | train | 43 | A | 182.21807099999933 | 0 | exploratory_train_seed43_A.log |
| exploratory/development | probe | 43 | A | 13.941131999999925 | 0 | exploratory_probe_seed43_A.log |
| exploratory/development | probe | 43 | B | 14.965242899999794 | 0 | exploratory_probe_seed43_B.log |
| exploratory/development | prepare | 44 | A | 73.87044590000005 | 0 | exploratory_prepare_seed44_A.log |
| exploratory/development | train | 44 | A | 207.94860889999927 | 0 | exploratory_train_seed44_A.log |
| exploratory/development | probe | 44 | A | 14.158367399999406 | 0 | exploratory_probe_seed44_A.log |
| exploratory/development | probe | 44 | B | 13.227277500000127 | 0 | exploratory_probe_seed44_B.log |
| exploratory/development | prepare | 45 | A | 70.71863920000033 | 0 | exploratory_prepare_seed45_A.log |
| exploratory/development | train | 45 | A | 149.3586132999999 | 0 | exploratory_train_seed45_A.log |
| exploratory/development | probe | 45 | A | 13.3003075000006 | 0 | exploratory_probe_seed45_A.log |
| exploratory/development | probe | 45 | B | 13.304021400000238 | 0 | exploratory_probe_seed45_B.log |
| exploratory/development | prepare | 46 | A | 71.41155810000055 | 0 | exploratory_prepare_seed46_A.log |
| exploratory/development | train | 46 | A | 149.60494930000004 | 0 | exploratory_train_seed46_A.log |
| exploratory/development | probe | 46 | A | 13.353178899999875 | 0 | exploratory_probe_seed46_A.log |
| exploratory/development | probe | 46 | B | 13.28686099999959 | 0 | exploratory_probe_seed46_B.log |
| exploratory/development | masks | 42 | A | 24.58869759999925 | 0 | exploratory_masks_seed42_A.log |
| exploratory/development | masked | 42 | A | 12.948333000000275 | 0 | exploratory_masked_seed42_A.log |
| exploratory/development | paired | 42 | A | 2.3885832000005394 | 0 | exploratory_paired_seed42_A.log |
| exploratory/development | curves | 42 | A | 2.789242699999704 | 0 | exploratory_curves_seed42_A.log |
| exploratory/development | masked | 43 | A | 12.978778499999862 | 0 | exploratory_masked_seed43_A.log |
| exploratory/development | paired | 43 | A | 2.3555665999992925 | 0 | exploratory_paired_seed43_A.log |
| exploratory/development | curves | 43 | A | 2.7332553000005646 | 0 | exploratory_curves_seed43_A.log |
| exploratory/development | masked | 44 | A | 12.899313500000062 | 0 | exploratory_masked_seed44_A.log |
| exploratory/development | paired | 44 | A | 2.2956162999998924 | 0 | exploratory_paired_seed44_A.log |
| exploratory/development | curves | 44 | A | 2.6253930000002583 | 0 | exploratory_curves_seed44_A.log |
| exploratory/development | masked | 45 | A | 12.762123299999985 | 0 | exploratory_masked_seed45_A.log |
| exploratory/development | paired | 45 | A | 2.297129600000517 | 0 | exploratory_paired_seed45_A.log |
| exploratory/development | curves | 45 | A | 2.6574264000000767 | 0 | exploratory_curves_seed45_A.log |
| exploratory/development | masked | 46 | A | 12.602011799999673 | 0 | exploratory_masked_seed46_A.log |
| exploratory/development | paired | 46 | A | 2.3055057000001398 | 0 | exploratory_paired_seed46_A.log |
| exploratory/development | curves | 46 | A | 2.6808378000005177 | 0 | exploratory_curves_seed46_A.log |
| exploratory/development | summary | 42 | A | 3.0767336999997497 | 0 | exploratory_summary_seed42_A.log |

## FACT — failures, retained

```json
[
  {
    "status_label": "exploratory/development",
    "event": "hard_stop",
    "seed": 42,
    "kind": "linear_probe",
    "model": "A",
    "error": "RuntimeError: Ultralytics already imported with different settings directory; run pilot CLI in a fresh process"
  },
  {
    "event": "runner_preflight_failed_before_any_fit",
    "error": "Empty-directory assertion counted an empty settings subdirectory as a file; corrected check in v2; v1 session preserved."
  }
]
```

## HYPOTHESIS

Context cues, weak labels and overfit are possible explanations. Masking/probes do not identify a causal linguistic mechanism. Signer/session independence unverified.

## PROPOSAL

STOP after Stage1b. Candidate delta/H2 unchanged; internal pilot remains exploratory. If seed SD >=abs(mean paired delta), or interval crosses -delta, pilot cannot decide; do not chase seeds. Stage2 requires researcher decision, addendum locking delta/H2/domain, validated folds/grid/budget, and separate approval.
