# exploratory/development — FACT: existing pilot paired analysis

| metric | A | B | delta_B_minus_A | ci95_low | ci95_high | lower_one_sided95 | bootstrap_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.9475002122060946 | 0.9319470937339606 | -0.015553118472134031 | -0.0399593946105572 | 0.0 | -0.03297199638663073 | 0.010548510710031805 |
| accuracy | 0.95 | 0.9333333333333333 | -0.016666666666666607 | -0.041666666666666664 | 0.0 | -0.03333333333333333 | 0.011161923736213136 |

Paired counts: {'both_correct': 112, 'a_only': 2, 'b_only': 0, 'both_wrong': 6}. Exact two-sided McNemar p=0.5. Union errors=8.

Wilson intervals apply to accuracy, not macro-F1. Paired bootstrap resamples videos within GT strata; extra B outputs retained. One training seed only; no measured seed variance. These post-hoc internal results are not confirmatory NI evidence. Bootstrap cannot invent unseen error categories; signer/session dependence may widen uncertainty.
