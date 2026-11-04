# exploratory/development — FACT: existing pilot paired analysis

| metric | A | B | delta_B_minus_A | ci95_low | ci95_high | lower_one_sided95 | bootstrap_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.956626061277224 | 0.9238073337649323 | -0.03281872751229176 | -0.06763508573527009 | -0.006886281750210443 | -0.06049979367324163 | 0.016218074145091108 |
| accuracy | 0.9583333333333334 | 0.925 | -0.033333333333333326 | -0.06666666666666667 | -0.008333333333333333 | -0.058333333333333334 | 0.01592487723759949 |

Paired counts: {'both_correct': 111, 'a_only': 4, 'b_only': 0, 'both_wrong': 5}. Exact two-sided McNemar p=0.125. Union errors=9.

Wilson intervals apply to accuracy, not macro-F1. Paired bootstrap resamples videos within GT strata; extra B outputs retained. One training seed only; no measured seed variance. These post-hoc internal results are not confirmatory NI evidence. Bootstrap cannot invent unseen error categories; signer/session dependence may widen uncertainty.
