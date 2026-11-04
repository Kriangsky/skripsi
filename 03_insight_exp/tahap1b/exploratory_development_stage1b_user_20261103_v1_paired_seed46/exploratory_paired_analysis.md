# exploratory/development — FACT: existing pilot paired analysis

| metric | A | B | delta_B_minus_A | ci95_low | ci95_high | lower_one_sided95 | bootstrap_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.9309718426908904 | 0.9391616672821838 | 0.00818982459129336 | -0.0007879026854230586 | 0.025582300650552676 | -0.0005414324756356503 | 0.007974774853003855 |
| accuracy | 0.9333333333333333 | 0.9416666666666667 | 0.008333333333333304 | 0.0 | 0.025 | 0.0 | 0.008121741809427452 |

Paired counts: {'both_correct': 112, 'a_only': 0, 'b_only': 1, 'both_wrong': 7}. Exact two-sided McNemar p=1.0. Union errors=8.

Wilson intervals apply to accuracy, not macro-F1. Paired bootstrap resamples videos within GT strata; extra B outputs retained. One training seed only; no measured seed variance. These post-hoc internal results are not confirmatory NI evidence. Bootstrap cannot invent unseen error categories; signer/session dependence may widen uncertainty.
