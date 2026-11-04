# exploratory/development — FACT: existing pilot paired analysis

| metric | A | B | delta_B_minus_A | ci95_low | ci95_high | lower_one_sided95 | bootstrap_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.9326249749869775 | 0.9308942013767204 | -0.0017307736102570859 | -0.024446151375811145 | 0.014827119478282084 | -0.019854161090689493 | 0.010158944539443554 |
| accuracy | 0.9333333333333333 | 0.925 | -0.008333333333333304 | -0.025 | 0.0 | -0.025 | 0.008088024881005169 |

Paired counts: {'both_correct': 111, 'a_only': 1, 'b_only': 0, 'both_wrong': 8}. Exact two-sided McNemar p=1.0. Union errors=9.

Wilson intervals apply to accuracy, not macro-F1. Paired bootstrap resamples videos within GT strata; extra B outputs retained. One training seed only; no measured seed variance. These post-hoc internal results are not confirmatory NI evidence. Bootstrap cannot invent unseen error categories; signer/session dependence may widen uncertainty.
