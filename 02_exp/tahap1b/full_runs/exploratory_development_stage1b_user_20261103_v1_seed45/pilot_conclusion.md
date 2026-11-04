# FACT — exploratory Classification, seed 45

30 epochs; input 224; fixed last.pt; independent initialization.

| model | scope | n | correct_n | accuracy | macro_f1 | macro_f1_definition | extra_class_predictions_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | common6 | 120 | 115 | 0.9583333333333334 | 0.956626061277224 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | common6 | 120 | 111 | 0.925 | 0.9238073337649323 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | full8 | 160 | 149 | 0.93125 | 0.9305668272045128 | Mean F1 over fixed ground-truth classes 0..7; extra predicted classes remain errors/FN. | 0 |

Common6 retains all B outputs. Full8 is a separate task. No claim of verified semantics or signer independence.
Technical flags retained: 497. Actual steps: {'A': 690, 'B': 900}.
No external evaluation, application benchmark or measured inference cost in this report. One seed is descriptive; no confirmatory conclusion.
