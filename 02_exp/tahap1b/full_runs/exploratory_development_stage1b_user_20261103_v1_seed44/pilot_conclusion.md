# FACT — exploratory Classification, seed 44

30 epochs; input 224; fixed last.pt; independent initialization.

| model | scope | n | correct_n | accuracy | macro_f1 | macro_f1_definition | extra_class_predictions_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | common6 | 120 | 114 | 0.95 | 0.9475002122060946 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | common6 | 120 | 112 | 0.9333333333333333 | 0.9319470937339606 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | full8 | 160 | 151 | 0.94375 | 0.9428515916307662 | Mean F1 over fixed ground-truth classes 0..7; extra predicted classes remain errors/FN. | 0 |

Common6 retains all B outputs. Full8 is a separate task. No claim of verified semantics or signer independence.
Technical flags retained: 497. Actual steps: {'A': 690, 'B': 900}.
No external evaluation, application benchmark or measured inference cost in this report. One seed is descriptive; no confirmatory conclusion.
