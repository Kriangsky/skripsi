# FACT — exploratory Classification, seed 43

30 epochs; input 224; fixed last.pt; independent initialization.

| model | scope | n | correct_n | accuracy | macro_f1 | macro_f1_definition | extra_class_predictions_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | common6 | 120 | 112 | 0.9333333333333333 | 0.9326249749869775 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | common6 | 120 | 111 | 0.925 | 0.9308942013767204 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 2 |
| B | full8 | 160 | 149 | 0.93125 | 0.9298634501033882 | Mean F1 over fixed ground-truth classes 0..7; extra predicted classes remain errors/FN. | 0 |

Common6 retains all B outputs. Full8 is a separate task. No claim of verified semantics or signer independence.
Technical flags retained: 497. Actual steps: {'A': 690, 'B': 900}.
No external evaluation, application benchmark or measured inference cost in this report. One seed is descriptive; no confirmatory conclusion.
