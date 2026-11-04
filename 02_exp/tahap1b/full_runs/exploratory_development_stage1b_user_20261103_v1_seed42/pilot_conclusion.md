# FACT — exploratory Classification, seed 42

30 epochs; input 224; fixed last.pt; independent initialization.

| model | scope | n | correct_n | accuracy | macro_f1 | macro_f1_definition | extra_class_predictions_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | common6 | 120 | 112 | 0.9333333333333333 | 0.9317756044925641 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | common6 | 120 | 111 | 0.925 | 0.9300418783462919 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 2 |
| B | full8 | 160 | 150 | 0.9375 | 0.9354640789917368 | Mean F1 over fixed ground-truth classes 0..7; extra predicted classes remain errors/FN. | 0 |

Common6 retains all B outputs. Full8 is a separate task. No claim of verified semantics or signer independence.
Technical flags retained: 497. Actual steps: {'A': 690, 'B': 900}.
No external evaluation, application benchmark or measured inference cost in this report. One seed is descriptive; no confirmatory conclusion.
