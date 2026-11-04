# exploratory/development — hand masked

| model | stratum | scope | balanced_task_chance | n | correct_n | accuracy | macro_f1 | macro_f1_definition | extra_class_predictions_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | landmarks_available | common6 | 0.16666666666666666 | 120 | 110 | 0.9166666666666666 | 0.914604258257509 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | landmarks_available | common6 | 0.125 | 120 | 109 | 0.9083333333333333 | 0.9055526702585527 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | landmarks_available | full8 | 0.125 | 160 | 147 | 0.91875 | 0.916875829574728 | Mean F1 over fixed ground-truth classes 0..7; extra predicted classes remain errors/FN. | 0 |
| A | all_including_unavailable | common6 | 0.16666666666666666 | 120 | 110 | 0.9166666666666666 | 0.914604258257509 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | all_including_unavailable | common6 | 0.125 | 120 | 109 | 0.9083333333333333 | 0.9055526702585527 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | all_including_unavailable | full8 | 0.125 | 160 | 147 | 0.91875 | 0.916875829574728 | Mean F1 over fixed ground-truth classes 0..7; extra predicted classes remain errors/FN. | 0 |

Chance is a uniform-output reference, not an empirical null for imbalanced strata. Missing landmarks remain in overall denominator. Face/body may remain; distribution shift and incomplete hand landmarks prevent a pure-background causal claim.
