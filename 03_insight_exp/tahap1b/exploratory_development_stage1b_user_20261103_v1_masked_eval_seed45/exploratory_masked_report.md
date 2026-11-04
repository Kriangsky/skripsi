# exploratory/development — hand masked

| model | stratum | scope | balanced_task_chance | n | correct_n | accuracy | macro_f1 | macro_f1_definition | extra_class_predictions_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | landmarks_available | common6 | 0.16666666666666666 | 120 | 113 | 0.9416666666666667 | 0.9406391242376554 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | landmarks_available | common6 | 0.125 | 120 | 110 | 0.9166666666666666 | 0.9149816409421746 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | landmarks_available | full8 | 0.125 | 160 | 141 | 0.88125 | 0.8786686157668278 | Mean F1 over fixed ground-truth classes 0..7; extra predicted classes remain errors/FN. | 0 |
| A | all_including_unavailable | common6 | 0.16666666666666666 | 120 | 113 | 0.9416666666666667 | 0.9406391242376554 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | all_including_unavailable | common6 | 0.125 | 120 | 110 | 0.9166666666666666 | 0.9149816409421746 | Mean F1 over fixed ground-truth classes 0..5; extra predicted classes remain errors/FN. | 0 |
| B | all_including_unavailable | full8 | 0.125 | 160 | 141 | 0.88125 | 0.8786686157668278 | Mean F1 over fixed ground-truth classes 0..7; extra predicted classes remain errors/FN. | 0 |

Chance is a uniform-output reference, not an empirical null for imbalanced strata. Missing landmarks remain in overall denominator. Face/body may remain; distribution shift and incomplete hand landmarks prevent a pure-background causal claim.
