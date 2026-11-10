# exploratory/development — all five seeds

| model | mean | std |
| --- | --- | --- |
| A | 0.9398997391307502 | 0.011577652345406817 |
| B | 0.9311704349008177 | 0.00547786904530329 |

{'status': 'exploratory/development', 'training_seeds': [42, 43, 44, 45, 46], 'seed42_is_replication': True, 'video_n': 120, 'mean_delta': -0.008729304229932344, 'ci95': [-0.03010145130566371, 0.007667945291901918], 'lower_one_sided95': -0.02558950457581278, 'video_only_ci95': [-0.02017397590776844, -4.169272461960138e-05], 'paired_delta_seed_sd': 0.01589842145086347, 'seed_only_ci95': [-0.02245936208819712, 0.002236875163160068], 'stop_cannot_decide': True, 'proceed_to_stage2': False, 'pooled_discordant_seed_video_counts': {'a_only': 8, 'b_only': 1}, 'pooled_mcnemar_descriptive_only': 0.0390625, 'warning': 'Pooled McNemar p is descriptive only: seed-video trials share videos, independence is false. No inferential pooled claim.', 'seed_variability': [{'model': 'A', 'mean': 0.9398997391307502, 'std': 0.011577652345406817}, {'model': 'B', 'mean': 0.9311704349008177, 'std': 0.00547786904530329}]}

FACT: all seed42..46 retained; seed42 replication. HYPOTHESIS: context/overfit explanations require diagnostics. PROPOSAL: request researcher decision; never auto-launch Stage2.
