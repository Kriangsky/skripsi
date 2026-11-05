# development evaluation

Mean paired common6 delta macro-F1: -0.00872930. Hierarchical percentile CI95% [-0.03022282, 0.00761214]; one-sided lower95% -0.02533393.

N=120 common6 videos; 160 full8 videos; all seeds [42, 43, 44, 45, 46]. 10000 bootstrap replicates, RNG 20261103. NI: not tested: development evidence. Margin status: NI_MARGIN_UNRESOLVED. H2: not tested: development evidence; lower bounds {'halo': 0.8300000000000001, 'tidur': 0.96}.

Sampling: resample videos within GT strata, same draw for A/B and all seeds; independently resample paired training seeds. Recompute F1 before averaging seed differences. Video-only intervals condition on observed seeds; seed-only intervals condition on observed videos. Five seeds give limited training-variability information. Seeds are not independent test datasets.

Class-interference mean rows contain marginal recall and difference CIs, plus Bonferroni simultaneous nominal 95% intervals over six differences (approximate bootstrap coverage). No class is declared most affected from rank alone. Full8 is a separate task. Restricted argmax and accuracy decomposition are diagnostic, without causal attribution.

Design-effect normal-SE stress tests: [{"design_effect": 1, "effective_n_multiplier": 1.0, "adjusted_se": 0.009390094823896081, "ci95_low": -0.027133551896184646, "ci95_high": 0.009674943436319958, "lower_one_sided95": -0.024174635758436057, "assumed_effective_video_n": 120.0}, {"design_effect": 1.5, "effective_n_multiplier": 0.6666666666666666, "adjusted_se": 0.011500470477447431, "ci95_low": -0.031269812170995476, "ci95_high": 0.013811203711130784, "lower_one_sided95": -0.02764589480641008, "assumed_effective_video_n": 80.0}, {"design_effect": 2, "effective_n_multiplier": 0.5, "adjusted_se": 0.013279599451923238, "ci95_low": -0.034756840884819735, "ci95_high": 0.017298232424955047, "lower_one_sided95": -0.030572301552891066, "assumed_effective_video_n": 60.0}]. Multiplying SE by sqrt(DE) is an assumption, not signer-cluster resampling. Unknown signer metadata prevents signer-population inference; one external signer would not support it. No measured efficiency endpoint is available from this analysis.
