# Precision/power planning — FACT / ASSUMPTION / PROPOSAL

## FACT

Only training seed42 exists; its reproduction is not a new seed. Actual paired counts are in paired analysis.
No seed variance has been measured. No external images/scores were read; zero model-training fits.
Correctness discordance is not the same as changed predicted label or macro-F1 influence.

## SIMULATION ASSUMPTIONS

External common6 planned n=300, 50 independent videos/GT class, five paired seeds. Hypothetical A accuracy .93,
correctness discordance .02/.05/.10, calibrated TRUE macro-F1 effects -.02/-.01/0/+.01, paired seed SD
0/.005/.01/.02. B predictions retain two extra outputs; macro-F1 is computed from full confusion, not accuracy.
Wrong-label choices: 20% extra B output for A-only errors, 10% for both-wrong. Both-wrong labels half shared,
half independently drawn. This is a disclosed generator, not estimated external behavior. Impossible combinations
are flagged in CSV, not dropped or assigned invented power.

Same video sample across seeds; independent Gaussian paired-seed mean SD/sqrt(5) adds unobserved seed variability.
95% CI width uses oracle delta-method video variance plus seed variance. NI uses one-sided normal critical 1.645.
This is APPROXIMATE planning power, not calibration of the final hierarchical percentile-bootstrap test.
It assumes independent videos; unknown signer/session clustering and domain shift can reduce precision.
Conditional empirical transport is optimistic: original error patterns only, no unseen discordances.
Five seeds are not 1,500 independent external videos. Report Monte Carlo SE in CSV.

### H1 at true macro-F1 delta zero

| correctness_discordance_assumed | paired_seed_sd_assumed | ci95_width_pp | ni_pass_probability | monte_carlo_se |
| --- | --- | --- | --- | --- |
| 0.02 | 0.0 | 3.1768589227584174 | 0.7886 | 0.004083014082757982 |
| 0.02 | 0.01 | 3.6284431470900955 | 0.697 | 0.004595552197505758 |
| 0.02 | 0.02 | 4.731289553740183 | 0.5013 | 0.004999983099971439 |
| 0.05 | 0.0 | 4.924515135760065 | 0.4871 | 0.004998335622984915 |
| 0.05 | 0.01 | 5.227237930196527 | 0.4434 | 0.004967861109169619 |
| 0.05 | 0.02 | 6.0451234518867505 | 0.3642 | 0.004812051121922958 |
| 0.1 | 0.0 | 6.920882808560376 | 0.3099 | 0.004624521488759675 |
| 0.1 | 0.01 | 7.139452773594197 | 0.2952 | 0.004561326122960296 |
| 0.1 | 0.02 | 7.758304394392357 | 0.2688 | 0.0044333571929182514 |

### H2, 50 videos per new class, threshold unchanged at .80

| true_accuracy_assumed | required_correct | probability_pass_one_class | probability_pass_both_if_independent |
| --- | --- | --- | --- |
| 0.8 | 45 | 0.04802721937073372 | 0.00230661380048458 |
| 0.85 | 45 | 0.21935333540143803 | 0.04811588575173577 |
| 0.9 | 45 | 0.6161230077242774 | 0.37960756064721 |
| 0.95 | 45 | 0.9622238270102226 | 0.9258746932661988 |
| 0.98 | 45 | 0.9995217827857861 | 0.9990437942632762 |

H2 above uses one-sided 95% Clopper-Pearson, a conservative transparent binomial reference for a fixed classifier,
or perfectly correlated errors across seeds. It is NOT exact power for the preregistered five-seed hierarchical CI.
No observed between-seed new-class variance exists. Both-pass product assumes independence between classes;
Frechet bounds in CSV allow unknown dependence. Do not pool seeds into n=250/class or replace primary CI silently.

## PROPOSAL — decision required, no protocol changes enacted

300 common6 videos may support NI when discordance and seed variation are small, but cannot guarantee a narrow CI.
Use the scenario tables, not the optimistic one-discordance pilot alone. Keep external confirmation with a prespecified
inconclusive outcome if desired; larger independently collected external data is an alternative requiring a new protocol.
Internal CV offers more development evidence, but reused/tuned internal folds do NOT become independent confirmation.
If CV is designated the main H1 evidence, explicitly limit the claim to development and report external corroboration
separately. Selection and testing on the same CV require nested separation for stronger inferential interpretation.
Do not widen delta or lower H2 merely to pass: any scientifically justified change needs researcher decision and
Addendum 02 before prospective training. Delta=.02 and H2=.80 remain unchanged. This analysis cannot decide for
the researcher. No claim that current single-seed pilot has confirmed H1 or H2.
