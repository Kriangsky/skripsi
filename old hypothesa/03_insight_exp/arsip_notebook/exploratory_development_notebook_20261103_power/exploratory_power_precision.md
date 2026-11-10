# exploratory/development — precision and power, not observed external results

FACT: Only seed42 has observed pilot results. Between-seed variance is unmeasured.
No external image or new training fit is used. NI margins .02/.03/.05 are planning scenarios only;
the candidate margin and H2 threshold remain unchanged. RNG20261103; 10,000 draws per scenario by default.

ASSUMPTIONS: true macro-F1 difference zero; A accuracy .93; correctness discordance .02/.05/.10;
paired seed SD0/.01/.02, five seed pairs sharing the same video sample. Joint wrong-label generator
is `calibrated_joint`: B retains extra output columns, with extra-class mass on discordant/both-wrong errors.
Oracle delta-method paired F1 variance plus independent Gaussian seed mean approximates intervals.
This is NOT exact hierarchical-bootstrap power, nor an estimate of unknown external domain shift.
Video independence assumed; signer/session correlation unmeasured. Five seeds are never five times the video n.

CV sizes: 5-fold over 300 internal videos/class implies 60 held-out/class, n360 common6 per fold;
combined OOF n1800. Those data include previously used train/val videos. OOF intervals below are optimistic
planning references: overlapping fitted training sets, configuration selection and signer dependence are
not modelled. They cannot establish independent confirmation or replace nested evaluation.

Illustration at 5% correctness discordance and hypothetical paired seed SD1 percentage point:

| domain | n_common6 | ni_margin | ci95_width_pp | ni_pass_probability | monte_carlo_se |
| --- | --- | --- | --- | --- | --- |
| external_common6 | 300 | 0.02 | 5.227237930196527 | 0.4434 | 0.004967861109169619 |
| external_common6 | 300 | 0.03 | 5.227237930196527 | 0.7258 | 0.004461102554302019 |
| external_common6 | 300 | 0.05 | 5.227237930196527 | 0.981 | 0.001365247230357931 |
| internal_CV_one_fold | 360 | 0.02 | 4.825164055086308 | 0.4931 | 0.004999523877330721 |
| internal_CV_one_fold | 360 | 0.03 | 4.825164055086308 | 0.7884 | 0.004084427009997853 |
| internal_CV_one_fold | 360 | 0.05 | 4.825164055086308 | 0.9915 | 0.0009180277773575238 |
| internal_CV_OOF_optimistic | 1800 | 0.02 | 2.6673910993597527 | 0.9029 | 0.0029609388713717133 |
| internal_CV_OOF_optimistic | 1800 | 0.03 | 2.6673910993597527 | 0.9976 | 0.0004893097178679326 |
| internal_CV_OOF_optimistic | 1800 | 0.05 | 2.6673910993597527 | 1.0 | 0.0 |

H2 n50/class, unchanged threshold .80. One-sided exact Clopper–Pearson lower bound must exceed .80.
Reference is a single classifier/perfectly correlated seed errors, not final five-seed hierarchical power.
Both-class product assumes independent class errors; Frechet bounds are in CSV.

| true_accuracy_assumed | n_per_class | required_correct | probability_pass_one_class | probability_pass_both_if_independent | both_probability_frechet_low | both_probability_frechet_high | method |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8 | 50 | 45 | 0.04802721937073372 | 0.00230661380048458 | 0.0 | 0.04802721937073372 | EXACT binomial + one-sided Clopper-Pearson; conditional single classifier/perfectly correlated seed errors; not pooled n=250 |
| 0.85 | 50 | 45 | 0.21935333540143803 | 0.04811588575173577 | 0.0 | 0.21935333540143803 | EXACT binomial + one-sided Clopper-Pearson; conditional single classifier/perfectly correlated seed errors; not pooled n=250 |
| 0.9 | 50 | 45 | 0.6161230077242774 | 0.37960756064721 | 0.23224601544855483 | 0.6161230077242774 | EXACT binomial + one-sided Clopper-Pearson; conditional single classifier/perfectly correlated seed errors; not pooled n=250 |
| 0.95 | 50 | 45 | 0.9622238270102226 | 0.9258746932661988 | 0.9244476540204452 | 0.9622238270102226 | EXACT binomial + one-sided Clopper-Pearson; conditional single classifier/perfectly correlated seed errors; not pooled n=250 |
| 0.98 | 50 | 45 | 0.9995217827857861 | 0.9990437942632762 | 0.9990435655715721 | 0.9995217827857861 | EXACT binomial + one-sided Clopper-Pearson; conditional single classifier/perfectly correlated seed errors; not pooled n=250 |

OPTIONS / CONSEQUENCES:

| Option | Consequence |
| --- | --- |
| Keep external n300 and scientifically fixed margin | Accept inconclusive NI outcomes when CI crosses boundary |
| Collect more independent external videos under a future approved protocol | More precision; cost and new collection before freeze |
| Use internal CV for development, retain external corroboration separately | More development data; selection bias and dependence remain |

No margin, H2 threshold or evidence domain is chosen here. Researcher decision and addendum locking these
items are mandatory BEFORE Stage2 CV tuning or Stage3; Stage1b still needs explicit scoped approval.
