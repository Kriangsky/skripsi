# exploratory/development — Stage 1a implementation plan

2026-11-03. Baseline: 81 unittest tests passed in 100.083s. Zero model fits authorized.

Reuse `analysis_paired.py`, `classification_research.py`, `power_precision.py` and legacy data/integrity helpers.
Keep candidate protocol, addenda, manuscripts, source datasets, splits and completed runs unchanged.
Researcher explicitly authorized updating the Stage 1b code gate in this chat: scoped approval suffices;
an addendum remains mandatory before Stage 2/3. No approval record will be fabricated.

1. Test and implement approval gate, no-augmentation train top-1/loss per epoch with model/RNG restoration.
2. Test and implement cached-landmark masking before letterbox; retain unavailable strata and save masks/coverage.
3. Test and implement frozen eval backbone feature extraction and six-class SGD linear heads, fixed last epoch.
4. Extend planning scenarios across NI margins .02/.03/.05 and internal CV sample sizes; disclose assumptions.
5. Create requested notebook calling modules only, explicit training=False gate and fresh output paths.
6. Recompute existing-pilot paired analysis and planning tables into a fresh exploratory output directory.
   Measure inference-only smoke; never infer an unmeasured head-training duration from it.
7. Run new and old tests, check notebook syntax/guards without Detection verifier, verify protected hashes.

Review focus: duplicate/mismatched identity/hash, missing landmarks, BN/RNG mutation, training without approval,
partial outputs/resume, all five seeds retained, six-class probe labels, extra B predictions retained.
Stop at A–E. Stage 1b proposal: A/B x seeds42..46, 224/30, plus ten heads; seed42 is replication.
