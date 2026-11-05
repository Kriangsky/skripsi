# Confirmatory addendum implementation plan

Spec: user-supplied Hypotheses and Endpoints task, 2026-11-05.
Scope: additive implementation; historical frozen source, notebooks, protocols and results remain unchanged.

1. Audit preregistration, Addendum 01, tracker, latest manuscript, split_lock, run configs,
   predictions and checkpoint bytes. Record conflicts; never resolve scientific decisions implicitly.
2. Add `05_kode/src/interference_metrics.py` and synthetic tests: strict one-video alignment,
   full/restricted argmax, fixed-class metrics, recall changes, intrusion and decomposition.
3. Add `confirmatory_statistics.py` and tests: paired stratified video and hierarchical seed/video
   bootstrap, recomputed F1, simultaneous recall intervals, NI boundary and design-effect stress tests.
4. Add explicit `confirmatory_evaluation.py` CLI: development reports, validated immutable freeze,
   separately approved external scoring. Test rejection before inference, drift and overwrite protections.
5. Write draft addendum X.1–X.7, machine-readable plan, readiness/conflicts and reproducibility inventory.
   Preserve H2 verbatim. Mark NI_MARGIN_UNRESOLVED; no final freeze or external scoring.
6. Run synthetic suite, existing suite and internal five-seed report only. Record commands and blockers.

Architecture: NumPy/pandas/SciPy already installed; reuse project class mapping and pixel preprocessing.
New modules are independent entry points, with no notebook auto-execution and no training calls.
Mean across all five paired seed scores is the estimand; videos, not seed-video rows, are sampling units.
Risk checks: missing/duplicate videos, mismatched labels/content, seed selection, dropped novel outputs,
unknown provenance, unsupported task, unresolved margin, tampered freeze and repeat scoring.

Ruling: user explicitly requests implementation now; proceed without a separate design approval.
Scientific conflicts block finalization, not synthetic implementation or development analysis.
