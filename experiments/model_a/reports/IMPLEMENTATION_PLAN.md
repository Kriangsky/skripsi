# Model A implementation plan

Goal: isolated six-class YOLO26 baseline; no official fit during construction.
Scope: new notebook, one importable helper, configuration, tests and evidence;
historical archive with compatibility junctions. No old training helper imports.

- Audit frozen identities, current environment, preprocessing, selected frames,
  review responses and callers before moves. Preserve data/provenance hashes.
- Archive historical runs/notebooks/reports using Git moves where tracked;
  preserve old directory access with junctions. Leave source manifests, caches,
  environment, code, preparation notebooks and active audit/protocol paths.
- Reuse historical selected-candidate identities only. Export incomplete
  1,800-row decision template; never invent missing frame selections.
- Block official fits on incomplete selection or unresolved semantic concerns.
  Technical smoke uses a separate, explicitly labelled subset of existing frames.
- Use one RGB letterbox transform for trainer, validator and standalone inference.
  Select best checkpoint by internal top-1; export canonical/output mappings.
- Test leakage/mapping/coverage guards, preprocessing, metrics, notebook execution,
  official checkpoint loading and a separate one-epoch smoke fit if feasible.

Risks checked: path escapes, changed source bytes, duplicate identities/content,
external contamination, incomplete frames, review ID namespace differences,
class-folder ordering, reruns/overwrite, serialization of custom transforms.
No Model B/C, tuning, new frame-selection recipe or external evaluation.
