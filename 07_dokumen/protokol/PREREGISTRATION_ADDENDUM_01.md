# PREREGISTRATION ADDENDUM 01 — v1.1, 2026-11-03

Parent: `PREREGISTRATION.md`, candidate v1, SHA256
`24e21743168f3457933f0f17db76e52d420b5f005da774a7f9230c71f1ac7a70`.
Parent file and its hash remain unchanged. This addendum and its SHA256 form the next protocol version.

## Approved scope — researcher decision, 2026-11-03

Stage 0 is approved subject to this addendum. The deliverable is a **full conference paper only**.
No demonstration, webcam product, or OpenCV application integration is planned. Existing image preparation
dependencies may still be reused; they do not imply an application deliverable.

The new title is **Effect of Vocabulary Expansion on YOLO26 Classification for BISINDO Phrase Recognition**.
Remove application-speed claims from the new title, abstract, method, and hypotheses. Historical manuscripts,
notebooks, run logs and bibliographic titles are immutable records, not the current study's claims.

The timing endpoint is **inference cost**, replacing latency terminology in the prospective paper/protocol.
Only measured values may be reported: batch1, FP32, 100 warmups, CUDA synchronization,
1,000 timed inputs ×5 repeats; preprocessing+transfer+inference+argmax, hardware and variance specified.
Any decode-inclusive cost is separate. Throughput does not establish application suitability.
No timing benchmark is executed in Stage 1a.

Swasono et al. (MALCOM 2025) is related work, not a numerical comparator. Detection/classification,
data, split, and endpoint differences preclude a matched comparison. Statements about their protocol must
cite the original paper, distinguish reported procedures from unspecified details, and avoid allegations.

## Authorization and unresolved decisions

Authorized now: A addendum; B read-only precision/power analysis; C Stage 1a code/tests and existing-artifact
paired analysis; D new paper skeleton. **Zero model-training fits authorized**, including smoke training.
Statistical Monte Carlo simulations are permitted; these are not model fits and do not read external test images.

H1 delta remains **0.02**. H2 threshold remains **0.80 for each new class**, with lower one-sided 95% CI
strictly above threshold for confirmatory success. Seeds remain 42–46. No threshold or domain change is made here.
External common6 has planned 300 videos (50/class); full8 400 includes those 300. External scores remain unopened.
Kaggle was historically preprocessed; the approved restriction remains one final scoring batch after freeze.

Analysis B will assess precision under disclosed discordance and paired seed-SD scenarios because only seed42
has been observed. Reproducing seed42 is not a second independent seed. Simulated power is conditional on its
generator/CI approximation, not measured external performance or an exact forecast of hierarchical-bootstrap power.
Possible changes to delta, H2 threshold or H1 evidence domain require researcher decision and **Addendum 02
before any prospective model training**. CV on reused internal data must remain labelled development evidence;
it cannot be renamed an untouched confirmatory test because it has more videos.

## Compatibility implementation

The Stage 0 snapshot tests protect original source bytes. New `classification_research.py` provides implemented
seed/epochs/imgsz/work options while reusing legacy sampling/selection, integrity and evaluation helpers.
Original commands and tests remain untouched; no fictitious flags are documented for the legacy CLI.
New prepared runs lock settings and require explicit scoped approval records before train; no overwrite/resume.
Every generated report distinguishes artifact facts, simulation assumptions and proposals.

This addendum authorizes implementation and analysis, not training or final confirmation. A final freeze still
requires configuration, manifests, source/dependency hashes, checkpoints, approval and external scoring plan.
