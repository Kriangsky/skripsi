# Exploratory YOLO26 classification pilot

Two independent ImageNet -> BISINDO runs; 30 epochs; seed 42. Epoch-30 last.pt used for both comparisons.
| Model | Evaluation | N | Accuracy | Macro-F1 | Extra-class errors |
|---|---|---:|---:|---:|---:|
| A | common6 | 120 | 0.9333 | 0.9318 | 0 |
| B | common6 | 120 | 0.9250 | 0.9300 | 2 |
| B | full8 | 160 | 0.9375 | 0.9355 | 0 |

Macro-F1 common6 uses the same six GT classes for both models. B keeps all eight output scores; no removal/renormalization of halo/tidur predictions.
Technical flagged frames retained: 497/640. Semantic audit assessed=0; known ambiguous/temporal ambiguity=0; unknown=640. Unknown does not mean valid.
Actual SGD steps: A=690, B=900. More data also yields more updates at equal epochs.

## apa_kabar / halo
Technical visual audit: 160 contact-sheet thumbnails viewed (120 stratified random + 40 additional flagged). Unadjudicated concerns: 3 possible preparation/resting frames; 2 cross-class pose-overlap examples. These are technical observations, not expert semantic verification. No frame or label was replaced.
- A/apa_kabar: 20/20 correct; wrong classes={}.
- B/apa_kabar: 20/20 correct; wrong classes={}.
- B/halo: 19/20 correct; wrong classes={'apa_kabar': 1}.

## Interpretation and next step
B minus A common6: accuracy=-0.0083; macro-F1=-0.0017. One seed is descriptive, not proof of superiority.
Full-eight accuracy=0.9375, compared with a balanced top-1 chance reference of 0.125 (not a significance test).
Results provide exploratory evidence to continue investigating classification, especially with verified semantic audits, signer metadata and multiple seeds. They do not establish thesis-methodology replacement.
Detection was not trained on a matched release, so these numbers cannot justify a causal accuracy advantage over detection or automatically returning to it. Classification labels the whole image; detection additionally localizes hands. Choose based on the thesis task and a later matched comparison.
No temporal comparator was trained. Errors may reflect bad video-to-frame labels, technical selection, visual ambiguity, signer/background cues or limited training.
Clothing/room cues recur in the audited image groups. High video-label accuracy does not establish that the model learned linguistically valid gestures. This is closed-set classification; background/unknown-gesture rejection, real-time latency and signer-independent generalization were not tested.
Before final experiments: assess semantic errors and label validity, collect participant metadata, freeze configuration, use multiple seeds and only then run external Kaggle testing.

## Limitations
- Exploratory pilot, not final thesis performance; one seed and 30 epochs.
- Labels come from frozen video metadata, not verified per-frame semantics; dataset is NOT fully verified.
- Technical flags are retained; MediaPipe hand detection is not gesture classification.
- Signer/session independence is unverified where metadata is absent; no signer-independent claim.
- Static-frame/label errors, background and signer cues may explain performance; no causal temporal claim.
- Classification and detection were not compared on a matched trained benchmark; architecture is not automatically replaced.
