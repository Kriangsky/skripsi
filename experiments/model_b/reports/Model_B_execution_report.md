# Model B execution report

Summary of actual executed notebook outputs and preserved artifacts. No notebook outputs or execution counts were synthesized.

## Execution and selection

Official run: `model_b_official_20261111T053828891203Z`. All 16 code cells executed, counts 1–16, no error outputs; three embedded figures. The Jupyter ipykernel used the repository virtual environment and NVIDIA RTX 4070 Laptop GPU. Training and prediction ran directly in notebook cells.

Eight seeded random trials completed, independently initialized from the official `yolo26n-cls.pt` asset. Pretrained SHA-256: `0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`. No Model A trained weights initialized Model B.

Selected trial 07: SGD, lr0=0.01, weight_decay=0.0001, batch=16; seed 42, image size 224, maximum 100 epochs, patience 20. Linear scheduler lrf=0.01, momentum=0.937, warmup_epochs=3, warmup_momentum=0.8, warmup_bias_lr=0.1, nbs=64; deterministic settings, workers=0, AMP disabled, dropout=0, no augmentation. Complete arguments appear in the executed notebook, trial manifest, args.yaml and actual_training_config.json.

Selected run completed 39 epochs. Its last tied best checkpoint is epoch 37; the stopper's last strict improvement was epoch 19. Framework fitness is eight-class macro-F1 rounded to five decimals. Checkpoint ties replace best.pt; early stopping resets only on strict improvements. Lowest trial ID breaks trial ties. All eight trials stopped early; none warm-started from another trial.

| Trial | Optimizer | LR | Weight decay | Batch | Epochs | Best epoch | Selection macro-F1 |
|---|---|---|---|---|---|---|---|
| 07 | SGD | 0.01 | 0.0001 | 16 | 39 | 37 | 0.99583 |
| 03 | SGD | 0.003 | 0.0001 | 16 | 46 | 26 | 0.99377 |
| 01 | AdamW | 0.001 | 0.0001 | 16 | 38 | 18 | 0.99375 |
| 08 | AdamW | 0.001 | 0.0005 | 16 | 35 | 29 | 0.99168 |
| 02 | SGD | 0.003 | 0.0005 | 32 | 51 | 37 | 0.99166 |
| 05 | SGD | 0.01 | 0.0005 | 32 | 40 | 40 | 0.98758 |
| 06 | AdamW | 0.0003 | 0.0005 | 32 | 33 | 13 | 0.98556 |
| 04 | SGD | 0.03 | 0.0001 | 16 | 48 | 48 | 0.98551 |

## Data and reused components

Verified 2,400 physical internal videos, 300/class: 1,920 train and 480 validation, 240/60 per class. All 1,800 shared samples retain exact Model A frame identities, hashes, labels and splits; 360 validation IDs pair with Model A. All source videos and selected frames were rehashed after training. Frozen split digest matched candidate configuration. No cross-split source-video or exact-frame byte/pixel leakage was detected.

Reused Model A's full-frame RGB letterbox transform, inference predictor, cached-frame ranking, canonical IDs, official pretrained asset, seed/hash helpers and frozen assignments. The new classes had 160 historical frame selections; those were retained. The other 440 used existing cached candidates ranked by Model A's method. No frames were extracted and no splits generated. Semantic labels remain unassessed; technical ranking is not semantic adjudication.

Model A audit: one documented baseline, 65 epochs, accuracy-selected checkpoint, internal-validation macro-F1 0.9944666391146623. No documented independent hyperparameter search. Its source notebook has no execution counts; existing completed evidence derived from external training artifacts. Model A evidence remained byte-identical.

Model A requires independent hyperparameter optimization before the final optimized Model A vs Model B comparison.

## Actual validation results

| Evaluation subset | Samples | Accuracy | Macro precision | Macro recall | Macro-F1 |
|---|---|---|---|---|---|
| All eight classes | 480 | 0.9958333333333333 | 0.9959016393442623 | 0.9958333333333333 | 0.9958330439613863 |
| Original six; eight-way predictions retained | 360 | 0.9944444444444445 | 0.9945355191256832 | 0.9944444444444445 | 0.9944440586151817 |

Two errors: bingung → apa_kabar; marah → kamu. Original-to-new intrusions: 0/360. New-to-original intrusions: 0/120. Full per-class tables, probabilities and confusion matrix are in the notebook and exports. Selection fitness and reloaded checkpoint metrics are recorded separately.

Actual search exposure: 330 total epochs, 633,600 training image presentations and 10,460 optimizer steps. Model A baseline exposure was 93,600 presentations. Per-trial API durations and comparable CSV training times are retained in manifests and `model_a_model_b_budget.csv/json`. Model A's actual optimizer-step count was not recorded and is left missing.

## Files and verification

Created only Model B artifacts: primary notebook; model_b_support.py; build_notebook.py; execute_notebook.py; test_model_b.py; verify_experiment.py; README.md; config/search_protocol.json; immutable selected-frame manifest; reports; separate smoke and official outputs.

Saved all eight trial best/last checkpoints, selected best/last copies, CSV/JSON trial histories, sample predictions with named eight-class probabilities, per-class metrics, confusion matrix, curves, configurations, source/dataset/split hashes, environment and checkpoint hashes. A complete executed notebook copy resides in the official run's `reproducibility/executed_notebook.ipynb`.

Four unit tests passed. `acceptance_verification.json` and `final_data_integrity.json` report PASS. All trial histories were finite; actual optimizer, learning rate, weight decay, batch, seed and epoch/patience arguments matched planned values. Model A files retained their hashes. Historical archive and raw data were not modified. Git changes are confined to new Model B files.

One preliminary smoke stopped on a Windows notebook-save error; its artifacts remain with failure status. Atomic replacement/retry fixed saving; a separate complete smoke passed all 16 cells. Restricted-sandbox kernel startup required local-socket permission. No remaining blocker.

## Inspect or repeat

Open `experiments/model_b/Model_B_YOLO26_8Class_Training.ipynb` in Jupyter or VS Code to inspect persisted results without rerunning.

To repeat: select the repository `.venv-bisindo/Scripts/python.exe`, restart kernel, Run All, then Save with outputs. This starts a new eight-trial experiment. Alternatively, from repository root:

```powershell
.venv-bisindo/Scripts/python.exe experiments/model_b/execute_notebook.py --mode official
```

Do not rerun the notebook authoring utility over the executed record. Existing official run copies remain preserved.

## Limits

Internal validation was reused for optimization and evaluation; these are not independent test results. No signer-independence claim. Model A has a smaller documented tuning budget and needs separate optimization. Vocabulary, sample count and hyperparameters vary together; performance differences cannot be attributed solely to vocabulary expansion. Technical frame scores do not establish semantic label validity. Model C, Kaggle evaluation and Model A retraining were not performed.
