# Model B — independent eight-class YOLO26

Primary artifact: `Model_B_YOLO26_8Class_Training.ipynb`. All training trials, model initialization, predictions, metrics, selection and figures execute directly in its cells. `model_b_support.py` supplies data guards and the custom trainer/validator only. `execute_notebook.py` runs Jupyter cells and saves outputs after each cell and on errors; it is not a separate training implementation.

From the repository root, execute:

```powershell
.venv-bisindo/Scripts/python.exe experiments/model_b/test_model_b.py
.venv-bisindo/Scripts/python.exe experiments/model_b/execute_notebook.py --mode smoke
.venv-bisindo/Scripts/python.exe experiments/model_b/execute_notebook.py --mode official
```

Smoke persists `reports/Model_B_smoke_executed.ipynb`. Official execution persists the primary notebook. Both copy the executed notebook into their unique run's `reproducibility/` directory on success. Failures retain genuine error outputs and partial execution counts. Never clear outputs.

Interactive: open the primary notebook in Jupyter/VS Code, choose `.venv-bisindo/Scripts/python.exe`, restart kernel, Run All, then Save. Default is eight official trials. Opening alone launches nothing. A second execution of the training cell in the same kernel raises. Restarting and running all creates another unique experiment; it never resumes stale weights. Smoke mode is set by the execution driver.

Search: seed 42, eight random configurations without replacement; SGD LR {0.003, 0.01, 0.03}, AdamW LR {0.0001, 0.0003, 0.001}, weight decay {0.0001, 0.0005}, batch {16, 32}. Each trial starts from the exact pinned official pretrained asset, never Model A. Max 100 epochs, patience 20, image size 224, deterministic settings, no augmentation, Model A's RGB letterbox transform and prediction preprocessing. Linear scheduler lrf=0.01; all remaining base arguments come from audited Model A configuration and print in section 03.

Selection: all-eight-class internal-validation macro-F1 controls checkpoint fitness and early stopping. Installed Ultralytics rounds training validation metrics to five decimals. Last tied epoch wins; lowest trial ID breaks trial ties. Saved FP16 checkpoint predictions may differ slightly from FP32 training EMA; notebook preserves both values. Failed trials remain on disk and stop execution for diagnosis. No extra final retraining.

Early stopping resets patience only on a strict fitness improvement. Tied epochs can replace `best.pt` without resetting patience. The selected run therefore stopped at epoch 39, with its last tied best checkpoint from epoch 37 and the stopper's last strict improvement at epoch 19.

Completed official execution: eight trials, 330 total epochs, 10,460 actual optimizer steps, 633,600 training image presentations. Selected trial 07 uses SGD, lr0=0.01, weight_decay=0.0001, batch=16. Eight-class validation: accuracy 0.9958333333, macro-F1 0.99583304396; common-six validation: accuracy 0.9944444444, macro-F1 0.99444405862. Original-to-new and new-to-original intrusion counts are zero. Full provenance and limits are in `reports/Model_B_execution_report.md` and the executed notebook.

Inspect the saved notebook without running it. `build_notebook.py` is the original authoring utility; do not rerun it over the executed notebook. An archived executed copy also resides in the official run's `reproducibility/` directory.

Data: 2,400 locked source videos, 300/class, 240 train + 60 validation/class. All 1,800 shared samples retain Model A's actual official-run selected frame, class and split. The new classes have 160 historical frame selections; retain those and choose the remaining 440 from existing cached candidates using Model A's technical ranking. No frame extraction, raw edits or split generation. Hashes verify videos, selected frames, candidate provenance and cross-split pixel identity. Data counts are assertions until the dataset-audit cell executes successfully.

Model A audit: one documented baseline official run, 65 completed epochs, accuracy-selected best epoch 46, validation macro-F1 0.9944666391146623. Source training notebook has zero execution counts; existing completed evidence was reconstructed/replayed from external training artifacts. Model B does not repeat that orchestration. Model A requires independent hyperparameter optimization before the final optimized Model A vs Model B comparison.

Limitations: validation reused for tuning; no independent generalization estimate. No signer-independence claim. Technical frame scores do not establish semantic label validity. Vocabulary size, sample count and optimized settings vary together. No Model C or Kaggle evaluation. Model A evidence hashes must remain unchanged.
