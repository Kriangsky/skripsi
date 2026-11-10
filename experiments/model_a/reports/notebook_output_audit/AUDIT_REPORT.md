# Model A notebook-output audit

Problem: opening the clean source notebook shows no saved results.
Cause: source has31 total cells, including16 Markdown and15 code cells; all source code counts are null and outputs empty. The executed evidence was saved separately. It has15 executed code cells (counts1–15),33 output blocks and no errors. All original source cell IDs/types/text match. Training ran externally, not in that notebook replay.

Fix: created `experiments/model_a/Model_A_YOLO26_6Class_Completed_Evidence.ipynb`. Part A explicitly labels static displays reconstructed from existing artifacts. Part B preserves all31 saved completed-results cells unchanged, including counts, timestamps and outputs. Original source and executed notebooks preserved. No kernel, training, frame generation, split edits, hyperparameter changes or metric recalculation.

Verified: source notebook SHA matches the official run snapshot; best.pt SHA matches the run manifest; completed65 epochs, selected checkpoint46, first strict maximum45; embedded metrics/run-manifest outputs exactly match saved JSON; embedded confusion PNG matches saved PNG byte-for-byte. History, per-class CSV, metric CSV and360 prediction records are present. All protected original files remain byte-identical. All restored image attachments use the saved image bytes. Notebook schema, save/reopen cell equality and static HTML conversion passed.

Missing: no required log/history/metric/confusion/checkpoint/prediction artifact is missing. Original in-notebook training execution counts/IOPub streams and kernel state cannot be recovered: the65-epoch fit ran outside Jupyter. Existing external log text is available and labelled as an artifact display, never an original notebook stream.

Evidence limitation: newly generated SHA receipts establish current preservation, not original creation times. Validation prediction files did not have separate contemporaneous per-file hashes; existing run metadata, checkpoint hashes, the saved replay displays and prior verification records provide the available linkage. No metric was recomputed during this audit.

Open the revised notebook to view outputs without running cells, or open `Model_A_completed_evidence.html` in this audit directory. `AUDIT_MANIFEST.json` records every input digest, cell inventory, preserved-cell offset and presentation source.
