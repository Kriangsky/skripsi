# 02 — Eksperimen

Artefak eksperimen dipindahkan utuh. Checkpoint/data/log tetap dalam paket aslinya.
Path root lama tetap berlaku melalui junction tersembunyi; tidak ada retraining/resume.

## Tahap1b — full A/B runs

| Seed | Folder run | Status |
|---|---|---|
| 42 (replikasi) | [A/B +release/config](tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42) | completed30epoch,224px,last.pt |
| 43 | [A/B +release/config](tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43) | completed30epoch,224px,last.pt |
| 44 | [A/B +release/config](tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44) | completed30epoch,224px,last.pt |
| 45 | [A/B +release/config](tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45) | completed30epoch,224px,last.pt |
| 46 | [A/B +release/config](tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46) | completed30epoch,224px,last.pt |

## Tahap1b — linear probes

Backbone frozen eval; head6 baru; train lama360,30epoch,SGD .01/.937/.0005,batch16; last head.

| Seed | A backbone | B backbone |
|---|---|---|
| 42 | [last head +log](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed42_attempt2) | [last head +log](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed42) |
| 43 | [last head +log](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed43) | [last head +log](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed43) |
| 44 | [last head +log](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed44) | [last head +log](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed44) |
| 45 | [last head +log](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed45) | [last head +log](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed45) |
| 46 | [last head +log](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed46) | [last head +log](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed46) |

Probe A seed42 attempt1 gagal sebelum model loading/optimization; [folder asli tetap disimpan](tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed42).
[Mask inputs dan coverage](tahap1b/mask_inputs/exploratory_development_stage1b_user_20261103_v1_exploratory_masks) berasal dari cache, bukan landmark detection baru.

## Arsip pipeline

| Paket | Keterangan |
|---|---|
| [experiment_yolo26](arsip_pipeline/experiment_yolo26) | Audit/preparation awal, legacy |
| [experiment_yolo26_v2](arsip_pipeline/experiment_yolo26_v2) | Detection preparation v2 |
| [experiment_yolo26_v3](arsip_pipeline/experiment_yolo26_v3) | Frozen split, candidate frames dan landmark cache |
| [experiment_yolo26_pilot](arsip_pipeline/experiment_yolo26_pilot) | Human feasibility pilot |
| [experiment_yolo26_training_pilot](arsip_pipeline/experiment_yolo26_training_pilot) | Detection pilot gates; bukan hasil Detection terlatih |
| [experiment_yolo26_classification_pilot](arsip_pipeline/experiment_yolo26_classification_pilot) | Classification pilot awal A/B seed42 |
| [experiment_yolo26_classification_notebook_20261102](arsip_pipeline/experiment_yolo26_classification_notebook_20261102) | Reproduksi Classification notebook |

## Log, kegagalan, dan organisasi

- [Sesi notebook peneliti](log_sesi/exploratory_development_stage1b_user_20261103_v1_session): event/timing/approval historis.
- [Preflight agen v1](kegagalan/exploratory_development_stage1b_agent_execution_20261103_v1): gagal sebelum fit; dipertahankan.
- [Riwayat error lengkap](../01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_failures.json).
- [Pemetaan folder dan verifikasi reorganisasi](organisasi_20261104/organization_manifest.json).
- [Rollback reorganisasi](organisasi_20261104/rollback.ps1).

## Kode dan notebook sumber

Sumber berada di05_kode; launcher memakai namespace runtime lama agar source locks tetap valid:
[Classification API](../05_kode/src/classification_research.py), [diagnostics](../05_kode/src/classification_diagnostics.py),
[workflow](../05_kode/src/classification_workflow.py), [runner proses terpisah](../05_kode/src/exploratory_stage1b_runner.py).
[Notebook Tahap1b asli](../06_notebook/eksperimen/BISINDO_Classification_Stage1b_Exploratory.ipynb) menyimpan error historis; eksekusi berhasil memakai runner proses terpisah.
[Protokol kandidat](../07_dokumen/protokol/PREREGISTRATION.md), [hash](../07_dokumen/protokol/PREREGISTRATION.sha256), dan [deviation log](../07_dokumen/protokol/DEVIATION_LOG.md) tidak diedit.

Navigasi: [Hasil](../01_hasil/README.md) · [Insight](../03_insight_exp/README.md).
