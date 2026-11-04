# 03 — Insight dari eksperimen

**exploratory/development**, internal validation. Ringkasan ini membaca laporan Tahap1b yang sudah selesai;
tidak menghitung ulang model, mengganti threshold, atau memulai eksperimen.

## FACT — hasil model

| Model | Scope | Macro-F1 mean ± SD |
|---|---|---|
| A | common6 | 0.9399 ± 0.0116 |
| B | common6 | 0.9312 ± 0.0055 |
| B | full8 | 0.9362 ± 0.0062 |

Mean paired delta B−A = **-0.008729**;
hierarchical bootstrap95% CI **[-0.030101; 0.007668]**.
Pilot belum dapat memutuskan NI; stop rule aktif. Semua5 seed masuk; seed42 replikasi.
Pooled McNemar hanya deskriptif karena video berulang lintas seed; tidak dipakai sebagai uji independen.

## FACT — kontrol diagnosis

| Model/scope | Hand-masked accuracy | Uniform-output chance |
|---|---|---|
| A/common6 | 0.9200 | 0.1667 |
| B/common6 | 0.9167 | 0.1250 |
| B/full8 | 0.8912 | 0.1250 |

| Backbone untuk probe6 | Macro-F1 mean ± SD |
|---|---|
| A | 0.9334 ± 0.0048 |
| B | 0.9259 ± 0.0092 |

## HYPOTHESIS

Cue luar ROI tangan, label video lemah, dan overfit adalah kemungkinan penjelasan; belum bukti sebab.
Masking dapat menyisakan wajah/tubuh atau tangan yang tidak tercakup landmark. Signer/sesi belum terverifikasi independen.
Train curve memakai actual weights; validation framework memakai EMA, sehingga gap perlu dibaca dengan caveat itu.

## PROPOSAL — gate berikutnya

Berhenti setelah Tahap1b. Delta/H2 tetap; tidak memilih seed/checkpoint dan tidak tuning Kaggle.
Tahap2 membutuhkan keputusan peneliti, addendum yang mengunci delta/H2/domain, integritas folds/grid/budget,
dan persetujuan fase terpisah. Reorganisasi folder bukan persetujuan Tahap2.

## Artefak analisis per seed

| Seed | Paired/CI/errors | Masking eval | Kurva |
|---|---|---|---|
| 42 | [CSV/MD](tahap1b/exploratory_development_stage1b_user_20261103_v1_paired_seed42) | [CSV/MD](tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed42) | [PNG](tahap1b/exploratory_development_stage1b_user_20261103_v1_curves_seed42/exploratory_learning_curves.png) |
| 43 | [CSV/MD](tahap1b/exploratory_development_stage1b_user_20261103_v1_paired_seed43) | [CSV/MD](tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed43) | [PNG](tahap1b/exploratory_development_stage1b_user_20261103_v1_curves_seed43/exploratory_learning_curves.png) |
| 44 | [CSV/MD](tahap1b/exploratory_development_stage1b_user_20261103_v1_paired_seed44) | [CSV/MD](tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed44) | [PNG](tahap1b/exploratory_development_stage1b_user_20261103_v1_curves_seed44/exploratory_learning_curves.png) |
| 45 | [CSV/MD](tahap1b/exploratory_development_stage1b_user_20261103_v1_paired_seed45) | [CSV/MD](tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed45) | [PNG](tahap1b/exploratory_development_stage1b_user_20261103_v1_curves_seed45/exploratory_learning_curves.png) |
| 46 | [CSV/MD](tahap1b/exploratory_development_stage1b_user_20261103_v1_paired_seed46) | [CSV/MD](tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed46) | [PNG](tahap1b/exploratory_development_stage1b_user_20261103_v1_curves_seed46/exploratory_learning_curves.png) |

- [Ringkasan hierarchical bootstrap](tahap1b/exploratory_development_stage1b_user_20261103_v1_exploratory_multiseed/exploratory_multiseed_summary.json).
- [Laporan lengkap](../01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_stage1b_report.md).
- [Power/presisi Tahap1a](../01_hasil/tahap1a/exploratory_development_stage1a_20261103/exploratory_power/exploratory_power_precision.md).
- [Arsip diagnosis Tahap1a](arsip_tahap1a/experiment_yolo26_classification_diag_v1): artefak historis, termasuk gate lama yang disupersede keputusan chat.
- [Arsip analisis notebook](arsip_notebook/).

Navigasi: [Hasil](../01_hasil/README.md) · [Eksperimen](../02_exp/README.md).
