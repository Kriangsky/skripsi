# BISINDO — deviation dan perubahan metodologi

Tanggal pencatatan: **2026-11-03, Asia/Bangkok**. Log append-only untuk perubahan berikutnya.

| ID | Tanggal | Perubahan | Alasan dan bukti | Dampak |
|---|---|---|---|---|
| D001 | 2026-11-03 | Peneliti menetapkan Classification sebagai jalur utama; Detection future work | `experiment_yolo26_training_pilot/reports/readiness.json`: ready=false, 0 verified positives pada seluruh target, metadata signer/session missing. v3 belum mempunyai positive semantic approval. Tidak berarti auto-box proposal tidak ada. | Metode Classification/ImageNet, top-1/macro-F1; Detection dan semua artefaknya tetap. Tidak menyatakan Classification lebih unggul. |
| D002 | 2026-11-03 | Pertanyaan baru: biaya vocabulary expansion dan NI, bukan target agar B menang | Keputusan peneliti, setelah pilot seed42 diketahui | Pilot lama eksploratif, tidak diperlakukan sebagai pengujian prospektif H1. Delta default .02; H2 .80 masing-masing kelas telah disetujui peneliti. Protokol keseluruhan masih menunggu persetujuan. |
| D003 | 2026-11-03 | Perjelas exposure Kaggle | Handoff dan artefak v2/v3 menunjukkan preprocessing external sudah terjadi; bukan model-selection score Classification | Peneliti menyetujui satu batch evaluasi model final setelah freeze. Jangan mengklaim Kaggle sama sekali belum dibuka. |
| D004 | 2026-11-03 | Rencana fold baru atas seluruh internal untuk development | Permintaan peneliti; split_lock lama tetap byte-identical | CV tidak menjadi independent confirmation karena internal val telah dipakai diagnosis. External sekali setelah freeze diusulkan sebagai domain konfirmatori. |

**Belum dilakukan pada revisi ini:** multi-seed, CV, training baru, masking, linear probes, McNemar/bootstrap baru,
evaluasi external, latency, atau perbaikan label. Tidak ada perubahan sumber video/split/run selesai.

Konflik snapshot lama: handoff 2026-11-02 masih menyebut notebook Classification belum dibuat dan metodologi belum berubah.
Artefak `BISINDO_YOLO26_Classification_End_to_End.ipynb` dan run notebook selesai membuktikan notebook tersebut sekarang ada;
keputusan peneliti 2026-11-03 menetapkan metodologi baru secara prospektif. Handoff lama tidak ditulis ulang.

Template entri berikut: ID, tanggal, protocol hash sebelumnya, perubahan, sebelum/sesudah hasil diketahui,
alasan, persetujuan peneliti, run terdampak, status exploratory/confirmatory, lokasi addendum/hash baru.

## D005 — 2026-11-03, paper-only scope and Stage 0 approval

Researcher approved Stage 0 conditional on Addendum 01; authorized A–D implementation/analysis only, zero training.
Parent protocol SHA256: `24e21743168f3457933f0f17db76e52d420b5f005da774a7f9230c71f1ac7a70`.
New output is a full conference paper, with no demo/webcam/application integration. Prospective timing is inference cost.
Historical manuscript, candidate protocol, Stage 0 notebook and completed results remain unchanged.
See `PREREGISTRATION_ADDENDUM_01.md` and its SHA256. This supersedes the earlier pending-approval status prospectively.

## D006 — 2026-11-03, Stage 1a implementation and precision analysis

Existing-pilot paired/bootstrap results and conditional power simulations are written under
`experiment_yolo26_classification_diag_v1/reports/`. They are exploratory analysis/planning, not new external scores.
Only seed42 is observed; between-seed variance is unknown. Simulated seed SD is explicitly hypothetical.
`classification_research.py` supplies parameterization in a private legacy-module namespace to preserve the source
hashes locked by Stage 0 tests. Original CLI/tests and completed experiment bytes remain untouched.
No prospective model training was started. Delta .02 and H2 .80 remain unchanged. Domain/threshold decisions
must be recorded in Addendum 02 before any prospective training. The analysis does not choose them for the researcher.
