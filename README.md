# BISINDO — workspace

**Hipotesis lama tidak dipakai lagi. Pengembangan dan eksperimen aktif sekarang berada di folder [`experiments/`](experiments/), dimulai dari Model A.**
Folder [`old hypothesa/`](old%20hypothesa/ARCHIVE_MANIFEST.md) menyimpan eksperimen hipotesis lama sebagai arsip historis, bukan acuan eksperimen aktif.

Eksperimen aktif mencakup **Model A (6 kelas)** dan **Model B (8 kelas)** berbasis YOLO26 classification. Semua hasil masih **exploratory/development**, bukan estimasi generalisasi independen.

| Model | Notebook dan panduan | Data train / validasi | Hasil validasi internal |
|---|---|---|---|
| A — baseline 6 kelas | [Notebook](experiments/model_a/Model_A_YOLO26_6Class_Training.ipynb) · [Audit dan penggunaan](experiments/model_a/reports/AUDIT_AND_USAGE.md) | 1.440 / 360 frame | Accuracy 99,44%; macro-F1 99,45% |
| B — 8 kelas | [Notebook dengan hasil eksekusi](experiments/model_b/Model_B_YOLO26_8Class_Training.ipynb) · [Panduan](experiments/model_b/README.md) · [Laporan eksekusi](experiments/model_b/reports/Model_B_execution_report.md) | 1.920 / 480 frame | Accuracy 99,58%; macro-F1 99,58% |

Model A baseline selesai 65 epoch. Model B menyelesaikan 8 trial dengan inisialisasi independen dari pretrained resmi; trial 07 terpilih berdasarkan macro-F1 validasi 8 kelas. Pada 360 sampel validasi enam kelas awal, Model B mencapai accuracy 99,44% dan macro-F1 99,44% dengan prediksi delapan kelas tetap dipertahankan.

**Batas interpretasi:** validasi Model B dipakai untuk tuning dan pemilihan checkpoint. Model A masih memerlukan optimasi hyperparameter independen sebelum perbandingan akhir kedua model yang dioptimasi. Jumlah kelas, jumlah data, dan konfigurasi berubah bersama; hasil ini belum membuktikan pengaruh penambahan kelas secara terisolasi. Belum ada evaluasi Kaggle atau klaim signer-independent.

**[Tracking proyek: dibuat, dicoba, berhasil, gagal, tertunda](07_dokumen/TRACKING_BISINDO.md)**.

| Section | Isi |
|---|---|
| [experiments](experiments/) | Eksperimen aktif Model A dan Model B |
| [old hypothesa](old%20hypothesa/ARCHIVE_MANIFEST.md) | Arsip hipotesis lama yang tidak lagi dipakai |
| [01_hasil](01_hasil/README.md) | Laporan, metrik, notebook hasil dan verifikasi |
| [02_exp](02_exp/README.md) | Full runs, linear probes, checkpoint, cache dan log |
| [03_insight_exp](03_insight_exp/README.md) | Interpretasi hasil, paired analysis, masking, kurva dan power |
| [04_dataset](04_dataset/README.md) | Video internal train dan external test; tidak diubah |
| [05_kode](05_kode/README.md) | Modul Python, tests, runner dan launcher |
| [06_notebook](06_notebook/README.md) | Notebook eksperimen, hasil, dan persiapan |
| [07_dokumen](07_dokumen/README.md) | Protokol/hash, draft paper historis, handoff dan panduan |
| [08_review](08_review/README.md) | Template review, ground truth dan respons reviewer |
| [09_environment](09_environment/README.md) | Akses environment Python existing |

Mulai dari notebook dan panduan model di tabel atas. Untuk memeriksa Model B tanpa training ulang:

```powershell
.venv-bisindo/Scripts/python.exe experiments/model_b/test_model_b.py
```

Dataset mentah, frame peserta, environment, dan checkpoint model tetap lokal sesuai `.gitignore`; clone GitHub saja tidak cukup untuk mengulang training. Notebook Model B menyimpan hasil eksekusi; membuka notebook tidak memulai training. Menjalankan ulang seluruh sel memulai eksperimen baru.

Folder bernomor mempertahankan kode, data, dokumen, dan bukti historis. Pada workspace lokal, path eksperimen lama tersedia melalui compatibility junctions; `.compat_runtime/` memakai hard links. Jangan menghapus atau mengedit alias tanpa memeriksa dependensinya. Junctions dan runtime lokal tidak otomatis dibuat oleh Git clone.

Untuk workflow historis, gunakan [launcher](05_kode/run.ps1) agar cwd/import tetap kompatibel. Lihat [catatan organisasi dan rollback](07_dokumen/organisasi_20261104_lengkap/README.md).
