# BISINDO — workspace

Seluruh file sudah disegmentasikan. Hasil penelitian tetap **exploratory/development**.

**[Tracking proyek: dibuat, dicoba, berhasil, gagal, tertunda](07_dokumen/TRACKING_BISINDO.md)**.

| Section | Isi |
|---|---|
| [01_hasil](01_hasil/README.md) | Laporan, metrik, notebook hasil dan verifikasi |
| [02_exp](02_exp/README.md) | Full runs, linear probes, checkpoint, cache dan log |
| [03_insight_exp](03_insight_exp/README.md) | Interpretasi hasil, paired analysis, masking, kurva dan power |
| [04_dataset](04_dataset/README.md) | Video internal train dan external test; tidak diubah |
| [05_kode](05_kode/README.md) | Modul Python, tests, runner dan launcher |
| [06_notebook](06_notebook/README.md) | Notebook eksperimen, hasil, dan persiapan |
| [07_dokumen](07_dokumen/README.md) | Protokol/hash, draft paper historis, handoff dan panduan |
| [08_review](08_review/README.md) | Template review, ground truth dan respons reviewer |
| [09_environment](09_environment/README.md) | Akses environment Python existing |

Root hanya memuat README dan konfigurasi Git standar. Alias folder lama serta environment disembunyikan.
`.compat_runtime/` mempertahankan namespace runtime lama memakai hard links, bukan salinan video/weights.
Jangan menghapus alias atau runtime itu: snapshot source, notebook dan checkpoint historis masih memakainya.
Kode/data di section dan runtime menunjuk file yang sama; edit pada salah satunya mengubah file yang sama.

Untuk menjalankan kode/tests/notebook, gunakan [launcher](05_kode/run.ps1) agar cwd/import tetap kompatibel.
Reorganisasi tidak menjalankan training, tidak mengubah kandidat protokol, dan tidak memulai Tahap2.
[Catatan organisasi dan rollback](07_dokumen/organisasi_20261104_lengkap/README.md).
