# 06 — Notebook

- [eksperimen](eksperimen/): notebook Detection, Classification, Stage0/1a dan notebook Tahap1b historis.
- [hasil](hasil/): notebook hasil Tahap1b, termasuk versi dengan output.
- [persiapan](persiapan/): notebook rename historis.

Untuk eksekusi notebook beku gunakan `05_kode/run.ps1 -Notebook`: Jupyter dibuka pada namespace
runtime lama sehingga `Path.cwd()`, source dan manifest tetap sesuai. Jangan menjalankan notebook beku
dari cwd folder ini; asumsi cwd-nya belum diubah demi menjaga hash/output lama.
Notebook hasil dengan output dapat dibaca langsung. Training memerlukan persetujuan fase/flag yang berlaku;
notebook Tahap1b asli menyimpan error historis, sementara eksekusi sukses memakai runner proses terpisah.
[Hasil terbaru](../01_hasil/README.md).
