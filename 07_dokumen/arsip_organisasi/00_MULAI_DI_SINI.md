# Navigasi proyek BISINDO

Organisasi folder: 4 November 2026. Hasil penelitian tetap **exploratory/development**.

| Section | Isi | Mulai dari |
|---|---|---|
| [01_hasil](01_hasil/README.md) | Laporan akhir, tabel hasil, verifikasi dan notebook hasil | [Laporan Tahap1b](01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_stage1b_report.md) |
| [02_exp](02_exp/README.md) | Run A/B, linear probes, checkpoints, cache eksperimen, log dan kegagalan | [Daftar eksperimen](02_exp/README.md) |
| [03_insight_exp](03_insight_exp/README.md) | Paired analysis, bootstrap, masking, kurva, power dan interpretasi | [Ringkasan insight](03_insight_exp/README.md) |

Kode `.py`, notebook sumber, protokol/hash dan dataset `train/` / `test/` tetap di root karena dipakai path/hash terkunci.
Folder output lama di root sekarang junction kompatibilitas tersembunyi, bukan salinan data. Path checkpoint lama tetap valid.
Jangan hapus junction tersebut selama artefak lama masih dipakai. Tampilkan hidden files bila perlu melihatnya.

Pemetaan dan bukti pemeriksaan: [organisasi_20261104](02_exp/organisasi_20261104/).
[Rollback](02_exp/organisasi_20261104/rollback.ps1) mengembalikan folder fisik ke nama/path semula tanpa mengubah isinya.
Tahap2 belum dijalankan; reorganisasi ini tidak menjalankan training.
