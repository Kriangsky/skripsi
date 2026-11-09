# TASK 2 ONLY — audit dan kandidat, 2026-11-09

Otorisasi: lampiran pengguna Task 2. Tidak mengesahkan keputusan ilmiah atau run.
Ponytail: reuse Settings, CONFIG, letterbox, SHA, modul statistik dan reader existing.

1. Snapshot source/protokol/metadata/run/checkpoint/notebook sebelum perubahan. Jangan mengulang ledger 45.250 file.
2. Baca sepuluh execution records, args.yaml, environment, results, optimizer counters, log dan metadata checkpoint.
   Uji transform tersimpan pada citra sintetik; bedakan bukti historis dari rekonstruksi source framework lokal.
3. Audit seed melalui handoff, protokol, approval, tabel fit dan failures. Laporkan keterbatasan urutan historis.
4. Buat satu JSON CANDIDATE PROPOSED_NOT_APPROVED: baseline hyperparameters, full frozen train/val,
   tanpa augmentasi/CV/search. Memo membandingkan NI vs estimasi, H2 verbatim dan dependency akademik.
5. Tambah sel read-only TASK 2 pada notebook primer; source/output/execution_count sel kode lama identik.
   Append tracker existing. Tidak mengubah analysis_plan atau protokol.
6. Tes struktur kandidat, transform, guards notebook, statistik/interference; jalankan seluruh safe notebook
   di salinan memory lalu simpan hanya output sel TASK 2. Verifikasi hash dan changed-file inventory.

Scope perubahan: folder bukti ini, satu test baru, notebook primer, tracker existing. Callers/config/exports
historis tidak diubah. Risiko: kandidat belum bisa dijalankan API Stage1b (TARGETS 60/20 dan gate stage1b);
implementasi final dan persetujuan hanya pekerjaan downstream. STOP sebelum TASK 3.
