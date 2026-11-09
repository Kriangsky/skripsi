# Brief konsultasi pembimbing — 9 November 2026

**Status: hasil DEVELOPMENT tersedia; eksperimen final dan evaluasi eksternal BLOCKED.**

**Judul lama:** *Effect of Training Set Expansion on YOLOv26 Performance for Real-Time BISINDO Phrase Recognition*.

**Usulan judul:** *Impact of Vocabulary Expansion on Frame-Based BISINDO Recognition: A Controlled YOLO26 Study*.

## Arah dan tujuan

Satu frame/video dengan satu label gambar lebih sesuai dengan Classification. Detection memerlukan endpoint lokalisasi; pengenalan temporal memerlukan urutan. Belum ada pembuktian Classification lebih unggul daripada Detection, dan perubahan metode belum mendapat bukti persetujuan akademik.

RQ: Apakah ekspansi vocabulary enam menjadi delapan kelas mengubah performa frame-level pada enam kelas awal, dan kelas mana yang paling terdampak? A dilatih independen pada saya, kamu, senang, bingung, marah, apa_kabar; B menambahkan halo dan tidur. Efek merupakan paket vocabulary **dan data/steps/head**, bukan efek kausal jumlah kelas saja.

## Data dan hasil yang sudah tersedia

| Bukti | Temuan | Batas |
| --- | --- | --- |
| TASK1 audit.json | 2.400 internal, 400 eksternal; train/val 1.920/480; 300/400 eksternal common6/full8 | Receipt audit dibaca; 45.250 checks tidak diulang |
| Human pilot, internal train | 160 frame, dua ekspor reviewer: 129/160 (80,625%) dan 138/160 (86,25%); kappa 0,82516 | Bukan adjudikasi linguistik bersertifikat |
| apa_kabar manusia | 9/20 dan 14/20 benar (45%/70%) | Sampel berbeda dari validation model |
| Stage1b, seluruh lima seed | A common6 macro-F1 0,93990 ± 0,01158; B 0,93117 ± 0,00548 | SD antarseed; 120 video yang sama |
| Paired Δ B−A | −0,0087293; CI95% [−0,0302228; +0,0076121]; SD Δ 0,0158984 | Development, bukan konfirmatori/eksternal |
| B full8 | Mean accuracy 0,9375; macro-F1 0,93619 | 160 video internal |
| Interferensi | Recall saya −3pp, marah −3pp, senang −1pp; tiga kelas lain 0 | Tidak menetapkan kelas paling terdampak secara inferensial |
| Masking tangan | Accuracy A common6 0,9200; B common6 0,91667 | Wajah/tubuh dapat tersisa; bukan bukti background-only |

Sumber ringkas: `VERIFIED_EXPERIMENT_RESULTS.md` §2–9 dan `EVIDENCE_MANIFEST.json`. Akurasi/F1 Stage1b dicek ulang secara aritmetika dari prediksi tersimpan; CI hanya dicocokkan dengan output statistik yang tersimpan. Tidak ada scoring baru. CI historis lain [−0,0301015; +0,0076679] juga dipertahankan dalam laporan.

## Status dan keputusan pembimbing

TASK0–2 menyelesaikan infrastruktur/audit/usulan, **bukan skripsi final**. Controller notebook tersedia. Final frame manifests masih null; semantic review, signer/session, lisensi dan versi/arsip Kaggle serta exposure attestation belum lengkap. Dataset eksternal pernah dipreprocess. NI margin belum memiliki justifikasi; H1 estimasi masih usulan. Full historical regression pernah 91/92; bukan seluruh tes hijau.

Mohon lima keputusan: (1) Classification frame-based menggantikan Detection; (2) judul/RQ dan batas klaim; (3) estimasi Δ+CI atau NI dengan margin substantif, mempertahankan H2 asli; (4) final A/B lima seed pada full locked data, equal epochs dan nasib CV/grid/faktorial lama; (5) persyaratan semantic review, provenance/exposure dan kelayakan klaim eksternal. Semua perlu catatan persetujuan nyata sebelum pelaksanaan.

## Narasi siap ucap ±90 detik

Penelitian awal saya membahas perluasan training set untuk pengenalan frasa BISINDO real-time. Namun eksperimen yang tersedia menggunakan satu frame per video dan menghasilkan satu label gambar. Karena itu saya mengusulkan fokus frame-based Classification, bukan klaim pemahaman gerakan temporal atau real-time. Pertanyaan utamanya: apakah perluasan dari enam menjadi delapan kelas mengubah kemampuan mengenali enam kelas awal, dan kelas mana yang terdampak? Dataset lokal mencakup 2.400 video internal dan 400 eksternal; audit terdahulu mencatat kecocokan file dan hash, tetapi validitas semantik serta identitas signer belum lengkap. Pilot dua reviewer pada 160 frame memperoleh 80,625 dan 86,25 persen; apa_kabar paling sulit, hanya 45 dan 70 persen. Sepuluh run development YOLO26n Classification telah selesai pada lima pasangan seed. Mean macro-F1 common6 adalah 0,93990 untuk A dan 0,93117 untuk B; selisih minus 0,00873 dengan interval 95 persen minus 0,03022 sampai plus 0,00761. Arah penurunan belum pasti dan ini bukan bukti kesetaraan. Masking tangan masih menyisakan akurasi tinggi, sehingga shortcut perlu menjadi keterbatasan. Efek yang diukur mencakup tambahan vocabulary, data, dan langkah optimisasi sekaligus. Training final dan scoring eksternal belum dilakukan. Saya meminta persetujuan perubahan metode, judul dan pertanyaan, analisis estimasi versus non-inferiority, serta cakupan final dan syarat provenance sebelum melanjutkan.
