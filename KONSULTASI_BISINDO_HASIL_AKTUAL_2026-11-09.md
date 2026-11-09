# Brief konsultasi skripsi BISINDO — hasil aktual per 9 November 2026

**Status penelitian:** pilot dan eksperimen *development* telah berjalan. Konfigurasi final masih kandidat, final full-data A/B dan *external evaluation* **belum dijalankan**. Angka dalam dokumen ini bukan hasil konfirmatori.

## 1. Usulan arah penelitian

**Judul awal:** *Effect of Training Set Expansion on YOLOv26 Performance for Real-Time BISINDO Phrase Recognition*.

**Judul yang diusulkan:** *Impact of Vocabulary Expansion on Frame-Based BISINDO Recognition: A Controlled YOLO26 Study*.

**Research question:** *Does expanding the training vocabulary from six to eight BISINDO classes change frame-level recognition performance on the original six classes, and which classes are most affected?*

**Alasan perubahan:** Data diwakili oleh **satu frame per video**, dengan satu label gestur per frame. Karena itu tugas YOLO26 **Classification** lebih langsung selaras dengan tujuan mengukur prediksi kelas daripada Detection berbasis *bounding box*. Penelitian tidak mengukur pemahaman urutan gerakan ataupun demonstrasi *real-time*. Penggantian task dan judul masih memerlukan persetujuan dosen.

**Kontribusi yang realistis:** mengukur efek gabungan penambahan dua kelas dan data training terhadap pengenalan enam kelas lama, memerinci perpindahan kesalahan, dan memeriksa generalisasi pada domain eksternal jika layak dievaluasi. Eksperimen ini **tidak mengisolasi efek kausal murni jumlah kelas**.

## 2. Data dan desain

- Delapan kelas (urutan tetap): `saya, kamu, senang, bingung, marah, apa_kabar, halo, tidur`.
- Dataset internal: **2.400 video**, train **1.920**, validation **480** (240/60 per kelas).
- Dataset eksternal Kaggle: **400 video**, 50 per kelas; common6 **300 video**.
- Satu frame per video; visual dapat ambigu secara semantik walaupun tajam secara teknis.
- Model A: YOLO26n Classification dari pretrained awal → enam kelas. Model B: pretrained awal yang sama → delapan kelas, **fine-tuning independen**, bukan melanjutkan A.
- Evaluasi utama yang diusulkan: selisih *paired macro-F1* B minus A pada video enam kelas yang sama; Model B selalu menggunakan prediksi dari seluruh delapan output.

Audit TASK 1 melaporkan 2.800 video fisik terverifikasi, split/ID/GT konsisten, dan tidak menemukan duplikasi identik lintas split pada cakupan yang diperiksa. Audit asli tidak dijalankan ulang dalam paket konsultasi ini. Ada tiga pasangan kandidat frame identik pada `senang_223`/`senang_224`, keduanya dalam training, bukan frame development terpilih. Kedekatan semantik/identitas signer belum dijamin.

## 3. Eksperimen development yang sudah selesai

Stage1b: YOLO26n-cls, pretrained ImageNet; 224×224 letterbox, 30 epoch, batch 16, SGD lr0,01, tanpa augmentasi, checkpoint `last.pt` (EMA). Lima seed training **42–46**. **Model A train/val 360/120** (60/20 per kelas; 690 optimizer steps); **Model B train/val 480/160** (60/20 per kelas; 900 optimizer steps). Dataset development dipasangkan dan *seed* training tidak merupakan lima dataset independen.

| Seed | Macro-F1 A, common6 | Macro-F1 B, common6 | B − A |
|---|---:|---:|---:|
| 42 | 0,93178 | 0,93004 | −0,00173 |
| 43 | 0,93262 | 0,93089 | −0,00173 |
| 44 | 0,94750 | 0,93195 | −0,01555 |
| 45 | 0,95663 | 0,92381 | −0,03282 |
| 46 | 0,93097 | 0,93916 | +0,00819 |
| **Rata-rata** | **0,93990** | **0,93117** | **−0,00873** |

**Hierarchical bootstrap 95% CI untuk mean Δ:** **[−0,03022; +0,00761]**. CI melewati nol: tidak membuktikan adanya penurunan maupun kesetaraan/non-inferiority. Bootstrap CI ini dibaca dari artefak yang tersimpan, tidak dihitung ulang dari awal dalam audit ini. Satu keluaran CI historis sedikit berbeda; angka point estimate sama.

**Model B full8:** mean accuracy **0,9375**, mean macro-F1 **0,93619** pada 160 video development internal. Recall kelas baru: `halo` rata-rata **0,93**, `tidur` **0,99** (masing-masing 20 video unik yang berulang pada lima seed). H2 eksternal **belum dapat dinyatakan lolos**.

**Perubahan recall kelas lama (mean B−A):** `saya −3pp`, `kamu 0`, `senang −1pp`, `bingung 0`, `marah −3pp`, `apa_kabar 0`. Angka tersebut deskriptif; jangan mengurutkannya sebagai perbedaan signifikan tanpa interval simultan.

**Intrusion ke kelas baru:** 0,67% dari pasangan prediksi development (4 kejadian dari 600 prediksi seed×video berulang); `saya→tidur` dan `marah→halo`. Prediksi `apa_kabar→halo` **tidak ditemukan** pada validation subset ini. Pada dekomposisi akurasi I3, komponen kesalahan tambahan akibat full8-vs-restricted argmax adalah **nol** karena contoh intrusion tersebut tetap salah saat dibatasi ke enam output; tidak berarti intrusion tidak ada.

**Transisi prediksi berpasangan:** dari lima seed yang masing-masing menilai 120 video yang sama, terdapat **1 kasus** A salah→B benar dan **8 kasus** A benar→B salah. Ini 600 evaluasi berulang terhadap 120 video unik, **bukan 600 observasi independen**. Hipotesis bahwa dua kelas tambahan membantu diskriminasi belum didukung data development ini.

## 4. Human pilot dan kemungkinan shortcut

Human pilot (berbeda dari sampel validasi model): 160 frame internal, 20/kelas; reviewer 1 benar **129/160 = 80,625%**, reviewer 2 benar **138/160 = 86,25%**, Cohen's κ **0,82516**. Untuk `apa_kabar`, benar **9/20 (45%)** dan **14/20 (70%)**. Reviewer belum terbukti merupakan validator linguistik tersertifikasi.

Sebaliknya, model A/B mencapai **100% recall `apa_kabar`** pada validation subset internal Stage1b. Ini bukan bukti bahwa AI lebih memahami bahasa isyarat; kondisi, sampel dan potensi petunjuk latar berbeda.

Kontrol *hand masking* yang sudah ada mempertahankan mean accuracy **0,9200 (A common6)**, **0,91667 (B common6)**, **0,89125 (B full8)**. Masking berbasis landmark tidak menghilangkan wajah/tubuh dan mungkin tidak sempurna. Temuan ini memunculkan *shortcut/context cue concern*, tetapi **bukan bukti model hanya membaca background**.

## 5. Hipotesis: mana yang boleh diklaim?

**H1 utama (menunggu persetujuan amendemen):** laporkan estimasi paired mean Δ macro-F1 dan CI95%, bukan memaksakan uji non-inferiority dengan δ=0,02 yang belum dijustifikasi substantif. Nilai Δ bisa positif, negatif, atau tidak konklusif.

**H2 historis (pertahankan):** untuk kelas `halo` dan `tidur`, masing-masing *lower one-sided 95% CI* recall harus **>0,80** jika klaim konfirmatori nantinya sah. Development tidak dapat dipakai untuk memutuskan H2.

**Hipotesis mekanistik eksploratif:** tambahan kelas dapat memperjelas pemisahan antarkelas lama atau justru menimbulkan *new-class interference*. Motivasi muncul setelah human pilot dan hasil development; **bukan H3 konfirmatori yang ditetapkan sebelumnya**. Analisis old-to-old confusion, intrusion dan transisi prediksi menguji pola prediksi, tetapi tidak membuktikan mekanisme kausal representasi.

## 6. Hambatan utama sebelum klaim final

1. **Persetujuan akademik:** revisi judul, research question, Detection → Classification dan batasan frame-based.
2. **Rencana statistik:** H1 estimation-only versus NI margin dengan justifikasi, serta status proposal CV/faktorial lama.
3. **Training final:** config yang kini masih `PROPOSED_NOT_APPROVED`, 5 seed A/B dengan full locked train/val, final manifests, hash dan log.
4. **External readiness:** versi/lisensi Kaggle, provenance signer/session, riwayat penggunaan untuk tuning dan kelayakan status *confirmatory*, audit label/semantic frame.
5. **Penulisan:** pembahasan masking, signer dependence, keterbatasan satu frame, bias dari context cues dan perbedaan train samples/optimizer steps.

Jangan menjanjikan real-time, terjemahan urutan gestur, signer-independent generalization, ataupun hasil external sebelum pengukuran dan bukti tersedia.

## 7. Kalimat yang bisa diucapkan saat konsultasi (±90 detik)

> Pak/Bu, saya ingin meminta persetujuan penyesuaian fokus skripsi. Judul awal saya menekankan training-set expansion dan pengenalan frasa BISINDO real-time. Tetapi data dan eksperimen yang benar-benar saya gunakan adalah satu frame per video dengan satu label kelas, sehingga saya mengusulkan fokus **frame-based YOLO26 Classification**. Pertanyaan utamanya: setelah kelas BISINDO diperluas dari enam menjadi delapan, apakah performa enam kelas lama berubah dan kelas mana yang terdampak?
>
> Data internal saya berjumlah 2.400 video, ditambah 400 video eksternal. Human pilot 160 frame menunjukkan `apa_kabar` cukup ambigu pada penilaian manusia. Saya sudah menjalankan lima pasangan eksperimen development A/B dengan lima seed. Rata-rata macro-F1 enam kelas lama adalah **93,99%** untuk A dan **93,12%** untuk B; selisih **−0,87 poin persentase**, dengan interval 95% yang masih melintasi nol. Model B juga mengenali delapan kelas, tetapi belum ada bukti development bahwa kelas tambahan meningkatkan diskriminasi kelas lama.
>
> Saya melihat kontribusi penelitian ini sebagai evaluasi terkontrol atas *trade-off* perluasan vocabulary, bukan klaim arsitektur baru atau sistem real-time. Sebelum training final dan evaluasi eksternal, saya ingin meminta arahan Bapak/Ibu tentang revisi metode, judul, hipotesis statistik, dan cakupan eksperimen final.

## 8. Lima jawaban pendek untuk pertanyaan dosen

**Kenapa Classification, bukan Detection?** Karena endpoint yang dinilai adalah satu kelas per gambar, tanpa kebutuhan memprediksi lokasi objek. Ini perubahan task yang harus disetujui, bukan bukti Classification unggul dalam semua kondisi.

**Apa novelty-nya?** Evaluasi berpasangan lima seed pada enam kelas lama, analisis per kelas, perpindahan kesalahan ke kelas baru, serta evaluasi eksternal yang diaudit; bukan sekadar menggunakan YOLO versi terbaru.

**Bukankah B mendapat data lebih banyak?** Betul. Desain ini mengukur ekspansi vocabulary **bersama** tambahan data, output head dan optimisasi; tidak mengisolasi efek jumlah kelas.

**Apa hasil yang diharapkan?** Estimasi perubahan dan kelas terdampak yang kredibel, bukan target B harus lebih tinggi. Hasil negatif atau tidak konklusif tetap dapat dipublikasikan bila protokol dan interpretasinya kuat.

**Kenapa data Kaggle belum diuji?** Evaluasi eksternal masih digate karena provenance, riwayat pemanfaatan, label semantik dan freeze final belum lengkap. Nilai internal tidak dipromosikan menjadi hasil final.

## 9. Sumber di paket ZIP

- `STAGE1B_TEN_RUNS.csv` — sepuluh training run development.
- `EVIDENCE/01_hasil/addendum/development_interference_20261105/common6_primary_results.csv` — paired Δ dan CI.
- `EVIDENCE/01_hasil/addendum/development_interference_20261105/class_interference.csv` — recall per kelas.
- `EVIDENCE/01_hasil/addendum/development_interference_20261105/intrusion_rates.csv` — I2.
- `EVIDENCE/01_hasil/addendum/development_interference_20261105/accuracy_decomposition.csv` — I3.
- `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_masking_summary.csv` — kontrol masking.
- `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/pilot_summary.csv` — human pilot.
- `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/report.md` — resep dan keputusan terbuka.
- `VERIFIED_EXPERIMENT_RESULTS.md` — laporan ekspor panjang serta batas verifikasi.

**Catatan audit arsip di sesi ini:** ZIP terbaca utuh, 202 berkas dalam evidence manifest cocok dengan SHA-256 yang tercatat. Hasil model dilihat dari CSV dan prediksi yang tersimpan; checkpoint tidak dijalankan, bootstrap tidak diulang, dan sumber video asli tidak diverifikasi ulang.
