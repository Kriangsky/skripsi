# BISINDO Vocabulary Expansion — prospective protocol candidate v1

Tanggal: **2026-11-03 (Asia/Bangkok)**. Status: **AWAITING_STAGE0_APPROVAL**.
Jalur utama yang diputuskan peneliti: YOLO26 Classification. Detection diarsipkan sebagai future work.
Ini usulan protokol prospektif, bukan preregistrasi retrospektif bagi pilot yang hasilnya sudah diketahui.
File pendamping `PREREGISTRATION.sha256` mengikat byte kandidat ini; hash saja bukan persetujuan atau lock run.
Tidak ada training baru yang diizinkan melalui Tahap 0.

## 1. Pertanyaan dan hipotesis

Pertanyaan: berapa biaya ekspansi vocabulary 6 → 8 kelas pada kelas lama, dan apakah sepadan dengan
kemampuan mengenali dua kelas baru, halo dan tidur?

- H1 utama: non-inferiority B terhadap A pada common6 macro-F1. Definisikan d = F1_B − F1_A.
  Margin delta = **0.02** (2 percentage points), H0: d ≤ −0.02, H1: d > −0.02.
  Delta boleh direvisi oleh peneliti sebelum run prospektif pertama; setiap revisi diberi versi/hash baru.
  Jangan menyebut d > −delta pada satu estimasi titik sebagai bukti non-inferiority tanpa CI.
- H2 kemampuan baru: accuracy B pada GT halo **dan** GT tidur masing-masing ≥ **0.80**.
  Ambang 80% per kelas disetujui peneliti pada 2026-11-03, sebelum run baru.
  Laporkan point estimate dan CI per kelas. Aturan konfirmatori yang diusulkan: lower one-sided 95% CI
  masing-masing kelas > 0.80 (intersection-union: kedua komponen harus lolos).
  Point estimate ≥ 0.80 saja disebut memenuhi target deskriptif, bukan lulus pengujian konfirmatori.
- H3 sekunder: D versus A_n pada desain faktorial, d_DA > 0. Tidak diuji sebagai B versus A dengan data lama tetap.
  H3 menambah data lama sekaligus vocabulary; peningkatan tidak boleh diatribusikan ke dua kelas baru saja.
  Laporkan juga C−A_n, B_n−A_n, D−C dan D−B_n untuk memperjelas intervensi; kontras lain eksploratif.

Hasil null, inferiority, non-inferiority atau inconclusive semuanya sah. Tidak ada target agar B menang.

## 2. Endpoint dan estimand

Primer: mean antar-seed dari paired common6 macro-F1 difference, seed **[42,43,44,45,46]**.
Macro-F1 dirata-ratakan atas enam kelas GT tetap 0..5; argmax B tetap atas seluruh delapan output.
Prediksi halo/tidur pada GT common6 tetap salah/FN. Common6 dan full8 selalu dilaporkan terpisah.
Top-1 accuracy, per-class precision/recall/F1, confusion matrices dan error flags sebagai endpoint pendukung.
H2 adalah recall/accuracy bersyarat pada GT masing-masing kelas, bukan precision atau pooled 39/40 saja.

**Estimand compute utama: epoch sama**, seluruh seed/arm pada konfigurasi terpilih yang sama.
Checkpoint: `last.pt` setelah epoch yang ditetapkan, tanpa early stopping/best-seed/best-checkpoint selection.
Best.pt boleh disimpan framework tetapi tidak dipakai mengganti endpoint resmi.
Estimand suplemen: optimizer steps sama; bukan klaim FLOPs/wall-time sama.
Pilot 60/kelas: A 30 epoch = 690 actual steps; B 23 epoch = 690 jika pipeline/batch identik.
Suplemen ini jelas berlabel berbeda epoch, dikecualikan dari aturan epoch-sama primer dan tidak mengganti H1.
Pada data/konfigurasi lain budget steps dihitung ulang dari aktual A; angka 690 tidak digeneralisasi.

## 3. Data, split dan provenance

Mapping tetap: saya=0, kamu=1, senang=2, bingung=3, marah=4, apa_kabar=5, halo=6, tidur=7.
Sumber kebenaran: split_lock/manifest/run records/predictions > analisis artefak > README > draft > usulan.
2.400 video internal, 400 Kaggle. Frozen split asli 1.920 internal train, 480 internal val tidak diubah.
Sampling seed = **42** terpisah dari training seed; urutan sampling label-stratified deterministik.
Satu video maksimum satu frame training. Policy otomatis existing dipakai ulang, label/prediction-blind.
Sampel sulit/flagged tetap masuk. Kegagalan teknis dicatat; jangan ganti dengan sampel mudah atau oversampling.
Shared-six IDs, frame pixels, label dan preprocessing identik antar arm pada konfigurasi sama.
Metadata signer/sesi tidak dibuat dari wajah/penampilan; signer independence tidak diklaim.
Corruption/decode, class mapping, exact duplicates dan detectable leakage tetap hard gates.

Tahap 1 mempertahankan manifest pilot 60 train + 20 val/kelas dan frozen split lama secara byte-identical.
Data audit manusia lama tidak digunakan untuk memilih ulang val/test atau mengubah jawaban/release manusia.

Tahap 2 menggunakan file folds baru atas 2.400 video internal, bukan edit split_lock.csv.
Kumpulkan metadata TEKNIS container (width, height, fps rational, codec, creation_time bila ada).
Usulan grouping konservatif: tuple teknis exact dengan timestamp creation_time valid jika tersedia;
tuple tanpa timestamp hanya proxy konteks teknis, bukan identitas sesi. Laporkan missingness/ukuran grup.
StratifiedGroupKFold, 5 fold, shuffle/random_state 42 jika API mendukung dan semua kelas tersedia.
Jika tidak ada proxy usable atau tidak mungkin menghasilkan lima fold dengan coverage memadai,
hentikan untuk persetujuan fallback StratifiedKFold per video; jangan memecah grup diam-diam.
Exact duplicate-content cluster tidak boleh terpisah antar train/val bahkan pada fallback.

CV memakai ulang internal validation yang sudah dianalisis dan dapat memasukkan video studi manusia dengan
frame policy tetap: CV adalah development evidence, **bukan untouched confirmatory holdout**.
Tidak ada jawaban reviewer dalam frame selection. Exposure lama dan overlap wajib dilaporkan.
Tidak menyebut technical proxy sebagai signer/session-independent split.

## 4. Tahap 1 — diagnosis murah, persetujuan terpisah untuk training

Work baru yang diusulkan: `experiment_yolo26_classification_diag_v1/`; tidak menimpa run selesai.

1. `analysis_paired.py` + unit tests: join one-to-one sample_id/video untuk common6, cek GT/hash,
   McNemar exact two-sided (binomial discordants; nol discordants → p=1), Wilson 95% CI accuracy,
   paired bootstrap macro-F1/accuracy per video, 10.000 replikasi, RNG statistik 20261103.
   Resampling stratified GT class dengan indeks video yang sama untuk A/B; tanpa menganggap frames/seeds sebagai video baru.
   Buat tabel union errors (aktual lama: 9 video), error per kelas, CSV/MD; ini post-hoc exploratory.
2. Parameterisasi API/CLI seed, epochs, imgsz dan work aman beserta tes; perintah lama tetap kompatibel,
   train/evaluate tetap menolak overwrite/resume. Propagasi parameter ke letterbox, release fingerprint,
   probe, dataset transforms, completion checks, evaluation dan conclusion; jangan hanya mengubah CONFIG global.
3. Setelah izin training: 10 run baru A/B × 5 seed, baseline 224/30; seed 42 diulang dan dilabeli replikasi,
   bukan tambahan seed independen. Semua lima seed wajib masuk tabel. Kontrol noise A vs A' memakai
   A(seed42) melawan A(seed43..46), kontras saling berkorelasi; tidak perlu empat training tambahan.
4. Context shortcut: masking semua tangan dari cached landmarks, fixed union padded ROI 15%, fill abu 114,
   sebelum letterbox. Simpan mask dan coverage; tidak ada landmark → laporkan strata unmasked/unavailable
   tersendiri, jangan hilangkan denominator keseluruhan. Landmark bukan GT semua tangan.
   Hand-masked = kontrol background/context-only yang masih dapat memuat wajah/tubuh; bukan background murni.
   Background murni hanya jika segmentasi person tersedia dan tervalidasi, selain itu tandai unavailable.
   Laporkan chance seimbang 1/6 dan 1/8 dengan keterbatasan distribution shift akibat masking.
5. Linear probe diagnostik: frozen backbone dalam eval mode, head baru enam kelas, train lama saja;
   10 head fits (A/B backbone × 5 seed), SGD lr .01, momentum .937, decay .0005, batch16,
   30 epoch, tanpa augmentation, last head, tidak memakai val untuk pemilihan/early stopping.
6. Kurva train/val loss dan top-1 per epoch, skala tugas A6/B8 jelas. Run lama tidak mencatat train top-1;
   instrumen baru harus menghitung train top-1 pada no-augmentation train set tanpa mengubah data/weights.

Stop keras: technical leakage/corruption/mapping, API unsupported, GPU unavailable, hash drift.
Jika SD seed macro-F1 A atau B ≥ |mean paired d|, atau CI d melintasi −delta,
pilot dinyatakan tidak dapat memutuskan; usulkan CV, jangan mengejar seed pemenang.
Jika CI sempit dan lulus/menolak NI, tetap laporkan tahap ini exploratory dan minta keputusan lanjut peneliti.
Semua progres ke Tahap 2 memerlukan persetujuan; tidak ada auto-launch.

## 5. Tahap 2 — CV dan faktorial terkontrol

Kapasitas per fold bisa berbeda pada grouped CV. Usulan n=120 hanya jika minimum train count per kelas/fold ≥240.
Jika lebih kecil: sebelum model CV apa pun, tetapkan n=min(120, floor(min_capacity/2)) dan hash addendum;
perubahan kapasitas perlu persetujuan, bukan penyesuaian setelah skor terlihat. Tidak ada oversampling.

| Arm faktorial | Kelas | Data lama/kelas | Data baru/kelas |
|---|---:|---:|---:|
| A_n | 6 | n | 0 |
| B_n | 8 | n | n |
| C | 6 | 2n | 0 |
| D | 8 | 2n | n |

Subset n nested dalam 2n; shared-six sama A_n/B_n dan C/D; data baru B_n/D identik.
Validation common6 sama semua arm dalam fold. Full fold train sampai 240/kelas terdapat pada C/D,
sedangkan A_n/B_n sengaja lebih kecil untuk memisahkan faktor jumlah contoh lama.

Grid terbatas **9 kandidat**, bukan seluruh 27 Cartesian combinations:

| ID | imgsz | epochs | regularisasi |
|---|---:|---:|---|
| G01 | 224 | 10 | baseline |
| G02 | 224 | 15 | baseline |
| G03 | 224 | 30 | baseline |
| G04 | 320 | 15 | baseline |
| G05 | 320 | 30 | baseline |
| G06 | 448 | 15 | baseline |
| G07 | 448 | 30 | baseline |
| G08 | 320 | 15 | label_smoothing=0.05 |
| G09 | 320 | 15 | proposed label-safe ringan |

G09 usulan: HSV hue ±0.01 turn, saturation/value gain 0.95..1.05; isotropic scale 0.95..1.0,
translate ≤0.02 canvas hanya pada ruang kosong letterbox. Seluruh isi frame tetap di canvas,
padding114, tanpa crop/flip/rotasi. Validasi label-safety sebelum arm dijalankan.
Jika belum dapat disahkan, G09 tidak dijalankan sampai persetujuan revisi grid sebelum CV.
Label smoothing tidak tersedia sebagai flag pada API terpasang; implementasi criterion cross-entropy
teruji harus dilakukan eksplisit, bukan mengirim flag fiktif atau menganggap setting aktif.

90 fits: 9 config × 5 fold × A_n/B_n, training seed42 untuk screening yang sama.
Pilih SATU config yang memaksimalkan mean 5-fold dari (F1_A_n,common6 + F1_B_n,common6)/2.
Tie numerik ≤1e-6: pilih imgsz lebih kecil, lalu epoch lebih sedikit, lalu ID grid terkecil.
Jangan memilih memakai d_BA atau hasil Kaggle. Semua skor config/fold dan failure tercatat.
Setelah selection, C/D pada lima fold menambah 10 fits; A_n/B_n terpilih memakai hasil CV yang sudah ada.
Faktorial CV bersifat development/post-selection, bukan pengujian H3 konfirmatori independen.
Suplemen baseline pilot step-matched: lima B(seed42..46), 23 epoch, 690 steps; A pasangan dari Tahap 1.
Budget tahap ini: **105 fits** (90 tuning + 10 C/D + 5 B step-matched), izin dapat dipecah per subfase.

OOF predictions untuk audit noise berasal hanya model yang tidak melatih video tersebut; pakai semua delapan
output untuk penandaan kelas termasuk marah. Confident-learning opsional memerlukan tes/API; ranking bukan adjudikasi.
Data utama tidak diubah. Daftar review ahli acak stratified + flagged terpisah, tanpa prediksi/confidence/ranking
di UI reviewer; laporkan sampling probability dan gunakan random strata untuk estimasi noise populasi.
Sensitivity analysis koreksi label terpisah dari endpoint utama, tanpa mengganti run/data utama.

Stop jika fold/group integrity tidak valid, semantic-safe transform belum disetujui, atau budget/API gagal.
CV selesai → pilih config sesuai aturan, bekukan manifest/config/source/checkpoint, lalu minta persetujuan final.

## 6. Tahap 3 — final setelah freeze dan persetujuan

Primer full-data: A_full=6×240 train, B_full=8×240 train, frozen internal val 60/kelas,
masing-masing seed42..46: **10 fits**. Internal val telah dipakai untuk pengembangan, jadi hanya descriptive/supportive;
primary confirmatory domain yang diusulkan adalah Kaggle eksternal pada release/config terkunci.
H1/H2 dilaporkan pada internal juga, dengan label reused-validation, bukan dua bukti independen.

Faktorial final jika disetujui: n=120, A_n/B_n/C/D dengan frozen validation yang sama.
C= A_full dapat dipakai ulang (bukan fit independen); D berisi new120, berbeda dari B_full new240.
Tambahan A_n, B_n, D ×5 seed = **15 fits**; total final **25 fits**, bukan mengubah nama B_full menjadi D.
H3 = D−A_n pada common6 external. Kontras faktorial memperlihatkan efek tambahan data lama dan vocabulary.
Jika arm faktorial final tidak disetujui sebelum freeze, H3 tidak diuji konfirmatori; laporkan not performed.

Kaggle sudah diproses saat persiapan v2/v3; jangan menyebutnya belum pernah dibuka.
Peneliti menyetujui pada 2026-11-03: **satu batch evaluasi model final sesudah freeze**, semua seed/arm terdaftar
di batch tersebut, bukan satu panggilan inference saja. Tidak ada tuning/model selection setelah hasil external terlihat.
Rencana common6=300 gambar adalah subset full8=400, bukan 700 unik; maksimum satu frame/video.
Jika technical failures, laporkan coverage/reason dan hentikan sebelum membuka skor; jangan silently exclude.
Frame policy/preprocessing, final external selection manifest, code/hash dan semua checkpoints dikunci sebelum batch skor.

Statistik final: d per seed, mean±SD (ddof=1), paired hierarchical bootstrap 10.000 replikasi,
resample seed-pairs dan video secara berpasangan dengan video draw sama lintas semua seed;
GT-stratified video resampling, RNG 20261103. Laporkan juga video-only conditional CI dan seed-only variability.
Jangan menghitung 5×jumlah video sebagai sampel video independen. Tidak ada signer-cluster CI tanpa metadata.
H1: lower one-sided 95% bootstrap CI d > −0.02 → NI; CI melintasi batas → inconclusive.
Jika upper one-sided 95% CI < −0.02 → evidence of inferiority beyond margin.
H2 pakai mean accuracy per kelas across seeds dan hierarchical CI. Wilson per-model/seed sebagai pendukung,
bukan Wilson pada seed×video pooled trials. H1 & H2 harus lolos untuk klaim trade-off memenuhi target.
H3 diuji setelah H1 dan H2 lolos (fixed gatekeeping), lower one-sided 95% CI D−A_n >0.
Kontras lain dan per-class intervals bersifat exploratory tanpa klaim konfirmatori multipel.

Latency/FPS hanya setelah diukur: GPU lokal, batch1, FP32, letterbox/RGB/transfer+inference+argmax,
100 warmups, 1.000 timed frames ×5 repeat, CUDA synchronize, mean/SD/median/p95 ms dan FPS=1000/mean_ms.
Catat model/seed/hardware; exclude file decode dari inti ini secara eksplisit, laporkan decode-inclusive video-frame
pipeline terpisah. Tidak mengklaim latency gesture temporal atau aplikasi webcam dari tensor inference saja.
Audit semantik sampel acak yang nanti diisi peneliti: estimasi noise dengan bobot strata dan CI;
targeted flagged audit tidak dianggap random population noise estimate.

## 7. Config, lock dan pelaporan

Checkpoint awal semua arm: yolo26n-cls.pt, SHA256
`0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`.
Seluruh init independen dari file yang sama, bukan A→B; classifier head diganti sesuai kelas arm.
Baseline aktual: SGD lr0 .01/lrf .01/momentum .937/decay .0005, warmup3, linear LR,
batch16/nbs16, ampFalse, workers0, patience0, deterministicTrue, no augmentation.
Learning schedule sama dalam setiap perbandingan equal-epoch; supplement berbeda horizon harus diungkap.
Tidak ada silent CPU fallback, batch reduction, resume, penghapusan seed atau penggantian checkpoint karena skor.

Sebelum run prospektif pertama: persetujuan Tahap 0 + snapshot immutable candidate + SHA256 + activation record
yang membatasi fase/run/estimasi waktu yang disetujui. Tahap implementasi tanpa training juga perlu persetujuan lanjut.
Sebelum konfirmatori: config winner, dataset/video-frame manifests, class mapping, folds, preprocessing,
source/tests hash, deps/hardware, initial checkpoint dan rencana external diikat pada final freeze record.
Tidak dapat mengunci config winner sekarang karena CV belum dijalankan. Run menolak draft/tampered lock.
Revisi pra-run memakai addendum bernomor; perubahan setelah hasil diketahui dicatat sebagai deviation,
bukan mengubah protokol lama agar terlihat prospektif. SHA kandidat Tahap 0 bukan final confirmatory lock.

Semua run menyimpan args aktual, training/sampling seed, dataset hash, initial/final checkpoint hash,
environment, logs, actual optimizer steps dan aturan checkpoint. Semua work baru dan menolak overwrite.
Pelaporan: semua seed/config/failure, flags/ambiguous/unknown, point estimates/CI, denominators, data exposure;
pisahkan fakta, hipotesis, exploratory dan confirmatory. Jangan menjanjikan B menang.

## 8. Budget dan keputusan bertahap (estimasi, bukan run yang sudah dilakukan)

Waktu dasar dari results.csv run pertama: A30=37.4169 s, B30=47.5833 s;
reproduksi notebook A30=32.0603 s, B30=40.7609 s. Itu waktu training tercatat, bukan prepare/end-to-end.
Resolusi/data/IO/diagnostic overhead dapat menaikkan waktu jauh; ukur ulang melalui smoke berizin.

| Fase | Fits baru yang diusulkan | Estimasi wall-time lokal | Persetujuan |
|---|---:|---|---|
| 0 | 0 | Dokumen, notebook read-only, tes | Saat ini saja |
| 1a statistik + parameterisasi | 0 | 15–45 menit implementasi/QA; statistik detik–menit | Minta setelah Tahap 0 |
| 1b multi-seed + probes | 10 full + 10 heads | 15–60 menit termasuk preparation/QA; GPU train dasar beberapa menit | Minta sebelum training |
| 1c masking/curves | 0 full fits | 5–30 menit, tergantung cache | Tercantum izin diagnosis |
| 2 tuning CV | 90 | 1–4 jam training/IO; persiapan metadata terpisah | Minta sebelum subfase |
| 2 faktorial + steps | 10 + 5 | 20–90 menit | Minta sebelum subfase |
| 3 primer final | 10 | 10–60 menit training, bergantung config | Minta setelah freeze |
| 3 faktorial opsional | 15 tambahan | 15–90 menit | Izin eksplisit terpisah |
| 3 external + latency | 0 fits | 5–30 menit, di luar audit ahli | Izin one-shot setelah freeze |

Rentang waktu adalah engineering estimate, bukan benchmark. Tidak ada fase training yang auto-start.

## 9. Revisi dokumen berikutnya

Setelah persetujuan lanjut, buat draft baru:
`Effect of Vocabulary Expansion on YOLO26 Classification for BISINDO Phrase Recognition.md`.
Draft lama tetap. Tulis Classification/ImageNet, halo/tidur, letterbox dan resolusi hasil ablation,
last.pt, satu frame/video, H1 NI/H2/H3, top-1/macro-F1. Pisahkan PILOT EKSPLORATIF lama
dari HASIL KONFIRMATORI yang saat ini **belum ada**. Latency hanya jika terukur.
Limitations wajib: weak video labels, metadata signer/sesi tidak tersedia, closed-set, single-frame,
ambiguitas apa_kabar/halo pada studi manusia, validation dipakai diagnosis, tanpa temporal comparator,
Detection tidak terlatih. Detection future work, tanpa klaim superiority atau temporal necessity.
