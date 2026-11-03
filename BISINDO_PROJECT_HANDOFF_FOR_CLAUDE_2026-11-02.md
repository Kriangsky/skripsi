# Handoff lengkap proyek thesis BISINDO YOLO26 untuk Claude

**Snapshot: 2 November 2026, Asia/Bangkok.**

Dokumen ini self-contained: konteks thesis, tujuan, keputusan, histori eksperimen, hasil nyata, kode yang tersedia, keterbatasan, dan pertanyaan lanjutan. Tidak perlu membaca chat sebelumnya untuk memahami inti masalah. Referensi file lokal disertakan jika source project juga tersedia.

Jika hanya Markdown ini yang diberikan, jangan menganggap Anda telah membuka source, weights, atau gambar lokal. Analisis fakta yang disertakan dan minta artefak tambahan bila perlu mengisolasi penyebab lebih lanjut.

## 0. Apa yang saya minta dari Claude

Saya ingin pendapat ilmiah dan teknis tentang **cara meningkatkan Model B agar berpeluang mengungguli Model A pada enam kelas lama yang sama**, dengan perbandingan yang fair, reproducible, dan dapat dipertanggungjawabkan dalam thesis/paper.

B lebih baik adalah **target/hipotesis**, bukan izin untuk memilih seed/checkpoint/subset/metrik yang menguntungkan B. Kritik desain eksperimen jika target tersebut tidak didukung intervensi yang ada. Jangan menjanjikan B pasti menang.

Saya butuh:

1. Diagnosis berbasis artefak aktual, dipisahkan dari hipotesis yang belum diuji.
2. Penjelasan apakah desain 6 -> 8 kelas memang memberi alasan B membaik pada kelas lama.
3. Rencana eksperimen bertahap yang praktis untuk GPU lokal, dengan kontrol dan kriteria keputusan sebelum hasil dilihat.
4. Usulan perbaikan data, frame, resolusi, atau regularisasi yang menjaga makna BISINDO dan tidak mencemari validation/test.
5. Pendapat apakah Classification dilanjutkan sebagai baseline, Detection kembali dikerjakan, atau keduanya dibandingkan melalui eksperimen sepadan.
6. Cara menyelaraskan paper dengan metode dan eksperimen yang benar-benar dijalankan.

**Jangan** menyarankan membuang frame sulit setelah melihat error, menyembunyikan prediksi kelas baru, memilih seed pemenang, menggunakan Kaggle untuk tuning, atau mengklaim hasil final dari pilot ini.

## 1. Thesis, konteks, dan tujuan proyek

Judul kerja: **Effect of Training Set Expansion on YOLO26 Performance for Real-Time BISINDO Phrase Recognition**. Nama file draft masih memakai ejaan `YOLOv26`.

Latar belakang: memperluas vocabulary pengenalan BISINDO dari enam menjadi delapan kelas, sambil mengukur dampaknya terhadap kelas lama, performa keseluruhan, dan nantinya latency/FPS aplikasi real-time.

Desain thesis awal adalah **YOLO26 Detection**, bukan Classification:

- Model A: pretrained COCO -> fine-tuning enam kelas.
- Model B: pretrained COCO yang sama -> fine-tuning delapan kelas **secara independen**.
- B tidak menggunakan hasil A sebagai initial weights. Ini bukan sequential/continual fine-tuning A -> B.
- Evaluasi utama memakai gambar enam kelas yang identik.
- Evaluasi tambahan B memakai seluruh delapan kelas.
- Internal data untuk train/validation. Kaggle untuk external test setelah konfigurasi final dibekukan.
- Satu video maksimal menghasilkan satu frame training terpilih; kandidat/frame berdekatan bukan sampel training independen tambahan.

Kemudian ditambahkan **Classification exploratory pilot** agar kelayakan learning bisa diuji tanpa menunggu seluruh bounding box disahkan. Ini **belum otomatis mengganti metodologi thesis Detection**.

## 2. Class mapping aktual: dikunci

```python
BASE = ["saya", "kamu", "senang", "bingung", "marah", "apa_kabar"]
EXTRA = ["halo", "tidur"]
CLASSES = BASE + EXTRA

# saya=0, kamu=1, senang=2, bingung=3,
# marah=4, apa_kabar=5, halo=6, tidur=7
```

`apa kabar` dinormalisasi sebagai `apa_kabar`. Mapping lama 0–5 harus sama pada A/B.

**Draft belum sinkron:** masih menyebut kelas tambahan `terima kasih` dan `tolong`. Dataset dan kode aktual memakai **halo dan tidur**.

## 3. Dataset dan frozen split

Dataset aktual **2.800 video**: 2.400 internal dan 400 Kaggle. Per kelas 300 internal + 50 external.

| Kelas | Internal train | Internal val | Kaggle external | Total video |
|---|---:|---:|---:|---:|
| saya | 240 | 60 | 50 | 350 |
| kamu | 240 | 60 | 50 | 350 |
| senang | 240 | 60 | 50 | 350 |
| bingung | 240 | 60 | 50 | 350 |
| marah | 240 | 60 | 50 | 350 |
| apa_kabar | 240 | 60 | 50 | 350 |
| halo | 240 | 60 | 50 | 350 |
| tidur | 240 | 60 | 50 | 350 |
| TOTAL | 1.920 | 480 | 400 | 2.800 |

Frozen split: `experiment_yolo26_v3/split_lock.csv`, diwarisi dari inventaris v2. Split internal 80:20, stratified per kelas, seed 42, **berdasarkan video**, bukan signer.

Metadata signer/sesi belum tersedia. Video berbeda tidak membuktikan signer atau sesi berbeda. Jangan menyebut hasil signer-independent. Penampilan/wajah tidak dipakai untuk membuat identitas signer.

Catatan histori dan exposure external:

- `marah` internal sudah dikoreksi menjadi 300 sebelum inventaris v2/v3 saat ini.
- Arsip awal `experiment_yolo26/dataset_audit.json` masih mencatat **2.803 video**, termasuk `marah` internal 303. Ini histori lama, bukan source of truth saat ini. Dokumen ini tidak menyimpulkan bagaimana tiga file lama dikoreksi/dipindahkan/dihapus.
- Kaggle belum dipakai untuk tuning atau evaluasi model BISINDO final. Persiapan v2/v3 lama memang memproses kandidat external. Jadi jangan menyatakan external data sama sekali belum pernah dibuka/diproses.
- Classification tidak memakai external images untuk train/val atau skor tuning; cached external hashes hanya dibandingkan untuk mencegah duplikasi.
- Sampling pilot tidak memindahkan video antar frozen split.
- Balanced class count tidak menjamin keragaman signer, sesi, background, atau outfit.

Kapasitas full-data yang **direncanakan**, bukan dataset Detection yang sudah disahkan:

| Model | Train | Internal val | External common6 | External full8 |
|---|---:|---:|---:|---:|
| A, enam kelas | 1.440 | 360 | 300 | tidak applicable |
| B, delapan kelas | 1.920 | 480 | 300 gambar sama | 400, termasuk 300 common6 |

Common6 merupakan subset full8; bukan 700 test images unik.

## 4. Environment dan batas operasional

- Workspace Windows: `C:\Users\Ricardo Cuthbert\Downloads\skrpsi bwabwa`.
- Shell PowerShell; kernel notebook existing `thesis_cuthbert`.
- Executable eksperimen: `.venv-bisindo/Scripts/python.exe`.
- Python 3.11.5.
- GPU NVIDIA GeForce RTX 4070 Laptop GPU, sekitar 8 GB VRAM.
- torch 2.14.0+cu126; torchvision 0.29.0+cu126; CUDA runtime 12.6.
- Ultralytics 8.4.197; numpy 1.26.4; pandas 2.2.3; MediaPipe 0.10.20; OpenCV 4.10.0.84.
- Package efektif tercatat pada environment JSON dan requirements-resolved setiap tahap. Full freeze juga memuat package warisan yang belum tentu dipakai pipeline.
- YOLO berjalan **lokal**, bukan API berbayar. Tidak memakai cloud training/HUB. Settings/logging eksternal dinonaktifkan dan settings disimpan di output baru.
- Tidak ada CPU training fallback atau silent batch reduction.
- Source video, class mapping, split, hasil v2/v3, studi manusia, dan run selesai tidak boleh ditimpa.

## 5. Ringkasan semua tahap eksperimen

| Tahap | Output | Yang benar-benar dilakukan | Status training BISINDO |
|---|---|---|---|
| Audit awal/legacy | `experiment_yolo26/` | Audit video, inventory, extracted midpoint; arsip 2.803 | Tidak ada hasil training tercatat |
| Detection preparation v2 | `experiment_yolo26_v2/` | Midpoint, MediaPipe box proposals, COCO smoke | Tidak dilatih; QC pending |
| Frame candidates v3 | `experiment_yolo26_v3/` | 33.250 kandidat, landmark cache, contact sheets, review UI | Tidak dilatih; 0 approved semantic positives |
| Blind human feasibility | `experiment_yolo26_pilot/` | 160 frame; dua ekspor reviewer lengkap; analisis | Selesai sebagai studi manusia |
| Controlled Detection pilot | `experiment_yolo26_training_pilot/` | Antrean 640, environment/API probe, gates | BLOCKED; 0 training run |
| Exploratory Classification pilot | `experiment_yolo26_classification_pilot/` | Dua training independen 30 epoch + evaluasi internal | SELESAI; dua run nyata |
| Multi-seed/ablation/external/full data | Belum ada | Belum dijalankan | Jangan dianggap punya hasil |

## 6. Legacy dan Detection v2: hasil serta pelajaran

Legacy memakai frame tengah. Audit lama tidak menemukan decode errors atau exact duplicate video/midpoint groups pada arsip yang diperiksa. Itu bukan jaminan signer/session independence.

v2 memakai dataset aktual 2.800 video:

| Kelas | Proposal detected / 350 | No hands |
|---|---:|---:|
| saya | 343 | 7 |
| kamu | 319 | 31 |
| senang | 349 | 1 |
| bingung | 350 | 0 |
| marah | 348 | 2 |
| apa_kabar | 327 | 23 |
| halo | 349 | 1 |
| tidur | 335 | 15 |
| Total | 2.720 | 80 |

Dari 80 kegagalan, 79 internal dan 1 external. **2.720 detections bukan recognition accuracy dan bukan bounding box yang sudah disahkan.** Semua 2.800 masih pending human QC.

COCO checkpoint smoke berhasil di GPU. Catatan satu frame: preprocess sekitar 30,92 ms, inference 7,89 ms, postprocess 8,07 ms. Ini bukan hasil model BISINDO, bukan latency rata-rata pipeline, dan bukan bukti FPS real-time thesis.

Checkpoint Detection:

```text
repo: Ultralytics/YOLO26
file: yolo26n.pt
revision: 070ac3c51435984d992ba92eb3f234daf4e4500d
SHA256: 9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
```

## 7. Frame candidates v3 dan automatic selection

Masalah midpoint: tangan terdeteksi atau frame di tengah video belum tentu menunjukkan gesture target. Ada preparation, resting, transition dan tahap gerakan berurutan.

v3 mengambil kandidat pada 20%,30%,40%,50%,60%,70%,80% dari `(frame_count - 1)`. Jika semua gagal teknis, fallback 5%,10%,15%,85%,90%,95%. Indeks dibulatkan dan dibuat unik; decode berurutan.

MediaPipe Hand Landmarker IMAGE mode, maksimal dua tangan, detection/presence confidence 0,5. Box proposal adalah union landmarks dengan padding 15% per sisi. MediaPipe **bukan classifier BISINDO**.

Heuristik teknis:

- Landmark finite, jumlah titik valid.
- Border margin 1% untuk flag near-border/possible clipping.
- Sisi tangan minimum 16 px, area minimum 256 px.
- ROI tidak kosong dan box tidak degenerat.
- Laplacian variance ROI tangan pada 128x128, minimum 60 untuk eligible.

Threshold belum dikalibrasi sebagai standar semantik. Tangan kedua yang diperlukan bisa tidak terdeteksi. Tanpa flag tidak berarti gesture benar; flagged tidak selalu berarti gambar tidak berguna.

Hasil v3:

- 2.800 video; 33.250 kandidat JPEG.
- 1.120 kandidat technically eligible; 32.130 flagged.
- 546 video mempunyai rekomendasi eligible pada laporan v3.
- 2.800 contact sheets dan cache JSON/landmarks per video.
- **0 approved semantic-positive frames**, 2.800 needs_review; Detection training tidak diizinkan.

UI v3 class-aware menyediakan pemilihan kandidat, alasan penolakan, pemilihan tangan relevan dan koreksi box manual. Ini berbeda dari blind human feasibility UI.

Policy otomatis pada human pilot/Classification sama untuk seluruh kelas:

1. Kandidat readable dengan provenance/hash valid.
2. Utamakan kandidat tanpa flag.
3. Jika semua flagged: jumlah flag paling sedikit.
4. Fokus ROI lebih tinggi, frame index lebih kecil, candidate ID sebagai tie-break.

Class label dan prediction confidence tidak dipakai untuk memilih frame. Rekomendasi teknis bukan semantic approval.

## 8. Human single-frame feasibility: protokol

Tujuan: mengukur apakah frame otomatis dapat menyampaikan delapan kelas sebelum mengasumsikan satu frame mewakili seluruh phrase.

- 20 video per kelas dari internal **train**, total 160; seed 42.
- Tidak mengambil Kaggle atau memindahkan split.
- Signer metadata tidak tersedia; tidak dibuat identitas palsu.
- Frame dipilih otomatis dan **reviewer tidak boleh menggantinya**.
- HTML kedua reviewer mengacak urutan konsisten, memakai opaque IDs; tanpa nama kelas sumber, video, folder atau rekomendasi kelas.
- Ground truth terpisah; gambar/HTML/manifest diikat hash.
- Pilihan: delapan kelas + ambiguous + invalid, dengan alasan preparation/transition/blur/clipped/temporal insufficient.
- Dua ekspor lengkap sudah diimpor. Reviewer identity teknis `reviewer_1`/`reviewer_2`; independensi dua manusia bersifat prosedural, bukan autentikasi oleh HTML.
- Semua 160 tetap denominator, termasuk invalid/ambiguous atau selection failure.

Teknis: 43 eligible, 117 flagged, 0 missing. Coverage teknis 26,875%; inspectable 100%.

## 9. Human feasibility: hasil nyata

| Metrik | Reviewer 1 | Reviewer 2 |
|---|---:|---:|
| Benar / semua | 129/160 | 138/160 |
| Accuracy seluruh sampel | 80,625% | 86,250% |
| 95% Wilson CI | 73,81–86,00% | 80,06–90,74% |
| Memilih salah satu kelas/identifiable | 133/160, 83,125% | 144/160, 90,000% |
| Accuracy conditional identifiable | 96,99% | 95,83% |
| Ambiguous | 27 | 12 |
| Invalid | 0 | 4 |
| Alasan preparation | 7 | 30 |
| Alasan transition | 33 | 27 |
| Alasan temporal insufficient | 4 | 4 |

Alasan dapat overlap. Conditional accuracy tidak boleh menggantikan accuracy seluruh sampel; subset mudah saja akan menimbulkan selection bias.

Per kelas, masing-masing n=20:

| Kelas | R1 benar | R1 accuracy | R2 benar | R2 accuracy |
|---|---:|---:|---:|---:|
| saya | 16 | 80% | 19 | 95% |
| kamu | 17 | 85% | 18 | 90% |
| senang | 17 | 85% | 14 | 70% |
| bingung | 18 | 90% | 19 | 95% |
| marah | 17 | 85% | 20 | 100% |
| apa_kabar | 9 | 45% | 14 | 70% |
| halo | 16 | 80% | 16 | 80% |
| tidur | 19 | 95% | 18 | 90% |

Agreement **135/160 = 84,375%**. Cohen's kappa **0,82515954**, dengan 10 kategori nominal terpisah: delapan kelas, ambiguous, invalid.

- 31 video unidentifiable oleh setidaknya satu reviewer; 12 oleh keduanya.
- Pasangan apa_kabar/halo menghasilkan 7 kesalahan pooled antar-reviewer, bukan pasti 7 video unik.
- Wilson CI mengasumsikan observasi independen. Repeated signer/session dapat membuatnya terlalu optimistis; belum ada CI cluster signer yang sah.

Confusion manusia berikut: baris=kelas sumber; kolom=jawaban. `A`=ambiguous, `I`=invalid.

### Reviewer 1

```csv
GT,saya,kamu,senang,bingung,marah,apa_kabar,halo,tidur,A,I
saya,16,0,0,0,0,0,0,0,4,0
kamu,0,17,0,0,0,0,0,0,3,0
senang,0,0,17,0,0,0,0,0,3,0
bingung,0,0,0,18,0,0,0,0,2,0
marah,0,0,0,0,17,0,0,0,3,0
apa_kabar,0,0,1,0,0,9,0,0,10,0
halo,0,0,0,0,0,3,16,0,1,0
tidur,0,0,0,0,0,0,0,19,1,0
```

### Reviewer 2

```csv
GT,saya,kamu,senang,bingung,marah,apa_kabar,halo,tidur,A,I
saya,19,0,0,0,0,0,0,0,1,0
kamu,0,18,0,0,0,0,0,0,2,0
senang,0,0,14,1,0,0,0,0,4,1
bingung,0,0,0,19,0,0,0,0,1,0
marah,0,0,0,0,20,0,0,0,0,0
apa_kabar,0,0,0,0,0,14,1,0,4,1
halo,0,0,0,0,0,3,16,0,0,1
tidur,0,0,1,0,0,0,0,18,0,1
```

Interpretasi: banyak frame menyampaikan kelas, tetapi apa_kabar sulit dan halo kadang menyerupainya. Informasi temporal atau preparation-frame selection masih hipotesis, belum kausal. Tidak ada temporal comparator atau klaim publikasi/final-model readiness.

## 10. Controlled Detection training pilot: implemented, tetapi blocked

Output terpisah: `experiment_yolo26_training_pilot/`.

Target 60 positive train + 20 positive val per kelas. A 360/120; B 480/160. Setiap positive perlu verifikasi semantic gesture, box valid, seluruh tangan relevan tercakup, dan audit batch. Metadata signer/sesi diwajibkan untuk subset independen.

Status aktual:

- Antrean review 640 video.
- Positive verified **0**, explicit video rejections **0**.
- 640 video antrean kekurangan metadata signer/sesi.
- Dataset selected/released untuk training **0/640**.
- GPU, official COCO checkpoint, SGD dan detection end-to-end dry forward lolos API probe.
- **Tidak ada run training Detection, detection mAP, atau perbandingan A/B Detection terlatih.**

Config pilot yang direncanakan: 30 epoch, 640 px, batch 16, seed 42, SGD, tanpa augmentation linguistik berisiko. Ini bukan full-data 100 epoch/MuSGD pada draft.

API audit menemukan detection checkpoint lama punya one-to-one branch/YAML end2end, tetapi flag runtime perlu diaktifkan eksplisit pada versi library sekarang. Hal itu ditangani di Detection pilot; bukan penyebab Classification B kalah.

Evaluator Detection telah dibuat: common6/full8, AP, fixed-threshold precision/recall/F1, FP/FN, confusion, apa_kabar/halo diagnostics. B/common6 mempertahankan prediksi kelas ekstra. Belum ada hasil model terlatih dari evaluator ini.

Pelajaran: gate tersebut melindungi anotasi Detection, tetapi terlalu berat untuk exploratory learning tanpa box. Gate lama tidak dihapus; ditambahkan Classification terpisah.

## 11. Keputusan dan scope exploratory Classification

Peneliti tetap memilih frame-based recognition. LSTM/TCN tidak diimplementasikan atau dilatih.

Instruksi pilot tambahan:

- `yolo26n-cls.pt` Classification; label dari metadata video.
- Tidak perlu bounding-box label atau approval MediaPipe.
- Missing signer/sesi dan semantic review belum lengkap menjadi warning, bukan blocker exploratory training.
- Corruption, class mapping, split consistency, decoding, known duplicates dan detectable leakage tetap hard gates.
- Tidak perlu menunggu manual review seluruh 640 gambar.
- Audit stratified random 10–20 per kelas + problematic cases; semua flagged tetap dalam denominator.
- Tidak mengganti metodologi thesis otomatis; bukan performa final.

## 12. Dataset Classification yang benar-benar dipakai

Sampling seed 42, 60 train + 20 val per kelas, dari frozen internal split. Tidak menggunakan confidence/prediksi model atau jawaban reviewer untuk memilih frame.

| Kelas | Train | Val | Train eligible | Train flagged | Val eligible | Val flagged |
|---|---:|---:|---:|---:|---:|---:|
| saya | 60 | 20 | 23 | 37 | 11 | 9 |
| kamu | 60 | 20 | 11 | 49 | 3 | 17 |
| senang | 60 | 20 | 17 | 43 | 1 | 19 |
| bingung | 60 | 20 | 4 | 56 | 0 | 20 |
| marah | 60 | 20 | 6 | 54 | 3 | 17 |
| apa_kabar | 60 | 20 | 10 | 50 | 5 | 15 |
| halo | 60 | 20 | 19 | 41 | 9 | 11 |
| tidur | 60 | 20 | 14 | 46 | 7 | 13 |

- A: 360 train + 120 val, ID 0–5.
- B: 480 train + 160 val, ID 0–7.
- Shared-six training dan validation images **byte-identical**; class IDs sama.
- **143 eligible, 497 flagged (77,66%), 0 selection failure. Semua flagged disertakan.**
- Semantic expert verification pada release Classification: **0**. Seluruh frame berstatus semantik not_assessed; label adalah weak video labels.
- 1.760 video internal tidak tersampling dicatat sebagai fixed-capacity exclusion, bukan quality exclusion.
- Overlap dengan video feasibility manusia: **31 pada train, 0 pada validation**. Jawaban reviewer tidak dipakai sebagai supervision atau pemilihan ulang frame.
- Human pilot menguji internal train; Classification metric memakai internal val. Jangan membandingkan 45%/70% manusia vs 100% model untuk apa_kabar sebagai benchmark head-to-head pada gambar sama.
- Metadata signer verified 0; tidak ada klaim signer/session independence.

ImageFolder mengurutkan folder secara alfabet. Prefix `00_saya`, `01_kamu`, ..., `07_tidur` menjaga ID original. Grafik/report kembali memakai nama canonical. Label nama kelas tidak digambar ke input model.

## 13. Preprocessing Classification

1. Satu frame dipilih otomatis memakai policy yang sama dengan feasibility pilot.
2. Full-resolution selected JPEG disimpan untuk audit, byte-identical dengan kandidat sumber.
3. Resize aspect-preserving + full-frame letterbox menjadi **224x224**, padding 114.
4. RGB float /255; Normalize mean=(0,0,0), std=(1,1,1).
5. Dataset transforms memakai Resize(square), ToTensor, Normalize. Tidak ada crop terhadap raw frame, flip, rotation, erasing atau jitter.

Tujuan: crop default classifier dapat memotong tangan pada frame landscape. Letterbox mempertahankan isi tetapi detail tangan kecil berkurang pada 224 px; background/outfit tetap masuk.

Tidak ada bounding-box labels Classification. Landmark lama hanya memberi petunjuk teknis, bukan syarat gesture benar atau seluruh tangan terdeteksi.

## 14. Checkpoint dan training controls Classification

Pretrained **ImageNet**, bukan COCO Detection:

```text
repo: Ultralytics/YOLO26
file: yolo26n-cls.pt
revision: 9da872c500a0a6fc9526cf09108ee249489a3895
SHA256: 0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81
```

Hash dicocokkan dengan upstream LFS repo resmi. Masing-masing model membuat objek baru dari checkpoint ini. B tidak memakai A.

Same pretrained checkpoint tidak berarti classifier head 6-output dan 8-output identik; head harus disesuaikan dengan jumlah kelas.

Config eksplisit yang disimpan saat run:

```python
CONFIG = {
    "task": "classify", "epochs": 30, "imgsz": 224,
    "batch": 16, "seed": 42, "deterministic": True,
    "optimizer": "SGD", "lr0": 0.01, "lrf": 0.01,
    "momentum": 0.937, "weight_decay": 0.0005,
    "warmup_epochs": 3.0, "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1, "cos_lr": False, "patience": 0,
    "nbs": 16, "device": 0, "workers": 0, "amp": False,
    "compile": False, "pretrained": True, "resume": False,
    "single_cls": False, "fraction": 1.0, "cache": False,
    "rect": False, "multi_scale": 0.0, "dropout": 0.0,
    "degrees": 0.0, "translate": 0.0, "scale": 0.0,
    "shear": 0.0, "perspective": 0.0,
    "fliplr": 0.0, "flipud": 0.0,
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "bgr": 0.0,
    "mosaic": 0.0, "mixup": 0.0, "cutmix": 0.0,
    "copy_paste": 0.0, "close_mosaic": 0,
    "erasing": 0.0, "auto_augment": None,
    "val": True, "save": True, "plots": True,
    "exist_ok": False, "verbose": True,
}
```

Beberapa field transform inherited dari konfigurasi umum/detection dinonaktifkan; actual classification dataset transforms tetap yang dijelaskan di atas.

Kontrol aktual:

- Hash awal dicek sebelum setiap model.
- Mapping/shared images diperiksa.
- GPU wajib; tidak ada fallback CPU atau silent batch reduction.
- Batch nominal 16. A memiliki 23 minibatch/epoch, terakhir 8; B 30 minibatch/epoch, semua 16. Tidak ada drop-last atau oversampling untuk meratakan batch.
- **Actual SGD steps A=690, B=900**. Equal epochs bukan equal compute/updates.
- Kelas lama masing-masing tetap melihat 60 gambar x 30 epoch = 1.800 paparan. Tidak ada tambahan gambar lama pada B.
- Evaluasi utama ditetapkan **last.pt epoch 30 pada keduanya**, bukan best B versus last A.
- Ultralytics juga menyimpan best.pt dan melakukan final validation best; angka itu tidak menggantikan hasil utama last.pt.
- results.csv berisi epoch 1–30 pada masing-masing run. Counter sementara 31 pada final validation callback bukan epoch training ke-31; metadata mempertahankan framework_callback_epoch terpisah.
- Seed sama tidak membuat dua dataset/output spaces berbeda memiliki trajectory optimisasi identik.

## 15. Audit kualitas Classification yang dilakukan

Per kelas: 10 random train + 5 random val, ditambah 5 flagged di luar primary sample. Total **120 random stratified + 40 tambahan flagged = 160 gambar**.

Delapan contact sheets telah dilihat asisten untuk **audit visual teknis**, bukan adjudikasi semantik ahli BISINDO.

Pengamatan:

- Tiga possible nonrepresentative/preparation/resting examples: satu label tidur dan dua label kamu.
- Dua contoh suspected cross-class static-pose overlap apa_kabar/halo, dengan tangan dekat pelipis.
- Room/background/clothing cues tampak berulang. Ini potensi shortcut, bukan bukti model pasti memakai cue tersebut.
- Tidak ada signer ID yang ditebak dari wajah/penampilan.
- **0 semantic expert-verified frames** pada release Classification.
- Known adjudicated ambiguity = 0 karena belum ada adjudikasi, **bukan bukti tidak ada ambiguity**. Semantik 640 frame masih unknown/unassessed.
- Frame dan label tidak diganti atau dikeluarkan setelah audit/hasil model.

Hasil dicatat terpisah dari manifest beku pada `audit/technical_visual_observations.csv` dan `audit/technical_visual_summary.json`.

UI opsional `audit/review.html` menyediakan valid/preparation/resting/transition/ambiguous/temporal_ambiguity/wrong_label/clipped/blur. Impor hanya diagnostic, tidak mengganti frame/label atau diam-diam mengubah denominator.

## 16. Hasil Classification: dua run nyata

Selesai di GPU, 30 epoch, seed 42. Top-1 memakai argmax seluruh output, tanpa class filter.

| Model | Evaluation | N | Benar | Accuracy | Macro-F1 | Prediksi kelas ekstra |
|---|---|---:|---:|---:|---:|---:|
| A | common6 | 120 | 112 | 93,3333% | 0,931776 | 0 |
| B | common6 | 120 | 111 | 92,5000% | 0,930042 | 2 |
| B | full8 | 160 | 150 | 93,7500% | 0,935464 | tidak applicable |

Macro-F1 common6 memakai **enam kategori GT yang sama** untuk A/B. B tetap memiliki delapan output. Prediksi halo/tidur pada kelas lama dihitung salah/FN untuk GT. Tidak ada masking/renormalization pada evaluasi resmi.

Full8 adalah tugas berbeda; jangan membandingkan 93,75% B/full8 dengan 93,33% A/common6 sebagai bukti B unggul pada tugas identik.

### Per-class common6

Masing-masing kelas mempunyai 20 GT. Recall dapat dihitung dari benar/20.

| Kelas | A benar | A precision | A F1 | B benar | B precision | B F1 |
|---|---:|---:|---:|---:|---:|---:|
| saya | 18/20 | 0,9474 | 0,9231 | 18/20 | 1,0000 | 0,9474 |
| kamu | 20/20 | 0,8696 | 0,9302 | 20/20 | 0,9091 | 0,9524 |
| senang | 19/20 | 1,0000 | 0,9744 | 19/20 | 1,0000 | 0,9744 |
| bingung | 20/20 | 0,9524 | 0,9756 | 20/20 | 0,8696 | 0,9302 |
| marah | 15/20 | 1,0000 | 0,8571 | 14/20 | 1,0000 | 0,8235 |
| apa_kabar | 20/20 | 0,8696 | 0,9302 | 20/20 | 0,9091 | 0,9524 |

### Per-class B/full8

| Kelas | Benar / 20 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| saya | 18 | 1,0000 | 0,9000 | 0,9474 |
| kamu | 20 | 0,9091 | 1,0000 | 0,9524 |
| senang | 19 | 1,0000 | 0,9500 | 0,9744 |
| bingung | 20 | 0,8696 | 1,0000 | 0,9302 |
| marah | 14 | 1,0000 | 0,7000 | 0,8235 |
| apa_kabar | 20 | 0,8696 | 1,0000 | 0,9302 |
| halo | 19 | 0,9500 | 0,9500 | 0,9500 |
| tidur | 20 | 0,9524 | 1,0000 | 0,9756 |

Apa_kabar recall 20/20 pada kedua model, tetapi masih menerima false positives. Halo B 19/20; satu -> apa_kabar. Kelas tersulit pada validation ini justru marah.

B tidak lebih buruk pada seluruh per-class metric: F1 common6 saya, kamu dan apa_kabar meningkat; senang tetap; bingung dan marah turun. Macro-F1 keseluruhan tetap sedikit lebih rendah. B juga benar pada **39/40 gambar kelas baru** halo/tidur; ini tambahan kemampuan fungsional, bukan bukti superiority pada common6.

## 17. Confusion matrices Classification lengkap

Baris = label video/GT metadata; kolom = prediksi. Urutan canonical mengikuti ID original.

### A common6

```csv
GT,saya,kamu,senang,bingung,marah,apa_kabar
saya,18,0,0,0,0,2
kamu,0,20,0,0,0,0
senang,0,0,19,0,0,1
bingung,0,0,0,20,0,0
marah,1,3,0,1,15,0
apa_kabar,0,0,0,0,0,20
```

### B common6

```csv
GT,saya,kamu,senang,bingung,marah,apa_kabar,halo,tidur
saya,18,0,0,1,0,0,0,1
kamu,0,20,0,0,0,0,0,0
senang,0,0,19,0,0,1,0,0
bingung,0,0,0,20,0,0,0,0
marah,0,2,0,2,14,1,1,0
apa_kabar,0,0,0,0,0,20,0,0
halo,0,0,0,0,0,0,0,0
tidur,0,0,0,0,0,0,0,0
```

### B full8

```csv
GT,saya,kamu,senang,bingung,marah,apa_kabar,halo,tidur
saya,18,0,0,1,0,0,0,1
kamu,0,20,0,0,0,0,0,0
senang,0,0,19,0,0,1,0,0
bingung,0,0,0,20,0,0,0,0
marah,0,2,0,2,14,1,1,0
apa_kabar,0,0,0,0,0,20,0,0
halo,0,0,0,0,0,1,19,0
tidur,0,0,0,0,0,0,0,20
```

Pada B/common6, baris kelas baru nol karena tidak ada GT halo/tidur. Kolomnya tetap ada agar extra-class errors terlihat.

## 18. Audit terbaru: mengapa skor B di bawah A?

Analisis post-hoc ini read-only: tidak training ulang atau mengubah skor resmi.

| Hasil paired pada 120 gambar identik | Jumlah |
|---|---:|
| Keduanya benar | 111 |
| A benar, B salah | 1 |
| A salah, B benar | 0 |
| Keduanya salah | 8 |

Seluruh delapan kesalahan A tetap salah pada B. B kehilangan **satu sampel tambahan**. Selisih accuracy **-0,8333 percentage point**, bukan bukti keunggulan umum A atau kegagalan besar B.

### Sampel pembeda tunggal

```text
sample_id: c344648e8dc5e8f1a15c
video: train/marah/marah_219.mp4
frozen split: validation
frame index: 34
label video: marah
A: marah, confidence 0.7036727667, benar terhadap metadata
B: kamu, confidence 0.8238225579, salah terhadap metadata
B probability marah: 0.0359912664, rank 4
quality flag: possible_hand_blur
ROI focus: 51.6164; technical threshold 60
raw image: 848 x 478
estimated landmark extent at input 224: sekitar 11 x 16 px
```

Ukuran tangan berasal dari rentang landmark cache, bukan bounding-box GT manusia. A benar pada frame yang sama; blur/resolusi belum otomatis menjelaskan mengapa hanya B salah. Ini petunjuk untuk ablation, bukan sebab kausal yang terbukti.

### Dua prediksi kelas tambahan bukan penyebab langsung selisih bersih

- GT marah, sample `032eb994a57dd0fbf676`: A salah -> saya; B salah -> halo.
- GT saya, sample `df79093028a6307795c2`: A salah -> apa_kabar; B salah -> tidur.

Keduanya sudah salah pada A. Diagnosis tambahan non-resmi mengambil argmax hanya enam output lama dari probabilitas B yang tersimpan. Hasil tetap **111/120**. Ini tidak mengubah skor utama dan tidak membenarkan memfilter output B pada laporan resmi.

Kesimpulan sempit yang sah: penurunan satu gambar berasal dari perubahan keputusan antar kelas lama, marah -> kamu. Belum dapat dipisahkan apakah penyebab training-nya variasi random, head, shared representation, data/label atau kombinasi.

### Semua sembilan gambar yang salah pada salah satu model

| sample_id | GT | A | Conf A | B | Conf B |
|---|---|---|---:|---|---:|
| 032eb994a57dd0fbf676 | marah | saya | 0,5850 | halo | 0,4543 |
| 180528e3b77a7ce444fa | marah | kamu | 0,4893 | apa_kabar | 0,3755 |
| 180c5230aa4c11674137 | saya | apa_kabar | 0,7157 | bingung | 0,7804 |
| 1ad79dfe0ff23b2caa1a | marah | kamu | 0,4572 | bingung | 0,4526 |
| 1fd8ed18034c48e061c4 | senang | apa_kabar | 0,5272 | apa_kabar | 0,6604 |
| 4484dfbcc18f5aac559d | marah | kamu | 0,9768 | kamu | 1,0000 |
| 539ed2712394ef6e7607 | marah | bingung | 0,9929 | bingung | 0,9927 |
| c344648e8dc5e8f1a15c | marah | marah | 0,7037 | kamu | 0,8238 |
| df79093028a6307795c2 | saya | apa_kabar | 0,4903 | tidur | 0,4444 |

Confidence B 1,0000 pada tabel adalah pembulatan dari sekitar 0,999970, bukan probability persis 1.

### Optimisasi dan indikasi overfit

| Catatan | A | B |
|---|---:|---:|
| Training loss epoch 30 | 0,00042 | 0,00055 |
| Validation loss epoch 30 | 0,27430 | 0,34765 |
| First epoch with maximum top-1 | 18 | 14 |
| Max top-1 pada scope masing-masing | 95% common6 | 95% full8 |
| Last top-1 pada scope masing-masing | 93,33% common6 | 93,75% full8 |

B validation loss 0,2835 pada epoch 14 naik menjadi 0,34765 pada epoch 30 saat train loss hampir nol. Pola ini konsisten dengan risiko overfit, tetapi bukan bukti kausal untuk selisih common6. Scope kurva A=6 kelas, B=8 kelas; ini bukan dua kurva common6 identik.

Menambah epoch ke 100 bukan solusi pertama yang terbukti. Memilih best.pt hanya untuk B juga tidak fair jika A tetap last.pt. Aturan checkpoint harus ditetapkan sebelum perbandingan dan diterapkan konsisten.

### Quality flag bukan penjelasan tunggal

Pada common6 val, kedua model mendapat 23 eligible dan 97 flagged:

- Eligible: A/B sama-sama 21/23 benar.
- Flagged: A 91/97, B 90/97 benar.
- Marah train 54/60 flagged; val 17/20 flagged.
- Tetapi bingung val 20/20 flagged dan A/B tetap 20/20 benar.

Jangan menyamakan technical flag dengan semantic invalid atau menganggap membuang flagged membuktikan peningkatan.

### Konfigurasi yang telah diaudit

- Hash checkpoint awal sama.
- Actual args sama kecuali data path, nama run dan output path.
- Shared images/class IDs konsisten.
- B bukan sequential fine-tuning dari A.
- Prediksi ekstra tidak disembunyikan dalam metrik resmi.

Tidak ditemukan kesalahan hitung/config pada jalur yang diperiksa yang menjelaskan selisih tersebut. Ini bukan pembuktian bahwa tidak mungkin ada bug lain atau bahwa sebab kausal sudah diketahui.

## 19. Mengapa lebih banyak kelas tidak otomatis membantu kelas lama?

- A: 6 x 60 = 360 training images.
- B: 6 x 60 lama + 2 x 60 baru = 480.
- B mendapat **0 tambahan contoh untuk tiap kelas lama**.
- Porsi data lama berubah 100% -> 75%, tetapi total paparan setiap kelas lama tetap sama selama 30 epoch.
- Jumlah output, objective competition, urutan minibatch dan optimizer steps ikut berubah.

B perlu memperoleh transfer fitur yang membantu kelas lama agar improve. Transfer positif mungkin, tetapi tidak dijamin. Satu selisih gambar juga belum membuktikan negative transfer sistematis.

Istilah training set expansion perlu jelas: **ekspansi vocabulary/classes** dan **penambahan jumlah contoh dalam kelas lama** adalah intervensi berbeda. Current A/B menguji tambahan label-space beserta data kelas baru, bukan pure data-volume effect pada label-space tetap.

## 20. Batasan yang tidak boleh hilang dari interpretasi

1. Satu training seed: belum ada mean/std atau paired delta stability antar-run.
2. Internal val 20 per kelas: satu gambar mengubah common6 accuracy 0,8333 pp.
3. Signer/session independence belum diverifikasi meskipun exact duplicate checks lolos.
4. Weak video labels: gesture pada setiap frame belum diverifikasi ahli.
5. Wajah, pakaian dan background ikut masuk; context shortcut mungkin.
6. Technical flags belum dikalibrasi sebagai semantic confidence.
7. Human study dan classifier val memakai sampel/split berbeda; tidak sah head-to-head langsung.
8. Closed-set classifier selalu memilih kelas; belum ada background/unknown gesture rejection benchmark.
9. Tidak ada temporal comparator; tidak ada bukti kausal perlu LSTM/TCN.
10. Belum ada Detection A/B terlatih pada release sepadan untuk membuktikan salah satu pendekatan lebih unggul.
11. Belum ada external Kaggle scores model terlatih.
12. Belum ada latency/FPS benchmark valid seluruh pipeline real-time BISINDO terlatih.
13. Equal epochs berbeda dari equal updates/compute; estimand perlu dinyatakan.
14. Validation sudah digunakan untuk diagnosis. Hasil perbaikan pada validation yang sama bukan final unbiased external evidence.

## 21. Ketidaksinkronan draft paper

File: `Effect of Training Set Expansion on YOLOv26 Performance for Real-Time BISINDO Phrase Recognition.md`.

Draft masih memuat:

- Extra classes terima kasih/tolong, bukan halo/tidur.
- Detection, pretrained COCO, 640x640, 100 epoch, MuSGD/cosine.
- mAP/precision/recall dan latency/FPS sebagai endpoint.
- Augmentation termasuk horizontal flip/random crop, padahal makna linguistik belum divalidasi dan transform tersebut dimatikan pada Classification pilot.
- Narasi annotations sudah diverifikasi dan detection failures dibuang. Status actual Detection masih 0 approved, sehingga narasi past-tense itu belum didukung implementasi/hasil sekarang.
- Template IEEE pada abstract/keywords dan boilerplate lain; belum manuskrip final.

Comparative analysis draft menyebut apakah tambahan frasa **strengthened or weakened** recognition kelas lama. Naskah lama tidak mensyaratkan B harus menang. Peneliti sekarang ingin mencari intervensi yang membuat B lebih baik; perubahan arah perlu menjadi hipotesis/desain transparan.

Jangan mengubah angka Classification accuracy/F1 menjadi Detection mAP, mendeskripsikan run ImageNet sebagai COCO, atau menyebutnya hasil final metode draft. Jangan mengubah draft otomatis sebelum peneliti memilih pendekatan final.

## 22. Eksperimen yang belum dijalankan

Berikut **usulan, bukan hasil**:

- Multi-seed A/B, misalnya 42,43,44,45,46 pada sampling dan split yang tetap.
- Resolusi 224 versus 320/448; full frame versus ROI/upper-body yang aman.
- Semantic data-quality/label audit dengan protokol independen.
- Regularisasi/dropout atau optimizer/LR search dengan budget dan aturan sama untuk A/B.
- Kontrol pemisahan efek jumlah data lama dan jumlah kelas.
- Matched optimizer-step comparison sebagai supplement equal-epoch experiment.
- Full-data training, beberapa seed, lalu Kaggle setelah konfigurasi final dikunci.
- Controlled Detection training dan perbandingan sepadan dengan Classification.
- Temporal comparator LSTM/TCN, video-level aggregation atau temporal recognition.
- Webcam deployment, unknown/background rejection, latency/FPS end-to-end.

Contoh kontrol factorial untuk diskusi, **belum dilaksanakan**:

| Arm | Kelas | Contoh per kelas lama | Contoh per kelas baru | Total train |
|---|---:|---:|---:|---:|
| A existing | 6 | 60 | 0 | 360 |
| B existing | 8 | 60 | 60 | 480 |
| C kontrol volume | 6 | 120 | 0 | 720 |
| D expanded B | 8 | 120 | 60 | 840 |

Common validation tetap sama. Shared images identik untuk pasangan A/B dan C/D; subset 60 dapat dibuat nested dalam 120. New-class subset B/D juga tetap. C/A dan D/B menguji tambahan data lama; B/A dan D/C menguji tambahan kelas pada data lama tetap.

Jika D menang atas A, jangan mengatribusikan semua peningkatan ke kelas baru: data lama juga bertambah. Compute/updates tetap perlu dilaporkan, dengan matched-compute control jika efeknya ingin dipisahkan.

## 23. Pertanyaan konkret kepada Claude

Tolong jawab berurutan:

1. Dari satu lost marah sample, delapan persistent errors, weak labels dan kurva, diagnosis apa yang terbukti? Apa yang belum diketahui?
2. Apakah B harus outperform A masuk akal ketika setiap kelas lama memakai data identik? Perlukah redefinisi hipotesis/intervensi?
3. Mana prioritas: multi-seed, review training marah, resolusi tangan, ROI aman, regularisasi, atau tambahan data lama? Jelaskan mekanismenya.
4. Apa ablation minimal untuk membedakan class interference/negative transfer, random variance, label noise, low-resolution detail loss dan context shortcuts?
5. Bagaimana kontrol agar peningkatan B tidak sekadar akibat compute lebih banyak atau selection bias?
6. Bagaimana memilih checkpoint fair dan melaporkan semua seed/uncertainty tanpa memilih pemenang?
7. Apakah peningkatan coverage delapan kelas dengan sedikit penurunan common6 bisa menjadi trade-off scalability yang layak diteliti? Jangan menjamin kelayakan publikasi.
8. Apakah Detection tetap jalur utama sesuai draft, atau Classification bisa menjadi final setelah revisi metode? Bukti tambahan apa yang dibutuhkan?
9. Apa yang harus direvisi di judul, metode, hipotesis, dan metrik paper; apa yang belum layak diklaim?
10. Buat rencana tiga tahap: diagnosis murah; controlled improvement pilot; final multi-seed/external. Untuk setiap tahap: perubahan, kontrol, metrik, risiko, kondisi lanjut/berhenti.

Jangan hanya menyarankan tambah epoch/data. Kaitkan usulan ke bukti atau mekanisme yang bisa diuji. Nyatakan jika tidak ada cara jujur menjamin B menang.

## 24. Aturan untuk rekomendasi/implementasi berikutnya

- Reuse project existing, jangan mulai dari nol.
- Jangan overwrite source video, mapping, split_lock, v2/v3, human pilot atau completed run.
- Jangan memakai jawaban manusia untuk memilih ulang validation/test atau memperbaiki pilot manusia beku.
- Jangan memilih frame berdasarkan confidence/prediksi YOLO atau validation score.
- Jangan membuang hard samples untuk menaikkan accuracy; laporkan denominator, flags, exclusions dan sebab.
- Jangan menjadikan missing signer/sesi atau review 640 gambar sebagai blocker exploratory Classification ketika belum tersedia. Tetap beri warning; validitas final tetap dibahas.
- Jangan melemahkan hard gates corruption/mapping/decode/known duplicate/detectable leakage.
- Jangan otomatis menerapkan flip atau crop yang mengubah arti gesture.
- Jangan post-hoc memilih best seed atau hanya best B checkpoint agar mengalahkan A.
- Jangan memfilter halo/tidur dari B/common6.
- Jangan membandingkan B/full8 versus A/common6 sebagai tugas identik.
- Jangan memakai Kaggle untuk tuning/model selection setelah test dibuka.
- Jangan klaim signer-independent, fully verified, real-time, superiority terhadap Detection, atau temporal necessity tanpa bukti.
- Jangan menjalankan ulang training untuk pertanyaan handoff ini; berikan analisis dan rencana terlebih dahulu.

## 25. Peta source dan artefak

```text
BISINDO_YOLO26_Training_Set_Expansion.ipynb   # existing Detection/v3 notebook
bisindo_pipeline.py                        # inventory, split, hash, box, QC/release/train Detection
frame_candidates.py                       # candidate/cache/technical checks/frame review
pilot_study.py                            # blind human feasibility/import/metrics
training_pilot.py                         # controlled Detection pilot, blocked
training_pilot_eval.py                    # Detection matching/AP/fixed-threshold diagnostics
classification_pilot.py                   # actual Classification prepare/train/evaluate

test_bisindo_pipeline.py
test_frame_candidates.py
test_pilot_study.py
test_training_pilot.py
test_classification_pilot.py

BISINDO_PIPELINE_README.md                  # arsip v2
FRAME_SELECTION_README.md                 # v3
PILOT_STUDY_README.md                      # human feasibility
TRAINING_PILOT_README.md                  # blocked Detection pilot
CLASSIFICATION_PILOT_README.md             # completed Classification

experiment_yolo26/                         # legacy stale 2.803 count
experiment_yolo26_v2/                      # corrected 2.800 midpoint preparation
experiment_yolo26_v3/                      # frozen split/candidates/cache/review UI
experiment_yolo26_pilot/                   # frozen human study + imported responses/analysis
experiment_yolo26_training_pilot/          # blocked Detection; runs empty
experiment_yolo26_classification_pilot/    # completed Classification A/B
```

**Notebook Classification end-to-end baru belum dibuat.** Permintaan notebook ditunda demi handoff ini. Notebook Detection existing tidak boleh dianggap sudah menjalankan Classification.

Artefak Classification:

```text
checkpoint.json
checkpoints/yolo26n-cls.pt
manifest.csv
release.json
selected_frames/<sample_id>.jpg
datasets/A/{train,val}/<numeric_class>/*.png
datasets/B/{train,val}/<numeric_class>/*.png
reports/experiment_config.json
reports/environment.json
reports/readiness.json
reports/verification.json
reports/frame_quality_counts.csv
reports/not_sampled_internal.csv
reports/training.log
audit/audit_samples.csv
audit/review.html
audit/<class>.jpg
audit/technical_visual_observations.csv
audit/technical_visual_summary.json
runs/model_A/{args.yaml,results.csv,optimizer_steps.json,weights/{best,last}.pt}
runs/model_B/{args.yaml,results.csv,optimizer_steps.json,weights/{best,last}.pt}
runs/model_A_execution.json
runs/model_B_execution.json
runs/trained_models.json
evaluation/comparison_metrics.csv
evaluation/per_class_metrics.csv
evaluation/predictions.csv
evaluation/confusion_A_common6.{csv,png}
evaluation/confusion_B_common6.{csv,png}
evaluation/confusion_B_full8.{csv,png}
evaluation/apa_kabar_halo_diagnostics.csv
evaluation/error_examples/
pilot_conclusion.md
```

Helper existing yang dapat digunakan ulang jika benar: hash/safe_path, class constants, sampling, letterbox, inspect_candidates, choose_candidate, folder mapping, release validation, metrics/matching/evaluation, dan audit import.

Classification saat ini memakai konstanta SIZE=224, TARGETS 60/20, CONFIG seed=42. CLI belum mempunyai flag bebas `--seed`, `--imgsz` atau loop multi-seed. Usulan ablation memerlukan implementasi teruji dan work directory baru; jangan mengarang CLI flag yang belum didukung.

## 26. Perintah existing dan verifikasi

Dari root project, memakai environment existing:

```powershell
.\.venv-bisindo\Scripts\python.exe classification_pilot.py prepare
.\.venv-bisindo\Scripts\python.exe classification_pilot.py probe
.\.venv-bisindo\Scripts\python.exe classification_pilot.py train
.\.venv-bisindo\Scripts\python.exe classification_pilot.py evaluate
```

Ini dokumentasi, bukan instruksi menjalankan ulang pada output selesai. Train/evaluate menolak overwrite/resume. Train otomatis mengevaluasi setelah dua model selesai. Command mendukung `--root`, `--source`, `--work`.

Untuk replikasi gunakan work directory baru dan checkpoint reference sama. Artifact/schema/paths harus valid. Belum ada notebook Classification baru untuk ini.

Audit semantik opsional:

```powershell
.\.venv-bisindo\Scripts\python.exe classification_pilot.py import-audit --file "C:\path\classification_quality_audit.json"
```

Tes:

```powershell
.\.venv-bisindo\Scripts\python.exe -m unittest test_classification_pilot test_training_pilot test_pilot_study test_bisindo_pipeline test_frame_candidates -v
```

Terakhir tercatat **46 passed, 0 failed**, termasuk regresi pipeline lama. Tes tidak dijalankan ulang hanya untuk menulis handoff. Independent checks sebelumnya mengonfirmasi accuracy/F1/matrix, argmax seluruh output, steps/epochs, shared byte-identical images, frozen split dan perlindungan source.

`verify_notebook.py` adalah verifier notebook Detection existing dan dapat menyimpan ulang output notebook itu. Jangan menggunakannya sebagai verifier Classification.

## 27. Status akhir dan sumber kebenaran

**Selesai:** preparation/candidates; studi manusia dan analisis; dua Classification training 30 epoch; evaluasi, audit visual teknis dan paired-error analysis.

**Belum selesai:** semantic-certified Detection dataset; Detection training; expert semantic verification Classification; multi-seed/ablation; external testing; deployment/latency benchmark; keputusan final metodologi thesis.

Jika sumber bertentangan, urutan otoritas:

1. Frozen split/manifest, run records, actual predictions/results CSV dan hashes.
2. Analisis yang dihitung dari artefak tersebut.
3. README tahap terkait, dengan membedakan archive/aktif.
4. Draft paper sebagai rancangan belum sinkron, bukan bukti pekerjaan sudah selesai.
5. Hipotesis/usulan diskusi, bukan hasil eksperimen.

Referensi eksternal implementasi, bukan benchmark project:

- Classification resmi: https://docs.ultralytics.com/tasks/classify
- Pinned classifier: https://huggingface.co/Ultralytics/YOLO26/blob/9da872c500a0a6fc9526cf09108ee249489a3895/yolo26n-cls.pt
- Pinned detection: https://huggingface.co/Ultralytics/YOLO26/blob/070ac3c51435984d992ba92eb3f234daf4e4500d/yolo26n.pt
- Model-selection overfitting: https://jmlr.org/papers/v11/cawley10a.html

**Inti permintaan:** bantu meningkatkan B melalui eksperimen yang masuk akal dan dapat diaudit, sambil tetap bersedia menyimpulkan tidak ada improvement jika bukti akhirnya demikian. Jangan menyelesaikan masalah dengan mengubah target evaluasi atau memilih angka yang akan dilaporkan.
