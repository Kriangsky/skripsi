# BISINDO — YOLO26 Training Set Expansion

Project skripsi **Effect of Training Set Expansion on YOLO26 Performance for Real-Time BISINDO Phrase Recognition**.
Penelitian menguji pengaruh perluasan kelas terhadap pengenalan BISINDO, khususnya apakah model delapan kelas
dapat mempertahankan atau meningkatkan performa pada enam kelas awal.

Repo memuat kode, notebook dengan output eksekusi, konfigurasi, metrik, dan laporan eksperimen.
Video, gambar peserta, cache, halaman review dengan gambar tertanam, environment, dan weights tetap lokal.

## Tujuan project

1. Menyiapkan pipeline dataset dengan split, pemilihan frame, dan eksperimen yang dapat ditelusuri.
2. Membandingkan Model A (6 kelas) dan Model B (8 kelas) pada validation enam kelas yang identik.
3. Mengevaluasi Model B pada delapan kelas sebagai tugas terpisah.
4. Memeriksa confusion, kualitas frame, dan kelayakan single-frame recognition sebelum eksperimen final.

Keunggulan Model B adalah hipotesis yang diuji, bukan hasil yang dipaksakan.
Walaupun judul menargetkan real-time, pilot ini belum membuktikan performa aplikasi real-time.

## Kelas dan dataset

| ID | Kelas | Model |
|---:|---|---|
| 0 | saya | A, B |
| 1 | kamu | A, B |
| 2 | senang | A, B |
| 3 | bingung | A, B |
| 4 | marah | A, B |
| 5 | apa_kabar | A, B |
| 6 | halo | B |
| 7 | tidur | B |

Inventory lokal: **2.400 video internal** (300/kelas) dan **400 video Kaggle** (50/kelas).
Frozen split v3: 240 train + 60 validation internal per kelas; Kaggle dialokasikan sebagai external test.
Folder sumber `train/` memuat video internal; pembagian train/validation model mengikuti frozen manifest.

Classification pilot mengambil 60 train + 20 validation per kelas, maksimal satu frame per video.
Model A: 360 train + 120 validation. Model B: 480 train + 160 validation.
Gambar enam kelas bersama identik pada kedua model. Kaggle tidak digunakan untuk training, validation, atau tuning.

## Approach sejauh ini

### Pipeline Detection

Persiapan awal menggunakan frame, MediaPipe, ROI/bounding box, dan review anotasi.
v2 merupakan arsip persiapan sebelumnya; v3 menambahkan beberapa kandidat frame dan antrean review gesture.
Controlled Detection training pilot masih diblokir quality gates anotasi/metadata.
Report persiapan bukan bukti bahwa model Detection sudah selesai dilatih.

### Pilot manusia

Studi single-frame feasibility melibatkan 160 video dan dua reviewer: accuracy sekitar 80,6% dan 86,3%,
Cohen's Kappa sekitar 0,825. Hasilnya dibekukan. Jawaban reviewer tidak digunakan untuk memilih frame validation.

### Exploratory Classification pilot

Pipeline tambahan menggunakan **YOLO26 Classification**, checkpoint **`yolo26n-cls.pt` pretrained ImageNet**.
ImageNet adalah dataset pretraining; arsitektur tetap YOLO26n Classification.
Model A dan B di-fine-tune independen dari checkpoint awal dengan hash identik, bukan sequential fine-tuning.

Frame dipilih otomatis dari kandidat v3; label berasal dari metadata video. Classification tidak membutuhkan bounding box.
Konfigurasi: 30 epoch, seed 42, batch 16, input 224, SGD, deterministic, letterbox tanpa crop/flip/augmentasi linguistik.
Evaluasi menggunakan `last.pt` epoch 30 pada keduanya.

Corruption, decoding, mapping, frozen split, duplicate content, dan detectable leakage tetap hard gates.
Semantik frame yang belum lengkap serta metadata signer/sesi yang tidak tersedia dicatat sebagai warning.
Seluruh **497 flagged frames** tetap disertakan; sampel sulit tidak dibuang diam-diam.

## Struktur folder

```text
.
├── README.md
├── BISINDO_YOLO26_Classification_End_to_End.ipynb   # Run Classification + output nyata
├── BISINDO_YOLO26_Training_Set_Expansion.ipynb      # Notebook pipeline Detection
├── classification_pilot.py                        # Prepare, gates, train A/B, evaluasi Classification
├── training_pilot.py                             # Controlled Detection pilot + utilitas bersama
├── training_pilot_eval.py                        # Evaluasi/matching Detection
├── bisindo_pipeline.py                           # Inventory, split, persiapan dataset
├── frame_candidates.py                           # Kandidat frame dan pemeriksaan teknis
├── pilot_study.py                                # Feasibility study/review manusia
├── audit_dataset.py                              # Audit dataset
├── *_template.html                               # Template UI review
├── test_*.py                                     # Unit tests
├── *_README.md                                   # Panduan setiap tahap
├── BISINDO_PROJECT_HANDOFF_FOR_CLAUDE_2026-11-02.md # Konteks dan riwayat lengkap
├── experiment_yolo26/                            # Arsip persiapan awal
├── experiment_yolo26_v2/                         # Report persiapan v2
├── experiment_yolo26_v3/                         # Inventory, frozen split, kandidat, report v3
├── experiment_yolo26_pilot/                      # Manifest dan hasil pilot manusia
├── experiment_yolo26_training_pilot/             # Readiness/blocker Detection pilot
├── experiment_yolo26_classification_pilot/        # Run Classification pertama
├── experiment_yolo26_classification_notebook_20261102/ # Reproduksi melalui notebook
├── train/                                       # Video internal; lokal, di-ignore
├── test/                                        # Video Kaggle; lokal, di-ignore
└── .venv-bisindo/                                # Environment lokal, di-ignore
```

Struktur artefak Classification:

```text
experiment_yolo26_classification_notebook_20261102/
├── checkpoint.json                  # Identitas/hash pretrained checkpoint
├── manifest.csv                     # Pilihan video/frame dan split
├── release.json                     # Identitas release dataset
├── pilot_conclusion.md              # Kesimpulan dan keterbatasan
├── reports/                         # Readiness, config, environment, logs, provenance
├── runs/
│   ├── model_A/                      # results.csv, curves, args
│   ├── model_B/
│   └── trained_models.json           # Hash weights dan actual optimizer steps
├── evaluation/                      # Prediksi, accuracy/F1, confusion matrices, diagnostics
├── audit/                           # Sampel dan catatan audit
├── datasets/                        # Gambar training; lokal, di-ignore
├── selected_frames/                 # Frame peserta; lokal, di-ignore
└── checkpoints/                     # Weights; lokal, di-ignore
```

Path di laporan dapat menunjuk file lokal yang sengaja tidak diunggah. SHA256/provenance dipertahankan untuk
membandingkan dengan artefak asli; keberadaan hash bukan berarti file tersebut tersedia di GitHub.

## Hasil pilot Classification

Notebook dieksekusi pada GPU lokal. Run notebook mereproduksi hasil run pertama:

| Model | Internal validation | Benar | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| A | Enam kelas bersama | 112/120 | 93,33% | 0,9318 |
| B | Enam kelas bersama | 111/120 | 92,50% | 0,9300 |
| B | Seluruh delapan kelas | 150/160 | 93,75% | 0,9355 |

Prediksi tambahan `halo`/`tidur` dari B pada common6 tetap dihitung sebagai kesalahan, tidak dihapus.
Macro-F1 common6 dirata-ratakan pada enam kelas ground truth; confusion B mempertahankan kolom prediksi tambahan.
Actual optimizer steps: **A 690**, **B 900**. Epoch sama tidak berarti jumlah updates sama.

- `apa_kabar`: benar 20/20 pada A dan B; masih menerima false positives dari kelas lain.
- `halo`: B benar 19/20; satu sampel diprediksi `apa_kabar`.
- `marah`: A benar 15/20, B 14/20.

![Confusion Model A common6](experiment_yolo26_classification_notebook_20261102/evaluation/confusion_A_common6.png)
![Confusion Model B common6](experiment_yolo26_classification_notebook_20261102/evaluation/confusion_B_common6.png)
![Confusion Model B full8](experiment_yolo26_classification_notebook_20261102/evaluation/confusion_B_full8.png)

Selisih common6 hanya satu gambar; satu seed tidak membuktikan keunggulan model.
Full8 dan common6 adalah tugas berbeda, bukan perbandingan langsung untuk membuktikan B lebih baik.

## Menjalankan lokal

1. Clone repo ini.
2. Pulihkan video lokal dan cache kandidat v3 sesuai frozen manifest asli. Gambar, `candidate_records/`, dan weights
   tidak didistribusikan di GitHub; clone saja belum cukup untuk meregenerasi training.
3. Siapkan environment Python/CUDA yang kompatibel dengan dependency versions pada laporan.
   Run tercatat memakai Python 3.11.5, Ultralytics 8.4.197, torch 2.14.0+cu126, torchvision 0.29.0+cu126.
4. Daftarkan kernel untuk environment yang dipakai:

   ```powershell
   python -m ipykernel install --user --name thesis_cuthbert --display-name thesis_cuthbert
   ```

5. Buka [notebook Classification](BISINDO_YOLO26_Classification_End_to_End.ipynb), pilih `thesis_cuthbert`.
6. Ganti `WORK_NAME` ke folder baru, lalu **Run All** dan simpan output.

Notebook menjalankan prepare → technical gates → GPU/API probe → training A/B → evaluasi.
Folder completed dibaca tanpa retraining; partial run tidak ditimpa. Gate failure menghentikan training.
Checkpoint diunduh dari sumber publik yang dipin dan diverifikasi hash; tidak memerlukan paid inference API.
Notebook memakai modul `.py` dalam repo, bukan file standalone tanpa dependensi/data.
Hardware/dependensi berbeda dapat mengubah hasil meskipun seed sama.

Unit tests dari root project:

```powershell
python -m unittest discover
```

## Batasan dan langkah berikutnya

- Label berasal dari metadata video; semantik seluruh frame belum disahkan ahli BISINDO.
- Signer/session independence belum diverifikasi; hasil tidak diklaim signer-independent.
- Background, pakaian, kualitas frame, dan pose ambigu dapat memengaruhi hasil.
- Log notebook mendukung traceability, bukan bukti tunggal validitas ilmiah.
- Belum ada trained benchmark Detection vs Classification pada release sepadan.
- Belum ada full-data multi-seed training atau external Kaggle evaluation untuk hasil Classification ini.
- Kesalahan belum membuktikan informasi temporal sebagai penyebab; perlu eksperimen pembanding.

Classification saat ini baseline exploratory. Pipeline Detection dan metodologi thesis tidak diganti otomatis.
Tahap final memerlukan audit data, keputusan metodologi/config dibekukan, beberapa seed, dan external testing.

## Dokumentasi lanjutan

- [Classification pilot](CLASSIFICATION_PILOT_README.md)
- [Controlled Detection pilot](TRAINING_PILOT_README.md)
- [Frame selection v3](FRAME_SELECTION_README.md)
- [Pilot manusia](PILOT_STUDY_README.md)
- [Arsip persiapan v2](BISINDO_PIPELINE_README.md)
- [Konteks lengkap dan seluruh eksperimen](BISINDO_PROJECT_HANDOFF_FOR_CLAUDE_2026-11-02.md)
