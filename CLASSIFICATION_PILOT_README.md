# Exploratory YOLO26 Classification pilot

## Notebook end-to-end

`BISINDO_YOLO26_Classification_End_to_End.ipynb` menjalankan prepare, technical gates, environment probe,
dua training independen 30 epoch, dan evaluasi melalui modul project existing. Output sel berisi log eksekusi nyata;
run notebook memakai folder baru `experiment_yolo26_classification_notebook_20261102/`, terpisah dari hasil awal di bawah.
Pilih kernel `thesis_cuthbert`. Ganti `WORK_NAME` untuk eksperimen baru, kemudian Run All dan simpan output.
Folder completed dibaca tanpa retraining; partial run tidak ditimpa. Untuk GitHub, sertakan notebook dengan output,
modul `.py` dan tests; dataset/cache tetap prasyarat lokal. Notebook ini bukan paket standalone tanpa modul/data.

Implementasi tambahan: `classification_pilot.py`, `test_classification_pilot.py`.
Output: `experiment_yolo26_classification_pilot/`. Detection, notebook, v2/v3, frozen split, dan feasibility pilot manusia tidak diubah.

## Hasil nyata

Training selesai pada GPU lokal, **30 epoch per model**, seed 42. Evaluasi memakai **last.pt epoch 30** untuk keduanya.
Angka di bawah adalah kesesuaian terhadap **label metadata video**, bukan bukti semua frame menunjukkan gesture valid.

| Model | Validation | N | Benar | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|
| A, enam kelas | common6 | 120 | 112 | 93,33% | 0,9318 |
| B, delapan kelas | common6 | 120 | 111 | 92,50% | 0,9300 |
| B, delapan kelas | full8 | 160 | 150 | 93,75% | 0,9355 |

B salah memprediksi kelas tambahan pada 2 gambar common6; prediksi tersebut **tidak dihapus**.
Macro-F1 common6 dihitung pada enam kelas GT yang sama, dengan prediksi tambahan tetap menyebabkan FN pada kelas sebenarnya.
Confusion B/common6 tetap memiliki delapan kolom prediksi.

- `apa_kabar`: recall 20/20 pada A dan B; tetap terdapat false positives dari kelas lain.
- `halo`: B benar 19/20; satu `halo` diprediksi `apa_kabar`.
- `marah`: A benar 15/20 dan B 14/20; paling sulit pada pilot ini.
- Actual optimizer steps: A **690**, B **900**. Dataset lebih besar juga menghasilkan lebih banyak updates pada jumlah epoch sama.

Selisih common6 hanya satu gambar; satu seed bukan bukti keunggulan salah satu model.
Classification layak diteruskan sebagai baseline exploratory dengan audit lanjutan. Hasil ini tidak membuktikan classification mengungguli detection;
belum ada comparison detection yang dilatih pada release sepadan. Metodologi thesis tidak diganti otomatis.

## Data dan gate

Sample tetap: **60 train + 20 val per kelas**, dari frozen internal split v3; seed 42, tanpa replacement.
A memakai 360 train + 120 val; B 480 train + 160 val. Shared-six images byte-identical dan class ID 0–5 sama.
Folder numerik `00_saya` ... `07_tidur` mencegah ImageFolder mengubah ID lewat urutan alfabet.
Setiap video menghasilkan maksimal satu frame. Label berasal dari kelas video; **tidak ada syarat bounding box atau approval MediaPipe**.
Cache landmark lama hanya digunakan ulang untuk technical frame selection, tanpa menjalankan detektor atau menyimpulkan kebenaran gesture.

Policy otomatis dipakai ulang dari `pilot_study.choose_candidate/inspect_candidates`: readable, eligible dahulu, lebih sedikit flag,
fokus ROI lebih tinggi, frame index lebih kecil. Tidak memakai class label/prediksi model dalam pemilihan frame.
Semua **497 flagged** tetap disertakan; 143 eligible dan 0 gagal seleksi. Tidak ada difficult-frame exclusion atau penggantian frame dari audit.
Daftar 1.760 video internal yang tidak tersampling tersimpan dengan alasan kapasitas tetap, bukan kualitas.
Semua video feasibility manusia dilarang masuk validation berdasarkan ID/hash; jawaban reviewer lama tidak digunakan.
Kaggle tidak dimasukkan ke train/validation atau dipakai untuk tuning. Hanya inventory/cached hashes dipakai untuk pemeriksaan duplikasi.

Hard gates: mapping, frozen source/split consistency, valid image decoding, hash, file corruption, exact video/image duplicate content,
shared A/B consistency, dan leakage yang dapat dideteksi. Known signer overlap pada frozen metadata juga ditolak jika ada.
Signer/sesi yang tidak tersedia dan review semantik belum lengkap menjadi **warning**, bukan blocker.
Near-duplicate pose, shared room/outfit dan participant overlap tidak otomatis terdeteksi oleh pemeriksaan hash.

## Audit kualitas

Antrean audit: **15 acak per kelas** (10 train + 5 val), ditambah 5 flagged per kelas. Total 160 gambar.
Delapan contact sheets telah dilihat untuk pemeriksaan visual teknis. Hasil di:

- `audit/technical_visual_observations.csv`
- `audit/technical_visual_summary.json`

Tercatat tiga possible preparation/resting frames dan dua contoh pose overlap lintas kelas. Keduanya dugaan visual,
**belum adjudikasi semantik ahli BISINDO**. Tidak dibuat signer ID dari penampilan orang.
Pakaian/latar terlihat berulang; model dapat menggunakan cue konteks. Accuracy tinggi tidak membuktikan linguistic gesture recognition.
Known semantic ambiguity = 0 **yang sudah diadjudikasi**, bukan berarti tidak ada ambiguity. Semantik seluruh 640 frame masih belum diverifikasi ahli.

UI audit opsional: `audit/review.html`. Nilai gesture validity/ambiguity/preparation/transition/wrong label dan catatan, lalu ekspor JSON.
Impor tidak mengganti frame, mengubah label, atau mengecualikan sampel dari eksperimen beku:

```powershell
.\.venv-bisindo\Scripts\python.exe classification_pilot.py import-audit --file "C:\path\classification_quality_audit.json"
```

Metadata sampling audit tetap beku; hasil pengamatan dan respons audit disimpan terpisah.
UI JavaScript diperiksa sintaksnya; interaksi browser nyata belum diuji dalam sesi ini.

## Pretrained model dan kontrol

[Dokumentasi classification resmi](https://docs.ultralytics.com/tasks/classify) menyatakan YOLO26-cls memakai pretrained ImageNet.
Ini berbeda dari pretrained COCO detection yang digunakan pipeline sebelumnya.

- Checkpoint: [Ultralytics/YOLO26 yolo26n-cls.pt, revision 9da872c500a0a6fc9526cf09108ee249489a3895](https://huggingface.co/Ultralytics/YOLO26/blob/9da872c500a0a6fc9526cf09108ee249489a3895/yolo26n-cls.pt).
- SHA256 diverifikasi terhadap upstream LFS: `0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`.
- A dan B masing-masing load file pretrained identik; B tidak menggunakan weights A.
- Runtime lokal: Ultralytics 8.4.197, torch 2.14.0+cu126, RTX 4070 Laptop GPU. Tidak memakai API berbayar/cloud.
- Epoch 30, imgsz 224, nominal batch 16, seed 42, deterministic, SGD, lr0 .01, lrf .01, momentum .937,
  weight decay .0005, warmup 3, linear LR, patience 0, nbs 16, AMP False.
- Seluruh augmentation linguistik berisiko dinonaktifkan. Full frame di-letterbox dengan aspect ratio terjaga ke 224×224,
  padding 114. Train/val menggunakan Resize(square), ToTensor RGB/255, Normalize(0,1), **tanpa random crop atau center crop terhadap raw frame**.
- Evaluasi langsung pada prepared square images dengan preprocessing sama; argmax seluruh output 6/8 kelas, tanpa masking.
- Actual optimizer steps dicatat setelah SGD step. Terdapat 30 baris epoch 1–30 pada results.csv setiap model.
  Ultralytics mengirim final validation callback dengan counter sementara 31; field framework_callback_epoch menyimpannya,
  sedangkan epoch training terkonfirmasi 30 berdasarkan results.csv. Tidak ada epoch ke-31 yang dilatih.

## Jalankan

```powershell
.\.venv-bisindo\Scripts\python.exe classification_pilot.py prepare
.\.venv-bisindo\Scripts\python.exe classification_pilot.py probe
.\.venv-bisindo\Scripts\python.exe classification_pilot.py train
```

Train mengecek gate teknis, source/release hashes dan GPU; jika valid, menjalankan kedua model lalu evaluasi.
Pada folder sekarang training/evaluasi **sudah selesai**; perintah train/evaluate menolak overwrite atau resume.
Tidak ada fallback training CPU, automatic batch reduction, tuning test set, atau perubahan pipeline detection.
Untuk replikasi baru, gunakan `--work` folder baru dan salin `checkpoint.json` sebagai referensi absolut ke pretrained file read-only yang sama.
Pertahankan versi source/environment, cache v3, class mapping, seed, dan seluruh preprocessing.

## Output utama

- `manifest.csv`, `release.json`: video/frame/mapping/split/hash/flag dan provenance label lemah, beku.
- `reports/experiment_config.json`, `environment.json`, `source_snapshot.json`, `verification.json`.
- `reports/frame_quality_counts.csv`, `not_sampled_internal.csv`, `pilot_readiness_report.md`, `training.log`.
- `runs/model_A`, `runs/model_B`: last/best weights, results.csv, training plots, optimizer steps.
- `runs/trained_models.json`, `model_A_execution.json`, `model_B_execution.json`: actual parameters, initial and trained hashes.
- `evaluation/comparison_metrics.csv`, `per_class_metrics.csv`, `predictions.csv`.
- `evaluation/confusion_A_common6.png`, `confusion_B_common6.png`, `confusion_B_full8.png` dan CSV.
- `evaluation/apa_kabar_halo_diagnostics.csv`, `error_examples/`.
- `pilot_conclusion.md`: interpretasi dan batasan empiris.

Tidak ada bounding-box annotation, external-test performance, unknown/background rejection benchmark, atau real-time latency benchmark.
Sebelum hasil final: validasi semantik, metadata participant/sesi, kontrol cue konteks, beberapa seed, kemudian external test setelah konfigurasi dibekukan.

## Tes

```powershell
.\.venv-bisindo\Scripts\python.exe -m unittest test_classification_pilot test_training_pilot test_pilot_study test_bisindo_pipeline test_frame_candidates -v
```

46 tes lulus, 0 gagal, termasuk regresi detection dan feasibility pilot. Dataset asli dan hasil sebelumnya tetap dipertahankan.
