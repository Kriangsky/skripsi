# Controlled YOLO26 training pilot

Implementasi lokal: `training_pilot.py`, `training_pilot_eval.py`, `test_training_pilot.py`.
Output terpisah: `experiment_yolo26_training_pilot/`.
Tidak memakai API berbayar, HUB, cloud training, akun, atau sequential fine-tuning.
Tidak mengubah notebook, dataset asli, split, anotasi v2/v3, atau pilot manusia.

## Status saat implementasi

- Antrean awal: **640 video internal**, 60 train + 20 validation per kelas; seed 42.
- Positive terverifikasi: **0**; reviewer belum menganotasi dataset training ini.
- Metadata signer/sesi: **belum tersedia**. Tidak dibuat identitas palsu.
- CUDA tersedia: NVIDIA GeForce RTX 4070 Laptop GPU.
- Ultralytics 8.4.197 / torch 2.14.0+cu126 mendukung checkpoint lokal YOLO26n dan SGD, diverifikasi melalui API.
- Dry forward tensor sintetis 1×3×64×64 menghasilkan output end-to-end 1×84×6. Tidak memakai gambar dataset atau training.
- Dataset release, visual batch audit, training dan evaluasi nyata **belum dapat dijalankan**.

Hasil feasibility manusia sebelumnya tidak sama dengan positive training beranotasi: jawaban kelas saja tidak memverifikasi bounding box/tangan relevan.

## Audit dan keputusan

Reuse `bisindo_pipeline.CLASSES/BASE`, hash, safe path, konversi box YOLO, pencatatan environment;
`frame_candidates.catalog` dan editor canvas/union tangan dari `frame_review_template.html`;
`pilot_study.inspect_candidates` untuk memvalidasi cache, hash dan kualitas teknis.

Tidak memanggil `bisindo_pipeline.prepare/validate_dataset/build_datasets/train_independent`: fungsi tersebut menulis ke output v3,
memerlukan review seluruh dataset termasuk external, menggunakan 100 epoch, atau mengizinkan CPU fallback.
Flow baru terpisah: tes → audit source → antrean review → validasi impor → gate metadata/positive → release beku → audit batch manusia → training independen → evaluasi.

Checkpoint existing dipakai read-only dari cache v2 dengan revision dan SHA256 terkunci.
Tidak ada download otomatis. Ultralytics settings/logging eksternal dinonaktifkan dan settings disimpan di folder pilot baru.
Checkpoint lama memiliki cabang one-to-one dan YAML end2end tetapi flag runtime baru belum aktif; pipeline mengaktifkan end2end secara eksplisit dan memverifikasinya untuk kedua model.

## 1. Siapkan / perbarui laporan

Dari folder project, gunakan environment existing:

```powershell
.\.venv-bisindo\Scripts\python.exe training_pilot.py prepare
```

`prepare` tidak menjalankan training. Source snapshot dicek read-only. Antrean train/val ditarik dari split beku.
Semua ID dan hash video pilot manusia dilarang masuk validation, terlepas dari apakah pernah dipakai untuk pengembangan.
Jawaban reviewer feasibility tidak dibaca untuk memilih frame.

Seluruh 2.400 video internal tetap tercatat di `reports/internal_inventory.csv`.
`excluded_samples.csv` membedakan pending review, penolakan eksplisit, development exclusion dan positive berlebih yang tidak terpilih.
Pending bukan penolakan. Rejection rate memakai denominator video yang sudah mendapat keputusan; kosong jika belum ada review.
Tidak ada oversampling. Sesudah penolakan, `prepare` memasukkan cadangan berikutnya dari urutan seed yang sama sampai target antrean terisi atau pool habis.

## 2. Metadata sungguhan

Isi `participant_metadata.csv` di folder output pilot, dengan kolom:

```text
video,signer,session
```

Salin daftar video dari `reports/metadata_template.csv`, lalu isi identitas signer dan sesi perekaman yang benar.
Template awal tidak ditimpa oleh prepare berikutnya; video cadangan baru dapat dilihat di `review_queue.csv`.
Jangan menganggap nomor video sebagai signer. Session ID harus mengidentifikasi sesi perekaman secara global.
Alternatif: `signer_metadata.csv` pada root jika sudah memiliki ketiga kolom tersebut.
Signer/sesi kosong, duplikat metadata, konflik signer dengan split lock, atau overlap train/validation memblokir training.
Tidak ada perpindahan video antar-split untuk memperbaiki overlap. Jika subset independen tidak tersedia, eksperimen tetap blocked.

## 3. Review gesture dan box

Buka `experiment_yolo26_training_pilot/review/frame_review.html` di Chrome/Edge lokal.
Ini UI **class-aware annotation**, terpisah dari UI blind feasibility. Kelas sumber diperlukan untuk verifikasi anotasi.

1. Periksa contact sheet dan kandidat; video sumber internal boleh dibuka untuk konteks.
2. Rekomendasi teknis menggunakan aturan sama untuk semua kelas: kandidat eligible dengan fokus terbesar; hanya proposal.
3. Pilih frame yang benar-benar menunjukkan gesture target. Editor existing memungkinkan memilih tangan relevan dan drag/edit box gabungan.
4. Centang verifikasi gesture, seluruh tangan yang diperlukan terlihat utuh/tercakup, dan box diperiksa pada frame tersebut.
5. Flag teknis butuh justifikasi manusia. Tangan yang memang diperlukan tetapi terpotong harus ditolak, bukan diloloskan lewat override.
6. Preparation/resting/transition, ambiguous, clipped, blur, wrong gesture dan temporal_ambiguity tersedia sebagai alasan.
7. Tolak kandidat individual untuk merekam masalah kandidat. Untuk menolak seluruh video, pilih alasan lalu klik **Tidak ada yang layak — needs_review** dan isi catatan.
8. Jika satu frame tidak merepresentasikan frasa, gunakan `temporal_ambiguity`; jangan memaksakan positive.
9. Isi reviewer lalu ekspor JSON. Draft hanya cadangan. Bulk approval dan pencarian frame baru dinonaktifkan.

Impor lewat:

```powershell
.\.venv-bisindo\Scripts\python.exe training_pilot.py import-review --file "C:\path\training_pilot_review.json"
```

Impor memvalidasi semua ID antrean, class ID, frame/video hash, cache MediaPipe, box, tiga attestation manusia, reviewer, timestamp, dan penolakan kandidat.
Tidak ada perubahan v3. Original JSON diarsipkan pada `review/imports/`; review store terlindungi digest.
Antrean berikutnya otomatis diperbarui. Gunakan HTML terbaru; draft/export stale ditolak.
Kandidat individual yang ditolak tidak otomatis berarti seluruh video ditolak; video tersebut tetap pending sampai keputusan eksplisit.

UI menautkan file kandidat/video/contact sheet v3 lokal; tetap simpan folder project pada lokasi yang sama.
Interaksi browser nyata belum diverifikasi dalam sesi implementasi. Pemeriksaan otomatis mencakup syntax/struktur UI dan validasi importer.

## 4. Release dan audit visual batch

Jika target 640 positive dan metadata independen sudah valid, prepare membangun:

- A: 360 train + 120 val, names/ID 0–5.
- B: 480 train + 160 val, names/ID 0–7.
- Shared-six images dan labels byte-identical; frozen split sama; satu video satu positive.
- YAML A/B dan B/common6 dengan vocabulary B tetap delapan kelas.
- Contact sheets berbox: `reports/contact_sheets/` (24 train + 8 val).

Release mengunci manifest, metadata, file dataset/YAML, dan contact sheets dengan hash. Tidak boleh diedit/ditimpa; gunakan work directory baru untuk eksperimen baru.
Manusia harus memeriksa seluruh batch contact sheets, lalu menjalankan:

```powershell
.\.venv-bisindo\Scripts\python.exe training_pilot.py approve-audit --reviewer "Nama reviewer" --confirmed
```

Jangan menjalankan approval sebelum audit sebenarnya. Gate selanjutnya memeriksa fingerprint approval dan sheet hashes.

## 5. Training pilot lokal

```powershell
.\.venv-bisindo\Scripts\python.exe training_pilot.py train
```

Perintah menjalankan prepare/gates ulang, lalu dua constructor YOLO dari **path/hash COCO yang sama**, bukan weights A → B.
Run existing, perubahan source/release, GPU hilang, perubahan optimizer/batch, atau OOM yang mencoba mengurangi batch dihentikan.
Tidak ada CPU fallback atau resume otomatis.

Config lengkap: `reports/experiment_config.json`. Parameter utama sama:
30 epoch, 640 px, batch 16, seed 42, deterministic, SGD, lr0=.01, lrf=.01, momentum=.937,
weight_decay=.0005, warmup=3, linear LR, patience=0, nbs=16, AMP=False.
Semua transformasi augmentation yang bisa mengubah gesture dinonaktifkan, termasuk flip/rotation/crop-scale/translate/mosaic/mixup/cutmix/color jitter.
Resize/letterbox preprocessing tetap diperlukan.

Actual optimizer steps dicatat setelah pemanggilan SGD, bukan diestimasi dari jumlah gambar.
`runs/model_A/optimizer_steps.json`, `runs/model_B/optimizer_steps.json` dan execution JSON merekam steps, config aktual, initial hash, dan release fingerprint.
Jumlah steps memang dapat berbeda karena ukuran training A/B berbeda. Logs/results/weights standar Ultralytics disimpan pada masing-masing run.

## 6. Evaluasi

Setelah dua run lengkap, training otomatis memanggil evaluasi. Untuk run lengkap yang belum dievaluasi:

```powershell
.\.venv-bisindo\Scripts\python.exe training_pilot.py evaluate
```

Evaluasi existing tidak ditimpa. Tidak ada evaluation pada Kaggle.

- A/common6 dan B/common6: **120 gambar yang sama**, B tetap delapan kelas tanpa `classes` filter.
- B/full8: 160 gambar, dilaporkan sebagai tugas terpisah.
- Official AP: conf=.001, IoU matching AP standar Ultralytics. Precision/recall curve diberi nama berbeda dari metrik fixed threshold.
- Fixed precision/recall/F1/FP/FN: confidence=.25, matching IoU=.50. Prediksi kelas tambahan B pada common6 tetap dihitung sebagai FP; jika menggantikan gesture benar, GT juga FN.
- Satu GT per image. Matching benar same-class IoU tertinggi; duplikat/unmatched adalah FP. Jika tidak ada correct match, wrong-class spatial match dicatat sebagai FP+FN off-diagonal. Axis matriks: true rows, predicted columns, background di akhir.
- AP common6 menghitung kelas yang memiliki ground truth; kelas tambahan tanpa GT tidak selalu menurunkan AP. Karena itu fixed-threshold dan confusion analysis wajib ikut dibaca.
- `apa_kabar_halo_diagnostics.csv` memuat seluruh sampel relevan, correct/incorrect, semua prediksi, confidence, IoU, FN, FP, flags anotasi dan visual kesalahan.
- `pilot_conclusion.md` menjawab hasil yang tersedia, bukan mengarang angka saat training blocked.

Satu seed/30 epoch bukan bukti keunggulan model. Pilot tidak membuktikan penyebab temporal tanpa comparator temporal.
Sesudah hasil pilot ditinjau, langkah lanjutan: anotasi full data, konfigurasi beku, beberapa seed, lalu external testing Kaggle.

## Tes

```powershell
.\.venv-bisindo\Scripts\python.exe -m unittest test_training_pilot test_pilot_study test_bisindo_pipeline test_frame_candidates -v
```

Tes release/import/training orchestration menggunakan fixture sintetis dan model tiruan di direktori temporer.
Tidak menjalankan gradient training pada dataset nyata atau membuat metrik eksperimen palsu.
