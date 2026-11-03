# Arsip pipeline BISINDO YOLO26 v2

Pipeline aktif sekarang memakai beberapa kandidat dan review gesture di
`experiment_yolo26_v3`. Lihat **FRAME_SELECTION_README.md** dan notebook terbaru.
Angka dan petunjuk di bawah adalah riwayat v2 dengan satu frame tengah, bukan alur aktif.

Notebook: `BISINDO_YOLO26_Training_Set_Expansion.ipynb`.
Implementation: `bisindo_pipeline.py`. Pilih kernel `.venv-bisindo/Scripts/python.exe`.

## Hasil persiapan lokal

Seluruh 2.800 video sudah diproses: 2.400 internal dan 400 Kaggle.
Per kelas: 300 internal (240 train, 60 val) dan 50 external test.
Tidak ada metadata signer yang tersedia. Split saat ini per video, stratified seed 42.
Tambahkan `signer_metadata.csv` dengan kolom `video,signer` untuk seluruh video internal
jika metadata tersedia; persiapan berikutnya akan memakai split signer-disjoint global.

| Kelas | Terdeteksi / 350 | Tanpa deteksi | Kegagalan internal | Kegagalan test |
|---|---:|---:|---:|---:|
| saya | 343 | 7 | 7 | 0 |
| kamu | 319 | 31 | 31 | 0 |
| senang | 349 | 1 | 1 | 0 |
| bingung | 350 | 0 | 0 | 0 |
| marah | 348 | 2 | 2 | 0 |
| apa_kabar | 327 | 23 | 22 | 1 |
| halo | 349 | 1 | 1 | 0 |
| tidur | 335 | 15 | 15 | 0 |
| Total | 2.720 | 80 | 79 | 1 |

Ini hitungan keberhasilan menghasilkan kandidat, **bukan akurasi anotasi**.
Semua 2.800 sampel masih pending review. Contoh overlay memperlihatkan bahwa
status detected juga bisa menghasilkan kotak yang belum mencakup tangan kedua.
Jangan menyamakan detected dengan benar. Kegagalan tidak otomatis dibuang.

## Buka hasil dan lakukan QC

**Cara cepat:** buka `experiment_yolo26_v2/annotation_review.html` di browser.
Scan grid seluruh gambar, gunakan search/filter, lalu klik hanya gambar yang salah.
Drag kotak baru dan simpan koreksi. Setelah memeriksa seluruh gambar, isi reviewer,
centang pernyataan review, lalu **Approve All & unduh review JSON**. Approval berlaku
untuk seluruh dataset, bukan hanya filter aktif. Seluruh kotak harus valid dahulu;
80 kegagalan deteksi tidak otomatis dikeluarkan. Unduh draft untuk cadangan progres.

Impor sekali melalui sel quick review di notebook (default membaca
`~/Downloads/bisindo_review_approved.json`), atau:

```python
pipeline.import_bulk_review(WORK, Path.home() / 'Downloads' / 'bisindo_review_approved.json')
pipeline.validate_dataset(ROOT, WORK)
```

CSV lama dicadangkan sebagai `annotations_before_bulk_<hash>.csv`. Review dari
snapshot berbeda, class/frame mismatch, serta kotak invalid ditolak sebelum CSV
diubah. Approval hanya berasal dari ekspor eksplisit reviewer; membuat HTML atau
menjalankan persiapan tidak menyetujui kandidat secara otomatis.

- `experiment_yolo26_v2/annotation_gallery.html`: semua 2.800 overlay, filter kelas/status.
- `experiment_yolo26_v2/annotations.csv`: indeks frame, hash video/JPEG, kandidat, status review.
- `experiment_yolo26_v2/failed_annotations.csv`: 80 sampel tanpa deteksi.
- `experiment_yolo26_v2/annotation_counts.csv`: rincian per sumber, kelas, split, dan status.
- `experiment_yolo26_v2/preview_contact_sheet.jpg`: contoh internal dan test; bukan bukti seluruh kotak benar.
- `experiment_yolo26_v2/validation_report.json`: seluruh penghalang QC/training.

Sebagai alternatif, widget notebook menampilkan gambar, slider x1/y1/x2/y2 ternormalisasi,
nama reviewer, catatan, tombol simpan, dan penolakan. Semua gambar perlu diperiksa;
80 kegagalan memerlukan penentuan kotak, atau perbaikan pemilihan frame secara
terdokumentasi. Pipeline saat ini memakai aturan frame tengah yang konsisten.
Tidak ada auto-approval. Perubahan box/class/frame membatalkan binding review.
Panggil `pipeline.render_report(WORK)` untuk memperbarui galeri setelah review.

## Checkpoint dan reproducibility

Checkpoint resmi dari https://huggingface.co/Ultralytics/YOLO26:

- file: `yolo26n.pt`
- revision: `070ac3c51435984d992ba92eb3f234daf4e4500d`
- SHA256: `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef`
- A dan B masing-masing membuat objek YOLO baru dari file yang sama.

MediaPipe Hand Landmarker: IMAGE mode, maksimal dua tangan, confidence .5,
union seluruh landmark tangan, padding 15% per sisi, clip ke batas gambar.
Asset versi 1 dan checksum tercatat pada kode. Sumber resmi:
https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/python.

Versi Python, package, GPU, CUDA dan full freeze tersimpan di
`experiment_yolo26_v2/environment.json` dan `requirements-resolved.txt`.
Venv lokal mewarisi package sistem; lock mencatat package yang efektif.
Untuk mesin baru, gunakan Python 3.11 dan install build CUDA yang tercatat melalui
index resmi PyTorch sebelum memasang requirements-resolved. Full freeze juga memuat
package warisan yang tidak dipakai pipeline. Jangan mengganti dependency setelah
persiapan tanpa memvalidasi ulang; fungsi training memeriksa versi paket tercatat.

## Training dan evaluasi

`RUN_TRAINING=False`, `RUN_EVALUATION=False` secara default.
Fungsi training tetap memeriksa QC, sumber data, hash review, mapping, dan split.
Release A/B dibuat berdasarkan hash manifest; semua gambar dan label enam kelas
bersama diperiksa byte-identik. Tidak ada training yang dijalankan pada revisi ini.

Evaluasi utama: common6; tambahan: B full8. mAP50/mAP50-95 dan per-class
precision/recall disimpan. Fixed-threshold precision/recall/F1 mikro menghitung
seluruh output, termasuk salah prediksi kelas baru. Threshold ditetapkan sebelum
test; mAP tidak dihitung dengan memfilter output kelas tambahan.

Latency benchmark FP32, batch 1: tahap inference dan wall-time predict dipisahkan,
warmup 10, tiga pengulangan, mean/p50/p95/FPS. Tidak termasuk video decode,
MediaPipe, webcam, atau UI. Smoke test checkpoint COCO berhasil pada GPU;
angkanya bukan hasil model BISINDO dan bukan benchmark tesis.

## Pemeriksaan lokal

```powershell
.\.venv-bisindo\Scripts\python.exe -m unittest test_bisindo_pipeline -v
.\.venv-bisindo\Scripts\python.exe verify_notebook.py
```

Verifikasi kedua menjalankan alur default notebook dan menyimpan output laporan;
widget hanya didefinisikan dalam pemeriksaan headless. Jalankan sel widget di
notebook untuk membukanya. Training/evaluasi tetap OFF.
