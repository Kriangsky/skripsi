# Frame selection BISINDO — pipeline aktif v3

Notebook: `BISINDO_YOLO26_Training_Set_Expansion.ipynb`, kernel `thesis_cuthbert`.
Implementasi: `frame_candidates.py`, dipanggil oleh `bisindo_pipeline.prepare()`.
Hasil: `experiment_yolo26_v3/`. Hasil v2 dan sumber video tetap dipertahankan.

## Hasil pemrosesan lokal

| Pemeriksaan | Hasil |
|---|---:|
| Video | 2.800 |
| Kandidat JPEG | 33.250 (7–13 per video) |
| Contact sheet | 2.800 |
| Kandidat lolos heuristik teknis | 1.120 |
| Video dengan rekomendasi teknis | 546 |
| Positive frame disetujui otomatis | 0 |
| Video menunggu review gesture | 2.800 |

Semua path sumber, hash video, class mapping dan split dibandingkan dengan inventory v2 dan identik.
Empat belas tes, sintaks JavaScript, serta alur notebook default lolos. Training/evaluasi tidak dijalankan.
Interaksi UI browser belum diuji langsung. Banyak kandidat memiliki flag dengan threshold awal ini;
angka tersebut bukan ukuran akurasi anotasi atau bukti bahwa sisa video tidak dapat digunakan.

## Kontrak eksperimen

- Class mapping tetap: saya, kamu, senang, bingung, marah, apa_kabar, halo, tidur.
- Split dibekukan dari `experiment_yolo26_v2/inventory.csv` ke `split_lock.csv`.
- Kandidat tidak menambah sampel training; satu video hanya boleh menghasilkan satu positive frame.
- Deteksi tangan tidak membuktikan gesture sesuai kelas. Nama video hanya memberi konteks kelas yang harus diverifikasi.
- Tidak ada label YOLO positif sampai reviewer memilih frame dan memverifikasi gesture.
- Video tanpa pilihan layak tetap tercatat sebagai `needs_review`; tidak dihapus dan tidak dipaksa berlabel.
- Training A/B independen dan gate QC tetap berlaku. Semua video perlu keputusan sebelum dataset utama dirilis.

## Kandidat dan pemeriksaan teknis

Tujuh titik: 20%,30%,40%,50%,60%,70%,80% dari `(frame_count - 1)`, pembulatan ke indeks integer unik.
Ekstraksi dilakukan dengan decode berurutan. MediaPipe Hand Landmarker IMAGE mode, maksimal dua tangan.
Jika semua kandidat utama gagal cek teknis, coba 5%,10%,15%,85%,90%,95%.
Permintaan manual “Cari frame lain” juga memproses rentang tambahan tersebut.

Kotak dibangun dari seluruh landmark tangan yang terdeteksi, dengan padding 15% per sisi.
Flag teknis:

- landmark tidak finite atau bukan 21 titik per tangan;
- landmark berada dalam 1% dari tepi gambar / di luar gambar;
- sisi tangan di bawah 16 px atau area di bawah 256 px;
- ROI tangan kosong atau box invalid;
- kemungkinan blur: Laplacian variance pada ROI tangan yang di-resize 128×128 di bawah 60.

Threshold fokus adalah heuristik awal, **belum dikalibrasi sebagai standar kualitas**.
Flag tidak membuktikan blur/occlusion; tidak adanya flag juga tidak membuktikan seluruh tangan utuh.
Lihat kandidat ukuran penuh dan konteks video. Satu tangan terdeteksi belum membuktikan
bahwa tangan kedua yang relevan juga terdeteksi. Reviewer boleh memilih subset tangan relevan
atau mengoreksi kotak, tetapi tidak boleh “memperbaiki” bagian tangan yang sebenarnya di luar frame.

Rekomendasi = kandidat lolos teknis dengan minimum fokus ROI tangan terbesar.
Rekomendasi tidak otomatis menjadi frame terpilih dan bukan klasifikasi preparation/transition/gesture.
Tidak ada pemodelan temporal atau classifier pose BISINDO baru dalam revisi ini.

## Review cepat

1. Buka `experiment_yolo26_v3/frame_review.html` di Chrome/Edge.
2. Scan contact sheet seluruh video. Filter kelas, internal/test, needs_review, atau tanpa kandidat lolos teknis.
3. Klik video, bandingkan kandidat. Putar video sumber bila perlu untuk memahami gesture.
4. Pilih kandidat yang sesuai kelas, bukan preparation/resting/transition. Pilih tangan relevan,
   atau drag satu bounding box gabungan. Centang verifikasi gesture, lalu simpan pilihan.
5. Untuk kandidat salah: pilih alasan (preparation, resting, transition, motion blur, clipped, wrong gesture, lainnya), lalu tolak.
6. Jika belum ada yang layak: tulis alasan dan minta frame lain. Ekspor dan impor JSON untuk menjalankan pencarian tambahan.
7. Jika tetap tidak ada: tandai `needs_review`. Jangan memilih frame jelek agar jumlah sampel terpenuhi.
8. Isi nama reviewer, unduh `bisindo_frame_review.json`, lalu impor lewat bagian 4 notebook.

Pilihan massal hanya atas konfirmasi eksplisit manusia setelah memeriksa gesture seluruh rekomendasi;
tidak berlaku otomatis pada kandidat yang ditolak, yang meminta frame tambahan, atau tanpa rekomendasi valid.
Untuk review bertahap, ekspor draft sebagai cadangan. Browser storage hanya bantuan, bukan satu-satunya salinan.

Setelah impor, buka ulang HTML baru karena manifest sudah berubah. JSON lama ditolak sebagai stale.
Kandidat dengan flag teknis memerlukan alasan override manusia bila tetap dipilih.
Keputusan ini bukan izin meloloskan preparation pose atau tangan yang benar-benar tidak terlihat.

## File hasil

| File/folder | Isi |
|---|---|
| `split_lock.csv` | Mapping dan split tetap |
| `candidates.csv` | Semua kandidat, indeks, JPEG/hash, landmark, fokus, status teknis, flag |
| `candidate_records/*.json` | Cache per video yang dapat dilanjutkan |
| `candidate_frames/*.jpg` | JPEG kandidat yang benar-benar dianalisis MediaPipe |
| `contact_sheets/*.jpg` | Kandidat dan box per video, indeks dan alasan flag |
| `frame_review.html` | Review semua video + editor pilihan/frame/tangan |
| `annotations.csv` | Manifest satu baris per video, termasuk keputusan frame dan penolakan |
| `needs_review.csv` | Video yang belum memiliki positive frame valid |
| `flagged_candidates.csv` | Alasan flag teknis per kandidat |
| `frames/*.jpg` | Hanya positive frame yang telah dipilih manusia |
| `candidate_labels/*.txt` | Satu baris YOLO per positive frame terpilih |
| `annotation_summary.json` | Jumlah kandidat, flag, pilihan, needs_review |
| `annotations_before_selection_*.csv` | Backup manifest sebelum impor review |

Kolom manifest penting: `frame_index`, `selected_candidate_id`, `frame_review_status`,
`review_status`, `rejection_reason`, `rejected_candidates` (JSON alasan per kandidat),
`quality_override_reason`, reviewer, hash gambar, serta digest keputusan frame dan box.
Frame belum dipilih memakai `frame_index=-1`, tanpa positive image/label.

## Gate dataset

`validate_dataset()` memeriksa semua gate sebelumnya, ditambah bahwa:

- frame dipilih dan gesture diverifikasi manusia;
- kandidat milik video yang tepat, indeks/hash sesuai, gambar tidak berubah;
- perubahan pilihan setelah review tidak diterima tanpa review baru;
- flag teknis punya justifikasi manusia bila di-override;
- tidak ada video yang hilang atau pindah split/class selama pemilihan;
- label sesuai JPEG yang dipilih, bukan frame lama dari cache midpoint.

Review partial diperbolehkan; release/training tetap tertahan selama ada `needs_review`.
Tidak dilakukan training atau benchmark model BISINDO dalam tahap pemilihan frame.

## Jalankan

```powershell
.\.venv-bisindo\Scripts\python.exe frame_candidates.py
.\.venv-bisindo\Scripts\python.exe -m unittest test_bisindo_pipeline test_frame_candidates -v
```

```python
import frame_candidates as frames
frames.import_review(ROOT, WORK, Path.home() / 'Downloads' / 'bisindo_frame_review.json')
pipeline.validate_dataset(ROOT, WORK)
```

Jika seluruh 13 titik masih melewatkan gesture, video tetap needs_review. Pemilihan timestamp
arbitrer atau sampling yang lebih rapat belum tersedia di UI; jangan memaksakan salah satu kandidat.
