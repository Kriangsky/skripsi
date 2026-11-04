# Pilot single-frame BISINDO

Implementasi terpisah: `pilot_study.py`, `test_pilot_study.py`.
Tidak mengubah notebook, dataset, `bisindo_pipeline.py`, `frame_candidates.py`, atau hasil v2/v3.
Tidak menjalankan training atau full-dataset review.

## Audit dan keputusan implementasi

- `bisindo_pipeline.CLASSES`, `sha`, `safe_path`, `save_json` dipakai kembali.
- `frame_candidates.assess_hands` dipakai untuk memeriksa ulang JPEG kandidat dengan landmark cache v3; MediaPipe tidak dijalankan ulang.
- `freeze_inventory()` / `prepare()` existing tidak dipanggil: fungsi tersebut menulis ulang inventaris/anotasi v3.
- Input read-only: `split_lock.csv`, `candidate_config.json`, `candidates.csv`, cache JSON dan JPEG v3; hash video terpilih diverifikasi terhadap split beku.
- Output dan caller baru: CLI terpisah, dua HTML mandiri, importer, analyzer, tes. Tidak ada perubahan API existing.
- Urutan implementasi: tes sampling/leakage → seleksi/provenance → UI/import → metrics/report → tes integrasi dan regresi.

## Protokol beku

20 video per kelas, seed 42, source=train dan split=train saja. Urutan input CSV tidak mempengaruhi sampling.
Hash/video/ID duplikat pada inventaris ditolak. Tidak ada confidence atau prediksi YOLO.
Jika signer tersedia, sampling round-robin kelompok signer teracak untuk memperluas keragaman.
Metadata signer parsial dilaporkan; data tanpa signer menjadi satu residual pool, tidak dibuat identitas baru.
`signer_metadata.csv` opsional, kolom `video,signer`; konflik dengan metadata beku ditolak.
Tanpa signer, semua video dalam kelas disampling seragam tanpa pengembalian; video berbeda tidak berarti signer berbeda.

Seleksi sama untuk semua kelas: frame terbaca → tanpa flag → jumlah flag terkecil → fokus ROI tertinggi → indeks frame terkecil → candidate ID.
Threshold tetap dari v3, tidak dikalibrasi memakai hasil review atau test eksternal.
Jika tidak ada kandidat lolos teknis, satu kandidat flagged tetap ditampilkan. Jika tidak ada gambar terbaca/cache, video tetap ada dengan status missing.
Corrupt image/hash kandidat dicatat sebagai unreadable; ketidakcocokan sumber/cache/katalog menghentikan persiapan.
Ini ukuran kualitas teknis, bukan bukti gesture benar, lengkap, atau semantik dapat dibedakan.

Gambar dipindahkan sebagai PNG lossless dari hasil decode JPEG kandidat: piksel sama, metadata sumber dibuang.
ID publik token acak kriptografis; tidak diturunkan dari kelas, hash video, nama file, atau seed publik.
Seed 42 mereproduksi daftar video. Token baru berbeda pada persiapan baru; manifest/HTML yang sudah dibuat dibekukan.
Daftar publik diurutkan berdasarkan token acak sebelum permutasi, sehingga seed publik tidak dapat mengembalikan urutan blok kelas.
Urutan HTML reviewer diacak terpisah dengan seed 43 dan 44, lalu tetap selama studi.
Impor memvalidasi hash manifest, laporan seleksi, gambar, kedua HTML, study ID, reviewer ID, seluruh ID sampel, kategori, alasan, dan timestamp dengan timezone.
Respons duplikat, parsial, stale, atau impor ulang ditolak. Missing frame wajib invalid.
Tidak ada fungsi mengganti frame dari hasil reviewer.

## Persiapan

Environment existing yang diverifikasi: `.venv-bisindo`, Python 3.11; executable sama dengan catatan environment v3.
Notebook/kernel `thesis_cuthbert` tidak diubah.

```powershell
.\.venv-bisindo\Scripts\python.exe pilot_study.py prepare
```

Folder output existing ditolak untuk melindungi studi beku. Jangan menghapus/mengulang pilot setelah reviewer mulai.
Dataset asli, v2/v3, dan split tetap utuh. Persiapan hanya membuka video terpilih untuk hash; tidak mengekstrak frame baru.

## Dua reviewer

1. Berikan **hanya** `experiment_yolo26_pilot/review/reviewer_1.html` kepada reviewer pertama dan `reviewer_2.html` kepada reviewer kedua.
2. File HTML sudah memuat gambar; bisa disalin sendiri dan dibuka offline di Chrome/Edge. Jangan berikan folder pilot lengkap, manifest privat, laporan seleksi, atau jawaban reviewer lain.
3. Gunakan dua orang yang memahami BISINDO, idealnya perangkat/profil browser terpisah. Jangan berdiskusi tentang jawaban selama review.
4. Setiap orang memilih salah satu dari 8 kelas, ambiguous, atau invalid; alasan boleh lebih dari satu. Tidak boleh melihat video sumber atau mengganti frame.
5. Browser menyimpan draft jika storage tersedia. Ekspor draft sebagai cadangan; impor draft hanya menerima identitas studi/reviewer yang cocok.
6. Setelah seluruh 160 sampel dijawab, klik **Ekspor final**. Serahkan JSON kepada koordinator.

Independensi reviewer bersifat prosedural; HTML offline tidak mengautentikasi identitas manusia.
Setiap HTML sekitar 106 MB karena memuat semua gambar PNG; pembukaan awal dapat memerlukan waktu.
Local storage hanyalah bantuan. Ekspor JSON adalah salinan portabel.
UI tidak menampilkan flag teknis, kelas sumber, rekomendasi, atau jawaban benar, termasuk sesudah selesai.
Kemungkinan cue visual dalam gambar itu sendiri tetap menjadi keterbatasan.

## Impor dan analisis

Ganti path JSON berikut dengan lokasi hasil reviewer. Jangan impor respons sintetis ke studi asli.

```powershell
.\.venv-bisindo\Scripts\python.exe pilot_study.py import --reviewer reviewer_1 --file "C:\path\reviewer_1_final.json"
.\.venv-bisindo\Scripts\python.exe pilot_study.py import --reviewer reviewer_2 --file "C:\path\reviewer_2_final.json"
.\.venv-bisindo\Scripts\python.exe pilot_study.py analyze
```

Sebelum dua respons lengkap tersedia, output analysis berstatus **PENDING**, tanpa accuracy atau confusion PNG palsu.
Sesudah impor lengkap, analyzer menghasilkan `pilot_summary.csv`, `per_class_results.csv`, dua matriks CSV,
`confusion_matrix.png`, `inter_reviewer_agreement.json`, dan `pilot_report.md`.

Accuracy_all selalu memakai seluruh 160 sampel. Ambiguous/invalid salah; missing tidak dibuang.
Accuracy_identifiable memakai subset yang oleh reviewer diberi salah satu label kelas; jumlah/coverage selalu dilaporkan.
Technical coverage = eligible/160; inspectable coverage = (eligible+flagged)/160.
Kappa memakai 10 kategori nominal: 8 kelas + ambiguous + invalid, tanpa menggabungkan abstention.
Kappa undefined ketika expected agreement=1. Persentase preparation/transition/reasons memakai denominator semua sampel; alasan bisa overlap.
"Sering tertukar" berarti >=2 kesalahan pooled pada pasangan kelas, bukan threshold signifikansi.
Wilson CI 95% tersedia untuk accuracy semua/subset dan accuracy semua per kelas.
Asumsi independensi video dapat gagal karena signer berulang; interval bukan CI cluster signer.

Report membedakan missing, flagged, salah kelas pada gambar terbaca, dan ambiguous/invalid pada gambar terbaca.
Alasan temporal reviewer tidak membuktikan kebutuhan temporal secara kausal; studi ini tidak punya pembanding video terkontrol.
Tidak ada keputusan final arsitektur, klaim publikasi Model A, atau peralihan LSTM/TCN otomatis.

## Verifikasi

```powershell
.\.venv-bisindo\Scripts\python.exe -m unittest test_pilot_study test_bisindo_pipeline test_frame_candidates -v
```

Tes integrasi memakai fixture sintetis di direktori temporer, termasuk analisis dengan jawaban sintetis; tidak menyentuh respons pilot asli.
Tes Node memeriksa syntax dan interaksi JS memakai DOM stub; bukan pengganti verifikasi visual di browser nyata.
Pemeriksaan UI langsung terhalang Computer Use: URL Chrome tidak dapat dipastikan oleh kontrol kebijakan.
Reviewer/koordinator masih perlu memeriksa tampilan dan unduhan file sebelum mulai evaluasi sebenarnya.

## Hasil persiapan lokal

160 sampel (20 per kelas): 43 eligible, 117 flagged, 0 missing. Technical coverage 26,875%; inspectable coverage 100%.
Signer tersedia: 0/160; keragaman atau independensi signer belum dapat dibuktikan.
26 tes lulus, 0 gagal: 12 tes pilot (termasuk JS dan integrasi sintetis) dan 14 tes regresi existing.
Audit output nyata: seluruh sampel cocok dengan split/hash sumber, 160/160 PNG memiliki piksel identik dengan JPEG kandidat,
dan payload kedua HTML hanya mengandung identitas studi/reviewer, ID sampel acak, serta gambar.
Belum ada respons manusia yang diimpor. Studi masih menunggu dua reviewer.
