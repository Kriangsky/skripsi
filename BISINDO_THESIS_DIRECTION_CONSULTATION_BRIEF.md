# DOKUMEN ARAH PENELITIAN & PERSIAPAN DEFENSE KONSULTASI
## Impact of Vocabulary Expansion on Frame-Based BISINDO Recognition: A Controlled YOLO26 Study

**Versi:** Draft konsultasi v1 — 9 November 2026  
**Fungsi dokumen:** Memahami arah skripsi, menyampaikan alasan perubahan metodologi kepada dosen, dan menyiapkan paper konferensi yang dapat dipertanggungjawabkan.  
**Status penting:** **USULAN, BELUM PERSETUJUAN AKADEMIK / BELUM HASIL FINAL.** Dokumen ini bukan pengganti preregistration, bukan keputusan pembimbing, dan bukan laporan eksperimen eksternal.

> **Inti yang harus dapat dijelaskan dalam satu kalimat:** Penelitian ini menguji apakah pelatihan model klasifikasi BISINDO dengan vocabulary delapan kelas—dibandingkan enam kelas—mengubah kemampuan mengenali enam kelas awal pada frame yang sama, serta kelas mana yang mengalami manfaat atau interferensi.

---

## 0. Ringkasan satu halaman: apa yang berubah dan kenapa?

**Judul lama:** *Effect of Training Set Expansion on YOLOv26 Performance for Real-Time BISINDO Phrase Recognition*.

**Judul utama yang diajukan:** *Impact of Vocabulary Expansion on Frame-Based BISINDO Recognition: A Controlled YOLO26 Study*.

Perubahan ini **bukan mengganti domain atau mengulang skripsi dari awal**. Topik tetap BISINDO; keluarga model tetap YOLO26; desain perbandingan tetap enam versus delapan kelas. Yang diperbaiki adalah **kesesuaian antara pertanyaan penelitian, jenis model, data, metrik, dan klaim**:

1. **Dari “real-time” menjadi “frame-based”:** sistem mengambil satu frame representatif per video dan belum mengevaluasi pipeline streaming end-to-end. Karena itu klaim real-time atau pemahaman sekuens gestur tidak layak menjadi judul utama.
2. **Dari Object Detection ke Classification:** endpoint penelitian adalah label gestur, bukan koordinat bounding box. Untuk output satu label per gambar, `yolo26n-cls.pt` lebih langsung sesuai. Ini **perubahan metodologis substantif** dan perlu persetujuan dosen.
3. **Dari “training set expansion” yang luas menjadi “vocabulary expansion” yang operasional:** perbandingan A=6 kelas dan B=8 kelas menyoroti dampak pada enam kelas yang sama. Namun ekspansi kelas **juga menambah contoh training**, mengubah ukuran head dan jumlah langkah optimisasi; efek jumlah kelas saja **tidak terisolasi secara kausal**.
4. **Dari skor tunggal menjadi analisis terkontrol:** lima seed, evaluasi pada video yang dipasangkan, perubahan macro-F1 enam kelas awal, perubahan recall kelas per kelas, serta pola salah prediksi ke kelas baru.
5. **Dari sekadar performa internal menjadi evaluasi eksternal yang jujur:** Kaggle direncanakan sebagai sumber evaluasi eksternal, tetapi status *untouched confirmatory* masih **BLOCKED** sampai provenance dan riwayat penggunaannya dapat dipertanggungjawabkan.

**Keputusan untuk konsultasi:** minta persetujuan atas (a) task Classification, (b) judul/RQ, (c) arah H1 statistik, dan (d) cakupan final A/B lima seed sebelum training final.

---

## 1. Pilihan judul dan konsekuensinya

| Opsi | Judul | Kelebihan | Catatan | Rekomendasi |
|---|---|---|---|---|
| **A** | **Impact of Vocabulary Expansion on Frame-Based BISINDO Recognition: A Controlled YOLO26 Study** | Konsisten dengan arah sekarang; ringkas; mencakup efek ekspansi | Kata *controlled* harus dibatasi pada recipe, split, dan evaluasi yang diseragamkan—bukan efek kausal jumlah kelas murni | **Utama** |
| B | *Evaluating Vocabulary Expansion in Frame-Based BISINDO Classification: A Paired YOLO26 Study* | Lebih eksplisit mengenai Classification dan *paired comparison* | Sedikit lebih teknis, tetapi mengurangi ambiguitas “controlled” | **Alternatif akademis terkuat** |
| C | *Class-Level Effects of BISINDO Vocabulary Expansion: A Multi-Seed YOLO26 Classification Study* | Mengangkat kontribusi analisis per kelas dan beberapa seed | Tidak menyebut frame-based; perlu dinyatakan jelas di abstrak | Alternatif |
| D | *Effects of Vocabulary-and-Data Expansion on Frame-Based BISINDO Classification Using YOLO26* | Paling hati-hati mengenai confounding kelas dan ukuran data | Lebih panjang, kurang elegan | Jika dosen sangat menekankan kausalitas |

**Pilihan kerja:** gunakan **Opsi A** saat konsultasi; pertimbangkan **Opsi B** jika pembimbing mempertanyakan makna *controlled*. Hindari “real-time”, “translation”, “continuous sign recognition”, dan “signer-independent” dalam judul sebelum ada bukti yang sesuai.

**Istilah yang penting:** versi resmi model adalah **YOLO26**; untuk task yang diusulkan checkpoint-nya **`yolo26n-cls.pt`**, bukan `yolo26n.pt` yang merupakan detection. Dokumentasi resmi: https://docs.ultralytics.com/models/yolo26 dan https://docs.ultralytics.com/tasks/classify.

---

## 2. Latar belakang masalah dan research gap

### 2.1 Masalah aplikatif

BISINDO memiliki gestur yang dapat melibatkan posisi tangan, konfigurasi jari, arah, gerakan, dan konteks temporal. Pengenalan pada satu frame dapat berguna sebagai studi *image-level classification*, tetapi tidak selalu cukup untuk menyimpulkan isi keseluruhan gestur dinamis. Contohnya `apa_kabar` dapat mempunyai tahapan gerak; sebuah frame persiapan atau transisi tidak selalu representatif secara semantik.

Ketika vocabulary klasifikasi berkembang, model harus mempelajari lebih banyak batas kelas. Dua fenomena dapat terjadi sekaligus:

- **Discrimination benefit:** supervisi kelas tambahan membantu model membedakan pola kelas awal yang sebelumnya bertumpang tindih secara visual.
- **New-class interference:** dua alternatif output baru justru menyerap prediksi yang benar atau salah dari kelas lama.

Kita tidak bisa memilih narasi yang disukai sebelum melihat hasil akhir.

### 2.2 Gap yang bisa diklaim secara masuk akal

Penelitian BISINDO sebelumnya telah mengevaluasi sistem berbasis YOLO untuk video real-time dan studi pengenalan isyarat berbasis video/multi-signer. Penelitian teoretis machine learning juga telah membahas keuntungan melihat lebih banyak kelas saat training. Jadi **“menemukan bahwa tambahan kelas mungkin membantu” bukan novelty baru**.

**Gap operasional studi ini:** pada delapan label BISINDO tertentu dan pipeline frame-based yang dibekukan, bagaimana penambahan dua kelas memengaruhi **enam kelas yang sama**, dan **ke mana kesalahan prediksi berpindah**? Kontribusi berasal dari kombinasi:

1. Perbandingan dua vocabulary secara *independent fine-tuning* dari checkpoint awal yang sama.
2. Evaluasi berpasangan pada frame/video/ground truth common6 yang identik.
3. Pelaporan variasi lima training seed dan ketidakpastian estimasi.
4. Analisis per kelas serta perpindahan kesalahan ke kelas baru.
5. Evaluasi domain eksternal dengan keterbatasan provenance dinyatakan secara eksplisit.

### 2.3 Posisi terhadap literatur

- **Swasono, Himamunanto, dan Maedjaja (2025)** melaporkan YOLO11 + OpenCV untuk BISINDO berbasis video real-time dengan enam kelas yang cocok dengan kelas awal studi ini. Pipeline yang dilaporkan bertumpu pada YOLO11 dan OpenCV; **bukan desain YOLO+LSTM yang dapat dipakai sebagai pembanding langsung**. Mereka mengukur aspek deteksi dan deployment; studi kita adalah klasifikasi frame dengan paired macro-F1. Jangan membandingkan angka mAP mereka langsung dengan macro-F1 kita. DOI: https://doi.org/10.57152/malcom.v5i3.2130.
- **Kindy, Leonali, dan Lucky (2025), *Word-Level BISINDO*** memperlihatkan pentingnya setting signer-dependent versus signer-independent pada data video BISINDO (1.600 video, 32 tanda, lima signer). Tidak bisa langsung menyetarakan hasilnya dengan eksperimen satu-frame kita. DOI: https://doi.org/10.1016/j.procs.2025.08.277.
- **Zhu et al. (2023), *Learning by Seeing More Classes*** meneliti bagaimana kelas tambahan saat training dapat berfungsi sebagai regularisasi dan memperbaiki representasi/generalization pada kelas awal. Ini **landasan konseptual** untuk ide discrimination benefit, bukan bukti bahwa hasil BISINDO kita pasti meningkat. Artikel tersebut berada di **IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)**, bukan TNNLS. DOI: https://doi.org/10.1109/TPAMI.2022.3225117.

**Kalimat positioning yang aman:** “Berbeda dari studi yang menitikberatkan performa sistem video real-time atau performa klasifikasi satu vocabulary, penelitian ini memusatkan evaluasi pada *retained-class performance* dan transisi kesalahan akibat ekspansi dari enam menjadi delapan kategori.” Ini masih perlu dikonfirmasi dengan tinjauan related work final agar tidak membuat klaim prioritas yang berlebihan.

---

## 3. Tujuan penelitian (*objectives*)

### 3.1 Tujuan umum

**Mengukur dan menjelaskan perubahan performa pengenalan frame BISINDO pada enam kelas awal setelah vocabulary training YOLO26 Classification diperluas dari enam menjadi delapan kelas, menggunakan desain A/B yang konsisten dan evaluasi berpasangan.**

### 3.2 Tujuan khusus

1. Membangun dan melaporkan dua kelompok eksperimen dengan *independent fine-tuning* dari checkpoint klasifikasi awal yang sama: Model A (6 kelas) dan Model B (8 kelas).
2. Mengestimasi perubahan **macro-F1 common6**, dihitung pada enam label awal yang sama, menggunakan frame yang sama untuk kedua model.
3. Mengidentifikasi kelas lama yang recall-nya membaik atau memburuk, beserta ketidakpastian perubahannya.
4. Mengukur apakah Model B mengalihkan prediksi dari kelas awal ke `halo` atau `tidur` (*new-class intrusion*).
5. Menilai apakah kesalahan antar-kelas lama berkurang, bertambah, atau hanya berpindah ke dua output baru.
6. Mengukur performa Model B pada vocabulary penuh delapan kelas, khususnya kemampuan mengenali `halo` dan `tidur`.
7. Membandingkan pola internal development dan evaluasi domain eksternal secara transparan, tanpa mengklaim generalisasi ke seluruh populasi signer BISINDO.
8. Menyediakan artefak yang dapat direproduksi: split lock, frame manifest, konfigurasi, seed, checkpoint hashes, prediksi, skrip statistik, dan log.

### 3.3 Bukan tujuan penelitian

- Mengembangkan arsitektur YOLO baru.
- Mengklaim sistem penerjemahan kalimat BISINDO, pengenalan gestur temporal penuh, atau real-time deployment.
- Mengisolasi pengaruh jumlah kelas dari seluruh pengaruh tambahan data/step/head secara kausal.
- Menunjukkan YOLO26 selalu lebih baik dibandingkan YOLO11, LSTM, atau metode lain.
- Menyimpulkan keberhasilan semua signer dari satu sumber test eksternal.

---

## 4. Rumusan masalah (*research questions*)

**RQ utama — tidak diubah:**

> **Does expanding the training vocabulary from six to eight BISINDO classes change frame-level recognition performance on the original six classes, and which classes are most affected?**

Versi Indonesia:

> Bagaimana penambahan vocabulary training dari enam menjadi delapan kelas BISINDO mengubah performa pengenalan berbasis frame pada enam kelas awal, dan kelas mana yang paling terdampak?

**Subpertanyaan untuk analisis, bukan perubahan RQ utama:**

- **RQ1 / endpoint utama:** Berapa perubahan macro-F1 Model B dikurangi Model A pada enam kelas awal dengan test yang sama?
- **RQ2 / I1:** Bagaimana perubahan recall pada masing-masing enam kelas awal?
- **RQ3 / I2–I3:** Seberapa sering prediksi kelas awal masuk ke `halo`/`tidur`, dan bagaimana kontribusi matematis pilihan output baru terhadap gap akurasi?
- **RQ4 / diagnostik eksploratif:** Adakah bukti berkurangnya kebingungan **antar-kelas lama** setelah menambahkan supervisi kelas baru, tanpa sekadar memindahkan kesalahan ke kelas baru?
- **RQ5 / kemampuan ekspansi:** Seberapa baik Model B mengenali seluruh delapan kelas, khususnya dua kelas tambahan?
- **RQ6 / external behavior:** Apakah pola tersebut terlihat pada data eksternal, dengan pembatasan klaim menurut kualitas provenance dan independensi dataset?

Semua subpertanyaan adalah cara mengurai **satu cerita ilmiah**. Jangan mengubah setiap subpertanyaan menjadi hipotesis konfirmatori baru tanpa amendemen protokol.

---

## 5. Kerangka konseptual: mengapa dua kelas baru bisa membantu atau merugikan?

### 5.1 Intuisi model A vs B

- **Model A** menerima contoh berlabel `saya`, `kamu`, `senang`, `bingung`, `marah`, `apa_kabar`; ia harus mengalokasikan probabilitas pada enam kelas tersebut.
- **Model B** menerima enam kelas yang sama **plus** `halo` dan `tidur`; ia memiliki delapan output dan memperoleh contoh supervisi eksplisit untuk kategori tambahan.

**Penambahan kelas bukan transfer learning berurutan dari A ke B.** A dan B masing-masing mulai dari pretrained YOLO26 Classification yang sama; eksperimennya bukan *continual learning* dan bukan uji *catastrophic forgetting*.

### 5.2 Hipotesis mekanisme: *discrimination benefit*

Jika visual `halo` terkadang mirip dengan `apa_kabar`, pelatihan dengan label `halo` bisa mendorong model membentuk fitur atau batas keputusan yang lebih spesifik. Secara teori hal ini dapat **mengurangi kebingungan beberapa pasangan kelas lama** atau meningkatkan prediksi benar pada frame tertentu.

**Tetapi:** data human pilot baru memberi indikasi adanya kebingungan `halo` ↔ `apa_kabar`. Itu belum membuktikan kemiripan semantik yang universal atau kausalitas. `tidur` juga tidak otomatis dianggap mirip dengan kelas lama tertentu.

### 5.3 Hipotesis saingan: *interference / competition*

Pada frame `apa_kabar`, Model B mungkin justru memberikan skor tertinggi kepada `halo`, sehingga recall `apa_kabar` turun. Bahkan jika error `apa_kabar` → `kamu` berkurang, hal itu **belum tentu manfaat diskriminasi** apabila error hanya berganti menjadi `apa_kabar` → `halo`.

### 5.4 Apa yang benar-benar diuji?

Kita mengamati **perubahan output dan kesalahan**. Kita **tidak** dapat langsung memastikan bahwa geometri embedding/backbone membaik hanya dari confusion matrix. Jika menyebut “improved representation” dalam Discussion, kata-katanya harus bersifat **kemungkinan penjelasan**, bukan temuan kausal yang telah dibuktikan. Tidak perlu menambah UMAP/t-SNE hanya untuk memperkuat narasi.

---

## 6. Hipotesis dan status keputusan statistik

### H1 — endpoint utama **BELUM DISETUJUI**

**Protokol historis:** Non-inferiority (NI) menggunakan margin kandidat δ = 0,02. Status sekarang **`NI_MARGIN_UNRESOLVED`** karena tidak ada alasan substantif yang cukup untuk menyatakan rugi dua poin macro-F1 sebagai batas yang dapat diterima. NI **tidak boleh dianggap aktif atau lolos** hanya karena angka pilot.

**Rekomendasi untuk dosen:** amendemen **estimation-focused primary analysis**, bukan mengarang nilai δ:

> Estimasi efek perluasan vocabulary sebagai selisih paired macro-F1 Model B − Model A pada enam kelas awal, disertai interval kepercayaan 95% dan variasi seluruh training seed.

Estimasi ini mengizinkan Δ positif, nol, atau negatif. CI yang mencakup nol berarti arah/magnitudo perubahan belum cukup presisi untuk kesimpulan pasti; **bukan bukti bahwa A dan B setara**.

Jika dosen memilih mempertahankan NI, margin harus ditentukan **sebelum external scoring**, berdasar alasan praktis/ilmiah independen dari hasil, dan seluruh perubahan protokol harus dicatat. **Jangan mengganti hasil protokol historis diam-diam**: amendemen harus menyatakan riwayat pilot yang sudah diamati.

### H2 — dipertahankan sesuai preregistration

> **H2 kemampuan baru:** accuracy Model B pada ground truth `halo` **dan** `tidur` masing-masing ≥ 0,80. Perlu melaporkan estimasi dan CI per kelas. Aturan konfirmatori yang diusulkan: **batas bawah one-sided 95% CI untuk masing-masing kelas harus > 0,80**; kedua syarat harus terpenuhi.

Pada task Classification, *accuracy conditional on each GT class* adalah **recall kelas tersebut**. Skor 0,81 secara deskriptif **tidak otomatis lolos H2** jika lower confidence bound belum lebih besar dari 0,80. Dengan 50 video eksternal per kelas, ambang ini cukup ketat; lima seed **bukan** 250 video independen.

### E1 — hipotesis mekanistik **eksploratif**, bukan H3 konfirmatori

> Expanding the vocabulary may reduce confusion among the original BISINDO classes by introducing additional class-specific supervision, while simultaneously creating errors into the newly introduced classes.

Versi Indonesia:

> Supervisi dua kelas tambahan berpotensi mengurangi kesalahan antar-kelas lama melalui pembelajaran batas keputusan yang lebih spesifik, tetapi juga berpotensi menambah salah prediksi kelas lama sebagai kelas baru.

**Mengapa eksploratif?** Ide pasangan `halo`–`apa_kabar` termotivasi oleh hasil human pilot yang **sudah dilihat**. Menyebutnya hipotesis konfirmatori yang ditentukan sebelum melihat data development akan keliru. Catat motivasi, definisi diagnostik, dan status eksploratif **sebelum** melihat hasil evaluasi eksternal; jangan mengubahnya menjadi H3 tanpa persetujuan.

### Hipotesis yang **tidak** diajukan

- “B pasti lebih baik dari A.”
- “Penambahan kelas membuat representasi gestur lebih baik secara kausal.”
- “Model mengerti gerakan temporal yang tidak direpresentasikan dalam frame.”
- “Penambahan data adalah satu-satunya penyebab perubahan.”

---

## 7. Dataset, kelas dan desain A/B yang harus dihafal

### 7.1 Mapping kelas **tetap**

| ID | Label | Kelompok |
|---:|---|---|
| 0 | `saya` | Original six |
| 1 | `kamu` | Original six |
| 2 | `senang` | Original six |
| 3 | `bingung` | Original six |
| 4 | `marah` | Original six |
| 5 | `apa_kabar` | Original six |
| 6 | `halo` | New class |
| 7 | `tidur` | New class |

**Peringatan:** `terima_kasih` dan `tolong` pada dokumen thesis lama **bukan** dua kelas tambahan pada dataset aktual.

### 7.2 Inventory dan split

| Data | Per kelas | 6 kelas awal | 8 kelas penuh | Pemakaian |
|---|---:|---:|---:|---|
| Internal semua video | 300 | 1.800 | **2.400** | Sumber train/val |
| Internal train (split terkunci) | 240 | **1.440** | **1.920** | Training A dan B |
| Internal val (split terkunci) | 60 | **360** | **480** | Development/selection terbatas sesuai protokol |
| Eksternal Kaggle | 50 | **300** | **400** | Evaluasi eksternal **setelah gate** |

Satu video menghasilkan **satu frame representatif untuk eksperimen utama**, bukan ratusan frame yang dianggap sampel independen.

**Berdasarkan laporan TASK 1 Codex yang disampaikan peneliti:** 2.800 video sumber ditemukan dan dicek kesesuaiannya, split dan mapping konsisten, pairing A/B sama, dan tiga pasangan frame identik pada `senang_223`/`senang_224` berada **di train yang sama**, bukan bukti kebocoran lintas split. Klaim ini belum diverifikasi independen oleh dokumen ini; *near duplicates*, validitas semantik frame, dan metadata eksternal masih terbatas.

### 7.3 Eksperimen final yang direncanakan

**A:** `yolo26n-cls.pt` (ImageNet pretraining) → 6 output → 1.440 frame train → 360 frame val.  
**B:** byte checkpoint awal yang sama → 8 output → 1.920 frame train → 480 frame val.

Kedua model:

- Dijalankan independen, **bukan** A lalu lanjut fine-tune menjadi B.
- Memakai **seed 42, 43, 44, 45, 46**, menghasilkan 5 pasangan A/B = 10 run final.
- Memakai frame awal yang sama untuk 6 kelas, split/video yang sama, pipeline evaluasi yang sama.
- Menggunakan konfigurasi yang sama pada parameter yang relevan; output head dan volume sampel berbeda sesuai desain.
- Melaporkan **semua run**, tidak hanya seed terbaik.

**Keterbatasan fairness:** equal epochs ≠ equal optimizer steps. Dengan full locked train, perkiraan jumlah step adalah A ≈ **2.700** dan B ≈ **3.600** pada 30 epoch, batch 16; harus diverifikasi saat eksekusi. Perbedaan ini merupakan bagian dari estimand *combined expansion*, bukan bukti efek class count murni. Jika dosen mewajibkan kontrol tambahan, satu **step-matched sensitivity check** dapat didiskusikan, tetapi bukan pekerjaan wajib tanpa alasan jelas.

### 7.4 Resep yang **terverifikasi dari Stage1b**, bukan persetujuan final

Menurut audit TASK 2 yang dilaporkan:

- Model YOLO26n Classification pretrained ImageNet.
- Input 224×224, **letterbox**, RGB; bukan random crop.
- Batch size 16, 30 epoch, SGD, lr awal 0,01.
- Tanpa crop, horizontal flip, vertical flip, RandAugment, atau random erasing.
- Checkpoint evaluasi `last.pt` (EMA, sesuai log).
- Lima seed yang sama; Stage1b memakai subset dengan optimizer steps aktual A=690 dan B=900, LR akhir≈0,00043.

**Candidate final** mempertahankan resep ini dengan seluruh train/val terkunci. Statusnya `PROPOSED_NOT_APPROVED`. Perubahan dari konfigurasi rencana historis (misalnya CV/faktorial) tidak boleh disembunyikan; perlu persetujuan akademik dan catatan deviasi.

**Provenance seed:** seed 42 sudah digunakan dalam pilot; daftar 42–46 tercatat kemudian pada dokumen/protokol awal November. Maka **jangan** mengklaim seluruh seed dipilih sebelum melihat hasil development. Alasan penggunaan lima seed: kompromi transparan antara variasi training dan biaya komputasi, **bukan angka seed yang sakral**.

---

## 8. Pengukuran: bagaimana membuktikan perubahan itu?

### 8.1 Endpoint utama: *paired Δ macro-F1*

Pada 300 video **common6 eksternal** yang sama untuk A dan B (jika manifest final tervalidasi), untuk setiap seed `s`:

\[
\Delta_s = \operatorname{MacroF1}_{B,s}^{\mathrm{common6}} - \operatorname{MacroF1}_{A,s}^{\mathrm{common6}}
\]

Estimasi utama adalah rata-rata lima selisih seed:

\[
\widehat{\Delta} = \frac{1}{5}\sum_{s\in\{42,43,44,45,46\}}\Delta_s.
\]

Macro-F1 dihitung sebagai rata-rata F1 **enam label awal yang tetap**. Dalam evaluasi Model B, prediksi menggunakan **argmax delapan output penuh**. Jika video `apa_kabar` diprediksi `halo`, itu **salah** dan **false negative** untuk `apa_kabar`. Jangan menutupi kelas baru saat menghitung angka utama.

**Mengapa paired?** Setiap sampel common6 dibandingkan pada video, frame, label dan urutan yang sama, sehingga perubahan A/B bukan artefak dari pemilihan sampel tes yang berbeda.

### 8.2 I1 — *class-level recall change*

Untuk kelas awal `k`:

\[
I1_k = \operatorname{Recall}_B(k)-\operatorname{Recall}_A(k).
\]

Laporkan recall masing-masing model, delta dan CI per kelas; jika mengklaim lebih dari satu kelas benar-benar terdampak, pertimbangkan interval simultan yang sudah disiapkan pada protokol. **Kelas dengan delta terbesar belum otomatis berbeda secara signifikan dari kelas lain.**

### 8.3 I2 — *novel-class intrusion*

Proporsi frame dari **enam GT awal** yang diprediksi Model B sebagai **`halo` atau `tidur`**, total maupun per kelas. Ini menjawab: “Apakah output kelas baru mengganggu prediksi lama?”

### 8.4 I3 — *accuracy-gap decomposition*

Definisikan `B_restricted`: prediksi diagnostik B ketika hanya enam output awal dipertimbangkan. Maka:

\[
\operatorname{Acc}_A-\operatorname{Acc}_B=
(\operatorname{Acc}_A-\operatorname{Acc}_{B\_restricted})+
(\operatorname{Acc}_{B\_restricted}-\operatorname{Acc}_B).
\]

Bagian kedua menunjukkan selisih yang muncul karena adanya pilihan output baru pada evaluasi kelas lama. **Ini identitas deskriptif, bukan identifikasi kausal atas perubahan pembelajaran internal.** `B_restricted` **dilarang** dipakai sebagai hasil B utama.

### 8.5 Analisis eksploratif diskriminasi yang diperkuat

Gunakan prediksi paired yang sudah tersedia untuk melihat transisi:

| Prediksi A | Prediksi B | Interpretasi |
|---|---|---|
| Benar | Benar | Stabil |
| Salah | Benar | Koreksi kesalahan oleh B |
| Benar | Salah ke kelas lama | Regresi ke kelas lama |
| Benar | Salah ke `halo`/`tidur` | New-class intrusion |
| Salah ke kelas lama | Salah ke kelas baru | Kesalahan **berpindah**, bukan otomatis membaik |

Analisis pasangan khusus (misalnya `apa_kabar` vs `halo`) boleh dijelaskan karena berasal dari human pilot, tetapi klasifikasikan sebagai **eksploratif yang termotivasi development**. Untuk menilai old-to-old confusion, tampilkan **jumlah benar dan salah yang berpasangan**, bukan hanya persentase kesalahan yang mengecil karena prediksi terdorong ke kelas baru.

### 8.6 Full8 dan H2

Pada eksternal penuh **400 video** (50×8): laporkan confusion matrix 8×8, accuracy, macro-F1, precision/recall/F1 per kelas, dan hasil `halo` serta `tidur`. Jangan membandingkan angka full8 macro-F1 B secara langsung dengan common6 macro-F1 A seolah task test-nya sama.

### 8.7 Statistik dan batas interpretasi

Draft analysis plan menyiapkan **10.000 paired GT-stratified video bootstrap replicates** yang mempertahankan pairing A/B, disertai resampling lima pasangan seed. Laporkan interval dua sisi 95%, SD antar-seed, interval alternatif *video-only*/*seed-only*, dan sensitivity assumptions yang sudah dipersiapkan. **Lima seed memberikan informasi terbatas tentang distribusi stochastic training**, sedangkan 300 video common6 bukan 1.500 sampel independen hanya karena diprediksi lima seed.

Jika Kaggle belum dapat ditetapkan sebagai sumber confirmatory yang independen, hasil harus diberi label **external exploratory** dan klaim statistik disesuaikan melalui keputusan/protokol yang didokumentasikan.

---

## 9. Temuan yang SUDAH ADA vs yang BELUM ADA

### 9.1 Human single-frame pilot — **sudah tercatat**

- 160 internal frame (20 per kelas), 2 reviewer.
- Reviewer 1: **129/160 = 80,6%**; Reviewer 2: **138/160 = 86,3%**.
- Cohen's κ ≈ **0,825** pada pelaporan pilot.
- `apa_kabar` adalah kelas yang sulit: 45% dan 70% benar dari masing-masing reviewer; beberapa frame `halo` disalahartikan sebagai `apa_kabar`.

**Kesimpulan yang boleh:** satu frame bisa ambigu untuk manusia; ada indikasi kelas tertentu sulit dibedakan. **Kesimpulan yang tidak boleh:** model terbukti memahami konteks lebih baik dari manusia atau data temporal telah terbukti penyebab salah prediksi. Human pilot dan model development **tidak boleh dibandingkan sebagai head-to-head tanpa kesamaan sampel/protokol**.

### 9.2 Stage1b development lima seed — **hasil awal, bukan hasil konfirmatori**

Menurut audit internal yang tersedia, pada **N=120 common6 validation** (bukan N=300 test eksternal):

- Mean paired Δ macro-F1 B−A = **−0,0087293042**.
- Hierarchical bootstrap CI 95% = **[−0,0302228186; +0,0076121399]**.
- **Interpretasi unit sampel:** ini **120 video yang sama diprediksi lima seed**, bukan 600 video test independen.
- CI mencakup nol; development ini tidak membuktikan B lebih baik, lebih buruk, atau non-inferior dalam pengertian konfirmatori.

**Interpretasi konsultasi yang benar:** pilot memberi sinyal perubahan rerata yang kecil dan ketidakpastian yang masih bermakna, sehingga evaluasi final dan analisis per kelas diperlukan.

### 9.3 Status berdasarkan laporan Codex terbaru

| Pekerjaan | Status yang **dilaporkan** | Makna |
|---|---|---|
| TASK 0 — Audit repo / notebook | PASS | Notebook controller dan test disiapkan; laporan, bukan audit independen kami |
| TASK 1 — Inventory/provenance audit | PASS | Angka/split/pairing cocok; metadata Kaggle belum lengkap |
| TASK 2 — Audit recipe/decision | PASS | Resep Stage1b diverifikasi, candidate final dibuat; **keputusan ilmiah BLOCKED** |
| TASK 3 — Final training readiness | Belum ada laporan selesai | Belum ada izin training |
| TASK 4 — Full-data final training | Belum dilaksanakan/terverifikasi | 10 checkpoint final belum tersedia |
| TASK 5 — Internal final validation | Menunggu model final | Tidak boleh dianggap hasil eksternal |
| TASK 6 — Freeze & external readiness | BLOCKED | Manifest, provenance, exposure review, approval belum lengkap |
| TASK 7 — External scoring | **BELUM** | Tidak ada external macro-F1 final |
| TASK 8 — Paper final | Draft | Belum submission-ready |

**Pembedaan penting:** Codex melaporkan test TASK 2 25/25 dan pemeriksaan ulang 11/11 lulus, serta perlindungan artefak historis. Pernyataan tersebut adalah **laporan eksekusi Codex yang dikirim peneliti**, bukan verifikasi langsung file lokal pada dokumen konsultasi ini.

---

## 10. Expected results: apa yang diharapkan dan apa “best case” yang benar?

**Expected results adalah tipe temuan yang ingin dapat diuji—bukan angka yang harus dipaksa keluar.** Terdapat beberapa skenario yang semuanya bisa menghasilkan diskusi akademis bermakna.

| Skenario | Pola hasil | Nilai ilmiah |
|---|---|---|
| **Best case aplikatif** | B mengenali dua kelas baru dengan baik **dan** kinerja common6 tetap tinggi/stabil atau meningkat; variasi antarseed kecil, eksternal mendukung | Membuktikan ekspansi ini praktis layak **dalam setting yang diuji**, dengan trade-off terbatas |
| **Best case analitis** | Ada pola per-class yang jelas: beberapa kelas lama membaik, sebagian terganggu, intrusion dapat dikuantifikasi, hasil konsisten dan CI informatif | Menjelaskan **mekanisme output dan trade-off** yang luput dari skor keseluruhan |
| **Trade-off negatif** | B turun, terutama pada kelas lama tertentu, dan kesalahan berpindah ke `halo`/`tidur` | Menunjukkan batas ekspansi dan titik risiko untuk desain vocabulary berikutnya |
| **Tidak konklusif** | Δ kecil, CI lebar atau domain shift besar; H2 tidak lolos | Tetap valid jika dilaporkan jujur, tetapi paper memerlukan pembahasan keterbatasan yang lebih kuat |

### Ilustrasi numerik hipotetis — **BUKAN HASIL PENELITIAN**

Contoh hanya untuk memahami interpretasi:

- A common6 macro-F1 = **0,86**, B common6 = **0,85** → Δ = **−0,01**.
- B mampu mengenali `halo` dan `tidur`, tetapi apakah H2 konfirmatori lolos tetap bergantung pada **lower confidence bound**, bukan hanya nilai rata-rata.
- Pada `apa_kabar`, misalnya error lama ke `kamu` turun, tetapi error baru ke `halo` meningkat. Ini **tidak otomatis berarti diskriminasi total membaik**.

Angka ini **tidak boleh dimasukkan ke Results**. Hapus ilustrasi numerik dari naskah submission dan ganti dengan output asli setelah scoring yang berizin.

### Definisi *best case* ilmiah

Best case untuk konferensi bukan “B harus lebih tinggi dari A” atau “accuracy 99%”. Best case adalah **hasil cukup presisi, konsisten, dapat direplikasi, menunjukkan pola kesalahan yang dapat dijelaskan, serta memiliki domain bukti yang dinyatakan jujur**. Skenario negatif yang dianalisis mendalam bisa lebih bernilai daripada skor tinggi pada split yang tidak independen.

---

## 11. Risiko ilmiah utama dan jawaban mitigasinya

| Risiko yang akan disorot reviewer | Dampak | Respons yang dapat dipertanggungjawabkan |
|---|---|---|
| **Class count berbarengan dengan jumlah sampel dan optimizer steps** | Klaim kausal “efek kelas saja” lemah | Gunakan istilah *combined vocabulary-and-data expansion* dan laporkan langkah aktual; satu step-matched sensitivity opsional jika disetujui |
| **Satu frame mewakili gestur dinamis** | Label frame dapat ambigu | Human pilot, pemilihan frame yang terdokumentasi, pemeriksaan validitas semantik; scope hanya frame-based |
| **Internal train/val mungkin signer-dependent** | Internal score terlalu optimistis | Sebut development, jangan klaim generalisasi signer dari val internal |
| **Kaggle signer metadata kosong dan riwayat penggunaan tidak lengkap** | Tidak sah menyebut untouched confirmatory / signer-independent | Review provenance, lisensi, riwayat tuning/preprocessing; jika tidak dapat dibuktikan, turunkan klaim ke external exploratory |
| **One external signer/domain** | Generalisasi populasi tidak dapat dibuktikan | Klaim transfer hanya ke sumber test tersebut, tanpa mengisolasi signer, kamera, pencahayaan, background |
| **Seed42 muncul sebelum daftar 42–46 ditetapkan** | Pemilihan seed tidak sepenuhnya prospektif | Nyatakan kronologi, pertahankan semua seed dan semua run, jangan pilih hasil terbaik |
| **Kegagalan test legacy / dead files** | Ragu pada reproducibility | Jangan hapus bukti lama; dokumentasikan satu stale test dan pisahkan dari validasi baru |
| **Potensi shortcut dari latar/background** | Model mungkin mempelajari faktor non-gestur | Satu kontrol yang terdefinisi dan benar-benar diperlukan setelah audit; jangan klaim tanpa artefak yang bisa ditelusuri |
| **Novelty hanya YOLO versi baru** | Reviewer menilai kontribusi inkremental | Fokus pada paired class expansion, interference, full8, dan external-domain behavior |
| **H1 NI margin 0,02 tanpa justifikasi** | Statistik rentan dianggap post hoc | Minta approval estimation-focused atau justifikasi NI prospektif independen dari hasil |

**Yang tidak boleh disimpulkan:** “signer-independent berhasil”, “human-like understanding”, “real-time system”, “class-incremental learning”, “terbukti dua kelas tambahan meningkatkan representasi”, atau “Model B lebih unggul di semua kondisi” tanpa bukti sesuai.

---

## 12. Langkah demi langkah penyelesaian penelitian — mengikuti TASK 0–8

Prinsip pelaksanaan: **satu TASK → audit/implementasi → test → bukti → STOP → review manusia → approval TASK berikutnya.** Notebook dipakai untuk orkestrasi dan jejak hasil, bukan sebagai pengganti artefak checkpoint/log/manifest.

### TASK 0 — Repo & notebook controller — **dilaporkan PASS**

Inventaris kode lama, penetapan modul aktif/historis, notebook utama yang tidak otomatis training/scoring, tes keamanan dan output tersimpan. Tidak membuang checkpoint atau notebook lama.

### TASK 1 — Data integrity & provenance audit — **dilaporkan PASS sebagai audit**

Pastikan jumlah 2.800 video dan split, class mapping, hash, pairing A/B, dan catatan duplicate. Catat keterbatasan semantic review dan metadata Kaggle. **PASS audit bukan berarti provenance Kaggle sudah resolved.**

### TASK 2 — Audit scientific decisions & recipe — **dilaporkan PASS sebagai audit**

Verifikasi Stage1b, lima seed, independent initialization, no augmentation, `last.pt`, serta candidate full-data recipe. Siapkan memo perubahan H1 dan cakupan CV/faktorial. **Candidate masih `PROPOSED_NOT_APPROVED`.**

### **KONSULTASI DOSEN — gerbang keputusan sebelum final training**

1. Setujui Classification vs Detection sebagai metode utama.
2. Setujui working title dan RQ 6→8.
3. Setujui H1: estimasi paired Δ dengan CI atau NI dengan margin ilmiah.
4. Setujui H2 dipertahankan sesuai protokol, lima seed, dan ruang lingkup A/B tanpa CV/faktorial tambahan bila memang layak.
5. Catat revisi yang diminta dan persetujuan secara nyata; **jangan membuat approval palsu di tracker**.

### TASK 3 — Final training readiness — **BELUM**

Verifikasi full-data manifest, data path, config frozen candidate, checkpoint policy, independent initialization, dry-run **tanpa optimizer step** dan perlindungan output. Tutup blocker training yang relevan; jangan menjalankan training hanya karena kode siap.

**Acceptance:** semua input training tervalidasi, keputusan akademik tercatat, tidak mengubah split, no external scoring, tidak ada duplikasi kode ilmiah tak perlu.

### TASK 4 — Final training A/B lima seed — **BELUM**

Lakukan 10 run independen pada resep yang disahkan; catat konfigurasi efektif, waktu, optimizer steps, library/hardware, log, model artifacts/hashes, dan kegagalan. **Jangan membuang run buruk.**

**Acceptance:** checkpoint final A/B untuk semua seed atau deviasi gagal yang dilaporkan jujur; tidak menggunakan Kaggle untuk memilih model.

### TASK 5 — Internal validation & diagnostic — **BELUM**

Gunakan data val internal sesuai split lock, hitung metrik common6 dan full8, per-seed variability dan I1–I3. Jika dipakai untuk keputusan, semua ketentuan penggunaannya harus transparan. Hasil tahap ini **development**, bukan uji final independen.

### TASK 6 — Freeze & external scoring readiness — **BLOCKED**

Selesaikan sumber/versi/lisensi Kaggle, metadata signer sesuai bukti, riwayat semua penggunaan eksternal, manifes frame 1/video, integritas dan QA, protokol H1/H2 final, config/checkpoint/source/hardware hashes, validation record, dan freeze. Jika Kaggle pernah dipakai secara score-guided untuk tuning, **jangan tetap menyebutnya untouched**.

### TASK 7 — One approved external scoring batch — **BELUM**

Hanya setelah gate dan approval terikat hash freeze; evaluasi common6 **300 video** dan B full8 **400 video** seluruh seed, pakai output B delapan kelas penuh. Simpan prediksi mentah, confusion, log dan hash. Tidak memilih seed atau threshold pasca-test. Jika batch gagal, jangan menghapus receipt atau diam-diam retry.

### TASK 8 — Analisis final, paper, submission — **BELUM**

Tulis tabel dan gambar dari hasil nyata: Δ macro-F1 + CI, per kelas, intrusion, full8, seed, external/internal gap; jelaskan keterbatasan dan related work yang benar. Cari konferensi dengan scope, template, CFP dan anggaran yang benar pada saat submission. Tidak ada jaminan diterima.

### Urutan keluaran minimal menuju paper

1. **Persetujuan metodologi tertulis** (minimal catatan konsultasi yang dapat diverifikasi).
2. **Frozen training recipe + split/frame manifest**.
3. **Sepuluh final checkpoint dan log**.
4. **Protocol/freeze evidence dan validasi Kaggle**.
5. **Satu evaluasi eksternal yang terotorisasi** atau penggantian klaim evidensial secara formal jika confirmatory tidak sah.
6. **Tabel statistik, confusion/error analysis, limitations, manuskrip akhir**.

---

## 13. Paket hasil dan visual yang sebaiknya masuk ke paper

**Tabel utama (wajib):**

1. Inventory internal dan eksternal + kelas/split/signer provenance.
2. Recipe A/B dan perbedaan data/optimizer steps.
3. Per-seed common6 macro-F1 A, B, Δ; mean dan SD.
4. Primary paired Δ macro-F1 dan interval dua sisi 95%.
5. Recall/precision/F1 enam kelas awal, perubahan recall dan CI.
6. Model B full8 performance termasuk `halo` dan `tidur`.

**Gambar bernilai tinggi (prioritas):**

- **Figure 1:** diagram data → frame selection → A6/B8 independent training → paired evaluation → interference metrics.
- **Figure 2:** confusion Model A common6, Model B common6 **6×8** (baris GT enam; kolom prediksi delapan), dan B full8 bila ruang cukup.
- **Figure 3:** forest plot Δ recall enam kelas awal beserta confidence intervals.
- **Figure 4:** bar/heatmap novel-class intrusion `halo`/`tidur` per kelas awal dan paired transition.

**Jangan menghias paper dengan grafik yang tidak menjawab RQ**, terutama embedding projection yang mudah diinterpretasikan berlebihan.

---

## 14. Cara menjelaskan ke pembimbing: naskah 90–120 detik

> **Pak/Bu, saya ingin mengajukan penyesuaian arah dan metodologi skripsi, bukan mengganti domain penelitian.** Semula judul saya menekankan training-set expansion dan real-time BISINDO phrase recognition memakai YOLO26 Detection. Setelah mengerjakan data dan pilot, saya menyadari desain aktual saya mengambil satu frame representatif per video, sehingga klaim real-time dan pemahaman gerakan penuh tidak sesuai dengan eksperimen yang benar-benar dapat saya evaluasi.
>
> Saya mengusulkan judul *Impact of Vocabulary Expansion on Frame-Based BISINDO Recognition: A Controlled YOLO26 Study*. Fokusnya adalah membandingkan dua model YOLO26 Classification yang dilatih **independen** dari checkpoint awal yang sama: model enam kelas dan model delapan kelas. Pertanyaannya bukan hanya mana yang lebih akurat, melainkan **apa dampak kelas baru terhadap enam kelas lama**, terutama apakah kebingungan berkurang atau justru prediksi kelas lama berpindah menjadi `halo` atau `tidur`.
>
> Saya sudah memiliki audit dataset dan eksperimen development lima seed. Hasil awal masih berasal dari validasi internal dan **belum** menjadi kesimpulan final. Saya berencana menilai perubahan paired macro-F1, recall per kelas, confusion dengan kelas baru, serta evaluasi eksternal setelah protokol dan provenance dibekukan.
>
> Saya ingin meminta persetujuan Bapak/Ibu atas perubahan Detection menjadi Classification, judul/RQ ini, pilihan analisis statistik utama, serta cakupan eksperimen final A/B lima seed. Jika arahnya disetujui, saya akan menyelesaikan training dan evaluasi final tanpa terus mengubah metodologi.

**Tips berbicara:** sampaikan RQ sebelum angka; tunjukkan diagram A/B; akui perubahan metodologis; jelaskan status data eksternal dengan jujur; minta keputusan spesifik, bukan sekadar “menurut Bapak/Ibu bagaimana?”.

---

## 15. Simulasi pertanyaan penguji/dosen dan jawaban ringkas

**Q1. Ini ganti topik atau ganti judul saja?**  
**A:** Perubahan metodologi substantif, tetapi satu domain dan inti eksperimen tetap dipertahankan. RQ dipersempit agar sesuai dengan satu frame/video dan klasifikasi label.

**Q2. Kenapa Classification, bukan Detection?**  
**A:** Outcome utama adalah identitas kelas gestur, bukan lokasi bounding box. Classification lebih langsung untuk evaluasi macro-F1 label gambar; konsekuensinya ia rentan pada shortcut background sehingga QA dan limitations harus jelas.

**Q3. Mengapa tidak real-time? Bukankah YOLO cepat?**  
**A:** Model cepat tidak sama dengan sistem real-time yang telah diuji dari kamera sampai output. Kita tidak mengukur pipeline streaming end-to-end dan tidak memodelkan urutan gestur.

**Q4. Apa bedanya dengan Swasono 2025?**  
**A:** Swasono menekankan implementasi YOLO11 + OpenCV pada video real-time dengan enam kelas. Kita mengukur perubahan performa kelas awal akibat ekspansi menjadi delapan kelas menggunakan paired classification dan analisis interferensi; metrik dan protokol tidak bisa dibandingkan secara angka langsung.

**Q5. Mengapa hanya dua kelas tambahan?**  
**A:** Itulah ekspansi yang tersedia pada dataset aktual dan desain terkontrol yang dibatasi dalam penelitian; tidak ada klaim mewakili seluruh vocabulary BISINDO. Dua kelas baru adalah `halo` dan `tidur`, bukan contoh arbitrer yang boleh diganti setelah hasil.

**Q6. Kenapa kelas tambahan bisa membuat kelas lama lebih baik?**  
**A:** Secara teori supervisi kelas baru dapat memaksa batas keputusan yang lebih spesifik. Tetapi juga bisa membuat *new-class intrusion*. Dua arah ini diuji sebagai penjelasan eksploratif; kita tidak menganggap peningkatan sebagai fakta sebelum hasilnya ada.

**Q7. Apa kontribusi ilmiahnya jika B malah turun?**  
**A:** Besar penurunan, kelas yang terdampak, salah prediksi ke kelas baru, dan stabilitas antarseed tetap bernilai untuk memahami trade-off ekspansi vocabulary. Klaim harus proporsional dengan interval ketidakpastian.

**Q8. Apakah ini eksperimen “controlled” kalau B lebih banyak data?**  
**A:** Controlled berarti awal checkpoint, split common6, preprocessing, recipe dan evaluasi dibuat sebanding. **Ini bukan eksperimen yang mengisolasi secara kausal jumlah kelas**. B juga menerima lebih banyak data/step dan head berbeda; kita mengestimasi efek paket ekspansi yang teramati.

**Q9. Kenapa lima seed 42–46?**  
**A:** Lima replikasi adalah kompromi komputasi yang dilaporkan seluruhnya. Angka seed sendiri tidak istimewa. Seed 42 telah dipakai pada pilot sebelum daftar final dikembangkan; kronologi itu kami laporkan, bukan kami klaim prospektif sejak awal.

**Q10. Kenapa hasil internal bisa tinggi tetapi hasil luar belum tentu?**  
**A:** Internal train/validation mungkin merekam signer yang sama. Kamera, background, sesi, dan signer sumber eksternal dapat berbeda, sehingga generalization gap perlu dievaluasi. Belum sah mengklaim signer-independence sebelum provenance benar.

**Q11. Apa bukti satu frame mewakili frasa?**  
**A:** Tidak selalu. Human pilot justru memperlihatkan ambiguitas pada `apa_kabar`. Karena itu ruang lingkupnya frame-level class recognition, bukan pengertian sekuens frasa.

**Q12. Kenapa tidak menggunakan LSTM?**  
**A:** LSTM/temporal model adalah pertanyaan lain yang memerlukan input urutan dan desain evaluasi berbeda. RQ sekarang sengaja menguji efek ekspansi vocabulary pada klasifikasi frame. Tidak perlu mengganti arsitektur tanpa justifikasi dari RQ.

**Q13. Mengapa memakai macro-F1 bukan mAP?**  
**A:** Ini **classification** satu label per gambar. Macro-F1 memberi bobot sama pada enam label awal, cocok untuk mengetahui performa antar-kelas. mAP adalah metrik utama untuk detection dan tidak dapat disetarakan langsung.

**Q14. Mengapa bukan non-inferiority dengan δ=0,02?**  
**A:** Margin NI adalah kerugian yang dianggap dapat diterima secara substantif; dokumen saat ini belum menjelaskannya. Saya mengusulkan analisis estimasi Δ dan CI, atau NI bila dosen dapat membantu menetapkan margin yang sah sebelum external scoring.

**Q15. Apa hasil minimum agar bisa menjadi paper?**  
**A:** Perbandingan A/B lengkap, metrik paired beserta ketidakpastian, analisis kelas dan intrusion, validitas data/provenance yang transparan, serta kesimpulan yang tidak melebihi bukti. Tidak ada ambang accuracy tunggal yang menjamin diterima konferensi.

**Q16. Bagaimana jika Kaggle ternyata pernah digunakan untuk tuning?**  
**A:** Tidak boleh disebut untouched confirmatory test. Riwayat itu harus diungkap, protokol/evidence tier direvisi secara resmi, dan hasil diberi status eksploratif atau gunakan dataset independen lain hanya jika disetujui dan tersedia; jangan diam-diam mengganti tes.

---

## 16. Keputusan yang perlu dicatat setelah konsultasi

Gunakan bagian ini sebagai **form catatan manual**, bukan approval otomatis:

- [ ] **Task:** YOLO26n Classification disetujui sebagai pengganti Detection (atau revisi yang ditetapkan).
- [ ] **Judul/RQ:** judul dan RQ final dikonfirmasi.
- [ ] **H1:** estimation-focused disetujui lewat amendemen, **atau** NI dengan margin dan alasan substantif.
- [ ] **H2:** ambang 0,80 dan ketentuan CI dipertahankan/dibahas sesuai protokol.
- [ ] **Desain:** lima seed, A/B independen, full locked data, no augmentation, `last.pt`, dan keputusan terkait CV/faktorial.
- [ ] **Eksploratif:** analisis old-to-old confusion dan new-class intrusion boleh dimasukkan sebagai diagnosis, tanpa mengubah endpoint utama.
- [ ] **External:** arahan tentang provenance, lisensi, status Kaggle, dan apa yang boleh diklaim.
- [ ] **Milestone:** apa yang wajib ada pada konsultasi berikutnya dan batas waktu yang disepakati.

**Tanggal konsultasi:** ___________________  
**Pembimbing:** ___________________  
**Ringkasan keputusan dan bukti persetujuan:** ___________________  
**Perubahan yang wajib dicatat dalam protokol:** ___________________

---

## 17. Referensi dan basis dokumen

### Sumber proyek yang menjadi basis utama

1. `#-FULL-PROJECT-HANDOFF-—-BISINDO-YOLO26-THESIS.txt` — konteks riset, human pilot, arah eksperimen, dan histori perubahan metode.
2. `HYPOTHESES_ENDPOINTS_ADDENDUM.md` — **draft/blocked**, X.1–X.7, H1/H2, endpoint, bootstrap, gate, dan status unresolved.
3. `external_scoring_readiness.md` — audit Stage1b, hasil development, keterbatasan exposure Kaggle, dan syarat freeze.
4. Laporan TASK 0, TASK 1, dan TASK 2 yang disampaikan peneliti di percakapan, terakhir **9 November 2026**. Detail terbaru (misalnya split, duplikasi f7/f10/f14, recipe full-data candidate) berasal dari laporan ini dan **belum diverifikasi langsung dengan berkas lokal TASK 2**.

### Literatur eksternal yang diverifikasi untuk konteks

1. Swasono, H. Y. A., Himamunanto, A. R., & Maedjaja, F. (2025). *Implementasi YOLO11 dan OpenCV untuk Pengenalan Frasa dalam Video Real-Time Bahasa Isyarat Tangan*. **MALCOM**, 5(3), 1061–1073. https://doi.org/10.57152/malcom.v5i3.2130
2. Kindy, G. O., Leonali, G., & Lucky, H. (2025). *Word-Level BISINDO: A Novel Video Indonesian Sign Language Dataset and Baseline Methods*. **Procedia Computer Science**, 269, 249–258. https://doi.org/10.1016/j.procs.2025.08.277
3. Zhu, F., Zhang, X.-Y., Wang, R.-Q., & Liu, C.-L. (2023). *Learning by Seeing More Classes*. **IEEE Transactions on Pattern Analysis and Machine Intelligence**, 45(6), 7477–7493. https://doi.org/10.1109/TPAMI.2022.3225117
4. Ultralytics (dokumentasi resmi). *YOLO26 Models*. https://docs.ultralytics.com/models/yolo26
5. Ultralytics (dokumentasi resmi). *Image Classification with Ultralytics YOLO*. https://docs.ultralytics.com/tasks/classify

### Status sumber dan aturan penggunaan

- Angka eksperimen development dan aturan hipotesis berasal dari dokumen proyek; bukan hasil scoring eksternal.
- Literatur digunakan untuk **positioning**, bukan sebagai bukti kinerja atau kesimpulan skripsi ini.
- Rekomendasi seperti estimation-focused H1, pemilihan judul alternatif, dan framing exploratory E1 merupakan **saran penulisan/metodologi yang masih menunggu persetujuan**.
- Seluruh angka contoh hipotetis dalam dokumen adalah **ilustrasi**, bukan hasil eksperimen.

---

## 18. Inti yang perlu dihafal sebelum bertemu dosen

**Kalimat penutup yang paling kuat untuk mempertahankan arah ini:**

> “Saya tidak sedang mengejar angka akurasi tertinggi atau mengklaim YOLO26 sebagai model terbaik. Saya ingin mengukur secara transparan apakah ekspansi enam ke delapan kelas mengubah kemampuan mengenali kelas lama, termasuk kemungkinan manfaat diskriminasi dan risiko interferensi. Karena data saya hanya satu frame per video, metodologi Classification frame-based dan paired evaluation lebih sesuai daripada klaim real-time yang belum diuji. Saya meminta persetujuan agar seluruh eksperimen final dan paper mengikuti batas bukti yang sama.”

**Jika pembimbing menyetujui:** selesaikan approval dan audit TASK 3 sebelum training final.  
**Jika pembimbing menolak:** catat alasan dan perubahan yang diwajibkan sebelum menyentuh pipeline final.  
**Apa pun hasil konsultasi:** jangan memalsukan status preregistration, mengganti seed/margin berdasarkan hasil eksternal, atau mengklaim generalisasi yang belum dibuktikan.
