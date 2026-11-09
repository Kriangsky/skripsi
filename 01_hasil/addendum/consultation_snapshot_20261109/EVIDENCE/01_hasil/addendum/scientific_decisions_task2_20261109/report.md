# TASK 2 — scientific decisions dan training configuration audit

Tanggal laporan: 2026-11-09 Asia/Bangkok. **Paket usulan; bukan persetujuan, amendemen atau freeze.**
Otorisasi hanya TASK 2 pada lampiran pengguna. TASK 3 tidak dijalankan.
Status validasi aktual dicatat di `validation.json`; final-readiness **BLOCKED** sampai keputusan/bukti dipenuhi.
Riwayat direktori parsial `scientific_decisions_task2_20261105/` dipertahankan; bukti baru memakai tanggal aktual.

## 1. Stage1b aktual: configured, defaults, effective

Audit mencakup sepuluh full runs A/B × seed42–46, bukan linear probe sebagai recipe final.
`effective_configuration.json` menyimpan per-run configured arguments, framework defaults, actual arguments,
hash args/checkpoint, versi/tanggal/model/transform checkpoint, counters, dan LR akhir.

| Komponen | Configured / default framework lokal | Bukti efektif / batas |
|---|---|---|
| Task/model | classify, yolo26n-cls, pretrained=true | checkpoint YAML scale=n, head6/8; log YOLO26n-cls, pretrained234/236 items; bukan Detection/COCO |
| Init | pinned ImageNet `yolo26n-cls.pt` | SHA256 `0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`; semua execution/trained records sama; source memuat model fresh per arm, bukan B melanjutkan A |
| Input | imgsz224 | selected full frame di-letterbox aspect-preserving gray114 menggunakan `classification_pilot.letterbox`, disimpan PNG224; loader BGR→RGB |
| Transform train/val | override trainer; bukan transform default | `Resize((224,224),bilinear,antialias=True) → ToTensor → Normalize(mean0,std1)`; normalisasi identitas, RGB/255, bukan ImageNet mean/std |
| Crop | scale=0; default Classification punya RandomResizedCrop/CenterCrop | override mengganti seluruh Compose train/val; tidak ada RandomResizedCrop, CenterCrop atau crop lain efektif |
| Flip | fliplr0 vs default.5; flipud0 | tidak ada HorizontalFlip/VerticalFlip dalam override atau Compose checkpoint |
| Auto augmentation | auto_augment=null vs default randaugment; erasing0 vs default.4 | tidak ada RandAugment/AutoAugment/AugMix/RandomErasing; HSV/geometry/mosaic/mixup/cutmix/copy-paste0; tidak ada augmentasi stokastik pada Compose |
| Optimizer | SGD vs default auto; lr0.01 | log constructor SGD(.01,.937), guard memeriksa classSGD; Nesterov true dari environment groups/source; bukan auto/MuSGD/AdamW |
| Batch/budget | batch16/nbs16 vs default nbs64; epochs30 | A360 train/120val; B480/160; batch terakhir A8/B16; accumulation1; counters690/900; seluruh30 hasil epoch lengkap |
| LR schedule | lrf.01,cos_lr=false,warmup3 | linear LambdaLR; akhir epoch30=.00043 semua groups, bukan .0001; seluruh epoch-end LR cocok rekonstruksi source lokal |
| Momentum/decay | .937/.0005; warmup_momentum.8,warmup_bias_lr.1 | warmup momentum .8→.937; bobot decay.0005, norm/bias0; nbs=batch membuat scaling decay1 |
| Warmup | 3epoch | source terpasang: round(3×batches), A69/B90; tidak mengasumsikan minimum100; bias mulai.1, groups lain0; target mengikuti schedule |
| Checkpoint | last.pt setelah30,patience0,resumefalse | sepuluh SHA cocok trained_models; last berisi EMA final; bukan best.pt walau framework menyimpan/melakukan final validation best |
| Seeds | training42–46,sampling42,deterministictrue | workers0,ampfalse,compilefalse; fresh model per arm, seed yang sama per pasangan; kontrol tidak membuktikan hasil bit-identical |

Argumen `augment=False` sendiri bukan bukti augmentasi training mati: ClassificationDataset membangun
pipeline train menurut mode; bukti penentu adalah override `ControlledTrainer.build_dataset`,
`ClassificationDataset.__getitem__` yang memakai `torch_transforms`, dan Compose tersimpan di checkpoint.
Compose tersimpan berasal dari attachment validation loader; source override train dan val sama.
Tidak tersedia trace transform setiap batch historis. Karena itu klaim train didukung source terkunci +
checkpoint validation + records, bukan rekaman visual seluruh batch atau replay training.

Semua10 Compose diuji pada RGB sintetik224: dua pemanggilan identik, tepat tensor RGB/255.
Tidak menjalankan model forward, backward, optimizer atau dataset scoring. Tidak menginstal paket.
Framework tercatat dan lokal: Ultralytics8.4.197, torch2.14.0+cu126, torchvision0.29.0+cu126;
environment historis CUDA12.6/RTX4070LaptopGPU. Ini bukan benchmark hardware baru.

Formula LR lokal: `lr(epoch)=.01*((1-epoch/30)*.99+.01)` untuk epoch index0..29 sesudah warmup.
Target .0001 berada pada index30, setelah30 epoch selesai. Rekonstruksi interpolation warmup cocok
seluruh90 nilai epoch×group per run (toleransi pembulatan CSV). Counter actual_optimizer_steps adalah
instrumentasi `optimizer_step` source; tidak tersedia serialized optimizer state akhir karena strip.
Checkpoint `epoch=-1, optimizer=None` berarti stripped, bukan nol epoch; bukti completion memakai results,
embedded train_results30 baris, optimizer_steps dan trained_models. `updates` juga dapat dihapus saat strip.

Source framework lokal dihash untuk reproducibility audit sekarang; hash framework historis saat training
tidak tersedia. Kesamaan versi dan kecocokan LR bukan bukti byte source framework historis identik.
Source project terhadap research_config historical diverifikasi pada validation record.

`init_seeds(args.seed+1+RANK)` pada proses single-device RANK=-1 memakai seed argumen.
Python/NumPy/Torch diset; dataloader punya generator dengan offset tetap yang bergantung seed/rank.
`torch.use_deterministic_algorithms(True,warn_only=True)`, cudnn deterministic, CUBLAS workspace diset
source lokal; PYTHONHASHSEED di-set runtime bukan bukti startup hash randomization dikunci.
Tidak ada RNG-state trace penuh atau uji lintas mesin; jangan menjanjikan determinisme mutlak.
Head6 vs8 berdimensi berbeda; identical pretrained backbone bytes tidak berarti seluruh tensor head A/B identik.

## 2. Audit provenance seed

`seed_provenance.json` memisahkan fakta dari yang hilang.

1. Handoff bernama2026-11-02, §22, sudah menyebut contoh42,43,44,45,46. Dokumen juga memuat hasil pilot42.
   Baris ini hadir di commit8b3bf45 bertanggal2026-11-03T12:13:54+07:00.
   Tanggal nama dokumen bukan timestamp pembuatan/approval independen.
2. Kandidat2026-11-03 §2 menetapkan daftar tersebut. D002 menyatakan desain baru setelah pilot42 diketahui;
   Addendum01 menyatakan hanya42 sudah diamati. Daftar bukan prospektif terhadap pilot pertama.
3. Approval per seed bertanggal2026-11-03, scope stage1b, merujuk explicit researcher chat.
   Ini record lokal; chat asli dan timestamp keputusan presisi tidak diautentikasi audit ini.
4. Tabel all_fits dan sepuluh run records mempertahankan semua42–46; tidak ditemukan registered seed terbuang.
   Seed42 pilot/notebook/replikasi tidak menambah seed independen. Probe42 gagal sebelum optimization,
   runner preflight0fit dan multi-work settings conflict tetap terlihat di failure history.

Alasan awal angka42, urutan43–46, dan tepat lima seed **tidak terdokumentasi**. Lima seed memungkinkan
deskripsi variasi, tetapi itu bukan justifikasi historis pemilihan atau bukti jumlah optimal/power memadai.
Tidak ada bukti memilih pemenang dari42–46 dalam records yang diaudit. Tidak dapat memastikan pilihan awal
bebas pengaruh hasil karena pilot42 sudah diketahui. Riwayat run di luar repo/chat/perangkat lain dan omitted
attempts belum diattest pemilik. Tidak menambah, mengganti atau mengecualikan seed.

## 3. SATU recipe final yang diusulkan

Machine-readable: `candidate_config.json`, **CANDIDATE / PROPOSED_NOT_APPROVED**.
Semua izin aksi false, approval null, protocol_amended false; tidak masuk analysis_plan aktif.

| Parameter | Stage1b | Usulan tunggal | Konsekuensi |
|---|---|---|---|
| Model/init/seeds | yolo26n-cls ImageNet pinned, independent A/B,42–46 | sama | tidak ada arsitektur/seed search |
| Resolution/preprocess | letterbox224 gray114,RGB/255 | sama untuk train/val/external | tidak mengandalkan generic Classification predict yang dapat center-crop |
| Augmentation | tidak ada | tetap tidak ada | label-safety konservatif; generalisasi/overfit belum terselesaikan |
| Optimizer/batch/epoch | SGD.01,.937,decay.0005,batch=nbs16,30 | sama | scheduler/warmup/control sama |
| Train/val per kelas |60/20 |240/60 sesuai frozen split | full internal TRAIN saja; val tidak digabung train |
| Train total A/B |360/480 |1440/1920 | data baru dan optimization budget bertambah |
| Val total A/B |120/160 |360/480 | reused development val; bukan holdout baru |
| Steps A/B |690/900 terukur |2700/3600 direncanakan |90/120 batches×30; hitung ulang dari runtime final |
| Warmup A/B |69/90 dari source+curves |270/360 direncanakan | proporsi3epoch sama; jumlah updates beda |
| Checkpoint |fixed last30 EMA |sama |tidak memilih best.pt/seed atau early-stop |
| CV/grid/factorial/step supplement |rencana historis, belum dilaksanakan |tidak dimasukkan kandidat ini |perubahan scope rencana memerlukan persetujuan eksplisit |

Full-data bukan perubahan split. Candidate §6 protokol memang memuat240/60, tetapi berpindah dari Stage1b
ke full-data tetap perubahan populasi training/validation dan exposure yang harus disetujui. Candidate ini
mempertahankan baseline, tidak menjalankan tuning/CV atau mengklaim recipe optimal. Rencana historis CV/grid,
105fits dan factorial/H3 tidak dihapus; usul melewati bagian itu harus dicatat dalam amendemen scope.
H3 tidak diuji; bukan dinyatakan gagal/terbukti. Tidak menambahkan experiment step-matched pada Task2.

Equal-epoch dipertahankan sesuai estimand primer historis. B melakukan900 vs A690 pada pilot dan planned3600
vs2700 pada full-data. Rasio bukan compute/FLOPs/wall-time sama. Estimand mencakup vocabulary, tambahan contoh
kelas baru, dan perubahan optimization budget; bukan efek kausal murni jumlah kelas. Mengubah menjadi
step-matched primer akan mengubah estimand dan memerlukan keputusan lain; tidak diusulkan di sini.

Horizontal flip berisiko mengubah handedness/relasi spasial; tidak ada bukti label-safe per gesture lokal.
Tidak menyatakan seluruh gesture pasti berubah makna. Tanpa validasi semantik, alasan menambah flip tidak ada.
Vertical flip/crop dapat merusak orientasi atau menghilangkan tangan/wajah; ditiadakan seperti baseline.
No-augmentation tidak membuktikan robustness; model dapat memakai signer/background cues. Internal pilot
hand masking tetap diagnostik, bukan eksperimen kausal tentang bahasa.

API existing `classification_research` masih Stage1b-only dan `TARGETS60/20`. JSON ini bukan argumen CLI
yang langsung runnable. Task3 harus merancang release final/parameter target dan approval scope tanpa
mengubah frozen historical modules/runs. Reuse selection/letterbox/evaluation/statistics; jangan copy algoritme.
Runtime final harus mencatat Compose aktual train/val, initial hashes, counters dan scheduler evidence.

## 4. Memo keputusan H1

| Pilihan | Persyaratan | Implikasi | Status |
|---|---|---|---|
| A: pertahankan NI | δ berdasarkan acceptable loss yang substantif, outcome-independent; persetujuan peneliti/pembimbing; addendum versi baru sebelum prospective fit/external scoring | H0 Δ≤−δ; sukses lower one-sided95% >−δ; equality gagal; tidak lolos bukan otomatis inferior | justifikasi .02 hilang; tidak memilih margin baru |
| B: estimasi paired Δ | persetujuan eksplisit mengubah tujuan H1 dan scope; addendum prospektif baru, judul/RQ/metode sinkron | mean lima paired Δmacro-F1 dan two-sided95% hierarchical CI; tidak ada keputusan NI/pass/fail margin atau klaim setara | **DIREKOMENDASIKAN; BELUM DISETUJUI** |

Rekomendasi B didasarkan pada tidak tersedianya justifikasi acceptable-loss dan tujuan mengukur perubahan,
bukan karena pilot gagal NI. Estimasi bukan NI yang dilonggarkan dan bukan uji superiority terselubung.
H1 setelah amendemen menjadi endpoint/pertanyaan estimasi; jangan mempertahankan label pengujian H1 NI
di abstract lalu memakai inference estimasi saja. CI yang mencakup nol bukan bukti kesetaraan.

Δ per seed = macro-F1 B−A pada identik common6 IDs/pixels/GT, rata-rata enam label lama tetap.
Argmax B tetap8 output; halo/tidur pada GT lama tetap kesalahan/FN. Primary mean lima pasangan seed,
two-sided95% hierarchical paired GT-stratified video/seed bootstrap10000 RNG20261103 menurut implementasi
existing. Jangan pool seed×video sebagai independen; laporkan seluruh seed, SD, video-only/seed-only sensitivity.
External count300/400 belum merupakan scored denominators final. Tidak ada bootstrap baru diperlukan audit ini.

Disclosure wajib dalam amendemen/manuskrip: pilot seed42 dan reproduksinya sudah terlihat sebelum kandidat
2026-11-03; seluruh Stage1b lima seed dan analisis/diagnostics internal sudah terlihat sebelum keputusan ini.
Laporan existing mencatat mean pairedΔ=-0.008729304229932344 dan interval melintasi−.02; dibaca, tidak dihitung
ulang. Jangan mengklaim keputusan2026-11-09 preregistered sebelum pilot. Perubahan berlaku prospektif untuk
run/evaluasi yang belum dilakukan; hash/timestamp baru memakai waktu nyata setelah approval, bukan backdate.
Kaggle sudah dipreprocess v2/v3; status belum scoring/tuning bukan attestation lengkap. Pemilik/reviewer wajib
mengungkap exposure lain. Tidak mengarang komitmen, tanda tangan atau izin.

## 5. H2 tetap persis

Sumber: PREREGISTRATION.md §1; byte kutipan juga disimpan `scientific_decisions.json`.

> - H2 kemampuan baru: accuracy B pada GT halo **dan** GT tidur masing-masing ≥ **0.80**.
>   Ambang 80% per kelas disetujui peneliti pada 2026-11-03, sebelum run baru.
>   Laporkan point estimate dan CI per kelas. Aturan konfirmatori yang diusulkan: lower one-sided 95% CI
>   masing-masing kelas > 0.80 (intersection-union: kedua komponen harus lolos).
>   Point estimate ≥ 0.80 saja disebut memenuhi target deskriptif, bukan lulus pengujian konfirmatori.

Accuracy bersyarat GT halo/tidur = recall masing-masing kelas, bukan precision atau pooled new-class accuracy.
Pertahankan mean lima seed dan hierarchical bounds sesuai addendum/analysis_plan; keduanya lower bound>.80,
equality gagal, point≥.80 saja deskriptif. Intersection-union tetap; tidak menambahkan Bonferroni dua komponen.
H1 opsiB tidak mengubah H2, GT/prediction scope, seed, threshold, tail atau success criteria.

## 6. Persetujuan akademik dan judul/RQ

Record keputusan peneliti Classification/D001 dan title Addendum01 **tersedia**. Bukti persetujuan formal
pembimbing/otoritas akademik untuk mengganti Detection→Classification, judul dan RQ **tidak ditemukan**.
Agent tidak bisa menyamakan approval chat peneliti dengan approval akademik.

Ajukan ke pembimbing dan, bila diwajibkan prosedur program, koordinator/program/penguji berwenang:
perubahan metode/task, endpoint macro-F1/H2, rancangan static-frame dan batas klaim, full-data tanpa search,
H1 estimasi vs NI, penghapusan klaim real-time/aplikasi, serta nasib factorial/H3 historical.
Nama pejabat, formulir dan rantai approval institusi belum tersedia; harus ditentukan dari prosedur resmi,
bukan dikarang. Persetujuan sebaiknya record nyata bertanggal/versi/artifact-bound, dengan syarat reviewer.

Judul keputusan peneliti Addendum01: **Effect of Vocabulary Expansion on YOLO26 Classification for BISINDO
Phrase Recognition**. Judul draft analysis_plan/notebook: **Impact of Vocabulary Expansion on Frame-Based
BISINDO Recognition: A Controlled YOLO26 Study**. Keduanya tidak disamakan sebagai satu approval akademik.
Usulan judul untuk approval: judul frame-based pada analysis_plan, karena satu frame/video bukan recognizer
temporal. Usulan RQ: “Berapa perubahan paired macro-F1 enam kelas lama ketika model independently fine-tuned
diperluas menjadi delapan kelas, dan apakah recall tiap kelas baru memenuhi aturan H2 asli?”
Jelaskan equal epochs dengan tambahan data/steps; controlled tidak berarti mengisolasi semua confound.
Tidak mengubah judul manuscript/protokol lama dalam Task2.

## 7. Temuan Task1 dibawa, belum selesai

- Tiga exact candidate-frame pairs = frame7,10,14 dari **dua video** senang_223 dan senang_224,
  setiap nomor berpasangan lintas kedua video. JPEG bytes dan decodedRGB sama per pasangan;
  bukan tiga pasangan video, bukan seluruh video byte-identical. Kedua video locked TRAIN;
  video bytes berbeda. Development hanya224/frame48, bukan tiga frame duplikat. Tidak ditemukan
  cross-split image identity pada scope Task1; tidak membuktikan bebas near-duplicate/semua-frame overlap.
  Full-data memasukkan keduanya: content-cluster/semantic review tetap downstream. Jangan deduplicate/edit split otomatis.
- Kaggle citation lokal bukan acquired-version receipt, archive hash, download/rename/subset provenance,
  atau license binding. Semuanya belum lengkap. Tidak menebak lisensi dari publiknya halaman.
- Signer/session external tidak tersedia (0/400); internal0/2400. Tidak ada klaim signer-independent/populasi signer.
- Exposure attestation belum lengkap: preprocessing historis terbukti; pencarian receipt internal-only tidak
  membuktikan tidak pernah ada scoring/tuning di luar records. Attestation hanya pemilik/reviewer nyata.
- Final internal/external frame manifests masih null; semantic labels belum expert-verified. Technical flags
  bukan gesture labels, reviewer training-pool pilot bukan adjudication external. Tidak diam-diam menyatakan solved.

Tidak mengulang45.250 file checks. Task2 rehash metadata/source/protocol/checkpoints untuk preservation saja,
bukan semua video/frame. Semua-data semantic/near-duplicate audit di luar Task2.

## 8. Integrasi, validasi dan batas

Notebook primer ditambah dua sel kode read-only TASK2 pada bagian konfigurasi/keputusan. Output nyata
baru hanya untuk sel tersebut. Semua source/output/execution_count sel kode lama dipertahankan; status lama
dibaca sebagai historis. Default dan opt-in gates training/freezing/scoring tetap menolak aksi.
Tracker tunggal existing diappend; tidak membuat tracker tandingan. `plan.md` hanya rencana kerja audit ini.

Inventory aktual dan hash preservation: `validation.json`. Candidate JSON tidak membekukan source/config,
tidak mencatat fabricated final checkpoint dan tidak dipromosikan menjadi experiment_config approved.
Tes memeriksa struktur/parameter/per-arm counts, ketidaksamaan step budgets, H2 verbatim, seed evidence,
sepuluh transform/counter records dan notebook guards/preservation. Suite statistik/interference existing
dipakai ulang. Full legacy suite tidak diulang; known stale notebook-output expectation Task0 tetap terbuka.

Audit attempt1 berhenti pada assertion string log optimizer akibat escapeANSI; belum mengubah notebook/protokol.
Attempt2 menormalisasi escapeANSI sebelum memeriksa constructorSGD. Bukti attempt disimpan, bukan training failure.

TASK3 membutuhkan approval recipe/H1/scope akademik, rencana menyelesaikan evidence blockers, final manifests
dan implementasi release final dengan scoped authorization. Izin LANJUT TASK3 sendiri tidak memberi izin fit,
external scoring, final protocol amendment atau freeze. Task4/6/7 tetap gate terpisah menurut tracker.

### Hasil aktual validasi

**TASK 2 STATUS: PASS audit/paket usulan; final readiness BLOCKED.** Validator selesai dengan exit0:
25/25 tests (candidate8, notebook3, confirmatory_statistics10, interference4), runtime compatibility exit0,
git diff --check exit0; seluruh notebook berjalan di kernel project, hanya2 selTASK2 outputs disimpan.
13 sel kode lama source/output/execution_count identik. 2.026 protected files diperiksa, drift0;
seluruh source project cocok recorded research_config lima seed, initial checkpoint hash cocok pin.

Attempt notebook sandbox awal macet sebelum output tersimpan; PIDs validator milik Task2 saja dihentikan.
Retry akses kernel lokal lulus. Penyebab stall tidak dipastikan; warning Windows ZMQ selector-thread tetap
tercatat dan tidak menggagalkan eksekusi. Snapshot/log awal tidak dihapus. Serialisasi line endings dirapikan
tanpa mengubah JSON code/output semantics; `final_verification.json` mengikat bytes setelah dokumentasi akhir.

Tidak ada training, model forward, Kaggle inference, external scoring, protocol amendment atau freeze.
Tidak ada data/split changes; Task3 belum dimulai. Full suite legacy dan all-frame/near-duplicate/semantic
review tidak diulang; risiko ilmiah dan known historical notebook-test mismatch tetap dicatat.

**MENUNGGU PERSETUJUAN: LANJUT TASK 3**
