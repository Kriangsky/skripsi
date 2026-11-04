# Tracking proyek BISINDO YOLO26

Terakhir diperbarui: **5 November 2026, Asia/Bangkok**. Dokumen tracking utama untuk status implementasi, percobaan, hasil, kegagalan, dan pekerjaan tertunda.

**Status sekarang: Tahap1b selesai; pilot belum dapat memutuskan non-inferiority (NI). Tahap2 belum dimulai.** Semua hasil Classification saat ini **exploratory/development**, bukan confirmation final.

## 1. Ringkasan status

| Bagian | Status | Catatan |
|---|---|---|
| Audit/inventory/split | Selesai | Dataset aktual2.800 video; split frozen berdasarkan video |
| Candidate-frame/landmark cache v3 | Selesai secara teknis | Bukan approval semantik gesture |
| Human single-frame feasibility | Selesai |160 sampel,2 reviewer; bukan benchmark head-to-head pada validation model |
| Detection training | BLOCKED / tidak dilakukan |0 verified positives; metadata signer/sesi belum memenuhi gate |
| Classification pilot awal +reproduksi notebook | Selesai | A/B seed42,30epoch,last.pt; bukan seed tambahan |
| Tahap0/protokol kandidat | Artefak tersedia | Kandidat/hash dipertahankan; bukan final confirmatory freeze |
| Tahap1a | Selesai | Kode, unit tests, paired analysis existing pilot, power planning;0 model fits fase1a |
| Tahap1b | Selesai |10 full fits +10 head fits, seluruhseed42–46,masking/paired/kurva |
| Organisasi workspace | Selesai |9 section,root README/config Git; runtime kompatibilitas tersembunyi |
| Tahap2 CV/tuning | BELUM | Membutuhkan keputusan/addendum/gates/persetujuan baru |
| Tahap3/final external scoring | BELUM | Belum ada config final/freeze/batch skor external |
| Draft paper final +addendum berikutnya | DITUNDA | Draft/addendum historis yang ada tidak direvisi pada fase ini |

## 2. Data dan batas interpretasi

- Frozen inventory: source=`test`,split=`test`: **400** video.
- Frozen inventory: source=`train`,split=`train`: **1920** video.
- Frozen inventory: source=`train`,split=`val`: **480** video.
- Mapping tetap: saya,kamu,senang,bingung,marah,apa_kabar,halo,tidur; ID0–5 sama untuk A/B.
- Classification pilot:60 train+20 validation/kelas. A360/120, B480/160; common6 memakai120 video sama; full8 B160 adalah tugas terpisah.
- Satu frame/video; label berasal dari metadata video, belum seluruhnya diverifikasi semantik ahli. Tidak membuang flagged/difficult frames berdasarkan error model.
- Signer/session independence belum terbukti. Interval video dapat terlalu optimistis bila signer/sesi berulang.
- Kaggle pernah diproses secara historis dalam preparation v2/v3. Belum dipakai sebagai skor tuning atau evaluasi model Classification final pada fase ini.
- Rencana external common6=300 adalah subset full8=400; angka itu rencana evaluasi, bukan denominator pilot yang sudah diuji.

Bukti: [frozen split](../02_exp/arsip_pipeline/experiment_yolo26_v3/split_lock.csv); [manifest Classification](../02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/manifest.csv).

## 3. Yang sudah dibuat

| Komponen | Artefak utama | Status penggunaan |
|---|---|---|
| Audit dan pipeline data | [bisindo_pipeline.py](../05_kode/src/bisindo_pipeline.py) | Dipakai; integritas, mapping, split dan provenance |
| Pemilihan kandidat frame | [frame_candidates.py](../05_kode/src/frame_candidates.py) | Dipakai; v3 cache/proposals, bukan classifier gesture |
| Human pilot +import/analysis | [pilot_study.py](../05_kode/src/pilot_study.py) | Selesai pada160 video |
| Controlled Detection gates/evaluator | [training_pilot.py](../05_kode/src/training_pilot.py) | Implementasi ada; training tetapblocked |
| Classification pilot legacy | [classification_pilot.py](../05_kode/src/classification_pilot.py) | Dipakai untuk pilot/reproduksi; last.pt |
| Classification parameterized API | [classification_research.py](../05_kode/src/classification_research.py) | Seed/epochs/imgsz/work/fingerprint/evaluation; no overwrite/resume |
| Paired statistics | [analysis_paired.py](../05_kode/src/analysis_paired.py) | One-to-one GT/hash, McNemar exact, Wilson, bootstrap, union errors |
| Power/precision planning | [power_precision.py](../05_kode/src/power_precision.py) | Simulasi dengan asumsi eksplisit; bukan angka external/pilot yang diarang |
| Masking/probe/train curves | [classification_diagnostics.py](../05_kode/src/classification_diagnostics.py) | Dipakai; mask15%/gray114,before letterbox; frozen eval head6 |
| Workflow notebook | [classification_workflow.py](../05_kode/src/classification_workflow.py) | Ada; multi-directory in-process punya konflik settings. Run sukses memakai isolated runner |
| Runner proses terpisah | [exploratory_stage1b_runner.py](../05_kode/src/exploratory_stage1b_runner.py) | Dipakai menyelesaikan Tahap1b tanpa mengubah source locked |
| Unit tests | [tests](../05_kode/tests) | Tes lama+baru; lihat section verifikasi |
| Notebook eksperimen | [eksperimen](../06_notebook/eksperimen) | Disimpan beserta output/error historis; tidak dibersihkan untuk menyembunyikan kegagalan |
| Notebook hasil executed | [BISINDO_Classification_Stage1b_Results_Exploratory_Executed.ipynb](../06_notebook/hasil/BISINDO_Classification_Stage1b_Results_Exploratory_Executed.ipynb) | Hasil/read-only rendering aktual dari artefak, termasuk plot |
| Protokol kandidat/hash | [PREREGISTRATION.md](protokol/PREREGISTRATION.md) | Tidak diedit; hash tetapvalid |
| Addendum01 historis | [PREREGISTRATION_ADDENDUM_01.md](protokol/PREREGISTRATION_ADDENDUM_01.md) | Sudah ada sebelumfaseini; tetaputuh, bukan addendum baru |
| Laporan Tahap1a | [exploratory_stage1a_report.md](../01_hasil/tahap1a/exploratory_development_stage1a_20261103/exploratory_stage1a_report.md) | Selesai; FACT/asumsi/proposal |
| Laporan Tahap1b | [exploratory_stage1b_report.md](../01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_stage1b_report.md) | Selesai; all seeds/failures/timing/diagnostics |
| Navigasi workspace | [README.md](../README.md) | Sembilan section,rootbersih |
| Launcher +hard-link guard | [run.ps1](../05_kode/run.ps1) | Cwd/importkompatibel; tolak linkdivergence tanpa menghapus versi |
| Manifest/rollback organisasi | [README.md](organisasi_20261104_lengkap/README.md) | Pemetaan,hash/fileIDs,CheckOnlyrollback tersedia |

## 4. Yang sudah dicoba dan hasilnya

### 4.1 Preparation dan human feasibility

- v3: **2800** video, **33250** candidate frames, **1120** technically eligible; selected positives **0**. Technical eligible bukan semantic approval.
- reviewer_1: **129/160**, accuracy_all **0.80625**; ambiguous=27,invalid=0.
- reviewer_2: **138/160**, accuracy_all **0.86250**; ambiguous=12,invalid=4.
- Agreement **0.84375**, Cohen kappa **0.825160**. Ambiguous/invalid tetap denominator.

Bukti: [annotation summary v3](../02_exp/arsip_pipeline/experiment_yolo26_v3/annotation_summary.json); [human pilot analysis](../02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/pilot_report.md).

### 4.2 Classification pilot awal — satu seed

| Model | Scope | N | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| A | common6 | 120 | 0.933333 | 0.931776 |
| B | common6 | 120 | 0.925000 | 0.930042 |
| B | full8 | 160 | 0.937500 | 0.935464 |

Reproduksi seed42 dari notebook tersedia terpisah; bukan seed baru dan bukan bukti multi-seed tambahan. Ini tetap pilot internal exploratory.

### 4.3 Tahap1a — analisis dan planning, tanpa training

- Paired video bootstrap10.000/RNG20261103; ekstra output B tidak dihapus; Wilson untuk accuracy, bukan F1.
- Power/presisi external300 dan CV internal disimulasikan dengan diskordansi/seed-SD eksplisit; margin .02/.03/.05 hanya skenario, tidak mengubah protokol.
- H2 n50/kelas dengan threshold .80 memakai referensi binomial yang dijelaskan; bukan exact power hierarchical lima seed.
- Train top1 lama tidak tersedia; instrumen baru menghitung train set tanpa augmentation, menjaga mode/weights/RNG.

### 4.4 Tahap1b — lima seed lengkap

10 full fits =2 full fits seed42 yang telah diselesaikan peneliti +8 full fits berikutnya oleh agen. Tidak melatih ulang full seed42 untuk mengganti hasil.10 head fits selesai; kegagalan probe awal terjadi sebelum optimization.

Konfigurasi full: A6/B8,224px,30epoch,seed42–46,pretrained pinned sama dan init independen,last.pt fixed. Probe: backbone frozen eval,head baru6kelas,train lama saja,SGD lr.01/momentum.937/decay.0005,batch16,30epoch,last head,tanpa val selection.

| Seed | Full A/B | Probe A/B | Paired delta F1 B−A | Catatan |
|---|---|---|---:|---|
| 42 | completed/completed | completed/completed | -0.001734 | Replikasi; probeA attempt1 gagal sebelum optimization,attempt2 selesai |
| 43 | completed/completed | completed/completed | -0.001731 | Semua fit selesai |
| 44 | completed/completed | completed/completed | -0.015553 | Semua fit selesai |
| 45 | completed/completed | completed/completed | -0.032819 | Semua fit selesai |
| 46 | completed/completed | completed/completed | 0.008190 | Semua fit selesai |

| Model | Scope | Macro-F1 mean ± SD | Accuracy mean ± SD |
|---|---|---:|---:|
| A | common6 | 0.939900 ± 0.011578 | 0.941667 ± 0.011785 |
| B | common6 | 0.931170 ± 0.005478 | 0.930000 ± 0.007454 |
| B | full8 | 0.936193 ± 0.006182 | 0.937500 ± 0.006250 |

- Mean paired delta **-0.008729**; hierarchical CI95% **[-0.030101; 0.007668]**, lowerone-sided95% **-0.025590**.
- **NI belum dapat diputuskan; stop rule aktif.** Tidak memilih seed/checkpoint dan tidak otomatis lanjut CV.
- Pooled discordants A-only=8,B-only=1; pooled McNemar p=0.0390625 **deskriptif saja**, karena video berulang lintas seed.

Bukti: [seluruhfit dan riwayaterror](../01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_all_fits.csv); [paired perseed](../01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_paired_per_seed.csv); [multiseed summary](../03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_exploratory_multiseed/exploratory_multiseed_summary.json).

### 4.5 Masking, linear probes, dan kurva

| Model/scope | Hand-masked accuracy mean | Uniform-output chance |
|---|---:|---:|
| A/common6 | 0.920000 | 0.166667 |
| B/common6 | 0.916667 | 0.125000 |
| B/full8 | 0.891250 | 0.125000 |

| Backbone | Probe6 macro-F1 mean ± SD |
|---|---:|
| A | 0.933427 ± 0.004828 |
| B | 0.925928 ± 0.009164 |

- Mask/coverage tersimpan,union ROI15%,fill114,sebelumletterbox. Semua640 pilotframe punya landmark; missing-landmark strata tetap diimplementasikan/diuji.
- Akurasi setelah masking tinggi **bukan bukti kausal** bahwa tangan tidak dibutuhkan: wajah/tubuh atau tangan yang luput dari landmark dapat tersisa; ada distribution shift.
- Kurva semua seed tersimpan. Train diagnostic memakai actual weights; validation framework memakai EMA, sehingga gap harus dibaca dengan caveat itu.

Bukti: [masking summary](../01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_masking_summary.csv); [probe summary](../01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_probe_summary.csv); [kurva/paired perseed](../03_insight_exp/README.md).

### 4.6 Waktu fase pertama — tercatat, bukan tebakan

| Fase | Detik |
|---|---:|
| Prepare seed42 | 71.913 |
| A training/results.csv | 53.124 |
| B training/results.csv | 67.186 |
| Pasangan train+evaluation | 139.930 |
| ProbeA,fresh process | 14.138 |
| ProbeB,fresh process | 14.667 |

Metrik fase ini berbeda scope, jangan dijumlah ganda: A/B results.csv ada di dalam pair wall; probe wall mencakup startup/extraction/eval. Semua waktu lengkap ada di [laporan Tahap1b](../01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_stage1b_report.md).

## 5. Yang gagal / terblokir / belum terbukti

| Item | Penyebab/bukti | Penanganan dan status sekarang |
|---|---|---|
| Detection training |0 verified positives; missing signer/session metadata | Gate dipertahankan; tidak ada Detection training/mAP yang diklaim |
| Membuktikan NI dari pilot | CI melintasi batas kandidat−.02; noise seed cukupbesar | Inconclusive; bukan eksperimen software gagal. Semua seed tetap dilaporkan |
| ProbeA seed42,attempt1 | Ultralytics sudah terimport dengan settings directory worklain | Gagal sebelum model loading/optimization; outputasli dipertahankan,attempt2 di folder baru selesai |
| Workflow multi-work dalam satu proses | Ultralytics settings global perproses | Source locked tidak diedit; eksekusi sukses memakai fresh subprocess perfit/evaluation. Pemanggilan multi-work in-process lama tetap tidak direkomendasikan |
| Runner preflight v1 | Assertion menghitung subfolder settings kosong sebagai file | Diperbaiki menjadi cekfile; v1 dipertahankan,v2 sukses;0 fit pada kegagalanpreflight |
| Tes gabungan pertama Tahap1a | Legacy mock sys.modules memicu reimport torchvision/C++ operator registration | Preload dependency pada collection tes; suitefinal92lulus |
| Render notebook hasil saat exit | IPython mencoba menulis history DB readonly | Rendering/output notebook tersimpan; ini warning cleanup non-training, bukan kegagalan fit |
| Junction dalam sandbox awal | Access denied | Junction lokal berhasil lewat izin workspace yang sesuai; target/path diverifikasi |
| File symbolic link untuk reorganisasi | Windows memerlukan administrator | Tidak dipakai; hardlinks +runtimecompat dipakai, diuji untuk mendeteksi atomic replacement/divergence |
| Pemeriksaan directoryID train saat organisasi | Metadata top-directoryID berubah pada pemindahan; bukan filevideo | Seluruh2.800 filevideo mempertahankan fileID/size dan mirror; validasi fokus pada file, bukan identitas topfolder |
| Launcher Jupyter melalui subcommand | `python -m jupyter lab` tidak menemukan entry point `jupyter-lab`,meski modul terpasang | Launcher diganti `python -m jupyterlab`; versi4.2.6 dan flag notebook-dir terverifikasi; tidak install/upgrade dependency |
| Penyusunan indeks | Satu SyntaxError generator sebelum write | Diperbaiki; indeks/tautan tervalidasi; tidak mengubah source/data |

Log kegagalan training/runner: [failure history](../01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_failures.json); [sesi notebook historis](../02_exp/log_sesi/exploratory_development_stage1b_user_20261103_v1_session/exploratory_events.jsonl).

## 6. Yang berhasil diverifikasi

- Baseline sebelum Tahap1a: **81 tes lulus**, 100.083detik.
- Akhir Tahap1a: **92 tes lulus**, 93.294detik. Bukti: [provenance tes1a](../01_hasil/tahap1a/exploratory_development_stage1a_20261103/exploratory_stage1a_provenance.json).
- Tahap1b:10 fixed-last checkpoints dan10 last heads,epoch1..30,steps A690/B900/head690,seed/hashes/source locks. Bukti: [verifikasi model](../01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_final_verification.json).
- Organisasi awal:48 folder dipisah; namespace checkpoint lama tetapada lewat junction. Bukti: [verifikasi awal](../02_exp/organisasi_20261104/organization_verification.json).
- Organisasi lengkap: **58 file dipindah**, **2800 filedataset tetap inode/size sama**, **182 hashbeku tetap**,10 checkpointterverifikasi. Bukti: [verifikasi lengkap](organisasi_20261104_lengkap/verification.json).
- Setelah organisasi: **29 tes terarah lulus,18.120detik** (Stage0/protokol,parameterisasi,paired,power). Bukan klaim menjalankan ulang seluruh suite/notebooktraining.
- Launcher dicek memakai runtime lama; rollbackCheckOnly lulus tanpa memindah/menghapus data. Guardhardlink lolos kasusnormal dan menolak atomic replacement sintetis.
- Protokol kandidat SHA256 tetap `24e21743168f3457933f0f17db76e52d420b5f005da774a7f9230c71f1ac7a70`.

## 7. Yang belum dilakukan / ditunda

- Addendum berikutnya yang mengunci delta/H2/domain konfirmatori dan keputusan power; tidak dibuat dalam fase organisasi/tracking.
- Tahap2: CV tuning, grid selection, faktorial C/D, suplemen step-matched; belum ada hasil.
- Tahap3: full-data final multi-seed, final freeze, satu batch external scores; belum ada hasil.
- Semantic adjudication lengkap, signer/session metadata/group independence; belum terpenuhi.
- Draft conference paper final yang menyelaraskan seluruh hasil; draft historis tersedia tetapi tidak diedit pada faseini.
- Demo/webcam/real-time application bukan deliverable. Jangan mengklaim FPS/latency aplikasi dari timing training/forward.

## 8. HIPOTESIS vs USULAN — keputusan berikutnya

**HIPOTESIS, belum terbukti:** context cues/weak labels/overfit mungkin menjelaskan accuracytinggi dan perbedaanA/B. Masking/probe tidak membuktikan mekanisme linguistik atau sebab.

**USULAN, belum diotorisasi:** tinjau hasil seluruhseed dan keterbatasan; putuskan margin/H2/domain serta kebutuhan data independen; tulis addendum sebelumTahap2/3; validasi folds/grid/budget dan beri persetujuan fase terpisah. Tidak mengejar seed pemenang atau tuning Kaggle.

## 9. Peta akses sekarang

- [Hasil](../01_hasil/README.md) · [Eksperimen](../02_exp/README.md) · [Insight](../03_insight_exp/README.md).
- [Dataset](../04_dataset/README.md) · [Kode/tests](../05_kode/README.md) · [Notebook](../06_notebook/README.md).
- [Dokumen](README.md) · [Review](../08_review/README.md) · [Environment](../09_environment/README.md).
- Launcher: `05_kode/run.ps1`; notebook beku dijalankan dari namespace `.compat_runtime/` agar cwd/snapshot tetap cocok.
- Hardlinks bukan salinan data. Jangan menghapus namespace/alias atau mengedit data beku; editor atomic-save dapat memisahkan link, dan launcher akan menolak divergence tanpa menghapus salah satu versi.
- [Manifest/rollback terbaru](organisasi_20261104_lengkap/README.md).

## 10. Aturan pembaruan tracker

Tambahkan setiap pekerjaan/percobaan baru dengan status,scope/seed/config,hasil aktual,error jika ada,bukti path,dan gate berikutnya. Pertahankan sejarah kegagalan. Pisahkan FACT/HYPOTHESIS/PROPOSAL. Angka hanya dari artefak/perhitungan, bukan ingatan.

| Tanggal | Aktivitas | Status | Hasil/error | Bukti | Tindak lanjut |
|---|---|---|---|---|
|2026-11-04–05| Segmentasi penuh +tracking | SELESAI | Tidak ada training baru; file/data/hash dipertahankan | Manifest dan verification pada organisasi lengkap | Menunggu keputusan riset berikutnya; tetap berhenti sebelumTahap2 |
