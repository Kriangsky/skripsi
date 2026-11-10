# exploratory/development — laporan Tahap 1a, 2026-11-03

## FACT — status dan cakupan

Baseline A: 81/81 tes lulus, 100.083 detik. Akhir E: 92/92 tes lulus, 93.294 detik.
Perintah: `.\.venv-bisindo\Scripts\python.exe -m unittest discover -v`.
Log akhir: `exploratory_tests_final.log`. Seluruh tes lama tetap dijalankan. Unit tests memakai fixtures/mock,
bukan pelatihan model BISINDO. Satu percobaan suite gabungan sebelumnya menemukan reload torchvision setelah
mock sys.modules lama; preload dependency pada koleksi tes mengisolasi registrasi operator C++ tersebut.
Percobaan final di atas lulus, tanpa mengubah dependency/environment.

Model fits baru=0, optimizer steps penelitian baru=0. External images/scores dibaca=0.
Read-only pada pilot selesai; semua output di direktori exploratory baru ini. Protokol kandidat/hash,
addendum, draft paper, video, split, notebook Detection, dan hasil run lama tidak diedit.
Hash kandidat valid: `24e21743168f3457933f0f17db76e52d420b5f005da774a7f9230c71f1ac7a70`.
Skema notebook divalidasi dengan nbformat; seluruh sel Python dikompilasi. Notebook tidak dieksekusi end-to-end;
`verify_notebook.py` Detection tidak digunakan. Training default False, tanpa approval file yang dibuat sesi ini.

## FACT — perubahan file

| File | Perubahan |
| --- | --- |
| classification_research.py | Gate Stage1b sesuai keputusan peneliti; callback train top1/loss; source fingerprint mencakup modul baru; cek epoch1..N dan partial evaluation |
| power_precision.py | API tabel exploratory: external dan CV, margin skenario .02/.03/.05; API lama tetap |
| classification_diagnostics.py | Union landmark mask15%/114 sebelum letterbox; mask/coverage/provenance; eval strata; frozen eval embeddings; head6 fixed SGD30; no-augmentation curve callback |
| classification_workflow.py | Fungsi notebook prepare/multiseed/train+probe/evaluation/paired/masking/curves; flag serta approval wajib; tanpa overwrite/resume |
| BISINDO_Classification_Vocabulary_Expansion.ipynb | Notebook baru exploratory/development; memanggil modul, RUN_TRAINING=False |
| test_exploratory_development.py | 11 tes baru: mode/RNG/weights, masker, strata unavailable, provenance, gate, notebook, parameter smoke dan power |
| exploratory_development_plan.md | Rencana implementasi dan batas scope |

`analysis_paired.py` dipakai ulang tanpa mengubah API/source. Parameterisasi existing diteruskan melalui SIZE,
CONFIG private namespace, dataset transforms, release/settings fingerprints, probe, fixed completion/evaluation,
judul plot dan conclusion. Sampling tetap42, training seeds42..46. Modul legacy/global original tidak dimutasi.
Perintah prepare/probe/train/evaluate legacy tetap tersedia; entry point baru melalui classification_research.py.

Review independen menemukan hardcoded smoke224 dan split masking yang belum terikat; keduanya diperbaiki
beserta regression tests. Coverage CSV kini diikat hash ke provenance/manifest, split/GT/video dibandingkan.

## FACT — paired pilot existing, common6

| metric | A | B | delta_B_minus_A | ci95_low | ci95_high | lower_one_sided95 | bootstrap_sd |
| --- | --- | --- | --- | --- | --- | --- | --- |
| macro_f1 | 0.9317756044925641 | 0.9300418783462919 | -0.0017337261462722076 | -0.025679347857453294 | 0.01566455224991803 | -0.020841948126293985 | 0.010670351600781183 |
| accuracy | 0.9333333333333333 | 0.925 | -0.008333333333333304 | -0.025 | 0.0 | -0.025 | 0.008088024881005169 |

Counts both-correct111, A-only1, B-only0, both-wrong8; union errors9; exact two-sided McNemar p=1.
Wilson accuracy CI, union errors dan error per kelas: `exploratory_paired/`.
Bootstrap video berpasangan, stratified GT,10000 replikasi,RNG20261103. B mempertahankan delapan output.
Full8 tetap tugas terpisah; tidak membandingkan B/full8 dengan A/common6 sebagai tugas identik.
Lower one-sided95% d=-.020841948 menyilang batas kandidat -.02: pilot ini tidak memutuskan NI.
Hanya seed42 teramati; reproduksi seed42 bukan seed tambahan. Tidak ada estimasi varians seed empiris.

## SIMULATION — power/presisi, bukan hasil external

Skenario ilustrasi berikut: true macro-F1 d=0, correctness discordance5%, paired seed SD1pp,
lima pasangan seed, video draw sama lintas seed. Generator mempertahankan ekstra output B.
Semua kombinasi discordance2/5/10%, seedSD0/1/2pp, margin2/3/5pp berada di CSV.
Lebar CI dua sisi95%; peluang NI memakai lower one-sided95% normal approximation.

| domain | n_common6 | ni_margin | ci95_width_pp | ni_pass_probability | monte_carlo_se |
| --- | --- | --- | --- | --- | --- |
| external_common6 | 300 | 0.02 | 5.227237930196527 | 0.4434 | 0.0049678611091696 |
| external_common6 | 300 | 0.03 | 5.227237930196527 | 0.7258 | 0.004461102554302 |
| external_common6 | 300 | 0.05 | 5.227237930196527 | 0.981 | 0.0013652472303579 |
| internal_CV_one_fold | 360 | 0.02 | 4.825164055086308 | 0.4931 | 0.0049995238773307 |
| internal_CV_one_fold | 360 | 0.03 | 4.825164055086308 | 0.7884 | 0.0040844270099978 |
| internal_CV_one_fold | 360 | 0.05 | 4.825164055086308 | 0.9915 | 0.0009180277773575 |
| internal_CV_OOF_optimistic | 1800 | 0.02 | 2.6673910993597527 | 0.9029 | 0.0029609388713717 |
| internal_CV_OOF_optimistic | 1800 | 0.03 | 2.6673910993597527 | 0.9976 | 0.0004893097178679 |
| internal_CV_OOF_optimistic | 1800 | 0.05 | 2.6673910993597527 | 1.0 | 0.0 |

CV satu fold mengasumsikan5-fold atas300 video internal/kelas:60 held-out/kelas (n360 common6), OOF n1800.
OOF di tabel hanya referensi optimistis: training folds overlap, selection/tuning dan signer/session correlation
belum dimodelkan. Ini tidak membuktikan confirmation independen. Asumsi video independen dapat terlalu optimistis.
Oracle delta-method + Gaussian paired seed mean bukan power exact dari hierarchical bootstrap akhir.
Marginnya hanya skenario; tidak ada delta/threshold H2/domain yang dipilih atau diubah.

H2 n50/kelas, threshold tetap .80. Referensi satu classifier/seed errors berkorelasi sempurna,
Clopper-Pearson lower one-sided95% >.80 memerlukan45/50 benar:

| true_accuracy_assumed | required_correct | probability_pass_one_class | probability_pass_both_if_independent |
| --- | --- | --- | --- |
| 0.8 | 45 | 0.0480272193707337 | 0.0023066138004845 |
| 0.85 | 45 | 0.219353335401438 | 0.0481158857517357 |
| 0.9 | 45 | 0.6161230077242774 | 0.37960756064721 |
| 0.95 | 45 | 0.9622238270102226 | 0.9258746932661988 |
| 0.98 | 45 | 0.999521782785786 | 0.9990437942632762 |

Peluang dua kelas mengasumsikan independensi antar kelas; Frechet bounds ada di CSV. Ini bukan exact power
hierarchical CI lima seed. Jangan menganggap5x50 sebagai250 video independen.

## FACT — masking dan kurva

Cache internal menghasilkan640 mask dan640 gambar masked224, tanpa inference landmark baru atau training:

| split | stratum | n | mean_coverage |
| --- | --- | --- | --- |
| train | landmarks_available | 480 | 0.057484155477971345 |
| val | landmarks_available | 160 | 0.0641646166018964 |

Tidak ada landmark-unavailable pada release pilot aktual ini; implementasi/tes tetap mempertahankan strata itu.
Mask adalah union semua landmark dengan padding15% pada union bounds, clipped ke frame, fill114 sebelum letterbox.
Landmark bukan GT kelengkapan tangan. Context-only masih dapat memuat wajah/tubuh, bukan background murni.
Masking evaluation pada model baru belum dijalankan; setelah izin akan melaporkan keseluruhan dan strata,
dengan reference chance A6=1/6 dan B8=1/8, serta caveat distribution shift/imbalance strata.
Kurva historical tersedia di `exploratory_historical_curves/`; train top1 lama unavailable, tidak direkayasa.
Callback baru merekam actual-training-weights top1/loss no augmentation dan memulihkan mode semua module/RNG.
Val Ultralytics memakai EMA: perbedaan sumber weights diungkap pada label kurva.

## HYPOTHESIS — belum terbukti

Cue konteks/background, noise label weak video, atau overfit dapat menjelaskan hasil pilot.
Masking, frozen probe dan kurva adalah diagnosis yang diusulkan; bukan bukti sebab atau jaminan B menang.

## PROPOSAL — opsi keputusan

| Opsi | Konsekuensi |
| --- | --- |
| Pertahankan margin ilmiah dan external n300 | Terima kemungkinan NI inconclusive; jangan memilih seed/checkpoint |
| Kumpulkan external independen lebih banyak melalui protokol baru | Presisi berpotensi membaik; biaya koleksi/audit dan freeze baru |
| CV internal sebagai development, external terpisah | Data development lebih banyak; selection bias/dependence tetap, bukan confirmation untouched |

Tidak memilih opsi untuk peneliti; delta dan ambang H2 tidak diubah.

## PROPOSAL — Tahap 1b, belum disetujui/dijalankan

10 full runs baru: A/B xseed42,43,44,45,46,224px,30epoch. Seed42 dilabeli replikasi.
Init tiap model independen dari pretrained pinned yang sama; main evaluation fixed last.pt, tanpa resume/picking.
10 linear head fits: backbone A/B setiap seed frozen eval, head baru6 kelas; train lama360 saja,
SGD lr.01/momentum.937/decay.0005,batch16,30epoch tanpa augmentation, last head; val hanya sesudah fit.
Semua seed/failure harus masuk tabel. Lanjutkan masking eval, per-seed paired bootstrap/McNemar dan kurva,
mean±SD, hierarchical CI (seed pairs + shared stratified video draw), conditional video/seed variability.
Pooled McNemar seed-video diberi label deskriptif karena repeated videos; jangan menyatakan pooled trials independen.

### Waktu berbasis artefak/ukur, bukan tebakan

| model | batch | repeats | seconds | seconds_per_batch |
| --- | --- | --- | --- | --- |
| A | 16 | 10 | 0.0449846999999863 | 0.0044984699999986 |
| B | 16 | 10 | 0.0537967000000207 | 0.005379670000002 |

Smoke di atas hanya FP32 resident GPU forward batch16, satu untimed warmup lalu10 repeat/CUDA synchronize,
bukan benchmark deployment atau training. Tidak mencakup decode/IO/backward/SGD/head fits.
Waktu training historis dari results.csv: A=37.4169s,
B=47.5833s. Ekstrapolasi5 pasangan historis=425.001s
(~7.08menit), hanya baseline training lama, bukan total baru dengan instrumentasi/probes/IO.
**Estimasi total terukur untuk10 full+10 head fits belum tersedia.** Mengukurnya memerlukan smoke training,
yang dilarang pada fase ini. Tidak membuat angka pengganti dari inference. Setelah persetujuan Tahap1b,
ukur fase fit/instrumentation/probe pertama dan tampilkan waktu tercatat; jangan memilih hasil smoke untuk analisis utama.

## Gate sebelum Tahap 2 atau 3

1. Addendum peneliti mengunci delta, threshold H2, domain evidence, inferensi/CI, multipel gatekeeping dan keputusan power;
   kandidat PREREGISTRATION.md/hash tetap utuh. Tidak dibuat dalam fase ini.
2. Persetujuan eksplisit subfase/budget; tidak auto-launch. Gate kode Stage1b saja kini cukup scoped approval sesuai keputusan chat;
   instruksi lama Addendum02 sebelum semua training adalah histori yang disupersede untuk Stage1b saja.
3. Semua lima seed dan replikasi42 dilaporkan. Terapkan stop bagian4: leakage/corruption/mapping/hash/API/GPU → stop;
   SD model seed >=|mean d| atau CI melintasi -delta → pilot tidak memutuskan; usulkan CV tanpa seed chasing.
4. Fold/video/group integrity dan kapasitas internal disahkan; signer/session metadata atau keterbatasannya eksplisit.
   CV reused/tuned tetap development; split_lock lama tidak dipindah/ditimpa.
5. Grid terbatas, selection rule/tie-break, semantic-safe transforms, kapasitas n dan budget CV dikunci sebelum skor.
   Tidak ada unsupported API flag, oversampling, Kaggle tuning atau checkpoint picking.
6. Sebelum final/external: freeze config/manifest/frame policy/source+tests/deps/checkpoints, approval satu batch scoring
   semua seed, common6 subset full8 dipisah; jangan menyebut Kaggle belum pernah diproses.

BERHENTI sebelum training. Menunggu persetujuan eksplisit Tahap1b. Draft paper/addendum tidak dikerjakan.
Risiko tersisa: weak labels, signer/session dependence, masked distribution shift, power conditional assumptions;
full/head production fits dan eksekusi notebook end-to-end belum diuji karena belum diizinkan.
