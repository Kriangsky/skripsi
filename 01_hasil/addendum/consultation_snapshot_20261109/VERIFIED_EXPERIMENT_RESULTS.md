# Verified experiment results — consultation 2026-11-09

## 1. Executive scientific summary

Snapshot lokal read-only; hanya menulis folder ekspor baru. Semua hasil model terukur di sini adalah **DEVELOPMENT internal**, bukan final/external. Sepuluh full Stage1b fits dan sepuluh frozen-backbone head fits tersedia. A common6 mean macro-F1=0,9398997391; B=0,9311704349. Mean paired Δ=−0,008729304229932344; CI95% terbaru [−0,030222818611814216; +0,007612139862988273]. CI mencakup nol: tidak menetapkan superiority, equivalence, non-inferiority, atau kepastian penurunan. Menambah halo/tidur masih relevan untuk meneliti trade-off kelas, bukan sudah terbukti memperbaiki diskriminasi.

**Evidence levels:** VERIFIED_FROM_ARTIFACT = saved numerical output dibaca dan cross-check terbatas (prediksi/metrik dihitung aritmetika; CI tidak direplikasi). REPORTED_NOT_INDEPENDENTLY_VERIFIED = laporan/receipt dibaca tetapi operasi aslinya tidak diulang. PLANNED_NOT_EXECUTED = rencana tanpa hasil. BLOCKED = prasyarat belum terpenuhi. UNKNOWN = bukti kurang. Hash membuktikan identitas byte, bukan kebenaran semantik atau execution history lengkap. Historical PASS tidak otomatis verifikasi independen.

## 2. Data inventory and provenance

| source | split | class_name | n |
| --- | --- | --- | --- |
| test | test | apa_kabar | 50 |
| test | test | bingung | 50 |
| test | test | halo | 50 |
| test | test | kamu | 50 |
| test | test | marah | 50 |
| test | test | saya | 50 |
| test | test | senang | 50 |
| test | test | tidur | 50 |
| train | train | apa_kabar | 240 |
| train | train | bingung | 240 |
| train | train | halo | 240 |
| train | train | kamu | 240 |
| train | train | marah | 240 |
| train | train | saya | 240 |
| train | train | senang | 240 |
| train | train | tidur | 240 |
| train | val | apa_kabar | 60 |
| train | val | bingung | 60 |
| train | val | halo | 60 |
| train | val | kamu | 60 |
| train | val | marah | 60 |
| train | val | saya | 60 |
| train | val | senang | 60 |
| train | val | tidur | 60 |

TASK1 `audit.json: observed` mencatat 2.400 internal, 400 external, train/val1.920/480, external common6/full8=300/400, 2.800 physical verified videos. Status angka receipt: VERIFIED_FROM_ARTIFACT sebagai isi saved audit; physical integrity saat ini **REPORTED_NOT_INDEPENDENTLY_VERIFIED** karena 45.250 checks tidak diulang. `class_split_counts` di atas berasal langsung JSON; per kelas internal300/train240/val60/external50. Split terkunci dengan SHA256 `03eb34d91c6764d14052d2d8f99f6996d5b54602d50d533b47b9246ad32f706e` menurut receipt. Lihat `audit_report.md` §1–5 dan `duplicate_impact.json`.

Stored audit: 0 duplicate video-byte groups; tiga pasangan candidate JPEG/RGB identik f7/f10/f14 pada dua video senang_223/224, keduanya TRAIN. Development hanya memakai224/f48, bukan frame duplikat. Tidak ada cross-split decoded identity pada scope audit. Tidak membuktikan tidak ada near-duplicate, overlap frame lain, signer leakage. Tidak menghapus/memindah data.

Kaggle citation lokal menunjuk BISINDO40 Kata mp4, bukan acquisition receipt. Versi, hash arsip asal, log unduh, subset/rename provenance, lisensi belum terikat ke400 file. Signer/session internal0/2400 dan external0/400 nonempty. Preprocessing eksternal v2/v3 sudah terjadi; kelengkapan riwayat scoring/tuning belum diattest. Jangan menyebut untouched atau signer-independent. `provenance.json` dan laporan TASK1 menyimpan bukti dokumenter, bukan autentikasi asal dataset.

## 3. Preprocessing and frame-selection history

TASK1 `frame_provenance`: 2.800 cache records, 33.250 kandidat; v3 positive-selected0, seluruh2.800 needs_review. Primary fractions20–80%, fallback5/10/15/85/90/95%. Pemilih teknis `pilot_study.choose_candidate` mempertimbangkan readable/flags/focus/frame index/candidate ID, bukan output model; tidak membuktikan representasi gesture valid. Development640 frame: semantic not_assessed640, flagged497, eligible143. Semua retained.

Full-frame aspect-preserving letterbox224×224, padding gray114, BGR→RGB, RGB/255; tidak crop/flip. Override ControlledTrainer mengganti Compose Classification default. Task2 effective configuration mencatat Resize224/ToTensor/identity Normalize; tidak ada augmentasi stokastik. `augment=False` saja tidak cukup sebagai bukti. Config/execution/source dibaca; transform per batch historis dan hash framework historis tidak tersedia. Tidak mereplay transform atau model dalam snapshot.

Human pilot berbeda: 160 internal training-pool frame, seed42, 20/kelas, technical eligible43/160 (26,875%), inspectable160/160. Ini bukan pemilihan ulang frame berdasarkan hasil review. Full-data final manifests internal/external masih null (`analysis_plan.json`, audit). Development manifests tidak dipromosikan final.

## 4. Human pilot

| reviewer_id | n | correct_n | accuracy_all | ambiguous_n | invalid_n | wrong_class_readable_n | accuracy_all_ci95_low | accuracy_all_ci95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reviewer_1 | 160 | 129 | 0.80625 | 27 | 0 | 4 | 0.738126102390117 | 0.8600131030673362 |
| reviewer_2 | 160 | 138 | 0.8625 | 12 | 4 | 6 | 0.8005890571519036 | 0.9074124513487551 |

Per kelas, denominator20 setiap reviewer; source `02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/per_class_results.csv`, row reviewer_id/class_name, columns correct_n/accuracy_all.

| reviewer_id | class_name | n | correct_n | accuracy_all |
| --- | --- | --- | --- | --- |
| reviewer_1 | saya | 20 | 16 | 0.8 |
| reviewer_1 | kamu | 20 | 17 | 0.85 |
| reviewer_1 | senang | 20 | 17 | 0.85 |
| reviewer_1 | bingung | 20 | 18 | 0.9 |
| reviewer_1 | marah | 20 | 17 | 0.85 |
| reviewer_1 | apa_kabar | 20 | 9 | 0.45 |
| reviewer_1 | halo | 20 | 16 | 0.8 |
| reviewer_1 | tidur | 20 | 19 | 0.95 |
| reviewer_2 | saya | 20 | 19 | 0.95 |
| reviewer_2 | kamu | 20 | 18 | 0.9 |
| reviewer_2 | senang | 20 | 14 | 0.7 |
| reviewer_2 | bingung | 20 | 19 | 0.95 |
| reviewer_2 | marah | 20 | 20 | 1.0 |
| reviewer_2 | apa_kabar | 20 | 14 | 0.7 |
| reviewer_2 | halo | 20 | 16 | 0.8 |
| reviewer_2 | tidur | 20 | 18 | 0.9 |

Agreement135/160=0,84375; expected agreement0,106328125; Cohen kappa0,8251595419179998, dihitung ulang dari paired sample IDs dua responses JSON dan cocok `inter_reviewer_agreement.json` keys. Confusion CSV diagonal/sum cocok129/160 dan138/160. Tidak teridentifikasi either31/both12; pooled apa_kabar↔halo errors7. Response exports dan private mapping dihash lokal; hanya hasil agregat disalin, tanpa private mapping atau identitas signer.

Reviewer1: salah kelas4, ambiguous27, invalid0; reviewer2 salah kelas6, ambiguous12, invalid4. Akurasi subset identifiable tinggi tidak menggantikan denominator semua160 karena selection bias. `apa_kabar` paling rendah untuk reviewer1 dan termasuk terendah reviewer2, bukan bukti temporal causation. Jumlah orang, keahlian linguistik dan independensi reviewer tidak diautentikasi dari ekspor. Tidak ada expert-certified full-label review. Wilson CI tercatat, tidak direplikasi; korelasi signer dapat mempersempit CI secara semu.

## 5. Classification training pilot

Historical `experiment_yolo26_classification_pilot` dan notebook reproduction20261102: A accuracy112/120=0,9333333/F1=0,9317756; B common6=111/120=0,925/F1=0,9300419; B full8=150/160=0,9375/F1=0,9354641. Sumber masing-masing `evaluation/comparison_metrics.csv`, rows model/scope. Nilai cocok saved rows; jangan menghitung reproduction sebagai seed independen. `pilot_conclusion.md` mencatat apa_kabar20/20 kedua model, halo19/20 (satu→apa_kabar), technical thumbnails160 tetapi expert-verified0/semantic-unassessed640. Human pilot dan model memakai pool/split berbeda; gap bukan bukti model memahami bahasa lebih baik.

Detection pilot `experiment_yolo26_training_pilot/pilot_conclusion.md`: NOT TRAINED, verified positives0/unreviewed2400; tidak ada hasil pembanding detection yang completed. Classification dipilih karena target satu label/frame, bukan hasil benchmark keunggulan terhadap detection/temporal.

## 6. Stage1b five-seed experiments

Seluruh sepuluh run berikut; common6 N120=20/kelas pada validation yang dipasangkan, B full8 N160. A train360/B480=60/kelas. Lima sampling manifests memakai sampling seed42, training seeds42–46; bukan lima dataset independen. Dua seed42 reused, delapan fits tambahan. Tidak cherry-pick seed.

YOLO26n-cls ImageNet, init SHA256 `0dd6f8dbc448870ac98a3cbb7156f923f7ce21fed3755d4019169ffffd279e81`; fresh per arm, bukan B meneruskan A. 224²,30 epochs,batch=nbs16,SGD lr0.01,momentum.937,Nesterov true,decay.0005,linear schedule lrf.01,warmup3 epochs,patience0,resume false,workers0,amp false,deterministic true. A23 batches/epoch=690 steps; B30=900. Equal epochs tidak menyamakan compute. Epoch30 LR tercatat.00043, bukan.0001. Checkpoint endpoint last.pt EMA30 menurut records, bukan best.pt. Semua sepuluh bytes last.pt dihash ulang dan cocok trained_models/effective_configuration. Metadata checkpoint tidak di-unpickle ulang; `CHECKPOINT_HASHES.json` membedakan verifikasi byte dan metadata historis. Stripped epoch−1/optimizer None bukan nol training; saved curve memiliki30 rows. Initial checkpoint pin sama pada seluruh records; metadata ImageNet/upstream bukan autentikasi independen baru.

Tabel: accuracy/precision/recall/F1 pada common6; full8_F1 hanya B. Macro precision/recall adalah rata-rata aritmetika saved per-class CSV (derivative eksplisit), bukan metrik training framework. Seconds adalah recorded fit duration, tidak termasuk prepare/inference; bukan benchmark efisiensi baru.

| model | seed | train | val | steps | seconds | accuracy | macro_precision | macro_recall | macro_F1 | full8_F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 42 | 360 | 120 | 690 | 53.1245 | 0.9333333333333333 | 0.9398133013693655 | 0.9333333333333333 | 0.9317756044925641 | N/A |
| B | 42 | 480 | 160 | 900 | 67.186 | 0.925 | 0.947957839262187 | 0.925 | 0.9300418783462919 | 0.9354640789917368 |
| A | 43 | 360 | 120 | 690 | 54.3363 | 0.9333333333333333 | 0.9435786435786435 | 0.9333333333333333 | 0.9326249749869775 | N/A |
| B | 43 | 480 | 160 | 900 | 98.6959 | 0.925 | 0.9507936507936507 | 0.925 | 0.9308942013767204 | 0.9298634501033882 |
| A | 44 | 360 | 120 | 690 | 71.4346 | 0.95 | 0.9570707070707071 | 0.95 | 0.9475002122060946 | N/A |
| B | 44 | 480 | 160 | 900 | 105.427 | 0.9333333333333333 | 0.9456654456654456 | 0.9333333333333333 | 0.9319470937339606 | 0.9428515916307662 |
| A | 45 | 360 | 120 | 690 | 54.0118 | 0.9583333333333334 | 0.9631093544137023 | 0.9583333333333334 | 0.956626061277224 | N/A |
| B | 45 | 480 | 160 | 900 | 69.782 | 0.925 | 0.938053613053613 | 0.925 | 0.9238073337649323 | 0.9305668272045128 |
| A | 46 | 360 | 120 | 690 | 54.0315 | 0.9333333333333333 | 0.9425465838509317 | 0.9333333333333333 | 0.9309718426908904 | N/A |
| B | 46 | 480 | 160 | 900 | 70.1394 | 0.9416666666666667 | 0.9491341991341992 | 0.9416666666666667 | 0.9391616672821838 | 0.9422182700801593 |

Lokasi lengkap tiap row pada `STAGE1B_TEN_RUNS.csv: source`; source row model/scope, columns n/accuracy/macro_f1. Precision/recall: sibling `per_class_metrics.csv`, rows model/scope/class_id. Steps: sibling `runs/model_*_execution.json:actual_optimizer_steps`; duration: `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_all_fits.csv`, kind=full/model/seed. Hash setiap checkpoint:

| model | seed | sha256 |
| --- | --- | --- |
| A | 42 | 03727737a647405bdc6154c925c4575f2b77bf45f48a2ef21a92d02d044f54b7 |
| B | 42 | e544a04b4798f1d76a2096c844fd768bd53e50aa2fa4e1c17a419eef3fbc67bd |
| A | 43 | 20533ef76136e29f105f96bcd9aeda09d530a2bcc65198beaf94bf9df4ebe5e7 |
| B | 43 | ddcd40f906c52f6228689eaed78235ece77b7372ad5de819c47976d5b229dd79 |
| A | 44 | 8b5af75865038f6c14a1dcd22cb9331495f0120ae92ff1b45ae6fe0d83f18db6 |
| B | 44 | 79fa00f026422a7d41dd51cf6e49f251a96418be55bea873076a0e00d71a8e80 |
| A | 45 | e03c41f67d1f64c84eaaedba9ddd51c1cd0c0384fb412a44b81d957b025c53ba |
| B | 45 | bbe8dac6bad0e452ef18543b795ae816008bf1673bbe99bf9e13aa6acba7bc76 |
| A | 46 | af538ffb291a31666a4861ce309714784eb1ed795d6896540bc828af86342242 |
| B | 46 | f5bb4d80995be7e668f4153bada040beca84467d968dce04e3e866e0cdb90fdb |

Per-class precision/recall/F1/TP/FP/FN seluruh seed tersedia di RESULTS_SUMMARY.csv dan original per_class CSV; confusion matrices A common6, B common6/full8 disalin untuk semua seed (bukan satu seed terbaik). Saved predictions dibaca dan N/correct/per-class TP/FP/FN/F1 dicek ulang tanpa model inference. Framework epoch validation dapat memakai best/EMA yang berbeda dari fixed last endpoint; jangan mengganti hasil endpoint dengan best score.

## 7. Paired A/B statistics

| seed | n | macro_f1_A | macro_f1_B | delta | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 120 | 0.9317756044925641 | 0.9300418783462919 | -0.0017337261462722076 |  |  |
| 43 | 120 | 0.9326249749869775 | 0.9308942013767204 | -0.0017307736102570859 |  |  |
| 44 | 120 | 0.9475002122060946 | 0.9319470937339606 | -0.015553118472134031 |  |  |
| 45 | 120 | 0.956626061277224 | 0.9238073337649323 | -0.03281872751229176 |  |  |
| 46 | 120 | 0.9309718426908904 | 0.9391616672821838 | 0.00818982459129336 |  |  |
| mean | 120 | 0.9398997391307502 | 0.9311704349008177 | -0.008729304229932344 | -0.030222818611814216 | 0.007612139862988273 |

Source `01_hasil/addendum/development_interference_20261105/common6_primary_results.csv`, rows seed42–46/mean, columns di atas; VERIFIED_FROM_ARTIFACT untuk saved estimates/CI, bukan independen bootstrap replication. Mean delta dan SD aritmetika diperiksa ulang. `analysis_details.json: seed_delta_sd`=0,01589842145086347 (ddof1). `exploratory_model_summary.csv`: A F1 SD0,011577652345406817; B0,0054778690453032575; B full8 mean F1=0,9361928436021125/SD0,006182415991676225. Accuracy means A0,9416667/B common6 0,93/B full8 0,9375. Five observed seeds memberi informasi variasi terbatas.

**Dua CI historis berbeda, keduanya dipertahankan:** summary Stage1b `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_exploratory_multiseed/exploratory_multiseed_summary.json` key ci95=[−0,03010145130566371; +0,007667945291901918], lower one-sided95=−0,02558950457581278. Output development20261105 row mean=[−0,030222818611814216; +0,007612139862988273], lower=−0,02533392899552548. Mean dan SD Δ sama. Dugaan penyebab: implementasi berbeda dapat menghasilkan realisasi bootstrap berbeda walau10.000/RNG20261103 sama; seed RNG tidak menjamin indeks/draws sama di implementasi berbeda. Tidak dilakukan replay untuk mengisolasi penyebab numerik setiap selisih; jangan menyebut salah satunya koreksi pasti. CI20261105 dipakai sebagai ringkasan analysis engine terkini, yang lama tetap dapat diperiksa.

`external_evaluation_report.md` bernama external tetapi header development dan provenance menyatakan internal. Bootstrap yang didokumentasikan: paired video resampling dalam strata GT, draw video yang sama untuk A/B dan semua seeds, paired training seeds diresample independen; F1 dihitung ulang setiap replicate sebelum mean Δ. N120 unique videos,5 seed pairs; **bukan N600 independen**.10000 reps, RNG20261103. `confirmatory_statistics.py` lines30–50 membangun video indices lalu seed multinomial weights. Tidak dieksekusi kembali.

Historical video-only CI=[−0,02017397590776844;−0,0000416927246196] dan seed-only CI=[−0,02245936208819712;+0,002236875163160068], conditional sensitivities, bukan pengganti hierarchical. Pooled McNemar counts A-only8/B-only1, p0,0390625 hanya deskriptif karena repeated videos, tidak inferensial. Stage1a seed42 McNemar exact p1, discordance1/120; sumber `01_hasil/tahap1a/.../exploratory_paired/exploratory_analysis.json` disertakan. Histori Stage1a “only seed42 exists” berlaku saat itu, bukan kontradiksi lima-seed sesudahnya.

## 8. I1/I2/I3 interference results

I1: recall mean per GT class (20 unique videos/class,5 paired seeds); file class_interference.csv rows mean. CI nominal95 dan Bonferroni simultaneous nominal95 approximate coverage, bukan ranking causal.

| class_name | n | recall_A | recall_B | delta_recall | delta_recall_ci95_low | delta_recall_ci95_high | delta_simultaneous95_low | delta_simultaneous95_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| saya | 20 | 0.9400000000000001 | 0.9099999999999999 | -0.030000000000000006 | -0.12 | 0.029999999999999916 | -0.19999999999999984 | 0.05999999999999994 |
| kamu | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| senang | 20 | 0.9800000000000001 | 0.97 | -0.010000000000000009 | -0.05999999999999983 | 0.0 | -0.08999999999999997 | 0.0 |
| bingung | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| marah | 20 | 0.7300000000000001 | 0.7 | -0.030000000000000027 | -0.09999999999999998 | 0.0 | -0.13000000000000012 | 0.0 |
| apa_kabar | 20 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

I2: intrusion ke halo/tidur pada GT common6. `intrusion_rates.csv`, rows mean:

| class_name | n | overall | halo | tidur |
| --- | --- | --- | --- | --- |
| overall | 120 | 0.006666666666666666 | 0.003333333333333333 | 0.003333333333333333 |
| saya | 20 | 0.02 | 0.0 | 0.02 |
| kamu | 20 | 0.0 | 0.0 | 0.0 |
| senang | 20 | 0.0 | 0.0 | 0.0 |
| bingung | 20 | 0.0 | 0.0 | 0.0 |
| marah | 20 | 0.02 | 0.02 | 0.0 |
| apa_kabar | 20 | 0.0 | 0.0 | 0.0 |

Mean overall0,0066667; setara4 predictions/600 repeated seed-video evaluations, bukan600 video independen. Pada tiap seed counts2,2,0,0,0 (cek file rows overall); saya→tidur dan marah→halo. Apa_kabar intrusion0 pada model validation ini, walau human training-pool ambiguity tinggi. Tidak menganggap human confusion pola model universal.

I3: identitas gap accuracy A−B = restricted_gap + intrusion_gap. Restricted argmax B pada6 output hanya diagnosis, tidak menggantikan primary full8 argmax. File accuracy_decomposition.csv:

| seed | accuracy_A | accuracy_B | accuracy_B_restricted | gap_A_minus_B | restricted_gap | intrusion_gap |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.9333333333333333 | 0.925 | 0.925 | 0.008333333333333304 | 0.008333333333333304 | 0.0 |
| 43 | 0.9333333333333333 | 0.925 | 0.925 | 0.008333333333333304 | 0.008333333333333304 | 0.0 |
| 44 | 0.95 | 0.9333333333333333 | 0.9333333333333333 | 0.016666666666666607 | 0.016666666666666607 | 0.0 |
| 45 | 0.9583333333333334 | 0.925 | 0.925 | 0.033333333333333326 | 0.033333333333333326 | 0.0 |
| 46 | 0.9333333333333333 | 0.9416666666666667 | 0.9416666666666667 | -0.008333333333333304 | -0.008333333333333304 | 0.0 |
| mean | 0.9416666666666667 | 0.9299999999999999 | 0.9299999999999999 | 0.011666666666666648 | 0.011666666666666648 | 0.0 |

Mean gap0,0116667, restricted_gap0,0116667, intrusion_gap0. Raw intrusion tidak sama dengan kehilangan akurasi akibat opsi baru: pada contoh intrusion, restricted old-class prediction juga dapat salah. Ini dekomposisi output, bukan bukti causal representation benefit. Full8 new-class recall halo mean0,93, tidur0,99 (20 videos/class shared across5 seeds), `exploratory_new_class_metrics.csv`; development lower95 halo0,83/tidur0,96 menurut analysis_details. H2 **not tested: development evidence**; point/lower bounds ini bukan kelulusan external H2.

## 9. Additional existing diagnostics

**Hand masking:** existing landmark-based padded box fill114,padding0,15,224² (`exploratory_mask_provenance.json`). Existing last.pt A/B ×5 seeds diuji pada120 common6/160 full8; mean accuracy A0,92±0,0315128; B common6 0,9166667±0,0131762; B full8 0,89125±0,0205396. Saved all-seeds rows dan per-seed reports disertakan. Mean0,92 memiliki bukti CSV, bukan angka tanpa dokumen. All-including-unavailable denominator dipertahankan; duplicate strata tidak dihitung dua kali. Wajah/tubuh/context tersisa, mask tidak sempurna/distribution shift; tidak membuktikan pure background causation. Mendukung kehati-hatian validitas konstruk revised RQ, bukan membuktikan efek vocabulary.

**Frozen-backbone representation probe:** sepuluh completed new six-class linear heads, train360/val120,30epochs,backbone eval,SGD.01/.937/.0005,batch16,source corresponding last.pt A/B tiap seed. Mean A F1=0,9334267350/accuracy0,935; B F1=0,9259280640/accuracy0,9283333. Sumber summary JSON tiap probe dan exploratory_linear_probes.csv/probe_summary.csv. Validation_used_for_selection=false menurut receipt. B backbone sudah melihat lebih banyak kelas/data, probe bukan isolasi causal/geometri embedding. Failed settings conflict seed42 A berhenti sebelum optimization; attempt2 selesai. Runner preflight0fit disimpan. Tidak mengecualikan seed karena kegagalan teknis.

**Learning curves:** semua epoch30 actual-weight train top1=1,0; framework EMA val berbeda (A0,93333–0,95833, B0,93125–0,94375). `exploratory_epoch30_curves.csv` dan original results30 rows. Gap bukan evaluasi model berbobot identik dan tidak membuktikan sebab overfit. Figures kurva existing dipertahankan, tanpa gambar sampel/contact sheets.

**Stage1a precision/power simulation:** existing completed10.000/scenario, bukan data eksternal. Asumsi external300/50 per new class,5seeds; simulated H1 trueΔ0,discordance.02,seedSD0 memberi NI pass probability.7886; seedSD.02 menjadi.5013. H2 binomial reference true accuracy.9, required45/50, probability one class.616123/both-if-independent.379608. Sumber power_precision_report.md tables dan scenario CSV. Ini hasil simulasi berbasis asumsi, REPORTED_NOT_INDEPENDENTLY_VERIFIED per kalkulasi; tidak dicampur sebagai observed model results di RESULTS_SUMMARY.csv. Historical report “no seed variance measured” berlaku sebelum Stage1b. Tidak menjadi bukti power final memadai.

**ROI-only/frame-selection quality:** tidak ditemukan completed matched ROI-only numerical comparator atau eksperimen causal membandingkan recipe frame selection pada current records yang diprioritaskan. Yang tersedia technical coverage/flags dan human pilot; bukan effect estimate selection policy. UNKNOWN di luar scoped discovery; tidak mengklaim pemeriksaan setiap arsip. No temporal/detection matched completed comparator. Rencana CV/grid105fits/factorial H3 belum memiliki completed hasil di bukti saat ini; bukan failed hypothesis.

## 10. Development versus external evidence

Seluruh metrics model dan pilot manusia berasal internal. External400 adalah inventory, **bukan jumlah scored predictions**. Tidak ada skor external final dalam paket. Receipt internal-only dan absence final scoring receipt di audit terdahulu bukan bukti exhaustive tidak pernah ada exposure di luar repo. Stage1b reused validation tetap development, meski paired statistics memakai nama confirmatory module. Tidak mengganti label hasil menjadi final.

## 11. Threats to validity

Frame-level weak labels, semantic validity belum dinilai; single-frame tidak memodelkan gerakan. Signer/session missing menghalangi cluster CI dan klaim generalisasi signer. Source/background cues, near-duplicates dan context leakage belum dieliminasi. Balanced small validation20/class dan5seeds membatasi presisi; degenerate CI kelas perfect hanya menggambarkan support empiris. Equal epochs mengubah data volume,head,optimizer steps; estimand combined vocabulary-and-data expansion, bukan efek kausal class count. Pilot42 diketahui sebelum penetapan seed list final; alasan lima seed/angka42 tidak didokumentasikan; tidak menjanjikan prospective preregistration terhadap semua pilot. No augmentation bukan bukti robustness. Checkpoint hashes tidak memvalidasi bahasa/lisensi. Historical framework/source/per-batch RNG trace tidak lengkap; deterministic flags bukan bitwise reproducibility guarantee.

## 12. Missing experiments and unresolved decisions

Final full-data A1440 train/360val dan B1920/480,10runs,belum dilakukan. Planned2700/3600steps bukan measured. Final manifests null,freeze absent; provenance/exposure dan academic approvals belum lengkap. Detection→Classification adalah perubahan substantif. Judul Addendum01 “Effect of Vocabulary Expansion on YOLO26 Classification for BISINDO Phrase Recognition” berbeda dari draft frame-based yang diajukan; tidak disamakan dengan approval pembimbing.

H1 NI candidateδ.02 **NI_MARGIN_UNRESOLVED**; rekomendasi estimasi Δ+two-sided CI belum approved. H2 asli accuracy conditional GT halo DAN tidur masing-masing≥.80, proposed confirmatory lower one-sided95 **> .80** pada keduanya; point target saja tidak cukup. Jangan mengubah threshold/margin setelah melihat external. Ide discrimination benefit eksploratif termotivasi pilot, bukan hipotesis konfirmatori baru atau hasil proven. CV/grid/factorial lama perlu keputusan scope eksplisit bila dilewati. TASK3 tidak dijalankan.

TASK0 controller `06_notebook/eksperimen/BISINDO_FINAL_EXPERIMENT_CONTROL.ipynb` tersedia; historical completion.json/new tests17, notebook12cells, legacy91/92 (satu `test_notebook_has_module_calls_and_training_false`, assertion1 is not None), full_regression_green=false. TASK1 saved report27/27; TASK2 saved report25/25,13 prior code cells preserved,2026 protected files drift0. Logs/JSON dibaca sebagai **historical receipts**, tes tidak diulang dalam snapshot dan tidak mengubah notebook outputs. TASK0–2 PASS audit/infrastructure tidak berarti scientific experiment completed. Partial Task2 directory20261105 bukan report lengkap;20261109 diprioritaskan.

## 13. Recommendations for final training and paper

Minta approval aktual untuk task/title/RQ, H1 estimasi vs NI substantif, final full-data five-seed equal-epoch scope/nasibCV-H3, dan semantic/provenance/exposure gates. Pertahankan H2 asli dan disclosure pilot. Selesaikan manifests/provenance/cluster/semantic decision sebelum prospective release/freeze; jangan backdate. Laporkan semua seed, fixed last endpoint,paired common6 dengan B eight-way argmax, per-class uncertainty dan diagnostics terpisah. Jelaskan kemungkinan manfaat diskriminasi sebagai mekanisme eksploratif; klaim akhir harus sesuai data eksternal yang benar-benar diperoleh dan independensinya. Tidak ada rekomendasi automatic training atau readiness conference submission.

## 14. Complete evidence references

Seluruh lokasi sumber relatif repository, bukan lokasi historis Windows di dalam records yang sudah direorganisasi. Dalam ZIP, salinan byte-identik berada di `EVIDENCE/<source-relative-path>` dan gambar di `FIGURES/<source-relative-path>`. `EVIDENCE_MANIFEST.json` mengikat SHA256,size,category,purpose,status; checkpoint hanya pointer/hash, tidak disalin. `RESULTS_SUMMARY.csv` memberi source_file per metric; row selector memakai model/seed/split atau reviewer/class dan nama kolom metric. Tabel detailed run precision/recall dihitung sebagai mean saved per-class metrics, diberi label derivative. Hash awal seluruh source yang dibaca dan protected source/notebook/protocol diperiksa sesudah ekspor. Tidak ada fresh video/file-integrity scan45.250; CI,model metadata dan historical tests tidak dieksekusi ulang.

- `01_hasil/addendum/data_inventory_task1_20261105/audit.json` — `EVIDENCE/01_hasil/addendum/data_inventory_task1_20261105/audit.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/data_inventory_task1_20261105/audit_report.md` — `EVIDENCE/01_hasil/addendum/data_inventory_task1_20261105/audit_report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `01_hasil/addendum/data_inventory_task1_20261105/duplicate_impact.json` — `EVIDENCE/01_hasil/addendum/data_inventory_task1_20261105/duplicate_impact.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/data_inventory_task1_20261105/final_verification.json` — `EVIDENCE/01_hasil/addendum/data_inventory_task1_20261105/final_verification.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/data_inventory_task1_20261105/manifest_catalog.json` — `EVIDENCE/01_hasil/addendum/data_inventory_task1_20261105/manifest_catalog.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/data_inventory_task1_20261105/provenance.json` — `EVIDENCE/01_hasil/addendum/data_inventory_task1_20261105/provenance.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/data_inventory_task1_20261105/test_task1_data_audit.log` — `EVIDENCE/01_hasil/addendum/data_inventory_task1_20261105/test_task1_data_audit.log`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/data_inventory_task1_20261105/validation.json` — `EVIDENCE/01_hasil/addendum/data_inventory_task1_20261105/validation.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `01_hasil/addendum/development_interference_20261105/accuracy_decomposition.csv` — `EVIDENCE/01_hasil/addendum/development_interference_20261105/accuracy_decomposition.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/development_interference_20261105/analysis_details.json` — `EVIDENCE/01_hasil/addendum/development_interference_20261105/analysis_details.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/development_interference_20261105/bootstrap_intervals.csv` — `EVIDENCE/01_hasil/addendum/development_interference_20261105/bootstrap_intervals.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/development_interference_20261105/class_interference.csv` — `EVIDENCE/01_hasil/addendum/development_interference_20261105/class_interference.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/development_interference_20261105/common6_primary_results.csv` — `EVIDENCE/01_hasil/addendum/development_interference_20261105/common6_primary_results.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/development_interference_20261105/external_evaluation_report.md` — `EVIDENCE/01_hasil/addendum/development_interference_20261105/external_evaluation_report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `01_hasil/addendum/development_interference_20261105/full8_results.csv` — `EVIDENCE/01_hasil/addendum/development_interference_20261105/full8_results.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/development_interference_20261105/input_provenance.json` — `EVIDENCE/01_hasil/addendum/development_interference_20261105/input_provenance.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/development_interference_20261105/intrusion_rates.csv` — `EVIDENCE/01_hasil/addendum/development_interference_20261105/intrusion_rates.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/final_control_task0/completion.json` — `EVIDENCE/01_hasil/addendum/final_control_task0/completion.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `01_hasil/addendum/final_control_task0/existing_regression_canonical_temp.log` — `EVIDENCE/01_hasil/addendum/final_control_task0/existing_regression_canonical_temp.log`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/final_control_task0/validation.json` — `EVIDENCE/01_hasil/addendum/final_control_task0/validation.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `01_hasil/addendum/scientific_decisions_task2_20261109/candidate_config.json` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/candidate_config.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/scientific_decisions_task2_20261109/effective_configuration.json` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/effective_configuration.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `01_hasil/addendum/scientific_decisions_task2_20261109/final_verification.json` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/final_verification.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/scientific_decisions_task2_20261109/report.md` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `01_hasil/addendum/scientific_decisions_task2_20261109/scientific_decisions.json` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/scientific_decisions.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/scientific_decisions_task2_20261109/seed_provenance.json` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/seed_provenance.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/scientific_decisions_task2_20261109/test_confirmatory_statistics.log` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/test_confirmatory_statistics.log`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/scientific_decisions_task2_20261109/test_final_control_notebook.log` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/test_final_control_notebook.log`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/scientific_decisions_task2_20261109/test_interference_metrics.log` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/test_interference_metrics.log`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/scientific_decisions_task2_20261109/test_task2_configuration.log` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/test_task2_configuration.log`; VERIFIED_FROM_ARTIFACT
- `01_hasil/addendum/scientific_decisions_task2_20261109/validation.json` — `EVIDENCE/01_hasil/addendum/scientific_decisions_task2_20261109/validation.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `01_hasil/tahap1a/exploratory_development_stage1a_20261103/exploratory_paired/exploratory_analysis.json` — `EVIDENCE/01_hasil/tahap1a/exploratory_development_stage1a_20261103/exploratory_paired/exploratory_analysis.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_all_fits.csv` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_all_fits.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_all_metrics.csv` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_all_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_epoch30_curves.csv` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_epoch30_curves.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_failures.json` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_failures.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_final_verification.json` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_final_verification.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_first_phase_timing.json` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_first_phase_timing.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_linear_probes.csv` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_linear_probes.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_masking_all_seeds.csv` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_masking_all_seeds.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_masking_summary.csv` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_masking_summary.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_model_summary.csv` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_model_summary.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_new_class_metrics.csv` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_new_class_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_paired_per_seed.csv` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_paired_per_seed.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_probe_summary.csv` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_probe_summary.csv`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_provenance.json` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_provenance.json`; VERIFIED_FROM_ARTIFACT
- `01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_stage1b_report.md` — `EVIDENCE/01_hasil/tahap1b/exploratory_development_stage1b_agent_execution_20261103_v2/exploratory_stage1b_report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `02_exp/arsip_pipeline/experiment_yolo26_classification_notebook_20261102/evaluation/comparison_metrics.csv` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_classification_notebook_20261102/evaluation/comparison_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_classification_notebook_20261102/pilot_conclusion.md` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_classification_notebook_20261102/pilot_conclusion.md`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/evaluation/comparison_metrics.csv` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/evaluation/comparison_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/pilot_conclusion.md` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_classification_pilot/pilot_conclusion.md`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/confusion_matrix.png` — `FIGURES/02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/confusion_matrix.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/inter_reviewer_agreement.json` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/inter_reviewer_agreement.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/per_class_results.csv` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/per_class_results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/pilot_report.md` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/pilot_report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/pilot_summary.csv` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/pilot_summary.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/reviewer_1_confusion.csv` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/reviewer_1_confusion.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/reviewer_2_confusion.csv` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/reviewer_2_confusion.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/verification.json` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_pilot/analysis/verification.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_pilot/responses/reviewer_1.json` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_pilot/responses/reviewer_2.json` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/arsip_pipeline/experiment_yolo26_training_pilot/pilot_conclusion.md` — `EVIDENCE/02_exp/arsip_pipeline/experiment_yolo26_training_pilot/pilot_conclusion.md`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/apa_kabar_halo_diagnostics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/apa_kabar_halo_diagnostics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/comparison_metrics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/comparison_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_A_common6.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_A_common6.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_A_common6.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_A_common6.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_B_common6.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_B_common6.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_B_common6.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_B_common6.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_B_full8.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_B_full8.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_B_full8.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/confusion_B_full8.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/per_class_metrics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/per_class_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/predictions.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/evaluation/predictions.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/manifest.csv` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/reports/experiment_config.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/reports/experiment_config.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_A/args.yaml` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_A/args.yaml`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_A/results.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_A/results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_A/results.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_A/results.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_A/weights/last.pt` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_A_execution.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_A_execution.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_B/args.yaml` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_B/args.yaml`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_B/results.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_B/results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_B/results.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_B/results.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_B/weights/last.pt` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_B_execution.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/model_B_execution.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/trained_models.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed42/runs/trained_models.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/apa_kabar_halo_diagnostics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/apa_kabar_halo_diagnostics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/comparison_metrics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/comparison_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_A_common6.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_A_common6.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_A_common6.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_A_common6.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_B_common6.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_B_common6.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_B_common6.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_B_common6.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_B_full8.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_B_full8.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_B_full8.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/confusion_B_full8.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/per_class_metrics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/per_class_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/predictions.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/evaluation/predictions.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/manifest.csv` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/reports/experiment_config.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/reports/experiment_config.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_A/args.yaml` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_A/args.yaml`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_A/results.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_A/results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_A/results.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_A/results.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_A/weights/last.pt` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_A_execution.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_A_execution.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_B/args.yaml` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_B/args.yaml`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_B/results.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_B/results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_B/results.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_B/results.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_B/weights/last.pt` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_B_execution.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/model_B_execution.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/trained_models.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed43/runs/trained_models.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/apa_kabar_halo_diagnostics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/apa_kabar_halo_diagnostics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/comparison_metrics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/comparison_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_A_common6.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_A_common6.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_A_common6.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_A_common6.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_B_common6.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_B_common6.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_B_common6.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_B_common6.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_B_full8.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_B_full8.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_B_full8.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/confusion_B_full8.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/per_class_metrics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/per_class_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/predictions.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/evaluation/predictions.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/manifest.csv` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/reports/experiment_config.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/reports/experiment_config.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_A/args.yaml` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_A/args.yaml`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_A/results.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_A/results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_A/results.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_A/results.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_A/weights/last.pt` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_A_execution.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_A_execution.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_B/args.yaml` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_B/args.yaml`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_B/results.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_B/results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_B/results.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_B/results.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_B/weights/last.pt` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_B_execution.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/model_B_execution.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/trained_models.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed44/runs/trained_models.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/apa_kabar_halo_diagnostics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/apa_kabar_halo_diagnostics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/comparison_metrics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/comparison_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_A_common6.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_A_common6.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_A_common6.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_A_common6.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_B_common6.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_B_common6.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_B_common6.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_B_common6.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_B_full8.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_B_full8.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_B_full8.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/confusion_B_full8.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/per_class_metrics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/per_class_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/predictions.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/evaluation/predictions.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/manifest.csv` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/reports/experiment_config.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/reports/experiment_config.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_A/args.yaml` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_A/args.yaml`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_A/results.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_A/results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_A/results.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_A/results.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_A/weights/last.pt` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_A_execution.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_A_execution.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_B/args.yaml` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_B/args.yaml`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_B/results.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_B/results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_B/results.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_B/results.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_B/weights/last.pt` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_B_execution.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/model_B_execution.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/trained_models.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed45/runs/trained_models.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/apa_kabar_halo_diagnostics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/apa_kabar_halo_diagnostics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/comparison_metrics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/comparison_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_A_common6.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_A_common6.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_A_common6.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_A_common6.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_B_common6.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_B_common6.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_B_common6.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_B_common6.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_B_full8.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_B_full8.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_B_full8.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/confusion_B_full8.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/per_class_metrics.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/per_class_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/predictions.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/evaluation/predictions.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/manifest.csv` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/reports/experiment_config.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/reports/experiment_config.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_A/args.yaml` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_A/args.yaml`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_A/results.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_A/results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_A/results.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_A/results.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_A/weights/last.pt` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_A_execution.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_A_execution.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_B/args.yaml` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_B/args.yaml`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_B/results.csv` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_B/results.csv`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_B/results.png` — `FIGURES/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_B/results.png`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_B/weights/last.pt` — pointer-only; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_B_execution.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/model_B_execution.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/trained_models.json` — `EVIDENCE/02_exp/tahap1b/full_runs/exploratory_development_stage1b_user_20261103_v1_seed46/runs/trained_models.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed42_attempt2/exploratory_probe_summary.json` — `EVIDENCE/02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed42_attempt2/exploratory_probe_summary.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed43/exploratory_probe_summary.json` — `EVIDENCE/02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed43/exploratory_probe_summary.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed44/exploratory_probe_summary.json` — `EVIDENCE/02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed44/exploratory_probe_summary.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed45/exploratory_probe_summary.json` — `EVIDENCE/02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed45/exploratory_probe_summary.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed46/exploratory_probe_summary.json` — `EVIDENCE/02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_A_seed46/exploratory_probe_summary.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed42/exploratory_probe_summary.json` — `EVIDENCE/02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed42/exploratory_probe_summary.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed43/exploratory_probe_summary.json` — `EVIDENCE/02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed43/exploratory_probe_summary.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed44/exploratory_probe_summary.json` — `EVIDENCE/02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed44/exploratory_probe_summary.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed45/exploratory_probe_summary.json` — `EVIDENCE/02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed45/exploratory_probe_summary.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed46/exploratory_probe_summary.json` — `EVIDENCE/02_exp/tahap1b/linear_probes/exploratory_development_stage1b_user_20261103_v1_probe_B_seed46/exploratory_probe_summary.json`; VERIFIED_FROM_ARTIFACT
- `02_exp/tahap1b/mask_inputs/exploratory_development_stage1b_user_20261103_v1_exploratory_masks/exploratory_mask_provenance.json` — `EVIDENCE/02_exp/tahap1b/mask_inputs/exploratory_development_stage1b_user_20261103_v1_exploratory_masks/exploratory_mask_provenance.json`; VERIFIED_FROM_ARTIFACT
- `03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power/h1_effect_sensitivity.md` — `EVIDENCE/03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power/h1_effect_sensitivity.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power/h1_power_scenarios.csv` — `EVIDENCE/03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power/h1_power_scenarios.csv`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power/h2_power_scenarios.csv` — `EVIDENCE/03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power/h2_power_scenarios.csv`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power/power_precision_report.md` — `EVIDENCE/03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power/power_precision_report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power/simulation_provenance.json` — `EVIDENCE/03_insight_exp/arsip_tahap1a/experiment_yolo26_classification_diag_v1/reports/power/simulation_provenance.json`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_exploratory_multiseed/exploratory_multiseed_summary.json` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_exploratory_multiseed/exploratory_multiseed_summary.json`; VERIFIED_FROM_ARTIFACT
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed42/exploratory_masked_metrics.csv` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed42/exploratory_masked_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed42/exploratory_masked_report.md` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed42/exploratory_masked_report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed43/exploratory_masked_metrics.csv` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed43/exploratory_masked_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed43/exploratory_masked_report.md` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed43/exploratory_masked_report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed44/exploratory_masked_metrics.csv` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed44/exploratory_masked_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed44/exploratory_masked_report.md` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed44/exploratory_masked_report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed45/exploratory_masked_metrics.csv` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed45/exploratory_masked_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed45/exploratory_masked_report.md` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed45/exploratory_masked_report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed46/exploratory_masked_metrics.csv` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed46/exploratory_masked_metrics.csv`; VERIFIED_FROM_ARTIFACT
- `03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed46/exploratory_masked_report.md` — `EVIDENCE/03_insight_exp/tahap1b/exploratory_development_stage1b_user_20261103_v1_masked_eval_seed46/exploratory_masked_report.md`; REPORTED_NOT_INDEPENDENTLY_VERIFIED
- `05_kode/src/analysis_paired.py` — `EVIDENCE/05_kode/src/analysis_paired.py`; VERIFIED_FROM_ARTIFACT
- `05_kode/src/classification_diagnostics.py` — `EVIDENCE/05_kode/src/classification_diagnostics.py`; VERIFIED_FROM_ARTIFACT
- `05_kode/src/classification_pilot.py` — `EVIDENCE/05_kode/src/classification_pilot.py`; VERIFIED_FROM_ARTIFACT
- `05_kode/src/classification_research.py` — `EVIDENCE/05_kode/src/classification_research.py`; VERIFIED_FROM_ARTIFACT
- `05_kode/src/confirmatory_statistics.py` — `EVIDENCE/05_kode/src/confirmatory_statistics.py`; VERIFIED_FROM_ARTIFACT
- `05_kode/src/frame_candidates.py` — `EVIDENCE/05_kode/src/frame_candidates.py`; VERIFIED_FROM_ARTIFACT
- `05_kode/src/interference_metrics.py` — `EVIDENCE/05_kode/src/interference_metrics.py`; VERIFIED_FROM_ARTIFACT
- `05_kode/src/pilot_study.py` — `EVIDENCE/05_kode/src/pilot_study.py`; VERIFIED_FROM_ARTIFACT
- `07_dokumen/TRACKING_BISINDO.md` — `EVIDENCE/07_dokumen/TRACKING_BISINDO.md`; VERIFIED_FROM_ARTIFACT
- `07_dokumen/protokol/HYPOTHESES_ENDPOINTS_ADDENDUM.md` — `EVIDENCE/07_dokumen/protokol/HYPOTHESES_ENDPOINTS_ADDENDUM.md`; VERIFIED_FROM_ARTIFACT
- `07_dokumen/protokol/PREREGISTRATION.md` — `EVIDENCE/07_dokumen/protokol/PREREGISTRATION.md`; VERIFIED_FROM_ARTIFACT
- `07_dokumen/protokol/PREREGISTRATION_ADDENDUM_01.md` — `EVIDENCE/07_dokumen/protokol/PREREGISTRATION_ADDENDUM_01.md`; VERIFIED_FROM_ARTIFACT
- `07_dokumen/protokol/analysis_plan.json` — `EVIDENCE/07_dokumen/protokol/analysis_plan.json`; VERIFIED_FROM_ARTIFACT
- `07_dokumen/protokol/external_scoring_readiness.md` — `EVIDENCE/07_dokumen/protokol/external_scoring_readiness.md`; VERIFIED_FROM_ARTIFACT
- `07_dokumen/protokol/reproducibility_manifest.json` — `EVIDENCE/07_dokumen/protokol/reproducibility_manifest.json`; VERIFIED_FROM_ARTIFACT
