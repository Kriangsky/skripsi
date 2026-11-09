# TASK 1 — Data inventory, provenance & integrity audit

**TASK 1 STATUS: PASS (audit selesai). External/final readiness tetap BLOCKED.**

Audit read-only, 2026-11-05 Asia/Bangkok. Tidak ada training, external inference, perubahan split/label/frame, atau TASK 2.

## 1. Expected versus observed

| Ukuran | Expected | Observed | Bukti |
|---|---:|---:|---|
| internal | 2400 | 2400 | split_lock + physical file identity/SHA256 |
| external | 400 | 400 | split_lock + physical file identity/SHA256 |
| internal_train | 1920 | 1920 | split_lock + physical file identity/SHA256 |
| internal_val | 480 | 480 | split_lock + physical file identity/SHA256 |
| external_common6 | 300 | 300 | split_lock + physical file identity/SHA256 |
| external_full8 | 400 | 400 | split_lock + physical file identity/SHA256 |

Seluruh 2.800 path video fisik tersedia; 2.800 hash file cocok metadata beku. Tidak ada path hilang/ekstra. Hash video diulang pada akhir audit, tanpa perubahan. Physical count dibedakan dari jumlah CSV; kesamaan set path dan hash mengikat keduanya.

| Kelas (ID) | Internal | Train | Val | External |
|---|---:|---:|---:|---:|
| saya (0) | 300 | 240 | 60 | 50 |
| kamu (1) | 300 | 240 | 60 | 50 |
| senang (2) | 300 | 240 | 60 | 50 |
| bingung (3) | 300 | 240 | 60 | 50 |
| marah (4) | 300 | 240 | 60 | 50 |
| apa_kabar (5) | 300 | 240 | 60 | 50 |
| halo (6) | 300 | 240 | 60 | 50 |
| tidur (7) | 300 | 240 | 60 | 50 |

## 2. Split/GT dan pairing

IDs/path unik; mapping ID/nama kelas, internal folder labels, source/split dan video identity konsisten. v2 annotations, v3 inventory/annotations dan split_lock mengacu identity beku yang sama. CSV netral400 baris unik dan lengkap; original_filename class tokens sesuai ground_truth; ini verifikasi konsistensi lokal, bukan semantik atau provenance archive.

Untuk setiap seed42,43,44,45,46: A common6 validation120 dan B full8 validation160. Shared common6 train360/val120 identik. Seluruh640 sampel, label, video SHA, selected-candidate SHA dan processed-image SHA konsisten lintas seed. A/B byte PNG shared benar-benar diperiksa, bukan hanya disandingkan baris CSV. `enriched_predictions` dan `align_predictions` existing memvalidasi paired IDs/GT/frame hashes dan full8 argmax B; tidak menghitung atau men-tuning hasil baru.

## 3. Frame provenance

Policy existing: `pilot_study.choose_candidate` memilih kandidat readable menurut quality_flags, jumlah flags, focus, frame_index dan candidate_id; policy teknis label/prediction-blind, bukan pemilihan dari model mistakes. `frame_candidates.py` menentukan primary fractions20–80% dan fallback5/10/15/85/90/95%; config/cache mengikat recipe dan split hash. TASK 1 tidak menjalankan selection maupun detector.

v3:2.800 JSON cache records dan33.250 kandidat; seluruh kandidat dihash dan didecode dari JPEG existing. v3 annotations masih needs_review2.800, selected positive frame0. Development640 frame berstatus semantic not_assessed; technical flagged497/eligible143. Technical validity tidak membuktikan gesture correctness.

Katalog manifest existing: 9 file pada `manifest_catalog.json`. Ada manifest pilot manusia160, Classification pilot640 dan lima seed640, serta legacy manifests. Final internal/external frame manifests belum ditetapkan (null dalam analysis_plan); tidak mempromosikan development manifest menjadi final.

Final manifest perlu complete coverage dan fields `sample_id,video,video_sha256,image_path,image_sha256,class_id,class_name`, selected-candidate/frame-index/preprocessing provenance dan split binding; final internal meliputi train/val, external satu frame per400 video. Keputusan final policy/review menunggu fase berikutnya.

## 4. Exact duplicate/leakage

| Pemeriksaan | Hasil aktual |
|---|---|
| Duplicate stored video hashes | 0 kelompok; metadata saja |
| Duplicate actual video bytes | 0 kelompok dari2.800 file |
| Duplicate candidate JPEG lintas video | 3 kelompok, masing-masing2 file |
| Duplicate decoded RGB lintas video | 3 kelompok yang sama |
| Cross-split decoded identity pada image yang diaudit | 0 kelompok |
| Unavailable/corrupt/hash mismatch | 0 dari45.250 file yang diperiksa |
| Near-duplicate, semua frame video, semantic/signer independence | UNVERIFIED; tidak dievaluasi |

**Temuan baru:** `train/senang/senang_223.mp4` dan `train/senang/senang_224.mp4` berbagi kandidat f7/f10/f14 identik byte dan RGB. Kedua video berada pada locked train. Hanya224 masuk manifest development, menggunakan f48; tiga frame duplikat ini bukan frame development yang dipakai. Tidak mengubah split atau membuang video. Detail digest/path: `duplicate_impact.json`. Shared frames membuktikan sebagian content berulang, bukan membuktikan keseluruhan video identik.

Scope decoded identity:33.250 v3 cached images,400 external v2 cached images,3.200 selected development JPEG dan5.600 A/B PNG dari lima seed. Repetisi file/sampel yang disengaja lintas arm/seed tidak dihitung sebagai video berbeda. Hash piksel menyertakan shape dan channel RGB uint8; bounded4 workers,max20M pixels/100MB perimage. Tidak decode ulang video mentah atau mengekstrak frame baru. Scan ini bukan jaminan tidak ada overlap temporal di frame lain, near-duplicate, crop/padding-equivalent content atau signer leakage.

## 5. Kaggle provenance/exposure evidence

| Topik | Status | Evidence | Keterbatasan |
|---|---|---|---|
| Dataset source | PARTIALLY VERIFIED | Historical draft reference [26] cites Anggi Yohanes Pardede, BISINDO 40 Kata mp4, Kaggle; URL in source passages. | Citation is local documentary evidence; no archive/version-to-local-byte binding verified. |
| Version/acquisition | UNVERIFIED | Draft lists access date Jun 10 2026; preparation records bind local hashes. | Draft date is not an acquisition receipt. Need dataset version ID, download/acquisition log, original archive hash, subset/rename mapping provenance. |
| License | UNVERIFIED | Scoped filename search recorded; no dataset license/version receipt found. | Need license text tied to acquired version, author attribution and reuse conditions. No license inferred from public availability. |
| Signer/session | UNVERIFIED | External nonempty signer=0/400; internal=0/2400. | Need trustworthy source records or reviewer-backed provenance; hashes do not establish signer independence. |
| Historical v2/v3 exposure | VERIFIED preprocessing | 400 external v2 annotation/frame records; 400 external v3 candidate-cache records and actual cached image byte/RGB checks in audit.json/file_integrity.jsonl. | Previously opened/preprocessed; not untouched. Draft describes filtering blurry/transitional frames, without complete underlying acquisition/selection audit trail. |
| External scoring/tuning history | UNVERIFIED completeness | 7 discovered completed Classification evaluation receipts have internal-only manifests; 0 final scoring-start receipts found in scoped search. | Absence in these records is not proof of absence elsewhere. Owner/reviewer must review historical external inference, score viewing, frame/threshold/model/checkpoint/hyperparameter selection; do not generate attestation. |
| Neutral filename GT | VERIFIED local consistency | 400 unique neutral filename/GT/original-filename CSV rows map to physical external files and locked class IDs; original filename class tokens match CSV. | Original archive labels and semantic gesture correctness unverified. Mapping CSV alone is not authoritative acquisition provenance. |
| Semantic/ambiguity review | PARTIAL, internal only | Two reviewer response exports cover 160 training-pool pilot samples; original exports match; counts below. Classification technical contact-sheet review has expert_verified_n=0 and semantic_unassessed_n=640. | No complete external semantic adjudication. Review responses are not relabelled into final manifests; reviewer expertise/independence not established by hashes. |

Reviewer pilot internal: reviewer1 ambiguous27/160; reviewer2 ambiguous12/160 dan invalid4/160. Bukan adjudikasi external atau seluruh640 frame Classification. Known ambiguous0 pada technical report bukan bukti ambiguity0. Semua respons dan labels lama tetap utuh.

Draft historis menyatakan external blurry/transitional frames dibuang. Itu bertentangan dengan klaim bahwa seluruh proses lama sudah terbukti prediction/quality-blind; yang terverifikasi hanya policy/cache lokal saat ini. Perlu original archive/version/subset-selection records dan owner exposure review. Dokumen draft tidak otomatis menjadi bukti pelaksanaan. Tidak menyimpulkan Kaggle untouched atau signer-independent.

## 6. Missing evidence / next actions

- UNVERIFIED: acquired Kaggle version, archive hash, acquisition log, lisensi dan provenance subset/rename; minta sumber asli/records yang mengikat400 video lokal.
- UNVERIFIED: signer/session dan independensi sumber; jangan membuat signer ID dari penampilan.
- UNVERIFIED: kelengkapan historical score viewing/tuning/frame-selection exposure; reviewer/owner harus meninjau riwayat aktual dan memberi attestation sendiri.
- Final internal/external manifests belum registered; semantic validity belum lengkap. TASK 2 meninjau scientific implications, academic approval dan keputusan final; tidak diselesaikan diam-diam di TASK 1.
- Tinjau pasangan senang223/224 untuk kebijakan content clusters sebelum desain final; tidak memindah atau menghapus data dalam audit ini.

## 7. Changes / validation

Created `05_kode/src/task1_data_audit.py`, `05_kode/tests/test_task1_data_audit.py` dan evidence folder ini; update notebook primer TASK 1 dan tracker existing. Tidak membuat notebook/tracker pesaing. Status notebook dan tes integrasi akhir dicatat pada `validation.json` setelah command aktual.

Actual audit command:

```powershell
& .venv-bisindo/Scripts/python.exe 05_kode/src/task1_data_audit.py --out 01_hasil/addendum/data_inventory_task1_20261105
& .venv-bisindo/Scripts/python.exe 01_hasil/addendum/data_inventory_task1_20261105/provenance_review.py
```

Audit exit0; 45.250 actual-file checks, end-of-audit video/metadata/protected-artifact hash mismatches0. `execution.json` berisi command/start/completion/exit. `file_integrity.jsonl` adalah ledger per file; `audit.json` memuat input/source hashes dan findings. `provenance.json` memuat evidence table/source passages/search commands. Full historical suite tidak diklaim hijau; kegagalan lama TASK 0 tetap direkam.

**Mandatory STOP:** TASK 2 belum diotorisasi.


Final integration results: safe full notebook execution passed; all12 prior code sources/outputs/counts preserved,
new TASK 1 output executed in project kernel. Relevant tests27/27 passed (audit4, notebook3, frame6, interference4,
statistics10). First integration attempt failed in text serialization/logging before saving primary; failed evidence retained.
`validation.json` covers code execution/tests. Subsequent current-task Markdown and LF normalization are verified separately
in `final_verification.json`; no code/output changed in that documentation pass. Historical full-suite91/92 from TASK 0
was not rerun; stale output expectation remains disclosed. See tracker section12 for exact commands and limitations.
