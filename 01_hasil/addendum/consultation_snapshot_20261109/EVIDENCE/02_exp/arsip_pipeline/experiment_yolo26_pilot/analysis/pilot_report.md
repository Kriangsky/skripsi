# Pilot single-frame BISINDO — exploratory

N=160; 20 per kelas; internal train saja; seed 42. Frame otomatis dibekukan sebelum review.
Technical coverage = frame lolos semua heuristik / semua sampel. Inspectable coverage mencakup frame flagged.
Ambiguous dan invalid selalu gagal pada accuracy_all. Subset identifiable berarti reviewer memilih salah satu kelas, bukan berarti jawabannya benar.

| Reviewer | Benar / semua | Accuracy semua (95% Wilson) | Identifiable n / semua | Accuracy subset |
|---|---|---|---|---|
| reviewer_1 | 129 / 160 | 0.806 (0.738–0.860) | 133 / 160 | 0.970 |
| reviewer_2 | 138 / 160 | 0.863 (0.801–0.907) | 144 / 160 | 0.958 |

Agreement=0.844; Cohen kappa=0.8251595419179998. Kategori: saya, kamu, senang, bingung, marah, apa_kabar, halo, tidur, ambiguous, invalid.
Technical coverage=0.269; inspectable coverage=1.000.
Tidak teridentifikasi oleh salah satu reviewer: 31; oleh keduanya: 12.
Pasangan sering tertukar: 1; kelas terlibat: 2. Definisi: >=2 kesalahan pooled; bukan uji signifikansi.

## Kelas sulit dan sumber kesalahan

- saya: accuracy semua reviewer_1=0.80, reviewer_2=0.95; tidak teridentifikasi pooled=5; teknis eligible/flagged/missing=13/7/0; tidak teridentifikasi pada eligible pooled=5; alasan temporal pooled=1.
- kamu: accuracy semua reviewer_1=0.85, reviewer_2=0.90; tidak teridentifikasi pooled=5; teknis eligible/flagged/missing=4/16/0; tidak teridentifikasi pada eligible pooled=0; alasan temporal pooled=1.
- senang: accuracy semua reviewer_1=0.85, reviewer_2=0.70; tidak teridentifikasi pooled=8; teknis eligible/flagged/missing=1/19/0; tidak teridentifikasi pada eligible pooled=2; alasan temporal pooled=1.
- bingung: accuracy semua reviewer_1=0.90, reviewer_2=0.95; tidak teridentifikasi pooled=3; teknis eligible/flagged/missing=0/20/0; tidak teridentifikasi pada eligible pooled=0; alasan temporal pooled=0.
- marah: accuracy semua reviewer_1=0.85, reviewer_2=1.00; tidak teridentifikasi pooled=3; teknis eligible/flagged/missing=2/18/0; tidak teridentifikasi pada eligible pooled=1; alasan temporal pooled=1.
- apa_kabar: accuracy semua reviewer_1=0.45, reviewer_2=0.70; tidak teridentifikasi pooled=15; teknis eligible/flagged/missing=4/16/0; tidak teridentifikasi pada eligible pooled=1; alasan temporal pooled=3.
- halo: accuracy semua reviewer_1=0.80, reviewer_2=0.80; tidak teridentifikasi pooled=2; teknis eligible/flagged/missing=12/8/0; tidak teridentifikasi pada eligible pooled=1; alasan temporal pooled=0.
- tidur: accuracy semua reviewer_1=0.95, reviewer_2=0.90; tidak teridentifikasi pooled=2; teknis eligible/flagged/missing=7/13/0; tidak teridentifikasi pada eligible pooled=0; alasan temporal pooled=1.
- reviewer_1: missing=0, ambiguous/invalid pada frame terbaca=27, salah kelas pada frame terbaca=4, alasan temporal=4, preparation=7, transition=33.
- reviewer_2: missing=0, ambiguous/invalid pada frame terbaca=16, salah kelas pada frame terbaca=6, alasan temporal=4, preparation=30, transition=27.

## Implikasi arsitektur
Nilai semua kelas dan agreement bersama coverage; accuracy subset saja mengandung selection bias. Tidak ada threshold kelayakan yang dipraregistrasikan, sehingga laporan ini tidak memberi keputusan lulus/gagal otomatis.
Kegagalan seleksi teknis dipisahkan dari jawaban semantik pada gambar terbaca. Jawaban temporal_insufficient adalah pendapat reviewer, bukan bukti kausal bahwa temporal model diperlukan. Pilot ini tidak membandingkan video dengan frame secara terkontrol.
Jika kelas tertentu tetap sulit pada frame teknis baik dan kedua reviewer konsisten menyebut informasi temporal kurang, pertimbangkan studi temporal LSTM/TCN berikutnya. Jangan mengganti arsitektur atau memilih ulang frame pilot dari respons ini.
Kelayakan YOLO26 single-frame harus diputuskan setelah meninjau per-class accuracy, confusion, alasan, dan konsistensi; hasil manusia tidak mengukur performa YOLO26. Tidak ada klaim kelayakan publikasi Model A.

## Keterbatasan
Wilson 95% mengasumsikan observasi independen. Korelasi signer/video dapat membuat interval terlalu sempit; tanpa metadata signer tidak dapat dihitung interval cluster yang sah. N=20 per kelas kecil. Pilot eksploratif pada training pool, bukan benchmark Kaggle external test. Tidak ada training, full-dataset review, atau tuning test set.
- Technical heuristics are not calibrated semantic validity checks.
- Signer-independent inference unavailable when signer metadata is missing; video != signer.
- Exploratory training-pool pilot; not an external-test benchmark.
- Reviewer independence requires separate people/devices and no answer sharing.
- Image appearance may itself reveal source cues; opaque files cannot eliminate visual confounding.
