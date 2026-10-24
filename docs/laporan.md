---
title: "Laporan Solusi Data Science Competition FIK FAIR 2026"
subtitle: "Prediksi Anonymous Competition Score: foundation model tabular, informasi urutan baris, dan koreksi residual bertingkat"
author: "Second Choice: Riantama Putra, Haikal Muhammad Royyan, Defina Aurellia Ningsih, Taufiq Muhammad Arif"
date: "Oktober 2026"
---

# 1. Pendahuluan

Kompetisi ini adalah regresi tabular. Tugasnya memprediksi *Anonymous Competition Score* dari 24 fitur numerik anonim yang sudah ditransformasi oleh panitia. Data latih berisi 382 baris dan data uji 96 baris. Metrik evaluasinya RMSE. Leaderboard public dihitung dari sekitar 30% data uji (±29 baris), sedangkan private dari ±67 baris.

Dengan data sekecil ini, risiko terbesarnya adalah *overfitting*, validasi yang terlalu optimis, dan mengejar skor public yang hanya dihitung dari 29 baris. Karena itu setiap ide diperlakukan sebagai hipotesis. Ide hanya dipertahankan kalau lolos validasi silang berulang pada fold yang sama, dan skor public tidak pernah dipakai untuk memilih parameter atau bobot.

Seluruh proses ada di notebook `final.ipynb`. Penomoran bagian di laporan ini mengikuti section notebook tersebut.

# 2. Eksplorasi Data (Notebook Section 3–5)

**Kualitas data.** Data bersih: tidak ada nilai hilang, tidak ada duplikat persis di dalam train maupun antara train dan test, dan setiap fitur memiliki 382 nilai unik. `record_id` adalah ID acak (D0001–D0478) tanpa korelasi berarti dengan target (ρ = 0,05), jadi tidak dipakai sebagai fitur.

**Target.** Distribusinya bimodal, dengan puncak sekitar 37 dan 57. Batas bawahnya 32,6 dan ekor kanannya memanjang sampai 100,4 (skewness 0,75). Baris dengan target di atas 70 hanya 7% dari data, tetapi menyumbang sekitar 41% *squared error* model awal. Ekor kanan ini menjadi fokus analisis sepanjang kompetisi.

**Struktur fitur.**

- Fitur-fiturnya sangat kolinear: 44 pasang memiliki |r| > 0,7, dengan nilai maksimum 0,94.
- Empat komponen utama menjelaskan 91% varians, dan hanya empat eigenvalue yang berada di atas batas noise Marchenko–Pastur. Fitur ini kemungkinan besar proyeksi dari sinyal berdimensi rendah.
- Meski begitu, sinyal target justru ada di komponen bervarians kecil. Regresi linear pada 4 PC memberi RMSE 11,98, sedangkan pada 24 PC 10,67. Jadi reduksi dimensi yang agresif merusak.
- Banyak fitur berhubungan dengan target dalam bentuk U terbalik. Spline aditif (RMSE 9,36) mengalahkan Ridge (10,73), yang berarti model non-linear diperlukan.

**Pergeseran train–test.** Sembilan dari 24 fitur lolos uji KS pada p < 0,05, tetapi pergeserannya kecil (≤ 0,3 SD). AUC *adversarial validation* hanya 0,53–0,59. Jarak tetangga terdekat test→train setara dengan train→train (0,179 vs 0,204 dalam jarak Euclidean terstandardisasi). Dari sini kami simpulkan *random K-fold* representatif untuk data uji, dan *reweighting* tidak diperlukan.

**Urutan baris file (temuan kunci).** Audit posisi baris menemukan bahwa `train.csv` hampir terurut menurun menurut target: korelasi Spearman antara posisi baris dan target −0,981. Pemeriksaan awal hanya menguji angka di `record_id` (yang memang acak), sehingga posisi baris di file sempat terlewat. Untuk test, label tidak tersedia, jadi urutannya diperiksa secara tidak langsung:

- Korelasi Spearman antara posisi baris test dan prediksi model fitur adalah −0,806.
- Pola pergeseran (*drift*) tiap fitur sepanjang file berkorelasi 0,873 antara train dan test.

Kedua bukti ini kuat mendukung, tetapi tidak membuktikan, bahwa `test.csv` juga terurut dengan cara yang sama. Pada 4 Oktober 2026 tim menanyakan hal ini langsung ke panitia. Panitia mengonfirmasi bahwa pemanfaatan urutan baris adalah bagian dari EDA dan preprocessing masing-masing peserta.

**Pasangan berdekatan.** Ada beberapa pasangan baris yang fiturnya sangat mirip tetapi targetnya berbeda (misalnya D0250/D0183 berjarak 0,204 dengan selisih target 2,74). Pasangan ini tidak dihapus, karena kemiripan fitur tidak membuktikan label salah. Kami mengujinya lewat *grouped split*, dijelaskan di Bagian 5.

# 3. Preprocessing dan Feature Engineering

Preprocessing berbeda untuk tiap keluarga model. Semua transformasi di-*fit* hanya pada partisi latih di setiap fold.

| Langkah | Dipakai untuk | Alasan dan bukti |
|---|---|---|
| Standardisasi lalu PCA *whitened* 10/12/14/16/20 dimensi | SVR | Kurva RMSE terhadap dimensi PCA bergerigi, sehingga rata-rata beberapa dimensi lebih stabil daripada satu dimensi |
| Target log(y − 30) yang distandardisasi | SVR | Mengurangi kemiringan target; SVR membaik dari 8,83 ke 8,35 |
| Prediksi anggota SVR di-*clip* ke rentang target train | SVR | Tanpa clip, satu baris diprediksi sampai 138 (maks target 100,4) |
| Fitur mentah, `norm_methods=['none']` | TabICLv2 | Normalisasi *power* internal merugikan pada data ini |
| *Pseudo-labelling* transduktif | TabICLv2, TabPFN-3.5, SVR | Lihat Bagian 4.1 |
| Posisi relatif baris dalam file, (i + 0,5)/n | Tahap urutan | Informasi urutan, dinormalisasi per file atau partisi |
| *Detrending* fitur dan target sepanjang file (Gaussian, lalu *local-linear*) | Tahap v11–v13 | Memisahkan tren urutan dari penyimpangan lokal tiap baris |
| Interaksi residual fitur × posisi tersentralisasi | Tahap v12–v13 | Efek fitur boleh berubah sepanjang file |

Beberapa ide preprocessing diuji dan **ditolak** karena tidak lolos validasi. Semuanya tetap dicatat agar proses seleksinya transparan:

- *Quantile transform*, PCA-denoise, dan ICA. ICA gagal membalik proyeksi panitia karena sumbernya *heavy-tailed*.
- Target log untuk model *foundation*.
- *Test-time jitter* dan *subagging context*.
- *Sample weighting* untuk ekor dan *classifier* khusus ekor.
- Model dua-tahap berbasis mode target.
- Pada tahap v13: *clipping* fitur dan *detrending* yang lebih lebar. Keduanya justru memperburuk RMSE.

# 4. Pemodelan (Notebook Section 6–9)

## 4.1 Lapisan fitur: foundation model dan SVR

Model *foundation* tabular memprediksi lewat *in-context learning*: data latih menjadi konteks, dan tidak ada pelatihan gradien pada data kompetisi. Pada benchmark awal, TabICLv2 (RMSE 7,63) mengalahkan TabPFN v2 (7,92), SVR multi-PCA (8,11), CatBoost (8,90), LightGBM (9,02), dan Ridge (10,77). Komponen yang dipakai:

- **TabICLv2**, dengan `norm_methods=['none']` dan *bagging* 4 seed.
- **TabPFN-3.5** (checkpoint Prior Labs, lisensi diterima, token disimpan sebagai Kaggle Secret). Dari TabPFN 2.5, 2.6, 3, dan 3.5, hanya 3.5 yang kompetitif.
- **SVR-plateau.** SVR RBF pada view PCA *whitened* dengan target log(y − 30), dirata-ratakan di grid C ∈ {10, 30} × skala ∈ {0,7; 1,0} × 5 dimensi PCA. Rata-rata atas "dataran" setting ini membuat error-nya lebih independen dari model *foundation*.

**Pseudo-labelling transduktif.** Baris query (fold validasi, atau data uji) pertama-tama diberi label oleh model. Lalu baris query dibagi ke k bagian, dan setiap bagian diprediksi ulang dengan bagian-bagian lain (yang sudah ber-*pseudo-label*) ditambahkan ke konteks. Dengan cara ini tidak ada baris yang pernah melihat *pseudo-label*-nya sendiri, dan tidak ada label test yang dipakai. Tujuannya memadatkan konteks di sekitar setiap query, karena analisis variogram menunjukkan error didominasi kepadatan sampel.

Prediksi TabICLv2-PL dan TabPFN-3.5-PL dirata-ratakan 50:50 (*FM-PL-mean*). Bobot NNLS di antara keduanya tidak stabil karena korelasi residualnya 0,98. Prediksi fitur akhir adalah 0,7 × FM-PL-mean + 0,3 × SVR-plateau-PL.

## 4.2 Lapisan urutan (v9)

Dari posisi relatif setiap baris test dibangun empat kandidat prediksi:

1. *Quantile map*: kurva isotonic menurun dari target train terhadap posisi relatif.
2. Proyeksi isotonic dari prediksi fitur.
3. Prediksi fitur yang diurutkan ulang sesuai urutan file.
4. Prediksi fitur mentah.

Tujuh aturan rata-rata ditetapkan sebelum evaluasi. Aturan dengan RMSE fold rata-rata terendah adalah **mean(quantile map, prediksi fitur terurut)**. Tidak ada bobot yang di-*fit* pada fold yang sedang dievaluasi. Alasannya, *quantile map* fold lain dibangun dari target fold yang dievaluasi, sehingga *fitting* bobot di sana akan membocorkan label.

## 4.3 Lapisan koreksi (v10–v13)

Lapisan urutan masih menyisakan dua sumber error:

- Posisi relatif sebuah baris di filenya sendiri hanya taksiran dari posisinya di data penuh.
- Target tidak monoton sempurna terhadap urutan; ada penyimpangan lokal yang mungkin dijelaskan oleh fitur.

Empat tahap menangani kedua error ini. Setiap tahap memilih satu kandidat dari katalog tetap lewat *inner* 5-fold CV pada baris latih *outer*. Setelah itu prediksinya dirata-ratakan **50:50 dengan versi sebelumnya**, dengan bobot yang ditetapkan sejak awal.

| Tahap | Ide | Pilihan akhir (CV pada seluruh train) |
|---|---|---|
| v10 | Koreksi rank: posisi query ditaksir dari posisi 10 tetangga terdekat di ruang PCA-12 *whitened*, dikalibrasi dengan tren *leave-one-out*, lalu dihaluskan dan dibuat monoton | k = 10, *strength* 1,0, *smoothing* 0,005 |
| v11 | *Detrending* Gaussian pada fitur dan target, lalu Ridge pada residual fitur untuk memprediksi penyimpangan target | Ridge α = 10, bobot 1,0 |
| v12 | *Detrending local-linear* (bias lebih kecil di ujung file) ditambah interaksi residual × posisi | *local-linear* + interaksi, Ridge α = 10 |
| v13 | *Kernel Ridge* RBF pada residual yang sama (48 input) | α = 0,1, γ = 2/48 |

Karena setiap tahap dirata-ratakan 50:50, prediksi akhir tersusun dari:

v13 = 1/16 · v9 + 1/16 · koreksi rank + 1/8 · komponen v11 + 1/4 · komponen v12 + 1/2 · komponen kernel.

Tahap v10–v13 hanya memakai fitur, target train, dan urutan file. Model *foundation* masuk ke prediksi akhir terutama lewat v9.

# 5. Strategi Validasi

- **Fold tetap.** Semua model dan tahap memakai `RepeatedKFold(5 fold × 3 repeat, seed 42)` yang sama, sehingga perbandingan antarversi berpasangan. Di dalam setiap split, urutan file dipertahankan dan posisi dinormalisasi per partisi, meniru file test yang terpisah.
- **Nested selection.** Setiap pilihan kandidat dibuat oleh *inner* CV yang hanya melihat baris latih *outer*. Bobot campuran tidak pernah di-*fit* pada label validasi.
- **Bootstrap berpasangan per record.** Ketiga prediksi repeat dari satu record di-*resample* bersama-sama, menghasilkan interval perubahan RMSE.
- **Simulasi public/private.** Blok query 96 baris dibagi menjadi 29/67 baris yang saling lepas, ratusan kali, untuk melihat sebaran skor dan ekor atasnya (persentil 95), bukan hanya rata-ratanya.
- **Grouped split.** Baris yang fiturnya berdekatan dikelompokkan memakai *connected components* (tanpa melihat target), dan setiap kelompok selalu berada di sisi split yang sama.
- **Learning curve.** Mengukur apakah error masih turun ketika data latih bertambah.
- **Aturan adopsi.** Pada fase fitur saja, perubahan hanya diadopsi bila gain nested > 0,01 dan P(bootstrap lebih baik) > 0,9.

# 6. Hasil

| Versi | Perubahan | RMSE CV | Public RMSE |
|---|---|---:|---:|
| v1 | TabICLv2 + SVR multi-PCA, blend NNLS | 7,41* | 7,947 |
| v2 | TabICLv2 tanpa normalisasi *power*, *bagging* 4 seed | 7,40* | 7,902 |
| v3 | *Pseudo-labelling* TabICLv2 | 7,32* | 7,888 |
| v5 | Rata-rata TabICLv2-PL dan TabPFN-3.5-PL | 7,30* | 7,888 |
| v6b | SVR-plateau dengan *clipping* | 7,26* | 7,837 |
| v8 | *Pseudo-labelling* untuk SVR | 7,23* | 7,900 |
| v9 | Pemetaan urutan baris | 3,613 | 2,387 |
| v10 | Koreksi rank dari tetangga fitur | 3,401 | 2,107 |
| v11 | Ridge pada residual ter-*detrend* | 3,303 | 1,955 |
| v12 | *Local-linear* + interaksi posisi | 3,179 | 1,825 |
| v13 | *Kernel Ridge* pada residual (**final**) | 3,016 | 1,722 |

\*RMSE blend nested (estimand v1–v8). Mulai v9, angkanya rata-rata RMSE dari 15 fold *outer*. Kedua estimand tidak dapat dibandingkan secara langsung.

Rincian validasi v13 terhadap v12:

- Lebih baik di **15/15 fold** dan di ketiga repeat (repeat 1: 3,067 → 2,870; repeat 2: 3,412 → 3,308; repeat 3: 3,181 → 2,998).
- Interval bootstrap perubahan RMSE: [−0,237; −0,073].
- Pada simulasi query 96 baris dengan seleksi nested penuh, komponen baru menurunkan rata-rata RMSE query dari 3,485 ke 3,200, persentil 95 RMSE private simulasi dari 4,779 ke 4,602, dan rata-rata selisih absolut public–private dari 0,827 ke 0,752.
- Pada *grouped split* (ambang jarak 0,35), komponen kernel 3,02 sedangkan residual linear 3,38. Gain-nya bertahan walaupun baris yang mirip dipisah ke sisi yang sama.
- *Learning curve* komponen kernel turun dari 3,73 (143 baris latih) ke 3,12 (286 baris). Error belum mencapai *plateau*.

Skor public bergerak ke arah yang sama dengan CV di setiap versi sejak v9. Meski begitu, skor public tidak dipakai untuk memilih apa pun. Gain CV juga tidak boleh dikurangkan dari skor public untuk "meramal" skor baru, karena 29 baris public sangat bising.

# 7. Pembahasan dan Keterbatasan

- **Ketergantungan pada urutan file.** Lapisan v9–v13 mengasumsikan `test.csv` diurutkan dengan cara yang sama seperti `train.csv`. Bukti tidak langsungnya kuat (Bagian 2), dan penggunaannya dikonfirmasi boleh oleh panitia. Namun metode ini tidak akan berlaku untuk data masa depan yang tidak terurut. Tanpa informasi urutan, solusi terbaik kami berbasis fitur saja (v8) berada di RMSE CV sekitar 7,2.
- **Ekor atas tetap paling sulit.** Pada OOF v12, 38 baris pertama (target tertinggi) masih menyumbang sekitar 36% *squared error*. Error di bagian ini muncul dengan dua tanda, jadi koreksi yang hanya menaikkan prediksi tidak dibenarkan.
- **Bias seleksi riset.** Validasi nested menghilangkan kebocoran seleksi parameter di dalam fold. Namun katalog kandidat dirancang setelah mempelajari 382 baris yang sama, dan tidak ada *holdout* lokal yang benar-benar belum tersentuh. Angka CV tetap estimasi pengembangan.
- **Pseudo-label.** Lapisan fitur memakai *pseudo-labelling* transduktif pada fitur test, tanpa label test apa pun. Seluruh pipeline karena itu tidak bebas *pseudo-label*.

# 8. Kepatuhan terhadap Aturan

- Hanya dataset resmi kompetisi yang dipakai. Tidak ada data eksternal, tidak ada upaya mengidentifikasi entitas asli, dan tidak ada rekonstruksi target test.
- Tidak ada *probing* leaderboard. Skor public hanya dicatat, tidak pernah dipakai untuk memilih parameter, bobot, atau mengubah prediksi per baris.
- `record_id` tidak diubah, dan file submission mengikuti `submission_template.csv`.
- Penggunaan urutan baris diungkapkan secara terbuka di notebook dan laporan ini, setelah dikonfirmasi oleh panitia pada 4 Oktober 2026.
- Model *pretrained* (TabICLv2, TabPFN-3.5) adalah library ML publik. Bobotnya tidak dilatih pada data yang berkaitan dengan target kompetisi.
- AI/LLM dipakai sebagai alat bantu *coding*, *debugging*, analisis, dan dokumentasi, sesuai aturan kompetisi.

# 9. Reproduksi

- **Notebook:** `notebooks/final.ipynb`, dibangun dari `notebooks/build_final.py`. Struktur: Pendahuluan, Inisialisasi, EDA (termasuk Dashboard EDA), Prapemrosesan, Skema Validasi, Pemodelan, Inferensi dan Submisi, Dashboard Solusi, Pasca-Analisis, Daftar Pustaka.
- **Mode `INFERENCE` (bawaan):** memuat keluaran lapisan fitur dari *run* GPU yang dibekukan (`artifacts/`: `feature_oof.npz`, `submission_v9.csv`, `benchmark_oof.csv`, dan `submission_v13.csv` sebagai referensi verifikasi), lalu menghitung ulang seluruh EDA, lapisan urutan, dan tahap v10–v13 di CPU dalam sekitar 75 detik. Hasilnya **identik per byte** dengan `submission_v13.csv`, dan notebook berhenti dengan *error* bila selisihnya melebihi 1e-6.
- **Mode `FULL`:** melatih ulang TabICLv2, TabPFN-3.5, dan SVR dengan *pseudo-labelling* di Kaggle 2× GPU T4 dengan internet aktif. Token Prior Labs disimpan sebagai Kaggle Secret `TABPFN_TOKEN`.
- **Seed:** fold `RepeatedKFold` seed 42. Seed *inner* per tahap adalah 100+i, 5200+i, 560100+i, dan 6000+i. Seed seleksi akhir adalah 4900, 5299, 560199, dan 6099.
- **Determinisme:** tahap v10–v13 deterministik. Model *foundation* di GPU dapat berbeda sangat sedikit antar-run, tetapi porsinya di prediksi akhir hanya 1/16.
- **Submission yang diserahkan:** SHA256 `0bd8ba26554f5163d14bfedbc70b9a134cfa458505e268c6ffb105e6cd07af14`.
- **Script analisis** (setiap script mencetak angka dan menyimpan gambar): `eda/` (EDA dasar), `eda/v2`–`eda/v9` (eksperimen per ronde, termasuk `58_deep_audit.py`, `59_preprocessing_ablation.py`, `60_nested_and_group_check.py`, `61_kernel_sensitivity.py`, dan `63_coherent_nested_queries.py`). Catatan lengkapnya ada di `docs/eda_findings.md` dan `docs/v10_*.md` sampai `docs/v13_*.md`.

# Referensi

1. FIK FAIR 2026. *Data Science Competition FIK FAIR 2026*. Kaggle, 2026. https://www.kaggle.com/competitions/data-science-competition-fikfair2026
2. Qu, J., Holzmüller, D., Varoquaux, G., Le Morvan, M. *TabICL: A Tabular Foundation Model for In-Context Learning on Large Data*. ICML 2025. (Paket `tabicl`, versi v2.)
3. Hollmann, N. et al. *Accurate predictions on small data with a tabular foundation model*. Nature 637, 2025. (Paket `tabpfn`, checkpoint TabPFN-3.5 dari Prior Labs.)
4. Pedregosa, F. et al. *Scikit-learn: Machine Learning in Python*. JMLR 12, 2011. Dokumentasi Kernel Ridge, Isotonic Regression, dan cross-validation untuk data berkelompok.
5. Lee, D.-H. *Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks*. ICML Workshop, 2013.

## Lampiran gambar yang disarankan

| Gambar | File |
|---|---|
| Urutan baris: posisi vs target, posisi test vs prediksi fitur, drift fitur | `results/v9/fikfair_v9/row_order.png` (atau output `final.ipynb`) |
| Validasi aturan urutan di 15 fold | `results/v9/fikfair_v9/order_simulation.png` |
| Audit duplikat, pasangan terdekat, korelasi | `outputs/deep_v13/audit.png` |
| Preprocessing sebelum/sesudah *detrending* | `outputs/deep_v13/preprocessing.png` |
| Ablasi preprocessing | `outputs/deep_v13/ablation.png` |
| Sensitivitas kernel (heatmap α × γ) | `outputs/deep_v13/sensitivity.png` |
| Nested, grouped split, *learning curve* | `outputs/deep_v13/confirmation.png` |
| Simulasi public/private | `outputs/deep_v13/coherent_nested.png` |
| Prediksi akhir sepanjang file test | `outputs/fikfair_v13/v13_predictions.png` |
