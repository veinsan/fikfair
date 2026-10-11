# Temuan EDA & Eksperimen — FIK FAIR DSC 2026

Semua angka dari `eda/*.py` (jalankan dari root: `.venv/bin/python eda/01_overview.py`, dst). Gambar di `outputs/eda/`.
CV = 5-fold × 3 repeat (fold identik untuk semua model). Baseline "tebak rata-rata" = RMSE 12.89.

| Script | Temuan | Keputusan |
|---|---|---|
| 01_overview | Tanpa missing/duplikat; tiap fitur 382 nilai unik; `record_id` = D0001–D0478 diacak, ρ(id,target)=0.05 | `record_id` dibuang |
| 01_overview | Target bimodal (~37 & ~57), lantai 32.6, ekor kanan s/d 100 (skew 0.75) | Uji log(y−30) per model |
| 02_shift | 9/24 fitur KS p<0.05, tapi shift kecil (≤0.3 SD); adversarial AUC 0.53–0.59 | Tanpa reweighting |
| 03_correlation_pca | Kolinear kuat (44 pasang \|r\|>0.7, max 0.94). 4 PC = 91% varians, hanya 4 eigen di atas noise edge Marchenko–Pastur | Fitur memang proyeksi low-dim |
| 03_correlation_pca | Linear di top-4 PC = 11.98, di 24 PC = 10.67 → sinyal target ada di PC varians kecil | Jangan reduksi agresif |
| 04_target_relationship | Banyak fitur berbentuk **inverted-U** (f01, f06, f14, f20). Ridge 10.73 → additive spline 9.36 | Butuh model non-linear |
| 04_target_relationship | Residual linear: under-predict ekor atas (+18 di desil teratas) | Ekor = sumber error utama |
| 05_neighbors | Jarak 1-NN test→train ≈ train→train; cuma 48 baris punya tetangga < 0.5 | Random KFold representatif |
| 06_preprocessing | Quantile / PCA-denoise / ICA: tidak ada gain nyata vs raw (selisih ≤0.15, dalam noise). PCA k=4 merusak (>10) | Raw untuk tree/FM, PCA-12 untuk kernel |
| 06_preprocessing | ICA source heavy-tailed (kurtosis 4–9) → **gagal** membalik proyeksi | ICA dibuang |
| 06_preprocessing | Log-target awalnya bikin SVR rusak (8.9→12.0) karena `epsilon=1` di skala log. Setelah target di-standardize: SVR 8.83→8.35, kNN membaik, ExtraTrees memburuk (8.92→9.37) | Transform target per keluarga model |
| 06b_svr_tuning | Kurva RMSE vs dim PCA bergerigi (k12=8.13, k14=8.62) | Rata-rata SVR k∈{10,12,14,16,20}: ≤ single k di 3 seed CV |
| 07_model_benchmark | TabICLv2 **7.63**, TabPFNv2(16) 7.92, SVR-multiPCA 8.11, GP-ARD(log) 8.58, CatBoost 8.90, ExtraTrees 8.89, LightGBM 9.02, Ridge 10.77 | Foundation model unggul |
| 07_model_benchmark | TabICL n_est=32 (7.65) dan +fitur PCA (7.69) tidak lebih baik | Default TabICL |
| 07_model_benchmark | GroupKFold (klaster similarity) memburukkan semua model 0.3–1.2, ranking tetap; TabICL paling stabil | Pilihan model robust |
| 08_ensemble | Korelasi residual TabICL–TabPFN 0.96; SVR paling beda (0.84) | NNLS: TabICL ≈0.72 + SVR ≈0.28 → nested **≈7.41** |
| 08_ensemble | 26 baris target>70 (7%) = ~41% SSE, bias +13 s/d +20 | Batas bawah error dari data |
| 08_ensemble | Linear stretch (slope 1.09) cuma −0.007 RMSE nested | Tidak dipakai |
| 08_ensemble | Public LB ~29 baris: RMSE model yang sama bisa 4.7–9.9 (interval 90%) | Pilih pakai CV, LB cuma sanity check |

## Soal "probing" leaderboard
Submission yang sengaja didesain buat mengekstrak info ground-truth test (mis. prediksi konstan / perturbasi per-baris buat nebak target test) masuk kategori **"eksploitasi terhadap sistem evaluator"** (Rules §11.2) → risiko diskualifikasi. Lagipula dengan ~29 baris public, sinyalnya tenggelam di noise (lihat tabel di atas). Analisis "salahnya di mana" dilakukan secara sah lewat residual OOF (08).

---

# v2: Deep Analysis Round 2 (setelah v1 public 7.947)

Script di `eda/v2/` (jalankan dari root: `.venv/bin/python eda/v2/08_reverse_engineering.py`, dst). Gambar `outputs/eda/v2_*.png`.

| Script | Hipotesis → ekspektasi teori | Hasil evaluasi | Keputusan |
|---|---|---|---|
| 08 (A) | Standardisasi pakai semua 478 baris → mean 0/sd 1 di train+test | Train saja: mean ±0.06; train+test: ±0.001 persis | Fit transform unsupervised di train+test = konsisten dengan pipeline panitia |
| 08 (B) | Kalau proyeksi dari k<24 indikator → eigen terkecil datar (noise floor) | Spektrum turun mulus, tidak ada floor | Full rank; tidak ada dimensi murni noise |
| 08 (C) | Source rank+bounded → flat (kurtosis<0), ICA-24 bisa pulihkan | 22–23 dari 24 source heavy-tailed (median kurtosis 5–9) | Proyeksi tidak bisa dibalik (indikator saling berkorelasi) → dibuang |
| 08 (D) | SIR/SAVE: target hidup di subruang kecil | 6 arah signifikan vs null permutasi, TAPI SVR di subruang SDR 9.6–10.6 vs 8.13 di PCA | Estimasi arah supervised terlalu noisy di ~300 baris/fold → dibuang |
| 09 | U-terbalik di banyak fitur = target turun kalau record menjauh dari pusat ("typicality") | Mahalanobis ρ=−0.39 (setara fitur terbaik). Hubungan satu arah: atipikal → pasti rendah; tipikal → bebas | Bantu Ridge (−0.32), CatBoost (−0.18); SVR +0.20; TabICL −0.007 → tidak dipakai (TabICL sudah menangkapnya) |
| 10 (a) | Variogram: nugget = noise tak tereduksi | Nugget ≈ 10 (RMSE ~3.2); MSE blend (54) tercapai di jarak ≈1.0 = median jarak tetangga terdekat | Error dibatasi kepadatan sampel, bukan model |
| 10 (b) | Kalau masih ada sinyal, residual OOF bisa diprediksi | CV R² residual ≤ 0 untuk Ridge/kNN/CatBoost (juga dengan pred sebagai input) | Informasi fitur sudah habis |
| 10 (c) | "Tail under-predicted" (v1) = bias yang bisa dikoreksi? | Kalibrasi per desil prediksi: gap ≤1.5 (9/10 desil), desil atas 0.02 | Itu regression-to-the-mean, bukan bias → koreksi narasi v1 |
| 11 | Knob internal TabICL | Semua dalam ±0.05. `norm=none`: 7.601 vs 7.634 (1 seed) dan 7.612 vs 7.647 (4 seed) | Pakai `norm=none`, bag 4 seed (seed 42 v1 ternyata "beruntung") |
| 12 | TabDPT (pretrain data riil) → error beda | RMSE 7.80, tapi korelasi residual vs TabICL 0.973 | Bobot NNLS 0 |
| 14 | Fine-tune TabICL (pinball loss, early stop di 10% fold train) | lr=1e-5: 7.694 (menang 1/15 fold); lr=3e-6: 7.627; freeze col+row: 7.654; zero-shot 7.601 | Makin kecil update makin dekat zero-shot → prior pretrained > belajar dari 382 baris. Tidak dipakai |
| 13 | Protokol seleksi (nested 5 seed, bootstrap, simulasi LB, adversarial) | v2 (TabICL-bag norm=none + SVR) nested 7.396 vs v1-bag 7.427; P(menang) public-29 ≈0.60, private-67 ≈0.64 | Pilih by CV; public LB tidak bisa mengkonfirmasi beda sekecil ini |

**Hasil notebook v2 (lokal CPU, identik dengan yang akan jalan di Kaggle):** TabICL-v2 7.612 vs v1 7.634; GroupKFold 7.896 vs 7.952; nested blend 7.396 vs 7.411 (bobot TabICL 0.725 + SVR 0.275). Prediksi test v2 vs v1: korelasi 0.9995, RMS beda 0.38 → public score v2 akan hampir sama dengan v1 (beda ± noise), dan itu **wajar**.

**Kesimpulan "mentok":** 3 foundation model dengan prior berbeda (sintetis TabICL/TabPFN, riil TabDPT) konvergen ke error yang sama (korelasi residual 0.96–0.97), residual tidak bisa dipelajari, dan variogram menunjukkan akurasi = resolusi sampling data. Gain yang tersisa hanya reduksi varians (seed bagging, `norm=none`) ≈ 0.03 RMSE.

---

# v3: Round 3 (setelah v2 public 7.902; Top-1 public 7.866)

Script di `eda/v3/` (helper `cvlib.py`). Gambar `outputs/eda/v3_*.png`. Semua TabICL pakai `norm=none`, fold 5×3 yang sama.

| Script | Hipotesis → ekspektasi teori | Hasil evaluasi | Keputusan |
|---|---|---|---|
| 15 views | Proyeksi panitia arbitrer & TabICL tidak rotation-invariant → basis lain (PCA/rotasi/ICA) = "augmentasi", error beda | PCA 7.72, rotasi acak 7.54–7.70, ICA 7.96 vs raw 7.60; korelasi residual rotasi vs raw 0.99; PCA = PCA-whitened persis (sanity: TabICL standardize kolom) | Basis panitia **bukan** arbitrer buat model; rotasi = cuma seed tambahan → tidak dipakai |
| 16A TTA jitter | Rata-rata prediksi atas salinan query ber-noise menghaluskan "controlled perturbation" | σ=0.03: −0.025 (noise), 0.1: +0.18, 0.3: +1.34; norm jitter σ=0.1 ≈0.5 ≈ sebagian besar jarak tetangga | Perturbasi panitia kecil, jitter mengaburkan sinyal → ditolak |
| 16B target | Log-target v1 jelek karena exp(E[log y]) < E[y] (Jensen) | exp(mean) 7.77 → via quantile 7.62 (diagnosis terbukti); rank-gauss 7.63; y mentah 7.60 | Diagnosis benar, tapi transform tetap tidak menang → y mentah |
| 16C/17 pseudo-label | Baris query yang di-pseudo-label memadatkan context (batas = kerapatan sampel, v2 variogram) | single seed 7.512 vs 7.601; kontrol seed 43: 7.610 (bukan efek seed); bag4: halves 7.547, **4-split 7.522** vs 7.612, menang 12/15 fold | **Dipakai** |
| 17 mekanisme | Kalau benar soal kerapatan, gain ~0 untuk baris yang punya tetangga dekat | Reduksi MSE per jarak ke train terdekat: <0.8: 0.02; 0.8–1.2: 1.07; **1.2–1.8: 3.35**; >1.8: 1.29 | Mekanisme terverifikasi, bukan leak |
| 18 sweep | Lebih banyak pseudo row → lebih baik, sampai jenuh | fraksi 0 / 0.5 / 0.75 / 0.875: OOF 7.429 / 7.357 / 7.332 / 7.321; blend 7.396 / 7.338 / 7.316 / 7.308 | Jenuh; pilih 4-split (biaya ½, rasio emulasi CV↔test 0.19 vs 0.19) |
| 19 seleksi | v3 vs v2 (submission aktif) | nested **7.316 vs 7.396** (sd 0.007); P(better) bootstrap 0.974; P(win) public-29 0.72, private-67 0.82; RMS beda test vs v2 0.54 | v3 = kandidat submit |

**Aturan & leakage:** label test tidak pernah dipakai; baris query tidak pernah melihat pseudo-label-nya sendiri (kalau iya, TabICL tinggal menyalin); hanya fitur test yang dipakai (semi-supervised standar), tanpa data eksternal.

---

# v4: Round 4 (setelah v3 public 7.888; v1→v2→v3 = 7.947→7.902→7.888, urutan sama dengan CV)

Script di `eda/v4/`. Gambar `outputs/eda/v4_*.png`. Fokus: apakah gain pseudo-label (PL) v3 bisa dipercaya, dan apakah masih ada yang bisa ditambah.

| Script | Hipotesis → ekspektasi teori | Hasil evaluasi | Keputusan |
|---|---|---|---|
| 20 learning curve | PL membantu karena data berlabel jarang → gain bisa MENGECIL di n=382 (test) vs n=306 (CV), jadi CV optimistis | Gain PL per n: 120: −0.02 (40% fold), 180: +0.07, 240: +0.12, 306: +0.08 | Gain tidak menyusut; PL butuh first-pass yang cukup akurat (n≳180). CV tidak optimistis karena ukuran context |
| 21 robust PL | PL kalah di GroupKFold = confirmation bias → pakai hanya pseudo-label yang yakin (IQR dari quantile TabICL) | IQR memprediksi error (ρ=0.31). keep 1.0: random +0.095 / group −0.119; 0.75: +0.060 / −0.031; 0.5: +0.026 / −0.003 | Trade-off satu garis, tidak ada yang menang di dua skema |
| 22 geometri test | PL gagal kalau query jauh dari data berlabel & saling bergerombol. Test mirip skema mana? | Median jarak ke berlabel / ke sesama query: test 1.24/1.83, random 1.21/1.81 (KS 0.08/0.07), group 1.47/1.48 (KS 0.39/0.30) | Test ≈ random folds → skenario group tidak mewakili test → PL penuh dipertahankan |
| 22 hedge | α·PL + (1−α)·plain sebagai kompromi | Blend+SVR: random gain +0.023…+0.085, group −0.006…−0.088 (monoton) | α=1 (sesuai geometri test) |
| 23 ekstensi | SVR transduktif (SVR juga dapat pseudo row); PL 2 putaran | SVR 8.11→8.04 tapi blend −0.005; 2 putaran −0.006 (sd antar seed ≈0.007) | Ditolak (di bawah noise) |

**Kesimpulan round 4:** v3 tervalidasi dan tetap model final. Notebook v4 = model v3 (prediksi identik) + bukti di atas, untuk dikumpulkan ke panitia.

---

# v5: Round 5 (checkpoint gated Prior Labs: TabPFN 2.5 / 2.6 / 3 / 3.5)

Script di `eda/v5/`. Gambar `outputs/eda/v5_*.png`. Token Prior Labs dibaca dari `~/.cache/tabpfn/auth_token` (lokal) atau Kaggle Secret `TABPFN_TOKEN`.

| Script | Hipotesis → ekspektasi teori | Hasil evaluasi | Keputusan |
|---|---|---|---|
| 24 zoo (plain) | Checkpoint lebih baru/khusus data kecil lebih akurat atau error-nya beda | 2.5: 7.85, 2.5-small 7.85, Real-2.5 7.83, Real-var 7.88, low-skew 7.79, 2.6: 7.84, 3: 7.70, **3.5: 7.66**, 3.5-fast 7.80 (TabICL 7.61). Korelasi residual vs ICL-PL 0.94–0.98; semua bobot blend ~0 | Hanya TabPFN-3.5 kompetitif |
| 24 zoo (PL) ungated | PL juga membantu model lain | TabPFN-v2 7.97→7.89, TabDPT 7.80→7.68, tapi blend v3 tidak membaik | Tidak dipakai |
| 25 TabPFN-3.5 + PL (3 repeat) | PL mengangkat 3.5 setara TabICL-PL; rata-rata dua prior = varians turun | PFN35-PL vs ICL-PL: −0.02 / −0.04 / +0.11 per repeat. PL dengan label dari TabICL (bukan dirinya) lebih buruk | Tiap model perlu pseudo-label sendiri |
| 26 seleksi | Blend NNLS 3 kolom | NNLS nested **lebih buruk** (7.358 vs 7.316): bobot ICL/PFN berayun 0.2–0.75 antar fold karena residual 0.98 (kolinear) | Diagnosis: instabilitas bobot, bukan model |
| 26 fix | Prediktor kolinear → rata-rata 50:50, fit hanya bobot SVR (1 derajat bebas) | Per repeat 7.468 → **7.443 (−0.025)**; pooled 7.316 → **7.296 (−0.020)**; bootstrap P(better) 0.88; P(win) public-29 0.59, private-67 0.62 | **v5 = mean(ICL-PL, PFN35-PL) + SVR** |
| 26 cek | Tambah FM ketiga (TabDPT-PL / TabPFN-v2-PL) ke rata-rata | 7.339 / 7.339 (lebih buruk) | Hanya pasangan yang sama kuat |

**Ekspektasi public:** 7.888 − ~0.02 ≈ 7.865, tepat di sekitar Top-1 (7.866); peluang terlihat menang di 29 baris public ≈ 0.59. Gain nyata tapi kecil.

---

# v6: Round 6 (setelah v5 public 7.88818 = seri dengan v3 7.88781; peringkat 3)

Script di `eda/v6/`. Gambar `outputs/eda/v6_*.png`. Arah: semua foundation model salah di baris yang sama (korelasi residual 0.95–0.98), jadi yang dicari adalah **anggota blend dengan error berbeda**.

| Script | Hipotesis → ekspektasi teori | Hasil evaluasi | Keputusan |
|---|---|---|---|
| 27 NN dari nol (pytabkit) | MLP/TabM/kernel-metrik punya bias induktif beda → error beda | RealMLP-TD 8.82 (corr 0.886), RealMLP-S 9.49, TabM 9.94 (0.828), xRFM 8.58 (0.895); blend v5 tidak membaik | Terlalu lemah di ~300 baris → ditolak |
| 28 local-context ICL | Target bervariasi lokal → context = k tetangga terdekat mempertajam | k=all 7.525, k=200 7.649, k=100 7.887; corr vs global 0.97–0.99 | Baris jauh tetap membawa informasi → ditolak |
| 29 audit record_id | Kalau id berurutan = entitas sama, random KFold bocor lewat tetangga id | Autokorelasi target lag 1–5 dalam pita null; jarak fitur baris id-bertetangga = acak (p=0.56); runs test z=+0.65 | Id bersih, CV aman |
| 30 kernel dari variogram | Variogram linear di origin → proses kasar → kernel Laplacian/Matern-½ > RBF | Laplacian 8.51, Matern-½ 8.35, Matern-3/2 8.04, RBF 8.09 | Teori gagal (kekasaran = noise target) → RBF |
| 30/31 regularisasi SVR | Anggota blend dinilai dari beda error-nya: SVR kurang teregularisasi lebih independen | C=1/3/10: blend 7.342/7.296/7.261; dataran C∈{10,30} × skala∈{0.7,1.0}: pooled 7.252–7.262, per-repeat 7.412–7.420 (v5: 7.296 / 7.443); RMSE SVR sendiri tidak membaik | Rata-rata 4 sel dataran |
| 32 seleksi v6 | v6 = mean(ICL-PL, PFN35-PL) + SVR-dataran | pooled **7.251** vs v5 7.296 vs v3 7.316; per-repeat 7.411 vs 7.443 vs 7.468 (lebih baik di 3/3 repeat); bootstrap P(better than v5) 0.84; P(win) public-29 0.51, private-67 0.56 | **v6 = kandidat** |

**Catatan public v5:** RMS beda prediksi v5 vs v3 = 0.58 tapi skor public hanya bergeser +0.0004 → konsisten dengan simulasi (peluang menang di 29 baris ≈ 0.59).

---

# v6b & v7: Round 7 (v6b public 7.837; Top-1 7.800)

**v6b** = v6 + tiap anggota SVR di-clip ke rentang target train (di v6 tanpa clip, anggota SVR memprediksi sampai 138 untuk D0263; CV tidak bisa memvalidasi ekstrapolasi itu). CV setara (7.2555 vs 7.2511), D0263 93.3 → 91.0.

Script di `eda/v7/`. Gambar `outputs/eda/v7_*.png`.

| Script | Hipotesis → ekspektasi teori | Hasil evaluasi | Keputusan |
|---|---|---|---|
| 33 pustaka anggota | Variasi target (log/sqrt/raw), view (PCA/raw-24D/PCA-24), loss (SVR/KRR), lokalitas (kNN), GP, tree → error berbeda → anggota ke-3 | 10 kandidat, semua delta blend ≥ 0 (pooled dan per-repeat); tidak ada yang lebih baik sebagai pengganti SVR-dataran | Sisi non-FM jenuh |
| 34 subagging context | OOF gabungan 0.16–0.19 lebih baik dari per-repeat = varians komposisi context besar → rata-rata model di subsample menghilangkannya | f=0.9: +0.020, f=0.8: +0.151, f=0.65: +0.290; seed-bag 8× di 100%: −0.006 | Bias context kecil > varians; selalu pakai 100% data |
| 35 A dekomposisi | Dari mana gain v5→v6b? | Baris target>70: −1022 SSE; 50–60: +515. Per prediksi: gain di baris yang diprediksi >70 | Gain = ekor |
| 35 B separabilitas ekor | Apakah ekor bisa dikenali? | AUC blend untuk y>70 = 0.87; ekor: target 80.5 vs prediksi 64.7; top-10 prediksi terkalibrasi (76.1 vs 74.8) | Sebagian ekor tak terlihat dari fitur |
| 35 C SVR berbobot ekor | sample_weight ke target ekstrem → kurang shrinkage | Delta ≥ 0 di semua λ | Ditolak |
| 36 classifier ekor | Classifier khusus > regressor dalam mengenali ekor | y>75: AUC 0.956 vs 0.888, tapi y>65/70 lebih buruk; stacking [blend, p] +0.04…+0.10 | Ditolak |
| bobot blend | Bobot di-fit di OOF gabungan; model akhir fit sekali → bobot SVR mungkin beda | per-repeat 0.278 vs gabungan 0.273; kurva datar 0.27–0.30 | Sudah optimal |
| PL 8-split di v6b | Lebih banyak pseudo row | −0.003 | Noise |

**v7** (hanya bisa diuji di GPU): TabICL-PL 8 seed, TabPFN-3.5-PL 16 estimator, TabPFN-3-PL sebagai anggota ketiga rata-rata FM. Aturan adopsi: ganti v6b hanya jika nested RMSE turun > 0.015.

---

# v8: Round 8 — mencari "satu faktor" di balik public 2.28 / 2.36 / 2.72 (tim lain ≥ 7.71)

Script di `eda/v8/`. Gambar `outputs/eda/v8_*.png`.

| Script | Hipotesis | Hasil | Keputusan |
|---|---|---|---|
| 37 F1 | Proyeksi hampir singular → info target di arah varians kecil | PC5–24 (≈9% varians fitur) menjelaskan 25% varians target secara linear vs PC1–4 15%; PC12/14/15/23/24 signifikan p<0.01 | Nyata, tapi model non-linear sudah memakainya |
| 37 F2–F4 | Basis/metrik lain membuat target deterministik | OLS putih = OLS mentah 10.75; aditif PC 9.78 (raw 9.26); 1-NN Mahalanobis 10.03 | Tidak ada |
| 37 F3 | Membalik transformasi rank | OLS pada rank 10.04 vs 10.75 mentah | Jejak rank nyata, tapi kecil |
| 37 F5 | Bocoran di digit/pembulatan | Digit terakhir seragam (0 tidak ada = trailing zero), korelasi digit–target ≤ 0.10 | Bersih |
| 38 | Sumber bounded (rank→bounded) bisa dipulihkan via kurtosis minimum | Arah kurtosis −1.49 (null Gaussian −1.12) = **bimodal** (struktur klaster), bukan sumber seragam | Sumber tidak terpulihkan |
| 39 | Apakah RMSE 2.3 mungkin dari fitur saja? (pasangan kembar) | Batas bawah tersirat 1.7–2.6 (CI lebar), ekstrapolasi ke jarak 0 ≈ 0 | **Tidak terbukti mustahil** — klaim round 2 "noise ~3" terlalu kuat |
| 40 | Target = Σ a·h(v·x+c) (inversi proyeksi + un-bounding) | Jaringan terstruktur 9.34–11.62 vs TabICL 7.53 | Hipotesis struktur gagal |
| 41 | Salin target kembaran Mahalanobis | Blend sudah setara untuk baris berkembaran; aturan 50/50: 7.3985 → 7.3951 (noise); hanya 3–7 baris test berkembaran | Ditolak |

**Interpretasi jujur:** tidak ditemukan struktur sah yang menurunkan CV di bawah ~7.25. Pola skor (tiga tim 2.3–2.7, sisanya ≥ 7.7; 2.3 ≈ batas bawah noise antar-kembaran; target maks 100.44 > 100 = indikasi skor 0–100 yang diberi perturbasi) **konsisten dengan** rekonstruksi target dari sumber data publik — yang dilarang (§2.2, §3, §11.1). Ini tidak terbukti; ada kemungkinan struktur sah yang belum kami temukan.

## Round 8 lanjutan (target 2.x)

| Script | Hipotesis | Hasil | Keputusan |
|---|---|---|---|
| 42 deflasi sumber | Histogram bimodal = tanda transformasi bounded (tanh); deflasi kurtosis memulihkan indikator satu per satu | Hanya 2 dari 24 arah lebih sub-Gaussian dari null; sumber tidak berbagi bentuk marginal; GAM di sumber 10.05 vs 9.22 fitur mentah | Proyeksi kemungkinan mencampur > 24 indikator → tidak bisa dibalik |
| 43 GP-ARD per basis | Target hampir deterministik → GP dengan skala per arah belajar fungsinya | raw 8.87 / whitened PC 8.79 / PC 8.82 (TabICL 7.53); GP sendiri menaksir noise kecil (0.03–0.12 varians std. log-y) | Fungsinya deterministik tapi terlalu kompleks untuk 300 sampel |
| 44 mode target | Bimodal = rezim biner tersembunyi | Mode rendah 38.2±3.1 (31%), tinggi 57.1±11.2; AUC mode dari fitur 0.94; two-stage 7.93 vs 7.53 | Ditolak |

**Penilaian setelah 8 round (~45 eksperimen):** tidak ada struktur sah di fitur yang mendekati RMSE 2.x. Dua penjelasan skor 2.28–2.72 yang konsisten dengan semua bukti: (a) probing public LB (29 baris; dengan 5 submit/hari target tiap baris public bisa direkonstruksi satu per satu dari selisih RMSE), atau (b) rekonstruksi target dari dataset publik. Keduanya dilarang (§11.2, §3) dan (a) tidak membantu private.

---

# v8: Round 9 — pseudo-label untuk anggota SVR

Script di `eda/v9/`.

| Script | Hipotesis | Hasil | Keputusan |
|---|---|---|---|
| 45 SVR-PL | SVR (anggota yang menggerakkan blend, kernel lokal) juga diuntungkan pseudo-label seperti FM; label first-pass = OOF FM di fold yang sama (tanpa bocor) | SVR sendiri 8.09 → 7.92; korelasi residual vs FM hampir sama (0.851→0.859); blend −0.018 pooled / −0.022 per-repeat; label dari blend v6b atau bobot 0.5 sedikit lebih kecil | Dipakai |
| 46 split | Berapa pseudo row untuk SVR? | 2/4/8-split: −0.027/−0.018/−0.022 pooled (tidak monoton = noise partisi) | Rata-rata 2/4/8 |
| 46 rata-rata split | Rata-rata menghapus noise partisi | pooled **7.2325 (−0.023)**, per-repeat 7.3877 (−0.029), lebih baik di 3/3 repeat; bootstrap P(better than v6b) **0.971**; P(win) public-29 0.62, private-67 0.69 | **v8** |

---

# Koreksi & temuan kebocoran urutan baris (4 Oktober 2026)

- **Koreksi:** audit `record_id` di round 6 (script 29) hanya menguji urutan **angka ID**, bukan **posisi baris di file CSV**. Klaim round 8 bahwa skor 2.x hanya bisa dijelaskan oleh probing atau data eksternal **ditarik**.
- **Temuan (audit `docs/score_gap_audit.md`, diverifikasi ulang):** Spearman(posisi baris train, target) = **−0.981** (baris pertama target 100.4, terakhir ~33). Spearman(posisi baris test, prediksi v8 dari fitur saja) = **−0.806** → test.csv kemungkinan besar juga terurut menurut target. Model posisi-saja (isotonic) mencapai RMSE diagnostik ~2.95–3.97. Ini penjelasan konkret skor 2.x.
- **Status aturan:** `data/competition_rules.md` §10.9 — diskualifikasi untuk "tindakan lain yang secara substansial menghilangkan kebutuhan untuk menyelesaikan tugas prediksi sebagaimana dirancang". Memprediksi target dari urutan baris test (artefak pengurutan panitia, bukan fitur) termasuk kategori ini menurut pembacaan kami; Top-3 wajib verifikasi (§8). **Tidak dipakai** di submission; disarankan melapor ke panitia.
- **Catatan waktu:** §9 menyebut periode submission berakhir **4 Oktober 2026 23:59 WIB**.

## v9: pemakaian urutan baris (setelah konfirmasi panitia)

Panitia mengonfirmasi bahwa memakai urutan baris file adalah bagian dari EDA/preprocessing peserta (aturan lokal di `data/competition_rules.md` berasal dari guidebook lama). Notebook `notebooks/v9.ipynb` = v8 + Section 8 "Row-Order Information":
- EDA urutan: Spearman posisi–target (train), posisi–prediksi fitur (test, harus < −0.5 atau notebook berhenti), dan kesamaan drift fitur sepanjang file train vs test.
- Simulasi split pemertahan urutan (300 × 80/20, posisi dinormalisasi per bagian) membandingkan: fitur saja, quantile map (posisi relatif → target train), proyeksi isotonic prediksi fitur, penataan ulang prediksi fitur, dan kombinasi NNLS yang di-cross-fit antar separuh split.
- Prediksi akhir = kombinasi dengan bobot dari semua split, diterapkan ke urutan test.csv.
Tidak dijalankan di lokal (instruksi); angka referensi dari audit: quantile map median ~3.97 di simulasi split independen.
