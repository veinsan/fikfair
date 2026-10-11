# Audit penyebab gap skor 2.x — 4 Oktober 2026

## Kesimpulan

Penjelasan baru yang paling kuat dari data lokal adalah **informasi target pada urutan baris CSV**. Audit sebelumnya menguji angka `record_id`, tetapi tidak posisi baris dalam file. Angka ID acak tidak berarti urutan baris acak. Temuan ini menjelaskan mekanisme yang bisa membawa skor ke kisaran 2.x tanpa membuktikan bahwa tim tertentu memakainya.

Leaderboard dibaca melalui Kaggle CLI pada audit ini: tiga skor teratas 2.2400426385, 2.2482931817, 2.2840260009; berikutnya 7.7110443504. Sumber: https://www.kaggle.com/competitions/data-science-competition-fikfair2026/leaderboard . Ini snapshot, bukan informasi metode peserta.

## Data dan bukti utama

Train 382 baris, test 96, 24 fitur numerik; tidak ada missing train atau duplikat vektor fitur train. Target 32.586413–100.439972. Sesuai file aturan lokal, public memakai 29 baris dan private 67.

| Pengukuran | Hasil | Arti |
|---|---:|---|
| Spearman angka ID vs target | +0.0485 | ID nyaris tidak informatif |
| Spearman posisi baris train vs target | **−0.9809** | Target hampir terurut menurun |
| Spearman posisi baris test vs prediksi v8 | −0.8063 | Model fitur juga melihat tren menurun sepanjang file test |
| Korelasi profil 24 korelasi fitur–posisi antara train dan test | 0.8727 | Urutan test mempunyai struktur fitur yang mirip urutan train |
| Isotonic dari posisi asli, 5-fold OOF | **RMSE 2.9533** | Sinyal posisi sangat kuat, tetapi global rank tetap membawa informasi held-out |
| Spline posisi asli, konfigurasi tetap | 3.4230 | Tidak semua pemulus otomatis menghasilkan 2.x |
| Rank train/valid dinormalisasi terpisah, 200 split 80/20 | median **3.9658**; persentil 5–95: 2.9026–5.5320 | Gap distribusi rank antarpartisi membuat transfer lebih sulit |

Hipotesis proses pembuatan: data diurutkan menurut target dasar atau indikator yang dekat dengan target, target akhir diberi perturbasi, lalu train/test dipisah sambil mempertahankan urutan relatif. Ini **inferensi**, bukan fakta tentang kode panitia. Urutan target yang tidak sepenuhnya monoton juga bisa timbul dari proses lain. Ground truth test tidak tersedia untuk membuktikan tren target test.

Pengujian posisi adalah **diagnostik kebocoran dataset**, bukan validasi model 24 fitur. Random KFold dengan posisi global dapat terlihat terlalu bagus karena posisi itu sudah berasal dari pengurutan seluruh dataset. Simulasi rank independen lebih mendekati pemisahan file, tetapi masih mengasumsikan proses split yang tidak diketahui.

## Apakah public kecil saja menjelaskan 2.x?

OOF v8 direkonstruksi dari anggota tersimpan, rata-rata tiga repeat, kemudian bobot blend dievaluasi dengan meta-KFold seed 42. RMSE 7.2067. Ini berbeda dari 7.2325 pada catatan lama karena angka lama merata-ratakan beberapa seed meta-KFold; angka audit bukan peningkatan model atau benchmark baru. Cross-validation pada matriks OOF tetap bukan nested CV penuh dari seluruh pemilihan model.

Pada 200.000 subset acak tanpa pengembalian, masing-masing 29 baris dari 382 residual tersebut:

- Median RMSE 6.9022; interval persentil 5–95: 4.6465–9.8121.
- 48 subset di bawah 3; **0 subset di bawah 2.3**.
- Hasil nol bukan probabilitas nol; simulasi hanya kondisional pada residual model sendiri, bukan residual model peserta lain.

Sebaliknya, dari simulasi rank independen di atas, 40.000 pengambilan subset public berukuran 29 menghasilkan **2.605% skor di bawah 2.3**. Subset berasal dari 200 split yang berulang sehingga bukan 40.000 eksperimen independen. Angka ini ilustrasi mekanisme, bukan estimasi peluang skor tim tertentu atau private leaderboard.

Artinya, variasi subset public saja kurang meyakinkan untuk menerangkan 7.x → 2.x pada pipeline fitur sekarang. **Informasi urutan + public kecil** merupakan penjelasan konkret yang cocok dengan bukti lokal.

## Apa yang membatasi hasil research sebelumnya?

1. Semua model utama belajar dari fitur anonim. Menambah model/ensemble tidak otomatis memulihkan informasi posisi yang tidak dimasukkan. Research sebelumnya menguji banyak prior pemodelan tetapi melewatkan metadata urutan file.
2. Sebanyak 26 target >70 menyumbang 40.86% SSE v8 pada audit ini. Namun setelah baris itu dikeluarkan, RMSE masih 5.7410. Masalah tidak hanya ekor target; perbaikan ekor saja belum menerangkan 2.x.
3. Learning curve tersimpan menunjukkan TabICL plain 10.01 pada n=120 menjadi 7.61 pada n=306. Keterbatasan sampel nyata, tetapi bukan bukti noise irreducible ~7.
4. Script `39_noise_floor_pairs.py` tidak memberikan lower bound model-free yang sah: fitur yang dekat tetapi berbeda tidak harus mempunyai prediksi sama tanpa asumsi kehalusan/Lipschitz. Untuk pasangan dengan x persis sama, dy²/4 adalah error minimum ketika keduanya diberi satu prediksi; pada pengulangan independen dengan noise identik, estimator varians noise memakai E[dy²]/2. Keduanya tidak boleh dipertukarkan. CI pasangan juga mengabaikan ketergantungan ketika satu baris muncul di beberapa pasangan.
5. Klaim pada catatan lama bahwa GP menaksir noise kecil berarti target deterministik terlalu kuat. Estimasi noise bergantung kernel, skala, dan fitting.
6. Bootstrap pada 3×382 residual yang memperlakukan repeat sebagai baris independen (misalnya script 32) dapat melebihkan kepastian. Repeat berasal dari observasi yang sama; resampling sebaiknya berkelompok per `record_id`. Seleksi puluhan eksperimen pada fold yang sama juga memberi selection optimism.

Kesimpulan lama bahwa tersisa hanya probing atau rekonstruksi eksternal perlu ditarik kembali. Keduanya bukan temuan audit, dan skor saja tidak cukup untuk menuduh peserta.

## Implikasi

Prioritas berikutnya adalah audit proses pengurutan dan split oleh panitia, serta reproduksi metode tim teratas. File aturan lokal §10.9 mencakup tindakan yang menghilangkan kebutuhan tugas prediksi; pemanfaatan urutan sebagai kanal target memerlukan penilaian panitia, bukan asumsi otomatis sah karena berasal dari CSV. Aturan bahwa evaluator mencocokkan ID hanya menjelaskan alignment submission, tidak membuktikan bahwa urutan file input bebas informasi.

Audit ini tidak membuat submission berbasis posisi, mengambil label test, atau menghubungi pihak lain. Model 24 fitur dan submission yang sudah ada tidak diubah.

## Reproduksi dan artefak

Jalankan dari root: `rtk proxy env MPLCONFIGDIR=/tmp/fikfair-mpl .venv/bin/python eda/47_score_gap_audit.py`.

Hasil: `outputs/score_gap_audit/metrics.json`, `residuals.csv`, `order_split_diagnostics.csv`, `feature_order_correlations.csv`, `submission_order_correlations.csv`, dan `score_gap_audit.png`. Skill dataset-profiling dan rigorous-evaluation dipakai untuk memisahkan profil data, angka diagnostik, dan klaim generalisasi.
