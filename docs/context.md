# Data Science Competition FIK FAIR 2026

Build a machine learning model to predict an anonymous competition target from 24 transformed numerical features.

## Overview

Data Science Competition FIK FAIR 2026 merupakan kompetisi machine learning yang menantang peserta untuk membangun model prediksi berdasarkan data numerik anonim.

Peserta akan diberikan training dataset yang terdiri atas 24 fitur numerik dan satu target. Peserta diminta memanfaatkan data tersebut untuk membangun model machine learning dan menghasilkan prediksi terhadap data test.

Submission prediksi dilakukan melalui Kaggle dan akan digunakan untuk menghasilkan skor serta leaderboard kompetisi.

### Description

# Problem Statement

Peserta diminta membangun model machine learning untuk memprediksi Anonymous Competition Score, yaitu target numerik yang disediakan khusus untuk kebutuhan kompetisi.

Identitas dan makna asli dari target tidak diberikan kepada peserta sebagai bagian dari desain kompetisi.

# Dataset

Dataset terdiri atas:

- train.csv — data training yang berisi 24 fitur dan target.
- test.csv — data test yang berisi 24 fitur tanpa target.
- submission_template.csv — template format submission.
- submission_example.csv — contoh format submission.
- data_dictionary.csv — informasi mengenai kolom yang tersedia.

# Features

Dataset peserta menyediakan 24 numerical transformed features:

`feature_01` hingga `feature_24`

Seluruh fitur bersifat:

- numerical
- dimensionless
- transformed
- tidak memiliki unit asli
- tidak mengandung missing value

Identitas serta pemetaan fitur terhadap source field asli tidak diberikan kepada peserta.

# Target

Target kompetisi adalah Anonymous Competition Score. Target:

- bersifat numerik;
- tidak menggunakan nama atau label sumber asli;
- merupakan nilai hasil konstruksi khusus untuk kompetisi;
- memiliki rentang nilai pada dataset training sekitar 32.59–100.44.

Formula dan proses pembentukan target tidak diberikan kepada peserta.

# Feature Transformation

Fitur peserta telah melalui beberapa tahap transformasi dan proyeksi dari sekumpulan indikator lingkungan/iklim.

Secara konseptual, proses meliputi:

- normalization/rank transformation;
- bounded transformation;
- private linear projection;
- standardization;
- controlled perturbation;
- rounding.

Nilai akhir fitur tidak merepresentasikan nilai asli dari source dataset dan tidak memiliki unit asli.

Detail parameter transformasi dan pemetaan terhadap source field tidak dipublikasikan.

### Evaluation

Submission peserta akan dievaluasi menggunakan:

**Primary Metric: RMSE (Root Mean Squared Error)**

Semakin kecil nilai RMSE, semakin baik performa model.

Leaderboard kompetisi terdiri atas:

- **Public Leaderboard:** digunakan untuk memberikan feedback selama kompetisi.
- **Private Leaderboard:** digunakan sebagai dasar peringkat akhir.

Peringkat akhir ditentukan berdasarkan Private Leaderboard.

### Submission

# Submission Format

Peserta wajib mengunggah file prediksi melalui sistem submission Kaggle.

File submission harus memiliki dua kolom:

`record_id,prediction`

Setiap `record_id` pada `test.csv` harus memiliki tepat satu nilai prediction.

Peserta disarankan menggunakan `submission_template.csv` sebagai dasar pembuatan file submission.

Contoh:

`record_id,prediction`

`D0383,61.293`

`D0384,54.827`

`D0385,72.114`

# Submission Limit

Setiap tim dapat melakukan maksimal 5 submission per hari selama periode kompetisi.

# External Data

Penggunaan external data tidak diperbolehkan untuk:

- mengidentifikasi entitas asli yang direpresentasikan oleh test record;
- memperoleh target test dari sumber eksternal;
- merekonstruksi target test secara langsung dari sumber publik;
- melakukan pencarian terhadap source entity berdasarkan karakteristik data test.

Paper, tutorial, dokumentasi publik, library machine learning, dan sumber pembelajaran dapat digunakan selama tidak digunakan untuk memperoleh atau merekonstruksi target test melalui external data.

# AI / LLM

AI atau Large Language Model (LLM) diperbolehkan sebagai alat bantu dalam pengembangan solusi, termasuk:

- coding;
- debugging;
- feature engineering;
- pencarian metode;
- dokumentasi.

Penggunaan AI/LLM tetap harus mengikuti seluruh aturan kompetisi, termasuk aturan mengenai external data dan data leakage.

# Notebook and Report

Selain submission Kaggle, peserta wajib mengumpulkan:

1. **Notebook (.ipynb)** yang berisi proses analisis, eksplorasi data, preprocessing, feature engineering, pemodelan, dan pembuatan prediksi.
2. **Laporan (.pdf)** yang menjelaskan proses dan pendekatan yang digunakan dalam notebook.

Notebook dan laporan dikumpulkan melalui Google Form yang disediakan oleh panitia.

### Citation

FIK FAIR 2026. Data Science Competition FIK FAIR 2026. [https://www.kaggle.com/competitions/data-science-competition-fikfair2026](https://www.kaggle.com/competitions/data-science-competition-fikfair2026), 2026. Kaggle.

## Dataset Description

# Data Science Competition 2026

Dataset ini digunakan dalam **Data Science Competition (DSC) FIK FAIR 2026** yang diselenggarakan oleh Fakultas Ilmu Komputer UPN “Veteran” Jakarta.

Dataset terdiri dari **478 observasi** dengan **24 fitur numerik anonim** yang telah melalui proses transformasi.

## Files

- **`train.csv`** - training set yang berisi fitur dan target.
- **`test.csv`** - test set yang digunakan untuk membuat prediksi.
- **`sample_submission.csv`** - contoh file submission dengan format yang benar.
- **`data_dictionary.csv`** - informasi mengenai fitur dan target yang tersedia dalam dataset.

## Columns

- **`record_id`** - ID unik dan anonim untuk setiap observasi. ID ini tidak memiliki makna geografis atau informasi tersembunyi.
- **`feature_01`** hingga **`feature_24`** - fitur numerik anonim yang telah ditransformasikan dan digunakan sebagai input dalam pemodelan.
- **`target`** - target pada training set yang digunakan sebagai dasar pembelajaran model.

## Data Split

- **Training set:** 382 observasi
- **Test set:** 96 observasi

Peserta menggunakan `train.csv` untuk melakukan eksplorasi data, preprocessing, feature engineering, dan pemodelan. Model kemudian digunakan untuk memprediksi target pada `test.csv`.

## Submission

File submission harus memiliki format:

- `record_id`
- `prediction`

Setiap `record_id` pada `test.csv` harus memiliki tepat satu prediksi.

Ground truth pada test set disimpan secara privat dan digunakan untuk melakukan evaluasi serta menentukan leaderboard.

## Competition Rules

## 1. Eligibility and Team

1. Setiap peserta wajib terdaftar sebagai peserta resmi Data Science Competition FIK FAIR 2026.
2. Peserta mengikuti ketentuan jumlah anggota tim sebagaimana ditetapkan dalam Guidebook resmi.
3. Setiap anggota hanya diperbolehkan tergabung dalam tim sesuai ketentuan panitia.
4. Penggunaan akun atau identitas peserta lain untuk mengikuti kompetisi tidak diperbolehkan.

## 2. Dataset

1. Peserta wajib menggunakan dataset resmi yang disediakan melalui halaman kompetisi.
2. Peserta tidak diperbolehkan memperoleh target test dari sumber lain.
3. Peserta tidak diperbolehkan mengubah, mengganti, atau memanipulasi `record_id` pada submission.
4. Peserta tidak diperbolehkan mencoba mengidentifikasi entitas asli yang direpresentasikan oleh test record untuk memperoleh target.

## 3. External Data

External data tidak boleh digunakan untuk:

- mengidentifikasi source entity dari test record;
- memperoleh target test secara langsung;
- merekonstruksi target test dari database atau sumber publik;
- melakukan pencocokan test record dengan data eksternal untuk memperoleh label/target.

Penggunaan:

- paper;
- tutorial;
- dokumentasi;
- library machine learning;
- referensi metode;

diperbolehkan selama tidak digunakan untuk memperoleh atau merekonstruksi target test.

## 4. AI / LLM

AI dan LLM diperbolehkan sebagai alat bantu pengembangan solusi, termasuk untuk:

- coding;
- debugging;
- feature engineering;
- pencarian metode;
- dokumentasi.

Penggunaan AI/LLM tidak membebaskan peserta dari aturan external data, data leakage, dan reproducibility.

## 5. Submission

1. Submission dilakukan melalui sistem submission Kaggle.
2. File submission harus berupa CSV.
3. File wajib memiliki kolom:
```autohotkey
record_id,prediction

```

1. Setiap `record_id` harus muncul tepat satu kali.
2. Seluruh `record_id` dari `test.csv` harus terdapat dalam submission.
3. Missing prediction tidak diperbolehkan.
4. Peserta wajib mengikuti format `submission_template.csv`.
5. Submission yang tidak memenuhi format dapat dinyatakan invalid.

## 6. Submission Limit

Setiap tim diperbolehkan melakukan maksimal **5 submission per hari**.

Batas submission berlaku untuk seluruh anggota dalam satu tim.

## 7. Evaluation and Ranking

Primary Metric:

**RMSE**

Semakin kecil nilai RMSE, semakin baik.

Tie-breaker:

**MAE**

Apabila terdapat kondisi skor yang sama sesuai ketentuan sistem penilaian, MAE digunakan sebagai tie-breaker.

## 8. Public and Private Leaderboard

Selama kompetisi berlangsung, peserta dapat melihat Public Leaderboard.

Private Leaderboard digunakan sebagai dasar penentuan peringkat akhir.

Pembagian evaluation set:

- Public: sekitar 30%
- Private: sekitar 70%

Private Leaderboard tidak ditampilkan selama periode kompetisi dan digunakan untuk menentukan peringkat akhir.

## 9. Notebook and Report

Setiap tim wajib mengumpulkan:

- Notebook dalam format `.ipynb`;
- Laporan dalam format `.pdf`.

Keduanya dikumpulkan melalui Google Form yang disediakan panitia.

Notebook wajib memuat proses analisis, eksplorasi data, preprocessing, feature engineering, pemodelan, dan pembuatan prediksi.

Laporan wajib menjelaskan proses dan pendekatan yang digunakan.

## 10. Reproducibility and Verification

Panitia berhak meminta tim dengan peringkat tertentu untuk menyerahkan source code, dependency/environment information, notebook, atau dokumentasi tambahan untuk keperluan verifikasi.

Peserta wajib memastikan solusi yang diserahkan dapat dijelaskan dan direproduksi sesuai ketentuan panitia.

## 11. Disqualification

Peserta dapat didiskualifikasi apabila terbukti:

1. memperoleh atau merekonstruksi target test menggunakan external data;
2. melakukan data leakage atau eksploitasi terhadap sistem evaluator;
3. memanipulasi leaderboard atau sistem submission;
4. membagikan private dataset, solution, atau informasi ground truth;
5. menggunakan identitas atau akun peserta lain secara tidak sah;
6. melakukan tindakan lain yang melanggar ketentuan resmi kompetisi.

## 12. Final Decision

Seluruh peserta wajib mengikuti Guidebook resmi Data Science Competition FIK FAIR 2026 serta pengumuman resmi panitia.

Dalam hal terdapat kondisi yang belum tercakup dalam Rules ini, panitia berhak memberikan keputusan teknis yang diperlukan untuk menjaga integritas kompetisi.

## Dataset License

This dataset is provided solely for participation in the Data Science Competition FIK FAIR 2026. Redistribution, commercial use, or use outside the competition is subject to the competition rules established by the organizer.
