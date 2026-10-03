# COMPETITION RULES — FIK FAIR DATA SCIENCE COMPETITION

## 1. Evaluation Metric

### 1.1 Primary metric
Metric utama adalah **RMSE (Root Mean Squared Error)**.

- Nilai RMSE yang lebih kecil memiliki peringkat lebih tinggi.
- RMSE dihitung terhadap ground truth test yang disimpan oleh panitia.
- Perhitungan menggunakan seluruh nilai prediksi yang valid pada subset leaderboard yang relevan.

### 1.2 Tie-breaker
Jika dua submission memiliki RMSE yang sama hingga **12 angka desimal**, maka dibandingkan dengan:

1. **MAE (Mean Absolute Error)** yang lebih kecil;
2. jika MAE juga sama hingga **12 angka desimal**, submission valid yang memiliki **timestamp server lebih awal**
   ditempatkan lebih tinggi.

Tidak ada skor manual, voting juri, atau bonus lain yang digunakan untuk menentukan ranking model.

---

## 2. Public dan Private Leaderboard

Test set terdiri dari **96 record** dan dibagi menjadi:

- **Public leaderboard:** 29 record
- **Private leaderboard:** 67 record

Pembagian record ditetapkan oleh panitia sebelum kompetisi dimulai dan tidak diberitahukan kepada peserta.

Public leaderboard digunakan untuk feedback selama kompetisi. **Ranking akhir ditentukan menggunakan
Private leaderboard**, bukan score Public leaderboard.

Score yang ditampilkan pada leaderboard dapat dibulatkan untuk tampilan, tetapi nilai internal evaluator
yang digunakan untuk ranking mengikuti ketentuan metric pada bagian 1.

---

## 3. Submission

### 3.1 Format
Submission wajib mengikuti schema berikut:

```text
record_id,prediction
```

- harus memiliki tepat 2 kolom;
- harus berisi tepat 96 baris data;
- setiap `record_id` test harus muncul tepat satu kali;
- tidak boleh ada `record_id` tambahan;
- `prediction` harus numerik dan finite;
- nama kolom harus persis sama dengan template.

Evaluator mencocokkan prediction berdasarkan `record_id`, sehingga urutan baris tidak menjadi bagian dari
penilaian.

### 3.2 Submission valid dan invalid

**Submission valid** adalah submission yang lolos seluruh pemeriksaan schema dan dapat dievaluasi oleh
sistem kompetisi.

**Submission invalid** adalah submission yang gagal pemeriksaan schema atau tidak dapat dievaluasi secara
normal. Submission invalid:
- tidak memperoleh score;
- tidak masuk leaderboard;
- tidak dianggap sebagai submission valid untuk batas harian.

Submission yang telah valid dan kemudian terbukti melanggar aturan kompetisi dapat dibatalkan oleh panitia
dan dapat menjadi dasar tindakan disipliner/disqualification.

Panitia dapat menerapkan rate limiting teknis terhadap pengiriman invalid yang berulang atau bersifat abusif.

### 3.3 Batas submission

Maksimum **5 submission valid per tim per hari**.

Batas harian mengikuti waktu **WIB (UTC+7)** dan reset pada pukul **00:00 WIB** setiap hari.

Timestamp server menjadi acuan resmi untuk menentukan kapan submission diterima.

---

## 4. External Data Policy

Kompetisi ini merupakan kompetisi **pure prediction**. Data eksternal boleh digunakan hanya untuk
pengetahuan atau metode umum, bukan untuk mengambil kembali identitas atau target test.

### 4.1 Yang diperbolehkan

Peserta boleh menggunakan:
- paper dan dokumentasi ilmiah;
- dokumentasi library/framework;
- tutorial dan referensi machine learning;
- sumber umum untuk memahami feature engineering atau algoritma;
- dataset eksternal untuk eksperimen/metode umum **selama dataset tersebut tidak digunakan untuk
  mengidentifikasi record test, mencocokkan record test dengan entitas asli, memperoleh target test,
  atau membangun label/prediction test dari nilai target eksternal yang cocok dengan record tersebut**.

### 4.2 Yang dilarang

Peserta dilarang menggunakan data eksternal, search engine, database, API, website, atau sumber lain untuk:

1. mengidentifikasi kota, negara, koordinat, atau entitas asli dari suatu `test record`;
2. melakukan **record linkage**, matching, join, reverse lookup, atau pencocokan pola yang bertujuan
   menghubungkan `record_id`/fitur test dengan entitas asli;
3. mencari atau mengambil nilai target asli yang berhubungan dengan entitas hasil identifikasi;
4. merekonstruksi rumus atau nilai target test dari sumber publik untuk menghasilkan prediction langsung;
5. menggunakan tabel, database, cache, atau hasil pencarian yang pada dasarnya menyediakan pasangan
   **entitas test → target**;
6. menggunakan data atau artefak pihak ketiga yang berisi label/target test atau informasi yang secara
   langsung mengungkap label tersebut.

Contoh shortcut yang **dilarang**:

```text
test record
→ identifikasi kota/entitas asli
→ cari nilai publik untuk entitas tersebut
→ gunakan nilai tersebut sebagai prediction
```

Hal yang dilarang tetap dilarang meskipun proses dilakukan secara manual, dengan script, API,
search engine, notebook, agent, atau AI/LLM.

### 4.3 Data eksternal untuk model

Pretrained model atau model yang dilatih menggunakan data eksternal **boleh digunakan** selama model/data
tersebut tidak digunakan sebagai sarana untuk mengambil kembali identitas atau target test secara spesifik.
Peserta tetap bertanggung jawab memastikan sumber data/model yang digunakan tidak memberikan akses langsung
ke target test.

---

## 5. AI / LLM Policy

AI/LLM **diperbolehkan** sebagai alat bantu pengembangan solusi, termasuk:
- coding;
- debugging;
- refactoring;
- feature engineering;
- pemilihan atau penjelasan metode;
- dokumentasi;
- membantu memahami error;
- membantu analisis train data dan eksperimen modeling.

Namun, AI/LLM tidak menjadi pengecualian terhadap External Data Policy.

Peserta **dilarang** menggunakan AI/LLM untuk:
- mengidentifikasi entitas asli dari test record;
- mencari target test dari pengetahuan atau sumber eksternal;
- melakukan lookup atau reverse lookup terhadap target;
- menghasilkan prediction test berdasarkan informasi target eksternal yang diperoleh secara spesifik.

Penggunaan AI/LLM untuk menulis kode atau membantu proses modeling tetap diperbolehkan selama proses
akhir mematuhi seluruh aturan kompetisi.

---

## 6. Pretrained Model dan Training Data

Pretrained model diperbolehkan dengan ketentuan:
- model tidak mengandung atau menyediakan ground truth test competition;
- model tidak digunakan untuk melakukan lookup target test secara spesifik;
- penggunaan model tidak melanggar External Data Policy;
- peserta dapat menjelaskan model dan sumber dependency yang digunakan jika diminta pada tahap verifikasi.

Peserta dilarang menggunakan dataset pihak ketiga yang diketahui mengandung target competition,
label test competition, atau pasangan identitas yang secara langsung memungkinkan target test diperoleh.

---

## 7. Team Integrity dan Collaboration

- Satu tim terdiri dari **1–3 peserta**.
- Satu peserta hanya boleh terdaftar pada **1 tim**.
- Keanggotaan lintas fakultas diperbolehkan selama memenuhi ketentuan pendaftaran resmi.

Selama periode kompetisi, peserta **tidak diperbolehkan berbagi solusi spesifik kompetisi** dengan tim lain,
termasuk:
- source code solusi;
- notebook solusi;
- prediction file;
- private/derived labels;
- hasil lookup target;
- model atau artifact yang secara khusus dilatih untuk kompetisi;
- strategi yang secara langsung memberikan akses terhadap target test.

Berbagi pengetahuan umum, paper, dokumentasi, tutorial, atau library yang tersedia secara publik tetap
diperbolehkan.

Anggota dalam tim yang sama boleh berbagi seluruh artifact solusi timnya.

---

## 8. Reproducibility dan Finalist Verification

Untuk **Top 3 teams**, pengumpulan bahan verifikasi bersifat **wajib** apabila diminta panitia.

Minimal bahan verifikasi:
- source code/notebook solusi;
- daftar dependency/environment, misalnya `requirements.txt`;
- penjelasan singkat pipeline;
- file konfigurasi atau artifact tambahan yang diperlukan untuk menjalankan solusi;
- informasi pretrained model atau dataset eksternal yang digunakan, jika ada.

Panitia dapat menggunakan bahan tersebut hanya untuk memverifikasi kepatuhan terhadap aturan kompetisi
dan tidak untuk mengubah hasil ranking secara manual.

Jika tim tidak dapat memberikan bahan verifikasi yang wajar atau terbukti menggunakan metode yang dilarang,
panitia dapat membatalkan submission terkait dan/atau menerapkan disqualification sesuai bagian 10.

---

## 9. Competition Timeline

Periode submission:

**22 September 2026 — 00:00 WIB** sampai **4 Oktober 2026 — 23:59 WIB**.

Submission yang diterima setelah batas akhir tidak dinilai.

Timestamp server kompetisi menjadi acuan resmi, bukan jam pada perangkat peserta.

---

## 10. Disqualification dan Pelanggaran

Tim dapat didiskualifikasi atau submission dapat dibatalkan apabila terbukti melakukan salah satu hal berikut:

1. memperoleh atau merekonstruksi target test melalui jalur yang dilarang;
2. mengidentifikasi entitas asli test untuk tujuan memperoleh target atau meningkatkan prediction secara langsung;
3. mengeksploitasi bug, celah evaluator, leaderboard, atau sistem submission untuk memperoleh informasi
   ground truth yang tidak seharusnya tersedia;
4. memanipulasi submission, score, evaluator, atau sistem leaderboard;
5. membagikan atau memperoleh private ground truth, answer key, atau artifact rahasia panitia;
6. menggunakan dataset/model pihak ketiga yang diketahui menyediakan target test secara langsung;
7. melakukan kolusi atau pertukaran solution-specific artifact antar tim selama kompetisi;
8. memberikan informasi palsu pada proses verifikasi;
9. melakukan tindakan lain yang secara substansial menghilangkan kebutuhan untuk menyelesaikan tugas
   prediksi sebagaimana dirancang dalam kompetisi.

Panitia dapat meminta klarifikasi atau bukti pendukung sebelum menetapkan tindakan berdasarkan dugaan
pelanggaran.

---

## 11. Organizer Data Security

File berikut adalah **organizer-only** dan tidak boleh diberikan kepada peserta:
- ground truth test / answer key;
- data master yang memuat identitas entitas asli;
- parameter atau artifact internal yang dapat memetakan fitur anonim ke sumber;
- evaluator private atau konfigurasi yang dapat mengungkap ground truth.

File private tidak boleh ditempatkan pada repository publik, paket dataset peserta, frontend publik,
atau lokasi lain yang dapat diakses peserta.

---

## 12. Final Ranking

Ranking akhir ditentukan dari **Private leaderboard** menggunakan aturan:

1. RMSE terkecil;
2. jika RMSE sama hingga 12 angka desimal, MAE terkecil;
3. jika RMSE dan MAE sama hingga 12 angka desimal, timestamp server submission valid yang lebih awal.

Tidak ada penilaian subjektif atau bonus manual yang ditambahkan ke ranking akhir.

---

## 13. Competition Files

Paket peserta berisi:

- `train.csv`
- `test.csv`
- `submission_template.csv`
- `data_dictionary.csv`
- `submission_example.csv`
- `baseline/baseline_mean.csv`
- `competition_rules.md`

Ground truth test, master data, dan artifact private panitia berada di luar paket peserta.
