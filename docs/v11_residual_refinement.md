# v11: EDA sesudah v10 public 2.1068098327

Skor public v10 berasal dari laporan pengguna. Angka itu tidak digunakan untuk mengoptimalkan bobot, memilih model, atau mengubah target/prediksi per baris. Hasil public v11 belum diketahui dan tidak ada submission dikirim oleh assistant.

## Diagnosis

Residual v10 masih besar pada 10% baris awal: RMSE 7.126, bias target minus prediksi +1.442, menyumbang 44.91% SSE. Penyimpangan target dari urutan tidak seluruhnya bisa diselesaikan dengan kurva monoton.

Tiga keluarga pendekatan diuji:

1. Stabilitas rank: dimensi PCA 8/12/16/20, rata-rata dimensi, subsampling 90% training sebanyak 16 kali dengan seed tetap.
2. Prior ketidakpastian rank: koreksi Gaussian dengan kovarians Brownian bridge.
3. Residual fitur: hilangkan tren posisi dari fitur dan target, fit Ridge pada residual training, dan prediksi penyimpangan target query relatif terhadap tren rank-nya.

Tidak ada foundation model yang dilatih ulang. Seluruh fit ringan menggunakan training partition saja. Data test hanya memberi fitur dan urutan yang sudah tersedia.

## Hasil dan pemilihan

Untuk outer fold yang sama dengan audit v10, inner 5-fold memilih satu dari 9 kandidat: dimensi 8, dimensi 12, rata-rata dimensi, Brownian bridge strength 0.5, subbagging 16, serta residual Ridge alpha 10/100 dengan kekuatan 0.5/1.0. Hasil kandidat terpilih dicampur dengan v10 dengan bobot tetap 50:50. Tidak ada bobot yang dipelajari dari outer-validation labels.

| Metode | Mean fold RMSE | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|---:|
| v10 | 3.4008 | 3.2766 | 3.5956 | 3.4433 |
| Estimator baru, seleksi inner | 3.3093 | 3.2043 | 3.5314 | 3.3158 |
| **v11: 50% v10 + 50% estimator baru** | **3.3031** | **3.1877** | **3.5095** | **3.3243** |

v11 membaik di 12/15 fold dan 3/3 repeat. Bootstrap berkelompok per record, menjaga ketiga repeat bersama, memberi interval persentil 95% delta RMSE [-0.18235, -0.02269]. Frekuensi bootstrap membaik 0.996, bukan probabilitas Bayesian bahwa model pasti lebih baik.

Simulasi 10.000 subset berukuran 29, masing-masing dari satu repeat: frekuensi menang terhadap v10 71.68%. Persentil 5/50/95 delta RMSE [-0.39717, -0.07842, +0.15891]. Ini simulasi kondisional pada residual CV, bukan peluang terukur mencapai skor tertentu di Kaggle.

Pada 200 split baru dengan parameter tetap, rank PCA12 tanpa residual rata-rata 3.6430. Residual Ridge10 strength0.5: 3.5809; strength1.0: 3.5906. Strength1.0 mengalahkan rank tanpa residual pada 64.5% split. Ini stress test pada dataset sama, bukan independent test. Variasi kekuatan residual menunjukkan pemilihan parameter masih tidak pasti.

Seleksi full-training menggunakan seed 5299 memilih **PCA12, 10 tetangga, koreksi rank strength1, smoothing0.005, residual Ridge alpha10 strength1**. Model ini dipilih dari training CV, bukan skor public. Variasi rank/Brownian/subbagging menjadi alternatif dalam seleksi inner, tetapi bukan final model terpilih.

## Letak perbaikan

| Posisi query | RMSE v10 | RMSE v11 |
|---|---:|---:|
| 0–10% | 7.1260 | 6.8206 |
| 10–25% | 4.0724 | 4.0676 |
| 25–50% | 3.0409 | 2.9266 |
| 50–75% | 2.3743 | 2.3625 |
| 75–90% | 1.1344 | 1.0202 |
| 90–100% | 1.3812 | 1.4157 |

Sebagian besar gain berasal dari ekor tinggi dan bagian tengah. Bagian paling akhir sedikit memburuk. Prediksi final boleh tidak monoton karena residual fitur; ini disengaja, sesuai target train yang memiliki penyimpangan lokal.

RMS perubahan test terhadap v10: 0.6744. Perubahan terbesar pada D0112: -2.0823; D0280: +1.2541. Angka ini adalah prediksi, bukan diagnosis kesalahan test karena label test tidak diketahui. Tidak ada penyesuaian manual per ID.

## Batas klaim

- Kandidat dirancang setelah mengeksplorasi dataset yang sama. Nested CV mengisolasi pemilihan kandidat dalam fold, tetapi tidak menghapus research-selection bias.
- OOF baseline berasal dari artefak lokal audit v10. Input final adalah file submission v10 yang sebenarnya dibuat sebelumnya. Skor Kaggle 2.1068098327 dilaporkan pengguna, tidak diukur melalui label test lokal.
- Pemetaan urutan tetap mengasumsikan proses train/test yang mempertahankan urutan relatif. Konfirmasi penggunaan urutan oleh panitia dilaporkan tim.
- Jangan mengurangi skor public 2.10681 dengan gain CV 0.0977 untuk mengklaim skor baru. Public v11 dapat memburuk.
- Artefak v10 tetap diperlukan. Untuk reproduksi end-to-end dari data mentah, simpan pula pipeline dan checkpoint pembentuk v9, lalu v10.

## Reproduksi

Notebook `notebooks/v11.ipynb` mandiri untuk Kaggle CPU. Attach official train/test/template dan `submission_v10.csv`. `outputs/v11_eda/v10_oof.csv` bersifat opsional untuk menampilkan perbandingan validasi berpasangan. Tanpanya, notebook tetap melakukan nested validation estimator baru dan menulis submission final.

Seluruh cell Python notebook dieksekusi secara berurutan melalui harness lokal. Hasilnya cocok dengan script audit pada toleransi absolut 1e-10. Final selection dan prediction diulang dua kali dalam lingkungan yang sama; CSV identik byte demi byte. Ini belum merupakan uji lintas hardware atau lintas versi library.

Output di `outputs/fikfair_v11/`:

- `submission_v11.csv`: kandidat utama, 96 ID unik sesuai test dan nilai finite.
- `manifest.json`: hash train/test/template/v10/submission, seed, konfigurasi, versi Python/library, platform.
- `source_hashes.json`: hash notebook dan script pembentuknya.
- `requirements-v11.txt`: pin NumPy, Pandas, SciPy, scikit-learn untuk perhitungan. Plot membutuhkan Matplotlib yang tersedia pada environment pengujian/Kaggle.
- `prediction_changes.csv` dan `v11_predictions.png`: perubahan dibanding v10.
- `validation.csv`, `fold_selection.csv`, `validation_predictions.npz`: bukti validasi.
- `v11_executed.ipynb`: notebook dengan output harness lokal tersimpan.

Environment pengujian: Python 3.12.14, NumPy 2.5.3, Pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1. PCA baru memakai full SVD dan subsampling memakai seed tetap 5100. Outer CV seed42, inner seed5200+fold, final selection seed5299. Jalankan dengan OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1 untuk mendekati lingkungan uji.

Urutan script: `51_v11_rank_eda.py` untuk keluarga rank, `53_v11_residual_check.py` untuk residual, `52_v11_nested.py` untuk nested validation gabungan, `notebooks/build_v11.py`, lalu `54_export_v11.py` untuk eksekusi dan pemeriksaan hasil. Semua script EDA di `eda/v9/`. Ini CPU ringan, tanpa training ulang foundation model.

## Referensi

Kovarians min(s,t)-s*t dari Brownian bridge sebagai limit empirical process dibahas dalam [catatan Statistics 210B Berkeley](https://www.stat.berkeley.edu/~aditya/resources/Lecture-ONE-210b-2014.pdf). Penggunaannya sebagai prior koreksi rank di sini adalah pendekatan pemodelan yang diuji, bukan jaminan teoretis performa kompetisi.

Rasional pengurangan varians melalui agregasi dijelaskan dalam [dokumentasi ensemble scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html). Eksperimen ini memakai subsampling tanpa pengembalian, bukan bootstrap klasik. Kandidat residual akhirnya terpilih melalui CV; referensi umum tersebut tidak menyediakan angka benchmark untuk kompetisi ini.
