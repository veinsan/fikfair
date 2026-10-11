# v10: perbaikan setelah public v9 2.3865618144

Tanggal analisis: 5 Oktober 2026. Skor v9 dilaporkan pengguna; target pembanding 2.2400426385. Tidak ada target test yang tersedia. Tidak ada submission dikirim pada pekerjaan ini, dan skor public tidak digunakan untuk mengoptimalkan prediksi per baris atau memilih parameter.

## Temuan EDA

Rekonstruksi validasi v9 dari artefak lokal mereproduksi rata-rata RMSE fold **3.6132**, sesuai output notebook Kaggle **3.613**. Prediksi lokal foundation model dapat berbeda sedikit dari Kaggle, sehingga kesesuaian ini bukan klaim identik bit demi bit. Baseline final test memakai file v9 Kaggle yang sebenarnya.

| Posisi relatif dalam validation file | RMSE v9 | Bias target minus prediksi |
|---|---:|---:|
| 0–10% | 7.827 | +3.096 |
| 10–25% | 4.034 | +0.468 |
| 25–50% | 3.117 | +0.321 |
| 50–75% | 2.534 | −0.686 |
| 75–90% | 1.586 | −1.148 |
| 90–100% | 1.536 | −0.831 |

Sekitar 10% baris awal menyumbang **47.77% SSE**. Selain penyusutan prediksi ekor, ada kesalahan pemetaan persentil: persentil ke-20 pada subset query tidak selalu tepat bersesuaian dengan persentil ke-20 pada subset training. Korelasi displacement rank sebenarnya dengan residual pemetaan kuantil adalah **−0.537**. Displacement sebenarnya hanya dipakai untuk diagnosis train, tidak sebagai input prediksi.

Hasil eksplorasi 15 fold yang sama:

- Naikkan bobot kuantil dari 50% ke 75%: RMSE 3.5223.
- Pemulusan kuantil ringan, tetap dicampur 50% prediksi terurut: 3.5652–3.5732.
- PCHIP: 3.6172, tidak membantu.
- Integrasi kurva target terhadap distribusi Beta rank query: 3.5295 tanpa campuran fitur, tetapi campurannya 3.6239.
- Koreksi posisi rank menggunakan tetangga fitur: kandidat awal 3.3655, sebelum koreksi optimisme pemilihan.

## Metode yang dipertahankan

1. Fit standardisasi dan PCA-12 whitening pada training partition saja.
2. Cari tetangga fitur training untuk setiap query, gunakan rata-rata posisi tetangga dengan bobot inverse squared distance.
3. Kalibrasikan perkiraan rank memakai tren leave-one-out pada training. Ini mengurangi bias bahwa tetangga cenderung tertarik ke bagian tengah distribusi.
4. Haluskan koreksi sepanjang urutan query, kecilkan pengaruhnya di batas file, lalu pertahankan urutan monoton.
5. Petakan rank yang dikoreksi ke kurva isotonic target training, dengan opsi pemulusan ringan.
6. Rata-ratakan hasilnya dengan file v9 memakai bobot tetap 50:50.

Intinya, fitur dipakai untuk memperbaiki posisi relatif query terhadap training. Ini berbeda dari v9, yang hanya mengurutkan ulang distribusi prediksi model fitur.

Metode koreksi rank itu sendiri tidak memakai pseudo-label atau pretrained model. Kandidat final 50:50 masih mewarisi pipeline pseudo-label v9 melalui input file v9.

## Verifikasi

Outer validation memakai 5-fold × 3 repeat yang sama dengan OOF lama. Pada masing-masing outer training partition, inner 5-fold memilih di antara kuantil biasa, kurva Beta, dan 12 konfigurasi koreksi rank. Tidak ada fitur atau label outer-validation yang digunakan dalam seleksi inner. Bobot campuran v9/rank tetap 50:50.

| Metode | Mean fold RMSE | Repeat 1 | Repeat 2 | Repeat 3 |
|---|---:|---:|---:|---:|
| v9 | 3.6132 | 3.5298 | 3.7910 | 3.6691 |
| Kuantil biasa | 3.6479 | 3.6786 | 3.7756 | 3.6569 |
| Koreksi rank, seleksi inner | 3.4286 | 3.2627 | 3.6201 | 3.4936 |
| **v10: 50% v9 + 50% koreksi rank** | **3.4008** | **3.2766** | **3.5956** | **3.4433** |

Repeat RMSE dihitung pada seluruh 382 baris dalam repeat, bukan rata-rata RMSE kelima fold. Jangan mencampur kedua estimand tersebut.

v10 menang di **11/15 fold** dan **3/3 repeat**. Bootstrap berpasangan 5.000 kali dengan unit `record_id` (ketiga repeat tetap berkelompok) memberi interval persentil 95% perubahan RMSE **[−0.3561, −0.0915]**. Ini ketidakpastian kondisional pada prediksi tersimpan, tidak mencakup seluruh efek retraining, ketergantungan training antar-fold, atau pemilihan desain metode.

Stress test 200 split baru, seed 20261005, membandingkan konfigurasi tetap k=10, strength=1 tanpa pemulusan terhadap kuantil biasa: rata-rata RMSE **3.6372 vs 3.8849**, menang **71% split**. Ini pengulangan pada dataset sama, bukan independent test. OOF foundation dari fold lama tidak dipakai pada split baru.

Simulasi public dengan 10.000 subset: pilih satu repeat, lalu 29 record berbeda. Frekuensi v10 mengalahkan v9 **78.95%**, persentil 5/50/95 perubahan RMSE **−0.6910/−0.1759/+0.2004**. Ini tidak berarti peluang 79% mengalahkan 2.24004; residual test dan identitas subset public tidak diketahui. Jangan mengurangi skor public 2.38656 dengan gain CV untuk mengklaim skor baru.

Metode dan baseline v9 didesain setelah mengeksplorasi dataset ini, dan bobot v9 0.7/0.3 serta aturan v9 dipilih pada research terdahulu. Nested selection memperbaiki evaluasi pemilihan parameter baru, bukan menghapus seluruh research-selection bias.

## Kandidat dan reproduksi

Seleksi 5-fold pada semua training memilih **k=10, strength=1, smoothing=0.005**. Output utama:

- `outputs/fikfair_v10/submission_v10.csv`: kandidat utama, 50% v9 + 50% koreksi rank.
- `outputs/fikfair_v10/submission_rank_only.csv`: pembanding yang tidak membutuhkan pseudo-label; bukti nested terhadap v9 lebih lemah daripada campuran.
- `outputs/fikfair_v10/prediction_changes.csv`: perubahan per ID.
- `outputs/fikfair_v10/v10_predictions.png`: grafik prediksi dan perubahan.
- `outputs/fikfair_v10/manifest.json`: konfigurasi dan provenance.
- `outputs/fikfair_v10/v10_executed.ipynb`: bukti eksekusi cell Python secara berurutan melalui harness lokal, bukan Jupyter kernel.
- `notebooks/v10.ipynb`: notebook mandiri untuk Kaggle CPU.

RMS perubahan v10 terhadap v9 **0.7215**. File memiliki 96 ID unik, ID sesuai test, dan prediksi finite. Notebook ringan berhasil dieksekusi seluruh cell kode; validasi sekitar 2.9 detik pada run lokal satu thread. Angka ini bukan estimasi pasti runtime Kaggle. Prediksi validasinya identik dengan implementasi script audit terpisah.

Untuk Kaggle: attach data kompetisi dan CSV v9 yang benar. Attach `outputs/order_refinement/v9_oof.csv` bila ingin tabel perbandingan berpasangan dengan v9; tanpa file itu, notebook tetap memvalidasi rank vs kuantil dan membuat submission. GPU, internet, dan token tidak diperlukan. Notebook memeriksa lima prediksi awal v9 agar tidak salah memilih file versi lain.

Script reproduksi: `48_order_refinement.py` untuk eksplorasi, `49_order_nested_validation.py` untuk nested validation dan stress test, `notebooks/build_v10.py` untuk membangun notebook, `50_export_v10.py` untuk export dan eksekusi diagnostik. Seluruh script berada di repo ini; gunakan prefix `rtk proxy` dan `.venv/bin/python`. Tidak ada training ulang foundation model.

## Referensi metode

Pemulusan estimator kuantil melalui pembobotan order statistics merupakan teknik yang terdokumentasi; lihat [SciPy Harrell–Davis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.hdquantiles.html). Kandidat kurva Beta di sini merupakan pendekatan integrasi rank-query tersendiri, **bukan** implementasi persis Harrell–Davis. [SciPy PCHIP](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html) menyediakan interpolasi yang mempertahankan monotonisitas; kandidat itu diuji dan tidak dipilih. Tidak ada benchmark publik yang membuktikan konfigurasi ini unggul pada kompetisi ini; bukti pemilihan berasal dari validasi di atas.
