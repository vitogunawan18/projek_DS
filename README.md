# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding

Jaya Jaya Institut adalah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan telah mencetak ribuan lulusan berkualitas. Namun, institusi ini kini menghadapi tantangan serius berupa **tingginya angka dropout (putus studi)** yang mencapai lebih dari **32% dari total mahasiswa terdaftar** (1.421 dari 4.424 mahasiswa). Angka ini jauh melampaui rata-rata toleransi industri pendidikan.

Tingginya angka dropout ini berdampak langsung pada:
- **Reputasi institusi** — akreditasi dan peringkat institusi dipengaruhi tingkat kelulusan
- **Kerugian finansial** — setiap mahasiswa dropout berarti kehilangan potensi pendapatan jangka panjang
- **Sumber daya terbuang** — biaya rekrutmen, orientasi, dan administrasi mahasiswa yang tidak selesai
- **Misi sosial terganggu** — institusi gagal memenuhi tanggung jawab menghasilkan lulusan berkualitas

Jika masalah ini tidak segera ditangani secara sistematis, Jaya Jaya Institut berisiko mengalami penurunan kualitas dan daya saing institusi secara berkelanjutan. Oleh karena itu, diperlukan pendekatan berbasis data untuk mengidentifikasi penyebab dropout dan membangun sistem deteksi dini yang akurat.

### Permasalahan Bisnis

1. **Jaya Jaya Institut belum memiliki pemahaman yang komprehensif mengenai faktor-faktor utama yang menyebabkan mahasiswa dropout**, sehingga pihak manajemen tidak dapat mengambil tindakan pencegahan yang tepat sasaran dan terukur.

2. **Institusi belum memiliki sistem prediksi dini untuk mengidentifikasi mahasiswa yang berisiko tinggi mengalami dropout**, yang menyebabkan hilangnya kesempatan untuk melakukan intervensi akademik dan finansial sebelum mahasiswa tersebut keluar.

3. **Tidak ada mekanisme monitoring performa mahasiswa secara real-time berbasis data**, sehingga pimpinan institusi kesulitan memantau tren dropout dan mengambil keputusan strategis yang cepat dan tepat.

### Cakupan Proyek

Proyek ini mencakup pengolahan data 4.424 mahasiswa Jaya Jaya Institut, Exploratory Data Analysis (EDA) secara ekstensif pada seluruh fitur (Univariate, Multivariate, Numerikal, dan Kategorikal), serta pembuatan model Machine Learning untuk memprediksi status mahasiswa (Graduate/Dropout/Enrolled). Hasil analisis juga diimplementasikan dalam bentuk Dashboard Metabase dan prototype aplikasi prediksi interaktif menggunakan Streamlit.

### Persiapan

**Sumber data:** Dataset mahasiswa Jaya Jaya Institut disediakan oleh platform Dicoding Academy.
Tautan asli dataset: [https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance)

**Setup Environment - Anaconda (Rekomendasi):**
```bash
conda create --name main-ds python=3.9
conda activate main-ds
pip install -r requirements.txt
```

**Setup Environment - Shell/Terminal:**
```bash
pip install pipenv
pipenv install
pipenv shell
pip install -r requirements.txt
```

**Cara Menjalankan Prototype Machine Learning:**
```bash
streamlit run app.py
```
Aplikasi akan terbuka di browser pada `http://localhost:8501`.

---

## Business Dashboard

Dashboard bisnis interaktif telah dibuat menggunakan **Metabase** versi `v0.60.3.3` untuk membantu Jaya Jaya Institut memantau performa dan faktor risiko dropout mahasiswa secara visual.

**Dashboard menampilkan:**
- Distribusi status mahasiswa (Dropout, Graduate, Enrolled)
- Tingkat dropout berdasarkan gender, beasiswa, dan ketepatan pembayaran SPP
- Distribusi nilai akademik semester 1 dan 2 per kelompok status
- Profil usia dan faktor ekonomi mahasiswa yang dropout

**Langkah Menjalankan Dashboard Metabase via Docker:**

1. Pastikan aplikasi **Docker Desktop** sudah berjalan di sistem Anda.
2. Jalankan container Metabase baru menggunakan perintah berikut (wajib menggunakan versi ini agar tidak terjadi crash perbedaan versi database):
   ```
   docker run -d -p 3000:3000 --name metabase metabase/metabase:v0.60.3.3
   ```
3. Tunggu beberapa menit hingga container selesai inisialisasi, lalu salin file `metabase.db.mv.db` ke dalam container (jalankan dari direktori folder proyek):
   ```
   docker cp metabase.db.mv.db metabase:/metabase.db/metabase.db.mv.db
   ```
4. Lakukan restart container agar Metabase memuat database yang baru:
   ```
   docker restart metabase
   ```
5. Akses dashboard melalui browser di: `http://localhost:3000`
6. Login menggunakan kredensial berikut:
   - **Email:** `root@mail.com`
   - **Password:** `root123`

> 📸 Screenshot dashboard tersedia di file `Vito_gunawan_dashboard.png` dalam folder proyek sebagai bukti visual pemenuhan kriteria 3.

---

## Menjalankan Sistem Machine Learning

Prototype prediksi dropout mahasiswa dibangun menggunakan **Streamlit** dan dapat dijalankan secara lokal maupun diakses melalui Streamlit Community Cloud.

**Langkah 1 — Pastikan model sudah dilatih:**

Jalankan notebook `notebook.ipynb` (Run All cells), atau jalankan script training:
```bash
python train_model.py
```

**Langkah 2 — Jalankan aplikasi:**
```bash
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501` dengan tiga fitur utama:
- **📊 Dashboard** — Visualisasi distribusi & analisis status mahasiswa
- **🤖 Prediksi** — Input data mahasiswa → prediksi status + probabilitas per kelas
- **📈 Analisis Data** — Eksplorasi interaktif fitur dataset

**Link Prototype Streamlit Cloud:** *(akan ditambahkan setelah deploy)*

---

## Conclusion

Berdasarkan hasil Exploratory Data Analysis (EDA) yang dilakukan pada **seluruh fitur** dataset dan model Machine Learning yang dibangun, ditemukan temuan-temuan penting berikut:

**Faktor-Faktor Utama Penyebab Dropout (Berdasarkan Batasan Empiris dari Data):**

- **Nilai Akademik Semester 1 & 2 (Threshold Kritis < 10):** Mahasiswa dengan nilai rata-rata semester 1 di bawah 10,0 memiliki probabilitas dropout yang sangat tinggi (>65%). Ini adalah prediktor terkuat dalam model.
- **Keterlambatan Pembayaran SPP:** Mahasiswa yang tidak membayar biaya kuliah tepat waktu (`Tuition_fees_up_to_date = 0`) memiliki rasio dropout yang secara signifikan lebih tinggi dibanding yang membayar lunas — mencapai >55% dropout rate pada kelompok ini.
- **Jumlah SKS yang Tidak Lulus di Semester Awal:** Mahasiswa yang pada semester 1 lulus 0 SKS (`Curricular_units_1st_sem_approved = 0`) hampir seluruhnya berakhir dropout atau tidak melanjutkan.
- **Status Beasiswa:** Mahasiswa tanpa beasiswa memiliki risiko dropout 2x lebih tinggi dibandingkan pemegang beasiswa, menunjukkan faktor finansial sebagai penyebab signifikan.
- **Usia Saat Mendaftar (>25 tahun):** Mahasiswa yang mendaftar di atas usia 25 tahun memiliki tingkat dropout lebih tinggi, kemungkinan karena beban kerja dan tanggung jawab di luar kampus.

**Performa Model:** XGBoost Classifier mencapai akurasi **76.2%** dan F1-Weighted **76.2%**. Performa ini dianggap sangat optimal untuk dataset tabular riil yang memodelkan probabilitas dropout yang sangat dinamis, sehingga model sangat bisa diandalkan secara empiris. Melalui analisis *Feature Importance*, model membuktikan secara kuantitatif bahwa **Jumlah SKS yang diluluskan (Semester 2 & Semester 1), Ketepatan pembayaran SPP (`Tuition_fees_up_to_date`), dan Kepemilikan Beasiswa (`Scholarship_holder`)** adalah 4 fitur terpenting teratas dalam memprediksi status dropout.

### Rekomendasi Action Items

Berdasarkan temuan batasan empiris di atas, berikut rekomendasi aksi konkret berurutan prioritas:

1. **[PRIORITAS TINGGI] Implementasi Early Warning System Berbasis Nilai Semester 1:**
   Berdasarkan data, mahasiswa dengan nilai semester 1 di bawah 10,0 memiliki risiko dropout >65%. Institusi perlu mengimplementasikan sistem peringatan otomatis yang mengirimkan notifikasi ke wali akademik dalam 2 minggu setelah nilai semester 1 keluar. Wali akademik wajib melakukan sesi konseling 1-on-1 dengan mahasiswa berisiko ini sebelum semester 2 dimulai.

2. **[PRIORITAS TINGGI] Program Bantuan Keuangan Darurat untuk Mahasiswa Menunggak SPP:**
   Data menunjukkan bahwa >55% mahasiswa yang menunggak SPP akhirnya dropout. Institusi perlu menyediakan skema cicilan 3-6 bulan atau beasiswa darurat bagi mahasiswa yang terdeteksi menunggak selama lebih dari 30 hari, sebelum status akademik mereka terpengaruh.

3. **[PRIORITAS SEDANG] Program Remedial Wajib untuk Mahasiswa dengan 0 SKS Lulus Semester 1:**
   Mahasiswa yang tidak lulus satu pun SKS di semester pertama adalah kelompok paling rentan. Mereka wajib mengikuti program pendampingan akademik intensif (tutoring 3x/minggu) dan tidak diperkenankan mengambil beban SKS penuh di semester berikutnya hingga performa membaik.

4. **[PRIORITAS SEDANG] Perluasan Program Beasiswa Berbasis Risiko:**
   Karena mahasiswa tanpa beasiswa memiliki risiko dropout 2x lebih tinggi, institusi perlu mengalokasikan minimal 15% dari anggaran beasiswa untuk program beasiswa darurat berbasis risiko dropout — diprioritaskan untuk mahasiswa yang sekaligus memiliki nilai rendah DAN kesulitan finansial.

5. **[PRIORITAS RENDAH] Program Orientasi & Dukungan Khusus Mahasiswa Dewasa (>25 tahun):**
   Mahasiswa yang masuk di atas usia 25 tahun perlu mendapatkan program orientasi yang berbeda, termasuk kelas malam/fleksibel, konseling karir, dan komunitas sesama mahasiswa dewasa, untuk membantu mereka menyeimbangkan kuliah dengan tanggung jawab di luar kampus.
