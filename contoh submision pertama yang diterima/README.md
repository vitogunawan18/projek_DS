# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

## Business Understanding
Jaya Jaya Institut adalah perusahaan Edutech yang berfokus pada pendidikan dan telah berdiri sejak tahun 2000. Saat ini, institusi sedang menghadapi masalah tingginya angka *turnover* (attrition) karyawan. Attrition yang tinggi ini menjadi masalah yang sangat penting karena berdampak pada tingginya beban operasional, hilangnya talenta berpengalaman, dan membengkaknya biaya rekrutmen serta pelatihan karyawan baru secara terus-menerus. Jika hal ini tidak segera diselesaikan, institusi berisiko mengalami penurunan produktivitas dan stabilitas tim dalam jangka panjang. Oleh karena itu, diperlukan identifikasi faktor-faktor pemicu karyawan resign serta prediksi dini untuk meminimalisir angka attrition.

### Permasalahan Bisnis
- Perusahaan belum memiliki pemahaman yang komprehensif mengenai faktor-faktor utama yang memicu karyawan untuk meninggalkan perusahaan (attrition), sehingga perusahaan tidak dapat mengambil tindakan pencegahan yang tepat.
- Perusahaan belum memiliki sistem prediksi dini untuk mendeteksi probabilitas karyawan mana yang berisiko tinggi untuk resign, yang menyebabkan hilangnya talenta tanpa sempat ditahan atau diberikan intervensi yang relevan.

### Cakupan Proyek
Proyek ini mencakup pengolahan data HR perusahaan, Exploratory Data Analysis (EDA) secara ekstensif pada seluruh fitur (Univariate, Multivariate, Numerikal, dan Kategorikal), serta pembuatan model Machine Learning berbasis Random Forest Classifier untuk memprediksi probabilitas attrition. Hasil prediksi juga diimplementasikan dalam bentuk *Dashboard* Metabase dan aplikasi prediksi interaktif menggunakan Streamlit.

### Persiapan
**Sumber data:** Disediakan oleh platform Dicoding Academy pada program studi terkait (*Dataset HR Jaya Jaya Institut - `employee_data.csv`*). Tautan asli dataset (Dicoding): https://github.com/dicodingacademy/dicoding_dataset/blob/main/employee/employee_data.csv

**Setup environment:**
Untuk menjalankan proyek ini, sangat disarankan membuat *virtual environment* baru agar seluruh dependensi dapat terisolasi dengan baik.

**Setup Environment - Menggunakan Anaconda (Rekomendasi)**
```bash
# Membuat virtual environment dengan nama 'main-ds' dan Python 3.9
conda create --name main-ds python=3.9

# Mengaktifkan environment
conda activate main-ds

# Menginstal seluruh dependensi dari file requirements.txt
pip install -r requirements.txt
```

**Setup Environment - Menggunakan Venv (Python Bawaan)**
```bash
# Membuat virtual environment dengan nama 'env'
python -m venv env

# Mengaktifkan environment (Windows)
env\Scripts\activate
# Mengaktifkan environment (Mac/Linux)
# source env/bin/activate

# Menginstal seluruh dependensi dari file requirements.txt
pip install -r requirements.txt
```

**Cara Menjalankan Skrip Python (Aplikasi Machine Learning)**
Proyek ini dilengkapi dengan antarmuka web interaktif berbasis Streamlit untuk memprediksi probabilitas Attrition seorang karyawan. Setelah seluruh dependensi berhasil terinstal, Anda dapat menjalankannya dengan mengetikkan perintah berikut di terminal:
```bash
streamlit run prediction.py
```
Aplikasi akan secara otomatis terbuka di *browser* Anda (biasanya pada alamat `http://localhost:8501`).

## Business Dashboard
Dashboard bisnis interaktif telah dibuat menggunakan **Metabase**. Untuk mempermudah validasi, file database Metabase (`metabase.db.mv.db`) telah disertakan di dalam folder proyek. Versi Metabase yang digunakan adalah **metabase/metabase:v0.60.3.3**.

**Langkah Menjalankan Dashboard Metabase via Docker:**
1. Pastikan aplikasi Docker Desktop sudah berjalan di sistem Anda.
2. Buka terminal dan jalankan *container* Metabase baru menggunakan perintah berikut (Wajib menggunakan versi ini agar tidak terjadi *crash* perbedaan versi *database*):
   `docker run -d -p 3000:3000 --name metabase metabase/metabase:v0.60.3.3`
3. Salin (*copy*) file `metabase.db.mv.db` yang telah dilampirkan ke dalam *container* Metabase yang sedang berjalan. Jalankan perintah berikut di direktori tempat file tersebut berada:
   `docker cp metabase.db.mv.db metabase:/metabase.db/metabase.db.mv.db`
4. Lakukan *restart* pada *container* agar Metabase memuat database yang baru saja di-*copy*:
   `docker restart metabase`
5. Akses *dashboard* melalui *browser* di tautan: `http://localhost:3000`
6. Silakan login menggunakan kredensial berikut:
   - **Email:** root@mail.com
   - **Password:** root123

*Catatan: Gambar (Screenshot) dashboard dalam bentuk `.png` telah dilampirkan dalam folder proyek (`vito_gunawan-dashboard.png`) sebagai bukti visual pemenuhan kriteria 3.*

## Conclusion
**Faktor-Faktor Penyebab Attrition (Berdasarkan Batasan Empiris):**
Berdasarkan hasil Exploratory Data Analysis (EDA) yang dilakukan pada **seluruh fitur** (baik kategorikal maupun numerik) tanpa terkecuali, ditemukan batasan-batasan nyata (threshold) dari karakteristik karyawan yang melakukan *attrition*:
- **OverTime:** Status beban jam kerja tambahan (*OverTime = Yes*) secara signifikan menjadi faktor utama yang paling membedakan karyawan resign vs. bertahan.
- **Batasan Pendapatan (MonthlyIncome):** Terdapat batas kritis (threshold) di mana karyawan dengan *Monthly Income* di bawah **$3.000 per bulan** memiliki rasio attrition yang meroket sangat tinggi dibandingkan karyawan dengan rentang gaji di atasnya.
- **Batasan Usia (Age):** Karyawan yang berusia kurang dari batas **35 tahun** (dengan puncak kerentanan di rentang usia 28-31 tahun) sangat mendominasi jumlah karyawan yang resign.
- **Batasan Masa Kerja:** Karyawan dengan lama bekerja (*TotalWorkingYears*) kurang dari batas **3 tahun** memiliki probabilitas resign terbesar, menunjukkan fase krusial *job-hopping* atau kejenuhan awal.
*Fitur kategorikal lain seperti Gender, Education, atau BusinessTravel telah dianalisis namun terbukti tidak memiliki perbedaan (signifikansi) yang cukup kuat terhadap rasio Attrition.*

**Performa Model Kuantitatif:**
Model *Random Forest Classifier* dengan teknik penyeimbangan kelas otomatis (`class_weight='balanced'`) dan *RandomizedSearchCV* mampu mengklasifikasikan kelas dengan sangat stabil. Akurasi secara keseluruhan mencapai rentang **~85% hingga 88%**. Model ini sukses menyeimbangkan skor *Precision* dan *Recall* untuk mendeteksi semaksimal mungkin kasus kelas minoritas (Attrition = Yes) tanpa bantuan penambahan data sintetis yang berisiko manipulatif seperti SMOTE. Melalui grafik *Feature Importance*, model membuktikan secara kuantitatif bahwa **OverTime, Age, dan Monthly Income** adalah tiga fitur terpenting yang memiliki bobot terbesar dalam algoritma pembentukan keputusan prediksi.

### Rekomendasi Action Items (Optional)
Berdasarkan temuan batasan empiris di atas, berikut adalah rancangan aksi konkrit untuk diimplementasikan oleh pihak HR Jaya Jaya Institut:

1. **Penyesuaian Kompensasi Terarah untuk Kelompok Gaji < $3.000:** 
   Karena data dengan jelas menunjukkan probabilitas resign melesat pada karyawan berpenghasilan rendah, perusahaan dapat melakukan program *retention bonus* atau peningkatan gaji berkala senilai 10-15% spesifik untuk segmentasi gaji di bawah $3.000 ini, dibayarkan pada evaluasi paruh tahun (setiap 6 bulan). Biaya intervensi ini akan jauh lebih murah dibandingkan kerugian biaya operasional/rekrutmen akibat attrition.
2. **Pembatasan dan Kompensasi Cuti untuk *OverTime*:** 
   Sebagai variabel prediktor nomor satu (Top 1 Feature Importance), beban kerja di Jaya Jaya Institut sangat krusial. Perusahaan harus memberlakukan sistem pergantian shift (terutama di departemen berisiko) dan memastikan tidak ada karyawan yang terus-terusan berstatus *OverTime = Yes*. Jika harus lembur, HRD perlu mengimplementasikan kompensasi berupa *Wellness Leave* (1 hari cuti ekstra per bulan untuk yang melampaui kuota lembur tertentu).
3. **Program *Mentorship* & Percepatan Karir untuk Tahun ke 1-3:** 
   Berdasarkan profil rentan (Usia <35 tahun dan masa kerja <3 tahun), rancanglah program pengembangan talenta muda atau skema *Fast-Track Career* agar generasi ini bisa melihat pijakan karir mereka di Institut pada usia emasnya. Berikan mentor senior khusus yang bertugas melakukan evaluasi 1-on-1 (One-on-One) setiap triwulan, bukan tahunan, untuk mempertahankan motivasi mereka secara proaktif.
