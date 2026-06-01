# ♻️ EcoSort AI: Smart Waste Forecasting & Classification

Selamat datang di repositori Data Science untuk proyek Capstone **EcoSort AI** (Coding Camp 2026). Proyek ini berfokus pada solusi cerdas pengelolaan sampah di wilayah Kota Bandung, menggabungkan klasifikasi limbah dan prediksi volume sampah untuk mendukung tata kelola lingkungan yang lebih baik.

## 🎯 Titik Awal Solusi
Beranjak dari pengumpulan dan analisis mendalam terhadap berbagai problematika lingkungan perkotaan, kami memformulasikan satu solusi utama yang berfokus pada efisiensi penyortiran dan pemantauan distribusi persampahan. Seluruh pengembangan pada tahap ini didasarkan pada definisi pertanyaan bisnis yang terukur untuk memastikan metrik kesuksesan yang kami buat tepat sasaran.

## 📂 Peta Perjalanan Data

Proyek ini dibangun dari nol, di mana kami sangat menghindari penggunaan dataset matang yang siap pakai secara instan. Kami menyusun struktur data yang kuat melalui tahapan yang komprehensif pada komponen berikut:

### 1. Eksplorasi & Klasifikasi (`dataset_klasifikasi_sampah.ipynb`)
Notebook ini merupakan pusat pemrosesan dari bahan mentah hingga menjadi informasi yang berharga.
- **Pembersihan Menyeluruh:** Meliputi tahapan *Data Gathering* dari berbagai sumber mentah, *Data Assessing* untuk menginspeksi kualitas struktur, hingga *Data Cleaning* yang dieksekusi secara mandiri untuk merapikan anomali.
- **Bercerita dengan Data:** *Exploratory Data Analysis* (EDA) dilakukan untuk menggali pola, di mana kami memastikan setiap langkah analisis selalu dibarengi dengan penjabaran tekstual (*Markdown*) agar logika pemrosesan dapat diikuti dengan mudah.
- **Keputusan Berbasis Bukti:** Setiap penarikan kesimpulan tidak kami biarkan menggantung, melainkan selalu divalidasi langsung oleh visualisasi data melalui tahapan *Explanatory Analysis*.
- **Kesiapan Pemodelan:** Hasil akhir pemrosesan dibungkus ke dalam format terkompresi (`.zip`) dan dikonfigurasi melalui `dataset_meta.json` yang sekaligus bertindak sebagai *Data Dictionary*. Dataset dipastikan dalam kondisi prima dan 100% siap diproses pada tahapan pemodelan.

### 2. Rekayasa Masa Depan (`Prediksi_sampah.ipynb`)
Notebook ini mengawal pengembangan fitur prediktif tambahan untuk peramalan tren dan volume.
- **Kaya Akan Konteks:** Melalui tahapan *Feature Engineering*, kami berinovasi menciptakan variabel-variabel (*features*) turunan yang lebih kaya dan informatif untuk menstimulasi akurasi model.
- **Keamanan Pipeline:** Struktur data dibentuk dengan pengawasan ketat sehingga informasi *target* prediksi masa depan tidak menyusup atau bocor ke dalam atribut pelatihan (*zero data leakage*).
- Output dari tahapan *Data Wrangling* di sini adalah dataset berformat `.csv` yang sudah matang dan siap santap untuk pelatihan algoritma.

## 📊 Jendela Wawasan Interaktif

Seluruh wawasan (*insight*), kesimpulan bisnis, dan analisis metrik yang kami peroleh tidak hanya berakhir di *notebook*. Kami telah merangkum dan menerjemahkannya menjadi sebuah presentasi visual yang interaktif melalui pengembangan dashboard menggunakan **Streamlit** (`app.py`).

Sebagai sentuhan akhir, dashboard ini telah berhasil dipublikasikan (*deploy*) ke Streamlit Cloud, sehingga dapat diakses secara terbuka oleh publik:

🌍 **Live Dashboard:** [EcoSort Dashboard - Capstone DBS](https://ecosortdashboardcapstonedbs.streamlit.app/)

---
*Dikembangkan oleh Tim Data Science untuk menuntaskan permasalahan riil dengan solusi berbasis data.*
