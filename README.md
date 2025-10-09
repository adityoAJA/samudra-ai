# samudra-ai
Paket Python untuk melakukan pengolahan koreksi bias model iklim global menggunakan arsitektur deep learning CNN-BiLSTM dan CONVLSTM2D

# SamudraAI 🌊

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![PyPI version](https://badge.fury.io/py/samudra-ai.svg)](https://pypi.org/project/samudra-ai/)
[![Python](https://img.shields.io/pypi/pyversions/samudra-ai.svg)](https://pypi.org/project/samudra-ai/)

Paket Python untuk koreksi bias model iklim menggunakan arsitektur deep learning CNN-BiLSTM. 

**SamudraAI** memudahkan peneliti dan praktisi di bidang ilmu iklim untuk menerapkan metode koreksi bias yang canggih pada data GCM (General Circulation Model) menggunakan data observasi sebagai referensi.

## Fitur Utama

* 🧠 **Arsitektur CNN-BiLSTM**: Menggabungkan kemampuan ekstraksi fitur spasial dari CNN dilanjutkan dengan pemahaman sekuens temporal dari LSTM.
* 🧠 **Arsitektur CONVLSTM2D**: Kemampuan ekstraksi fitur spasial dan temporal dari CNN dan LSTM yang jalan secara simultan.
* 📂 **Antarmuka Sederhana**: API yang bersih dan mudah digunakan, terinspirasi oleh `scikit-learn`.
* 🛠️ **Pra-pemrosesan Terintegrasi**: Fungsi bawaan untuk memuat, memotong, dan menormalisasi data iklim dalam format NetCDF.
* 💾 **Model Persistent**: Kemampuan untuk menyimpan model yang telah dilatih dan memuatnya kembali untuk inferensi di kemudian hari.

## Instalasi

Anda dapat menginstal SamudraAI langsung dari PyPI menggunakan pip:

```bash
pip install samudra-ai
```

## Best Practice

* ✅ Disarankan menggunakan TensorFlow GPU untuk performa optimal
* ✅ Disarankan memiliki memory / RAM yang cukup untuk pengolahan data dengan resolusi tinggi dan luasan domain yang besar
* ✅ Jalankan pelatihan secara penuh di lingkungan lokal
* ⚠️ Hindari mencampur save/load model .keras antar environment yang berbeda
* ⚠️ Menggunakan Docker tetap bisa berjalan, namun proses save and load (penggunaan no.5) tidak bisa diproses karena perbedaan env
* 💡 Format .nc hasil koreksi bisa langsung digunakan untuk plotting dan analisis

## Lisensi

Proyek ini dilisensikan di bawah **MIT License**. Lihat file `LICENSE` untuk detailnya.
