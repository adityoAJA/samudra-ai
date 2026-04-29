# samudra-ai
Library Python samudra-ai versi ke-2

# SamudraAI 🌊

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
[![PyPI version](https://badge.fury.io/py/samudra-ai.svg)](https://pypi.org/project/samudra-ai/)
[![Python](https://img.shields.io/pypi/pyversions/samudra-ai.svg)](https://pypi.org/project/samudra-ai/)

Paket Python versi 2untuk melakukan pengolahan koreksi bias model iklim global menggunakan arsitektur deep learning CNN-BiLSTM dan CONVLSTM2D

**SamudraAI** memudahkan peneliti dan praktisi di bidang ilmu iklim untuk menerapkan metode koreksi bias yang canggih pada data GCM (General Circulation Model) menggunakan data observasi sebagai referensi.

## Fitur Utama

* 🧠 **Arsitektur CNN-BiLSTM**: Menggabungkan kemampuan ekstraksi fitur spasial dari CNN dilanjutkan dengan pemahaman sekuens temporal dari LSTM.
* 🧠 **Arsitektur CONVLSTM2D**: Kemampuan ekstraksi fitur spasial dan temporal dari CNN dan LSTM yang jalan secara simultan.
* 🧠 **Antarmuka Sederhana**: API yang bersih, sederhana dan mudah digunakan.
* 🛠️ **Pra-pemrosesan Terintegrasi**: Fungsi bawaan untuk memuat, memotong, dan menormalisasi data iklim dalam format NetCDF.
* 🛠️ **Tersedia transformasi log/expm**: Fungsi bawaan untuk memuat, memotong, dan menormalisasi data iklim khususnya data hujan.
* 🛠️ **Tersedia feature seasonal**: Fungsi bawaan untuk memasukkan faktor pola musiman khususnya pada data hujan.
* 💾 **Model Persistent**: Kemampuan untuk menyimpan model yang telah dilatih dan memuatnya kembali untuk diterapkan pada data dimasa depan.

## Instalasi

Anda dapat menginstal SamudraAI langsung dari PyPI menggunakan pip:

```bash
pip install samudra-ai
```

## Penggunaan Library
```
https://github.com/adityoAJA/samudra-ai/blob/cc71adc811d5b16ece2d6db74044124974eb1136/How-to-use-samudra-ai.txt
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
