# samudra-ai
Paket Python untuk melakukan pengolahan koreksi bias model iklim global menggunakan arsitektur deep learning CNN-BiLSTM

# SamudraAI 🌊

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Paket Python untuk koreksi bias model iklim menggunakan arsitektur deep learning CNN-BiLSTM. 

**SamudraAI** memudahkan peneliti dan praktisi di bidang ilmu iklim untuk menerapkan metode koreksi bias yang canggih pada data GCM (General Circulation Model) menggunakan data observasi sebagai referensi.

## Fitur Utama

* 🧠 **Arsitektur CNN-BiLSTM**: Menggabungkan kemampuan ekstraksi fitur spasial dari CNN dengan pemahaman sekuens temporal dari LSTM.
* 📂 **Antarmuka Sederhana**: API yang bersih dan mudah digunakan, terinspirasi oleh `scikit-learn`.
* 🛠️ **Pra-pemrosesan Terintegrasi**: Fungsi bawaan untuk memuat, memotong, dan menormalisasi data iklim dalam format NetCDF.
* 💾 **Model Persistent**: Kemampuan untuk menyimpan model yang telah dilatih dan memuatnya kembali untuk inferensi di kemudian hari.

## Instalasi

Anda dapat menginstal SamudraAI langsung dari PyPI menggunakan pip:

```bash
pip install samudra-ai
```

## Cara Penggunaan Cepat (Quick Start)

Berikut adalah alur kerja dasar untuk menggunakan `SamudraAI`.

### 1. Siapkan Data Anda
Pastikan Anda memiliki data dalam format `xarray.DataArray`:
* `gcm_hist_data`: Data GCM historis (sebagai input `X`).
* `obs_data`: Data observasi/reanalysis (sebagai target `y`).
* `gcm_future_data`: Data GCM masa depan yang ingin dikoreksi.

```bash
### 2. import model
from samudra_ai import SamudraAI
from samudra_ai.data_loader import load_and_mask_dataset

### 3. Load GCM dan Observasi
gcm = load_and_mask_dataset("canesm5_historical_1993_2014.nc", "zos", (-15, 10), (90, 145), ("1993", "2014"))
obs = load_and_mask_dataset("cmems_obs_1993_2024.nc", "sla", (-15, 10), (90, 145), ("1993", "2014"))

### 4. Inisialisasi dan Training Model
model = SamudraAI(time_seq=9)
model.fit(gcm, obs, epochs=100)
model.plot_history(output_dir="hasil_plot/")

### 5. Simpan dan/atau muat ulang model
model.save("canesm5_model_final")
model = SamudraAI.load("canesm5_model_final")

### 6. Evaluasi Historical dan Simpan Hasil Koreksi
eval_df, corrected_hist = model.evaluate_and_plot(
    raw_gcm_data=gcm,
    ref_data=obs,
    var_name_ref="sla",
    output_dir="hasil_evaluasi/",
    save_corrected_path="canesm5_historical_terkoreksi.nc"
)

### 6. Koreksi Proyeksi SSP
ssp = load_and_mask_dataset("canesm5_ssp245_2015_2100.nc", "zos", (-15, 10), (90, 145), ("2025", "2100"))
corrected_proj = model.correction(ssp, save_path="canesm5_ssp245_terkoreksi.nc")
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
