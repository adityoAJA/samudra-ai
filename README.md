# samudra-ai
Paket Python untuk melakukan pengolahan koreksi bias model iklim global menggunakan arsitektur deep learning CNN-BiLSTM

# SamudraAI 🌊

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![PyPI version](https://img.shields.io/pypi/v/samudra-ai.svg)
![Python versions](https://img.shields.io/pypi/pyversions/samudra-ai.svg)

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
```bash
pip install numpy pandas xarray tensorflow scikit-learn matplotlib h5netcdf cftime scikit-image joblib
```

## Cara Penggunaan Cepat (Quick Start)

Berikut adalah alur kerja dasar untuk menggunakan `SamudraAI`.

### 1. Siapkan Data Anda
Pastikan Anda memiliki data dalam format `xarray.DataArray`:
* `gcm_hist_data`: Data GCM historis (sebagai input `X`).
* `obs_data`: Data observasi/reanalysis (sebagai target `y`).
* `gcm_future_data`: Data GCM masa depan yang ingin dikoreksi.

### 2. Latih Model
Impor kelas `SamudraAI`, inisialisasi, dan latih model dengan data Anda.

```python
import xarray as xr
from samudra_ai import SamudraAI

# Asumsikan data sudah dimuat ke dalam objek xarray DataArray
# gcm_hist_data = xr.open_dataarray(...)
# obs_data = xr.open_dataarray(...)

# Inisialisasi model
model = SamudraAI(time_seq=9)

# Latih model
# Proses ini akan menangani normalisasi, pembuatan sekuens, dan training
history = model.fit(
    x_data_hist=gcm_hist_data,
    y_data_obs=obs_data,
    epochs=50,
    batch_size=8
)

print("✅ Model selesai dilatih!")
```

### 3. Simpan Model yang Telah Dilatih
Sangat direkomendasikan untuk menyimpan model Anda agar tidak perlu melatih ulang.

```python
# Simpan semua komponen (model, scaler, konfigurasi) ke sebuah direktori
model.save("model_tersimpan/model_ssh_indo")
```

### 4. Lakukan Prediksi (Koreksi)
Muat kembali model yang tersimpan dan gunakan untuk mengoreksi data di masa depan.

```python
# Asumsikan gcm_future_data sudah dimuat
# gcm_future_data = xr.open_dataarray(...)

# Muat model dari direktori
loaded_model = SamudraAI.load("model_tersimpan/model_ssh_indo")

# Lakukan koreksi pada data baru
corrected_data = loaded_model.predict(gcm_future_data)

# Tampilkan atau simpan hasilnya
print(corrected_data)
# corrected_data.to_netcdf("hasil_koreksi.nc")
```

## Lisensi

Proyek ini dilisensikan di bawah **MIT License**. Lihat file `LICENSE` untuk detailnya.
