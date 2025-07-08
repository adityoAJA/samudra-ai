# library

import numpy as np
import xarray as xr
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# Kita akan mengimpor fungsi dari modul kita sendiri
from .preprocessing import create_sequences

def perform_correction(
    model: Model,
    data_to_correct: xr.DataArray,
    scaler_X: MinMaxScaler,
    scaler_y: MinMaxScaler,
    time_window: int,
) -> xr.DataArray:
    """
    Menjalankan pipeline inferensi lengkap pada data baru menggunakan model yang sudah dilatih.

    Args:
        model (Model): Model Keras yang sudah dilatih.
        data_to_correct (xr.DataArray): DataArray input yang akan dikoreksi 
                                        (misalnya, data GCM masa depan).
        scaler_X (MinMaxScaler): Scaler yang sudah di-fit pada data input training.
        scaler_y (MinMaxScaler): Scaler yang sudah di-fit pada data target training.
        time_window (int): Ukuran jendela waktu yang digunakan saat training.

    Returns:
        xr.DataArray: DataArray hasil koreksi dengan koordinat yang sesuai.
    """
    print(f"🔬 Performing inference on data with shape {data_to_correct.shape}...")

    # 1. Persiapan Data Input
    #    - Konversi ke NumPy dan tangani NaN
    inference_input = np.nan_to_num(data_to_correct.values)
    #    - Normalisasi menggunakan scaler_X yang sudah ada
    inference_input_reshaped = inference_input.reshape(inference_input.shape[0], -1)
    inference_input_scaled = scaler_X.transform(inference_input_reshaped).reshape(inference_input.shape)
    
    # 2. Buat Sekuens Time-Series
    #    - Tambahkan dimensi channel sebelum membuat sekuens
    inference_seq = create_sequences(
        inference_input_scaled[..., np.newaxis], 
        time_window=time_window
    )
    
    if inference_seq.shape[0] == 0:
        print("⚠️ Warning: Not enough data to create any sequences for inference. Returning an empty DataArray.")
        return xr.DataArray(
            np.empty((0, *data_to_correct.shape[1:])),
            coords={
                "time": [],
                "latitude": data_to_correct.latitude.values,
                "longitude": data_to_correct.longitude.values
            },
            dims=["time", "latitude", "longitude"]
        )

    # 3. Jalankan Prediksi
    print(f"🧠 Running model.predict() on {inference_seq.shape[0]} sequences...")
    scaled_predictions = model.predict(inference_seq)
    scaled_predictions = scaled_predictions.squeeze(axis=-1)  # Hapus dimensi channel

    # 4. Pasca-Proses Hasil Prediksi
    #    - Lakukan inverse transform untuk mengembalikan ke skala asli
    predictions_flat = scaled_predictions.reshape(scaled_predictions.shape[0], -1)
    predictions_original_scale = scaler_y.inverse_transform(predictions_flat)
    final_predictions = predictions_original_scale.reshape(scaled_predictions.shape)
    
    # 5. Rekonstruksi menjadi DataArray dengan Koordinat yang Benar
    #    - Prediksi dimulai setelah jendela waktu pertama, jadi sesuaikan koordinat waktu
    valid_time_coords = data_to_correct.time.values[time_window - 1:]
    
    #    - Pastikan panjang waktu cocok dengan jumlah prediksi
    num_predictions = final_predictions.shape[0]
    valid_time_coords = valid_time_coords[:num_predictions]
    
    corrected_da = xr.DataArray(
        final_predictions,
        coords={
            "time": valid_time_coords,
            "latitude": data_to_correct.latitude.values,
            "longitude": data_to_correct.longitude.values,
        },
        dims=["time", "latitude", "longitude"],
        name=data_to_correct.name or "corrected_data"
    )
    
    print("✅ Inference finished.")
    return corrected_da
