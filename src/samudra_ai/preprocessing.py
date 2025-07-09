# preprocessing.py

import xarray as xr
import numpy as np
import pandas as pd
import cftime
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

from .utils import to_np_datetime64_safe

def load_and_mask_dataset(
    file_path: str,
    var_name: str,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    time_range: Tuple[str, str]
) -> xr.DataArray:
    """Membuka, memotong, dan me-masking dataset NetCDF dari path file."""
    with xr.open_dataset(file_path, engine="h5netcdf", decode_times=True) as data:
        if var_name not in data.variables:
            raise ValueError(f"Variabel '{var_name}' tidak ditemukan. Tersedia: {list(data.variables.keys())}")
        if 'time' not in data.coords:
            raise ValueError("Koordinat 'time' tidak ditemukan dalam dataset.")

        # Deteksi nama dimensi lat/lon dari skrip asli
        lat_names = ["lat", "latitude", "j", "y"]
        lon_names = ["lon", "longitude", "i", "x"]
        detected_lat = next((lat for lat in lat_names if lat in data.dims), None)
        detected_lon = next((lon for lon in lon_names if lon in data.dims), None)
        if detected_lat is None or detected_lon is None:
            raise ValueError("Dimensi latitude dan longitude tidak ditemukan dalam dataset.")

        # Logika validasi dan seleksi waktu dari skrip asli
        time_type = type(data.time.values[0])
        if issubclass(time_type, (cftime.Datetime360Day, cftime.DatetimeNoLeap, cftime.DatetimeGregorian)):
            dt1 = pd.to_datetime(time_range[0])
            dt2 = pd.to_datetime(time_range[1])
            start_time_obj = time_type(dt1.year, dt1.month, dt1.day)
            end_time_obj = time_type(dt2.year, dt2.month, dt2.day)
        else:
            start_time_obj = np.datetime64(time_range[0], 'D')
            end_time_obj = np.datetime64(time_range[1], 'D')
        
        start_time_sel = data.time.sel(time=start_time_obj, method="nearest").values
        end_time_sel = data.time.sel(time=end_time_obj, method="nearest").values

        sliced_data = data[var_name].sel(time=slice(start_time_sel, end_time_sel))
        
        # Slicing spasial
        masked_data = sliced_data.sel(
            {detected_lat: slice(lat_range[0], lat_range[1]),
             detected_lon: slice(lon_range[0], lon_range[1])}
        ).dropna(dim="time", how="all")

        if masked_data.size == 0:
            raise ValueError("Data kosong setelah proses slicing dan masking.")
        return masked_data

def normalize_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Menormalisasi data input (X) dan target (y) menggunakan MinMaxScaler."""
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_reshaped = X.reshape(X.shape[0], -1)
    X_scaled = scaler_X.fit_transform(X_reshaped).reshape(X.shape)
    
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_reshaped = y.reshape(y.shape[0], -1)
    y_scaled = scaler_y.fit_transform(y_reshaped).reshape(y.shape)
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def create_sequences(data: np.ndarray, time_window: int) -> np.ndarray:
    """Membuat sekuens time-series dari sebuah array."""
    num_sequences = len(data) - time_window + 1
    if num_sequences <= 0:
        # Mengembalikan array kosong dengan shape yang benar jika data tidak cukup
        return np.array([]).reshape(0, time_window, *data.shape[1:])
    
    sequences = np.array([data[i:i+time_window] for i in range(num_sequences-1)])
    return sequences
