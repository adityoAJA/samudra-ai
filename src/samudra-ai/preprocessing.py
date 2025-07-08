# library

import xarray as xr
import numpy as np
import pandas as pd
import cftime
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

# Impor dari modul utilitas kita sendiri
from .utils import to_np_datetime64_safe

def standardize_dims(ds: xr.Dataset) -> xr.Dataset:
    """Menyeragamkan nama dimensi spasial dan waktu ke ['time', 'latitude', 'longitude']."""
    rename_dict = {}
    lat_names = ["lat", "y", "j"]
    lon_names = ["lon", "x", "i"]
    
    detected_lat = next((name for name in lat_names if name in ds.dims), None)
    if detected_lat and detected_lat != "latitude":
        rename_dict[detected_lat] = "latitude"

    detected_lon = next((name for name in lon_names if name in ds.dims), None)
    if detected_lon and detected_lon != "longitude":
        rename_dict[detected_lon] = "longitude"
        
    if "year" in ds.dims and "year" != "time":
        rename_dict["year"] = "time"

    return ds.rename(rename_dict) if rename_dict else ds

def load_and_mask_dataset(
    file_path: str,
    var_name: str,
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    time_range: Tuple[str, str]
) -> xr.DataArray:
    """Membuka, memotong, dan me-masking dataset NetCDF dari path file."""
    with xr.open_dataset(file_path, engine="h5netcdf", decode_times=True) as data:
        data = standardize_dims(data)
        
        if var_name not in data.variables:
            raise ValueError(f"Variabel '{var_name}' tidak ditemukan. Tersedia: {list(data.variables.keys())}")
        
        if 'time' not in data.coords:
            raise ValueError("Koordinat 'time' tidak ditemukan dalam dataset.")
            
        # Logika validasi waktu yang detail dari skrip asli Anda
        time_values = data['time'].values
        time_type = type(time_values[0])
        
        if issubclass(time_type, (cftime.Datetime360Day, cftime.DatetimeNoLeap, cftime.DatetimeGregorian)):
            dt1 = pd.to_datetime(time_range[0])
            dt2 = pd.to_datetime(time_range[1])
            start_time_obj = time_type(dt1.year, dt1.month, dt1.day)
            end_time_obj = time_type(dt2.year, dt2.month, dt2.day)
        else:
            start_time_obj = np.datetime64(time_range[0], 'D')
            end_time_obj = np.datetime64(time_range[1], 'D')
        
        sliced_data = data[var_name].sel(
            time=slice(start_time_obj, end_time_obj),
            latitude=slice(lat_range[0], lat_range[1]),
            longitude=slice(lon_range[0], lon_range[1])
        ).dropna(dim="time", how="all")

        if sliced_data.size == 0:
            raise ValueError("Data kosong setelah proses slicing dan masking.")
            
        return sliced_data

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
    sequences = []
    if len(data) < time_window:
        return np.array([]).reshape(0, time_window, *data.shape[1:])
    for i in range(len(data) - time_window + 1):
        sequences.append(data[i : i + time_window])
    return np.array(sequences)
