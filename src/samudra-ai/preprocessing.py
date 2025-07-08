# library

import xarray as xr
import numpy as np
import pandas as pd
import cftime
from sklearn.preprocessing import MinMaxScaler
from skimage.transform import resize
from typing import Tuple, Optional, Dict

# --- Helper Functions ---

def to_np_datetime64_safe(t):
    """Konversi waktu (termasuk cftime) ke numpy.datetime64 dengan aman."""
    try:
        if isinstance(t, (cftime.DatetimeNoLeap, cftime.DatetimeGregorian)):
            return np.datetime64(t.strftime('%Y-%m-%d'), 'D')
        return np.datetime64(str(t), 'D')
    except Exception:
        return np.datetime64(pd.to_datetime(t).date(), 'D')

def standardize_dims(ds: xr.Dataset) -> xr.Dataset:
    """Menyeragamkan nama dimensi spasial dan waktu ke ['time', 'latitude', 'longitude']."""
    rename_dict = {}
    # ... (logika Anda sudah bagus) ...
    return ds.rename(rename_dict) if rename_dict else ds

# --- Main Preprocessing Functions ---

def load_and_mask_dataset(
    file_path: str, 
    var_name: str, 
    lat_range: Tuple[float, float], 
    lon_range: Tuple[float, float], 
    time_range: Tuple[str, str]
) -> xr.DataArray:
    """
    Membuka, memotong, dan me-masking dataset NetCDF dari path file.
    
    Args:
        file_path (str): Path menuju file .nc.
        var_name (str): Nama variabel yang akan diekstrak.
        lat_range (Tuple[float, float]): Rentang lintang (min, max).
        lon_range (Tuple[float, float]): Rentang bujur (min, max).
        time_range (Tuple[str, str]): Rentang waktu ('YYYY-MM-DD', 'YYYY-MM-DD').

    Returns:
        xr.DataArray: DataArray yang sudah dipotong dan di-masking.
    """
    # Membuka dataset di dalam fungsi
    with xr.open_dataset(file_path, engine="h5netcdf", decode_times=True) as data:
        # Validasi awal
        if var_name not in data.variables:
            raise ValueError(f"Variabel '{var_name}' tidak ditemukan. Tersedia: {list(data.variables.keys())}")
        
        # ... sisa logika validasi dan slicing Anda yang sudah bagus bisa dimasukkan di sini ...
        # ... (Kode untuk validasi waktu, seleksi, dan slicing) ...
        
        # Contoh slicing (disederhanakan, gunakan logika Anda yang lebih lengkap)
        sliced_data = data[var_name].sel(
            time=slice(time_range[0], time_range[1]),
            latitude=slice(lat_range[0], lat_range[1]),
            longitude=slice(lon_range[0], lon_range[1])
        ).dropna(dim="time", how="all")

        if sliced_data.size == 0:
            raise ValueError("Data kosong setelah slicing.")
            
        return sliced_data

def normalize_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Menormalisasi data input (X) dan target (y) menggunakan MinMaxScaler."""
    # ... (Kode Anda sudah sempurna) ...
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_reshaped = X.reshape(X.shape[0], -1)
    X_scaled = scaler_X.fit_transform(X_reshaped).reshape(X.shape)
    
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_reshaped = y.reshape(y.shape[0], -1)
    y_scaled = scaler_y.fit_transform(y_reshaped).reshape(y.shape)
    
    return X_scaled, y_scaled, scaler_X, scaler_y

def create_sequences(data: np.ndarray, time_window: int) -> np.ndarray:
    """Membuat sekuens time-series dari sebuah array."""
    # ... (Kode Anda sudah sempurna) ...
    sequences = []
    if len(data) < time_window:
        return np.array([]).reshape(0, time_window, *data.shape[1:])
    for i in range(len(data) - time_window + 1):
        sequences.append(data[i : i + time_window])
    return np.array(sequences)

# --- Output Function ---

def save_to_netcdf(
    data_array: xr.DataArray,
    var_name: str,
    output_path: str,
    attributes: Optional[Dict[str, str]] = None,
    clip_range: Optional[Tuple[Optional[float], Optional[float]]] = None
):
    """
    Menyimpan DataArray ke file NetCDF dengan metadata yang fleksibel.

    Args:
        data_array (xr.DataArray): Data yang akan disimpan.
        var_name (str): Nama untuk variabel di dalam file NetCDF.
        output_path (str): Path lengkap untuk file output (misal: 'hasil/data.nc').
        attributes (Optional[Dict[str, str]]): Atribut untuk ditambahkan ke variabel.
        clip_range (Optional[Tuple[...]]): Rentang untuk memotong data (min, max).
    """
    # Menghapus logika hardcoded
    if clip_range is not None:
        data_array = data_array.clip(min=clip_range[0], max=clip_range[1])

    dataset = data_array.to_dataset(name=var_name)
    
    if attributes:
        dataset[var_name].attrs.update(attributes)
    
    dataset.to_netcdf(output_path, engine="h5netcdf")
    print(f"✅ Hasil disimpan: {output_path}")
