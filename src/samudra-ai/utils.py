# library

import xarray as xr
import numpy as np
import pandas as pd
import cftime
import os
from typing import Tuple, Optional, Dict

def to_np_datetime64_safe(t):
    """Konversi waktu (termasuk cftime) ke numpy.datetime64 dengan aman."""
    try:
        if isinstance(t, (cftime.DatetimeNoLeap, cftime.DatetimeGregorian)):
            return np.datetime64(t.strftime('%Y-%m-%d'), 'D')
        return np.datetime64(str(t), 'D')
    except Exception:
        return np.datetime64(pd.to_datetime(t).date(), 'D')

def get_lat_lon_coords(data_array: xr.DataArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mengekstrak koordinat latitude dan longitude dari DataArray.

    Args:
        data_array (xr.DataArray): DataArray input.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple berisi array latitude dan longitude.
    """
    lat_name = next((coord for coord in data_array.coords if 'lat' in coord.lower()), None)
    lon_name = next((coord for coord in data_array.coords if 'lon' in coord.lower()), None)

    if lat_name is None or lon_name is None:
        raise ValueError("Koordinat latitude/longitude tidak ditemukan dalam DataArray.")

    return data_array.coords[lat_name].values, data_array.coords[lon_name].values

def save_to_netcdf(
    data_array: xr.DataArray,
    var_name: str,
    output_path: str,
    attributes: Optional[Dict[str, str]] = None,
):
    """
    Menyimpan DataArray ke file NetCDF dengan metadata yang fleksibel.

    Args:
        data_array (xr.DataArray): Data yang akan disimpan.
        var_name (str): Nama untuk variabel di dalam file NetCDF.
        output_path (str): Path lengkap untuk file output (misal: 'hasil/data.nc').
        attributes (Optional[Dict[str, str]]): Atribut untuk ditambahkan ke variabel.
    """
    # Pastikan direktori output ada
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    dataset = data_array.to_dataset(name=var_name)
    
    if attributes:
        dataset[var_name].attrs.update(attributes)
    
    dataset.to_netcdf(output_path, engine="h5netcdf")
    print(f"💾 Hasil disimpan ke: {output_path}")
