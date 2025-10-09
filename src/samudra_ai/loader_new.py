import os
import xarray as xr
import pandas as pd
import numpy as np
import cftime
from .utils import standardize_dims

def load_and_mask_dataset(file_path: str, var_name: str, lat_range: tuple, lon_range: tuple, time_range: tuple) -> xr.DataArray:
    print(f"-> Memuat dan memproses file: {os.path.basename(file_path)}...")

    # === Validasi time_range ===
    if not (isinstance(time_range, (list, tuple)) and len(time_range) == 2):
        raise ValueError("time_range harus tuple (start, end) dengan format YYYY-MM-DD")
    start_dt, end_dt = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
    if start_dt >= end_dt:
        raise ValueError("time_range: tanggal awal harus lebih kecil dari tanggal akhir")

    # === Pilih engine sesuai jenis file ===
    engine = "netcdf4" if "gn" in file_path.lower() else "h5netcdf"

    with xr.open_dataset(file_path, engine=engine, decode_times=True) as data:
        if var_name not in data.variables:
            raise ValueError(f"Variabel '{var_name}' tidak ditemukan di dataset.")

        # === Deteksi & rename waktu ===
        time_candidates = ["time", "valid_time", "date", "Time", "t"]
        detected_time = next((t for t in time_candidates if t in data.dims or t in data.coords), None)
        if not detected_time:
            raise ValueError("Dimensi waktu tidak ditemukan.")
        if detected_time != "time":
            data = data.rename({detected_time: "time"})

        # === Konversi range waktu ===
        time_type = type(data.time.values[0])
        if 'cftime' in str(time_type):
            start_time, end_time = time_type(start_dt.year, start_dt.month, start_dt.day), time_type(end_dt.year, end_dt.month, end_dt.day)
        else:
            start_time, end_time = np.datetime64(start_dt, 'D'), np.datetime64(end_dt, 'D')

        sliced_data = data[var_name].sel(time=slice(start_time, end_time))

        # === Deteksi dimensi spasial ===
        lat_names = ["lat", "latitude", "j", "y"]
        lon_names = ["lon", "longitude", "i", "x"]
        detected_lat = next((lat for lat in lat_names if lat in sliced_data.dims), None)
        detected_lon = next((lon for lon in lon_names if lon in sliced_data.dims), None)
        if not detected_lat or not detected_lon:
            raise ValueError("Dimensi lat/lon tidak ditemukan.")

        # === Sort lat/lon ===
        if not np.all(np.diff(sliced_data[detected_lat]) > 0):
            sliced_data = sliced_data.sortby(detected_lat)
        if not np.all(np.diff(sliced_data[detected_lon]) > 0):
            sliced_data = sliced_data.sortby(detected_lon)

        # === Deteksi grid gaussian ===
        is_gaussian = (
            'gaussian' in data.attrs.get('grid_label', '').lower() or
            'gn' in file_path.lower() or
            len(sliced_data[detected_lat]) > 180  # biasanya gaussian grid punya >180 latitude
        )

        if is_gaussian:
            print("⚠️ Gaussian grid (gn) terdeteksi → melakukan interpolasi ke grid reguler 1°x1°...")
            new_lat = np.arange(-90, 90.1, 1.0)
            new_lon = np.arange(0, 360, 1.0)
            try:
                sliced_data = sliced_data.interp({detected_lat: new_lat, detected_lon: new_lon})
            except Exception as e:
                print("❌ Gagal interpolasi gaussian grid:", e)
                raise

        # === Slicing spasial ===
        masked_data = sliced_data.sel(
            {detected_lat: slice(*lat_range), detected_lon: slice(*lon_range)}
        ).dropna(dim="time", how="all")

        if masked_data.size == 0:
            raise ValueError("Data kosong setelah slicing.")

        # === Standarisasi dimensi agar selalu lat/lon konsisten ===
        return standardize_dims(masked_data)
