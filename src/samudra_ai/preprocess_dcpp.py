# File: src/samudra_ai/preprocess_dcpp.py
import xarray as xr
import pandas as pd
import numpy as np

def preprocess_dcpp(file_path: str, var_name: str, lat_range=None, lon_range=None, time_range=None):
    print(f"-> Preprocessing DCPP file: {file_path} (var: {var_name})")

    ds = xr.open_dataset(file_path, engine="h5netcdf", decode_times=True)

    if var_name not in ds:
        raise ValueError(f"Variabel {var_name} tidak ada dalam file {file_path}")

    da = ds[var_name]

    # === Hapus dimensi tidak relevan ===
    for dim in ["initial_time", "ensemble", "member", "zlev"]:
        if dim in da.dims:
            da = da.isel({dim: 0}).squeeze(dim)

    # Kalau masih ada dimensi extra selain time/lat/lon → ambil indeks pertama
    while len(da.dims) > 3:
        for d in da.dims:
            if d not in ["time", "lat", "lon"]:
                da = da.isel({d: 0}).squeeze(d)

    # === Deteksi & rename dimensi waktu ===
    time_candidates = ["time", "valid_time", "date", "Time", "t"]
    detected_time = next((t for t in time_candidates if t in da.dims or t in da.coords), None)
    if not detected_time:
        raise ValueError("Dimensi waktu tidak ditemukan di file DCPP.")

    if detected_time != "time":
        da = da.rename({detected_time: "time"})

    # === Slicing waktu kalau diberikan ===
    if time_range:
        try:
            start_dt = pd.to_datetime(time_range[0])
            end_dt = pd.to_datetime(time_range[1])
        except Exception:
            raise ValueError("Format time_range harus bisa dibaca pandas, misal '1981-01-01'")
        if start_dt >= end_dt:
            raise ValueError("time_range: tanggal awal harus lebih kecil dari tanggal akhir")

        time_type = type(da.time.values[0])
        if "cftime" in str(time_type):
            start_time = time_type(start_dt.year, start_dt.month, start_dt.day)
            end_time = time_type(end_dt.year, end_dt.month, end_dt.day)
        else:
            start_time = np.datetime64(start_dt, "D")
            end_time = np.datetime64(end_dt, "D")

        start_time_sel = da.time.sel(time=start_time, method="nearest").values
        end_time_sel = da.time.sel(time=end_time, method="nearest").values
        da = da.sel(time=slice(start_time_sel, end_time_sel))

    # === Slicing lat/lon kalau diberikan ===
    if lat_range and lon_range:
        lat_names = ["lat", "latitude", "j", "y"]
        lon_names = ["lon", "longitude", "i", "x"]
        detected_lat = next((lat for lat in lat_names if lat in da.dims), None)
        detected_lon = next((lon for lon in lon_names if lon in da.dims), None)
        if detected_lat and detected_lon:
            da = da.sel({detected_lat: slice(*lat_range), detected_lon: slice(*lon_range)})

    # === Pastikan urutan dimensi standar ===
    if set(["time", "lat", "lon"]).issubset(set(da.dims)):
        da = da.transpose("time", "lat", "lon")

    return da
