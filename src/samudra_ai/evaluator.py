# File: src/samudra_ai/evaluator.py
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from shapely.geometry import Point
from shapely.prepared import prep
from scipy.interpolate import interpn
from .utils import compute_metrics, standardize_dims

def evaluate_model(ref_data, raw_gcm_data, corrected_data, var_name, output_dir):
    print("\n🔎 Memulai proses evaluasi dan plotting otomatis...")

    # Standarisasi dimensi time-lat-lon
    ref_data = standardize_dims(ref_data)
    raw_gcm_data = standardize_dims(raw_gcm_data)
    corrected_data = standardize_dims(corrected_data)

    # Potong ke time length minimum
    time_len = min(len(ref_data.time), len(raw_gcm_data.time), len(corrected_data.time))
    ref_sliced = ref_data.isel(time=slice(0, time_len))
    raw_sliced = raw_gcm_data.isel(time=slice(0, time_len))
    corr_sliced = corrected_data.isel(time=slice(0, time_len))

    # Ambil grid target dari referensi
    target_lat = ref_sliced['lat'].values
    target_lon = ref_sliced['lon'].values
    target_grid_lat, target_grid_lon = np.meshgrid(target_lat, target_lon, indexing='ij')
    target_points = np.array([target_grid_lat.ravel(), target_grid_lon.ravel()]).T

    def manual_interp(source_da, target_da):
        # Pastikan dimensi [time, lat, lon]
        if source_da.dims[-2:] != ("lat", "lon"):
            source_da = source_da.transpose("time", "lat", "lon")

        source_lat = source_da['lat'].values
        source_lon = source_da['lon'].values

        # Validasi monoton
        if not (np.all(np.diff(source_lat) > 0) or np.all(np.diff(source_lat) < 0)):
            raise ValueError("Latitude harus monoton. Urutkan ulang dimensi 'lat'.")
        if not (np.all(np.diff(source_lon) > 0) or np.all(np.diff(source_lon) < 0)):
            raise ValueError("Longitude harus monoton. Urutkan ulang dimensi 'lon'.")

        source_points = (source_lat, source_lon)
        interpolated_values = np.array([
            interpn(
                source_points, source_da.isel(time=t).values, target_points,
                method='linear', bounds_error=False, fill_value=np.nan
            ).reshape(target_grid_lat.shape)
            for t in range(source_da.shape[0])
        ])
        # return xr.DataArray(interpolated_values, coords=ref_sliced.coords, dims=ref_sliced.dims)
        return xr.DataArray(
            interpolated_values,
            coords={"time": target_da["time"], "lat": target_da["lat"], "lon": target_da["lon"]},
            dims=["time", "lat", "lon"]
        )

    # Interpolasi raw GCM ke grid ref
    raw_aligned = manual_interp(raw_sliced, ref_sliced)

    # Rata-rata spasial jadi timeseries
    ref_series = ref_sliced.mean(dim=['lat', 'lon'])
    raw_series = raw_aligned.mean(dim=['lat', 'lon'])
    corr_series = corr_sliced.mean(dim=['lat', 'lon'])

    # Hitung metrik evaluasi
    metrics_raw = compute_metrics(ref_series.values, raw_series.values)
    metrics_corr = compute_metrics(ref_series.values, corr_series.values)
    results_df = pd.DataFrame([
        {'Source': 'GCM Asli', **dict(zip(['Correlation', 'RMSE', 'Bias', 'MAE'], metrics_raw))},
        {'Source': 'GCM Terkoreksi', **dict(zip(['Correlation', 'RMSE', 'Bias', 'MAE'], metrics_corr))},
    ])

    # Simpan summary metrics
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        results_df.to_excel(os.path.join(output_dir, "summary_metrics.xlsx"), index=False)

    # Simpan timeseries per timestep
    df_line = pd.DataFrame({
        "Time": ref_series.time.values,
        "GCM Asli": raw_series.values,
        "GCM Terkoreksi": corr_series.values,
        "Reanalysis (Obs)": ref_series.values
    })

    if output_dir:
        df_line.to_excel(os.path.join(output_dir, "timeseries_comparison.xlsx"), index=False)

    # Plot timeseries
    plt.figure(figsize=(15, 6))
    sns.lineplot(data=df_line)
    plt.title(f"Grafik Perbandingan Time Series {var_name.upper()}")
    plt.grid(True)
    if output_dir:
        plt.savefig(os.path.join(output_dir, "timeseries_comparison.png"), dpi=300)
    plt.show()

    # Plot bar metrik
    df_melt = results_df.melt(id_vars=['Source'], var_name='Metric', value_name='Value')
    plt.figure(figsize=(10, 7))
    sns.barplot(data=df_melt, x='Metric', y='Value', hue='Source', palette='viridis')
    plt.title("Perbandingan Metrik Evaluasi")
    if output_dir:
        plt.savefig(os.path.join(output_dir, "metrics_barplot.png"), dpi=300)
    plt.show()

    return results_df, corr_sliced

def _create_land_mask(lat, lon, resolution="10m"):
    """Buat masker daratan berdasarkan NaturalEarthFeature."""
    land_feature = cfeature.NaturalEarthFeature("physical", "land", resolution)
    land_geom = list(land_feature.geometries())
    land_prep = [prep(g) for g in land_geom]

    mask = np.zeros((len(lat), len(lon)), dtype=bool)
    for i, la in enumerate(lat):
        for j, lo in enumerate(lon):
            pt = Point(lo, la)
            if any(g.contains(pt) for g in land_prep):
                mask[i, j] = True
    return mask

def mask_land(da, resolution="10m"):
    """Masking hanya daratan."""
    lat = da["lat"].values
    lon = da["lon"].values
    mask = _create_land_mask(lat, lon, resolution)
    return da.where(mask)


def mask_ocean(da, resolution="10m"):
    """Masking hanya lautan (kebalikan dari mask_land)."""
    lat = da["lat"].values
    lon = da["lon"].values
    mask = _create_land_mask(lat, lon, resolution)
    return da.where(~mask)

def plot_spatial_comparison(corrected_da, raw_gcm_da, ref_da, var_name, output_dir=None,
                            units="", mask_type=None):
    
    print("🌍 Membuat plot spasial perbandingan rata-rata...")

    # Hitung rata-rata sepanjang waktu
    out_mean = corrected_da.mean(dim="time")
    gcm_mean = raw_gcm_da.mean(dim="time")
    ref_mean = ref_da.mean(dim="time")

    # Masking opsional
    if mask_type == "land":
        out_mean, gcm_mean, ref_mean = mask_land(out_mean), mask_land(gcm_mean), mask_land(ref_mean)
    elif mask_type == "ocean":
        out_mean, gcm_mean, ref_mean = mask_ocean(out_mean), mask_ocean(gcm_mean), mask_ocean(ref_mean)

    datasets = [out_mean, gcm_mean, ref_mean]
    titles = ["Model Terkoreksi", "GCM Asli", "Observasi (Reanalysis)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"projection": ccrs.PlateCarree()})

    for ax, da, title in zip(axes, datasets, titles):
        im = ax.pcolormesh(
            da["lon"], da["lat"], da,
            cmap="coolwarm", transform=ccrs.PlateCarree()
        )
        ax.set_aspect('auto')
        ax.coastlines()
        ax.set_title(title)

    # === buat axis khusus untuk colorbar di bawah ===
    cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])  
    # [left, bottom, width, height] dalam koordinat figure (0–1)
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(units)

    plt.suptitle(f"Peta Perbandingan Rata-rata Spasial {var_name.upper()}", fontsize=14)
    # plt.tight_layout(rect=[0, 0.15, 1, 0.95])
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.2, wspace=0.25)

    save_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"spatial_comparison_{var_name}_{mask_type or 'global'}.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches="tight", format="jpg")
        print(f"✅ Peta spasial disimpan ke: {save_path}")

    plt.show()
    return save_path