import sys
import xarray as xr

# Impor komponen-komponen dari paket samudra_ai yang sudah diinstal
# Perhatikan kita tidak lagi mengimpor dari file lokal, tapi dari paket
from samudra_ai import SamudraAI
from samudra_ai.preprocessing import load_and_mask_dataset
from samudra_ai.utils import save_to_netcdf
from samudra_ai.plotting import plot_loss_curve, plot_mae_curve

# ===================================================================
# 1. KONFIGURASI
# Ganti nilai-nilai di bawah ini sesuai dengan data Anda.
# ===================================================================

# --- Path File Data ---
# Pastikan file-file ini ada di folder yang sama dengan skrip ini
FILE_GCM_HISTORICAL = "gcm_historical.nc"  # Ganti dengan nama file GCM historis Anda
FILE_OBSERVATION = "obs_reanalysis.nc"      # Ganti dengan nama file observasi Anda
FILE_GCM_PROJECTION = "gcm_proyeksi.nc"     # Ganti dengan nama file GCM proyeksi Anda

# --- Nama Variabel di dalam file NetCDF ---
VAR_NAME_GCM = "zos"  # Ganti dengan nama variabel di file GCM Anda (misal: 'tos', 'zos')
VAR_NAME_OBS = "zos"  # Ganti dengan nama variabel di file observasi Anda

# --- Rentang Spasial dan Temporal ---
# Rentang Lintang (Latitude)
LAT_RANGE = (-15, 10)
# Rentang Bujur (Longitude)
LON_RANGE = (90, 145)

# Rentang Waktu untuk Training
TIME_RANGE_HISTORICAL = ("1993-01-01", "2014-12-31")
TIME_RANGE_OBSERVATION = ("1993-01-01", "2014-12-31")

# Rentang Waktu untuk Prediksi/Koreksi
TIME_RANGE_PROJECTION = ("2025-01-01", "2100-12-31")

# --- Parameter Model ---
TIME_SEQUENCE = 9  # Ukuran jendela waktu untuk input LSTM
EPOCHS = 50        # Jumlah epoch untuk training (bisa disesuaikan)
BATCH_SIZE = 8     # Ukuran batch

# --- Path Output ---
MODEL_SAVE_PATH = "model_samudra_ai_final"
CORRECTED_OUTPUT_FILE = "hasil_koreksi/koreksi_proyeksi_final.nc"
PLOT_LOSS_FILE = "hasil_plot/kurva_loss.png"
PLOT_MAE_FILE = "hasil_plot/kurva_mae.png"

# ===================================================================
# 2. MEMUAT DAN MEMPERSIAPKAN DATA
# ===================================================================
print("🌊 [Langkah 1/5] Memuat dan memproses data...")

try:
    # Muat data training (input dan target)
    print(f"-> Memuat data GCM historis: {FILE_GCM_HISTORICAL}")
    gcm_hist_data = load_and_mask_dataset(
        file_path=FILE_GCM_HISTORICAL,
        var_name=VAR_NAME_GCM,
        lat_range=LAT_RANGE,
        lon_range=LON_RANGE,
        time_range=TIME_RANGE_HISTORICAL
    )

    print(f"-> Memuat data Observasi: {FILE_OBSERVATION}")
    obs_data = load_and_mask_dataset(
        file_path=FILE_OBSERVATION,
        var_name=VAR_NAME_OBS,
        lat_range=LAT_RANGE,
        lon_range=LON_RANGE,
        time_range=TIME_RANGE_OBSERVATION
    )
    
    # Muat data yang akan dikoreksi
    print(f"-> Memuat data GCM proyeksi: {FILE_GCM_PROJECTION}")
    gcm_proj_data = load_and_mask_dataset(
        file_path=FILE_GCM_PROJECTION,
        var_name=VAR_NAME_GCM,
        lat_range=LAT_RANGE,
        lon_range=LON_RANGE,
        time_range=TIME_RANGE_PROJECTION
    )
    
    print("✅ Data berhasil dimuat.")
    print(f"   - Ukuran data GCM historis: {gcm_hist_data.shape}")
    print(f"   - Ukuran data Observasi: {obs_data.shape}")
    print(f"   - Ukuran data GCM proyeksi: {gcm_proj_data.shape}")

except FileNotFoundError as e:
    print(f"❌ ERROR: File tidak ditemukan! {e}")
    sys.exit() # Hentikan skrip jika file tidak ada
except Exception as e:
    print(f"❌ ERROR saat memuat data: {e}")
    sys.exit()

# ===================================================================
# 3. TRAINING MODEL
# ===================================================================
print("\n🧠 [Langkah 2/5] Memulai training model SamudraAI...")

# Inisialisasi model dengan parameter yang sudah ditentukan
model = SamudraAI(time_seq=TIME_SEQUENCE)

# Latih model menggunakan data historis dan observasi
history = model.fit(
    x_data_hist=gcm_hist_data,
    y_data_obs=obs_data,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

print("✅ Training model selesai.")

# Visualisasikan hasil training
plot_loss_curve(history, title="Kurva Loss Training", output_path=PLOT_LOSS_FILE)
plot_mae_curve(history, title="Kurva MAE Training", output_path=PLOT_MAE_FILE)

# ===================================================================
# 4. MENYIMPAN MODEL
# ===================================================================
print(f"\n💾 [Langkah 3/5] Menyimpan model ke direktori: {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)

# ===================================================================
# 5. MEMUAT MODEL & PREDIKSI
# ===================================================================
print(f"\n🔄 [Langkah 4/5] Memuat model dari direktori: {MODEL_SAVE_PATH}...")

# Hapus model lama dari memori untuk simulasi penggunaan riil
del model 

# Muat model yang sudah disimpan
loaded_model = SamudraAI.load(MODEL_SAVE_PATH)
print("✅ Model berhasil dimuat kembali.")

print("\n🔬 Melakukan prediksi/koreksi pada data proyeksi...")
corrected_data = loaded_model.predict(gcm_proj_data)
print("✅ Prediksi selesai.")
print(f"   - Ukuran data hasil koreksi: {corrected_data.shape}")

# ===================================================================
# 6. MENYIMPAN HASIL AKHIR
# ===================================================================
print(f"\n📦 [Langkah 5/5] Menyimpan hasil koreksi ke file: {CORRECTED_OUTPUT_FILE}...")

# Siapkan atribut untuk file NetCDF
attributes = {
    'title': 'Hasil Koreksi Bias GCM menggunakan SamudraAI',
    'institution': 'Nama Institusi Anda',
    'source_model': FILE_GCM_PROJECTION,
    'correction_method': 'CNN-BiLSTM (SamudraAI v0.1.0)'
}

# Simpan hasil ke file NetCDF
save_to_netcdf(
    data_array=corrected_data,
    var_name=VAR_NAME_GCM,
    output_path=CORRECTED_OUTPUT_FILE,
    attributes=attributes
)

print("\n🎉🎉🎉 SELURUH PROSES SELESAI! 🎉🎉🎉")
