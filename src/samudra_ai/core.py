# File: src/samudra_ai/core.py
import os
import json
import joblib
import logging
import xarray as xr
import numpy as np
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .models import build_cnn_bilstm, ModelConfig
from .model2 import build_convlstm2d, ModelConfig2
from .utils import NumpyEncoder, save_to_netcdf
from .data_loader import load_and_mask_dataset
from .trainer import prepare_training_data, plot_training_history
from .evaluator import evaluate_model, plot_spatial_comparison

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SamudraAI:
    def __init__(self, time_seq: int = 9, lstm_units: int = 64, learning_rate: float = 1e-4, config: ModelConfig = None):
        self.time_seq = time_seq
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.config = config
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.history = None
        self.obs_for_metadata = None

    def fit(self, x_data_hist: xr.DataArray, y_data_obs: xr.DataArray, epochs: int = 100, batch_size: int = 8, test_size: float = 0.2, seed: int = 42):
        logger.info("🧠 Memulai training model CNN-BiLSTM...")
        self.obs_for_metadata = y_data_obs.copy(deep=True)

        X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_training_data(
            x_data_hist, y_data_obs, self.time_seq, seed
        )
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        input_shape = X_train.shape[1:]
        output_shape = y_train.shape[1:3]

        # ✅ pakai config custom kalau ada, kalau tidak build default
        if self.config is None:
            self.config = ModelConfig(rnn_units=self.lstm_units, learning_rate=self.learning_rate)

        self.model = build_cnn_bilstm(input_shape, output_shape, self.config)
        self.model.summary()

        callbacks = [EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(patience=5, verbose=1)]
        self.history = self.model.fit(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            validation_data=(X_test, y_test), callbacks=callbacks, verbose=1
        )

        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        logger.info(f"✅ Training selesai -> Final Val Loss (MSE): {loss:.4f}, Final Val MAE: {mae:.4f}")

    def correction(self, data_to_correct: xr.DataArray, save_path: str = None) -> xr.DataArray:
        if not all([self.model, self.scaler_X, self.scaler_y, self.obs_for_metadata is not None]):
            raise RuntimeError("Model belum dilatih. Jalankan .fit() terlebih dahulu.")

        logger.info(f"🔬 Mengoreksi data dengan shape: {data_to_correct.shape}...")

        X_future = data_to_correct.fillna(0).values
        X_future_scaled = self.scaler_X.transform(X_future.reshape(X_future.shape[0], -1)).reshape(X_future.shape)
        X_future_seq = np.array([X_future_scaled[i:i+self.time_seq] for i in range(len(X_future_scaled) - self.time_seq)])

        X_future_seq = X_future_seq[..., np.newaxis]  # ✅ bentuk ke 5D
        print("🧪 Input to model.predict:", X_future_seq.shape)
        print("🧪 Any NaN in input?:", np.isnan(X_future_seq).any())

        pred_scaled = self.model.predict(X_future_seq, batch_size=1).squeeze()

        flat_pred = pred_scaled.reshape(pred_scaled.shape[0], -1)

        # Ambil shape dari data observasi (tanpa waktu)
        obs_shape = self.obs_for_metadata.isel(time=0).shape
        expected_size = np.prod(obs_shape)

        if flat_pred.shape[1] != expected_size:
            raise ValueError(
                f"Mismatch: Hasil prediksi ({flat_pred.shape[1]}) tidak cocok dengan target obs_ref "
                f"({expected_size}) berdasarkan shape {obs_shape}"
            )

        pred_original = self.scaler_y.inverse_transform(flat_pred).reshape((pred_scaled.shape[0], *obs_shape))

        obs_mask = self.obs_for_metadata.isel(time=0).notnull()
        lat_dim_name = next((d for d in obs_mask.dims if 'lat' in d.lower()), 'lat')
        lon_dim_name = next((d for d in obs_mask.dims if 'lon' in d.lower()), 'lon')

        valid_time = data_to_correct.time.values[self.time_seq - 1 : self.time_seq - 1 + len(pred_original)]

        predicted_da = xr.DataArray(
            pred_original,
            coords={
                "time": valid_time,
                lat_dim_name: obs_mask.coords[lat_dim_name],
                lon_dim_name: obs_mask.coords[lon_dim_name]
            },
            dims=["time", lat_dim_name, lon_dim_name],
            name="corrected"
        )

        if predicted_da.shape[1:] == obs_mask.shape:
            predicted_masked = predicted_da.where(obs_mask, drop=False)
        else:
            logger.warning("Ukuran mask observasi tidak cocok. Koreksi akan disimpan tanpa masking akhir.")
            predicted_masked = predicted_da

        # Jika disediakan save_path, simpan otomatis
        if save_path:
            save_to_netcdf(predicted_masked, save_path)

        logger.info("✅ Koreksi selesai.")
        return predicted_masked

    def evaluate_and_plot(
        self,
        raw_gcm_data: xr.DataArray,
        ref_data: xr.DataArray,
        var_name_ref: str,
        output_dir: str = "hasil_evaluasi",
        save_corrected_path: str = None,
        mask_type: str = None):
        
        corrected_data = self.correction(raw_gcm_data)

        # Debug log hasil prediksi
        try:
            min_val = np.nanmin(corrected_data.values)
            max_val = np.nanmax(corrected_data.values)
            print(f"✅ Hasil koreksi -> min: {min_val:.4f}, max: {max_val:.4f}, shape: {corrected_data.shape}")
        except Exception as e:
            print("⚠️ Gagal menghitung min/max hasil koreksi:", e)

        # Auto-trim obs agar cocok dengan hasil koreksi
        if corrected_data.sizes['time'] != ref_data.sizes['time']:
            start_index = self.time_seq - 1
            ref_data = ref_data.isel(time=slice(start_index, start_index + corrected_data.sizes['time']))

        # Simpan hasil koreksi jika diminta
        if save_corrected_path:
            save_to_netcdf(corrected_data, save_corrected_path)

        # === Plot spasial perbandingan ===
        spatial_path = plot_spatial_comparison(
            corrected_da=corrected_data,
            raw_gcm_da=raw_gcm_data,
            ref_da=ref_data,
            var_name=var_name_ref,
            output_dir=output_dir,
            units=ref_data.attrs.get("units", ""),
            mask_type=mask_type 
        )
        logger.info(f"🌍 Peta spasial disimpan ke: {spatial_path}")

        return evaluate_model(ref_data, raw_gcm_data, corrected_data, var_name_ref, output_dir)

    def plot_history(self, output_dir: str = None):
        if not self.history:
            raise ValueError("Model belum dilatih.")
        plot_training_history(self.history, output_dir)

    def save(self, path: str):
        logger.info(f"💾 Menyimpan model ke direktori: {path}...")
        os.makedirs(path, exist_ok=True)

        # Simpan model dan scaler
        self.model.save(os.path.join(path, "model.keras"))
        joblib.dump(self.scaler_X, os.path.join(path, "scaler_X.gz"))
        joblib.dump(self.scaler_y, os.path.join(path, "scaler_y.gz"))

        # Simpan metadata observasi dengan backend yang lebih stabil
        try:
            self.obs_for_metadata.to_netcdf(
                os.path.join(path, "obs_metadata.nc"),
                engine="h5netcdf"
            )
        except Exception as e:
            logger.warning(f"❗ Gagal menyimpan obs_metadata.nc: {e}")

        # Simpan konfigurasi model
        config = {
            'time_seq': self.time_seq,
            'lstm_units': self.lstm_units,
            'learning_rate': self.learning_rate
        }
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(config, f, cls=NumpyEncoder)

        logger.info(f"✅ Model dan komponen berhasil disimpan.")

    @classmethod
    def load(cls, path: str):
        logger.info(f"🔄 Memuat model dari direktori: {path}...")
        required_files = ["config.json", "model.keras", "scaler_X.gz", "scaler_y.gz", "obs_metadata.nc"]
        for fname in required_files:
            full_path = os.path.join(path, fname)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File tidak ditemukan: {full_path}")

        with open(os.path.join(path, "config.json"), 'r') as f:
            config = json.load(f)
        instance = cls(**config)
        instance.model = keras_load_model(
            os.path.join(path, "model.keras"),
            custom_objects={"LeakyReLU": LeakyReLU},
            compile=False)
        instance.scaler_X = joblib.load(os.path.join(path, "scaler_X.gz"))
        instance.scaler_y = joblib.load(os.path.join(path, "scaler_y.gz"))
        instance.obs_for_metadata = xr.open_dataarray(os.path.join(path, "obs_metadata.nc"))
        instance.model.compile(
            optimizer=Adam(learning_rate=instance.learning_rate),
            loss='mse',
            metrics=['mae'])
        logger.info(f"✅ Model dan komponen berhasil dimuat.")
        return instance
    
class SamudraAI2:
    def __init__(self, time_seq: int = 9, config: ModelConfig2 = None):
        self.time_seq = time_seq
        self.config = config
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.history = None
        self.obs_for_metadata = None

    def fit(self, x_data_hist, y_data_obs, epochs: int = None, batch_size: int = None, test_size: float = 0.2, seed: int = 42):
        logger.info("🧠 Memulai training model ConvLSTM2D...")
        self.obs_for_metadata = y_data_obs.copy(deep=True)

        X_train, X_test, y_train, y_test, scaler_X, scaler_y = prepare_training_data(
            x_data_hist, y_data_obs, self.time_seq, seed
        )
        self.scaler_X, self.scaler_y = scaler_X, scaler_y

        input_shape = X_train.shape[1:]    # (time_seq, lat, lon, 1)
        output_shape = y_train.shape[1:3]  # (lat, lon)

        if self.config is None:
            self.config = ModelConfig2()

        self.model = build_convlstm2d(input_shape, output_shape, self.config)
        self.model.summary()

        callbacks = [EarlyStopping(patience=10, restore_best_weights=True), ReduceLROnPlateau(patience=5, verbose=1)]
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs or self.config.epochs,
            batch_size=batch_size or self.config.batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks, verbose=1
        )

    def correction(self, data_to_correct: xr.DataArray, save_path: str = None) -> xr.DataArray:
        if not all([self.model, self.scaler_X, self.scaler_y, self.obs_for_metadata is not None]):
            raise RuntimeError("Model belum dilatih. Jalankan .fit() terlebih dahulu.")

        logger.info(f"🔬 Mengoreksi data dengan shape: {data_to_correct.shape}...")

        X_future = data_to_correct.fillna(0).values
        X_future_scaled = self.scaler_X.transform(X_future.reshape(X_future.shape[0], -1)).reshape(X_future.shape)
        X_future_seq = np.array([X_future_scaled[i:i+self.time_seq] for i in range(len(X_future_scaled) - self.time_seq)])

        X_future_seq = X_future_seq[..., np.newaxis]  # ✅ bentuk ke 5D
        print("🧪 Input to model.predict:", X_future_seq.shape)
        print("🧪 Any NaN in input?:", np.isnan(X_future_seq).any())

        pred_scaled = self.model.predict(X_future_seq, batch_size=1).squeeze()

        flat_pred = pred_scaled.reshape(pred_scaled.shape[0], -1)

        # Ambil shape dari data observasi (tanpa waktu)
        obs_shape = self.obs_for_metadata.isel(time=0).shape
        expected_size = np.prod(obs_shape)

        if flat_pred.shape[1] != expected_size:
            raise ValueError(
                f"Mismatch: Hasil prediksi ({flat_pred.shape[1]}) tidak cocok dengan target obs_ref "
                f"({expected_size}) berdasarkan shape {obs_shape}"
            )

        pred_original = self.scaler_y.inverse_transform(flat_pred).reshape((pred_scaled.shape[0], *obs_shape))

        obs_mask = self.obs_for_metadata.isel(time=0).notnull()
        lat_dim_name = next((d for d in obs_mask.dims if 'lat' in d.lower()), 'lat')
        lon_dim_name = next((d for d in obs_mask.dims if 'lon' in d.lower()), 'lon')

        valid_time = data_to_correct.time.values[self.time_seq - 1 : self.time_seq - 1 + len(pred_original)]

        predicted_da = xr.DataArray(
            pred_original,
            coords={
                "time": valid_time,
                lat_dim_name: obs_mask.coords[lat_dim_name],
                lon_dim_name: obs_mask.coords[lon_dim_name]
            },
            dims=["time", lat_dim_name, lon_dim_name],
            name="corrected"
        )

        if predicted_da.shape[1:] == obs_mask.shape:
            predicted_masked = predicted_da.where(obs_mask, drop=False)
        else:
            logger.warning("Ukuran mask observasi tidak cocok. Koreksi akan disimpan tanpa masking akhir.")
            predicted_masked = predicted_da

        # Jika disediakan save_path, simpan otomatis
        if save_path:
            save_to_netcdf(predicted_masked, save_path)

        logger.info("✅ Koreksi selesai.")
        return predicted_masked

    def evaluate_and_plot(
        self,
        raw_gcm_data: xr.DataArray,
        ref_data: xr.DataArray,
        var_name_ref: str,
        output_dir: str = "hasil_evaluasi",
        save_corrected_path: str = None,
        mask_type: str = None):

        logger.info("📊 Evaluasi model dengan data referensi...")
        
        corrected_data = self.correction(raw_gcm_data)

        # Debug log hasil prediksi
        try:
            min_val = np.nanmin(corrected_data.values)
            max_val = np.nanmax(corrected_data.values)
            print(f"✅ Hasil koreksi -> min: {min_val:.4f}, max: {max_val:.4f}, shape: {corrected_data.shape}")
        except Exception as e:
            print("⚠️ Gagal menghitung min/max hasil koreksi:", e)

        # Auto-trim obs agar cocok dengan hasil koreksi
        if corrected_data.sizes['time'] != ref_data.sizes['time']:
            start_index = self.time_seq - 1
            ref_data = ref_data.isel(time=slice(start_index, start_index + corrected_data.sizes['time']))

        # Simpan hasil koreksi jika diminta
        if save_corrected_path:
            save_to_netcdf(corrected_data, save_corrected_path)

        # === Plot spasial perbandingan ===
        spatial_path = plot_spatial_comparison(
            corrected_da=corrected_data,
            raw_gcm_da=raw_gcm_data,
            ref_da=ref_data,
            var_name=var_name_ref,
            output_dir=output_dir,
            units=ref_data.attrs.get("units", ""),
            mask_type=mask_type 
        )
        logger.info(f"🌍 Peta spasial disimpan ke: {spatial_path}")

        return evaluate_model(ref_data, raw_gcm_data, corrected_data, var_name_ref, output_dir)

    def plot_history(self, output_dir: str = None):
        if not self.history:
            raise ValueError("Model belum dilatih.")
        plot_training_history(self.history, output_dir)

    def save(self, path: str):
        logger.info(f"💾 Menyimpan model ke direktori: {path}...")
        os.makedirs(path, exist_ok=True)

        # Simpan model dan scaler
        self.model.save(os.path.join(path, "model2.keras"))
        joblib.dump(self.scaler_X, os.path.join(path, "scaler_X2.gz"))
        joblib.dump(self.scaler_y, os.path.join(path, "scaler_y2.gz"))

        # Simpan metadata observasi dengan backend yang lebih stabil
        try:
            self.obs_for_metadata.to_netcdf(
                os.path.join(path, "obs_metadata2.nc"),
                engine="h5netcdf"
            )
        except Exception as e:
            logger.warning(f"❗ Gagal menyimpan obs_metadata.nc: {e}")

        # Simpan konfigurasi model
        config = {
            'time_seq': self.time_seq,
            'lstm_units': self.lstm_units,
            'learning_rate': self.learning_rate
        }
        with open(os.path.join(path, "config2.json"), 'w') as f:
            json.dump(config, f, cls=NumpyEncoder)

        logger.info(f"✅ Model dan komponen berhasil disimpan.")

    @classmethod
    def load(cls, path: str):
        logger.info(f"🔄 Memuat model dari direktori: {path}...")
        required_files = ["config2.json", "model2.keras", "scaler_X2.gz", "scaler_y2.gz", "obs_metadata2.nc"]
        for fname in required_files:
            full_path = os.path.join(path, fname)
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"File tidak ditemukan: {full_path}")

        with open(os.path.join(path, "config.json"), 'r') as f:
            config = json.load(f)
        instance = cls(**config)
        instance.model = keras_load_model(
            os.path.join(path, "model.keras"),
            custom_objects={"LeakyReLU": LeakyReLU},
            compile=False)
        instance.scaler_X = joblib.load(os.path.join(path, "scaler_X.gz"))
        instance.scaler_y = joblib.load(os.path.join(path, "scaler_y.gz"))
        instance.obs_for_metadata = xr.open_dataarray(os.path.join(path, "obs_metadata.nc"))
        instance.model.compile(
            optimizer=Adam(learning_rate=instance.learning_rate),
            loss='mse',
            metrics=['mae'])
        logger.info(f"✅ Model CONVLSTM2D dan komponen berhasil dimuat.")
        return instance