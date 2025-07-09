# core.py

import numpy as np
import xarray as xr
import joblib
import json
import os
from tensorflow.keras.models import load_model as keras_load_model, Model
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

from .preprocessing import normalize_data, create_sequences
from .models import build_cnn_bilstm
from .training import set_seeds, get_default_callbacks
from .plotting import plot_history_metric
from .utils import NumpyEncoder

class SamudraAI:
    def __init__(self, time_seq: int = 9, lstm_units: int = 64, learning_rate: float = 1e-4):
        self.time_seq = time_seq
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        self.model: Model = None
        self.scaler_X = None
        self.scaler_y = None
        self.history = None
        self.obs_mask = None # Untuk menyimpan mask dari data observasi
        self.target_coords = None
        self.target_dims = None

    def fit(self, x_data_hist: xr.DataArray, y_data_obs: xr.DataArray, epochs: int = 100, batch_size: int = 8, validation_split: float = 0.2, seed: int = 42):
        set_seeds(seed)
        
        # Logika krusial dari skrip asli: menyamakan panjang data
        min_samples = min(x_data_hist.shape[0], y_data_obs.shape[0])
        x_np = np.nan_to_num(x_data_hist.isel(time=slice(0, min_samples)).values)
        y_np = np.nan_to_num(y_data_obs.isel(time=slice(0, min_samples)).values)
        
        # Simpan informasi koordinat & mask dari data observasi (resolusi tinggi)
        self.target_coords = y_data_obs.coords
        self.target_dims = y_data_obs.dims
        self.obs_mask = xr.where(np.isnan(y_data_obs.isel(time=0)), 0, 1)

        X_scaled, y_scaled, self.scaler_X, self.scaler_y = normalize_data(x_np, y_np)
        
        # Logika pembuatan sekuens dari skrip asli
        X_seq = np.array([X_scaled[i:i+self.time_seq] for i in range(len(X_scaled) - self.time_seq)])
        y_seq = np.array([y_scaled[i+self.time_seq-1] for i in range(len(y_scaled) - self.time_seq)])
        
        X_seq = X_seq[..., np.newaxis]
        y_seq = y_seq[..., np.newaxis]
        
        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=validation_split, random_state=seed, shuffle=False)
        
        input_shape = X_train.shape[1:]
        output_shape = y_train.shape[1:]
        
        self.model = build_cnn_bilstm(input_shape=input_shape, output_shape=output_shape, lstm_units=self.lstm_units)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])
        self.model.summary()
        
        callbacks = get_default_callbacks()
        
        print(f"🚀 Memulai training untuk {epochs} epochs...")
        self.history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks, verbose=1)
        
        loss, mae = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"✅ Training dan evaluasi selesai -> MSE: {loss:.4f}, MAE: {mae:.4f}")
        return self.history

    def predict(self, data_to_correct: xr.DataArray) -> xr.DataArray:
        if not self.model: raise RuntimeError("Model belum dilatih.")
        
        X_future = np.nan_to_num(data_to_correct.values)
        X_future_reshaped = X_future.reshape(X_future.shape[0], -1)
        X_future_scaled = self.scaler_X.transform(X_future_reshaped).reshape(X_future.shape)
        
        X_future_seq = np.array([X_future_scaled[i:i+self.time_seq] for i in range(len(X_future_scaled) - self.time_seq + 1)])
        X_future_seq = X_future_seq[..., np.newaxis]

        if X_future_seq.shape[0] == 0: return xr.DataArray()

        pred_scaled = self.model.predict(X_future_seq).squeeze()
        
        pred_flat = pred_scaled.reshape(pred_scaled.shape[0], -1)
        pred_original = self.scaler_y.inverse_transform(pred_flat)
        final_predictions = pred_original.reshape(pred_scaled.shape)
        
        valid_time = data_to_correct.time.values[self.time_seq - 1:]
        num_predictions = final_predictions.shape[0]

        corrected_da = xr.DataArray(
            final_predictions,
            coords={
                "time": valid_time[:num_predictions],
                self.target_dims[1]: self.target_coords[self.target_dims[1]],
                self.target_dims[2]: self.target_coords[self.target_dims[2]]
            },
            dims=self.target_dims
        )
        
        # Logika masking dari skrip asli
        mask_interp = self.obs_mask.interp_like(corrected_da, method="nearest")
        corrected_masked_da = corrected_da.where(mask_interp == 1)
        
        return corrected_masked_da

    def plot_loss(self, title="Kurva Loss", output_path=None):
        plot_history_metric(self.history, 'loss', title, 'Loss (MSE)', output_path)

    def plot_mae(self, title="Kurva MAE", output_path=None):
        plot_history_metric(self.history, 'mae', title, 'MAE', output_path)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.keras"))
        joblib.dump(self.scaler_X, os.path.join(path, "scaler_X.gz"))
        joblib.dump(self.scaler_y, os.path.join(path, "scaler_y.gz"))
        
        target_info = {
            'coords': {k: v.values.tolist() for k, v in self.target_coords.items() if k != 'time'},
            'dims': self.target_dims,
            'obs_mask': self.obs_mask.values.tolist()
        }
        config = {'time_seq': self.time_seq, 'lstm_units': self.lstm_units, 'learning_rate': self.learning_rate}

        with open(os.path.join(path, "config.json"), 'w') as f: json.dump(config, f, cls=NumpyEncoder)
        with open(os.path.join(path, "target_info.json"), 'w') as f: json.dump(target_info, f, cls=NumpyEncoder)
        print(f"✅ Model dan komponen berhasil disimpan di: {path}")

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "config.json"), 'r') as f: config = json.load(f)
        instance = cls(**config)
        
        with open(os.path.join(path, "target_info.json"), 'r') as f: target_info = json.load(f)
        instance.target_dims = tuple(target_info['dims'])
        instance.target_coords = {k: xr.DataArray(v, dims=target_info['coords'][k]['dims']) for k, v in target_info['coords'].items()}
        instance.obs_mask = xr.DataArray(target_info['obs_mask'], coords={'latitude': instance.target_coords['latitude'], 'longitude': instance.target_coords['longitude']}, dims=['latitude', 'longitude'])

        instance.model = keras_load_model(os.path.join(path, "model.keras"), custom_objects={"LeakyReLU": LeakyReLU})
        instance.scaler_X = joblib.load(os.path.join(path, "scaler_X.gz"))
        instance.scaler_y = joblib.load(os.path.join(path, "scaler_y.gz"))
        print(f"✅ Model dan komponen berhasil dimuat dari: {path}")
        return instance
