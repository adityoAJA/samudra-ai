import numpy as np
import xarray as xr
import joblib
import json
import os
from tensorflow.keras.models import load_model as keras_load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Impor dari modul-modul kita sendiri
from .preprocessing import normalize_data, create_sequences
from .models import build_cnn_bilstm
from .training import set_seeds, get_default_callbacks, train_model
from .inference import perform_correction

class SamudraAI:
    """
    Kelas utama untuk melakukan koreksi bias model iklim menggunakan arsitektur CNN-BiLSTM.
    """
    def __init__(self, time_seq: int = 9, lstm_units: int = 64, learning_rate: float = 1e-4):
        """
        Inisialisasi model dengan hiperparameter.
        """
        self.time_seq = time_seq
        self.lstm_units = lstm_units
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        
    def fit(self, x_data_hist: xr.DataArray, y_data_obs: xr.DataArray, epochs: int = 100, batch_size: int = 8, validation_split: float = 0.2, seed: int = 42):
        """
        Melatih model koreksi bias.

        Args:
            x_data_hist (xr.DataArray): Data GCM (input).
            y_data_obs (xr.DataArray): Data observasi (target).
            epochs (int): Jumlah epoch training.
            batch_size (int): Ukuran batch.
            validation_split (float): Persentase data untuk validasi.
            seed (int): Random seed untuk reproducibility.
        """
        # 1. Atur reproducibility
        set_seeds(seed)
        
        # 2. Persiapan Data
        X_np = np.nan_to_num(x_data_hist.values)
        y_np = np.nan_to_num(y_data_obs.values)
        
        X_scaled, y_scaled, self.scaler_X, self.scaler_y = normalize_data(X_np, y_np)
        
        X_seq = create_sequences(X_scaled[..., np.newaxis], self.time_seq)
        y_seq = create_sequences(y_scaled[..., np.newaxis], self.time_seq)
        
        # Ambil hanya y pada time step terakhir untuk setiap sekuens
        y_seq = y_seq[:, -1, ...]
        
        X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=validation_split, random_state=seed, shuffle=False)
        
        # 3. Bangun dan Compile Model
        input_shape = X_train.shape[1:]
        output_shape = y_train.shape[1:]
        
        self.model = build_cnn_bilstm(input_shape=input_shape, output_shape=output_shape, lstm_units=self.lstm_units)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])
        self.model.summary()
        
        # 4. Training
        callbacks = get_default_callbacks()
        history = train_model(
            self.model, X_train, y_train, 
            validation_data=(X_val, y_val), 
            epochs=epochs, batch_size=batch_size, callbacks=callbacks
        )
        
        return history

    def predict(self, data_to_correct: xr.DataArray) -> xr.DataArray:
        """Menggunakan model untuk mengoreksi data baru."""
        if not all([self.model, self.scaler_X, self.scaler_y]):
            raise RuntimeError("Model belum dilatih atau di-load. Jalankan .fit() atau .load() terlebih dahulu.")
        
        return perform_correction(
            model=self.model, data_to_correct=data_to_correct,
            scaler_X=self.scaler_X, scaler_y=self.scaler_y,
            time_window=self.time_seq
        )

    def save(self, path: str):
        """Menyimpan model, scaler, dan konfigurasi ke sebuah direktori."""
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model.keras"))
        joblib.dump(self.scaler_X, os.path.join(path, "scaler_X.gz"))
        joblib.dump(self.scaler_y, os.path.join(path, "scaler_y.gz"))
        
        config = {'time_seq': self.time_seq, 'lstm_units': self.lstm_units, 'learning_rate': self.learning_rate}
        with open(os.path.join(path, "config.json"), 'w') as f:
            json.dump(config, f)
        print(f"✅ Model dan komponen berhasil disimpan di direktori: {path}")

    @classmethod
    def load(cls, path: str):
        """Memuat model, scaler, dan konfigurasi dari sebuah direktori."""
        with open(os.path.join(path, "config.json"), 'r') as f:
            config = json.load(f)
            
        instance = cls(**config)
        instance.model = keras_load_model(os.path.join(path, "model.keras"))
        instance.scaler_X = joblib.load(os.path.join(path, "scaler_X.gz"))
        instance.scaler_y = joblib.load(os.path.join(path, "scaler_y.gz"))
        print(f"✅ Model dan komponen berhasil dimuat dari direktori: {path}")
        return instance
