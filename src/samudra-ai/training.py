# library

import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, History, Callback
from tensorflow.keras.models import Model
from typing import List, Tuple

def set_seeds(seed_value: int = 42):
    """
    Mengatur random seeds untuk reproducibility.

    Args:
        seed_value (int): Nilai seed yang akan digunakan.
    """
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print(f"🌱 Seeds set to {seed_value} for reproducible results.")

def get_default_callbacks(
    early_stopping_patience: int = 10,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.5,
) -> List[Callback]:
    """
    Membuat dan mengembalikan daftar callbacks standar untuk training.

    Args:
        early_stopping_patience (int): Jumlah epoch tanpa peningkatan sebelum training berhenti.
        reduce_lr_patience (int): Jumlah epoch tanpa peningkatan sebelum learning rate dikurangi.
        reduce_lr_factor (float): Faktor pengurangan learning rate.

    Returns:
        List[Callback]: Daftar objek Keras Callback.
    """
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=early_stopping_patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=reduce_lr_factor,
        patience=reduce_lr_patience,
        verbose=1,
    )
    
    return [early_stopping, reduce_lr]

def train_model(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    validation_data: Tuple[np.ndarray, np.ndarray],
    epochs: int,
    batch_size: int,
    callbacks: List[Callback],
) -> History:
    """
    Menjalankan proses training model.

    Args:
        model (Model): Model Keras yang sudah di-compile.
        X_train (np.ndarray): Data training input.
        y_train (np.ndarray): Data training target.
        validation_data (Tuple): Data validasi (X_val, y_val).
        epochs (int): Jumlah epoch training.
        batch_size (int): Ukuran batch.
        callbacks (List[Callback]): Daftar callback yang akan digunakan.

    Returns:
        History: Objek history dari hasil training.
    """
    print(f"🚀 Starting model training for {epochs} epochs...")
    
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
    )
    
    print("✅ Model training finished.")
    return history
