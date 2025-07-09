# training.py

import tensorflow as tf
import numpy as np
import random
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, History, Callback
from typing import List

def set_seeds(seed_value: int = 42):
    """Mengatur random seeds untuk reproducibility."""
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    print(f"🌱 Seeds set to {seed_value} for reproducible results.")

def get_default_callbacks() -> List[Callback]:
    """Mengembalikan daftar callbacks standar untuk training."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
    return [early_stopping, reduce_lr]
