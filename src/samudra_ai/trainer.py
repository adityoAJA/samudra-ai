# File: src/samudra_ai/trainer.py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import os

def prepare_training_data(x_data, y_data, time_seq, seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    min_samples = min(x_data.shape[0], y_data.shape[0])
    x_np = np.nan_to_num(x_data.isel(time=slice(0, min_samples)).values)
    y_np = np.nan_to_num(y_data.isel(time=slice(0, min_samples)).values)

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(x_np.reshape(x_np.shape[0], -1)).reshape(x_np.shape)
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_scaled = scaler_y.fit_transform(y_np.reshape(y_np.shape[0], -1)).reshape(y_np.shape)

    X_seq = np.array([X_scaled[i:i+time_seq] for i in range(len(X_scaled) - time_seq)])
    y_seq = np.array([y_scaled[i+time_seq-1] for i in range(len(y_scaled) - time_seq)])

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq[..., np.newaxis], y_seq, test_size=0.2, random_state=seed, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler_X, scaler_y

def plot_training_history(history, output_dir=None):
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Kurva Loss Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "loss_curve.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title("Kurva MAE Training")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.legend()
    if output_dir:
        plt.savefig(os.path.join(output_dir, "mae_curve.png"))
    plt.show()