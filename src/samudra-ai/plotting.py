# library

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
from typing import Optional

def plot_loss_curve(
    history: History,
    title: str = "Model Training vs. Validation Loss",
    output_path: Optional[str] = None,
):
    """
    Membuat dan menampilkan plot kurva loss dari history training.

    Args:
        history (History): Objek history yang dikembalikan oleh model.fit().
        title (str): Judul untuk plot.
        output_path (Optional[str]): Path untuk menyimpan gambar. 
                                     Jika None, gambar hanya ditampilkan.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["loss"], label="Training Loss")
        plt.plot(history.history["val_loss"], label="Validation Loss")
        plt.title(title, fontsize=16)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss (MSE)", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"📉 Plot loss disimpan ke: {output_path}")

        plt.show()

    except KeyError as e:
        print(f"Error: Kunci {e} tidak ditemukan dalam history. Pastikan 'loss' dan 'val_loss' ada.")


def plot_mae_curve(
    history: History,
    title: str = "Model Training vs. Validation MAE",
    output_path: Optional[str] = None,
):
    """
    Membuat dan menampilkan plot kurva MAE dari history training.

    Args:
        history (History): Objek history yang dikembalikan oleh model.fit().
        title (str): Judul untuk plot.
        output_path (Optional[str]): Path untuk menyimpan gambar.
                                     Jika None, gambar hanya ditampilkan.
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history["mae"], label="Training MAE")
        plt.plot(history.history["val_mae"], label="Validation MAE")
        plt.title(title, fontsize=16)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Mean Absolute Error (MAE)", fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=300)
            print(f"📏 Plot MAE disimpan ke: {output_path}")

        plt.show()

    except KeyError as e:
        print(f"Error: Kunci {e} tidak ditemukan dalam history. Pastikan 'mae' dan 'val_mae' ada.")
