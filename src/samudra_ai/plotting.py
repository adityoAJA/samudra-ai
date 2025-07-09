# plotting.py

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import History
from typing import Optional
import os

def plot_history_metric(history: History, metric: str, title: str, y_label: str, output_path: Optional[str] = None):
    """Fungsi generik untuk mem-plot metrik training dan validasi."""
    try:
        plt.figure(figsize=(10, 6))
        plt.plot(history.history[metric], label=f"Training {y_label}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {y_label}")
        plt.title(title, fontsize=16)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if output_path:
            # ================================================= #
            # PERBAIKAN DI SINI: Buat direktori jika belum ada
            # ================================================= #
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            # ================================================= #

            plt.savefig(output_path, dpi=300)
            print(f"📈 Plot disimpan ke: {output_path}")

        plt.show()

    except KeyError as e:
        print(f"Error: Kunci {e} tidak ditemukan dalam history. Pastikan metrik training benar.")
