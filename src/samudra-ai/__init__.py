"""
SamudraAI: Paket Python untuk Koreksi Bias Model Iklim.

Paket ini menyediakan implementasi model deep learning CNN-BiLSTM untuk
melakukan koreksi bias pada data output dari General Circulation Models (GCMs)
berdasarkan data observasi atau reanalysis.
"""

# Membuat kelas SamudraAI tersedia saat user melakukan `import samudra_ai`
from .core import SamudraAI

# Mendefinisikan apa yang diimpor saat `from samudra_ai import *`
__all__ = ['SamudraAI']

# Informasi versi paket
__version__ = "0.1.0"
