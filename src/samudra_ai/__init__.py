# File: src/samudra_ai/__init__.py
from .core import SamudraAI, SamudraAI2
from .data_loader import load_and_mask_dataset
from .trainer import prepare_training_data, plot_training_history
from .evaluator import evaluate_model, plot_spatial_comparison, mask_land, mask_ocean
from .preprocess_dcpp import preprocess_dcpp

__version__ = "1.3.1"
__all__ = [
    'SamudraAI',
    'SamudraAI2',
    'load_and_mask_dataset',
    'prepare_training_data',
    'plot_training_history',
    'evaluate_model',
    'plot_spatial_comparison',
    "preprocess_dcpp",
    'mask_land',
    'mask_ocean',
]
