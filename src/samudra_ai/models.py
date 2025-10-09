# # File: src/samudra_ai/models.py
from typing import Tuple, List, Optional
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv2D, BatchNormalization,
    Dropout, Flatten, Bidirectional, LSTM, Dense, Reshape, LeakyReLU
)
from tensorflow.keras.optimizers import Adam


class ModelConfig:
    """Konfigurasi fleksibel untuk CNN-BiLSTM Samudra-AI"""
    def __init__(
        self,
        filters: List[int] = [16, 32, 64, 128],
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = "linear",
        rnn_units: int = 64,
        dropout_rates: List[float] = [0.0, 0.1, 0.2, 0.0],
        leaky_relu_alpha: float = 0.1,
        batch_size: int = 32,
        epochs: int = 50,
        learning_rate: float = 1e-4
    ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.rnn_units = rnn_units
        self.dropout_rates = dropout_rates
        self.leaky_relu_alpha = leaky_relu_alpha
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate


def build_cnn_bilstm(
    input_shape: Tuple,
    output_shape: Tuple,
    config: Optional[ModelConfig] = None
) -> Model:
    """
    Bangun arsitektur CNN-BiLSTM sesuai baseline,
    tapi fleksibel untuk variabel harian/bulanan dan berbagai parameter.
    """
    if config is None:
        config = ModelConfig()  # pakai default baseline

    target_height, target_width = output_shape[0], output_shape[1]
    inputs = Input(shape=input_shape, name="input_layer")
    x = inputs

    # === TimeDistributed CNN layers ===
    for i, f in enumerate(config.filters):
        x = TimeDistributed(
            Conv2D(f, config.kernel_size, padding="same", activation=config.activation)
        )(x)
        x = TimeDistributed(LeakyReLU(config.leaky_relu_alpha))(x)
        x = TimeDistributed(BatchNormalization())(x)

        # Tambahkan dropout sesuai urutan kalau tersedia
        if i < len(config.dropout_rates) and config.dropout_rates[i] > 0:
            x = TimeDistributed(Dropout(config.dropout_rates[i]))(x)

    # === Flatten dan Dropout ===
    x = TimeDistributed(Flatten())(x)
    x = Dropout(config.dropout_rates[-1] if len(config.dropout_rates) > 0 else 0.2)(x)

    # === BiLSTM layers ===
    x = Bidirectional(LSTM(config.rnn_units, return_sequences=True))(x)
    x = Bidirectional(LSTM(config.rnn_units, return_sequences=False))(x)

    # === Dense ke output ===
    x = Dense(target_height * target_width)(x)
    outputs = Reshape((target_height, target_width, 1))(x)

    # === Compile model ===
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss="mse",
        metrics=["mae"]
    )

    return model
