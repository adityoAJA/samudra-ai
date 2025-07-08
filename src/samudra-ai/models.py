# library

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    TimeDistributed,
    Conv2D,
    BatchNormalization,
    LeakyReLU,
    Flatten,
    LSTM,
    Dense,
    Reshape,
    Dropout,
    Bidirectional,
)
from typing import Tuple

def build_cnn_bilstm(
    input_shape: Tuple[int, int, int, int],
    output_shape: Tuple[int, int, int],
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
) -> Model:
    """
    Membangun arsitektur model CNN-BiLSTM untuk koreksi bias.

    Arsitektur ini terdiri dari beberapa layer CNN yang diaplikasikan
    pada setiap time step, diikuti oleh layer Bi-directional LSTM.

    Args:
        input_shape (Tuple): Bentuk data input (time_steps, height, width, channels).
        output_shape (Tuple): Bentuk data output yang diinginkan (height, width, channels).
        lstm_units (int): Jumlah unit dalam setiap layer LSTM. Default: 64.
        dropout_rate (float): Rate untuk layer Dropout. Default: 0.2.

    Returns:
        Model: Model Keras yang belum di-compile.
    """
    # Ekstrak dimensi target dari output_shape
    target_height, target_width, _ = output_shape
    
    inputs = Input(shape=input_shape)

    # === BLOK CNN (Time Distributed) ===
    # Layer 1
    x = TimeDistributed(Conv2D(16, (3, 3), padding="same", activation="linear"))(inputs)
    x = TimeDistributed(LeakyReLU(0.1))(x)
    x = TimeDistributed(BatchNormalization())(x)

    # Layer 2
    x = TimeDistributed(Conv2D(32, (3, 3), padding="same", activation="linear"))(x)
    x = TimeDistributed(LeakyReLU(0.1))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.1))(x) # Sesuai skrip asli

    # Layer 3
    x = TimeDistributed(Conv2D(64, (3, 3), padding="same", activation="linear"))(x)
    x = TimeDistributed(LeakyReLU(0.1))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x) # Sesuai skrip asli

    # Layer 4
    x = TimeDistributed(Conv2D(128, (3, 3), padding="same", activation="linear"))(x)
    x = TimeDistributed(LeakyReLU(0.1))(x)
    x = TimeDistributed(BatchNormalization())(x)

    # === BLOK RECURRENT (Bi-LSTM) ===
    x = TimeDistributed(Flatten())(x)
    x = Dropout(dropout_rate)(x)
    
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)

    # === BLOK OUTPUT ===
    x = Dense(target_height * target_width)(x)
    outputs = Reshape((target_height, target_width, 1))(x)

    # Buat model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Jangan compile model di sini.
    # Biarkan proses training yang menanganinya.
    
    return model
