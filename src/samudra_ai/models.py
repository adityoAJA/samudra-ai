# src/samudra_ai/models.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, TimeDistributed, Conv2D, BatchNormalization, LeakyReLU,
    Flatten, LSTM, Dense, Reshape, Dropout, Bidirectional
)
from typing import Tuple

def build_cnn_bilstm(input_shape: Tuple, output_shape: Tuple, lstm_units: int = 64) -> Model:
    """Membangun arsitektur model CNN-BiLSTM sesuai skrip referensi."""
    target_height, target_width = output_shape[0], output_shape[1]
    inputs = Input(shape=input_shape)

    x = TimeDistributed(Conv2D(16, (3, 3), padding='same', activation='linear'))(inputs)
    x = TimeDistributed(LeakyReLU(0.1))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Conv2D(32, (3, 3), padding='same', activation='linear'))(x)
    x = TimeDistributed(LeakyReLU(0.1))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.1))(x)

    x = TimeDistributed(Conv2D(64, (3, 3), padding='same', activation='linear'))(x)
    x = TimeDistributed(LeakyReLU(0.1))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Dropout(0.2))(x)

    x = TimeDistributed(Conv2D(128, (3, 3), padding='same', activation='linear'))(x)
    x = TimeDistributed(LeakyReLU(0.1))(x)
    x = TimeDistributed(BatchNormalization())(x)

    x = TimeDistributed(Flatten())(x)
    x = Dropout(0.2)(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    x = Bidirectional(LSTM(lstm_units, return_sequences=False))(x)

    x = Dense(target_height * target_width)(x)
    outputs = Reshape((target_height, target_width, 1))(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model
