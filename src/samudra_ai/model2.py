from typing import Tuple, List, Optional
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, ConvLSTM2D, BatchNormalization, Conv2D, Lambda,
    SpatialDropout2D, SpatialDropout3D, TimeDistributed
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

class ModelConfig2:
    """Konfigurasi fleksibel untuk ConvLSTM2D Samudra-AI"""
    def __init__(
        self,
        conv_filters: List[int] = [16],         # filter untuk layer Conv2D awal
        lstm_filters: List[int] = [16, 32, 64, 128],  # filter tiap layer ConvLSTM2D
        kernel_size: Tuple[int, int] = (3, 3),
        activation: str = "relu",
        dropout_rates: List[float] = [0.1, 0.2, 0.3, 0.4],  # dropout sesuai layer ConvLSTM
        learning_rate: float = 1e-3,
        batch_size=8,
        epochs=100,      
    ):
        self.conv_filters = conv_filters
        self.lstm_filters = lstm_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.dropout_rates = dropout_rates
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs


def build_convlstm2d(
    input_shape: Tuple,
    output_shape: Tuple,
    config: Optional[ModelConfig2] = None
) -> Model:
    """Bangun arsitektur ConvLSTM2D sesuai baseline, tapi fleksibel."""
    if config is None:
        config = ModelConfig2()

    target_height, target_width = output_shape[0], output_shape[1]
    inputs = Input(shape=input_shape, name="input_layer")

    x = inputs

    # === Step 1: Conv2D awal ===
    for f in config.conv_filters:
        x = TimeDistributed(Conv2D(f, config.kernel_size, padding="same", activation=config.activation))(x)
        x = TimeDistributed(BatchNormalization())(x)

    # === Step 2: ConvLSTM stack ===
    for i, f in enumerate(config.lstm_filters):
        return_seq = i < len(config.lstm_filters) - 1
        x = ConvLSTM2D(f, config.kernel_size, padding="same",
                       return_sequences=return_seq,
                       activation=config.activation)(x)
        x = BatchNormalization()(x)

        if i < len(config.dropout_rates):
            if return_seq:
                x = TimeDistributed(SpatialDropout2D(config.dropout_rates[i]))(x)
            else:
                x = SpatialDropout2D(config.dropout_rates[i])(x)

    # === Resize ke target resolusi ===
    x = Lambda(lambda t: tf.image.resize(t, (target_height, target_width), method="bilinear"))(x)

    # === Conv2D final → output ===
    outputs = Conv2D(1, (1, 1), padding="same", activation="linear")(x)

    # === Compile ===
    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(config.learning_rate), loss="mse", metrics=["mae"])

    return model