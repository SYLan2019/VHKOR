import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from lib.model import CNNmulti
from tensorflow.keras.models import Model
from lib.data.Dataloader import X1_train, X2_train, X3_train


# position embedding
def positional_embedding(maxlen, model_size):
    PE = np.zeros((maxlen, model_size))
    for i in range(maxlen):
        for j in range(model_size):
            if j % 2 == 0:
                PE[i, j] = np.sin(i / 10000 ** (j / model_size))
            else:
                PE[i, j] = np.cos(i / 10000 ** ((j-1) / model_size))
    PE = tf.constant(PE, dtype=tf.float32)
    return PE


def transformer_cross(
    inputs1, inputs2,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    dropout: float = 0,
    kernel_size: int = 1,
):
    #pos = positional_embedding(maxlen=128, model_size=1)
    #inputs = pos + inputs
    """Encoder: Attention and Normalization and Feed-Forward."""
    # 1. Attention and Normalization:
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs1, inputs2)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs1

    # 2. Feed Forward Part:
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs1.shape[-1], kernel_size=kernel_size)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


def transformer_self(
    inputs,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    dropout: float = 0,
    kernel_size: int = 1,
):
    #pos = positional_embedding(maxlen=128, model_size=1)
    #inputs = pos + inputs
    """Encoder: Attention and Normalization and Feed-Forward."""
    # 1. Attention and Normalization:
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # 2. Feed Forward Part:
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=kernel_size)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)

    return x + res


def encoder(
    inputs,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    num_layers: int,
    dropout: float = 0,
    kernel_size: int = 1,
):
    x = inputs
    for i in range(num_layers):
        x = transformer_self(x, head_size, num_heads, ff_dim, dropout, kernel_size)

    x = layers.Flatten(input_shape=(128, 1))(x)
    output = keras.layers.Dense(63, activation='softmax')(x)
    return output


def encoder_cross(
    inputs1, inputs2,
    head_size: int,
    num_heads: int,
    num_layers: int,
    ff_dim: int,
    dropout: float = 0,
    kernel_size: int = 1,
):
    x = inputs2
    if num_layers == 1:
        output = transformer_cross(inputs1, inputs2, head_size, num_heads, ff_dim, dropout, kernel_size)
    else:
        for i in range(num_layers):
            x = transformer_cross(inputs1, x, head_size, num_heads, ff_dim, dropout, kernel_size)
        output = x
    return output


class CustLayer(layers.Layer):
    def __init__(self, **kwargs):
        super(CustLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='weights', shape=(len(input_shape), 1), initializer='uniform', trainable=True)

    def call(self, inputs, **kwargs):
        return keras.backend.dot(keras.backend.stack(inputs, axis=-1), self.w)[..., -1]


def build_T_cross():
    x1 = keras.layers.Input(X1_train.shape[1:], name='Input_1')
    x2 = keras.layers.Input(X2_train.shape[1:], name='Input_2')
    x3 = keras.layers.Input(X3_train.shape[1:], name='Input_3')
    in1 = x1
    in2 = x2
    in3 = x3
    for i in range(1, len(CNNmulti.CNN_H().layers)):
        x1 = CNNmulti.CNN_H().layers[i](x1)
    for j in range(1, len(CNNmulti.CNN_K().layers)):
        x2 = CNNmulti.CNN_K().layers[j](x2)
    for k in range(1, len(CNNmulti.CNN_V().layers)):
        x3 = CNNmulti.CNN_V().layers[k](x3)
    # x1 = transformer_self(x1, head_size=1, num_heads=2, ff_dim=128)
    # x2 = transformer_self(x2, head_size=1, num_heads=2, ff_dim=128)
    # x3 = transformer_self(x3, head_size=1, num_heads=2, ff_dim=128)
    merger = CustLayer()([x1, x2, x3])
    # merger = layers.Dense(1, activation='relu')(merger)
    cross1 = encoder_cross(merger, x1, head_size=1, num_heads=3, num_layers=4, ff_dim=128)
    cross2 = encoder_cross(merger, x2, head_size=1, num_heads=3, num_layers=4, ff_dim=128)
    cross3 = encoder_cross(merger, x3, head_size=1, num_heads=3, num_layers=4, ff_dim=128)
    self1 = encoder(cross1, head_size=1, num_heads=3, ff_dim=128, num_layers=4)
    self2 = encoder(cross2, head_size=1, num_heads=3, ff_dim=128, num_layers=4)
    self3 = encoder(cross3, head_size=1, num_heads=3, ff_dim=128, num_layers=4)
    full = layers.concatenate([self1, self2, self3])

    output = layers.Dense(63, activation='softmax')(full)
    model = Model(inputs=[in1, in2, in3], outputs=output)
    return model


