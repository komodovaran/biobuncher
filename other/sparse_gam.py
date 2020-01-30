"""Keras implementation of the k-sparse autoencoder.
"""
from tensorflow.python.keras.layers import (
    Input,
    Dense,
    Flatten,
    Reshape,
    Conv2DTranspose,
    Conv2D,
    Activation,
)
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from lib.tfcustom import KSparse, UpdateSparsityLevel, calculate_sparsity_levels
import numpy as np
import matplotlib.pyplot as plt
from st_predict import _pca
import tensorflow as tf
import tensorflow.keras.backend as K

if __name__ == "__main__":
    """Example of how to use the k-sparse autoencoder to learn sparse features of MNIST digits."""
    # Process MNIST
    X = np.load("../data/preprocessed/fake_tracks_type_3_img.npz")["data"]

    x_train, x_val = train_test_split(X)

    input_shape = x_train.shape[1:]

    # Define autoencoder parameters
    EPOCHS = 10
    BATCH_SIZE = 128
    KERNEL_SIZE = 3
    LATENT_DIM = 16
    LAYER_FILTERS = [32, 64]

    INIT_SPARSITY = 128
    END_SPARSITY = 15
    sparsity_levels = calculate_sparsity_levels(
        INIT_SPARSITY, END_SPARSITY, EPOCHS
    )

    # k_sparse = KSparse(sparsity_levels = sparsity_levels, name = 'KSparse')(x)

    # Build the Autoencoder Model
    inputs = Input(shape=input_shape, name="encoder_input")
    x = inputs

    for filters in LAYER_FILTERS:
        x = Conv2D(
            filters=filters,
            kernel_size=KERNEL_SIZE,
            strides=2,
            activation="relu",
            padding="same",
            data_format="channels_last",
        )(x)

    # Generate the latent vector
    x = Flatten()(x)
    x = Dense(LATENT_DIM, name="latent_vector")(x)
    k_sparse = KSparse(sparsity_levels = sparsity_levels, name = 'KSparse')(x)

    # Decoder
    x = Dense(input_shape[0] * input_shape[1] * input_shape[2])(k_sparse)
    x = Reshape(input_shape)(x)

    for filters in LAYER_FILTERS[::-1]:
        x = Conv2DTranspose(
            filters=filters,
            kernel_size=KERNEL_SIZE,
            strides=1,
            activation="relu",
            padding="same",
            data_format="channels_last",
        )(x)

    x = Conv2DTranspose(
        filters=input_shape[-1],
        kernel_size=KERNEL_SIZE,
        padding="same",
        data_format="channels_last",
    )(x)

    outputs = Activation(None, name="decoder_output")(x)

    # Autoencoder = Encoder + Decoder
    autoencoder = Model(inputs, outputs, name="autoencoder")
    autoencoder.summary()

    autoencoder.compile(loss="mse", optimizer="adam")

    # Train the autoencoder
    autoencoder.fit(
        x_train,
        x_train,
        validation_data=(x_val, x_val),
        epochs=100,
        batch_size=BATCH_SIZE,
        callbacks = [UpdateSparsityLevel()]
    )
# pca, _ = _pca(encoded, embed_into_n_components = 2)
#
# plt.scatter(pca[:,0], pca[:, 1])
# plt.show()
