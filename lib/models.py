import datetime
from glob import glob
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model
import os

from lib.tfcustom import (
    KLDivergenceLayer,
    VariableRepeatVector,
    ResidualConv1D,
    mse_loss,
    kullback_leibler_loss,
    nll,
)


def create_vae():
    original_dim = 784
    intermediate_dim = 256
    latent_dim = 2
    epsilon_std = 1.0

    x = Input(shape=(original_dim,))
    h = Dense(intermediate_dim, activation="relu")(x)

    z_mu = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var])
    z_sigma = Lambda(lambda t: K.exp(0.5 * t))(z_log_var)

    eps = Input(
        tensor=K.random_normal(
            stddev=epsilon_std, shape=(K.shape(x)[0], latent_dim)
        )
    )
    z_eps = Multiply()([z_sigma, eps])
    z = Add()([z_mu, z_eps])

    d = Dense(intermediate_dim, input_dim=latent_dim, activation="relu")(z)
    x_pred = Dense(original_dim, activation="sigmoid")(d)

    vae = Model(inputs=[x, eps], outputs=x_pred)
    vae.compile(optimizer="rmsprop", loss=nll)

    return vae


def single_lstm_autoencoder(
    n_timesteps, n_features, latent_dim, activation=None
):
    """
    Parameters
    ----------
    n_features:
        Number of extracted_features in tom_data
    latent_dim:
        Latent dimension, i.e. how much it should be compressed
    n_timesteps:
        Number of timesteps in tom_data

    Returns
    -------
    encoder:
        Extract latent vector from tom_data
    decoder:
        Decode latent vector to tom_data
    autoencoder:
        Encode tom_data to latent vector and recreate from that
    """

    # ENCODER
    inputs = Input(shape=(n_timesteps, n_features))
    x = CuDNNLSTM(units=latent_dim, return_sequences=True, name="encoded",)(
        inputs
    )
    x = TimeDistributed(Dense(n_features))(x)
    x = Activation(None)(x)

    autoencoder = Model(inputs=inputs, outputs=x)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def lstm_autoencoder(n_timesteps, n_features, latent_dim, activation="elu"):
    def gelu(x):
        return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))

    if activation == "gelu":
        activation = gelu

    input = Input(shape=(n_timesteps, n_features))

    x = Bidirectional(CuDNNLSTM(latent_dim), name="encoded", merge_mode="mul")(
        input
    )

    x = Activation(activation)(x)

    x = VariableRepeatVector()([input, x])

    x = Bidirectional(
        CuDNNLSTM(latent_dim, return_sequences=True), merge_mode="mul"
    )(x)

    x = Activation(activation)(x)

    x = TimeDistributed(Dense(n_features))(x)
    x = Activation(None)(x)

    autoencoder = Model(inputs=input, outputs=x)
    autoencoder.compile(
        optimizer="adam", loss="mse", metrics=["mse"],
    )
    return autoencoder


def oneway_lstm_autoencoder(
    n_timesteps=None, n_features=2, latent_dim=64, activation="relu"
):
    input = Input(shape=(n_timesteps, n_features))

    x = CuDNNLSTM(latent_dim, name="encoded")(input)
    x = Activation("sigmoid")(x)

    x = VariableRepeatVector()([input, x])
    x = CuDNNLSTM(latent_dim, return_sequences=True)(x)

    x = Activation(activation)(x)

    x = TimeDistributed(Dense(n_features))(x)
    x = Activation(None)(x)

    autoencoder = Model(inputs=input, outputs=x)
    autoencoder.compile(
        optimizer="adam", loss="mse", metrics=["mse"],
    )
    return autoencoder


def conv_autoencoder(n_timesteps, n_features, latent_dim):
    """
    Parameters
    ----------
    n_features:
        Number of extracted_features in data
    latent_dim:
        Latent dimension, i.e. how much it should be compressed
    n_timesteps:
        Number of timesteps in tom_data

    Returns
    -------
    encoder:
        Extract latent vector from tom_data
    decoder:
        Decode latent vector to tom_data
    autoencoder:
        Encode tom_data to latent vector and recreate from that
    """
    p = {"padding": "same", "kernel_initializer": "he_uniform"}
    activation = "relu"

    # ENCODER
    inputs = Input((n_timesteps, n_features))
    mask = Masking(mask_value=0)(inputs)

    x = Conv1D(64, 1, **p)(mask)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = MaxPool1D(5, padding="same")(x)

    x = Conv1D(64, 3, **p)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = MaxPool1D(5, padding="same")(x)

    # Encoding and reshaping
    x = Flatten()(x)
    x = Dense(latent_dim, activation=None, name="encoded")(x)
    x = Dropout(0.2)(x)
    x = Reshape((latent_dim, 1))(x)
    x = Conv1D(1, 1, activation=None, padding="same")(x)

    # DECODER
    x = UpSampling1D(5)(x)

    x = Conv1D(64, 3, **p)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = UpSampling1D(3)(x)

    x = Conv1D(64, 3, **p)(x)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    # To make shape fit again
    outputs = Conv1D(n_features, 1, activation=None, **p)(x)

    # AUTOENCODER
    autoencoder = Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=Adam(lr=0.001), loss="mse")
    return autoencoder


def residual_conv_autoencoder(n_timesteps, n_features, latent_dim):
    """
    Autoencoder with residuals. Requires a large number of datapoints to be viable.
    Pooling layers fix the dimensionality

    Parameters
    ----------
    n_features:
        Number of extracted_features in tom_data
    latent_dim:
        Latent dimension, i.e. how much it should be compressed
    n_timesteps:
        Number of timesteps in tom_data

    Returns
    -------
    encoder:
        Extract latent vector from tom_data
    decoder:
        Decode latent vector to tom_data
    autoencoder:
        Encode tom_data to latent vector and recreate from that
    """
    p = {"padding": "same", "kernel_initializer": "he_uniform"}

    # Convolutions control (filters, size) if padding = "same"
    activation = "relu"
    latent_dim = 20

    # ENCODER
    inputs = Input((n_timesteps, n_features))
    mask = Masking(mask_value=0)(inputs)

    x = Conv1D(2, 1, **p)(mask)
    x = BatchNormalization()(x)
    x = Activation(activation)(x)

    x = ResidualConv1D(4, 31, activation, pool=True)(x)
    x = ResidualConv1D(4, 31, activation)(x)

    x = ResidualConv1D(8, 15, activation, pool=True)(x)
    x = ResidualConv1D(8, 15, activation)(x)

    x = ResidualConv1D(16, 7, activation, pool=True)(x)
    x = ResidualConv1D(16, 7, activation)(x)

    x = Flatten()(x)
    x = Dense(latent_dim, activation=None, name="encoded")(x)
    x = Dropout(0.2)(x)
    x = Reshape((latent_dim, 1))(x)
    x = Conv1D(1, 1, activation=None, padding="same")(x)

    # DECODER
    x = ResidualConv1D(16, 7, activation)(x)
    x = ResidualConv1D(16, 7, activation)(x)

    x = UpSampling1D(5)(x)

    x = ResidualConv1D(8, 15, activation, pool=True)(x)
    x = ResidualConv1D(8, 15, activation)(x)

    x = UpSampling1D(3)(x)

    x = ResidualConv1D(4, 31, activation, pool=True)(x)
    x = ResidualConv1D(4, 31, activation)(x)

    outputs = Conv1D(n_features, 1, activation=None, **p)(x)

    # AUTOENCODER
    autoencoder = Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer=Adam(lr=0.001), loss="mse")
    return autoencoder


def model_builder(
    model_build_f, build_args, patience=3, model_dir=None, chkpt_tag=None
):
    """Loads model and callbacks"""
    # set a directory in case None is set initially
    if chkpt_tag is None:
        chkpt_tag = ""

    _model_dir = Path(
        "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + chkpt_tag
    )

    initial_epoch = 0
    if model_dir is None:
        print("no model directory set. Creating new model.")
        model_dir = _model_dir
        model = model_build_f(*build_args)
    else:
        try:
            print("Loading model from specified directory")
            latest_ver = sorted(
                glob(Path(model_dir).joinpath("model_???")), reverse=True,
            )[0]
            initial_epoch = int(
                latest_ver[-3:]
            )  # get the last 3 values in dir name as epoch
            model = tf.keras.models.load_model(str(latest_ver))
        except IndexError:
            print("no model found. Creating new model.")
            model_dir = _model_dir
            model = model_build_f(*build_args)

    # callbacks
    es = EarlyStopping(patience=patience)
    rl = ReduceLROnPlateau(patience=5)
    tb = TensorBoard(log_dir=model_dir.as_posix())
    mca = ModelCheckpoint(
        filepath=model_dir.joinpath("model_{epoch:03d}.h5").as_posix(),
        save_best_only=True,
    )

    os.system("chmod -R 777 {}".format(model_dir))

    callbacks = [mca, tb, es, rl]
    return model, callbacks, initial_epoch


def lstm_vae(
    n_timesteps, n_features, intermediate_dim, kl_weight, eps=1, z_dim=2
):
    """
    Variational autoencoder for variable length time series. Cannot sample over
    the input space, because of variable-length time series compatibility
    """

    def _sample(args):
        """
        The sampling function to draw a latent vector from a normal distribution
        in z with a mu and a sigma
        """
        z_mean, z_log_var = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], z_dim), mean=0.0, stddev=eps
        )
        return z_mean + K.exp(z_log_var / 2) * epsilon

    inputs = Input(shape=(n_timesteps, n_features))
    # encode -> (latent_dim, )
    xe = Bidirectional(CuDNNLSTM(intermediate_dim, return_sequences=False))(
        inputs
    )

    xe = Activation("elu")(xe)

    # create latent n-dimensional (2D here) manifold
    z_mean = Dense(z_dim, name="z_mean")(xe)
    z_log_var = Dense(z_dim, name="z_var")(xe)

    # sample vector from the latent distribution
    z = Lambda(_sample, name="z_sample")([z_mean, z_log_var])

    # Repeat so it fits into LSTM
    xd = VariableRepeatVector()([inputs, z])
    xd = Bidirectional(
        CuDNNLSTM(intermediate_dim, return_sequences=True, name="decoder")
    )(xd)

    xd = Activation("elu")(xd)
    # Make sure the final activation is linear and correct dimensionality
    outputs = TimeDistributed(Dense(n_features, activation=None))(xd)

    # Start with 0 weight for the KL loss, and slowly increase with callback
    re_loss = mse_loss(inputs=inputs, outputs=outputs)
    kl_loss = kullback_leibler_loss(z_mean=z_mean, z_log_var=z_log_var)
    vae_loss = kl_weight * kl_loss + re_loss

    vae = Model(inputs, outputs)
    vae.add_loss(vae_loss)
    vae.compile(optimizer="adam")
    vae.summary()
    return vae
