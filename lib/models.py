import datetime
import os
from glob import glob
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    CuDNNLSTM,
    Dense,
    Dropout,
    Flatten,
    Input,
    Lambda,
    Masking,
    MaxPool1D,
    Reshape,
    TimeDistributed,
    UpSampling1D,
)
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import L1L2

import tensorflow_probability as tfp

# define Tensorflow probability distributions
Bernoulli = tfp.distributions.Bernoulli
OneHotCategorical = tfp.distributions.OneHotCategorical
RelaxedOneHotCategorical = tfp.distributions.RelaxedOneHotCategorical
KL = tfp.distributions.kl_divergence

from lib.tfcustom import (
    KLDivergenceLayer,
    ResidualConv1D,
    VariableRepeatVector,
    gelu,
)


def lstm_autoencoder(n_timesteps, n_features, latent_dim, activation="elu"):
    """
    LSTM autoencoder with bidirectionality
    """
    if activation == "gelu":
        activation = gelu

    i = Input(shape=(n_timesteps, n_features))

    x = Bidirectional(CuDNNLSTM(latent_dim), merge_mode="mul")(i)
    x = Activation(activation, name="encoded")(x)

    x = VariableRepeatVector()([i, x])
    x = Bidirectional(
        CuDNNLSTM(latent_dim, return_sequences=True), merge_mode="mul"
    )(x)
    x = Activation(activation)(x)
    x = TimeDistributed(Dense(n_features))(x)
    x = Activation(None)(x)

    autoencoder = Model(inputs=i, outputs=x)
    autoencoder.compile(
        optimizer="adam", loss="mse", metrics=["mse"],
    )
    return autoencoder


def lstm_autoencoder_zdim(
    n_timesteps, n_features, latent_dim, activation="elu", zdim=8
):
    """
    LSTM autoencoder, but with a Dense unit to downsample the feature vector.
    The LSTM units keep their unit size.
    """
    if activation == "gelu":
        activation = gelu

    i = Input(shape=(n_timesteps, n_features))
    x = Bidirectional(CuDNNLSTM(latent_dim), merge_mode="mul")(i)
    x = Activation(activation)(x)
    x = Dense(zdim, name="encoded")(x)
    x = Activation(activation)(x)
    x = VariableRepeatVector()([i, x])
    x = Bidirectional(
        CuDNNLSTM(latent_dim, return_sequences=True), merge_mode="mul"
    )(x)
    x = Activation(activation)(x)
    x = TimeDistributed(Dense(n_features))(x)
    x = Activation(None)(x)

    autoencoder = Model(inputs=i, outputs=x)
    autoencoder.compile(
        optimizer="adam", loss="mse", metrics=["mse"],
    )
    return autoencoder


def lstm_autoencoder_sparse(
    n_timesteps, n_features, latent_dim, activation="elu", l1_reg=0.00,
):
    """
    LSTM autoencoder with sparsity constraints to avoid learning redundancies
    between different samples
    """
    if activation == "gelu":
        activation = gelu

    i = Input(shape=(n_timesteps, n_features))
    x = Bidirectional(
        CuDNNLSTM(latent_dim, kernel_regularizer=L1L2(l1=l1_reg)),
        merge_mode="mul",
        name="encoded",
    )(i)

    x = Dense(latent_dim)(x)
    x = Activation("softmax")(x)
    x = VariableRepeatVector()([i, x])

    x = Bidirectional(
        CuDNNLSTM(latent_dim, return_sequences=True), merge_mode="mul"
    )(x)
    x = Activation(activation)(x)
    x = TimeDistributed(Dense(n_features))(x)
    x = Activation(None)(x)

    autoencoder = Model(inputs=i, outputs=x)
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
    i = Input((n_timesteps, n_features))
    mask = Masking(mask_value=0)(i)

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
    x = UpSampling1D(5)(x)

    x = ResidualConv1D(16, 7, activation)(x)
    x = ResidualConv1D(16, 7, activation)(x)

    x = UpSampling1D(3)(x)

    x = ResidualConv1D(8, 15, activation, pool=True)(x)
    x = ResidualConv1D(8, 15, activation)(x)

    x = ResidualConv1D(4, 31, activation, pool=True)(x)
    x = ResidualConv1D(4, 31, activation)(x)

    x = Conv1D(n_features, 1, activation=None, **p)(x)

    # AUTOENCODER
    autoencoder = Model(inputs=i, outputs=x)
    autoencoder.compile(optimizer=Adam(lr=0.001), loss="mse")
    return autoencoder


def lstm_vae_bidir(
    n_timesteps,
    n_features,
    intermediate_dim,
    kl_weight,
    eps=1,
    z_dim=2,
    activation=None,
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

    if activation == "gelu":
        activation = gelu

    inputs = Input(shape=(n_timesteps, n_features))
    # encode -> (latent_dim, )
    xe = Bidirectional(CuDNNLSTM(intermediate_dim, return_sequences=False))(
        inputs
    )

    xe = Activation(activation)(xe)

    # Create a n-dimensional distribution to sample from
    z_mu = Dense(z_dim, name="z_mu")(xe)
    z_log_var = Dense(z_dim, name="z_var")(xe)

    # Add a layer that calculates the KL loss and returns the values
    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var, kl_weight])

    # sample vector from the latent distribution
    z = Lambda(_sample, name="encoded")([z_mu, z_log_var])

    # Repeat so it fits into LSTM
    xd = VariableRepeatVector()([inputs, z])
    xd = Bidirectional(
        CuDNNLSTM(intermediate_dim, return_sequences=True, name="decoded")
    )(xd)

    xd = Activation(activation)(xd)
    # Make sure the final activation is linear and correct dimensionality
    outputs = TimeDistributed(Dense(n_features, activation=None))(xd)

    # Start with 0 weight for the KL loss, and slowly increase with callback
    vae = Model(inputs, outputs)
    vae.compile(optimizer="adam", loss="mse")
    vae.summary()
    return vae


def lstm_cat_vae(
    n_timesteps,
    n_features,
    intermediate_dim,
    activation=None,
    num_dist=1,
    z_dim=2,
):
    """
    Variational autoencoder for variable length time series. Cannot sample over
    the input space, because of variable-length time series compatibility
    """
    if activation == "gelu":
        activation = gelu

    def reparameterize(logits_z):
        tau = 1  # temperature

        # generate latent sample using Gumbel-Softmax for categorical variables
        z = RelaxedOneHotCategorical(tau, logits_z).sample()
        z_hard = tf.cast(tf.one_hot(tf.argmax(z, -1), z_dim), z.dtype)
        z = tf.stop_gradient(z_hard - z) + z
        return z

    # Encode
    input = Input(shape=(n_timesteps, n_features))
    logits_z = Bidirectional(
        CuDNNLSTM(intermediate_dim, return_sequences=False)
    )(input)
    logits_z = Activation(activation)(logits_z)
    logits_z = Dense(z_dim)(logits_z)

    # Sample latent
    z = Lambda(reparameterize, name="encoded")(logits_z)

    # Decode
    x = VariableRepeatVector()([input, z])
    x = Bidirectional(
        CuDNNLSTM(intermediate_dim, return_sequences=True, name="decoded")
    )(x)
    x = Activation(activation)(x)
    output = TimeDistributed(Dense(n_features, activation=None))(x)

    # Calculate loss
    logits_pz = tf.ones_like(logits_z) * (1.0 / z_dim)
    q_cat_z = OneHotCategorical(logits=logits_z)
    p_cat_z = OneHotCategorical(logits=logits_pz)
    KL_qp = KL(q_cat_z, p_cat_z)
    KL_qp_sum = tf.reduce_sum(KL_qp)
    ELBO = tf.reduce_mean(-KL_qp_sum)
    loss = -ELBO

    vae = Model(input, output)
    vae.compile(optimizer="adam", loss="mse")
    vae.add_loss(loss)
    vae.summary()
    return vae


def lstm_vae_bidir_stacked(
    n_timesteps,
    n_features,
    intermediate_dim,
    kl_weight,
    eps=1,
    z_dim=2,
    activation=None,
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

    if activation == "gelu":
        activation = gelu

    inputs = Input(shape=(n_timesteps, n_features))
    # encode -> (latent_dim, )
    xe = Bidirectional(CuDNNLSTM(intermediate_dim, return_sequences=True))(
        inputs
    )

    xe = Bidirectional(CuDNNLSTM(intermediate_dim, return_sequences=False))(xe)

    xe = Dense(intermediate_dim)(xe)

    xe = Activation(activation)(xe)

    # Create a n-dimensional distribution to sample from
    z_mu = Dense(z_dim, name="z_mu")(xe)
    z_log_var = Dense(z_dim, name="z_var")(xe)

    # Add a layer that calculates the KL loss and returns the values
    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var, kl_weight])

    # sample vector from the latent distribution
    z = Lambda(_sample, name="encoded")([z_mu, z_log_var])

    # Repeat so it fits into LSTM
    xd = VariableRepeatVector()([inputs, z])
    xd = Bidirectional(CuDNNLSTM(intermediate_dim, return_sequences=True))(xd)
    xd = Bidirectional(
        CuDNNLSTM(intermediate_dim, return_sequences=True, name="decoded")
    )(xd)

    xd = Activation(activation)(xd)
    # Make sure the final activation is linear and correct dimensionality
    outputs = TimeDistributed(Dense(n_features, activation=None))(xd)

    # Start with 0 weight for the KL loss, and slowly increase with callback
    vae = Model(inputs, outputs)
    vae.compile(optimizer="adam", loss="mse")
    vae.summary()
    return vae


def lstm_vae_unidir(
    n_timesteps,
    n_features,
    intermediate_dim,
    kl_weight,
    eps=1,
    z_dim=2,
    activation=None,
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

    if activation == "gelu":
        activation = gelu

    inputs = Input(shape=(n_timesteps, n_features))

    xe = CuDNNLSTM(intermediate_dim, return_sequences=False)(inputs)
    xe = Activation(activation)(xe)

    z_mu = Dense(z_dim, name="z_mu")(xe)
    z_log_var = Dense(z_dim, name="z_var")(xe)
    z_mu, z_log_var = KLDivergenceLayer()([z_mu, z_log_var, kl_weight])
    z = Lambda(_sample, name="encoded")([z_mu, z_log_var])

    xd = VariableRepeatVector()([inputs, z])
    xd = CuDNNLSTM(intermediate_dim, return_sequences=True, name="decoded")(xd)
    xd = Activation(activation)(xd)
    outputs = TimeDistributed(Dense(n_features, activation=None))(xd)

    # Start with 0 weight for the KL loss, and slowly increase with callback
    vae = Model(inputs, outputs)
    vae.compile(optimizer="adam", loss="mse")
    vae.summary()
    return vae


def model_builder(
    model_build_f,
    build_args,
    patience=3,
    model_dir=None,
    chkpt_tag=None,
    weights_only=False,
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
        save_weights_only=weights_only,
    )

    callbacks = [mca, tb, es, rl]
    return model, callbacks, initial_epoch, model_dir
