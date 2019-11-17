import datetime
from glob import glob
from pathlib import Path

import tensorflow.python as tf
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *


class ResidualConv1D:
    """
    Performs convolutions with residual connections
    """

    def __init__(self, filters, kernel_size, activation, pool = False):
        self.kernel_size = kernel_size
        self.activation = activation
        self.pool = pool
        self.p = {
            "padding"           : "same",
            "kernel_initializer": "he_uniform",
            "strides"           : 1,
            "filters"           : filters,
        }

    def build(self, x):
        res = x
        if self.pool:
            x = MaxPooling1D(1, padding = "same")(x)
            res = Conv1D(kernel_size = 1, **self.p)(res)

        out = Conv1D(kernel_size = 1, **self.p)(x)

        out = BatchNormalization()(out)
        out = Activation(self.activation)(out)
        out = Conv1D(kernel_size = self.kernel_size, **self.p)(out)

        out = BatchNormalization()(out)
        out = Activation(self.activation)(out)
        out = Conv1D(kernel_size = self.kernel_size, **self.p)(out)

        out = Add()([res, out])

        return out

    def __call__(self, x):
        return self.build(x)


def create_vae(n_features, n_timesteps, latent_dim = 10):
    def _sampling(args):
        """
        Reparameterization trick by sampling from an isotropic unit Gaussian. instead of sampling from Q(z|X),
            sample epsilon = N(0,I)
            z = z_mean + sqrt(var) * epsilon

        # Arguments
            args (tensor): mean and log of variance of Q(z|X)

        # Returns
            z (tensor): sampled latent vector
        """
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape = (batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def _loss():
        reconstruction_loss = tf.keras.losses.mse(
            K.flatten(inputs), K.flatten(outputs)
        )
        reconstruction_loss *= n_timesteps * n_features
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis = -1)
        kl_loss *= -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss)
        return vae_loss

    p = {"padding": "same", "kernel_initializer": "he_uniform"}

    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape = (n_timesteps, n_features), name = "encoder_input")
    x = Flatten()(inputs)
    x = Dense(512, activation = "relu")(x)

    x = BatchNormalization()(x)
    x = Dense(128, activation = "relu")(x)
    z_mean = Dense(latent_dim, name = "z_mean")(x)
    z_log_var = Dense(latent_dim, name = "z_log_var")(x)

    # use reparameterization trick to push the sampling out as input
    z = Lambda(_sampling, output_shape = (latent_dim,), name = "z")(
        [z_mean, z_log_var]
    )
    # build decoder model
    latent_inputs = Input(shape = (latent_dim,), name = "z_sampling")
    x = Dense(128, activation = "relu")(latent_inputs)
    x = BatchNormalization()(x)
    x = Dense(512, activation = "relu")(x)
    outputs = Dense(n_timesteps * n_features, activation = "sigmoid")(x)

    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name = "encoder")
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name = "decoder")
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name = "vae_mlp")

    vae.summary()

    vae.add_loss(_loss())
    vae.compile(optimizer = "adam")
    return vae


def build_lstm_autoencoder(n_features, latent_dim, n_timesteps):
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

    repeat_dim = (n_timesteps // latent_dim) * latent_dim
    lstm_units = 10

    if repeat_dim != n_timesteps:
        raise ValueError(
            "Latent dim {} cannot be multiplied up to full dim {}".format(
                latent_dim, n_timesteps
            )
        )

    # ENCODER
    inputs = Input(shape = (None, n_features))
    ez = LSTM(units = lstm_units, return_sequences = False)(inputs)
    ez = Activation("relu")(ez)
    eo = Dense(units = latent_dim)(ez)

    encoder = Model(inputs = inputs, outputs = eo)

    # DECODER
    latent_inputs = Input(shape = (latent_dim,))
    dz = RepeatVector(repeat_dim)(latent_inputs)
    dz = LSTM(units = lstm_units, return_sequences = True)(dz)
    dz = Activation("relu")(dz)
    outputs = TimeDistributed(Dense(n_features))(dz)

    decoder = Model(inputs = latent_inputs, outputs = outputs)

    # AUTOENCODER
    outputs = decoder(encoder(inputs))
    autoencoder = Model(inputs = inputs, outputs = outputs)
    autoencoder.compile(optimizer = "adam", loss = "mse")
    return autoencoder


def build_conv_autoencoder(n_features, latent_dim, n_timesteps, activation = "relu"):
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

    # filters, size

    # ENCODER
    ei = Input(shape = (n_timesteps, n_features))
    ez = Conv1D(4, 7, **p)(ei)  # 50x4
    ez = BatchNormalization()(ez)
    ez = Activation(activation)(ez)

    ez = Conv1D(8, 5, **p)(ez)  # 50x8
    ez = BatchNormalization()(ez)
    ez = Activation(activation)(ez)

    ez = Conv1D(16, 3, **p)(ez)  # 50x16
    ez = BatchNormalization()(ez)
    ez = Activation(activation)(ez)

    ez = Flatten()(ez)  # 800,
    eo = Dense(units = latent_dim, activation = None)(ez)  # latent_dim,
    encoder = Model(inputs = ei, outputs = eo)

    # DECODER
    latent_inputs = Input(shape = (latent_dim,))  # latent_dim.,
    dz = Dense(n_timesteps * 16)(
        latent_inputs
    )  # restore datapoints from latent
    dz = Reshape((n_timesteps, 16))(dz)  # reshape to correct dimension
    dz = BatchNormalization()(dz)
    dz = Activation(activation)(dz)

    dz = Conv1D(16, 3, **p)(dz)  # 50x16
    dz = BatchNormalization()(dz)
    dz = Activation(activation)(dz)

    dz = Conv1D(8, 5, **p)(dz)  # 50x8
    dz = BatchNormalization()(dz)
    dz = Activation(activation)(dz)

    dz = Conv1D(4, 7, **p)(dz)  # 50x4
    dz = BatchNormalization()(dz)
    dz = Activation(activation)(dz)

    do = Conv1D(n_features, 1, activation = None, **p)(dz)
    decoder = Model(inputs = latent_inputs, outputs = do)

    # AUTOENCODER
    do = decoder(encoder(ei))
    autoencoder = Model(inputs = ei, outputs = do)
    autoencoder.compile(optimizer = "sgd", loss = "mse")
    return autoencoder


def build_dilated_conv_autoencoder(n_features, latent_dim, n_timesteps):
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
    p = {"padding": "same", "kernel_initializer": "he_uniform"}

    # filters, size

    # ENCODER
    ei = Input(shape = (n_timesteps, n_features))

    ez = Conv1D(2, 1, dilation_rate = 7)(ei)
    ez = BatchNormalization()(ez)
    ez = Activation("elu")(ez)

    ez = Conv1D(4, 7, **p, dilation_rate = 5)(ez)  # 100x4
    ez = BatchNormalization()(ez)
    ez = Activation("elu")(ez)

    ez = Conv1D(6, 7, **p, dilation_rate = 3)(ez)  # 100x6
    ez = BatchNormalization()(ez)
    ez = Activation("elu")(ez)

    ez = Flatten()(ez)  # 100*2*6,
    eo = Dense(units = latent_dim, activation = None)(ez)  # 10
    encoder = Model(inputs = ei, outputs = eo)

    # DECODER
    latent_inputs = Input(shape = (latent_dim,))  # latent_dim.,
    dz = Dense(n_timesteps * 6)(latent_inputs)  # restore datapoints from latent
    dz = Reshape((n_timesteps, 6))(dz)  # reshape to correct dimension
    dz = BatchNormalization()(dz)
    dz = Activation("elu")(dz)

    dz = Conv1D(6, 7, **p, dilation_rate = 3)(dz)  # 50x4
    dz = BatchNormalization()(dz)
    dz = Activation("elu")(dz)

    dz = Conv1D(4, 7, **p, dilation_rate = 5)(dz)  # 50x4
    dz = BatchNormalization()(dz)
    dz = Activation("elu")(dz)

    do = Conv1D(2, 1, activation = None, **p, dilation_rate = 7)(dz)
    decoder = Model(inputs = latent_inputs, outputs = do)

    # AUTOENCODER
    do = decoder(encoder(ei))
    autoencoder = Model(inputs = ei, outputs = do)
    autoencoder.compile(optimizer = "adam", loss = "mse")

    return autoencoder


def build_residual_conv_autoencoder(
    n_features, latent_dim, n_timesteps, activation = "relu"
):
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

    # ENCODER
    ei = Input((n_timesteps, n_features))
    ez = Conv1D(2, 7, **p)(ei)
    ez = BatchNormalization()(ez)
    ez = Activation(activation)(ez)

    ez = ResidualConv1D(4, 31, activation, pool = True)(ez)
    ez = ResidualConv1D(4, 31, activation)(ez)

    ez = ResidualConv1D(8, 15, activation, pool = True)(ez)
    ez = ResidualConv1D(8, 15, activation)(ez)

    ez = ResidualConv1D(16, 7, activation, pool = True)(ez)
    ez = ResidualConv1D(16, 7, activation)(ez)

    ez = Flatten()(ez)  # 800
    eo = Dense(units = latent_dim, activation = None)(ez)  # 10

    encoder = Model(inputs = ei, outputs = eo)

    # DECODER
    latent_inputs = Input(shape = (latent_dim,))  # 10
    dz = Dense(n_timesteps * 16)(latent_inputs)  # 800
    dz = Reshape((n_timesteps, 16))(dz)  # 50x16
    dz = BatchNormalization()(dz)
    dz = Activation(activation)(dz)

    dz = ResidualConv1D(16, 7, activation, pool = True)(dz)
    dz = ResidualConv1D(16, 7, activation)(dz)

    dz = ResidualConv1D(8, 15, activation, pool = True)(dz)
    dz = ResidualConv1D(8, 15, activation)(dz)

    dz = ResidualConv1D(4, 31, activation, pool = True)(dz)
    dz = ResidualConv1D(4, 31, activation)(dz)

    do = Conv1D(2, 1, **p, activation = None)(dz)  # 50x2

    decoder = Model(inputs = latent_inputs, outputs = do)

    # AUTOENCODER
    do = decoder(encoder(ei))
    autoencoder = Model(inputs = ei, outputs = do)
    autoencoder.compile(optimizer = "adam", loss = "mse")
    return autoencoder


def model_builder(model_build_f, build_args, patience = 3, model_dir = None, chkpt_tag = None):
    """Loads model and callbacks"""
    # set a directory in case None is set initially
    if chkpt_tag is None:
        chkpt_tag = ""

    _model_dir = Path(
        "models/"
        + datetime.datetime.now().strftime("%Y%m%d-%H%M")
        + chkpt_tag
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
                glob(Path(model_dir).joinpath("model_???")), reverse = True,
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
    es = EarlyStopping(patience = patience)
    tb = TensorBoard(log_dir = model_dir.as_posix())
    mca = ModelCheckpoint(
        filepath = model_dir.joinpath("model_{epoch:03d}.h5").as_posix(),
        save_best_only = True,
    )
    callbacks = [mca, tb, es]
    return model, callbacks, initial_epoch
