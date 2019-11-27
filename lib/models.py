import datetime
from glob import glob
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.python.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import backend as K
from tensorflow.keras.losses import KLDivergence, Huber


class VariableRepeatVector:
    """
    Tidies up the call to a lambda function by integrating it in a
    layer-like wrapper

    The two usages are identical:
    decoded = VariableRepeatVector()([inputs, encoded])
    decoded = Lambda(variable_repeat)([inputs, encoded])
    """

    @staticmethod
    def variable_repeat(x):
        # matrix with ones, shaped as (batch, steps, 1)
        step_matrix = K.ones_like(x[0][:, :, :1])
        # latent vars, shaped as (batch, 1, latent_dim)
        latent_matrix = K.expand_dims(x[1], axis=1)
        return K.batch_dot(step_matrix, latent_matrix)

    def __call__(self, x):
        return Lambda(self.variable_repeat)(x)


class ResidualConv1D:
    """
    Performs convolutions with residual connections
    """

    def __init__(self, filters, kernel_size, activation, pool=False):
        self.kernel_size = kernel_size
        self.activation = activation
        self.pool = pool
        self.p = {
            "padding": "same",
            "kernel_initializer": "he_uniform",
            "strides": 1,
            "filters": filters,
        }

    def build(self, x):
        res = x
        if self.pool:
            x = MaxPooling1D(1, padding="same")(x)
            res = Conv1D(kernel_size=1, **self.p)(res)

        out = Conv1D(kernel_size=1, **self.p)(x)

        out = BatchNormalization()(out)
        out = Activation(self.activation)(out)
        out = Conv1D(kernel_size=self.kernel_size, **self.p)(out)

        out = BatchNormalization()(out)
        out = Activation(self.activation)(out)
        out = Conv1D(kernel_size=self.kernel_size, **self.p)(out)

        out = Add()([res, out])

        return out

    def __call__(self, x):
        return self.build(x)


# def create_vae(n_features, n_timesteps, latent_dim = 10):
#     def _sampling(args):
#         """
#         Reparameterization trick by sampling from an isotropic unit Gaussian. instead of sampling from Q(z|X),
#             sample epsilon = N(0,I)
#             z = z_mean + sqrt(var) * epsilon
#
#         # Arguments
#             args (tensor): mean and log of variance of Q(z|X)
#
#         # Returns
#             z (tensor): sampled latent vector
#         """
#         z_mean, z_log_var = args
#         batch = K.shape(z_mean)[0]
#         dim = K.int_shape(z_mean)[1]
#         # by default, random_normal has mean = 0 and std = 1.0
#         epsilon = K.random_normal(shape = (batch, dim))
#         return z_mean + K.exp(0.5 * z_log_var) * epsilon
#
#     def _loss():
#         reconstruction_loss = tf.keras.losses.mse(
#             K.flatten(inputs), K.flatten(outputs)
#         )
#         reconstruction_loss *= n_timesteps * n_features
#         kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#         kl_loss = K.sum(kl_loss, axis = -1)
#         kl_loss *= -0.5
#         vae_loss = K.mean(reconstruction_loss + kl_loss)
#         return vae_loss
#
#     # p = {"padding": "same", "kernel_initializer": "he_uniform"}
#
#     # VAE model = encoder + decoder
#     # build encoder model
#     inputs = Input(shape = (n_timesteps, n_features), name = "encoder_input")
#     x = Flatten()(inputs)
#     x = Dense(512, activation = "relu")(x)
#
#     x = BatchNormalization()(x)
#     x = Dense(128, activation = "relu")(x)
#     z_mean = Dense(latent_dim, name = "z_mean")(x)
#     z_log_var = Dense(latent_dim, name = "z_log_var")(x)
#
#     # use reparameterization trick to push the sampling out as input
#     z = Lambda(_sampling, output_shape = (latent_dim,), name = "z")(
#         [z_mean, z_log_var]
#     )
#     # build decoder model
#     latent_inputs = Input(shape = (latent_dim,), name = "z_sampling")
#     x = Dense(128, activation = "relu")(latent_inputs)
#     x = BatchNormalization()(x)
#     x = Dense(512, activation = "relu")(x)
#     outputs = Dense(n_timesteps * n_features, activation = "sigmoid")(x)
#
#     # instantiate encoder model
#     encoder = Model(inputs, [z_mean, z_log_var, z], name = "encoder")
#     # instantiate decoder model
#     decoder = Model(latent_inputs, outputs, name = "decoder")
#     # instantiate VAE model
#     outputs = decoder(encoder(inputs)[2])
#     vae = Model(inputs, outputs, name = "vae_mlp")
#
#     vae.summary()
#
#     vae.add_loss(_loss())
#     vae.compile(optimizer = "adam")
#     return vae


# def vae_lstm(n_features, n_timesteps, latent_dim = 10):
#     def _sampling(args):
#         """
#         Reparameterization trick by sampling from an isotropic unit Gaussian. instead of sampling from Q(z|X),
#             sample epsilon = N(0,I)
#             z = z_mean + sqrt(var) * epsilon
#
#         # Arguments
#             args (tensor): mean and log of variance of Q(z|X)
#
#         # Returns
#             z (tensor): sampled latent vector
#         """
#         z_mean, z_log_var = args
#         batch = K.shape(z_mean)[0]
#         dim = K.int_shape(z_mean)[1]
#         # by default, random_normal has mean = 0 and std = 1.0
#         epsilon = K.random_normal(shape = (batch, dim))
#         return z_mean + K.exp(0.5 * z_log_var) * epsilon
#
#     def _loss():
#         reconstruction_loss = tf.keras.losses.mse(
#             K.flatten(inputs), K.flatten(outputs)
#         )
#         reconstruction_loss *= n_timesteps * n_features
#         kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
#         kl_loss = K.sum(kl_loss, axis = -1)
#         kl_loss *= -0.5
#         vae_loss = K.mean(reconstruction_loss + kl_loss)
#         return vae_loss
#
#     repeat_dim = (n_timesteps // latent_dim) * latent_dim
#
#     # build encoder model
#     inputs = Input(shape = (n_timesteps, n_features), name = "encoder_input")
#
#     # ENCODER
#     inputs = Input(shape = (None, n_features))
#     ez = CuDNNLSTM(units = 32, return_sequences = False)(inputs)
#     ez = Activation("tanh")(ez)
#     z_mean = Dense(latent_dim, name = "z_mean")(ez)
#     z_log_var = Dense(latent_dim, name = "z_log_var")(ez)
#
#     # use reparameterization trick to push the sampling out as input
#     z = Lambda(_sampling, output_shape = (latent_dim,), name = "z")(
#         [z_mean, z_log_var]
#     )
#
#     # build decoder model
#     latent_inputs = Input(shape = (latent_dim,), name = "z_sampling")
#     dz = RepeatVector(repeat_dim)(latent_inputs)
#     dz = CuDNNLSTM(units = 32, return_sequences = True)(dz)
#     dz = Activation("sigmoid")(dz)
#     outputs = TimeDistributed(Dense(n_features))(dz)
#
#     # instantiate encoder model
#     encoder = Model(inputs, [z_mean, z_log_var, z], name = "encoder")
#     # instantiate decoder model
#     decoder = Model(latent_inputs, outputs, name = "decoder")
#     # instantiate VAE model
#     outputs = decoder(encoder(inputs)[2])
#     vae = Model(inputs, outputs, name = "vae_mlp")
#
#     vae.summary()
#
#     vae.add_loss(_loss())
#     vae.compile(optimizer = "adam")
#     return vae


def single_lstm_autoencoder(n_timesteps, n_features, latent_dim, activation = None):
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
    x = LSTM(units=latent_dim, recurrent_dropout = 0.2, return_sequences=True, name="encoded")(
        inputs
    )
    x = TimeDistributed(Dense(n_features))(x)
    x = Activation(None)(x)

    autoencoder = Model(inputs=inputs, outputs=x)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder



def multi_lstm_autoencoder(n_timesteps, n_features, latent_dim, activation = "relu"):

    input = Input(shape=(n_timesteps, n_features))

    x = Bidirectional(CuDNNLSTM(latent_dim), name = "encoded")(input)
    x = Activation(activation)(x)

    x = VariableRepeatVector()([input, x])

    x = Bidirectional(CuDNNLSTM(latent_dim, return_sequences = True))(x)
    x = Activation(activation)(x)
    x = TimeDistributed(Dense(n_features))(x)
    x = Activation(None)(x)

    autoencoder = Model(inputs=input, outputs=x)
    autoencoder.compile(optimizer="adam", loss="mse")
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
    callbacks = [mca, tb, es, rl]
    return model, callbacks, initial_epoch
