"""
https://arxiv.org/abs/1312.6114
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import tensorflow as tf
from tensorflow.keras import backend as K, metrics
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from lib.tfcustom import VariableRepeatVector
from tensorflow.python.keras.layers import CuDNNLSTM

batch_size = 128
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 200
epsilon_std = 1


def get_timeseries_data(length=50, n_each_class=200):
    """
    Make 3 types of sequence data with variable length
    """
    data = []
    for _ in range(n_each_class):
        r = np.random.normal
        l = np.linspace
        i = np.random.randint

        x_noisy = np.column_stack(
            (
                (np.cos(l(i(1, 5), 5, length)) + r(0, 0.2, length)),
                ((1 + np.sin(l(i(1, 20), 5, length)) + r(0, 0.2, length))),
            )
        )

        x_wavy = np.column_stack(
            (
                (np.cos(l(0, i(1, 5), length)) + r(0, 0.2, length)),
                ((2 + np.sin(l(i(1, 20), 20, length)) + r(0, 0.2, length))),
            )
        )

        x_spikes = np.column_stack(
            (
                (np.cos(l(i(1, 5), 20, length)) + r(0, 0.2, length)) ** 3,
                (
                    (1 + np.sin(l(i(1, 20), 20, length)) + r(0, 0.2, length))
                    ** 3
                ),
            )
        )

        # Randomly cut the begining of traces and fill in with zeroes to mimick short traces
        zero = np.random.randint(1, length // 2)
        # x_noisy[0:zero] = 0
        # x_wavy[0:zero] = 0
        # x_spikes[0:zero] = 0

        data.append(x_noisy)
        data.append(x_wavy)
        data.append(x_spikes)

    data = np.array(data)
    data = data.reshape((-1, length, 2))

    X_train, X_test = sklearn.model_selection.train_test_split(
        data, train_size=0.8
    )

    mu = np.mean(X_train, axis=(0, 1))
    sg = np.std(X_train, axis=(0, 1))
    X_train = (X_train - mu) / sg
    X_test = (X_test - mu) / sg

    return X_train, X_test


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
        shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=epsilon_std
    )
    return z_mean + K.exp(z_log_var / 2) * epsilon


def crossentropy_loss(inputs, outputs):
    return K.cast(
        K.shape(K.flatten(inputs)), tf.float32
    ) * metrics.binary_crossentropy(K.flatten(inputs), K.flatten(outputs))


def mse_loss(inputs, outputs):
    return tf.reduce_sum(metrics.mse(inputs, outputs))


def kullback_leibner_loss(z_mean, z_log_var):
    return - 0.5 * tf.reduce_sum(1 + z_log_var - z_mean ** 2 - tf.exp(z_log_var), 1)


def vae_model():
    """VAE model"""
    inputs = Input(shape=(784, 1))

    # x = LSTM(intermediate_dim, return_sequences = False)(inputs)

    h = Dense(intermediate_dim, activation="relu")(inputs)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # we instantiate these layers separately so as to reuse them later
    decoder_h = Dense(intermediate_dim, activation="relu")
    decoder_mean = Dense(original_dim, activation="sigmoid")
    h_decoded = decoder_h(z)
    outputs_pre = decoder_mean(h_decoded)

    outputs = Reshape((784, 1))(outputs_pre)

    # instantiate VAE model
    vae = Model(inputs, outputs)
    reconstr_loss = mse_loss(inputs, outputs)

    kl_loss = kullback_leibner_loss(z_mean, z_log_var)

    vae_loss = tf.reduce_mean(reconstr_loss + kl_loss)

    vae.add_loss(vae_loss)
    vae.compile(optimizer="rmsprop")
    vae.summary()

    encoder = Model(inputs, z_mean)
    decoder_input = Input(shape=(latent_dim,))
    _h_decoded = decoder_h(decoder_input)
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    return vae, encoder, generator


def conv_vae_2d():
    inputs = Input(
        shape=(28, 28, 1)
    )  # must have last dimension to be recognized as image

    x = Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = Conv2D(64, 3, padding="same", activation="relu", strides=(2, 2))(x)
    x = Conv2D(64, 3, padding="same", activation="relu")(x)
    x = Conv2D(64, 3, padding="same", activation="relu")(x)

    # need to know the shape of the network here for the decoder
    shape_before_flattening = K.int_shape(x)

    x = Flatten()(x)
    x = Dense(32, activation="relu")(x)

    # Two outputs, latent mean and (log)variance
    z_mu = Dense(latent_dim)(x)
    z_log_sigma = Dense(latent_dim)(x)

    # sample vector from the latent distribution
    z = Lambda(sampling)([z_mu, z_log_sigma])

    ## DECODER ARCHITECTURE
    # decoder takes the latent distribution sample as input
    decoder_input = Input(K.int_shape(z)[1:])

    # Expand to 784 total pixels
    x = Dense(np.prod(shape_before_flattening[1:]), activation="relu")(
        decoder_input
    )

    # unflatten back to image
    x = Reshape(shape_before_flattening[1:])(x)

    # use Conv2DTranspose to reverse the conv layers from the encoder
    x = Conv2DTranspose(
        32, 3, padding="same", activation="relu", strides=(2, 2)
    )(x)
    x = Conv2D(1, 3, padding="same", activation="sigmoid")(x)

    # decoder model statement
    decoder = Model(decoder_input, x)

    # apply the decoder to the sample from the latent distribution
    outputs = decoder(z)

    xent_loss = crossentropy_loss(inputs, outputs)
    kl_loss = kullback_leibner_loss(z_mu, z_log_sigma)
    vae_loss = K.mean(xent_loss + kl_loss)

    vae = Model(inputs, outputs)
    vae.add_loss(vae_loss)
    vae.compile(optimizer="adam")

    return vae


def lstm_vae_1d():
    inputs = Input(shape=(50, 2))  # (timesteps, features)

    xe = CuDNNLSTM(128, return_sequences=False)(inputs)

    # need to know the shape of the network here for the decoder
    shape_before_flattening = K.int_shape(xe)
    x = Flatten()(xe)

    # Two outputs, latent mean and (log)variance
    z_mu = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # sample vector from the latent distribution
    z = Lambda(sampling)([z_mu, z_log_var])

    ## DECODER ARCHITECTURE
    # decoder takes the latent distribution sample as input
    decoder_input = Input(K.int_shape(z)[1:])

    x = Dense(np.prod(shape_before_flattening[1:]), activation="relu")(K.int_shape(z)[1:])

    # TODO: make it a single model for VariableRepeatVector to work
    # x = VariableRepeatVector()([inputs, x])
    # print(x.shape)
    x = RepeatVector(50)(x)
    x = CuDNNLSTM(128, return_sequences=True)(x)
    x = TimeDistributed(Dense(2, activation = None))(x)

    # decoder model statement
    decoder = Model(decoder_input, x)

    # apply the decoder to the sample from the latent distribution
    outputs = decoder(z)

    # xent_loss = crossentropy_loss(inputs, outputs)

    reconstr_loss = mse_loss(inputs, outputs)
    kl_loss = kullback_leibner_loss(z_mu, z_log_var)
    vae_loss = tf.reduce_sum(kl_loss + reconstr_loss)

    vae = Model(inputs, outputs)
    vae.add_loss(vae_loss)
    vae.compile(optimizer="rmsprop")
    vae.summary()
    return vae


def lstm_vae_1d_repeatvector():
    inputs = Input(shape=(50, 2))  # (timesteps, features)

    xe = CuDNNLSTM(128, return_sequences=False)(inputs)

    # need to know the shape of the network here for the decoder
    shape_before_flattening = K.int_shape(xe)
    x = Flatten()(xe)

    # Two outputs, latent mean and (log)variance
    z_mu = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    # sample vector from the latent distribution
    z = Lambda(sampling)([z_mu, z_log_var])

    ## DECODER ARCHITECTURE
    x = Dense(np.prod(shape_before_flattening[1:]), activation="elu")(z)
    x = VariableRepeatVector()([inputs, x])

    x = CuDNNLSTM(128, return_sequences=True)(x)
    outputs = TimeDistributed(Dense(2, activation = None))(x)

    reconstr_loss = mse_loss(inputs, outputs)
    kl_loss = kullback_leibner_loss(z_mu, z_log_var)
    vae_loss = tf.reduce_sum(kl_loss + reconstr_loss)

    vae = Model(inputs, outputs)
    vae.add_loss(vae_loss)
    vae.compile(optimizer="rmsprop")
    vae.summary()
    return vae


# train the VAE on MNIST digits
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# x_train = x_train.astype("float32") / 255.0
# x_test = x_test.astype("float32") / 255.0

# x_train = np.expand_dims(x_train, axis = -1)
# x_test = np.expand_dims(x_test, axis = -1)

# x_train = resample(x_train, n_samples = 1000)
# x_test = resample(x_test, n_samples = 1000)

# Train the VAE on simulated timeseries
x_train, x_test = get_timeseries_data(length=50, n_each_class=1000)

vae = lstm_vae_1d_repeatvector()
vae.fit(
    x=x_train,
    y=None,
    shuffle=True,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, None),
)

x_pred = vae.predict(x_test[0:10])

fig, ax = plt.subplots(ncols=2, nrows=4)

for i in range(4):
    ax[i, 0].plot(x_test[i])
    ax[i, 1].plot(x_pred[i])
plt.show()

quit()

# x_test_encoded = encoder.predict(x_test, batch_size = batch_size)
# plt.figure(figsize = (6, 6))
# plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1])
# plt.colorbar()
# plt.show()

# encoded = encoder.predict(x_train[0:5])

#
# # display a 2D manifold of the digits
# n = 15  # figure with 15x15 digits
# digit_size = 28
# figure = np.zeros((digit_size * n, digit_size * n))
# # linearly spaced coordinates on the unit square were transformed through the inverse CDF (ppf) of the Gaussian
# # to produce values of the latent variables z, since the prior of the latent space is Gaussian
# grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
# grid_y = norm.ppf(np.linspace(0.05, 0.95, n))
#
# for i, yi in enumerate(grid_x):
#     for j, xi in enumerate(grid_y):
#         z_sample = np.array([[xi, yi]])
#         x_decoded = generator.predict(z_sample)
#         digit = x_decoded[0].reshape(digit_size, digit_size)
#         figure[
#             i * digit_size : (i + 1) * digit_size,
#             j * digit_size : (j + 1) * digit_size,
#         ] = digit
#
# plt.figure(figsize=(10, 10))
# plt.imshow(figure, cmap="Greys_r")
# plt.show()
