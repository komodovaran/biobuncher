import datetime
from glob import glob
from pathlib import Path

import sklearn.model_selection
import tensorflow.python as tf
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *

X = np.ones(shape=(500, 100, 2))
N_TIMESTEPS = X.shape[1]
N_FEATURES = X.shape[2]
CALLBACK_TIMEOUT = 5
EPOCHS = 100
BATCH_SIZE = 32
LATENT_DIM = 5


def build_lstm_autoencoder(n_features, latent_dim, n_timesteps):
    repeat_dim = (n_timesteps // latent_dim) * latent_dim
    lstm_units = 10

    # ENCODER
    ei = Input(shape=(None, n_features))
    ez = LSTM(units=lstm_units, return_sequences=False)(ei)
    ez = Activation("relu")(ez)
    eo = Dense(units=latent_dim)(ez)
    encoder = Model(inputs=ei, outputs=eo)

    # DECODER
    latent_inputs = Input(shape=(latent_dim,))
    dz = RepeatVector(repeat_dim)(latent_inputs)
    dz = LSTM(units=lstm_units, return_sequences=True)(dz)
    dz = Activation("relu")(dz)
    do = TimeDistributed(Dense(n_features))(dz)
    decoder = Model(inputs=latent_inputs, outputs=do)

    # AUTOENCODER
    do = decoder(encoder(ei))
    autoencoder = Model(inputs=ei, outputs=do)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def build_conv_autoencoder(n_features, latent_dim, n_timesteps):
    p = {"padding": "same", "kernel_initializer": "he_uniform"}

    # ENCODER
    ei = Input((n_timesteps, n_features))
    ez = Conv1D(16, 3, **p)(ei)
    ez = BatchNormalization()(ez)
    ez = Activation("relu")(ez)

    ez = Flatten()(ez)
    eo = Dense(units=latent_dim, activation=None)(ez)
    encoder = Model(inputs=ei, outputs=eo)

    # DECODER
    latent_inputs = Input(shape=(latent_dim,))
    dz = Dense(n_timesteps * 16)(latent_inputs)
    dz = Reshape((n_timesteps, 16))(dz)
    dz = BatchNormalization()(dz)
    dz = Activation("relu")(dz)

    dz = Conv1D(16, 3, **p)(dz)
    dz = BatchNormalization()(dz)
    dz = Activation("relu")(dz)

    do = Conv1D(n_features, 1, activation=None, **p)(dz)
    decoder = Model(inputs=latent_inputs, outputs=do)

    # AUTOENCODER
    do = decoder(encoder(ei))
    autoencoder = Model(inputs = ei, outputs = do)
    autoencoder.compile(optimizer = "adam", loss = "mse")
    return autoencoder


def prepare_data(X, train_size=0.8):
    X_train, X_test = sklearn.model_selection.train_test_split(
        X, train_size=train_size, random_state=1
    )
    X_train_len, X_test_len = len(X_train), len(X_test)

    X_train, X_test = [
        tf.data.Dataset.from_tensor_slices((tf.constant(Xi), tf.constant(Xo)))
        for (Xi, Xo) in ((X_train, X_train), (X_test, X_test))
    ]

    X_train, X_test = [
        data.shuffle(buffer_size=10 * BATCH_SIZE).batch(BATCH_SIZE)
        for data in (X_train, X_test)
    ]
    return (X_train, X_test), (X_train_len, X_test_len)


def model_builder(model_dir, model_build_f, build_args):
    initial_epoch = 0
    if model_dir is None:
        print("no model directory set. Creating new model.")
        # Create new directory with current time
        model_dir = Path(
            "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        model = model_build_f(*build_args)
    else:
        try:
            print("Loading model from specified directory")

            # Get latest model from list of directories
            latest = sorted(
                glob(model_dir.joinpath("model_???").as_posix()), reverse=True
            )[0]

            # get the last 3 values in dir name as epoch
            initial_epoch = int(latest[-3:])
            model = model_build_f(*build_args)
            # model.set_weights(weights)
        except IndexError:
            print("no model found. Creating new model.")
            model_dir = Path(
                "models/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            )
            model = model_build_f(*build_args)

    # callbacks here
    mca = ModelCheckpoint(
        filepath=model_dir.joinpath("model_{epoch:03d}").as_posix(),
        save_best_only=True,
        save_weights_only = True
    )

    callbacks = [mca]
    return model, callbacks, initial_epoch


if __name__ == "__main__":
    (X_train, X_test), (X_train_len, X_test_len) = prepare_data(X)

    model, callbacks, initial_epoch = model_builder(
        model_dir=None,
        model_build_f=build_lstm_autoencoder,
        build_args=(N_FEATURES, LATENT_DIM, N_TIMESTEPS),
    )

    model.summary()
    model.fit(
        x=X_train.repeat(),
        validation_data=X_test.repeat(),
        epochs=EPOCHS,
        steps_per_epoch=X_train_len // BATCH_SIZE,
        validation_steps=X_test_len // BATCH_SIZE,
        initial_epoch=0,
        callbacks=callbacks,
    )
