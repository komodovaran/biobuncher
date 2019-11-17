import os

import numpy as np
import sklearn.model_selection
import tensorflow.python as tf
from itertools import product
import lib.math
import lib.models
from lib.models import build_conv_autoencoder, build_residual_conv_autoencoder

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "" # disable CUDA device for testing


def _model_type(model_build_f):
    """Returns name of model from builder function"""
    return str(model_build_f.__name__[6:])


def _get_data(path):
    """
    Loads all traces
    """
    X = np.load(path)["data"]
    X = X[:, :, [0, 1]]
    return X


def _prepare_data(X, train_size=0.8):
    X_train, X_test = sklearn.model_selection.train_test_split(
        X, train_size=train_size
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


if __name__ == "__main__":
    _INPUT_NPZ = (
        # "results/intensities/tracks-cme-catidx_resampled-50.npz",
        # "results/intensities/tracks-cme_split-c1_resampled-50.npz",
        "results/intensities/tracks-tpy_roi-int_resampled-50.npz",
    )
    _LATENT_DIM = (5,) #10, 15)
    _MODELF = (build_conv_autoencoder, build_residual_conv_autoencoder)

    for _input_npz, _latent_dim, _modelf in product(
        _INPUT_NPZ, _LATENT_DIM, _MODELF
    ):
        CONTINUE_DIR = None
        INPUT_NPZ = _input_npz
        LATENT_DIM = _latent_dim

        try:
            X_raw = _get_data(INPUT_NPZ)
        except FileNotFoundError:
            continue

        X = lib.math.normalize_tensor(X_raw, per_feature=False)
        if X.max() > 1:
            raise ValueError("Check normalization!")

        N_TIMESTEPS = X.shape[1]
        N_FEATURES = X.shape[2]
        EARLY_STOPPING = 10
        EPOCHS = 10
        BATCH_SIZE = 32

        TAG = "_" + _model_type(_modelf)
        TAG += "_dim={}_".format(LATENT_DIM)
        TAG += "_data={}".format(INPUT_NPZ.split("/")[-1])

        (X_train, X_test), (X_train_len, X_test_len) = _prepare_data(X)

        model, callbacks, initial_epoch = lib.models.model_builder(
            model_dir=CONTINUE_DIR,
            chkpt_tag=TAG,
            patience=EARLY_STOPPING,
            model_build_f=_modelf,
            build_args=(N_FEATURES, LATENT_DIM, N_TIMESTEPS),
        )

        model.summary()
        model.fit(
            x=X_train.repeat(),
            validation_data=X_test.repeat(),
            epochs=EPOCHS,
            steps_per_epoch=X_train_len // BATCH_SIZE,
            validation_steps=X_test_len // BATCH_SIZE,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
        )
