import itertools
import os

import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow.python as tf

import lib.math
import lib.old_models

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _get_data(path):
    """
    Loads all traces
    """
    print(path)
    X = np.load(path, allow_pickle = True)["data"]
    if X.shape[0] < 100:
        raise ValueError("File is suspiciously small. Recheck!")
    return X


def _preprocess(
    X, path, train_size = 0.8, per_sample_norm = True, per_feature_norm = True
):
    """
    Preprocess data into tensors and appropriate train/test sets
    """
    X_train, X_test = sklearn.model_selection.train_test_split(
        X, train_size = train_size, random_state = 0
    )

    if per_sample_norm == "sample":
        # normalization per sample, independent of whole distribution
        X_train, X_test = [
            lib.math.maxabs_tensor(X, per_feature = per_feature_norm)
            for X in (X_train, X_test)
        ]
        np.savez(
            path[:-4] + "_traintest.npz", X_train = X_train, X_test = X_test
        )
    elif per_sample_norm == "dataset":
        # normalize with train set max
        X_train_max = np.max(X_train, axis = (0, 1, 2))
        X_train, X_test = [
            lib.math.div0(X, X_train_max) for X in (X_train, X_test)
        ]
        np.savez(
            path[:-4] + "_traintest.npz",
            X_train = X_train,
            X_test = X_test,
            scale = X_train_max,
        )
    else:
        raise ValueError

    # Calculate known lengths before converting into tensors
    X_train_len, X_test_len = len(X_train), len(X_test)

    if len(X.shape) > 1:
        X_train, X_test = [
            tf.data.Dataset.from_tensor_slices(
                (tf.constant(Xi), tf.constant(Xo))
            )
            for (Xi, Xo) in ((X_train, X_train), (X_test, X_test))
        ]
    else:
        X_train, X_test = [
            tf.data.Dataset.from_tensor_slices(
                (tf.constant(Xi), tf.constant(Xo))
            )
            for (Xi, Xo) in ((X_train, X_train), (X_test, X_test))
        ]

    X_train, X_test = [
        data.shuffle(buffer_size = 10 * BATCH_SIZE).batch(BATCH_SIZE)
        for data in (X_train, X_test)
    ]
    return (X_train, X_test), (X_train_len, X_test_len)


if __name__ == "__main__":
    EARLY_STOPPING = 10
    EPOCHS = 1000
    BATCH_SIZE = 64
    CONTINUE_DIR = None
    MODELF = lib.old_models.variable_lstm_autoencoder
    INPUT_NPZ = "results/intensities/tracks-cme_split-c1_res.npz"

    _LATENT_DIM = (50,)
    _NORMTYPE = ("dataset",)
    _FEATUREWISE = (False,)
    _BIDIRECTIONAL = (True,)

    for (
        _latent_dim,
        _normtype,
        _featurewise,
        _bidirectional,
    ) in itertools.product(
        _LATENT_DIM, _NORMTYPE, _FEATUREWISE, _BIDIRECTIONAL,
    ):

        X_raw = _get_data(INPUT_NPZ)

        (X_train, X_test), (X_train_len, X_test_len) = _preprocess(
            X_raw,
            per_sample_norm = _normtype,
            per_feature_norm = _featurewise,
            path = INPUT_NPZ,
        )

        n_timesteps = X_raw.shape[1]
        n_features = X_raw.shape[2]

        build_args = [
            n_features,
            n_timesteps,
            _latent_dim,
            _bidirectional,
        ]

        TAG = "_{}".format(MODELF.__name__)
        TAG += "_dim={}".format(_latent_dim)
        TAG += "_norm={}".format(_normtype)
        if _bidirectional in build_args:
            TAG += "_bidir={}".format(_bidirectional)
        TAG += "_prftr={}".format(_featurewise)
        TAG += "_data={}".format(INPUT_NPZ.split("/")[-1])

        model, callbacks, initial_epoch = lib.old_models.model_builder(
            model_dir = CONTINUE_DIR,
            chkpt_tag = TAG,
            patience = EARLY_STOPPING,
            model_build_f = MODELF,
            build_args = build_args,
        )
        model.summary()

        model.fit(
            x = X_train.repeat(),
            validation_data = X_test.repeat(),
            epochs = EPOCHS,
            steps_per_epoch = X_train_len // BATCH_SIZE,
            validation_steps = X_test_len // BATCH_SIZE,
            initial_epoch = initial_epoch,
            callbacks = callbacks,
        )
