import itertools
import os

import numpy as np
import seaborn as sns
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf
import tensorflow.keras.utils
from imblearn.under_sampling import RandomUnderSampler

import lib.math
import lib.models
from lib.tfcustom import VariableTimeseriesBatchGenerator

sns.set_style("darkgrid")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def _get_data(path):
    """
    Loads all traces
    """
    X = np.load(path, allow_pickle=True)["data"]
    if X.shape[0] < 100:
        raise ValueError("File is suspiciously small. Recheck!")
    return X


def _get_labels(path):
    """
    Loads labels
    """
    y = np.load(path, allow_pickle=True)["y"]
    return y


def preprocess(X, y, n_features, max_batch_size, train_size):
    """
    Preprocess data into tensors and appropriate train/test sets
    """
    idx = np.arange(0, len(X), 1)

    (
        X_train,
        X_test,
        idx_train,
        idx_test,
    ) = sklearn.model_selection.train_test_split(X, idx, train_size=train_size)

    if y is not None:
        y = y.reshape(-1, 1)
        y_train, y_test = y[idx_train, ...], y[idx_test, ...]

        ru = RandomUnderSampler()
        _, y_train = ru.fit_resample(X = y_train, y = y_train)
        selected = ru.sample_indices_
        X_train = X_train[selected]

        y_train, y_test = [
            tensorflow.keras.utils.to_categorical(y, num_classes = 2)
            for y in (y_train, y_test)
        ]

    else:
        y_train, y_test = None, None

    mu, sg, *_ = lib.math.array_stats(X)
    X_train, X_test = [
        lib.math.standardize(X, mu, sg) for X in (X_train, X_test)
    ]

    data, lengths, sizes = [], [], []

    for (X, y) in (X_train, y_train), (X_test, y_test):
        # Batch into variable batches to speed up (but see caveats)
        Gen = VariableTimeseriesBatchGenerator(
            X=X.tolist(),
            y=y,
            max_batch_size=max_batch_size,
            shuffle_samples=True,
            shuffle_batches=True,
        )

        steps_per_epoch = Gen.steps_per_epoch
        batch_sizes = Gen.batch_sizes

        X = tf.data.Dataset.from_generator(
            generator=Gen,
            output_types=(tf.float64, tf.int64),
            output_shapes=((None, None, n_features), (None, 2)),
        )
        sizes.append(batch_sizes)
        lengths.append(steps_per_epoch)
        data.append(X)

    info = idx_train, idx_test, mu, sg
    return data, lengths, info


if __name__ == "__main__":
    MODELF = (lib.models.lstm_classifier,)

    INPUT_X = ("data/preprocessed/combined_filt5_var.npz",)
    INPUT_Y = ("results/saved_labels/combined_filt5_var__clust_[2].npz",)

    N_TIMESTEPS = None
    EARLY_STOPPING = 3
    EPOCHS = 100
    TRAIN_TEST_SIZE = 0.8
    BATCH_SIZE = (4,)
    CONTINUE_DIR = None

    LATENT_DIM = (128,)
    KEEP_ONLY = (0, None)

    # Add iterables here
    for (
        _input_x,
        _input_y,
        _batch_size,
        _latent_dim,
        _modelf,
        _keep_only,
    ) in itertools.product(
        INPUT_X, INPUT_Y, BATCH_SIZE, LATENT_DIM, MODELF, KEEP_ONLY,
    ):

        X_raw = _get_data(_input_x)
        y = _get_labels(_input_y)

        if _keep_only is not None:
            X_raw = np.array([x[:, _keep_only].reshape(-1, 1) for x in X_raw])

        N_FEATURES = X_raw[0].shape[-1]

        # Pre-define loss so it gets compiled in the graph
        KL_LOSS = tf.Variable(0.0)

        build_args = [
            N_TIMESTEPS,
            N_FEATURES,
            _latent_dim,
        ]

        TAG = "_{}".format(_modelf.__name__)
        TAG += "_data={}".format(_input_x.split("/")[-1])  # input data
        TAG += "_dim={}".format(_latent_dim)  # LSTM latent dimension
        TAG += "_bat={}".format(_batch_size)  # batch size

        if _keep_only is not None:
            TAG += "_single={}".format(
                _keep_only
            )  # Keep only one of the features

        data, lengths, info = preprocess(
            X=X_raw,
            y=y,
            n_features=N_FEATURES,
            max_batch_size=_batch_size,
            train_size=TRAIN_TEST_SIZE,
        )

        (Xy_train, Xy_test), (train_steps, test_steps) = data, lengths

        model, callbacks, initial_epoch, model_dir = lib.models.model_builder(
            model_dir=CONTINUE_DIR,
            tag =TAG,
            weights_only=False,
            patience=EARLY_STOPPING,
            model_build_f=_modelf,
            build_args=build_args,
        )

        model.summary()
        model.fit(
            x=Xy_train.repeat(),
            validation_data=Xy_test.repeat(),
            epochs=EPOCHS,
            steps_per_epoch=train_steps,
            validation_steps=test_steps,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
        )

        # Save indices and normalization values to the newly created model directory
        np.savez(os.path.join(model_dir, "info.npz"), info=info)
