import itertools
import os
import re

import numpy as np
import seaborn as sns
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf

import lib.math
import lib.models
from lib.tfcustom import (
    AnnealingVariableCallback,
    VariableTimeseriesBatchGenerator,
)

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


def _preprocess(X, n_features, max_batch_size, train_size):
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

    mu, sg, *_ = lib.math.array_stats(X)
    X_train, X_test = [
        lib.math.standardize(X, mu, sg) for X in (X_train, X_test)
    ]

    data, lengths, sizes = [], [], []
    for X in X_train, X_test:
        # Batch into variable batches to speed up (but see caveats)
        Xi = VariableTimeseriesBatchGenerator(
            X=X.tolist(),
            max_batch_size=max_batch_size,
            shuffle_samples=True,
            shuffle_batches=True,
        )

        steps_per_epoch = Xi.steps_per_epoch
        batch_sizes = Xi.batch_sizes

        X = tf.data.Dataset.from_generator(
            generator=Xi,
            output_types=(tf.float64, tf.float64),
            output_shapes=((None, None, n_features), (None, None, n_features)),
        )
        sizes.append(batch_sizes)
        lengths.append(steps_per_epoch)
        data.append(X)

    info = idx_train, idx_test, mu, sg
    return data, lengths, info


if __name__ == "__main__":
    MODELF = (lib.models.lstm_vae_bidir,)

    INPUT_NPZ = ("data/preprocessed/combined_filt5_var.npz",)

    N_TIMESTEPS = None
    EARLY_STOPPING = 3
    EPOCHS = 100
    TRAIN_TEST_SIZE = 0.8
    BATCH_SIZE = (64,)
    CONTINUE_DIR = None

    LATENT_DIM = (64,)
    ACTIVATION = (
        None,
        "relu",
    )
    ZDIM = (64,)
    EPS = (0.001,)
    SINGLE_FEATURE = (None,)
    ANNEAL_TIME = (20,)

    # Add iterables here
    for (
        _input_npz,
        _batch_size,
        _latent_dim,
        _activation,
        _eps,
        _zdim,
        _anneal_time,
        _single_feature,
        _modelf,
    ) in itertools.product(
        INPUT_NPZ,
        BATCH_SIZE,
        LATENT_DIM,
        ACTIVATION,
        EPS,
        ZDIM,
        ANNEAL_TIME,
        SINGLE_FEATURE,
        MODELF,
    ):

        X_raw = _get_data(_input_npz)

        if _single_feature is not None:
            X_raw = np.array(
                [x[:, _single_feature].reshape(-1, 1) for x in X_raw]
            )

        N_FEATURES = X_raw[0].shape[-1]

        # Pre-define loss so it gets compiled in the graph
        KL_LOSS = tf.Variable(0.0)

        build_args = [
            N_TIMESTEPS,
            N_FEATURES,
            _latent_dim,
            KL_LOSS,
            _eps,
            _zdim,
            _activation,
        ]

        TAG = "_{}".format(_modelf.__name__)
        TAG += "_data={}".format(_input_npz.split("/")[-1])  # input data
        TAG += "_dim={}".format(_latent_dim)  # LSTM latent dimension
        TAG += "_act={}".format(_activation)  # activation function
        TAG += "_bat={}".format(_batch_size)  # batch size
        TAG += "_eps={}_zdim={}_anneal={}".format(
            _eps, _zdim, _anneal_time
        )  # vae parameters
        if _single_feature is not None:
            TAG += "_single={}".format(
                _single_feature
            )  # Keep only one of the features

        data, lengths, info = _preprocess(
            X=X_raw,
            n_features=N_FEATURES,
            max_batch_size=_batch_size,
            train_size=TRAIN_TEST_SIZE,
        )
        (X_train, X_test), (X_train_steps, X_test_steps) = data, lengths

        model, callbacks, initial_epoch, model_dir = lib.models.model_builder(
            model_dir=CONTINUE_DIR,
            chkpt_tag=TAG,
            weights_only=False,
            patience=EARLY_STOPPING,
            model_build_f=_modelf,
            build_args=build_args,
        )

        if re.search(string=_modelf.__name__, pattern="vae") is not None:
            print("re-initialized KL loss")
            KL_LOSS.assign(value=0.0)
            callbacks.append(
                AnnealingVariableCallback(
                    var=KL_LOSS,
                    anneal_over_n_epochs=_anneal_time,
                    anneals_starts_at=2,
                )
            )

        model.summary()
        model.fit(
            x=X_train.repeat(),
            validation_data=X_test.repeat(),
            epochs=EPOCHS,
            steps_per_epoch=X_train_steps,
            validation_steps=X_test_steps,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
        )

        # Save indices and normalization values to the newly created model directory
        np.savez(os.path.join(model_dir, "info.npz"), info=info)
