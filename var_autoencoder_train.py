import itertools
import os

import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf
from lib.math import ragged_stat
from lib.utils import timeit
import lib.models
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# os.environ["CUDA_VISIBLE_DEVICES"] = "" # disable CUDA device for testing


class SingleBatchGenerator:
    """
    Callable generator to yield single tensor arrays
    """

    def __init__(self, X):
        self.X = X

    def __call__(self):
        for i in range(len(self.X)):
            xi = np.expand_dims(self.X[i], axis=0)
            yield xi, xi


# def ragged_max(arr):
#     """
#     Returns the maximum value of a ragged array
#     """
#     arr = np.array(arr)
#     return np.concatenate(arr).ravel().max()


def _get_data(path):
    """
    Loads all traces
    """
    X = np.load(path, allow_pickle=True)["data"]
    if X.shape[0] < 100:
        raise ValueError("File is suspiciously small. Recheck!")
    return X


@timeit
def _preprocess(X, path, train_size=0.8):
    """
    Preprocess data into tensors and appropriate train/test sets
    """
    X_train, X_test = sklearn.model_selection.train_test_split(
        X, train_size=train_size, random_state=0, shuffle = False
    )

    X_stat = np.row_stack(X_train)

    mu = np.mean(X_stat, axis=(0))
    sg = np.std(X_stat, axis=(0))

    np.savez(
        path[:-4] + "_traintest.npz",
        X_train=X_train,
        X_test=X_test,
        scale=(mu, sg),
    )

    # Normalize after saving
    X_train = np.array([(xi - mu) / sg for xi in X_train])
    X_test = np.array([(xi - mu) / sg for xi in X_test])

    fig, ax = plt.subplots(nrows=5, ncols=5)
    ax = ax.ravel()
    for i in range(len(ax)):
        ax[i].plot(X_test[i])
    plt.show()

    data, lengths = [], []
    for X in X_train, X_test:
        len_X = len(X)
        # Have to be lists of arrays for the generator to work
        X = X.tolist()
        # Wrap as callable generator
        X = SingleBatchGenerator(X)
        # Convert to tensorflow dataset
        X = tf.data.Dataset.from_generator(
            generator=X,
            output_types=(tf.float64, tf.float64),
            output_shapes=((1, None, 2), (1, None, 2)),
        )

        lengths.append(len_X)
        data.append(X)
    return data, lengths


if __name__ == "__main__":
    EARLY_STOPPING = 10
    EPOCHS = 1000
    N_FEATURES = 2
    N_TIMESTEPS = None
    CONTINUE_DIR = None
    MODELF = lib.models.single_lstm_autoencoder
    INPUT_NPZ = "results/intensities/tracks-cme_split-c1_var.npz"

    _LATENT_DIM = (10,) #)64, 128)
    _ACTIVATION = (None,)#("relu", "selu", "elu", "tanh", None)

    for (_latent_dim, _activation) in itertools.product(_LATENT_DIM, _ACTIVATION):

        X_raw = _get_data(INPUT_NPZ)

        build_args = [N_TIMESTEPS, N_FEATURES, _latent_dim, _activation]

        TAG = "_{}".format(MODELF.__name__)
        TAG += "_dim={}".format(_latent_dim)
        TAG += "_variable"
        TAG += "_data={}".format(INPUT_NPZ.split("/")[-1])

        (X_train, X_test), (X_train_len, X_test_len) = _preprocess(
            X=X_raw, path=INPUT_NPZ
        )

        model, callbacks, initial_epoch = lib.models.model_builder(
            model_dir=CONTINUE_DIR,
            chkpt_tag=TAG,
            patience=EARLY_STOPPING,
            model_build_f=MODELF,
            build_args=build_args,
        )
        model.summary()
        model.fit(
            x=X_train.repeat(),
            validation_data=X_test.repeat(),
            epochs=EPOCHS,
            steps_per_epoch=X_train_len,
            validation_steps=X_test_len,
            initial_epoch=initial_epoch,
            callbacks=callbacks,
        )
