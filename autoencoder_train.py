import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import sklearn.model_selection
import sklearn.preprocessing
import tensorflow as tf

import lib.math
import lib.models
from lib.tfcustom import VariableBatchGenerator, AnnealingVariableCallback
from lib.plotting import sanity_plot
from lib.utils import timeit
from tensorflow.keras import backend as K

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# os.environ["CUDA_VISIBLE_DEVICES"] = "" # disable CUDA device for testing


def _get_data(path):
    """
    Loads all traces
    """
    X = np.load(path, allow_pickle = True)["data"]
    if X.shape[0] < 100:
        raise ValueError("File is suspiciously small. Recheck!")
    return X


@timeit
def _preprocess(X, path, max_batch_size, train_size):
    """
    Preprocess data into tensors and appropriate train/test sets
    """
    idx = np.arange(0, len(X), 1)

    (
        X_train,
        X_test,
        idx_train,
        idx_test,
    ) = sklearn.model_selection.train_test_split(X, idx, train_size = train_size)

    # sanity_plot(X_test, "before normalization")

    mu, sg, *_ = lib.math.array_stats(X)
    X_train, X_test = [
        lib.math.standardize(X, mu, sg) for X in (X_train, X_test)
    ]

    np.savez(
        path[:-4] + "_traintest.npz",
        X_train = X_train,
        X_test = X_test,
        idx_train = idx_train,
        idx_test = idx_test,
        scale = (mu, sg),
    )

    # sanity_plot(X_test, "after normalization")

    data, lengths, sizes = [], [], []
    for X in X_train, X_test:
        # Batch into variable batches to speed up
        Xi = VariableBatchGenerator(
            X = X.tolist(), max_batch_size = max_batch_size, shuffle = True
        )

        steps_per_epoch = Xi.steps_per_epoch
        batch_sizes = Xi.batch_sizes

        X = tf.data.Dataset.from_generator(
            generator = Xi,
            output_types = (tf.float64, tf.float64),
            output_shapes = ((None, None, 2), (None, None, 2)),
        )
        sizes.append(batch_sizes)
        lengths.append(steps_per_epoch)
        data.append(X)

    # Take a single batch and plot
    # for n, (Xi, _) in enumerate(data[1]):
    #     sanity_plot(Xi.numpy(), "batch {}".format(n))
    #     if n == 3:
    #         break

    fig, ax = plt.subplots(ncols = 2)
    ax[0].hist(sizes[0], label = "batch sizes train")
    ax[1].hist(sizes[1], label = "batch sizes test")
    for a in ax:
        a.legend(loc = "upper left")
    plt.savefig("plots/variable_batch_{}.pdf".format(max_batch_size))
    plt.show()
    return data, lengths


if __name__ == "__main__":
    EARLY_STOPPING = 3
    EPOCHS = 1000
    N_FEATURES = 2
    MAX_BATCH_SIZE = 32
    BATCH_SIZE = [4, 12, 32, 64]
    TRAIN_TEST_SIZE = 0.8
    N_TIMESTEPS = None
    CONTINUE_DIR = None
    LATENT_DIM = 32
    ACTIVATION = "elu"
    MERGE = "mul"

    MODELF = lib.models.lstm_autoencoder
    INPUT_NPZ = (
        # "results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8_var.npz",
        # "results/intensities/tracks-CLTA-TagRFP_EGFP-Aux1-A7D2_var.npz",
        # "results/intensities/tracks-CLTA-TagRFP_EGFP-Aux1-A7D2_EGFP-Gak-F6_var.npz",  # smallest
        "results/intensities/tracks-cme_var.npz",
    )

    # _ZDIM = (2, 3)
    # _EPS = (0.1, 0.5, 1)

    # Pre-define loss so it gets compiled in the graph
    kl_loss = K.variable(0.0)

    iters = itertools.product(INPUT_NPZ, BATCH_SIZE)
    for (_input_npz, _batch_size) in iters:
        build_args = [N_TIMESTEPS, N_FEATURES, LATENT_DIM]

        TAG = "_{}".format(MODELF.__name__)
        TAG += "_dim={}".format(LATENT_DIM)
        TAG += "_activ={}".format(ACTIVATION)
        # TAG += "_eps={}_zdim={}".format(_eps, _zdim)
        TAG += "_merge={}".format(MERGE)
        TAG += "_batch={}"
        TAG += "_data={}".format(_input_npz.split("/")[-1])

        X_raw = _get_data(_input_npz)

        (X_train, X_test), (X_train_steps, X_test_steps) = _preprocess(
            X = X_raw,
            path = _input_npz,
            max_batch_size = _batch_size,
            train_size = TRAIN_TEST_SIZE,
        )

        K.set_value(kl_loss, 0.0)

        model, callbacks, initial_epoch = lib.models.model_builder(
            model_dir = CONTINUE_DIR,
            chkpt_tag = TAG,
            patience = EARLY_STOPPING,
            model_build_f = MODELF,
            build_args = build_args,
        )

        # avc = AnnealingVariableCallback(
        #     var=kl_loss, anneal_over_n_epochs=20, anneals_starts_at=30)
        # callbacks.append(avc)

        model.summary()
        model.fit(
            x = X_train.repeat(),
            validation_data = X_test.repeat(),
            epochs = EPOCHS,
            steps_per_epoch = X_train_steps,
            validation_steps = X_test_steps,
            initial_epoch = initial_epoch,
            callbacks = callbacks,
        )
