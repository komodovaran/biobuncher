import numpy as np
import sklearn.utils
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Activation, Add, BatchNormalization, Conv1D, Lambda, Layer, MaxPooling1D
import lib.utils

class KLDivergenceLayer(Layer):
    """
    Takes the mu and sigma, and the KL loss variable (compiled in graph) and
    calculates the loss, adds it to the model, then returns the variables.
    Keep all values in the __call__, otherwise the layer will break because
    it lacks other components required
    """
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var, kl_weight = inputs

        # kl loss for batch
        kl_batch = -0.5 * K.sum(
            1 + log_var - K.square(mu) - K.exp(log_var), axis=-1
        )

        # Add the mean loss for the batch
        self.add_loss(kl_weight * K.mean(kl_batch), inputs=inputs)

        # Return the inputs
        return mu, log_var


class SingleBatchGenerator:
    """
    Callable generator to yield single samples as tensors
    """

    def __init__(self, X):
        self.X = X

    def __call__(self):
        for i in range(len(self.X)):
            xi = np.expand_dims(self.X[i], axis=0)
            yield xi, xi


class VariableTimeseriesBatchGenerator:
    """
    Callable generator to yield batches of same-length timeseries. Provide
    indices to keep track of where each sample ends up.
    """

    def __init__(
        self, X, max_batch_size, shuffle_samples, shuffle_batches, indices=None, y = None
    ):
        if indices is None:
            indices = np.arange(0, len(X), 1)
        self.batch_by_length(
            X, max_batch_size, shuffle_samples, shuffle_batches, indices, y
        )

        if y is None:
            self.y = None

    @staticmethod
    def chunk(l, n):
        """
        Breaks a list into chunk size of at most n
        """
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def batch_by_length(
        self, X, max_batch_size, shuffle_samples, shuffle_batches, indices, y = None
    ):
        """
        Batches a list of variable-length samples into equal-sized tensors
        to speed up training
        """
        # Shuffle samples before batching
        if shuffle_samples:
            if y is not None:
                X, y, indices = sklearn.utils.shuffle(X, y, indices)
            else:
                X, indices = sklearn.utils.shuffle(X, indices)

        lengths = [len(xi) for xi in X]
        length_brackets = np.unique(lengths)

        # initialize empty batches for each length
        X_batches = [[] for _ in range(len(length_brackets))]
        idx_batches = [[] for _ in range(len(length_brackets))]
        y_batches = [[] for _ in range(len(length_brackets))]

        if len(X_batches) != len(length_brackets):
            raise ValueError

        # Go through each sample and find out where it belongs
        for i in range(len(X)):
            xi = X[i]
            idx = indices[i]

            # Find out which length bracket it belongs to
            (belongs_to,) = np.where(len(xi) == length_brackets)[0]

            # Place sample there
            X_batches[belongs_to].append(xi)
            idx_batches[belongs_to].append(idx)

            if y is not None:
                yi = y[i]
                y_batches[belongs_to].append(yi)

        # Break into smaller chunks so that a batch is at most max_batch_size
        dataset = []
        index = []
        labels = []
        for j in range(len(X_batches)):
            sub_batch = list(self.chunk(X_batches[j], max_batch_size))
            sub_idx = list(self.chunk(idx_batches[j], max_batch_size))

            if y is not None:
                sub_y = list(self.chunk(y_batches[j], max_batch_size))
            else:
                sub_y = []

            for k in range(len(sub_batch)):
                dataset.append(sub_batch[k])
                index.append(sub_idx[k])
                if y is not None:
                    labels.append(sub_y[k])

        # Now transform each batch to a tensor
        dataset = [np.array(batch) for batch in dataset]
        index_set = [np.array(index_batch) for index_batch in index]
        if y is not None:
            labels = [np.array(label_batch) for label_batch in labels]

        # Shuffle batches of different lengths (not individual samples)
        if shuffle_batches:
            if y is not None:
                dataset, index_set, labels = sklearn.utils.shuffle(dataset, index_set, labels)
            else:
                dataset, index_set = sklearn.utils.shuffle(dataset, index_set)

        for b in dataset:
            if len(b) > max_batch_size:
                raise ValueError

        # Set steps per epoch for dataset to get the right number of steps
        self.batches = dataset
        self.indices = np.array(lib.utils.flatten_list(index_set))
        self.steps_per_epoch = len(dataset)
        self.batch_sizes = [len(batch) for batch in dataset]
        self.y = labels

    def __call__(self):
        """
        Returns a single batch of samples per call, as (input, output) tuple,
        for sample reconstruction by autoencoder
        """
        for i in range(len(self.batches)):
            xi = self.batches[i]
            if self.y is not None:
                yi = self.y[i]
                yield xi, yi
            else:
                yield xi, xi


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
            x = MaxPooling1D(1, padding= "same")(x)
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


class AnnealingVariableCallback(Callback):
    """
    Slowly increases a variable from 0 to 1 over a given amount of epochs
    """
    def __init__(self, var, anneals_starts_at, anneal_over_n_epochs):
        super(AnnealingVariableCallback, self).__init__()
        self.var = var
        self.anneal_starts_at = anneals_starts_at
        self.anneal_over_n_epochs = anneal_over_n_epochs

    def on_epoch_end(self, epoch, logs={}):
        if epoch > self.anneal_starts_at:
            new_var = min(
                K.get_value(self.var) + (1.0 / self.anneal_over_n_epochs), 1.0,
            )
            K.set_value(self.var, new_var)


def mse_loss(inputs, outputs):
    return K.mean(K.square(inputs - outputs))


def kullback_leibler_loss(z_mean, z_log_var):
    """
    We call it log variance, because it will be exponentiated when put into the equation.
    This helps with numerical stability, as the model only has to focus on the log variance.
    """
    # kl = -0.5 * K.sum(1 + z_log_var - z_mean ** 2 - tf.exp(z_log_var), axis = -1)
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return K.sum(kl_loss)


def gelu(x):
    return 0.5 * x * (1 + K.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


def nll(y_true, y_pred):
    """ Negative log likelihood (Bernoulli). """
    # keras.losses.binary_crossentropy gives the mean
    # over the last axis. we require the sum
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)



class KSparse(Layer):
    '''k-sparse Keras layer.

    # Arguments
        sparsity_levels: np.ndarray, sparsity levels per epoch calculated by `calculate_sparsity_levels`
    '''

    def __init__(self, sparsity_levels, **kwargs):
        self.sparsity_levels = tf.constant(sparsity_levels, dtype = tf.int32)
        self.k = tf.Variable(initial_value = self.sparsity_levels[0])
        self.uses_learning_phase = True
        super().__init__(**kwargs)

    def call(self, inputs, mask = None):
        def sparse():
            kth_smallest = tf.sort(inputs)[..., K.shape(inputs)[-1] - 1 - self.k]
            return inputs * K.cast(K.greater(inputs, kth_smallest[:, None]), K.floatx())

        return K.in_train_phase(sparse, inputs)

    def get_config(self):
        config = {'k': self.k}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class UpdateSparsityLevel(Callback):
    """
    Update sparsity level at the beginning of each epoch.
    """
    def on_epoch_begin(self, epoch, logs = {}):
        l = self.model.get_layer('KSparse')
        K.set_value(l.k, l.sparsity_levels[epoch])


def calculate_sparsity_levels(initial_sparsity, final_sparsity, n_epochs):
    """
    Calculate sparsity levels per epoch. Initial sparsity should be slightly
    lower than embedding size

    # Arguments
        initial_sparsity: int
        final_sparsity: int
        n_epochs: int
    """
    return np.hstack((np.linspace(initial_sparsity, final_sparsity, n_epochs // 2, dtype = np.int),
                      np.repeat(final_sparsity, (n_epochs // 2) + 1)))[:n_epochs]