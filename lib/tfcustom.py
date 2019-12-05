import numpy as np
import sklearn.utils
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.keras.callbacks import Callback
from tensorflow_core.python.keras.engine import Layer
from tensorflow_core.python.keras.layers import Lambda, MaxPooling1D, Conv1D, BatchNormalization, Activation, Add


class KLDivergenceLayer(Layer):

    """ Identity transform layer that adds KL divergence
    to the final model loss.
    """

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mu, log_var = inputs
        kl_batch = -0.5 * K.sum(
            1 + log_var - K.square(mu) - K.exp(log_var), axis=-1
        )
        self.add_loss(K.mean(kl_batch), inputs=inputs)

        return inputs


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


class VariableBatchGenerator:
    """
    Callable generator to yield single tensor arrays
    """

    def __init__(self, X, max_batch_size, shuffle):
        self.batch_by_length(X, max_batch_size, shuffle)

    @staticmethod
    def chunks(l, n):
        """
        Breaks a list into chunk size of at most n
        """
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def batch_by_length(self, X, max_batch_size, shuffle):
        """
        Batches a list of variable-length samples into equal-sized tensors
        to speed up training
        """
        # First, shuffle all samples
        if shuffle:
            X = sklearn.utils.shuffle(X)

        lengths = [len(xi) for xi in X]
        length_brackets = np.unique(lengths)

        # initialize empty batches for each length
        length_batches = [[] for _ in range(len(length_brackets))]
        if not len(length_batches) == len(length_brackets):
            raise ValueError

        # Go through each sample and find out where it belongs
        for xi in X:
            # Find out which length bracket it belongs to
            (idx,) = np.where(len(xi) == length_brackets)
            idx = idx[0]

            # Place sample there
            length_batches[idx].append(xi)

        # Break into smaller chunks so that a batch is at most max_batch_size
        dataset = []
        for b in length_batches:
            sub_divided = list(
                self.chunks(b, max_batch_size)
            )  # multiple batches
            for s in sub_divided:
                dataset.append(s)

        # Now transform each batch to a tensor
        dataset = [np.array(batch) for batch in dataset]

        # Shuffle batches
        if shuffle:
            dataset = sklearn.utils.shuffle(dataset)

        for b in dataset:
            if len(b) > max_batch_size:
                raise ValueError

        # Set steps per epoch for dataset to get the right number of steps
        self.X = dataset
        self.steps_per_epoch = len(dataset)
        self.batch_sizes = [len(batch) for batch in dataset]

    def __call__(self):
        for i in range(len(self.X)):
            xi = self.X[i]
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