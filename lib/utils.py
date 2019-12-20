import itertools
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import parmap
from tensorflow.python.keras.utils import Sequence
import datetime

def time_now():
    """
    Returns the current YMD-HM formatted date
    """
    return datetime.datetime.now().strftime("%Y%m%d-%H%M")

def est_proc():
    """
    Estimates a good number of processes for multithreading, due to overhead
    of throwing too many CPUs at a problem
    """
    c = mp.cpu_count()
    if c <= 8:
        return c
    else:
        return c // 4


def timeit(method):
    """Decorator to time functions and methods for optimization"""

    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print("'{}' {:.2f} ms".format(method.__name__, (te - ts) * 1e3))
        return result

    return timed


def flatten_list(input_list, as_array = False):
    """
    Parameters
    ----------
    input_list:
        flattens python list of lists to single list
    as_array:
        True returns numpy array, False returns iterable python list
    Returns
    -------
    flattened list in the chosen format
    """
    flat_lst = list(itertools.chain.from_iterable(input_list))
    if as_array:
        flat_lst = np.array(flat_lst)
    return flat_lst


def ravel_ragged(array):
    """
    Unravels ragged numpy arrays (arrays with sub-arrays of different sizes)
    """
    return np.ravel(list(itertools.chain(*array)))


def groupby_parallel_apply(
    grouped_df, func, f_args = None, concat = True, n_jobs = -1
):
    """
    Runs Pandas groupby functions in parallel.
    Set concat = True to concatenate subgroups to a new dataframe
    """
    if n_jobs == -1:
        n_jobs = est_proc()

    groups = [group for _, group in grouped_df]

    if f_args is not None:
        results = parmap.map(func, groups, *f_args, pm_processes = n_jobs)
    else:
        results = parmap.map(func, groups, pm_processes = n_jobs)

    if concat:
        results = pd.concat(results, sort = False)
    return results


def initialize_df_columns(df, new_columns):
    """
    Initializes a list of new columns with zeros
    """
    return df.assign(**{c: 0 for c in new_columns})


def pairwise(array):
    """Unpacks elements of an array (1,2,3,4...) into pairs, i.e. (1,2), (3,4), ..."""
    return zip(array[0::2], array[1::2])


def batch_to_numpy(tf_data, include_label = False, n_batches = -1):
    """
    Parameters
    ----------
    tf_data
        tf.Dataset
    n_batches
        Number of batches to take (size defined by dataset)
        'n_batches = -1' loads the entire set
    Returns
    -------
    Numpy array
    """
    if include_label:
        x, y = [], []
        for xi, yi in tf_data.take(n_batches):
            x.append(xi)
            y.append(yi)
        x, y = [np.concatenate(d, axis = 0) for d in (x, y)]
        return x, y
    else:
        x = [xi.numpy() for xi, _ in tf_data.take(n_batches)]
        x = np.concatenate(x, axis = 0)
        return x


def run_length_encoding(arr):
    """
    run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy
    returns: tuple (runlengths, startpositions, values)
    """
    ia = np.array(arr)  # force numpy

    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = np.array(ia[1:] != ia[:-1])  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        runlen = np.diff(np.append(-1, i))  # run lengths
        pos = np.cumsum(np.append(0, runlen))[:-1]  # positions
        no_adj_duplicates = ia[i]
        return runlen, pos, no_adj_duplicates


def remove_zero_padding(arr_true, arr_pred = None, padding = "before"):
    """
    Removes zero-padding to display predictions

    Parameters
    ----------
    arr_true:
        True array (has zero padding)
    arr_pred:
        Predicted array
    padding:
        Whether padding comes 'before' or 'after' the actual array

    Returns
    -------
    (arr_true, arr_pred) with padded indices removed
    """
    runlen, *_ = run_length_encoding(arr_true)

    if arr_pred is not None:
        if len(arr_true) != len(arr_pred):
            raise ValueError("Arrays must be equal length")
        arrays = arr_true, arr_pred
    else:
        arrays = (arr_true,)

    if padding == "before":
        cut = runlen[0]
        s = np.index_exp[cut:]
    elif padding == "after":
        cut = runlen[-1]
        s = np.index_exp[:-cut]
    else:
        raise ValueError
    return [arr[s] for arr in arrays]


def sample_groups(df, size, by):
    """
    Parameters
    ----------
    df:
        DataFrame to sample
    size:
        Number of samples
    by:
        List of DataFrame.groupby() keys

    Returns
    -------
    Subsampled DataFrame with the same properties
    """
    g = df.groupby(by)
    n_groups = np.arange(g.ngroups)
    np.random.shuffle(n_groups)
    sample = df[g.ngroup().isin(n_groups[:size])]
    return sample


def ts_tensor_to_df(X):
    """
    Converts timeseries tensor of shape (n_samples, n_timesteps, n_features) to
    DataFrame with 'time' and 'id' columns.
    """
    n_samples, n_timesteps, n_features = X.shape
    X = X.reshape(-1, n_features)

    df = pd.DataFrame(X)
    df["time"] = df.index % n_timesteps
    df["id"] = np.repeat(range(n_samples), n_timesteps)
    return df


def ts_to_stationary(df, groupby = None):
    """
    Differences over the timeseries datafram
    to make it stationary. Pads beginning.
    """
    # Avoid breaking these columns in groupby
    time_col = df.pop("time")
    id_col = df["id"]

    # Group by ID
    g = df.groupby("id") if groupby is not None else df
    g = g.diff().replace(np.nan, 0, inplace = False)

    # Add columns back
    g["time"] = time_col
    g["id"] = id_col
    return g


def sample_max_normalize_3d(X, squeeze = True):
    """
    Sample-wise max-value normalization of 3D array (tensor).
    This is not feature-wise normalization, to keep the ratios between extracted_features intact!
    """
    if len(X.shape) == 2:
        X = X[np.newaxis, :, :]
    assert len(X.shape) == 3
    arr_max = np.max(X, axis = (1), keepdims = True)
    X = X * (1 / arr_max)

    if squeeze:
        return np.squeeze(X)
    else:
        return X


def remove_parent_dir(path, n):
    """
    Removes n directories from the left side of the path
    """
    return Path(*Path(path).parts[n + 1:]).as_posix()


class BucketedSequence(Sequence):
    """
    A Keras Sequence (dataset reader) of input sequences read in bucketed bins.
    Assumes all inputs are already padded using `pad_sequences` (where padding
    is prepended).
    https://github.com/tbennun/keras-bucketed-sequence
    """

    @staticmethod
    def _roundto(val, batch_size):
        return int(math.ceil(val / batch_size)) * batch_size

    def __init__(self, num_buckets, batch_size, seq_lengths, x_seq, y):
        self.batch_size = batch_size
        # Count bucket sizes
        bucket_sizes, bucket_ranges = np.histogram(
            seq_lengths, bins = num_buckets
        )

        # Obtain the (non-sequence) shapes of the inputs and outputs
        input_shape = (1,) if len(x_seq.shape) == 2 else x_seq.shape[2:]
        output_shape = (1,) if len(y.shape) == 1 else y.shape[1:]

        # Looking for non-empty buckets
        actual_buckets = [
            bucket_ranges[i + 1] for i, bs in enumerate(bucket_sizes) if bs > 0
        ]
        actual_bucketsizes = [bs for bs in bucket_sizes if bs > 0]
        bucket_seqlen = [int(math.ceil(bs)) for bs in actual_buckets]
        num_actual = len(actual_buckets)
        print("Training with %d non-empty buckets" % num_actual)

        self.bins = [
            (
                np.ndarray([bs, bsl] + list(input_shape), dtype = x_seq.dtype),
                np.ndarray([bs] + list(output_shape), dtype = y.dtype),
            )
            for bsl, bs in zip(bucket_seqlen, actual_bucketsizes)
        ]
        assert len(self.bins) == num_actual

        # Insert the sequences into the bins
        bctr = [0] * num_actual
        for i, sl in enumerate(seq_lengths):
            for j in range(num_actual):
                bsl = bucket_seqlen[j]
                if sl < bsl or j == num_actual - 1:
                    self.bins[j][0][bctr[j], :bsl] = x_seq[i, -bsl:]
                    self.bins[j][1][bctr[j], :] = y[i]
                    bctr[j] += 1
                    break

        self.num_samples = x_seq.shape[0]
        self.dataset_len = int(
            sum([math.ceil(bs / self.batch_size) for bs in actual_bucketsizes])
        )
        self._permute()

    def _permute(self):
        # Shuffle bins
        random.shuffle(self.bins)

        # Shuffle bin contents
        for i, (xbin, ybin) in enumerate(self.bins):
            index_array = np.random.permutation(xbin.shape[0])
            self.bins[i] = (xbin[index_array], ybin[index_array])

    def on_epoch_end(self):
        self._permute()

    def __len__(self):
        """ Returns the number of minibatches in this sequence. """
        return self.dataset_len

    def __getitem__(self, idx):
        idx_begin, idx_end = self.batch_size * idx, self.batch_size * (idx + 1)

        # Obtain bin index
        for i, (xbin, ybin) in enumerate(self.bins):
            rounded_bin = self._roundto(xbin.shape[0], self.batch_size)
            if idx_begin >= rounded_bin:
                idx_begin -= rounded_bin
                idx_end -= rounded_bin
                continue

            # Found bin
            idx_end = min(xbin.shape[0], idx_end)  # Clamp to end of bin

            return xbin[idx_begin:idx_end], ybin[idx_begin:idx_end]

        raise ValueError("out of bounds")


def get_index(*args, index):
    """
    Returns indexed args.
    """
    return [a[index] for a in args]



def count_adjacent_values(arr):
    """
    Returns start index and length of segments of equal values.

    Example for plotting several axvspans:
    --------------------------------------
    adjs, lns = lib.count_adjacent_true(score)
    t = np.arange(1, len(score) + 1)

    for ax in axes:
        for starts, ln in zip(adjs, lns):
            alpha = (1 - np.mean(score[starts:starts + ln])) * 0.15
            ax.axvspan(xmin = t[starts], xmax = t[starts] + (ln - 1), alpha = alpha, color = "red", zorder = -1)
    """
    arr = arr.ravel()

    n = 0
    same = [(g, len(list(l))) for g, l in itertools.groupby(arr)]
    starts = []
    lengths = []
    for v, l in same:
        _len = len(arr[n : n + l])
        _idx = n
        n += l
        lengths.append(_len)
        starts.append(_idx)
    return starts, lengths