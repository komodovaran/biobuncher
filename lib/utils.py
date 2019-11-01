import itertools
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd


def timeit(method):
    """Decorator to time functions and methods for optimization"""

    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print("'{}' {:.2f} ms".format(method.__name__, (te - ts) * 1e3))
        return result

    return timed


def flatten_list(input_list, as_array=False):
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


@timeit
def groupby_parallel_apply(grouped_df, func, concat=True):
    """
    Runs Pandas groupby functions in parallel.
    Set concat = True to concatenate subgroups to a new dataframe
    """
    with Pool(cpu_count()) as p:
        results_list = p.map(func, [group for _, group in grouped_df])
    if concat:
        results_list = pd.concat(results_list)
    return results_list


#


def pairwise(array):
    """Unpacks elements of an array (1,2,3,4...) into pairs, i.e. (1,2), (3,4), ..."""
    return zip(array[0::2], array[1::2])


def batch_to_numpy(tf_data, include_label=False, n_batches=-1):
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
        x, y = [np.concatenate(d, axis=0) for d in (x, y)]
        return x, y
    else:
        x = [xi.numpy() for xi, _ in tf_data.take(n_batches)]
        x = np.concatenate(x, axis=0)
        return x


def rle(arr):
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


def remove_zero_padding(arr_true, arr_pred=None, padding="before"):
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
    runlen, *_ = rle(arr_true)

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


def ts_to_stationary(df, groupby=None):
    """
    Differences over the timeseries datafram
    to make it stationary. Pads beginning.
    """
    time_col = df.pop("time")
    id_col = df["id"]
    if groupby is not None:
        g = df.groupby("id")
    else:
        g = df

    g = g.diff().replace(np.nan, 0)
    g.join()
    
    
    return g
