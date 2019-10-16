import itertools
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd


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


def timeit(method):
    """Decorator to time functions and methods for optimization"""

    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print("'{}' {:.2f} ms".format(method.__name__, (te - ts) * 1e3))
        return result

    return timed


@timeit
def groupby_parallel_apply(grouped_df, func):
    """
    Runs Pandas groupby functions in parallel
    """
    with Pool(cpu_count()) as p:
        results_list = p.map(func, [group for _, group in grouped_df])
    return pd.concat(results_list)