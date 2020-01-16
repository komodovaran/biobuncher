import streamlit as st
from multiprocessing import Pool, cpu_count
import pandas as pd
import numpy as np
import parmap
from lib.utils import timeit
np.random.seed(1)

@timeit
def groupby_parallel_apply(grouped_df, func, option):
    """
    Runs Pandas groupby functions in parallel.
    Set concat = True to concatenate subgroups to a new dataframe
    """
    groups = [group for _, group in grouped_df]

    threads = cpu_count() - 1

    if option == "a":
        with Pool(threads) as p:
            results = [r for r in p.imap_unordered(func, groups)]
    elif option == "b":
        with Pool(threads) as p:
            results = p.map(func, groups)
    elif option == "c":
        results = parmap.map(func, groups)
    else:
        raise ValueError
    return results


def applyf(group):
    return len(group)


df = pd.DataFrame({"a" : np.random.randint(0, 1, 10000),
                   "id": np.random.randint(1, 10000, 10000)})

result = groupby_parallel_apply(df.groupby(["id"]), applyf, option = "a")
result = groupby_parallel_apply(df.groupby(["id"]), applyf, option = "b")
result = groupby_parallel_apply(df.groupby(["id"]), applyf, option = "c")