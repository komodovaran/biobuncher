import ray
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
import time
ray.init()

def timeit(method):
    """Decorator to time functions and methods for optimization"""
    def timed(*args, **kwargs):
        ts = time.time()
        result = method(*args, **kwargs)
        te = time.time()
        print("'{}' {:.2f} ms".format(method.__name__, (te - ts) * 1e3))
        return result
    return timed

def groupby_parallel_apply(grouped_df, func, column):
    """Parallel groupby apply"""
    groups = [group[column] for _, group in grouped_df]
    with Pool(cpu_count()) as p:
        results = p.map(func, groups)
    results = pd.concat(results, sort = False)
    return results

n = 10000
df = pd.DataFrame({"a" : np.random.normal(0, 1, n),
                   "id" : np.random.randint(0, n//10, n)})

@ray.remote(num_cpus = cpu_count())
def f(x):
    return x * x

def g(x):
    return x * x


@timeit
def _1():
    futures = []
    for _, group in df.groupby("id"):
        result = f.remote(group["a"])
        futures.append(result)
    return pd.concat(ray.get(futures), sort = False)

@timeit
def _2():
    return groupby_parallel_apply(grouped_df = df.groupby("id"), func = g, column = "a")

print(np.alltrue(_1() == _2()))