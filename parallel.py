from multiprocessing import Pool, cpu_count
from time import time
import numpy as np
import pandas as pd

np.random.seed(1)

def applyf(group):
    return len(group)

n = 100000

df = pd.DataFrame({"a" : np.random.randint(0, 1, n),
                   "id": np.random.randint(1, n**2, n)})

start = time()

with Pool(cpu_count()) as p:
    results = p.map(applyf, [group for _, group in df.groupby("id")])

stop = time()

print("elapsed: {:.1f} ms".format((stop - start)*1000))