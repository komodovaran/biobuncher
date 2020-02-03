import lib.math
import parmap
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def process(xi):
    ts = lib.math.resample_timeseries(xi, new_length = 200)
    return ts

X = np.load("data/preprocessed/combined_filt20_var.npz", allow_pickle = True)["data"]

mp_results = parmap.map(process, tqdm(X))
newX = np.array(mp_results)

print(newX.shape)

fig, ax = plt.subplots(nrows = 3, ncols = 2)
for i in range(3):
    ax[i, 0].plot(X[i])
    ax[i, 1].plot(newX[i])
plt.show()

np.savez("data/preprocessed/combined_filt20.npz", data = newX)