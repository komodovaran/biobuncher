import numpy as np
from glob import glob
from lib.math import normalize_tensor


paths = glob("../results/intensities/*resampled*.npz")
arrays = [np.load(p)["data"] for p in paths]

for X in arrays:
    is_one = normalize_tensor(X, per_feature = True)[0].round(3) == 1.
    print(np.sum(is_one, axis = 0)) # rowwise sum should yield (1, 1) or (1,0)