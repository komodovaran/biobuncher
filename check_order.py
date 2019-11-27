import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def _get_train_test_npz(path):
    """
    Loads train/test and normalization factor
    """
    f = np.load(path, allow_pickle=True)
    X_train, X_test, (mu, sg) = f["X_train"], f["X_test"], f["scale"]

    # Standardize
    X_train = np.array([(xi - mu) / sg for xi in X_train])
    X_test = np.array([(xi - mu) / sg for xi in X_test])
    return X_train, X_test, (mu, sg)


BY = ["file", "particle"]
df = pd.DataFrame(pd.read_hdf("results/intensities/tracks-cme_split-c1.h5"))
X_train, X_test, _ = _get_train_test_npz("results/intensities/tracks-cme_split-c1_var_traintest.npz")

grouped_df = df.groupby(BY)
groups = [group for _, group in grouped_df]

idx = 16

fig, ax = plt.subplots(nrows = 2)
ax[0].plot(groups[idx][["int_c0", "int_c1"]])
ax[1].plot(X_train[idx])
plt.show()