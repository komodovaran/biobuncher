from glob import glob
import numpy as np
import matplotlib.pyplot as plt

def maxabs_tensor(X, per_feature=False):
    """
    Sample-wise max-value normalization of 3D array (tensor) either per-feature or per-sample
    """
    if len(X.shape) == 2:
        X = X[np.newaxis, :, :]
    if not len(X.shape) == 3:
        raise ValueError("Shape not a tensor")

    if per_feature:
        axis = 1
    else:
        axis = (1, 2)
    arr_max = np.max(X, axis = axis, keepdims = True)
    return np.squeeze(X / arr_max)

for per_feature in (True,False):
    X = np.random.normal(0, 100000, 10000).reshape(-1, 100, 2)
    n_features = X.shape[-1]
    X_norm = maxabs_tensor(X, per_feature = per_feature)
    if per_feature:
        # Each colum should have max 1, so sum should be equal to number of columns
        print(X_norm[0].max(axis = 0))
        assert X_norm[0].max(axis = 0).sum() == X_norm.shape[2]
    else:
        # Each colum should have max 1, so sum should be equal to number of columns
        print(X_norm[0].max(axis = 0))
        assert X_norm[0].max(axis = 0).sum() <= X_norm.shape[2]

    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    fig.suptitle("per feature = {}".format(per_feature))
    ax = ax.ravel()
    ax[0].set_title("feature 0")
    ax[1].set_title("feature 1")
    ax[0].plot(X[0, :, 0])
    ax[1].plot(X[0, :, 1], color = "orange")
    ax[2].plot(X_norm[0, :, 0])
    ax[3].plot(X_norm[0, :, 1], color = "orange")
    ax[2].set_ylim(-1, 1)
    ax[3].set_ylim(-1, 1)
    plt.tight_layout()
    plt.show()

    # # Then load some real data
    PATHS = glob("../results/intensities/*re*.npz")
    arrays = [np.load(p)["data"] for p in PATHS]
    for p, a in zip(PATHS, arrays):
        print("Loaded {}\nshape: {}\ndtype: {}\n\n".format(p, a.shape, a[0].dtype))
        print("min: {:.2f}, max: {:.2f}".format(a.min(), a.max()))