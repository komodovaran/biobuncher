import numpy as np
import matplotlib.pyplot as plt

def ragged_stat(arr, f):
    """
    Returns the statistic of a ragged array, given a function
    """
    arr = np.array(arr)
    return f(np.concatenate(arr).ravel())

X = np.load("../results/intensities/tracks-cme_split-c1_var.npz", allow_pickle = True)["data"]

X_plot = X[0:25].copy()

fig, ax = plt.subplots(nrows = 5, ncols = 5)
fig.suptitle("Normal")
ax = ax.ravel()
for i in range(len(X_plot)):
    xi = X[i]
    ax[i].plot(xi[:, 0], color = "orange")
    ax[i].plot(xi[:, 1], color = "cyan")
plt.tight_layout()

fig, ax = plt.subplots(nrows = 5, ncols = 5)
fig.suptitle("Rescaled Clathrin")
ax = ax.ravel()
for i in range(len(X_plot)):
    xi = X[i]
    m = np.max(xi, axis = 0)
    xi[:, 0] = xi[:, 0] * (np.max(m) / np.min(m))
    xi = ((xi - ragged_stat(X_plot, np.mean)) / ragged_stat(X_plot, np.std))
    ax[i].plot(xi[:, 0], color = "orange")
    ax[i].plot(xi[:, 1], color = "cyan")
plt.tight_layout()
plt.show()

# fig, ax = plt.subplots(nrows = 2)
# ax = ax.ravel()
# means = np.zeros((len(X), 2))
# means_normalized = means.copy()
# for i in range(len(X)):
#     xi = X[i]
#     m = np.max(xi, axis = 0)
#     means[i, :] = m
#
#     # normalize clathrin
#     xi[:, 0] = xi[:, 0] * (np.max(m) / np.min(m))
#     np.max(xi, axis = 0)
#     means_normalized[i, :] = m
# ax[0].scatter(means[:, 0], means[:, 1])
# ax[1].scatter(means_normalized[:, 0], means_normalized[:, 1])
# plt.show()