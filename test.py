import numpy as np
import matplotlib.pyplot as plt

from lib.math import kde_2d

n_components = 3
import lib.globals
import os.path

# noinspection PyUnresolvedReferences
import mpl_scatter_density

encodings_name = "20200203-0424_lstm_vae_bidir_data=combined_filt5_var.npz_dim=128_act=None_bat=4_eps=1_zdim=16_anneal=1___umap__combined_filt20_var_+_fake_tracks_type_3.npz"

zz = np.load(
        os.path.join(lib.globals.umap_dir, encodings_name),
        allow_pickle=True,
    )

X = zz["umap"]

X = X[0:5000]

# Extract x and y
x = X[:, 0]
y = X[:, 1]
c = np.array([0] * 2000 + [1] * 1000 + [2] * 2000)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection="scatter_density")

# density
ax.scatter_density(X[:, 0], X[:, 1], cmap = "viridis", dpi = 20)

# labels
# for label in np.unique(c):
#     if label == 0:
#         continue
#     ax.scatter_density(X[c==label][:, 0], X[c==label][:, 1], c = c[c==label], cmap = "Accent", dpi = 20, alpha = 0.5)


xx, yy, zz = kde_2d(X[:, 0], X[:, 1])
ax.contour(xx, yy, zz, cmap = "coolwarm", alpha = 0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.title('2D Gaussian Kernel density estimation')
plt.show()