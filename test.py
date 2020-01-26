import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from lib.plotting import rearrange_labels

np.random.seed(1)

X, _ = make_blobs(n_samples=500, centers=10, n_features=2)

clf = KMeans(n_clusters=10)
cluster_labels = clf.fit_predict(X)

cluster_labels, ctrs = rearrange_labels(X=X, cluster_labels=cluster_labels)

fig, ax = plt.subplots()
for i, m in enumerate(ctrs):
    ax.annotate(
        xy=m[[0, 1]],
        s=i,
        bbox=dict(boxstyle="square", fc="w", ec="grey", alpha=0.9),
    )
ax.scatter(X[:, 0], X[:, 1], c=cluster_labels)

plt.show()
