import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial.distance as ssd
from scipy.cluster import hierarchy
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
from lib.plotting import dendrogram_ts_layout

n_timeseries = 50

fig, axes = dendrogram_ts_layout(n_timeseries = n_timeseries)

X, _ = make_blobs(n_samples = 500, centers = 10, n_features = 2)
clf = MiniBatchKMeans(n_clusters = n_timeseries)
labels = clf.fit_predict(X)
ctrs = clf.cluster_centers_
square_distance_mat = np.round(euclidean_distances(ctrs), 2)
condensed_dist = ssd.squareform(square_distance_mat)
z = hierarchy.linkage(condensed_dist, method = "single", metric = "euclidean")

timeseries = []

for i, ax in enumerate(axes):
    if i == 0:
        continue
    ax.plot(np.random.normal(0, 1, 200))

hierarchy.dendrogram(z, ax = axes[0], orientation = 'left')

plt.show()