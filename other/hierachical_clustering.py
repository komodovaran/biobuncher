import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster import hierarchy
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics.pairwise import euclidean_distances
import scipy.spatial.distance as ssd
import seaborn as sns

sns.set_style("darkgrid")


np.random.seed(1)

X, _ = make_blobs(n_samples = 500, centers = 10, n_features = 2)

clf = MiniBatchKMeans(n_clusters = 10)
labels = clf.fit_predict(X)

fig, ax = plt.subplots(ncols = 2)
ax[0].scatter(X[:, 0], X[:, 1], c = labels)

ctrs = clf.cluster_centers_

for i in range(len(ctrs)):
    ax[0].annotate(
        xy = ctrs[i],
        s = i,
        bbox = dict(boxstyle = "square", fc = "w", ec = "grey", alpha = 0.9),
    )

square_distance_mat = np.round(euclidean_distances(ctrs), 2)

# convert the redundant n*n square matrix form into a condensed nC2 array

# condensed dist[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
condensed_dist = ssd.squareform(square_distance_mat)
z = hierarchy.linkage(condensed_dist, method = "single", metric = "euclidean")
dn1 = hierarchy.dendrogram(z, ax = ax[1], orientation = 'top')
plt.show()
