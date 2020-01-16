import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
from sklearn.datasets import make_blobs
from lib.math import histpoints_w_err

# generate random sample, two components
np.random.seed(0)

# concatenate the two datasets into the final training set
centers = [[1, 1], [-1, -1]]
X_train, _ = make_blobs(1000, n_features = 2, centers = centers, cluster_std = 0.4)

x = np.linspace(-2., 2., 4)
y = np.linspace(-2., 2., 4)

# fit a Gaussian Mixture Model with two components
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

probabilities = clf.predict_proba(X_train).round(2)
label = np.argmax(probabilities, axis = 1)
top_prop = probabilities.max(axis = 1)


# display predicted scores by the model as a contour plot
X, Y = np.meshgrid(x[1:], y[1:])
XX = np.array([X.ravel(), Y.ravel()]).T
Z = clf.predict(XX)
Z = Z.reshape(X.shape)

ndcounts, bin_edges = np.histogramdd(X_train, bins = (x, y), density=True)

proba = -clf.score_samples(XX).reshape(X.shape)
proba = np.exp(proba)

volume = np.product(np.diff(bin_edges)[:, 0])
proba /= proba.sum()

print(proba)
print(np.sum(proba, axis = (0, 1)))
print(ndcounts * volume)
print(np.sum(ndcounts * volume))
quit()

proba = proba.reshape(X.shape)

plt.contourf(X, Y, Z, extend = "both", cmap = "viridis", alpha = 0.5)
CS = plt.contour(X, Y, proba, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 20), cmap = "Greys")
plt.scatter(X_train[:, 0], X_train[:, 1], s = 50, c = label, edgecolors = "grey", zorder = 100, cmap = "viridis")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.tight_layout()
plt.show()