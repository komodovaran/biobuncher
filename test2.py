import matplotlib.pyplot as plt
import numpy as np

x = np.random.uniform(0, 1, 100)
y = x

pts = np.column_stack((x, y))
mean = np.mean(pts, axis = 0)

plt.scatter(pts[:, 0], pts[:, 1])
plt.annotate(xy = mean, s = "mid", bbox = dict(boxstyle = "square", fc = "w", alpha = 1))
plt.show()
