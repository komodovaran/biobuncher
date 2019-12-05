import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)

x = np.random.uniform(0, 1, 100)
y = np.random.uniform(0, 1, 100)


plt.scatter(x, y)
coords = np.column_stack((x, y)).round(2)

ind = np.lexsort((coords[:,0],coords[:,1]))
coords = coords[ind]

for i, (x, y) in enumerate(coords):
    plt.text(s = i, x = x + 0.01, y = y + 0.01)
    plt.plot(x, y, color = "red")
plt.gca().invert_yaxis()
plt.show()