import numpy as np
import matplotlib.pyplot as plt

a = np.random.normal(0, 1, 100)
b = a + 10

ab = np.column_stack((a,b))

plt.plot(ab)
plt.plot(ab / ab.max(axis = (0, 1)))
plt.show()