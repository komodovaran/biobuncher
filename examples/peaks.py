import numpy as np
import lib.math
import matplotlib.pyplot as plt

a = np.array([1,2,3,4,9,8,5,3,2,1])
b = a*100

plt.plot(a/a.max())
plt.plot(b/b.max())
plt.show()