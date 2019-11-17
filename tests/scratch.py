import numpy as np

a = np.random.normal(0, 1, 20).reshape(10, 2)
print(a)
print()
print(a.mean(axis = 0))