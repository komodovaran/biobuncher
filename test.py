import numpy as np
from mayavi import mlab

mlab.options.offscreen = True

n = 5000
x = np.random.rand(n)
y = np.random.rand(n)
z = np.random.rand(n)
s = np.sin(x)**2 + np.cos(y)

mlab.points3d(x, y, z, s, colormap="RdYlBu", scale_factor=0.02, scale_mode='none')
