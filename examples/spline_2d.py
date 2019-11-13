import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import skimage.data
import lib.utils
import lib.math
import skimage.io

# z = skimage.data.stereo_motorcycle()[0][..., 0]
z = skimage.io.imread("/Users/johannes/Downloads/elsp_0410.png")[..., 0]
print(z.shape)
bg = lib.math.smooth_spline_2d(z, s = 1e10, kx = 3, ky = 3)

fig, ax = plt.subplots(nrows = 2)
ax[0].imshow(z)
ax[1].imshow(z - bg)
plt.show()