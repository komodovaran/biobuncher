import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import skimage.data
import lib.utils
import lib.math

z = skimage.data.stereo_motorcycle()[0][..., 0]
print(z.shape)
smooth = lib.math.smooth_spline_2d(z, s = 3e8, kx = 3, ky = 3)

fig, ax = plt.subplots(nrows = 2)
ax[0].imshow(z)
ax[1].imshow(smooth)
plt.show()