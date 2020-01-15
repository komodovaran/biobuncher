import matplotlib.pyplot as plt
import numpy as np
import skimage.transform

y = []
for i in range(200):
    trace_c0 = [1, 2, 3, 5, 6, 2, 1, 2, 3, 5]
    trace_c0 = np.array(trace_c0, dtype = float)
    trace_c0 *= np.random.randint(1, 20)
    trace_c0 *= np.random.normal(1, 0.2, len(trace_c0))

    trace_c1 = np.flip(trace_c0) * 3

    newtrace = np.row_stack((trace_c0, trace_c1))
    y.append(newtrace)

y = np.dstack(y)
y = np.swapaxes(y, 0, 2)

# plt.plot(traces[0])
y = np.array([t / t.max(axis = (0, 1)) for t in y])

y = y[..., 0]

x = np.tile(np.arange(0, 10, 1), y.shape[0]).reshape(y.shape)
bins = 20

heatmap, xlabels, ylabels = np.histogram2d(y.ravel(), x.ravel(), bins=bins)
plt.plot(y[0])
plt.show()

plt.imshow(heatmap)
plt.show()