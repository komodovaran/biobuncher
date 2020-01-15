import matplotlib.pyplot as plt
import numpy as np

max_frame = 100
h, w = 300, 300
n_samples = 40

fig, axes = plt.subplots(nrows = 2)
axes = axes.ravel()

for i, condition in enumerate((True, False)):
    ax = axes[i]
    uniform_random = condition
    samples = []
    for j in range(n_samples):
        appears_at = np.random.randint(0, max_frame)

        if uniform_random:
            x_pos = np.random.uniform(0, w)
            y_pos = np.random.uniform(0, h)
        else:
            x_pos = np.random.normal(w//2, 50, 1).clip(0, w)
            y_pos = np.random.normal(h // 2, 50, 1).clip(0, h)

        samples.append((x_pos, y_pos, appears_at))

    samples = np.row_stack(samples)
    f = ax.scatter(samples[:, 0], samples[:, 1], c = samples[:, 2])
    plt.colorbar(f, label = "time of apperance", ax = ax)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("uniform random" if i == 0 else "not uniform random")
plt.show()