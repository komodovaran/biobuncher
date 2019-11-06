import numpy as np
from photutils import datasets


if __name__ == "__main__":
    # Create x and y indices
    h, w = 200, 300
    x, y = _create_grid(h=h, w=w)

    # create tom_data
    img = []
    for _ in range(10):
        randx = np.random.randint(10, w - 10)
        randy = np.random.randint(10, h - 10)
        amp = 100
        d = gaussian_2d(
            xy_array=(x, y),
            amplitude=amp,
            pos_x=randx,
            pos_y=randy,
            sigma_x=3,
            sigma_y=3,
            rotation=0,
            offset=0,
        )
        d = d + np.random.normal(0, 5, d.shape)
        d += 200  # add offset
        img.append(d)
    img = np.sum(img, axis=0)
    img = img.reshape(h, w)
    print(img.mean())
    print(np.std(img))