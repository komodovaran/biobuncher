import matplotlib.pyplot as plt
import numpy as np
from lib.tfcustom import VariableTimeseriesBatchGenerator
import pandas as pd

if __name__ == "__main__":
    np.random.seed(0)

    original_data = []
    for _ in range(300):
        xi = np.zeros((np.random.randint(10, 100), 2))
        original_data.append(xi)
    original_data = np.array(original_data)

    gen = VariableTimeseriesBatchGenerator(
        X=original_data,
        max_batch_size=32,
        shuffle_samples=True,
        shuffle_batches=True,
    )

    indices = gen.indices
    new_array = []
    for batchx, batchy in gen():
        for xi in batchx:
            new_array.append(xi)
    new_array = np.array(new_array)

    original_id = 200
    (idx,) = np.where(indices == original_id)[0]
    assert len(original_data[original_id]) == len(new_array[idx])