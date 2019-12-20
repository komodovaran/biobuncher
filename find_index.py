import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    np.random.seed(0)

    original_npz = np.load(
        "../results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8_var.npz",
        allow_pickle=True,
    )["data"]
    original_df = df = pd.DataFrame(
        pd.read_hdf(
            "../results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8_var.h5"
        )
    )

    grouped_df = original_df.groupby(["file", "particle"])
    keys = list(grouped_df.groups.keys())

    # Original
    INDEX_TO_FIND = 5446
    grp = grouped_df.get_group(keys[INDEX_TO_FIND])

    fig, ax = plt.subplots(ncols = 3)
    ax[0].plot(original_npz[INDEX_TO_FIND])
    ax[1].plot(grp[["int_c0", "int_c1"]])
    print(grp["file"][0])

    plt.show()