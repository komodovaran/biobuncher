import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from lib.globals import groupby


def main(inputs, output):
    """
    Args:
        inputs (iter of str):
        output (str):
    """
    arrs_ls, dfs_ls = [], []
    lengths = []
    indices = []

    ids = []
    for i, npz_path in tqdm(enumerate(sorted(inputs))):
        h5_path = npz_path[:-3] + "h5"

        npz = np.load(npz_path, allow_pickle=True)["data"]
        h5 = pd.DataFrame(pd.read_hdf(h5_path))
        h5.reset_index(drop=True, inplace=True)

        # Reference to dataset it was combined from
        h5["source"] = os.path.basename(npz_path)

        # ID for sub-dataset
        h5["sub_id"] = h5.groupby(groupby).ngroup()

        # ID for combined dataset
        ids.append(h5["sub_id"].values)

        ngroups = h5.groupby(groupby).ngroups

        lengths.append(ngroups)
        idx = np.arange(0, ngroups, 1)

        # Create indices for each dataframe separately and stack them in the same order
        sub_index = pd.DataFrame(
            {"idx": idx, "file": os.path.basename(h5_path)}
        )

        arrs_ls.append(npz)
        dfs_ls.append(h5)
        indices.append(sub_index)

        if not ngroups == len(npz):
            raise ValueError("npz and h5 are not the same number of samples!")

    arrs = np.concatenate(arrs_ls)  # type: np.ndarray
    dfs = pd.concat(dfs_ls, sort=False)  # type: pd.DataFrame
    idxs = pd.concat(indices, sort=False)  # type: pd.DataFrame

    if not len(arrs) == len(idxs):
        raise ValueError("Number of samples and indices are different")

    dfs["id"] = dfs.groupby(["file", "particle", "source"]).ngroup()

    np.savez(output + ".npz", data=arrs)
    dfs.to_hdf(output + ".h5", key="df")
    idxs.to_hdf(output + "_idx.h5", key="df")

    # Test some random indices
    idx = 5381, 53179, 6712, 358
    for i in idx:
        grp = dfs[dfs["id"] == dfs["id"].max() - i + 1]
        if not len(arrs[-i]) == len(grp):
            raise ValueError(
                "Indexing failed. Group order not preserved correctly"
            )


if __name__ == "__main__":
    INPUTS = (
        "data/preprocessed/tracks-CLTA-TagRFP EGFP-Aux1-A7D2 EGFP-Gak-F6_filt20_var.npz",
        "data/preprocessed/tracks-CLTA-TagRFP EGFP-Aux1-A7D2_filt20_var.npz",
        "data/preprocessed/tracks-CLTA-TagRFP EGFP-Gak-A8_filt20_var.npz",
    )
    OUTPUT = "data/preprocessed/combined_filt20_var"

    main(inputs=INPUTS, output=OUTPUT)
