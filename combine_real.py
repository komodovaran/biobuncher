import numpy as np
import pandas as pd
import os

INPUTS = (
    "results/intensities/tracks-CLTA-TagRFP EGFP-Aux1-A7D2 EGFP-Gak-F6_filt5_var.npz",
    "results/intensities/tracks-CLTA-TagRFP EGFP-Aux1-A7D2_filt5_var.npz",
    "results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8_filt5_var.npz",
)

OUTPUT = "results/intensities/combined_filt5_var"

arrs_ls, dfs_ls = [], []
indices = []
for npz_path in INPUTS:
    h5_path = npz_path[:-3] + "h5"

    npz = np.load(npz_path, allow_pickle = True)["data"]
    h5 = pd.DataFrame(pd.read_hdf(h5_path))

    ngroups = h5.groupby(["file" ,"particle"]).ngroups
    idx = np.arange(0, ngroups, 1)

    if not ngroups == len(npz):
        raise ValueError("npz and h5 are not the same number of samples!")

    # Create indices for each dataframe separately and stack them in the same order
    sub_index = pd.DataFrame({"idx" : idx,
                              "file": os.path.basename(h5_path)})

    arrs_ls.append(npz)
    dfs_ls.append(h5)
    indices.append(sub_index)

arrs = np.concatenate(arrs_ls) # type: np.ndarray
dfs = pd.concat(dfs_ls, sort = False) # type: pd.DataFrame
idxs = pd.concat(indices, sort = False) # type: pd.DataFrame

np.savez(OUTPUT + ".npz", data = arrs)
dfs.to_hdf(OUTPUT + ".h5", key = "df")
idxs.to_hdf(OUTPUT + "_idx.h5", key = "df")