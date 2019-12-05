import numpy as np
import pandas as pd
from tqdm import tqdm
import parmap


def remove_none(ls):
    return [i for i in ls if i is not None]


def _get_data(path):
    """
    Loads all traces and converts them to a padded tensor
    """
    print(path)
    df = pd.DataFrame(pd.read_hdf(path))
    return df


def _filter(args):
    _, group = args
    if len(group) >= FILTER_SHORT:
        return group[["int_c0", "int_c1"]].values
    else:
        return None


def _process(df, path, by):
    """
    Zero-pad everything to match max-length of video, then do fourier transform
    """
    grouped_df = df.groupby(by)
    X_variable = parmap.map(_filter, tqdm(grouped_df), pm_processes = 4)
    X_variable = np.array(remove_none(X_variable))
    np.savez(path[:-3] + "_var.npz", data = X_variable, allow_pickle = True)
    print("File used: {}".format(path))
    print("X_variable:  {}".format(X_variable.shape))
    print(
        "{} traces discarded (too short)".format(
            grouped_df.ngroups - len(X_variable)
        )
    )
    print("Example check: ", X_variable[-1].shape)


if __name__ == "__main__":
    INPUT = (
        "results/intensities/tracks-cme.h5",
        # "results/intensities/tracks-CLTA-TagRFP_EGFP-Aux1-A7D2.h5",
        # "results/intensities/tracks-CLTA-TagRFP_EGFP-Aux1-A7D2_EGFP-Gak-F6.h5",
        # "results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8.h5",
    )

    BY = ["file", "particle", "split"]
    FILTER_SHORT = 20

    for i in INPUT:
        df = _get_data(i)
        if "split" not in df.columns:
            df["split"] = np.zeros(len(df))
        _process(df = df, path = i, by = BY)
