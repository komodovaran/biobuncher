import numpy as np
import pandas as pd
from tqdm import tqdm
import parmap

pd.set_option("display.max_columns", 100)


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
    check1 = group["t"].values[0] == FIRST_FRAME  # check that it's not first
    check2 = group["t"].values[-1] == LAST_FRAME  # or last frame
    check3 = (
        len(group) >= LAST_FRAME
    )  # check that it's not longer than maximum length
    check4 = len(group) <= FILTER_SHORT  # check that it's longer than minimum
    if any(
        [check1, check2, check3, check4]
    ):  # if any checks fail, discard trace
        return None
    else:
        return group


def _process(df, path, by, filter):
    """
    Save as filtered arrays, to make downstream processing easier
    """

    grouped_df = df.groupby(by)
    df_filtered = pd.concat(parmap.map(_filter, tqdm(grouped_df), pm_processes=16))
    arrays = [g[COLUMNS].values for _, g in tqdm(df_filtered.groupby(BY))]

    df_filtered.to_hdf(path[:-3] + "_filt{}_var.h5".format(filter), key="df")
    np.savez(path[:-3] + "filt{}_var.npz".format(filter), data=arrays, allow_pickle=True)

    print("File used: {}".format(path))
    print("{} traces kept".format(len(arrays)))
    print("{} traces discarded".format(grouped_df.ngroups - len(arrays)))
    print("Example check: ", arrays[-1].shape)


if __name__ == "__main__":
    INPUT = (
        "results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8.h5",
        "results/intensities/tracks-CLTA-TagRFP EGFP-Aux1-A7D2.h5",
        "results/intensities/tracks-CLTA-TagRFP EGFP-Aux1-A7D2 EGFP-Gak-F6.h5",
        "results/intensities/tracks-cme.h5",
    )

    BY = ["file", "particle"]
    FILTER_SHORT = 5 # default 5
    COLUMNS = ["int_c0", "int_c1"]

    for i in INPUT:
        df = _get_data(i)
        print(df.size)
        print(df.head())
        FIRST_FRAME = df["t"].min()
        LAST_FRAME = df["t"].max()
        _process(df=df, path=i, by=BY, filter = FILTER_SHORT)
