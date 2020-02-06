import numpy as np
import pandas as pd
from tqdm import tqdm
import parmap
import re

pd.set_option("display.max_columns", 100)


def _filter(args):
    """
    Parallel filter function
    """
    _, group = args
    # check that it's not first
    check1 = group["f"].values[0] == FIRST_FRAME
    # or last frame
    check2 = group["f"].values[-1] == LAST_FRAME
    # check that it's not longer than maximum length
    check3 = len(group) >= LAST_FRAME
    # check that it's longer than minimum
    check4 = len(group) <= MIN_LEN
    # if any checks fail, discard trace
    if any([check1, check2, check3, check4]):
        return None
    else:
        return group


def _process_to_arrays(df, path, filter):
    """
    Save as filtered arrays, to make downstream processing easier
    """
    by = ["file", "particle"]

    columns = []
    for c in sorted(df.columns):
        r = re.search(pattern="int_c*", string=c)
        if r is not None:
            columns.append(r.string)

    grouped_df = df.groupby(by)
    df_filtered = pd.concat(
        parmap.map(_filter, tqdm(grouped_df), pm_processes=16), sort=False
    )

    arrays = [g[columns].values for _, g in tqdm(df_filtered.groupby(by))]

    df_filtered.to_hdf(path[:-3] + "_filt{}_var.h5".format(filter), key="df")
    np.savez(
        path[:-3] + "_filt{}_var.npz".format(filter),
        data=arrays,
        allow_pickle=True,
    )

    print("File used: {}".format(path))
    print("{} traces kept".format(len(arrays)))
    print("{} traces discarded".format(grouped_df.ngroups - len(arrays)))
    print("Example check: ", arrays[-1].shape)


def main(input, min_len):
    for i in input:
        # Need to globally declare for the parallel function to easily grab
        global FIRST_FRAME
        global LAST_FRAME
        global MIN_LEN

        df = pd.DataFrame(pd.read_hdf(i))
        FIRST_FRAME = df["t"].min()
        LAST_FRAME = df["t"].max()
        _process_to_arrays(df=df, path=i, filter=min_len)


if __name__ == "__main__":
    INPUT_DF = (
        "data/preprocessed/tracks-CLTA-TagRFP EGFP-Aux1-A7D2 EGFP-Gak-F6_filt5_var.h5",
        "data/preprocessed/tracks-CLTA-TagRFP EGFP-Aux1-A7D2_filt5_var.h5",
        "data/preprocessed/tracks-CLTA-TagRFP EGFP-Gak-A8_filt5_var.h5",
    )
    MIN_LEN = 5  # default 5

    main(input=INPUT_DF, min_len=MIN_LEN)
