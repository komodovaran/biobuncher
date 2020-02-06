import numpy as np
import pandas as pd
from tqdm import tqdm
import parmap
import re

pd.set_option("display.max_columns", 100)


def _filter(args):
    """
    Parallel filter function

    Note:
        t column: actual real time (seconds, miliseconds, etc)
        f column: relative apperance time in video [26, 27, 28...]
    """
    _, group = args

    failsafes = []
    # check that it's not first
    if REMOVE_EDGES:
        fail1 = group["f"].values[0] == FIRST_FRAME
        # or last frame
        fail2 = group["f"].values[-1] == LAST_FRAME

        failsafes.extend([fail1, fail2])

    # check that group is not somehow longer than maximum length
    fail3 = len(group) >= LAST_FRAME
    # check that it's longer than minimum
    fail4 = len(group) <= MIN_LEN

    failsafes.extend([fail3, fail4])

    # if any fails, discard trace
    if any(failsafes):
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

    for i, (_, group) in enumerate(grouped_df):
        if group["f"].values[0] != 1:
            print(group[["f", "t"]].head(5))

    try:
        df_filtered = pd.concat(
            parmap.map(_filter, tqdm(grouped_df), pm_processes=16), sort=False
        )
    except ValueError:
        print(
            "Couldn't concatenate anything!\n"
            "Maybe all traces were removed in filtering?"
        )
        return

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


def main(input, min_len, remove_edges):
    for i in input:
        # Need to globally declare for the parallel function to easily grab
        global FIRST_FRAME
        global LAST_FRAME
        global MIN_LEN
        global REMOVE_EDGES

        df = pd.DataFrame(pd.read_hdf(i))

        FIRST_FRAME = df["f"].min()
        LAST_FRAME = df["f"].max()
        if LAST_FRAME == 0:
            raise ValueError(
                "Couldn't determine max frame, as last timepoint in 't' column "
                "is zero. Fix the 't' column before attempting to run this "
                "script"
            )
        _process_to_arrays(df=df, path=i, filter=min_len)


if __name__ == "__main__":
    INPUT_DF = (
        # "data/preprocessed/tracks-CLTA-TagRFP EGFP-Aux1-A7D2 EGFP-Gak-F6_filt5_var.h5",
        # "data/preprocessed/tracks-CLTA-TagRFP EGFP-Aux1-A7D2_filt5_var.h5",
        # "data/preprocessed/tracks-CLTA-TagRFP EGFP-Gak-A8_filt5_var.h5",
        # "data/preprocessed/test-CS2_CAV-GFP_VLP-CF640_filt5_var.h5",
    )
    MIN_LEN = 5  # default 5
    REMOVE_EDGES = False

    main(input=INPUT_DF, min_len=MIN_LEN, remove_edges = REMOVE_EDGES)
