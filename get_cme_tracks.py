from glob import glob
import pandas as pd
import scipy.io
from multiprocessing import Pool, cpu_count


def _matlab_tracks_to_pandas(mat_path):
    """
    Converts CME-derived ProcessedTracks.mat to Pandas DataFrame format.
    Single point features are marked with trailing underscores because Pandas
    can't deal with variable length columns formats
    """
    mat_file = scipy.io.loadmat(mat_path)
    m = pd.DataFrame(mat_file["tracks"][0])

    df = []
    for n in range(len(m)):
        x = m.loc[n]

        group_len = len(x["t"].T[:, 0])
        group = pd.DataFrame(
            {
                "file"      : mat_path,
                "particle"  : n,
                "t"         : x["t"].T[:, 0],
                # "f"           : x["f"].T[:, 0],
                "int_c0"    : x["A"].T[:, 0],
                "int_c1"    : x["A"].T[:, 1],
                "pval_Ar_c0": x["pval_Ar"].T[:, 0],
                "pval_Ar_c1": x["pval_Ar"].T[:, 1],
                "isPSF_c0"  : x["isPSF"].T[:, 0],
                "isPSF_c1"  : x["isPSF"].T[:, 1],
                # "visibility__": x["visibility"].T[:, 0].repeat(group_len),
                "catIdx__"    : x["catIdx"].T[:, 0].repeat(group_len),
                # "lifetime_s__": x["lifetime_s"].T[:, 0].repeat(group_len),
                # "isCCP"       : x["isCCP"].T[:, 0].repeat(group_len)
            }
        )
        # forward fill all NaNs
        group.fillna(method = "ffill", inplace = True)

        df.append(group)
    return pd.concat(df)

if __name__ == "__main__":
    PATHS = sorted(glob("data/kangmin_data/**/ProcessedTracks.mat", recursive = True))
    RESULTS_PATH = "results/intensities/tracks-cme.h5"

    with Pool(cpu_count()) as p:
        df = pd.concat(p.map(_matlab_tracks_to_pandas, PATHS))

    print("NaNs:\n:", df.isna().sum())

    # ALl traces
    df.to_hdf(RESULTS_PATH, key = "df")

    # Only good traces as determined by CME
    df[df["catIdx__"] <= 4].to_hdf(RESULTS_PATH[:-3] + "-catidx.h5", key = "df")