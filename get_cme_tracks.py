from glob import glob
import pandas as pd
import scipy.io
from multiprocessing import Pool, cpu_count
from lib.utils import remove_parent_dir


def cme_tracks_to_pandas(mat_path, rm_n_parent_dir = 4):
    """
    Converts CME-derived ProcessedTracks.mat to Pandas DataFrame format.
    Single point extracted_features are marked with trailing underscores because
    Pandas can't deal with variable length columns formats

    Set the number of parent directories to remove, to get the right subfolder in the file
    """
    mat_file = scipy.io.loadmat(mat_path)

    m = pd.DataFrame(mat_file["tracks"][0])

    df = []
    for n in range(len(m)):
        x = m.loc[n]

        group_len = len(x["t"].T[:, 0])
        group = pd.DataFrame(
            {
                "file"      : remove_parent_dir(mat_path, rm_n_parent_dir),
                "particle"  : n,
                "t"         : x["t"].T[:, 0], # actual video time for particle
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
    IN_PATH = "/media/tklab/linux-data/Data/CLTA-TagRFP EGFP-Gak-A8/**/ProcessedTracks.mat"#"data/kangmin_data/**/ProcessedTracks.mat"
    OUT_PATH = "results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8.h5"

    files = sorted(glob(IN_PATH, recursive = True))
    print("Found files:")
    [print(f) for f in files]

    with Pool(cpu_count()) as p:
        df = pd.concat(p.map(cme_tracks_to_pandas, files))

    print("NaNs:\n:", df.isna().sum())
    print("Number of files in df: {}".format(len(df["file"].unique())))

    # ALl traces
    df.to_hdf(OUT_PATH, key = "df")

    # Separate tracks with only good traces as determined by CME (might not be used)
    df[df["catIdx__"] <= 4].to_hdf(OUT_PATH[:-3] + "-catidx.h5", key = "df")