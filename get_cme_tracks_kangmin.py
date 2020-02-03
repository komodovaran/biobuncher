import re
from glob import glob

import numpy as np
import pandas as pd
import parmap
import scipy.io


def cme_tracks_to_pandas(mat_path, project_name):
    """
    Converts CME-derived ProcessedTracks.mat to Pandas DataFrame format.
    Single point extracted_features are marked with trailing underscores because
    Pandas can't deal with variable length columns formats
    """
    matlab_file = scipy.io.loadmat(mat_path)
    m = pd.DataFrame(matlab_file["tracks"][0])

    df = []
    for n in range(len(m)):
        x = m.loc[n]

        # Find out where parent dirs can be skipped
        real_dir = re.search(string = mat_path, pattern = project_name)

        # Create path from actual directory
        filepath = mat_path[real_dir.start():]

        track_len = len(x["t"].T[:, 0])
        group = pd.DataFrame(
            {
                "file": np.repeat(filepath, track_len),
                "particle": n,
                "t": x["t"].T[:, 0],  # actual video time for particle
                "f": x["f"].T[:, 0],
                "x": x["x"].T[:, 0],
                "y": x["y"].T[:, 0],
                "int_c0": x["A"].T[:, 0],
                "int_c1": x["A"].T[:, 1],
                "pval_Ar_c0": x["pval_Ar"].T[:, 0],
                "pval_Ar_c1": x["pval_Ar"].T[:, 1],
                "isPSF_c0": x["isPSF"].T[:, 0],
                "isPSF_c1": x["isPSF"].T[:, 1],
                "A_pstd_c0": x["A_pstd"].T[:, 0],
                "A_pstd_c1": x["A_pstd"].T[:, 1],
                "c_pstd_c0": x["c_pstd"].T[:, 0],
                "c_pstd_c1": x["c_pstd"].T[:, 1],
                "sigma_r_c0": x["sigma_r"].T[:, 0],
                "sigma_r_c1": x["sigma_r"].T[:, 1],
                "SE_sigma_r_c0": x["SE_sigma_r"].T[:, 0],
                "SE_sigma_r_c1": x["SE_sigma_r"].T[:, 1],
                "catIdx__": x["catIdx"].T[:, 0].repeat(track_len),
                # "visibility__": x["visibility"].T[:, 0].repeat(track_len),
                # "lifetime_s__": x["lifetime_s"].T[:, 0].repeat(track_len),
                # "isCCP"       : x["isCCP"].T[:, 0].repeat(track_len)
            }
        )
        # forward fill all NaNs
        group.fillna(method="ffill", inplace=True)

        df.append(group)
    return pd.concat(df)


def main(names, input, output):
    for name in names:
        _input = input.format(name)
        _output = output.format(name)

        experiment_name = (
            _input.split("/")[-3].replace(" ", "_").replace("-", "_").upper()
        )

        print("Processed experiment search string: ", experiment_name)
        files = sorted(glob(_input, recursive=True))
        print("\nFound files:")
        [print(f) for f in files]
        print()

        accepted_files = []
        for f in files:
            name_i = (
                f.split("/")[-5].replace(" ", "_").replace("-", "_").upper()
            )
            if name_i == experiment_name:
                accepted_files.append(f)
            else:
                print("Rejected: {}".format(f))
        print("\nAccepted files:")
        [print(f) for f in accepted_files]

        df = pd.concat(parmap.map(cme_tracks_to_pandas, files, project_name = name))

        print("NaNs:\n:", df.isna().sum())
        print("Number of files in df: {}".format(len(df["file"].unique())))

        # ALl traces
        # df.to_hdf(_output, key="df")


if __name__ == "__main__":
    # Project name goes into curly path
    PROJECT_NAMES = (
        "CLTA-TagRFP EGFP-Aux1-A7D2 EGFP-Gak-F6",
        "CLTA-TagRFP EGFP-Aux1-A7D2",
        "CLTA-TagRFP EGFP-Gak-A8",
    )

    # Search for tracks in this path. ** means multiple wildcard subdirectories
    SEARCH_PATTERN = "../../../Data/{}/**/ProcessedTracks.mat"

    # Output to a file that also contains the project name in the curly bracket
    OUTPUT_NAME = "data/preprocessed/tracks-{}.h5"

    main(names=PROJECT_NAMES, input=SEARCH_PATTERN, output=OUTPUT_NAME)