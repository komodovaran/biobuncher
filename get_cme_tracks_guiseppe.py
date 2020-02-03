import re
from glob import glob

import h5py
import numpy as np
import pandas as pd
import parmap


def index(m, track_id, column):
    """
    Indexing in the matlab h5 file returns a reference only. This reference is
    then used to go back and find the values in the file.
    """
    ref = m[column][track_id][0]
    return np.ravel(m[ref][:])


def cme_tracks_to_pandas(mat_path, project_name):
    """
    Converts CME-derived ProcessedTracks.mat to Pandas DataFrame format.

    This version was specifically created for matlab 7.3 files.

    Add extra columns as required.
    """

    COLUMNS = "A", "f", "t", "x", "y", "z"

    h5file = h5py.File(mat_path, "r")
    m = h5file["tracks"]
    n_tracks = len(m["A"])

    df = []
    for i in range(n_tracks):
        A, f, t, x, y, z = [index(m=m, track_id=i, column=c) for c in COLUMNS]
        track_len = len(A)

        # Find out where parent dirs can be skipped
        real_dir = re.search(string=mat_path, pattern=project_name)

        # Create path from actual directory
        filepath = mat_path[real_dir.start() :]

        group = pd.DataFrame(
            {
                "file": np.repeat(filepath, track_len),
                "particle": np.repeat(i, track_len),
                "int_c0": A,
                "f": f,
                "t": t,
                "x": x,
                "y": y,
                "z": z,
            }
        )

        df.append(group)

        group.fillna(method="ffill", inplace=True)

    return pd.concat(df)


def main(names, input, output):
    for name in names:
        _input = input.format(name)
        _output = output.format(name)

        files = sorted(glob(_input, recursive=True))
        print("\nFound files:")
        [print(f) for f in files]
        print()

        df = pd.concat(
            parmap.map(cme_tracks_to_pandas, files, project_name=name)
        )

        print("Number of files in df: {}".format(len(df["file"].unique())))

        # ALl traces
        df.to_hdf(_output, key="df")

        print("Each trace will be tagged with 'file' like:")
        print(df["file"].values[0])


if __name__ == "__main__":
    # Project name goes into curly path
    PROJECT_NAMES = ("Test",)

    # Search for tracks in this path. ** means multiple wildcard subdirectories
    SEARCH_PATTERN = "../../../Data/{}/**/ProcessedTracks.mat"

    # Output to a file that also contains the project name in the curly bracket
    OUTPUT_NAME = "data/preprocessed/test-{}.h5"

    main(names=PROJECT_NAMES, input=SEARCH_PATTERN, output=OUTPUT_NAME)
