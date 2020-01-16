import numpy as np
import os.path

def main(real_npz, fake_npz):
    data, lengths = [], []
    for i in (real_npz, fake_npz):
        arr = np.load(i, allow_pickle = True)["data"]
        data.append(arr)
        lengths.append(len(arr))
    data = np.concatenate(data)

    print("Combined: ", data.shape)
    print("The last {} indices will be fake data".format(lengths[-1]))

    savename = REAL_NPZ[:-4] + "_+_" + os.path.basename(FAKE_NPZ)
    np.savez(savename, data = data)


if __name__ == "__main__":
    REAL_NPZ = "data/preprocessed/tracks-CLTA-TagRFP EGFP-Gak-A8_filt5_var.npz"
    FAKE_NPZ = "data/preprocessed/fake_tracks_type_2.npz"
    main(real_npz = REAL_NPZ, fake_npz = FAKE_NPZ)