import numpy as np
import os.path

REAL_NPZ = "results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8_filt5_var.npz"
FAKE_NPZ = "results/intensities/fake_tracks_type_2.npz"

inputs = REAL_NPZ, FAKE_NPZ
data, lengths = [], []
for i in inputs:
    arr = np.load(i, allow_pickle = True)["data"]
    data.append(arr)
    lengths.append(len(arr))

data = np.concatenate(data)

print("Combined: ", data.shape)
print("The last {} indices will be fake data".format(lengths[-1]))
savename = REAL_NPZ[:-4] + "_+_" + os.path.basename(FAKE_NPZ)

np.savez(savename, data = data)