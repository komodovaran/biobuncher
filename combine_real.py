import numpy as np

INPUT_NPZ = (
    "results/intensities/tracks-CLTA-TagRFP EGFP-Aux1-A7D2 EGFP-Gak-F6_filt5_var.npz",
    "results/intensities/tracks-CLTA-TagRFP EGFP-Aux1-A7D2_filt5_var.npz",
    "results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8_filt5_var.npz",
)

data = []
for _input_npz in INPUT_NPZ:
    _d = np.load(_input_npz, allow_pickle = True)["data"]
    data.append(_d)
    print(_d.shape)

data = np.concatenate(data)

print("Combined: ", data.shape)
np.savez("results/intensities/combined_filt5_var.npz", data = data)