import numpy as np

f = np.load("results/intensities/tracks-CLTA-TagRFP EGFP-Gak-A8_var.npz", allow_pickle = True)["data"]
print(len(f))

f = np.load("results/intensities/tracks-cme_var.npz", allow_pickle = True)["data"]
print(len(f))