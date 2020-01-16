import numpy as np

f = np.load("data/preprocessed/tracks-CLTA-TagRFP EGFP-Gak-A8_var.npz", allow_pickle = True)["data"]
print(len(f))

f = np.load("data/preprocessed/tracks-cme_var.npz", allow_pickle = True)["data"]
print(len(f))