import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results/_CLTA-TagRFP EGFP-Aux1-A7D2_Cell1_1s_TagRFP_GFP-ND20 Exp100ms RFP-ND12 Exp60ms - 1_Cy3  TIRF Q-1.tif.csv")

fig, axes = plt.subplots(nrows = 5, ncols = 2)
axes = axes.ravel()
for i, (n, particle) in enumerate(df.groupby("particle")):
    if i == 10:
        break
    axes[i].plot(particle["signal"])
plt.show()