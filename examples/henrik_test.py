import streamlit as st
from lib.statbib import *
import seaborn as sns


@st.cache()
def MakeDat(mu, std, N):

    dat = [np.random.normal(m, s, N) for m, s in zip(mu, std)]

    return np.array(dat).T


mu = [-1, 2, 3, 1]
std = [1, 0.5, 2, 1]
N = 1000
data = MakeDat(mu, std, N)

st.write(data)

def Repackage(x, bins):
    return np.reshape(x, (4, int(len(x) / 4)))


@st.cache(suppress_st_warning=True)
def GetFit(data, bins):
    params = []
    xs, ys, stds = np.zeros((4, bins)), np.zeros((4, bins)), np.zeros((4, bins))
    res = np.zeros((4, bins))

    for i in range(4):
        st.write(i)
        d = data[:, i]
        m, s = np.mean(d), np.std(d)
        min, max = m - 2 * s, m + 2 * s
        bw = (max - min) / bins
        st.write(bw)
        xs[i], ys[i], stds[i] = histogram(d, range=(min, max), bins=bins)

        fitparams, err, pval = BChi2Fit(
            d, bins, gauss_extended, bound=(min, max), mu=1, sigma=1, N=1000
        )
        params.append(fitparams)
        res[i] = ys[i] - bw * gauss_extended(xs[i], *fitparams)
    return res.T, params


st.write(data.shape)

fitY = []
bins = st.slider("bins", 10, 200, 20)

res, params = GetFit(data, bins)

st.write(res)

fig, ax = plt.subplots(1, 1)
sns.violinplot(data=res, ax=ax)
ax.set(ylabel="")
fig1, axs = plt.subplots(1, 2)
axs[0].bar(np.arange(4), [m for m in [p[1] for p in params]])
axs[0].set(xlabel="Dimension", ylabel="Mu")
axs[1].bar(np.arange(4), [m for m in [p[2] for p in params]])
axs[1].set(xlabel="Dimension", ylabel="Sigma")
fig1.tight_layout()
st.pyplot(fig1)
st.pyplot(fig)
