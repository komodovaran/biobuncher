import numpy as np
from skimage import io
from scipy import fftpack
import streamlit as st
import matplotlib.pyplot as plt

def fft_bg_2d(image, K = 2, percentile = 10):
    """
    Background correction with Fast Fourier Transform on a 2D image.
    Args:
        K:
            Block size set to zero. Higher number gives more "wiggliness".
        percentile:
            Percentile above which to filter out. Higher number adds more noise.
    """
    M, N = image.shape

    F = fftpack.fftn(image)
    F_magnitude = np.abs(F)
    F_magnitude = fftpack.fftshift(F_magnitude)

    F_magnitude[M // 2 - K: M // 2 + K, N // 2 - K: N // 2 + K] = 0

    # Find all peaks higher than the 98th percentile
    peaks = F_magnitude < np.percentile(F_magnitude, percentile)

    # Shift the peaks back to align with the original spectrum
    peaks = fftpack.ifftshift(peaks)

    # Make a copy of the original (complex) spectrum
    F_dim = F.copy()

    # Set those peak coefficients to zero
    F_dim = F_dim * peaks.astype(int)

    # Do the inverse Fourier transform to get back to an image.
    # Since we started with a real image, we only look at the real part of
    # the output.
    image_filtered = np.real(fftpack.ifft2(F_dim))

    spectrum = np.log(1 + F_magnitude)
    after_sup = np.log(1 + np.abs(F_dim))

    return image_filtered, spectrum, after_sup

if __name__ == "__main__":
    video = io.imread("/Users/johannes/Documents/Code/particle_tracking/data/kangmin_data/B_CLTA-TagRFP EGFP-Aux1 EGFP-GAK F6-1/Cell1_1s/TagRFP/GFP-ND20 Exp100ms RFP-ND12 Exp100ms - 1_Cy3  TIRF Q-1.tif")

    st.write(video.shape)

    frame = video[0, ...]
    image_filtered, F_dim, peaks = fft_bg_2d(frame, K = 10, percentile = 1)

    fig, ax = plt.subplots(nrows = 2, ncols = 2)
    ax = ax.ravel()
    ax[0].imshow(frame.clip(0,2000))
    ax[1].imshow(image_filtered.clip(0,2000))
    ax[2].imshow(np.log10(1 + np.abs(F_dim)))
    ax[3].imshow(peaks)
    st.write(fig)