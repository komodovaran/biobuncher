import base64
from io import StringIO

import matplotlib.pyplot as plt
import streamlit as st
from matplotlib import numpy as np
import lib.utils

@lib.utils.timeit
def svg_write(fig, center=True):
    """
    Renders a matplotlib figure object to SVG.
    Disable center to left-margin align like other objects.
    """
    # Save to stringIO instead of file
    imgdata = StringIO()
    fig.savefig(imgdata, format="svg")

    # Retrieve saved string
    imgdata.seek(0)
    svg_string = imgdata.getvalue()

    # Encode as base 64
    b64 = base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = '<p style="text-align:center; display: flex; justify-content: {};">'.format(css_justify)
    html = r'{}<img src="data:image/svg+xml;base64,{}"/>'.format(
        css, b64
    )

    # Write the HTML
    st.write(html, unsafe_allow_html=True)

@lib.utils.timeit
def write(fig):
    st.write(fig)


if __name__ == "__main__":
    fig, ax = plt.subplots(nrows=5, ncols=5)
    ax = ax.ravel()

    for i in range(len(ax)):
        x = np.arange(0, np.pi * np.random.randint(2, 10), 0.1)
        y = np.sin(x)

        ax[i].plot(x, y)

    plt.tight_layout()

    st.subheader("Peasant pixels")
    write(fig)

    st.subheader("Glorious SVG")
    svg_write(fig, center = False)
