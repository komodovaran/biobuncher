import base64
from io import StringIO

import matplotlib.lines
import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

import lib.math
import lib.utils


def create_legend_handles(colors):
    """
    Creates the handles for a legend

    Example:
    -------
    fig.legend(create_legend_handles(("salmon", "lightgreen")), ["TagRFP", "EGFP"])
    """
    return [matplotlib.lines.Line2D([], [], color = c) for c in colors]


def plot_c0_c1(
    int_c0 = None,
    int_c1 = None,
    alpha = 1,
    color0 = "salmon",
    color1 = "seagreen",
    ax = None,
    frame = None,
    separate_ax = False,
):
    """
    Easy plotting of Clathrin (channel 0) and Auxilin (channel 1)
    """
    if int_c0 is None:
        int_c0 = []
    if int_c1 is None:
        int_c1 = []

    if ax is None:
        fig, ax = plt.subplots()

    if frame is None:
        frame = range(len(int_c1))
    ax.plot(frame, int_c0, color = color0, alpha = alpha)

    if separate_ax:
        ax_ = ax.twinx()
        ax.tick_params(axis = 'y', colors = color0)
        ax_.tick_params(axis = 'y', colors = color1)
    else:
        ax_ = ax
    ax_.plot(frame, int_c1, color = color1, alpha = alpha)

    if separate_ax:
        return ax, ax_
    else:
        return ax,


def plot_c0_c1_errors(
    mean_int_c0,
    mean_int_c1,
    std_c0,
    std_c1,
    color0 = "salmon",
    color1 = "seagreen",
    separate_ax = False,
    frame = None,
    ax = None,
):
    if ax is None:
        fig, ax = plt.subplots()
    if frame is None:
        frame = range(len(mean_int_c1))
    ax.plot(frame, mean_int_c0, color = color0)
    ax.fill_between(
        x = range(len(mean_int_c1)),
        y1 = mean_int_c0 - std_c0,
        y2 = mean_int_c0 + std_c0,
        facecolor = color0,
        alpha = 0.4,
    )
    ax.tick_params(axis = "y", colors = color0)

    if separate_ax:
        ax_ = ax.twinx()
    else:
        ax_ = ax
    ax_.plot(frame, mean_int_c1, color = color1)
    ax_.fill_between(
        x = range(len(mean_int_c1)),
        y1 = mean_int_c1 - std_c1,
        y2 = mean_int_c1 + std_c1,
        facecolor = color1,
        alpha = 0.4,
    )
    ax_.tick_params(axis = "y", colors = color1 if separate_ax else "black")

    for a in ax, ax_:
        a.set_yticks(())
        a.set_xticks(())
    return ax if not separate_ax else ax, ax_


def sanity_plot(X, title):
    """
    Sanity check plot
    """
    fig, ax = plt.subplots(nrows = 3, ncols = 3)
    plt.suptitle(title)
    ax = ax.ravel()
    for n in range(len(ax)):
        try:
            ax[n].plot(X[n])
        except IndexError:
            fig.delaxes(ax[n])
    plt.tight_layout()
    plt.savefig("")
    plt.show()


def svg_write(fig, center = True):
    """
    Renders a matplotlib figure object to SVG.
    Disable center to left-margin align like other objects.
    """
    # Save to stringIO instead of file
    imgdata = StringIO()
    fig.savefig(imgdata, format = "svg")

    # Retrieve saved string
    imgdata.seek(0)
    svg_string = imgdata.getvalue()

    # Encode as base 64
    b64 = base64.b64encode(svg_string.encode("utf-8")).decode("utf-8")

    # Add some CSS on top
    css_justify = "center" if center else "left"
    css = '<p style="text-align:center; display: flex; justify-content: {};">'.format(
        css_justify
    )
    html = r'{}<img src="data:image/svg+xml;base64,{}"/>'.format(css, b64)

    # Write the HTML
    st.write(html, unsafe_allow_html = True)


def get_colors(cmap, n_colors):
    """Extracts n colors from a colormap"""
    norm = Normalize(vmin = 0, vmax = n_colors)
    return [plt.get_cmap(cmap)(norm(i)) for i in range(0, n_colors)]


def mark_trues(arr, ax, color, alpha):
    """
    Marks all True in array as stretches of solid coloring.
    E.g.
    [True True True False False]
    will be shaded at the first 3 datapoints
    """
    adjs, lns = lib.utils.count_adjacent_values(arr)
    position = range(len(arr))
    for idx, ln in zip(adjs, lns):
        if arr[idx] == 0:
            continue
        ax.axvspan(
            xmin = position[idx],
            xmax = position[idx] + ln,
            alpha = alpha,
            facecolor = color,
        )
