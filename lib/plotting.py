import base64
from io import StringIO

import matplotlib.cm
import matplotlib.lines
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt, gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap

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
    html = r'{}<img src="data:image/svg+xml;base64,{}" download=file.svg/>'.format(css, b64)

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

def plot_timeseries_percentile(timeseries, ax, n_percentiles = 10, min_percentile = 0, max_percentile = 100, color = "red"):
    """
    Plots varying percentiles of timeseries, as a type of contouring.
    Timeseries must be shape (samples, length)
    """
    if color == "red":
        colormap = matplotlib.cm.Reds_r
    elif color == "blue":
        colormap = matplotlib.cm.Blues_r
    elif color == "black":
        colormap = matplotlib.cm.Greys_r
    elif color == "purple":
        colormap = matplotlib.cm.Purples_r
    elif color == "orange":
        colormap = matplotlib.cm.Oranges_r
    else:
        raise NotImplementedError("Colormap not implemented for given color")

    cmap = colormap(np.arange(colormap.N))
    # Set alpha
    cmap[:, -1] = np.linspace(0, 1, colormap.N)
    # Create new colormap
    cmap = ListedColormap(cmap)

    percentiles = np.linspace(min_percentile, max_percentile, n_percentiles)
    length = timeseries.shape[1]

    sdist = np.zeros((n_percentiles, length))
    for i in range(n_percentiles):
        for t in range(length):
            sdist[i, t] = np.percentile(timeseries[:, t], percentiles[i])

    half = int((n_percentiles - 1) / 2)

    ax.plot(np.arange(0, length, 1), sdist[half, :], color = color, alpha = 0.35)
    for i in range(half):
        ax.fill_between(np.arange(0, length, 1), sdist[i, :], sdist[-(i + 1), :], facecolor = cmap(i / half), edgecolor = cmap(i / half))
    return ax


def rearrange_labels(X, cluster_labels, sort_on_column=0):
    """
    Sorts and rearranges cluster labels, so they appear in order of appereance
    instead of randomly.

    Args:
        X (np.array):
        cluster_labels (np.array):
        sort_on_column (int):

    Returns:
        Sorted cluster labels and corresponding centers
    """
    labels, ctrs = [], []
    for i in range(len(set(cluster_labels))):
        Xi = X[cluster_labels == i]
        ctr = np.mean(Xi, axis=0)
        labels.append(i)
        ctrs.append(ctr)

    ctrs = np.row_stack(ctrs)
    labels = np.array(labels).reshape(-1, 1)

    # sort on x column
    new_order = ctrs[:, sort_on_column].argsort()

    labels_new = labels[new_order]
    ctrs_new = ctrs[new_order]

    # replace labels with their new variants in the array of cluster labels
    np.put(cluster_labels, labels, labels_new)
    return cluster_labels, ctrs_new


def despine_ax(ax, top = True, right = True, left = True, bottom = True):
    """
    Removes spine from ax.
    """
    if top:
        ax.spines['top'].set_visible(False)
    if right:
        ax.spines['right'].set_visible(False)
    if left:
        ax.spines['left'].set_visible(False)
    if bottom:
        ax.spines['bottom'].set_visible(False)


def dendrogram_ts_layout(n_timeseries):
    """
    Multi-layout plot for a dendrogram and timeseries. Returns figure and axes.
    axes[0] is the dendrogram. The rest is for timeseries, top to bottom.

    Args:
        n_timeseries (int)
    """
    fig = plt.figure(figsize = (8, n_timeseries*2))
    outer_grid = fig.add_gridspec(nrows = 1, ncols = 2, wspace=0.2, hspace=0.2)

    outer_grid[0].subgridspec(1, 1, wspace = 0.0, hspace = 0.0)
    fig.add_subplot(outer_grid[0])

    ax_right = outer_grid[1].subgridspec(n_timeseries, 1, wspace=0.0, hspace=0.0)
    for i in range(n_timeseries):
        ax = fig.add_subplot(ax_right[i])
        ax.set_xticks(())
        ax.set_yticks(())
        despine_ax(ax)
        fig.add_subplot(ax)

    axes = fig.get_axes()
    despine_ax(axes[0], bottom = False)
    return fig, axes


def dendrogram_multi_ts_layout(
    n_timeseries_types, n_sub_grid, return_extra_column=False
):
    """
    Multi-layout plot for a dendrogram and timeseries. Returns figure and axes.
    axes[0] is the dendrogram. The rest is for timeseries, top to bottom. An extra
    column can be added to display some overview of timeseries (like mean trace)

    Args:
        n_sub_grid (int)
        n_timeseries_types (int)
        return_extra_column (bool)
    """
    fig = plt.figure(figsize=(10, n_timeseries_types * 2.5))

    # wspace: horizontal spacing, hspace: height spacing
    outer_grid = fig.add_gridspec(
        nrows=1, ncols=2 + return_extra_column, wspace=0.2, hspace=0.2
    )

    outer_grid[0].subgridspec(1, 1, wspace=0.0, hspace=0.0)
    fig.add_subplot(outer_grid[0])

    ax_left = fig.get_axes()[0]

    # Space between individual trace groups here
    ax_right = outer_grid[1].subgridspec(
        n_timeseries_types, 1, wspace=0.1, hspace=0.1
    )

    if return_extra_column:
        ax_extra = outer_grid[2].subgridspec(n_timeseries_types, 1)

    axes, axes_extra = [], []
    # Per timeseries type
    for i in range(n_timeseries_types):
        sub_axes = []
        inner_grid_right = gridspec.GridSpecFromSubplotSpec(
            n_sub_grid,
            n_sub_grid,
            subplot_spec=ax_right[i],
            wspace=0.0,
            hspace=0.0,
        )

        # Mini grid for sub-timeseries
        for j in range(n_sub_grid ** 2):
            ax = fig.add_subplot(inner_grid_right[j])
            ax.set_xticks(())
            ax.set_yticks(())
            despine_ax(ax)

            fig.add_subplot(ax)
            sub_axes.append(ax)
        axes.append(sub_axes)

        if return_extra_column:
            e_ax = fig.add_subplot(ax_extra[i])
            e_ax.set_xticks(())
            e_ax.set_yticks(())
            despine_ax(e_ax)
            axes_extra.append(e_ax)

    if return_extra_column:
        return fig, ax_left, axes, axes_extra
    else:
        return fig, ax_left, axes