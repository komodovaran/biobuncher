import matplotlib.lines
from matplotlib import pyplot as plt
import numpy as np


def create_legend_handles(colors):
    """
    Creates the handles for a legend

    Example:
    -------
    fig.legend(create_legend_handles(("salmon", "lightgreen")), ["TagRFP", "EGFP"])
    """
    return [matplotlib.lines.Line2D([], [], color=c) for c in colors]


def plot_c0_c1(int_c0, int_c1, alpha = 1, bg_c0=None, bg_c1=None, ax=None, frame=None, separate_ax = False):
    """
    Easy plotting of Clathrin (channel 0) and Auxilin (channel 1)
    """
    if ax is None:
        fig, ax = plt.subplots()

    if frame is None:
        frame = range(len(int_c1))
    ax.plot(frame, int_c0, color="salmon", alpha = alpha)
    if bg_c0 is not None:
        ax.plot(frame, bg_c0, color="darkred", alpha = alpha)

    if separate_ax:
        ax_ = ax.twinx()
    else:
        ax_ = ax
    ax_.plot(frame, int_c1, color="seagreen", alpha = alpha)
    if bg_c1 is not None:
        ax_.plot(frame, bg_c1, color="darkgreen", alpha = alpha)

    for a in ax, ax_:
        a.set_yticks(())
    return ax if not separate_ax else ax, ax_


def plot_c0_c1_errors(mean_int_c0, mean_int_c1, separate_ax = False, frame=None, ax=None):
    std_c0, std_c1 = [
        np.std(mean_int, axis=0, keepdims=True)
        for mean_int in (mean_int_c0, mean_int_c1)
    ]

    if ax is None:
        fig, ax = plt.subplots()
    if frame is None:
        frame = range(len(mean_int_c1))
    ax.plot(frame, mean_int_c0, color="salmon")
    ax.fill_between(
        x=range(len(mean_int_c1)),
        y1=mean_int_c0 - std_c0,
        y2=mean_int_c0 + std_c0,
        facecolor="salmon",
        alpha=0.5,
    )
    ax.tick_params(axis = "y", colors = "salmon")

    if separate_ax:
        ax_ = ax.twinx()
    else:
        ax_ = ax
    ax_.plot(frame, mean_int_c1, color="seagreen")
    ax_.fill_between(
        x=range(len(mean_int_c1)),
        y1=mean_int_c1 - std_c1,
        y2=mean_int_c1 + std_c1,
        facecolor="seagreen",
        alpha=0.5,
    )
    ax_.tick_params(axis = "y", colors = "seagreen" if separate_ax else "black")

    # for a in ax, ax_:
    #     a.set_yticks(())
    #     a.set_xticks(())
    return ax if not separate_ax else ax, ax_
