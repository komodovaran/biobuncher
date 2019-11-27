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


def plot_c0_c1(
    int_c0,
    int_c1,
    alpha=1,
    color0="salmon",
    color1="seagreen",
    ax=None,
    frame=None,
    separate_ax=False,
):
    """
    Easy plotting of Clathrin (channel 0) and Auxilin (channel 1)
    """
    if ax is None:
        fig, ax = plt.subplots()

    if frame is None:
        frame = range(len(int_c1))
    ax.plot(frame, int_c0, color=color0, alpha=alpha)

    if separate_ax:
        ax_ = ax.twinx()
    else:
        ax_ = ax
    ax_.plot(frame, int_c1, color=color1, alpha=alpha)

    # for a in ax, ax_:
    #     a.set_yticks(())
    return ax if not separate_ax else ax, ax_


def plot_c0_c1_errors(
    mean_int_c0,
    mean_int_c1,
    color0="salmon",
    color1="seagreen",
    separate_ax=False,
    frame=None,
    ax=None,
):
    std_c0, std_c1 = [
        np.std(mean_int, axis=0, keepdims=True)
        for mean_int in (mean_int_c0, mean_int_c1)
    ]

    if ax is None:
        fig, ax = plt.subplots()
    if frame is None:
        frame = range(len(mean_int_c1))
    ax.plot(frame, mean_int_c0, color=color0)
    ax.fill_between(
        x=range(len(mean_int_c1)),
        y1=mean_int_c0 - std_c0,
        y2=mean_int_c0 + std_c0,
        facecolor=color0,
        alpha=0.4,
    )
    ax.tick_params(axis="y", colors=color0)

    if separate_ax:
        ax_ = ax.twinx()
    else:
        ax_ = ax
    ax_.plot(frame, mean_int_c1, color=color1)
    ax_.fill_between(
        x=range(len(mean_int_c1)),
        y1=mean_int_c1 - std_c1,
        y2=mean_int_c1 + std_c1,
        facecolor=color1,
        alpha=0.4,
    )
    ax_.tick_params(axis="y", colors=color1 if separate_ax else "black")

    for a in ax, ax_:
        # a.set_yticks(())
        a.set_xticks(())
    return ax if not separate_ax else ax, ax_
