import matplotlib.lines
from matplotlib import pyplot as plt


def create_legend_handles(colors):
    """
    Creates the handles for a legend

    Example:
    -------
    fig.legend(create_legend_handles(("salmon", "lightgreen")), ["TagRFP", "EGFP"])
    """
    return [matplotlib.lines.Line2D([], [], color = c) for c in colors]


def _plot_c0_c1(int_c0, int_c1, bg_c0=None, bg_c1=None, ax=None, frame=None):
    """
    Easy plotting of Clathrin (channel 0) and Auxilin (channel 1)
    """
    if ax is None:
        fig, ax = plt.subplots()

    if frame is None:
        frame = range(len(int_c1))
    ax.plot(frame, int_c0, color="salmon")
    if bg_c0 is not None:
        ax.plot(frame, bg_c0, color="darkred")

    ax_ = ax.twinx()
    ax_.plot(frame, int_c1, color="seagreen")
    if bg_c1 is not None:
        ax_.plot(frame, bg_c1, color="darkgreen")

    for a in ax, ax_:
        a.set_yticks(())
    return ax, ax_