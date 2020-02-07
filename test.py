import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from lib.plotting import despine_ax


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
    fig = plt.figure(figsize=(9, n_timeseries_types * 2.5))

    outer_grid = fig.add_gridspec(
        nrows=1, ncols=2 + return_extra_column, wspace=0.1, hspace=0.3
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


fig, ax_left, axes, axes_extra = dendrogram_multi_ts_layout(
    n_timeseries_types=3, n_sub_grid=2, return_extra_column=True
)

for ax in axes[0]:
    ax.plot(np.random.normal(0, 1, 100))

for ax in axes[1]:
    ax.plot(np.random.normal(0, 1, 100))

for ax in axes[2]:
    ax.plot(np.random.normal(0, 1, 100))

ax_left.plot(np.random.normal(0, 1, 100))

axes_extra[2].plot(np.random.normal(0, 1, 100))

plt.show()
