from itertools import product
import numpy as np
import plotly.express as px
import streamlit as st
import plotly.subplots
import plotly.graph_objs as go


def make_secondary_y(nrows, ncols):
    return np.reshape(
        nrows * ncols * [{"secondary_y": True}], newshape = (nrows, ncols)
    ).tolist()


nrows, ncols = 3, 3

fig = plotly.subplots.make_subplots(
    rows = nrows,
    cols = ncols,
    specs = make_secondary_y(nrows, ncols),
    horizontal_spacing = 0.01,
    vertical_spacing = 0.01,
)

for r, c in product(range(nrows), range(ncols)):
    y1 = np.random.normal(0, 1, 50)
    y2 = np.random.normal(10,0.1, 50)
    x = np.arange(0, 50, 1)

    f1 = px.line(
        x = x,
        y = y1,
        color_discrete_sequence = ["red"],
    )

    f2 = px.line(
        x = x,
        y = y2,
        color_discrete_sequence = ["green"],
    )

    l = np.random.randint(0, 50, 1)

    p = dict(row = r + 1, col = c + 1)
    fig.add_trace(f1["data"][0], **p)
    fig.add_trace(f2["data"][0], **p, secondary_y = True)
    fig.update_layout(shapes = [go.layout.Shape(
        type = "line",
        x0 = l,
        x1 = l,
        y0 = min(y1),
        y1 = max(y1),
        line = dict(color = "RoyalBlue",width = 5)
    )])


p = dict(showticklabels = False, zeroline = False, showgrid = False, automargin = True)
fig.update_xaxes(**p)
fig.update_yaxes(**p)
fig.update_layout(showlegend = False)
st.write(fig)