import plotly.express as px
import plotly.graph_objs as go
import numpy as np


def plot_image(arr, title):
    return {
        "data": [go.Heatmap(z=np.flip(arr, 0), colorscale="Viridis", showscale=False)],
        "layout": {
            "xaxis": {"showgrid": False, "zeroline": False, "visible": False},
            "yaxis": {
                "showgrid": False,
                "zeroline": False,
                "scaleanchor": "x",
                "visible": False,
            },
            "title": title,
        },
    }
