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


def get_scatterplot(data):
    return px.scatter(
        data_frame=data,
        x="ls_x",
        y="ls_y",
        color="y_label",
        opacity=0.6,
        category_orders={"y_label": np.arange(0, 10)},
        hover_data=["index"],
        custom_data=["index"],
    )
