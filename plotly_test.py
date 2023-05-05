import dash_html_components as html
import dash_core_components as dcc
import dash
from latent_space_visualisation import get_model
from dash.dependencies import Input, Output, State
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objs as go

# Load model
model = tf.keras.models.load_model("model.pb")
decoder = model.get_layer("decoder")
encoder = model.get_layer("encoder")

# Generate some sample data
(_, _), (x_test, y_test) = get_model.get_data("mnist")
x_test_encoded = encoder.predict(x_test)
# x_test_pred = decoder.predict(x_test_encoded)

# min_x, max_x = np.floor(min(x_test_encoded)), np.ceil(max(x))
# min_y, max_y = np.floor(min(y)), np.ceil(max(y))

# xx, yy = np.meshgrid(
#     np.arange(min_x, max_x + 0.1, 0.1), np.arange(min_y, max_y + 0.1, 0.1)
# )
# uniform_grid = np.c_[xx.flatten(), yy.flatten()]


# Create a Dash app
app = dash.Dash(__name__)

# Define the layout of the app

# app.layout = html.Div(
#     [
#         dcc.Graph(
#             id="scatterplot",
#             figure={
#                 "data": [
#                     {
#                         "x": uniform_grid[:, 0],
#                         "y": uniform_grid[:, 1],
#                         "mode": "markers",
#                         "marker": {"opacity": 0},
#                         "showlegend": False,
#                         "hovertemplate": "x: %{x}<br>y: %{y}<extra></extra>",
#                     },
#                     {
#                         "x": x,
#                         "y": y,
#                         "mode": "markers",
#                         "marker": {"opacity": 1},
#                         "name": "Data",
#                     },
#                 ],
#                 "layout": {"title": "Scatter Plot", "hovermode": "closest"},
#             },
#         ),
#         html.Div(id="output"),
#     ]
# )

# TODO May want to convert latent space to df and use px.scatter so points can be easily colored.
app.layout = html.Div(
    [
        dcc.Graph(
            id="scatterplot",
            figure={
                "data": [
                    {
                        "x": x_test_encoded[:, 0],
                        "y": x_test_encoded[:, 1],
                        "mode": "markers",
                        "marker": {
                            "opacity": 0.7,
                            "color": y_test,
                            "showscale": False,
                            "showlegend": True,
                        },
                        "name": "Data",
                    },
                ],
                "layout": {
                    "title": "Scatter Plot",
                    "hovermode": "closest",
                    # "showlegend": True,
                },
            },
        ),
        html.Div(id="output"),
        dcc.Graph(id="clicked-point-plot"),
    ]
)


@app.callback(
    Output("clicked-point-plot", "figure"),
    Input("scatterplot", "clickData"),
)
def display_clicked_point(clickData):
    if clickData:
        point = clickData["points"][0]
        x_clicked = point["x"]
        y_clicked = point["y"]

        ls = np.array([[x_clicked, y_clicked]], dtype=np.float32)

        image = get_image_from_latent_space(ls)

        return plot_image(image)
    else:
        return {"data": [], "layout": {}}


def get_image_from_latent_space(ls, decoder_model=decoder):
    return decoder_model.predict(ls)


def plot_image(arr):
    return {
        "data": [go.Heatmap(z=arr[0, :, :, 0], colorscale="Viridis", showscale=False)],
        "layout": {
            "xaxis": {"showgrid": False, "zeroline": False, "visible": False},
            "yaxis": {
                "showgrid": False,
                "zeroline": False,
                "scaleanchor": "x",
                "visible": False,
            },
        },
    }


if __name__ == "__main__":
    app.run_server(debug=True)
