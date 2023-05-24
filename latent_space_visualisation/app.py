import dash_html_components as html
import dash_core_components as dcc
import dash
from latent_space_visualisation import get_model, plotting, utils
from dash.dependencies import Input, Output, State
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from PIL import Image

# Load model
model = tf.keras.models.load_model("../model.pb")
decoder = model.get_layer("decoder")
encoder = model.get_layer("encoder")

# Generate some sample data
(_, _), (x_test, y_test) = get_model.get_data("mnist")
x_test_encoded = encoder.predict(x_test)

latent_space_df = pd.DataFrame(
    {"ls_x": x_test_encoded[:, 0], "ls_y": x_test_encoded[:, 1], "y_label": y_test}
).reset_index()


app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc.Graph(
            id="scatterplot",
            figure=px.scatter(
                data_frame=latent_space_df,
                x="ls_x",
                y="ls_y",
                color="y_label",
                opacity=0.6,
                category_orders={"y_label": np.arange(0, 10)},
                hover_data=["index"],
                custom_data=["index"],
            ),
            style={"width": "50%", "display": "inline-block", "height": "80vh"},
        ),
        html.Div(
            [
                dcc.Graph(id="image_1", figure={}),
                dcc.Graph(id="image_2", figure={}),
                dcc.Graph(id="image_3", figure={}),
            ],
            style={
                "width": "50%",
                "display": "inline-block",
                "vertical-align": "top",
            },
        ),
        html.Div(
            [
                html.Label("Point 1:    "),
                html.Label("X Coordinate"),
                dcc.Input(id="x1-input", type="number", value=-10),
                html.Label("Y Coordinate"),
                dcc.Input(id="y1-input", type="number", value=10),
            ]
        ),
        html.Div(
            [
                html.Label("Point 2:    "),
                html.Label("X Coordinate"),
                dcc.Input(id="x2-input", type="number", value=10),
                html.Label("Y Coordinate"),
                dcc.Input(id="y2-input", type="number", value=-10),
            ]
        ),
        html.Div(
            [
                html.Label("Number of Steps"),
                dcc.Input(id="n_steps", type="number", value=5),
            ]
        ),
        html.Div([html.Button(id="path", n_clicks=0, children="Generate Path")]),
        html.Div([html.Button(id="walk", n_clicks=0, children="Walk Path")]),
    ],
    style={"width": "100%"},
)


@app.callback(
    [
        Output("image_1", "figure"),
        Output("image_2", "figure"),
    ],
    Input("scatterplot", "clickData"),
)
def plot_img_from_scatterplot(clickData):
    if clickData:
        point = clickData["points"][0]
        x_clicked = point["x"]
        y_clicked = point["y"]

        ls = np.array([[x_clicked, y_clicked]], dtype=np.float32)
        print(ls.shape)

        predicted_image = utils.get_image_from_latent_space(ls, decoder_model=decoder)[
            0, :, :, 0
        ]

        try:
            index = point["customdata"][0]
            original_image = x_test[index]
        except Exception:
            return {}, plotting.plot_image(predicted_image, "Generated Image")

        return plotting.plot_image(
            original_image, "Original Image"
        ), plotting.plot_image(predicted_image, "Generated Image")
    else:
        return {}, {}


@app.callback(
    Output("scatterplot", "figure"),
    Input("path", "n_clicks"),
    State("x1-input", "value"),
    State("y1-input", "value"),
    State("x2-input", "value"),
    State("y2-input", "value"),
    State("n_steps", "value"),
    State("scatterplot", "figure"),
)
def generate_path(n_clicks, x1, y1, x2, y2, n_steps, figure):
    if n_clicks:
        global latent_space_arr
        # Get the current data
        data = figure["data"]

        path_x = np.linspace(x1, x2, n_steps)
        path_y = np.linspace(y1, y2, n_steps)
        latent_space_arr = np.array([path_x, path_y]).T

        data.append(
            go.Scatter(
                x=path_x,
                y=path_y,
                mode="lines+markers",
                marker={
                    "size": 15,
                    "opacity": 1,
                    "color": "black",
                    "symbol": "x",
                },
                line={"width": 4},
                name="LS Walk",
            )
        )

        # Update the figure with the new data
        figure.update({"data": data})
    return figure


@app.callback(
    Output("image_3", "figure"), Input("walk", "n_clicks"), State("image_3", "figure")
)
def walk_path(n_clicks, figure):
    if n_clicks:
        idx = n_clicks - 1
        print(latent_space_arr[n_clicks].shape)
        im = utils.get_image_from_latent_space(
            latent_space_arr[np.newaxis, idx], decoder_model=decoder
        )[0, :, :, 0]
        return plotting.plot_image(im, f"Plot_{n_clicks}")
    return figure


if __name__ == "__main__":
    app.run_server(debug=True)
