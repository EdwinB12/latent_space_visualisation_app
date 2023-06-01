from dash import html
from dash import dcc
import dash
from mnist_latent_space_exploration import get_model, plotting, utils
from dash.dependencies import Input, Output, State
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.pb")
decoder = model.get_layer("decoder")
encoder = model.get_layer("encoder")

# Generate some sample data
(_, _), (x_test, y_test) = get_model.get_data("mnist")
x_test = x_test[::5]
y_test = y_test[::5]
x_test_encoded = encoder.predict(x_test)

latent_space_df = pd.DataFrame(
    {"ls_x": x_test_encoded[:, 0], "ls_y": x_test_encoded[:, 1], "y_label": y_test}
).reset_index()

# TODO: Plot grid on figure so anywhere on figure can be clicked. Alternatively, find a better way of making whole plot clickable
# Generate a uniform grid 20% bigger than latent space limits
grid_x_range = (
    latent_space_df["ls_x"].min() - latent_space_df["ls_x"].min() * 0.2,
    latent_space_df["ls_x"].max() + latent_space_df["ls_x"].max() * 0.2,
)
grid_y_range = (
    latent_space_df["ls_y"].min() - latent_space_df["ls_y"].min() * 0.2,
    latent_space_df["ls_y"].max() + latent_space_df["ls_y"].max() * 0.2,
)
grid = utils.get_background_grid(grid_x_range, grid_y_range, 0.1)

app = dash.Dash(__name__)
app.layout = html.Div(
    [
        dcc.Graph(
            id="scatterplot",
            figure=plotting.get_scatterplot(latent_space_df),
            style={"width": "50%", "display": "inline-block", "height": "80vh"},
        ),
        html.Div(
            [
                dcc.Graph(id="image_1", figure={}),
                dcc.Graph(id="image_2", figure={}),
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
        dcc.Interval(id="interval", interval=2000, disabled=True),
    ],
    style={"width": "100%"},
)


# TODO: Make outline of point black when clicked.
@app.callback(
    Output("image_1", "figure", allow_duplicate=True),
    Output("image_2", "figure", allow_duplicate=True),
    Input("scatterplot", "clickData"),
    prevent_initial_call="initial_duplicate",
)
def plot_img_from_scatterplot(clickData):
    if clickData:
        # Get clickdata from point
        point = clickData["points"][0]
        x_clicked = point["x"]
        y_clicked = point["y"]

        ls = np.array([[x_clicked, y_clicked]], dtype=np.float32)

        predicted_image = utils.get_image_from_latent_space(ls, decoder_model=decoder)[
            0, :, :, 0
        ]

        try:
            index = point["customdata"][0]
            original_image = x_test[index]
        except Exception:
            return (
                {},
                plotting.plot_image(predicted_image, "Generated Image"),
            )

        return (
            plotting.plot_image(original_image, "Original Image"),
            plotting.plot_image(predicted_image, "Generated Image"),
        )
    else:
        return {}, {}


@app.callback(
    Output("scatterplot", "figure", allow_duplicate=True),
    Input("path", "n_clicks"),
    State("x1-input", "value"),
    State("y1-input", "value"),
    State("x2-input", "value"),
    State("y2-input", "value"),
    State("n_steps", "value"),
    State("scatterplot", "figure"),
    prevent_initial_call=True,
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
    dash.dependencies.Output("interval", "disabled"),
    dash.dependencies.Input("walk", "n_clicks"),
)
def start_interval(n_clicks):
    if n_clicks:
        print("Banana")
        if n_clicks % 2 == 0:
            print("Banana2")
            return True
        return False
    print("Banana1")
    return True


@app.callback(
    Output("image_1", "figure"),
    Output("image_2", "figure"),
    Output("scatterplot", "figure"),
    Input("interval", "n_intervals"),
    State("interval", "disabled"),
    State("image_1", "figure"),
    State("image_2", "figure"),
    State("scatterplot", "figure"),
)
def walk_path(n_intervals, is_disabled, figure1, figure2, scatterplot):
    """
    Continously walk along the created path
    """
    if not is_disabled:
        idx = (n_intervals - 1) % len(latent_space_arr)
        print(idx)

        data_names = [data_dict["name"] for data_dict in scatterplot["data"]]

        if "Selected Points" in data_names:
            scatterplot["data"].pop(-1)
        else:
            pass

        # TODO Plot the sample on top and then remove it. Too difficult to change the color of a marker
        scatterplot["data"].append(
            go.Scatter(
                x=[latent_space_arr[idx, 0]],
                y=[latent_space_arr[idx, 1]],
                mode="markers",
                marker={
                    "size": 15,
                    "opacity": 1,
                    "color": "green",
                    "symbol": "x",
                },
                line={"width": 4},
                name="Selected Points",
            )
        )

        im = utils.get_image_from_latent_space(
            latent_space_arr[np.newaxis, idx], decoder_model=decoder
        )[0, :, :, 0]

        return {}, plotting.plot_image(im, f"Generated Image: Step {idx}"), scatterplot
    return {}, figure2, scatterplot


if __name__ == "__main__":
    app.run_server(debug=True)
