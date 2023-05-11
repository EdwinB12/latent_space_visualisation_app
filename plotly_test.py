import dash_html_components as html
import dash_core_components as dcc
import dash
from latent_space_visualisation import get_model
from dash.dependencies import Input, Output, State
import numpy as np
import tensorflow as tf
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd

# Load model
model = tf.keras.models.load_model("model.pb")
decoder = model.get_layer("decoder")
encoder = model.get_layer("encoder")

# Generate some sample data
(_, _), (x_test, y_test) = get_model.get_data("mnist")
x_test_encoded = encoder.predict(x_test)

latent_space_df = pd.DataFrame(
    {"ls_x": x_test_encoded[:, 0], "ls_y": x_test_encoded[:, 1], "y_label": y_test}
).reset_index()

# x_test_pred = decoder.predict(x_test_encoded)

# min_x, max_x = np.floor(min(x_test_encoded)), np.ceil(max(x))
# min_y, max_y = np.floor(min(y)), np.ceil(max(y))

# xx, yy = np.meshgrid(
#     np.arange(min_x, max_x + 0.1, 0.1), np.arange(min_y, max_y + 0.1, 0.1)
# )
# uniform_grid = np.c_[xx.flatten(), yy.flatten()]


# Create a Dash app
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
                category_orders={"y_label": np.arange(0, 10)},
                hover_data=["index"],
                custom_data=["index"],
            ),
            style={"width": "50%", "display": "inline-block", "height": "100vh"},
        ),
        html.Div(
            [dcc.Graph(id="image_1", figure={}), dcc.Graph(id="image_2", figure={})],
            style={"width": "50%", "display": "inline-block", "vertical-align": "top"},
        ),
    ],
    style={"width": "100%"},
)


@app.callback(
    [Output("image_1", "figure"), Output("image_2", "figure")],
    Input("scatterplot", "clickData"),
)
def display_clicked_point(clickData):
    if clickData:
        point = clickData["points"][0]
        x_clicked = point["x"]
        y_clicked = point["y"]
        index = point["customdata"][0]

        ls = np.array([[x_clicked, y_clicked]], dtype=np.float32)

        predicted_image = get_image_from_latent_space(ls)[0, :, :, 0]
        original_image = x_test[index]

        return plot_image(original_image, "Original Image"), plot_image(
            predicted_image, "Generated Image"
        )
    else:
        return {}, {}


def get_image_from_latent_space(ls, decoder_model=decoder):
    return decoder_model.predict(ls)


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


if __name__ == "__main__":
    app.run_server(debug=True)
