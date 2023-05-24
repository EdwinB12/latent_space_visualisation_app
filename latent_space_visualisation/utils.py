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


def get_image_from_latent_space(ls, decoder_model):
    return decoder_model.predict(ls)


def img_arr_to_gif(img_arr, title, **kwargs):
    imgs = [Image.fromarray(img) for img in img_arr]
    imgs[0].save(f"tmp/{title}.gif", save_all=True, append_images=imgs[1:], **kwargs)
    print("Gif Saved")
