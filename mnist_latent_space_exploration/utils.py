import numpy as np
import pandas as pd
from PIL import Image


def get_image_from_latent_space(ls, decoder_model):
    return decoder_model.predict(ls)


def img_arr_to_gif(img_arr, title, **kwargs):
    imgs = [Image.fromarray(img) for img in img_arr]
    imgs[0].save(f"tmp/{title}.gif", save_all=True, append_images=imgs[1:], **kwargs)
    print("Gif Saved")


def get_background_grid(x_range, y_range, spacing):
    x_arr, y_arr = np.arange(x_range[0], x_range[1], spacing), np.arange(
        y_range[0], y_range[1], spacing
    )
    grid = np.meshgrid(x_arr, y_arr)
    grid = np.vstack(list(map(np.ravel, grid))).T
    return pd.DataFrame(data=grid, columns=["x", "y"])


if __name__ == "__main__":
    df = get_background_grid((0, 100), (0, 100), 1)
    print(df.head())
