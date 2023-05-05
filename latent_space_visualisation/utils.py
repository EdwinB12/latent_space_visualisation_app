import tensorflow as tf
from tensorflow import keras


def get_latent_space(input_image, encoder, **kwargs):
    return encoder.predict(input_image, **kwargs)


def get_model_output(decoder, latent_space, **kwargs):
    return decoder.predict(latent_space, **kwargs)
