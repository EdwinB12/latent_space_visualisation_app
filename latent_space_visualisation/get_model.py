import tensorflow as tf
from tensorflow import keras


def build_encoder(input_layer, latent_space_size):
    x = keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input_layer)
    x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = keras.layers.MaxPooling2D((2, 2), padding="same")(x)
    x = keras.layers.Flatten()(x)
    output_layer = keras.layers.Dense(latent_space_size)(x)

    encoder = keras.models.Model(input_layer, output_layer, name="encoder")

    return encoder


def build_decoder(latent_space_size):
    input_layer = keras.Input(latent_space_size)
    x = keras.layers.Dense(units=7 * 7 * 32, activation=tf.nn.relu)(input_layer)
    x = tf.keras.layers.Reshape(target_shape=(7, 7, 32))(x)
    x = keras.layers.Conv2DTranspose(
        64, (3, 3), strides=2, activation="relu", padding="same"
    )(x)
    x = keras.layers.Conv2DTranspose(
        32, (3, 3), strides=2, activation="relu", padding="same"
    )(x)
    output_layer = keras.layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(
        x
    )

    model = keras.models.Model(input_layer, output_layer, name="decoder")

    return model


def build_model(input_shape, latent_space_size):
    input_layer = keras.Input(input_shape)

    encoder = build_encoder(
        input_layer=input_layer, latent_space_size=latent_space_size
    )
    decoder = build_decoder(latent_space_size=latent_space_size)

    encoder_output = encoder(input_layer)
    decoder_output = decoder(encoder_output)

    ae = keras.models.Model(input_layer, decoder_output, name="autoencoder")

    return ae


def get_data(dataset_name):
    assert dataset_name in ["mnist"]

    if dataset_name == "mnist":
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    LATENT_SPACE_SIZE = 2
    INPUT_SHAPE = (28, 28, 1)

    (x_train, y_train), (x_test, y_test) = get_data("mnist")

    model = build_model(input_shape=INPUT_SHAPE, latent_space_size=LATENT_SPACE_SIZE)

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.fit(
        x=x_train, y=x_train, epochs=10, validation_data=(x_test, x_test), batch_size=32
    )

    model.save("../model.pb")
