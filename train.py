from AutoEncoder import AutoEncoder
from tensorflow.keras.datasets import mnist # type: ignore
import tensorflow as tf
LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 20



def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # normalize images
    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):

    autoencoder = AutoEncoder(
        input_shape=(28,28,1), # shape of MNIST imgs
        conv_filters=(32,64,64,64), # hyperparameters
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2 
    )

    autoencoder.summary()
    autoencoder.compile(learning_rate=learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train, _, _, _ = load_mnist()
    autoencoder = train(x_train[:10000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
    autoencoder2 = AutoEncoder.load("model")
    autoencoder2.summary()



