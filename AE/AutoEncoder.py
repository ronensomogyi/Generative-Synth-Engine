import os
import pickle
from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense, Reshape, Conv2DTranspose, Activation # type: ignore
from keras import backend as K
from keras.optimizers import Adam # type: ignore
from keras.losses import MeanSquaredError # type:ignore
import numpy as np

class AutoEncoder:
    """
    Deep Convolutional autoencoder with mirrored 
    encoder and decoder components
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        
        self.input_shape = input_shape 
        self.conv_filters = conv_filters 
        self.conv_kernels = conv_kernels 
        self.conv_strides = conv_strides
        self.latent_space_dim :int = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()


    """ Utility Methods """

    def summary(self):
        """ Prints summary of layer structure etc. """
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        """ Compiles the model, built in Keras tool """
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss =  MeanSquaredError() # Mean Squared Error
        self.model.compile(optimizer=optimizer, loss=mse_loss)


    def train(self, x_train, batch_size, num_epochs):
        """ Uses built in Keras tools to train model """
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)


    def reconstruct(self, images):
        """ 
        Takes: a list of images
        Returns: list of images encoded then decoded,
        list of points in the latent space corresponding to the images
        """
        latent_represenation = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_represenation)
        return reconstructed_images, latent_represenation


    @classmethod
    def load(cls, save_folder="."):
        """ 
        Creates new instance of AutoEncoder class and loads it w/ pretrained
        parameters and weights
        """
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)

        autoencoder = AutoEncoder(*parameters) # * means positional arguments
        weights_path = os.path.join(save_folder, "model.weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)


    def save(self, save_folder="."):
        """ Saves parameters and weights into specified directory, defaut wd """
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def _create_folder_if_it_doesnt_exist(self, folder):
        if not (os.path.exists(folder)):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape ,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "model.weights.h5")
        self.model.save_weights(save_path)

    """ End of Utility Methods """


    """ Construction Methods """

    def _build(self):
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()



    def _build_autoencoder(self):
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")


    def _build_encoder(self):
        """
        Uses an architecture consiting of Keras' Input, Conv2D, ReLU, BatchNormalization layers 
        to construct a Model which encodes an input into a specified latent space.
        """
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input # for _build_autoencoder()
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        return Input(shape=self.input_shape, name="encoder_input")
    
    def _add_conv_layers(self, encoder_input):
        """ Creates all conv blocks in encoder """
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x # graph of layers
    
    def _add_conv_layer(self, layer_index, x):
        """ 
        Adds a convolutional block to a graph of layers 
        consiting of conv 2d + ReLU + batch normalization
        """
        layer_num = layer_index + 1 # layers are zero indexed
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index], # num filters at specific layer index
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_num}"
        )

        x = conv_layer(x) # getting a keras layer and applying it to the graph of layers x
        x = ReLU(name=f"encoder_relu_{layer_num}")(x) # first parenthesis instantiates the layer and second applies it to x
        x = BatchNormalization(name=f"encoder_bn_{layer_num}")(x)
        return x

    def _add_bottleneck(self, x):
        """ 
        Flatten data and add bottleneck (Dense layer)
        """

        self._shape_before_bottleneck = x.shape[1:] # width, height, num channels
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x
    

    def _build_decoder(self):
        """
        Uses an architecture conssisting of Keras' Input, Dense, Reshape, Conv2DTranspose,
        ReLU, BatchNormalization layers to consrtuct a Model which decodes a point from a 
        latent space into an output
        """
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input) 
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder") # instantiate keras model
    
    def _add_decoder_input(self):
        return Input(shape=(self.latent_space_dim,), name="decoder_input") # need to pass latent dim as tuple
    
    def _add_dense_layer(self, decoder_input):
        num_neurons = np.prod(self._shape_before_bottleneck) # prod[w, h, num_channels] -> k neurons
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        return reshape_layer

    def _add_conv_transpose_layers(self, x):
        """ 
        Add convolutional transpose blocks
        """
        # loop thru all comv layers in reverse order and stop at the first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)

        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            filters=1, 
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer


    """ End of Construction Methods"""




