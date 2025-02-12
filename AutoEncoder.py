
from keras import Model
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Flatten, Dense # type: ignore
from keras import backend as K


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

        self._build()


    def summary(self):
        self.encoder.summary()

    def _build(self):
        self._build_encoder()
        #self._build_decoder()
        #self._build_autoencoder()


    def _build_encoder(self):
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
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
    


if __name__ == "__main__":
    autoencoder = AutoEncoder(
        input_shape=(28,28,1),
        conv_filters=(32,64,64,64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim= 2 
    )
    autoencoder.summary()



