import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Dense,
    Reshape,
    Conv2D,
    Conv2DTranspose,
    LeakyReLU,
    ReLU,
    BatchNormalization
)
from tensorflow.keras import layers
from tensorflow.keras import initializers

class GeneratorBlock(layers.Layer):
    def __init__(self, units=32, kernel_size=3, strides=2, initializer=None):
        super(GeneratorBlock, self).__init__()
        self.conv_trans = Conv2DTranspose(
            units, kernel_size=kernel_size, strides=strides, padding="same", use_bias=True)
        self.relu = ReLU()
        self.bn = BatchNormalization()

    def call(self, input_tensor):
        x = self.conv_trans(input_tensor)
        x = self.bn(x)
        return self.relu(x)

class Generator(tf.keras.Model):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.hidden_units = config["gen_hidden_units"]
        self.output_dims = config["output_dims"]
        self.net = self.build_net(config["latent_dims"])

    def call(self, inputs):
        return self.net(inputs)

    def build_net(self, latent_dims):
        # model input
        inputs = tf.keras.Input(shape=(latent_dims,))
        x = inputs
        x = Dense(7 * 7 * 256)(x)
        x = Reshape((7, 7, 256))(x)

        # hidden layers
        # for units in self.hidden_units:
        x = GeneratorBlock(128, kernel_size=2)(x)
        # x = GeneratorBlock(128, kernel_size=4, strides=1)(x)
        x = GeneratorBlock(64)(x)
        
        # model output
        x = Conv2DTranspose(self.output_dims, kernel_size=5, padding="same", activation='tanh')(x)
        return tf.keras.Model(inputs, x, name='generator')

