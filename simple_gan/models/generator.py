import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Activation, ReLU, BatchNormalization
from tensorflow.keras import layers
from tensorflow.keras import initializers

class GeneratorBlock(layers.Layer):
    def __init__(self, units=32, initializer=None):
        super(GeneratorBlock, self).__init__()
        self.dense = Dense(units, use_bias=False)
        self.leaky_relu = ReLU(negative_slope=0.2)
        self.bn = BatchNormalization()

    def call(self, input_tensor):
        x = self.dense(input_tensor)
        x = self.bn(x)
        return self.leaky_relu(x)

class Generator(tf.keras.Model):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.hidden_units = config["gen_hidden_units"]
        self.output_dims = tf.math.reduce_prod(config["image_dims"])
        self.net = self.build_net(config["latent_dims"])

    def call(self, inputs):
        return self.net(inputs)

    def build_net(self, latent_dims):
        # model input
        inputs = tf.keras.Input(shape=(latent_dims,))
        x = inputs

        # hidden layers
        for idx, units in enumerate(self.hidden_units):
            if idx == 0:
                x = Dense(units, kernel_initializer=initializers.RandomNormal(stddev=0.02), use_bias=False)(x)
                x = BatchNormalization()(x)
                x = ReLU(negative_slope=0.2)(x)
            else:
                x = GeneratorBlock(units)(x)
        
        # model output
        x = Dense(self.output_dims, activation='tanh')(x)
        return tf.keras.Model(inputs, x, name='generator')

