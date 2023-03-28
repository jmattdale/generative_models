import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Activation, ReLU
from tensorflow.keras.activations import relu
from tensorflow.keras import layers
from tensorflow.keras import initializers

class DiscriminatorBlock(layers.Layer):
    def __init__(self, units=32, dropout_rate=0.3, kernel_init=None):
        super(DiscriminatorBlock, self).__init__()
        if kernel_init is not None:
            self.dense = Dense(units, kernel_initializer=kernel_init)
        else:
            self.dense = Dense(units)
        self.drop_out = Dropout(dropout_rate)
        self.leaky_relu = ReLU(negative_slope=0.2)

    def call(self, input_tensor):
        x = self.dense(input_tensor)
        x = self.leaky_relu(x)
        return self.drop_out(x)

class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.hidden_units = config["disc_hidden_units"]
        self.dropout_rate = config["dropout_rate"]
        self.image_dims = tf.math.reduce_prod(config["image_dims"])
        self.net = self.build_net(self.image_dims)

    def call(self, inputs):
        return self.net(inputs)

    def build_net(self, image_dims):
        # model input
        inputs = tf.keras.Input(shape=(image_dims,))
        x = inputs

        # hidden layers
        for idx, units in enumerate(self.hidden_units):
            if idx == 0:
                x = DiscriminatorBlock(units, self.dropout_rate, kernel_init=initializers.RandomNormal(stddev=0.02))(x)
            else:
                x = DiscriminatorBlock(units, self.dropout_rate)(x)
        
        # model output
        x = layers.Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs, x, name='discriminator')

