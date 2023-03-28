import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Reshape, Dense, Dropout, Flatten, Activation, LeakyReLU, Conv2D, BatchNormalization
from tensorflow.keras.activations import relu
from tensorflow.keras import layers
from tensorflow.keras import initializers

class DiscriminatorBlock(layers.Layer):
    def __init__(self, units=32):
        super(DiscriminatorBlock, self).__init__()
        self.conv = Conv2D(units, kernel_size=4, strides=2, padding="same", use_bias=False)
        self.bn = BatchNormalization()
        self.leaky_relu = LeakyReLU(alpha=0.2)

    def call(self, input_tensor):
        x = self.conv(input_tensor)
        # x = self.bn(x)
        return self.leaky_relu(x) 

class Discriminator(tf.keras.Model):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.hidden_units = config["disc_hidden_units"]
        self.dropout_rate = config["dropout_rate"]
        self.image_dims = config["image_dims"]
        self.net = self.build_net()

    def call(self, inputs):
        return self.net(inputs)

    def build_net(self):
        # model input
        inputs = tf.keras.Input(shape=self.image_dims)
        x = inputs

        # hidden layers
        for idx, units in enumerate(self.hidden_units):
            x = DiscriminatorBlock(units)(x)
        
        # model output
        x = Flatten()(x)
        # x = Dropout(self.dropout_rate)(x)
        x = Dense(1, activation="sigmoid")(x)
        return tf.keras.Model(inputs, x, name='discriminator')

