import tensorflow as tf
import numpy as np

def get_dataset(batch_size, dataset="digit"):
    if dataset == "digit":
        (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
    else:
        (x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
        
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train.reshape(x_train.shape[0], 784)

    return tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder=True)