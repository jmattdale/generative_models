import os
import gdown

import tensorflow as tf
import numpy as np
from zipfile import ZipFile

def get_dataset(batch_size, dataset='digit'):
    '''
    repo currently only works for MNIST datasets
    '''
    if dataset == 'digit' or dataset == 'fashion':
        if dataset == "digit":
            (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
        else:
            (x_train, _), (_, _) = tf.keras.datasets.fashion_mnist.load_data()
            
        x_train = (x_train.astype(np.float32) - 127.5) / 127.5
        x_train = np.expand_dims(x_train, -1)

        return tf.data.Dataset.from_tensor_slices(x_train).batch(batch_size, drop_remainder=True)

    if not os.path.exists("../data/celeba_gan"):
        os.makedirs("../data/celeba_gan")

        url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
        output = "../data/celeba_gan/data.zip"
        gdown.download(url, output, quiet=True)

        with ZipFile("../data/celeba_gan/data.zip", "r") as zipobj:
            zipobj.extractall("../data/celeba_gan")

    dataset = tf.keras.utils.image_dataset_from_directory(
        "../data/celeba_gan", label_mode=None, image_size=(64, 64), batch_size=batch_size
    )
    dataset = dataset.unbatch()
    # dataset = dataset.take(batch_size * 3)
    dataset = dataset.cache()
    dataset = dataset.map(lambda x: x / 255.0, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset