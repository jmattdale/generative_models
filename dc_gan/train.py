import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from models.gan import GAN
from common.callbacks import GANMonitor
from common.config import load_config
from common.data import get_dataset


def main(config):
    dataset = get_dataset(config["batch_size"], dataset='digit')

    gan = GAN(config)

    # gen = gan.generator
    # gen.built=True
    # gen = gen.get_layer('generator')
    # print(gen.summary())
    # exit()

    gan.compile(
        d_optimizer=tf.keras.optimizers.Adam(lr=config["lr"]),
        g_optimizer=tf.keras.optimizers.Adam(lr=config["lr"]),
        loss_fn=tf.keras.losses.BinaryCrossentropy(),
    )



    gan.fit(dataset,
            epochs=config["epochs"],
            callbacks=[GANMonitor(config), 
                       TensorBoard(log_dir=os.path.join(config["model_dir"], "tensorboard"))]
    )


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    main(config)