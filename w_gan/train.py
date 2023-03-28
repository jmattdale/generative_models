import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from models.wgan import WGAN
from common.callbacks import GANMonitor
from common.config import load_config
from common.data import get_dataset

# Define the loss functions for the discriminator,
# which should be (fake_loss - real_loss).
# We will add the gradient penalty later to this loss function.
def critic_loss(real_img, fake_img):
    real_loss = tf.reduce_mean(real_img)
    fake_loss = tf.reduce_mean(fake_img)
    return fake_loss - real_loss


# Define the loss functions for the generator.
def generator_loss(fake_img):
    return -tf.reduce_mean(fake_img)

def main(config):
    dataset = get_dataset(config["batch_size"], dataset='digit')

    wgan = WGAN(config)

    wgan.compile(
        d_optimizer=tf.keras.optimizers.Adam(lr=config["lr"], beta_1=0.5, beta_2=0.9),
        g_optimizer=tf.keras.optimizers.Adam(lr=config["lr"], beta_1=0.5, beta_2=0.9),
        g_loss_fn=generator_loss,
        c_loss_fn=critic_loss
    )



    wgan.fit(dataset,
            epochs=config["epochs"],
            callbacks=[GANMonitor(config), 
                       TensorBoard(log_dir=os.path.join(config["model_dir"], "tensorboard"))]
    )


if __name__ == "__main__":
    config_path = sys.argv[1]
    config = load_config(config_path)
    main(config)