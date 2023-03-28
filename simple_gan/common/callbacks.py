import os
import tensorflow as tf

class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, config):
        self.num_images = config["display_images"]
        self.latent_dims = config["latent_dims"]
        self.image_dims = config["image_dims"]
        self.writer = tf.summary.create_file_writer(
            os.path.join(config["model_dir"], "tensorboard")
        )

    def on_batch_end(self, batch, logs=None):
        noise_vectors = tf.random.normal(shape=[self.num_images, self.latent_dims])
        generated_images = self.model.generator(noise_vectors)
        generated_images = tf.reshape(generated_images,
                                      [self.num_images, self.image_dims[0], self.image_dims[1], 1]
        )
        
        with self.writer.as_default(step=batch):
            tf.summary.image("generated_images", generated_images)