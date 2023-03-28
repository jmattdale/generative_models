import tensorflow as tf
from tensorflow.keras import layers

from models.discriminator import Discriminator
from models.generator import Generator

class GAN(tf.keras.Model):
    def __init__(self, config):
        super(GAN, self).__init__()
        self.batch_size = config["batch_size"]
        self.latent_dims = config["latent_dims"]
        self.discriminator = Discriminator(config)
        self.generator = Generator(config)

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    def train_generator(self, real_images):
        noise_vectors = tf.random.normal(shape=[self.batch_size, self.latent_dims])
        incorrect_labels = tf.ones((self.batch_size, 1))

        with tf.GradientTape() as tape:
            generated_images = self.generator(noise_vectors)
            descriminator_preds = self.discriminator(generated_images)
            generator_loss = self.loss_fn(incorrect_labels, descriminator_preds)
        grads = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        self.g_loss_metric.update_state(generator_loss)

    def train_discriminator(self, real_images):
        noise_vectors = tf.random.normal(shape=[self.batch_size, self.latent_dims])

        generated_images = self.generator(noise_vectors)
        combined_images = tf.concat([generated_images, real_images], axis=0)
        labels = tf.concat([
            tf.zeros((self.batch_size, 1)),
            tf.ones((self.batch_size, 1))], axis=0
        )

        # Add random noise to the labels - helps prevent mode collapse
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            descriminator_preds = self.discriminator(combined_images)
            descriminator_loss = self.loss_fn(labels, descriminator_preds)
        grads = tape.gradient(descriminator_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_variables))
        self.d_loss_metric.update_state(descriminator_loss)

    def train_step(self, real_images):
        self.train_generator(real_images)
        self.train_discriminator(real_images)

        return {
            "generator_loss": self.g_loss_metric.result(),
            "discriminator_loss": self.d_loss_metric.result()
        }
