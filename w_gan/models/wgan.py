import tensorflow as tf
from tensorflow.keras import layers

from models.critic import Critic
from models.generator import Generator

class WGAN(tf.keras.Model):
    def __init__(self, config):
        super(WGAN, self).__init__()
        self.batch_size = config["batch_size"]
        self.latent_dims = config["latent_dims"]
        self.critic_steps= config["critic_extra_train_steps"]
        self.grad_penalty_weight = config["grad_penalty_weight"]
        self.critic = Critic(config)
        self.generator = Generator(config)

    def compile(self, d_optimizer, g_optimizer, c_loss_fn, g_loss_fn):
        super().compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.c_loss_fn = c_loss_fn
        self.g_loss_fn = g_loss_fn
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    def gradient_penalty(self, real_images, fake_images):
        alpha = tf.random.normal([self.batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated = real_images * alpha - fake_images * (1-alpha)

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            # 1. get the descriminator output for the interpolated image
            pred = self.critic(interpolated, training=True)

        # 2. Calculate the gradient w.r.t to this interpolated image
        grads = tape.gradient(pred, [interpolated])[0]

        # 3. calculate the norm of the gradients
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1)**2)
        return gp


    @tf.function
    def train_generator(self, real_images):
        noise_vectors = tf.random.normal(shape=[self.batch_size, self.latent_dims])

        with tf.GradientTape() as tape:
            generated_images = self.generator(noise_vectors)
            critic_logits = self.critic(generated_images)
            generator_loss = self.g_loss_fn(critic_logits)

        grads = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        self.g_loss_metric.update_state(generator_loss)

    @tf.function
    def train_critic(self, real_images):
        for _ in range(self.critic_steps):
            noise_vectors = tf.random.normal(shape=[self.batch_size, self.latent_dims])

            generated_images = self.generator(noise_vectors)

            # Train the critic
            with tf.GradientTape() as tape:
                fake_images_logits = self.critic(generated_images)
                real_images_logits = self.critic(real_images)

                critic_cost = self.c_loss_fn(real_images_logits, fake_images_logits)
                gp = self.gradient_penalty(real_images, generated_images)
                c_loss = critic_cost + gp * self.grad_penalty_weight

            grads = tape.gradient(c_loss, self.critic.trainable_variables)
            self.d_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))
            self.d_loss_metric.update_state(c_loss)

    @tf.function
    def train_step(self, real_images):
        self.train_generator(real_images)
        self.train_critic(real_images)

        return {
            "generator_loss": self.g_loss_metric.result(),
            "discriminator_loss": self.d_loss_metric.result()
        }
