import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

from data_preparator import DataPreparator

"""Based on: https://keras.io/examples/generative/wgan_gp/ (15.09.2021) - DD.MM.YYY"""
"""Based on: https://github.com/kongyanye/cwgan-gp/blob/master/cwgan_gp.py (15.09.2021) - DD.MM.YYY"""


# TODO: Bei den Datensätzen darauf achten, dass es sie dann immer als float64 vorhanden sind,
#  eventuell ist der cast aber nicht notwendig dafür
# tf.keras.backend.set_floatx('float64')

# TODO: Hier könnten das Skript aufgeteilt werden in: WGAN-Model und WGAN-Generator
class WGAN(keras.Model):
    def __init__(
            self,
            discriminator,
            generator,
            latent_dim,
            channel_number,
            num_classes,
            discriminator_extra_steps=3,
            gp_weight=10.0,
    ):
        super(WGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.channel_number = channel_number
        self.num_classes = num_classes
        self.d_steps = discriminator_extra_steps
        self.gp_weight = gp_weight

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn):
        super(WGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn

    def gradient_penalty(self, batch_size, real_eeg, fake_eeg):
        """ Calculates the gradient penalty.

        This loss is calculated on an interpolated image
        and added to the discriminator loss.
        """
        # Get the interpolated eeg data
        # TODO: Check if its legit
        # alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0, dtype=tf.float64)
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_eeg - real_eeg
        interpolated = real_eeg + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            # 1. Get the discriminator output for this interpolated eeg data.
            pred = self.discriminator(interpolated, training=True)

        # 2. Calculate the gradients w.r.t to this interpolated eeg data.
        grads = gp_tape.gradient(pred, [interpolated])[0]
        # 3. Calculate the norm of the gradients.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, train_data):
        real_eeg_data, one_hot_labels = train_data

        # Add dummy dimensions to the labels so that they can be concatenated with
        # the eeg data. This is for the discriminator.
        eeg_one_hot_labels = one_hot_labels[:, :, None, None]

        eeg_one_hot_labels = tf.repeat(
            eeg_one_hot_labels, repeats=[self.channel_number]
        )
        eeg_one_hot_labels = tf.reshape(
            eeg_one_hot_labels, (-1, self.channel_number, self.num_classes)
        )

        # Sample random points in the latent space and concatenate the labels.
        # This is for the generator.
        batch_size = tf.shape(real_eeg_data)[0]

        # For each batch, we are going to perform the
        # following steps as laid out in the original paper:
        # 1. Train the generator and get the generator loss
        # 2. Train the discriminator and get the discriminator loss
        # 3. Calculate the gradient penalty
        # 4. Multiply this gradient penalty with a constant weight factor
        # 5. Add the gradient penalty to the discriminator loss
        # 6. Return the generator and discriminator losses as a loss dictionary

        # Train the discriminator first. The original paper recommends training
        # the discriminator for `x` more steps (typically 5) as compared to
        # one step of the generator. Here we will train it for 3 extra steps
        # as compared to 5 to reduce the training time.
        for i in range(self.d_steps):
            # Get the latent vector
            # Sample random points in the latent space.
            # TODO: Check if its legit
            # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=tf.float64)
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            random_vector_labels = tf.concat(
                [random_latent_vectors, one_hot_labels], axis=-1
            )
            # real_eeg_data_and_labels = tf.concat([real_eeg_data, eeg_one_hot_labels], -1)
            real_eeg_data_and_labels = tf.concat([real_eeg_data, eeg_one_hot_labels], axis=-1)
            with tf.GradientTape() as tape:
                # Generate fake eeg data from the latent vector
                fake_eeg_data = self.generator(random_vector_labels, training=True)
                fake_eeg_data = tf.concat([fake_eeg_data, eeg_one_hot_labels], axis=-1)
                # Get the logits for the fake eeg data
                fake_logits = self.discriminator(fake_eeg_data, training=True)
                # Get the logits for the real eeg data
                real_logits = self.discriminator(real_eeg_data_and_labels, training=True)

                # Calculate the discriminator loss using the fake and real image logits
                d_cost = self.d_loss_fn(real_eeg_data=real_logits, fake_eeg_data=fake_logits)

                # Calculate the gradient penalty
                gp = self.gradient_penalty(batch_size, real_eeg_data_and_labels, fake_eeg_data)

                # Add the gradient penalty to the original discriminator loss
                d_loss = d_cost + gp * self.gp_weight

            # Get the gradients w.r.t the discriminator loss
            d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
            # Update the weights of the discriminator using the discriminator optimizer
            self.d_optimizer.apply_gradients(
                zip(d_gradient, self.discriminator.trainable_variables)
            )

        # Train the generator
        # Get the latent vector
        # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        # TODO: Hier muss ich noch mals überprüfen, wo ich genau die zusammenlegung der Daten eigentlich haben möchte,
        #  ich glaube nämlich das ich die auf der axis=0 statt auf der letzten haben möchte,
        #  ich bin mir aber nicht ganz so sicher
        # TODO: Check if its legit
        # random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim), dtype=tf.float64)
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        random_latent_vector_labels = tf.concat(
            [random_latent_vectors, one_hot_labels], axis=-1
        )
        with tf.GradientTape() as tape:
            # Generate fake eeg data using the generator
            generated_eeg_data = self.generator(random_latent_vector_labels, training=True)
            # Get the discriminator logits for eeg data
            generated_eeg_data = tf.concat([generated_eeg_data, eeg_one_hot_labels], -1)
            gen_eeg_data_logits = self.discriminator(generated_eeg_data, training=True)
            # Calculate the generator loss
            g_loss = self.g_loss_fn(gen_eeg_data_logits)

        # Get the gradients w.r.t the generator loss
        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.g_optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )
        return {"d_loss": d_loss, "g_loss": g_loss}

    def generate_eeg_data_sample(self, class_label):
        # self.generator.load_weights('../cwgan_gp/generator')
        noise = np.random.normal(0, 1, (1, self.latent_dim + self.num_classes))
        generate_eeg_sample = self.generator.predict([noise, np.array(class_label).reshape(-1, 1)])
        return generate_eeg_sample


class ConditionalWassersteinGanGenerator:

    def __init__(self,
                 batch_size,
                 eeg_channel_number,
                 eeg_time_steps,
                 num_classes,
                 latent_dimensions,
                 network_settings,
                 buffer_size=1024
                 ):

        # EEG_DATA_SHAPE = (64, 795)
        self.batch_size = batch_size
        # self.BATCH_SIZE = 512

        # Size of the noise vector
        self.eeg_channel_number = eeg_channel_number

        # TODO: Was sollte ich bei der wahl des 'latent space' betrachten,
        #  ich habe hier jetzt einfach die anzahl der Zeitpunkt in einem Datensample verwendet
        self.latent_dimensions = latent_dimensions  # Number of tTime steps for one eeg-data sample block
        self.num_time_steps = eeg_time_steps

        self.num_classes = num_classes

        self.buffer_size = buffer_size
        self.network_settings = network_settings

        # TODO: Für die Layer architecture das ganze noch mals angucken
        self.generator_in_channels = self.latent_dimensions + self.num_classes
        self.discriminator_in_channels = self.latent_dimensions + self.num_classes
        print(self.generator_in_channels, self.discriminator_in_channels)

        # Instantiate the optimizer for both networks
        # (learning_rate=0.0002, beta_1=0.5 are recommended)
        self.generator_optimizer = keras.optimizers.Adam(
            # learning_rate=0.0002, beta_1=0.5, beta_2=0.9
            learning_rate=self.network_settings.GENERATOR_LEARNING_RATE,
            beta_1=self.network_settings.GENERATOR_BETA_1,
            beta_2=self.network_settings.GENERATOR_BETA_2
        )
        self.discriminator_optimizer = keras.optimizers.Adam(
            # learning_rate=0.0002, beta_1=0.5, beta_2=0.9
            learning_rate=self.network_settings.DISCRIMINATOR_LEARNING_RATE,
            beta_1=self.network_settings.DISCRIMINATOR_BETA_1,
            beta_2=self.network_settings.DISCRIMINATOR_BETA_2
        )

    # Discrimnator
    def conv_block(self,
                   x,
                   filters,
                   activation,
                   kernel_size=(3,),
                   strides=(1,),
                   padding="same",
                   use_bias=True,
                   use_bn=False,
                   use_dropout=False,
                   drop_value=0.5,
                   ):
        x = layers.Conv1D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)
        if use_bn:
            x = layers.BatchNormalization()(x)
        x = activation(x)
        if use_dropout:
            x = layers.Dropout(drop_value)(x)
        return x

    def get_discriminator_model(self, naming):
        eeg_input = layers.Input(shape=(self.eeg_channel_number, self.discriminator_in_channels))
        x = layers.ZeroPadding1D((2, 2))(eeg_input)
        x = self.conv_block(
            x,
            # 64,
            self.network_settings.DISCRIMINATOR_FILTERS_LAYER_1,
            activation=layers.LeakyReLU(0.2),
            kernel_size=(5,),
            strides=(2,),
            use_bias=True,
            use_bn=False,
            use_dropout=False,
            drop_value=0.3,
        )
        x = self.conv_block(
            x,
            # 128,
            self.network_settings.DISCRIMINATOR_FILTERS_LAYER_2,
            activation=layers.LeakyReLU(0.2),
            kernel_size=(5,),
            strides=(2,),
            use_bias=True,
            use_bn=False,
            use_dropout=True,
            drop_value=0.3,
        )
        x = self.conv_block(
            x,
            # 256,
            self.network_settings.DISCRIMINATOR_FILTERS_LAYER_3,
            activation=layers.LeakyReLU(0.2),
            kernel_size=(5,),
            strides=(2,),
            use_bias=True,
            use_bn=False,
            use_dropout=True,
            drop_value=0.3,
        )
        x = self.conv_block(
            x,
            # 512,
            self.network_settings.DISCRIMINATOR_FILTERS_LAYER_4,
            kernel_size=(5,),
            strides=(2,),
            activation=layers.LeakyReLU(0.2),
            use_bn=False,
            use_bias=True,
            use_dropout=False,
            drop_value=0.3,
        )

        x = layers.Flatten()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(1)(x)

        d_model = keras.models.Model(eeg_input, x, name=f'{naming}_discriminator')
        return d_model

    # Generator
    def upsample_block(self,
                       x,
                       filters,
                       activation,
                       kernel_size=(3,),
                       # strides=(1,),
                       strides=1,
                       up_size=2,
                       padding="same",
                       use_bn=False,
                       use_bias=True,
                       use_dropout=False,
                       drop_value=0.3,
                       ):
        x = layers.UpSampling1D(up_size)(x)
        x = layers.Conv1D(
            filters, kernel_size, strides=strides, padding=padding, use_bias=use_bias
        )(x)

        if use_bn:
            x = layers.BatchNormalization()(x)

        if activation:
            x = activation(x)
        if use_dropout:
            x = layers.Dropout(drop_value)(x)
        return x

    def get_generator_model(self, naming):
        # noise = layers.Input(shape=(self.noise_dim + 5,))
        noise = layers.Input(shape=(self.generator_in_channels,))
        x = layers.Dense(self.eeg_channel_number * self.generator_in_channels, use_bias=False)(noise)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)

        x = layers.Reshape((self.eeg_channel_number, self.generator_in_channels))(x)
        x = self.upsample_block(
            x,
            # 128,
            self.network_settings.GENERATOR_FILTERS_LAYER_1,
            layers.LeakyReLU(0.2),
            # strides=(1,),
            strides=1,
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = self.upsample_block(
            x,
            # 64,
            self.network_settings.GENERATOR_FILTERS_LAYER_2,
            layers.LeakyReLU(0.2),
            # strides=(1,),
            strides=1,
            use_bias=False,
            use_bn=True,
            padding="same",
            use_dropout=False,
        )
        x = self.upsample_block(
            x,
            self.num_time_steps,
            activation=layers.Activation("tanh"),
            kernel_size=4,
            strides=8,
            use_bias=False,
            use_bn=True
        )
        # At this point, we have an output which has the same shape as the input, (32, 32, 1).
        # We will use a Cropping2D layer to make it (28, 28, 1).
        # x = layers.Cropping1D((2, 2))(x)  # TODO: Überprüfen ob ich das Cropping benötige oder einfach raus lassen kann

        g_model = keras.models.Model(noise, x, name=f'{naming}_generator')
        return g_model

    # g_model = get_generator_model()
    # g_model.summary()

    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to this loss function.
    def discriminator_loss(self, real_eeg_data, fake_eeg_data):
        real_loss = tf.reduce_mean(real_eeg_data)
        fake_loss = tf.reduce_mean(fake_eeg_data)
        return fake_loss - real_loss

    # Define the loss functions for the generator.
    def generator_loss(self, fake_eeg):
        return -tf.reduce_mean(fake_eeg)

    # Set the number of epochs for trainining.
    # epochs = 20
    # epochs = 1

    def generate_cw_gan(self, naming, data_sample, is_float_64, epochs=1):
        d_model = self.get_discriminator_model(naming)
        g_model = self.get_generator_model(naming)
        d_model.summary()
        g_model.summary()

        # Instantiate the WGAN model.
        self.wgan = WGAN(
            discriminator=d_model,
            generator=g_model,
            latent_dim=self.latent_dimensions,
            discriminator_extra_steps=self.network_settings.DISCRIMINATOR_EXTRA_STEPS,
            channel_number=self.eeg_channel_number,
            num_classes=self.num_classes
        )

        # Compile the WGAN model.
        self.wgan.compile(
            d_optimizer=self.discriminator_optimizer,
            g_optimizer=self.generator_optimizer,
            g_loss_fn=self.generator_loss,
            d_loss_fn=self.discriminator_loss,
        )

        _data = data_sample[0]
        _labels = data_sample[1]
        # TODO: Check if its legit
        all_labels = keras.utils.to_categorical(_labels, self.num_classes)
        if is_float_64:
            all_labels = tf.cast(all_labels, dtype=tf.float64)

        # Create tf.data.Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((_data, all_labels))
        dataset = dataset.shuffle(buffer_size=self.buffer_size).batch(self.batch_size)

        # print(f"Shape of training eeg-data: {_data.shape}")
        # print(f"Shape of training labels: {all_labels.shape}")
        # Start training the model.
        self.wgan.fit(dataset, batch_size=self.batch_size, epochs=self.network_settings.CW_GAN_TRAININGS_EPOCHS)

    def generate_augmented_data(self, number_of_samples_per_class):

        label_for_generating = []
        for x in range(self.num_classes):
            label_for_generating.append([x] * number_of_samples_per_class)

        label_for_generating = list(np.concatenate(label_for_generating))
        random.shuffle(label_for_generating)

        print(f'For {self.num_classes} classes with '
              f'{number_of_samples_per_class} samples we will get {len(label_for_generating)} labels')

        generated_data = []
        for label in label_for_generating:
            # The list contains a single element and by taking this, we do not need to reshape in a later step
            generated_data.append(self.wgan.generate_eeg_data_sample(label)[0])

        print(f'We got {len(generated_data)} generated data samples')
        data = (np.array(generated_data), np.array(label_for_generating))
        return data
