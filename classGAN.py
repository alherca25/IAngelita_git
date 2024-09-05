# Quiero que este código sea el predilecto para la generación del modelo, pero me da error de dimensionado

class GAN:
    def __init__(self):
        self.generator = self.make_generator_model()
        self.discriminator = self.make_discriminator_model()
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self.BUFFER_SIZE = 60000
        self.BATCH_SIZE = 256
        self.EPOCHS = 50

    def make_generator_model(self):
        model = tf.keras.Sequential()
        model.add(Dense(8*8*256, use_bias=False, input_shape=(100,)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Reshape((8, 8, 256)))
        model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        model.add(Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        # Añadir una capa adicional para alcanzar 128x128
        model.add(Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

        # Capa final para producir la imagen de salida de 128x128x1
        model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        return model

    def make_discriminator_model(self):
        model = tf.keras.Sequential()
        model.add(Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 1]))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        model.add(LeakyReLU())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(1))
        return model

    def discriminator_loss(self, real_output, fake_output):
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def generator_loss(self, fake_output):
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)

    @tf.function
    def train_step(self, images):
        noise = tf.random.normal([self.BATCH_SIZE, 100])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)

            real_output = self.discriminator(images, training=True)
            fake_output = self.discriminator(generated_images, training=True)

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

    def train(self, dataset):
        for epoch in range(self.EPOCHS):
            for image_batch in dataset:
                self.train_step(image_batch)
            print(f'Época {epoch + 1} completada, quedan {self.EPOCHS - epoch - 1} épocas')

    def generate_images(self, num_images):
        noise = tf.random.normal([num_images, 100])
        generated_images = self.generator(noise, training=False)
        return generated_images
    
# Crear una instancia del modelo GAN y entrenarlo
gan = GAN()
gan.train(resized_images)
