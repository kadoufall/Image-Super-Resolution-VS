import os

import tensorflow as tf
from scipy import misc

import isr_model
import isr_util

tf.enable_eager_execution()


def train_step(feature, label, loss_model, generator, discriminator, g_optimizer, d_optimizer):
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        generated_images = generator(feature)

        real_output = discriminator(label)
        generated_output = discriminator(generated_images)

        g_loss = isr_model.create_g_loss(generated_output, generated_images, label, loss_model)
        d_loss = isr_model.create_d_loss(real_output, generated_output)

    gradients_of_generator = g_tape.gradient(g_loss, generator.variables)
    gradients_of_discriminator = d_tape.gradient(d_loss, discriminator.variables)

    g_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
    d_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))

    return g_loss, d_loss


def train(train_log_dir, train_image_ds, test_image_ds, epochs, checkpoint_dir):
    generator = isr_model.Generator()
    discriminator = isr_model.Discriminator()
    g_optimizer, d_optimizer = isr_model.create_optimizers()

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(g_optimizer=g_optimizer,
                                     d_optimizer=d_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
    loss_model = isr_util.vgg19()

    for epoch in range(epochs):
        all_g_cost = all_d_cost = 0

        step = 0

        it = iter(train_image_ds)
        while True:
            try:
                image_batch, label_batch = next(it)
                step = step + 1
                g_loss, d_loss = train_step(image_batch, label_batch, loss_model, generator, discriminator,
                                            g_optimizer, d_optimizer)
                all_g_cost = all_g_cost + g_loss
                all_d_cost = all_d_cost + d_loss
            except StopIteration:
                break

        generate_and_save_images(train_log_dir, generator, epoch + 1, test_image_ds)

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)



def generate_and_save_images(train_dir, model, epoch, test_image_ds):
    dir = train_dir + str(epoch) + '/'

    feature_batch, _ = next(iter(test_image_ds))

    if tf.gfile.Exists(dir):
        tf.gfile.DeleteRecursively(dir)
    tf.gfile.MakeDirs(dir)

    predictions = model(feature_batch)

    for i, pred in enumerate(predictions):
        if i > 5:
            break
        misc.imsave(dir + 'image_{:02d}.png'.format(i), pred)
