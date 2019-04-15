import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


class _IdentityBlock(tf.keras.Model):
    def __init__(self, filter, stride, data_format):
        super(_IdentityBlock, self).__init__(name='')

        bn_axis = 1 if data_format == 'channels_first' else 3

        self.conv2a = tf.keras.layers.Conv2D(
            filter, (3, 3), strides=stride, data_format=data_format, padding='same', use_bias=False)
        # self.bn2a = tf.keras.layers.BatchNormalization(axis=bn_axis)
        self.prelu2a = tf.keras.layers.PReLU(shared_axes=[1, 2])

        self.conv2b = tf.keras.layers.Conv2D(
            filter, (3, 3), strides=stride, data_format=data_format, padding='same', use_bias=False)
        # self.bn2b = tf.keras.layers.BatchNormalization(axis=bn_axis)

    def call(self, input_tensor):
        x = self.conv2a(input_tensor)
        # x = self.bn2a(x)
        # x = tf.nn.leaky_relu(x)

        x = self.prelu2a(x)
        x = self.conv2b(x)
        # x = self.bn2b(x)

        x = x + input_tensor
        return x


def pixelShuffler(inputs, scale=2):
    size = tf.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = inputs.get_shape().as_list()[-1]

    # Get the target channel size
    channel_target = c // (scale * scale)
    channel_factor = c // channel_target

    shape_1 = [batch_size, h, w, channel_factor // scale, channel_factor // scale]
    shape_2 = [batch_size, h * scale, w * scale, 1]

    # Reshape and transpose for periodic shuffling for each channel
    input_split = tf.split(inputs, channel_target, axis=3)
    output = tf.concat([phaseShift(x, scale, shape_1, shape_2) for x in input_split], axis=3)

    return output


def phaseShift(inputs, scale, shape_1, shape_2):
    # Tackle the condition when the batch is None
    X = tf.reshape(inputs, shape_1)
    X = tf.transpose(X, [0, 1, 3, 2, 4])

    return tf.reshape(X, shape_2)


class Generator(tf.keras.Model):
    def __init__(self, data_format='channels_last'):
        super(Generator, self).__init__(name='')

        if data_format == 'channels_first':
            self._input_shape = [-1, 3, 32, 32]
            self.bn_axis = 1
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1, 32, 32, 3]
            self.bn_axis = 3

        self.conv1 = tf.keras.layers.Conv2D(
            64, kernel_size=9, strides=1, padding='SAME', data_format=data_format)

        self.prelu1 = tf.keras.layers.PReLU(shared_axes=[1, 2])

        self.res_blocks = [_IdentityBlock(64, 1, data_format) for _ in range(16)]

        self.conv2 = tf.keras.layers.Conv2D(
            64, kernel_size=3, strides=1, padding='SAME', data_format=data_format)
    
        self.upconv1 = tf.keras.layers.Conv2D(
            256, kernel_size=3, strides=1, padding='SAME', data_format=data_format)
        self.prelu2 = tf.keras.layers.PReLU(shared_axes=[1, 2])

        self.upconv2 = tf.keras.layers.Conv2D(
            256, kernel_size=3, strides=1, padding='SAME', data_format=data_format)
        self.prelu3 = tf.keras.layers.PReLU(shared_axes=[1, 2])

        self.conv4 = tf.keras.layers.Conv2D(
            3, kernel_size=9, strides=1, padding='SAME', data_format=data_format)

    def call(self, inputs):
        x = tf.reshape(inputs, self._input_shape)

        x = self.conv1(x)
        # x = tf.nn.leaky_relu(x)
        x = self.prelu1(x)
        x_start = x

        for i in range(len(self.res_blocks)):
            x = self.res_blocks[i](x)

        x = self.conv2(x)
        x = x + x_start

        x = self.upconv1(x)
        x = pixelShuffler(x)
        x = self.prelu2(x)

        x = self.upconv2(x)
        x = pixelShuffler(x)
        x = self.prelu3(x)

        x = self.conv4(x)
        x = tf.nn.tanh(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, data_format='channels_last'):
        super(Discriminator, self).__init__(name='')

        if data_format == 'channels_first':
            self._input_shape = [-1, 3, 128, 128]
            self.bn_axis = 1
        else:
            assert data_format == 'channels_last'
            self._input_shape = [-1, 128, 128, 3]
            self.bn_axis = 3

        self.conv1 = tf.keras.layers.Conv2D(
            64, kernel_size=3, strides=1, padding='SAME', data_format=data_format)

        self.conv2 = tf.keras.layers.Conv2D(
            64, kernel_size=3, strides=2, padding='SAME', data_format=data_format)
        # self.bn2 = tf.keras.layers.BatchNormalization(axis=self.bn_axis)

        self.conv3 = tf.keras.layers.Conv2D(
            128, kernel_size=3, strides=1, padding='SAME', data_format=data_format)
        # self.bn3 = tf.keras.layers.BatchNormalization(axis=self.bn_axis)

        self.conv4 = tf.keras.layers.Conv2D(
            128, kernel_size=3, strides=2, padding='SAME', data_format=data_format)
        # self.bn4 = tf.keras.layers.BatchNormalization(axis=self.bn_axis)

        self.conv5 = tf.keras.layers.Conv2D(
            256, kernel_size=3, strides=1, padding='SAME', data_format=data_format)
        # self.bn5 = tf.keras.layers.BatchNormalization(axis=self.bn_axis)

        self.conv6 = tf.keras.layers.Conv2D(
            256, kernel_size=3, strides=2, padding='SAME', data_format=data_format)
        # self.bn6 = tf.keras.layers.BatchNormalization(axis=self.bn_axis)

        self.conv7 = tf.keras.layers.Conv2D(
            512, kernel_size=3, strides=1, padding='SAME', data_format=data_format)
        # self.bn7 = tf.keras.layers.BatchNormalization(axis=self.bn_axis)

        self.conv8 = tf.keras.layers.Conv2D(
            512, kernel_size=3, strides=2, padding='SAME', data_format=data_format)
        # self.bn8 = tf.keras.layers.BatchNormalization(axis=self.bn_axis)

        self.fc1 = tf.keras.layers.Dense(1024)
        self.fc2 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = tf.reshape(inputs, self._input_shape)
        x = self.conv1(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv3(x)
        # x = self.bn3(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv4(x)
        # x = self.bn4(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv5(x)
        # x = self.bn5(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv6(x)
        # x = self.bn6(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv7(x)
        # x = self.bn7(x)
        x = tf.nn.leaky_relu(x)
        x = self.conv8(x)
        # x = self.bn8(x)
        x = tf.nn.leaky_relu(x)
        # x = self.flatten(x)
        x = self.fc1(x)
        x = tf.nn.leaky_relu(x)
        x = self.fc2(x)
        # x = tf.nn.sigmoid(x)
        return x


def create_g_loss(d_output, g_output, labels, loss_model):
    gene_ce_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(d_output), d_output)

    vgg_loss = tf.keras.backend.mean(tf.keras.backend.square(loss_model(labels) - loss_model(g_output)))

    # mse_loss = tf.keras.backend.mean(tf.keras.backend.square(labels - g_output))

    g_loss = vgg_loss + 1e-3 * gene_ce_loss
    # g_loss = mse_loss + 1e-3 * gene_ce_loss
    return g_loss


def create_d_loss(disc_real_output, disc_fake_output):
    disc_real_loss = tf.losses.sigmoid_cross_entropy(tf.ones_like(disc_real_output), disc_real_output)

    disc_fake_loss = tf.losses.sigmoid_cross_entropy(tf.zeros_like(disc_fake_output), disc_fake_output)

    disc_loss = tf.add(disc_real_loss, disc_fake_loss)
    return disc_loss


def create_optimizers():
    g_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8)
    d_optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, beta2=0.999, epsilon=1e-8)

    return g_optimizer, d_optimizer
