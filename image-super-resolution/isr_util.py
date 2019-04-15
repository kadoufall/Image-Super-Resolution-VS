import os
import random

import tensorflow as tf
from PIL import Image
from scipy import misc

tf.enable_eager_execution()


def delete_or_makedir(dir):
    if tf.gfile.Exists(dir):
        tf.gfile.DeleteRecursively(dir)
    tf.gfile.MakeDirs(dir)


def resize_image(filename, hr_dir, resize_dir):
    image = Image.open(os.path.join(hr_dir, filename))

    half_the_width = image.size[0] / 2
    half_the_height = image.size[1] / 2
    image = image.crop(
        (
            half_the_width - 64,
            half_the_height - 64,
            half_the_width + 64,
            half_the_height + 64
        )
    )

    file, _ = os.path.splitext(filename)
    image.save(os.path.join(resize_dir, file + '-resized.png'))


def prepare_train_dirs(checkpoint_dir, train_log_dir, delete_train_log_dir=False):
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(checkpoint_dir):
        tf.gfile.MakeDirs(checkpoint_dir)

    # Cleanup train log dir
    if delete_train_log_dir:
        delete_or_makedir(train_log_dir)


def preprocess_image(image_path, training=False):
    image_size = 128
    k_downscale = 4
    downsampled_size = image_size // k_downscale

    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    if training:
        # 在训练图像时，利用随机翻转对图像进行预处理来增加训练数据
        image = tf.image.random_flip_left_right(image)
    
        # 随机调整属性，使训练得到的模型尽可能小的受到无关因素的影响．
        image = tf.image.random_saturation(image, 0.95, 1.05)  # 饱和度
        image = tf.image.random_brightness(image, 0.05)  # 亮度
        image = tf.image.random_contrast(image, 0.95, 1.05)  # 对比度
        image = tf.image.random_hue(image, 0.05)  # 色相

    label = (tf.cast(image, tf.float32) - 127.5) / 127.5  # normalize to [-1,1] range

    feature = tf.image.resize_images(image, [downsampled_size, downsampled_size], tf.image.ResizeMethod.BICUBIC)
    feature = (tf.cast(feature, tf.float32) - 127.5) / 127.5  # normalize to [-1,1] range

    # if training:
    #     feature = feature + tf.random.normal(feature.get_shape(), stddev=0.03)

    return feature, label


def load_data(data_dir, training=False):
    filenames = tf.gfile.ListDirectory(data_dir)
    filenames = [os.path.join(data_dir, f) for f in filenames]
    random.shuffle(filenames)

    image_count = len(filenames)

    image_ds = tf.data.Dataset.from_tensor_slices(filenames)
    image_ds = image_ds.map(lambda image_path: preprocess_image(image_path, training=training))

    BATCH_SIZE = 30
    image_ds = image_ds.batch(BATCH_SIZE)

    # image_ds = image_ds.prefetch(buffer_size=400)

    return image_ds


def save_feature_label(train_log_dir, test_image_ds):
    feature_batch, label_batch = next(iter(test_image_ds))

    feature_dir = train_log_dir + '0_feature/'
    label_dir = train_log_dir + '0_label/'

    delete_or_makedir(feature_dir)
    delete_or_makedir(label_dir)

    for i, feature in enumerate(feature_batch):
        if i > 5:
            break
        misc.imsave(feature_dir + '{:02d}.png'.format(i), feature)

    for i, label in enumerate(label_batch):
        if i > 5:
            break
        misc.imsave(label_dir + '{:02d}.png'.format(i), label)


def vgg19():
    vgg19 = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    loss_model = tf.keras.Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    loss_model.trainable = False
    return loss_model
