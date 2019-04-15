import os

import tensorflow as tf

import isr_train

import isr_util

# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# config = tf.ConfigProto()
# # config.gpu_options.per_process_gpu_memory_fraction = 0.8
# config.gpu_options.allow_growth = True
# tf.keras.backend.set_session(tf.Session(config=config))

CHECKPOINT_DIR = 'checkpoint/'
TRAIN_LOG_DIR = 'train_log/'

train_hr_dir = 'dataset/data/'
test_hr_dir = 'dataset/data-test/'

resize_train_dir = 'dataset/train/'
resize_test_dir = 'dataset/test/'

tf.enable_eager_execution()


def resize_data():
    train_files = tf.gfile.ListDirectory(train_hr_dir)
    test_files = tf.gfile.ListDirectory(test_hr_dir)

    isr_util.delete_or_makedir(resize_train_dir)
    isr_util.delete_or_makedir(resize_test_dir)

    for file in train_files:
        isr_util.resize_image(file, train_hr_dir, resize_train_dir)

    for file in test_files:
        isr_util.resize_image(file, test_hr_dir, resize_test_dir)


def train():
    isr_util.prepare_train_dirs(CHECKPOINT_DIR, TRAIN_LOG_DIR, True)

    train_image_ds = isr_util.load_data(resize_train_dir, training=True)
    test_image_ds = isr_util.load_data(resize_test_dir, training=False)

    isr_util.save_feature_label(TRAIN_LOG_DIR, test_image_ds)

    isr_train.train(TRAIN_LOG_DIR, train_image_ds, test_image_ds, 1000, CHECKPOINT_DIR)


if __name__ == '__main__':
    resize_data()

    # train()
