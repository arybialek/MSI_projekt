import os
import random
from typing import Tuple

import fire
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# This doesn't work properly with tf.keras
# from callbacks import TensorBoard
from tensorflow.python.keras.callbacks import TensorBoard


def train(data_dir: str,
          model_id: str,
          batch_size: int = 8,
          rate: float = 0.5,
          target_size: Tuple[int, int] = (299, 299),
          lr: float = 1e-3,
          epochs: int = 25,
          seed: int = 1234,
          log_dir: str = './logs',
          ):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    keras.backend.set_session(sess)

    # Directories for training, validation and test splits.
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # Rescales all images by 1/255
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    # Generators.
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        color_mode='grayscale',
                                                        class_mode='binary',
                                                        shuffle=True,
                                                        )

    validation_generator = val_datagen.flow_from_directory(val_dir,
                                                           target_size=target_size,
                                                           batch_size=batch_size,
                                                           color_mode='grayscale',
                                                           class_mode='binary',
                                                           shuffle=False,
                                                           )

    train_steps = len(train_generator)
    val_steps = len(validation_generator)

    # Instantiating a small convent for positives and negatives classification.
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights=None,
                       input_shape=(target_size[0], target_size[1], 1)))
    # Version from tf.keras.applications
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=['acc'],
                  )

    cbs = []
    cbs.append(TensorBoard(os.path.join(log_dir, model_id, 'tb_logs')))

    """
        Save model with lowest validation loss (check on every epoch end if current
        model performs better than the old one).
    """
    cbs.append(ModelCheckpoint(
        '{}/{}/resnet50_min_val_loss.h5'.format(log_dir, model_id),
        save_best_only=True,
        monitor='val_loss', mode='min'))
    cbs.append(
        CSVLogger(filename='{}/{}/train_log.csv'.format(log_dir, model_id)))

    model.fit_generator(train_generator, steps_per_epoch=train_steps,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=val_steps, callbacks=cbs,
                        shuffle=False,
                        )

    model.save('{}/{}/resnet50.h5'.format(log_dir, model_id))


if __name__ == '__main__':
    fire.Fire(train)
