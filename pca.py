import os

import numpy as np
import tensorflow as tf
from sklearn.base import clone
from sklearn.decomposition import PCA
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

log_dir = 'path_to_dir'
os.makedirs(log_dir)
cross_val_dir = 'path_to_directory'

k = 10
target_size = (299, 299)
batch_size = 1
epochs = 40

for fold in range(k):
    fold_dir = os.path.join(cross_val_dir, 'fold_{}/'.format(fold))

    train_val_dir = os.path.join(fold_dir, 'train_val/')
    test_dir = os.path.join(fold_dir, 'test/')

    # Generators.
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_val_dir,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        color_mode='grayscale',
                                                        class_mode='binary',
                                                        shuffle=False,
                                                        )

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=target_size,
                                                      batch_size=batch_size,
                                                      color_mode='grayscale',
                                                      class_mode='binary',
                                                      shuffle=False,
                                                      )

    flat_train_images = []
    train_labels = []

    flat_test_images = []
    test_labels = []

    for _ in range(len(train_generator)):
        image, label = next(train_generator)
        flat_train_images.append(image.flatten())
        train_labels.append(label)

    for _ in range(len(test_generator)):
        image, label = next(test_generator)
        flat_test_images.append(image.flatten())
        test_labels.append(label)

    model = PCA(n_components=2, random_state=0)
    pca_train_images = model.fit_transform(flat_train_images)
    pca_test_images = clone(model).fit_transform(flat_test_images)

    os.makedirs('{}/{}'.format(log_dir, fold))

    np.save('{}/{}/pca_train_images.npy'.format(log_dir, fold),
            pca_train_images)
    np.save('{}/{}/pca_train_labels.npy'.format(log_dir, fold),
            np.array(train_labels))
    np.save('{}/{}/pca_test_images.npy'.format(log_dir, fold), pca_test_images)
    np.save('{}/{}/pca_test_labels.npy'.format(log_dir, fold),
            np.array(test_labels))
