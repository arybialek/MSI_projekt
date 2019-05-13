from typing import List, Union, Tuple

import tensorflow as tf
from functools import partial


def parse_example(encoded_data: tf.image,
                  multiclass: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply the 'parser' on each record read from all tfrecords files.
       Parse records into data.

    label - full multi-class labels, 0 is negative, 1 is benign calcification,
            2 is benign mass, 3 is malignant calcification, 4 is malignant mass

    label_normal - 0 for negative and 1 for positive

    Args:
        encoded_data: serialized/encoded data read by map() function from
                      tf.data API passed to this parse_example function to
                      extract actual data
        multiclass: should the function return multiclass label or merge it into a single class

    Returns:
        Tuple containing image and label.
    """
    features = tf.parse_single_example(
        encoded_data,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'label_normal': tf.FixedLenFeature([], tf.int64),
            'image': tf.FixedLenFeature([], tf.string)
        })

    # Extract the data.
    if multiclass:
        label = features['label']
    else:
        label = features['label_normal']

    # Flat image of numbers.
    image = tf.decode_raw(features['image'], tf.uint8)

    # Reshape and scale the image.
    image = tf.reshape(image, [299, 299, 1])
    return image, label


def normalize(image, label):
    """

    Args:
        image
        label

    Returns:
        Tuple of normalized image tensor and label.

    """
    image = tf.to_float(image)
    norm_image = tf.div(tf.subtract(image, tf.reduce_min(image)),
                        tf.subtract(tf.reduce_max(image), tf.reduce_min(image)))
    return norm_image, label


def prepare_dataset(filenames: Union[str, List[str]],
                    multiclass: bool = False) -> tf.data.Dataset:
    """Prepare data generator from tfrecords filenames.

    Args:
        filenames: path or a list of paths to tfrecords files
        multiclass: should multiclass label be used instead of binary one

    Returns:
        Dataset generator.
    """
    # Create tf.data generator.
    dataset = tf.data.TFRecordDataset(filenames)
    parser = partial(parse_example, multiclass=multiclass)
    return dataset.map(parser).batch(1)
