import os
from uuid import \
    uuid4  # uuid4() generates random unique filename (without extension)

import fire
import imageio
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tqdm import tqdm  # Show progressbar during loop iteration

from ddsm_generation.ddsm_utils import prepare_dataset

# Allow for-loop iteration over tf.Dataset object (reading .tfrecords)
tf.enable_eager_execution()

# Map multi-class labels from .tfrecords into their actual names
labels_dict = {
    0: "negative",
    1: "benign_calcification",
    2: "benign_mass",
    3: "malignant_calcification",
    4: "malignant_mass",
}

# Map binary labels from .tfrecords into their actual names
labels_normal_dict = {
    0: "negative",
    1: "positive",
}


def generate_ddsm(ddsm_data_dir: str,
                  out_dir: str,
                  file_extension: str,
                  multiclass: bool = True,
                  just_preview: bool = False):
    """

    Args:
        ddsm_data_dir: Path to directory with raw ddsm data (should contain .tfrecords and .npy)
        out_dir: directory that will contain ddsm generation results
        file_extension: Extension of the files that will be saved. Can be '.tiff', '.png' or
                        any other valid image file extension
        multiclass: If True, will generate dataset with all the classes split into separate
                    directories. If False, will generate dataset with binary class split
        just_preview: If True, images from .tfrecords will only show them with matplotlib,
                      not save them to the disk. When False, it will actually save the images
                      to files
    """
    # Create a list of all the files from DDSM raw data dir ending with .tfrecords
    ddsm_train = [os.path.join(ddsm_data_dir, file) for file in
                  os.listdir(ddsm_data_dir) \
                  if file.endswith('.tfrecords')]

    # Creates a dataset that reads all examples from all tfrecords files
    ddsm_train_dataset = prepare_dataset(ddsm_train, multiclass=multiclass)

    # Create directories for all the labels that will be used during DDSM dataset generation
    labels = labels_dict if multiclass else labels_normal_dict
    for label in labels.values():
        os.makedirs(os.path.join(out_dir, label), exist_ok=True)

    # Iterate over examples from the ddsm_train_dataset.
    for image, label_idx in tqdm(ddsm_train_dataset,
                                 desc='Processing .tfrecords'):
        if just_preview:
            plt.imshow(image[0, :, :, 0], cmap='gray')
            plt.title('Label {}'.format(int(label_idx)))
            plt.show()
        else:
            # Generate image with random identifier (unique!)
            imageio.imwrite('{}/{}/{}.{}'.format(out_dir,
                                                 labels[int(label_idx)],
                                                 uuid4(),
                                                 file_extension),
                            image[0, :, :, 0])

    # Now that we parsed original training data, let's parse the rest of the data from .npy files
    npy_files = [os.path.join(ddsm_data_dir, file) for file in
                 os.listdir(ddsm_data_dir) \
                 if file.endswith('.npy')]

    for npy_file in npy_files:
        if 'cv10_data.npy' in npy_file:
            ddsm_val_data = np.load(npy_file)
        elif 'cv10_labels.npy' in npy_file:
            ddsm_val_labels = np.load(npy_file)
        elif 'test10_data.npy' in npy_file:
            ddsm_test_data = np.load(npy_file)
        elif 'test10_labels.npy' in npy_file:
            ddsm_test_labels = np.load(npy_file)

    if not multiclass:
        ddsm_val_labels[ddsm_val_labels > 0] = 1
        ddsm_test_labels[ddsm_val_labels > 0] = 1

    for image, label_idx in tqdm(zip(ddsm_val_data, ddsm_val_labels),
                                 desc="Processing ddsm val .npy"):
        # Generate image with random identifier (unique!)
        imageio.imwrite('{}/{}/{}.{}'.format(out_dir,
                                             labels[int(label_idx)],
                                             uuid4(),
                                             file_extension),
                        image[:, :, 0])

    for image, label_idx in tqdm(zip(ddsm_test_data, ddsm_test_labels),
                                 desc="Processing ddsm test .npy"):
        # Generate image with random identifier (unique!)
        imageio.imwrite('{}/{}/{}.{}'.format(out_dir,
                                             labels[int(label_idx)],
                                             uuid4(),
                                             file_extension),
                        image[:, :, 0])


if __name__ == "__main__":
    fire.Fire(generate_ddsm)
