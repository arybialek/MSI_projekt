import errno
import os
import random
from typing import List

import fire
from sklearn.model_selection import train_test_split


def create_symlink(src, dst):
    """

    Args:
        src: Path to source file
        dst: Path to destination symlink that will be created
    """
    try:
        # Try to create a symlink
        os.symlink(src, dst)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # If the symlink already exists, remove it...
            os.remove(dst)
            # And create a new one
            os.symlink(src, dst)


def split(ddsm_data_dir: str,
          out_dir: str,
          file_extension: str,
          filter_classes: List = [],
          train_size: float = 0.7,
          balance_data: bool = True,
          ):
    """

    Args:
        ddsm_data_dir: this directory should contain subdirectories for each class
        out_dir:
        file_extension:
        filter_classes:
        train_size:
        balance_data:

    Returns:

    """
    classes = [dir for dir in os.listdir(ddsm_data_dir)]

    if filter_classes is not None:
        filtered_classes = []
        for cls in classes:
            if cls in filter_classes:
                filtered_classes.append(cls)
        classes = filtered_classes

    if balance_data:
        num_examples = {}
        for cls in classes:
            num_examples[cls] = len(
                [file for file in os.listdir(os.path.join(ddsm_data_dir, cls)) if
                 file.endswith(file_extension)])

    for cls in classes:
        # Returns full paths to files from a given class
        files = [os.path.join(ddsm_data_dir, cls, file) for file in \
                 os.listdir(os.path.join(ddsm_data_dir, cls)) if file.endswith(file_extension)]

        random.shuffle(files)

        if balance_data:
            least_examples = min(num_examples.values())
            files = files[:least_examples]

        """
            Use sklearn library function to automatically split the data. It can't, however,
            directly split the data into three datasets so we need to run it twice. First, split the
            data into train (70%) and combined val and test (30%). Then split val_test into half
            (15% of whole dataset for val and 15% for test).
        """
        train, val_test = train_test_split(files, test_size=(1. - train_size))
        val, test = train_test_split(val_test, test_size=0.5)

        train_dir = os.path.join(out_dir, 'train', cls)
        val_dir = os.path.join(out_dir, 'val', cls)
        test_dir = os.path.join(out_dir, 'test', cls)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        for file in train:
            dst = os.path.join(train_dir, os.path.basename(file))
            create_symlink(src=file, dst=dst)

        for file in val:
            dst = os.path.join(val_dir, os.path.basename(file))
            create_symlink(src=file, dst=dst)

        for file in test:
            dst = os.path.join(test_dir, os.path.basename(file))
            create_symlink(src=file, dst=dst)


if __name__ == '__main__':
    fire.Fire(split)
