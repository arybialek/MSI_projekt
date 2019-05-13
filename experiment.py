import random

import fire
import tensorflow as tf

from train_resnet import train


def experiment(log_dir: str,
               epoch: int = 200,
               seed: int = 1234,
               ):
    # PHASE 1:
    lr = 1.28006404e-05
    batch_size = 12
    rate = [0.2, 0.3, 0.5]
    random.seed(seed)

    for rat in rate:
        train(
            data_dir='path_to_directory',
            model_id=str(rat),
            batch_size=batch_size,
            rate=rat,
            target_size=(299, 299),
            lr=lr,
            epochs=epoch,
            seed=seed,
            log_dir=log_dir
        )
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    fire.Fire(experiment)
