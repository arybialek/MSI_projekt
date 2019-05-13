import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

cross_val_dir = 'path_to_dir'

report = pd.DataFrame(
    index=np.arange(10),
    columns=['Precision', 'Recall', 'Accuracy',
             'F1-score'])

for i in range(10):
    tf.keras.backend.clear_session()

    fold_dir = os.path.join(cross_val_dir, 'fold_{}/'.format(i))
    test_dir = os.path.join(fold_dir, 'test/')

    test_generator = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_generator.flow_from_directory(test_dir,
                                                        target_size=(299, 299),
                                                        batch_size=12,
                                                        color_mode='grayscale',
                                                        class_mode='binary',
                                                        shuffle=False, )

    model = load_model(
        'path_to_directory/{}/resnet50_min_val_loss.h5'.format(
            i))
    predictions = model.predict_generator(test_generator,
                                          steps=len(test_generator), verbose=1)
    ground_truth = test_generator.classes

    predictions[predictions < 0.5] = 0
    predictions[predictions >= 0.5] = 1

    tn, fp, fn, tp = confusion_matrix(ground_truth,
                                      predictions.astype(np.uint8)).ravel()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * (precision * recall) / (precision + recall)

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    report.loc[i] = [precision, recall, accuracy, f1]
    report.to_csv('metrics_resnet.csv', mode='w')
