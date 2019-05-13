import os

import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint, \
    CSVLogger
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

data_path = 'path_to_directory'
log_dir = 'path_to_directory'

k = 10
target_size = (299, 299)
lr = 1.28006404e-05
batch_size = 12
epochs = 40
model_id = 'vanilla_resnet'

train = os.path.join(data_path, 'train/')
val = os.path.join(data_path, 'val/')
test = os.path.join(data_path, 'test/')

classes = os.listdir(train)
data = {classes[0]: [], classes[1]: []}

for cls in classes:
    data[cls] += [os.path.join(train, cls, file) for file in
                  os.listdir(os.path.join(train, cls)) if
                  file.endswith('.png')]
    data[cls] += [os.path.join(val, cls, file) for file in
                  os.listdir(os.path.join(val, cls)) if
                  file.endswith('.png')]
    data[cls] += [os.path.join(test, cls, file) for file in
                  os.listdir(os.path.join(test, cls)) if
                  file.endswith('.png')]

random.shuffle(data[classes[0]])
random.shuffle(data[classes[1]])

num_samples_fold = [len(data[classes[0]]) // k, len(data[classes[1]]) // k]

cross_val_dir = 'path_to_directory'
#os.makedirs(cross_val_dir, exist_ok=True)

for fold in range(k):
    test_data_fold = {}
    train_data_fold = {}
    for n, cls in enumerate(classes):
        train_data_fold[cls] = data[cls][
                               num_samples_fold[n] * fold: num_samples_fold[n] * (fold + 1)
                               ]
        test_data_fold[cls] = \
            data[cls][:num_samples_fold[n] * fold] + data[cls][num_samples_fold[n] * (fold + 1):
                                                               num_samples_fold[n] * k]

    train_data = {}
    val_data = {}
    for cls in classes:
        train_data[cls] = train_data_fold[cls][:(len(train_data_fold[cls]) // 2)]
        val_data[cls] = train_data_fold[cls][(len(train_data_fold[cls]) // 2):]

    fold_dir = os.path.join(cross_val_dir, 'fold_{}/'.format(fold))
    os.makedirs(fold_dir, exist_ok=True)
    os.makedirs(os.path.join(fold_dir, 'test'))
    os.makedirs(os.path.join(fold_dir, 'test', classes[0]))
    os.makedirs(os.path.join(fold_dir, 'test', classes[1]))

    os.makedirs(os.path.join(fold_dir, 'train'))
    os.makedirs(os.path.join(fold_dir, 'train', classes[0]))
    os.makedirs(os.path.join(fold_dir, 'train', classes[1]))

    os.makedirs(os.path.join(fold_dir, 'val'))
    os.makedirs(os.path.join(fold_dir, 'val', classes[0]))
    os.makedirs(os.path.join(fold_dir, 'val', classes[1]))

    for cls in test_data_fold.keys():
        for file in test_data_fold[cls]:
            src = file
            dst = os.path.join(fold_dir, 'test', cls, os.path.basename(file))
            try:
                os.symlink(src, dst)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(dst)
                    os.symlink(src, dst)

    for cls in train_data.keys():
        for file in train_data[cls]:
            src = file
            dst = os.path.join(fold_dir, 'train', cls, os.path.basename(file))
            try:
                os.symlink(src, dst)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(dst)
                    os.symlink(src, dst)

    for cls in val_data.keys():
        for file in val_data[cls]:
            src = file
            dst = os.path.join(fold_dir, 'val', cls, os.path.basename(file))
            try:
                os.symlink(src, dst)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    os.remove(dst)
                    os.symlink(src, dst)

    train_val_dir = os.path.join(fold_dir, 'train_val/')
    # val_dir = os.path.join(fold_dir, 'val/')
    test_dir = os.path.join(fold_dir, 'test/')

    # Generators.
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    # val_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(train_val_dir,
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

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=target_size,
                                                      batch_size=batch_size,
                                                      color_mode='grayscale',
                                                      class_mode='binary',
                                                      shuffle=False,
                                                      )

    train_steps = len(train_generator)
    # val_steps = len(validation_generator)
    test_steps = len(test_generator)

    # Instantiating a small convent for positives and negatives classification.
    model = Sequential()
    model.add(ResNet50(include_top=False, pooling='avg', weights=None,
                       input_shape=(target_size[0], target_size[1], 1)))
    # Version from tf.keras.applications
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(lr=lr),
                  metrics=['acc'],
                  )

    cbs = []
    cbs.append(
        TensorBoard(os.path.join(log_dir, model_id, str(fold), 'tb_logs')))

    """
        Save model with lowest validation loss (check on every epoch end if current
        model performs better than the old one).
    """
    cbs.append(ModelCheckpoint(
        '{}/{}/{}/resnet50_min_val_loss.h5'.format(log_dir, model_id,
                                                   str(fold)),
        save_best_only=True,
        monitor='loss', mode='min'))
    cbs.append(
        CSVLogger(filename='{}/{}/{}/train_log.csv'.format(log_dir, model_id,
                                                           str(fold))))

    model.fit_generator(train_generator, steps_per_epoch=train_steps,
                        epochs=epochs,
                        # validation_data=validation_generator,
                        # validation_steps=val_steps,
                        callbacks=cbs,
                        shuffle=False,
                        )

    model.save('{}/{}/{}/resnet50.h5'.format(log_dir, model_id, str(fold)))

    test_loss, test_acc = model.evaluate_generator(test_generator,
                                                   steps=test_steps)
    with open('cross_val_results_40.txt', 'a+') as file:
        file.write('Fold: {}\n'.format(fold))
        file.write('Test loss: {}\n'.format(test_loss))
        file.write('Test accuracy: {}\n'.format(test_acc))

    tf.keras.backend.clear_session()
