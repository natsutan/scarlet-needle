#!/usr/bin/env python3
# https://www.dlology.com/blog/how-to-leverage-tensorflows-tfrecord-to-train-keras-model/
import sys
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

#path_tfrecords_train = 'gs://${BUCKET}/ILSVRC2011/tfrecord/train.tfrecord'
# path_tfrecords_test = 'gs://${BUCKET}/ILSVRC2011/tfrecord/val.tfrecord'

path_tfrecords_train = 'D:/data/tfrecordtrain/train_224x224_n50.tfrecord'
path_tfrecords_test = 'D:/data/tfrecordtrain/val_224x224_n50.tfrecord'


tf.logging.set_verbosity(tf.logging.INFO)

LIST_OF_LABELS = ['bottle','aeroplane', 'cat', 'bus', 'horse', 'monitor', 'motorbike', 'bicycle',
                  'cow', 'dinint_table', 'potted_plants', 'chair', 'sofa', 'bird', 'dog', 'train', 
                  'person', 'sheep', 'boat', 'car']

HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
NCLASSES = len(LIST_OF_LABELS)

def cnn_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, NUM_CHANNELS)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(NCLASSES, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])
    return model

def imgs_input_fn(filenames, perform_shuffle=False, repeat_count=1, batch_size=1):
    def _parse_function(serialized):
        features = \
            {
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            }
        # Parse the serialized data so we get a dict with our data.
        parsed_example = tf.parse_single_example(serialized=serialized,
                                                 features=features)
        # Get the image as raw bytes.
        image_shape = tf.stack([HEIGHT, WIDTH, NUM_CHANNELS])
        image_raw = parsed_example['image']
        #label = to_categorical(parsed_example['label'], NCLASSES)
       # label = tf.cast(['label'], tf.int64)
        # Decode the raw bytes so it becomes a tensor with type.
        image = tf.decode_raw(image_raw, tf.uint8)
        image = tf.cast(image, tf.float32)
        image = tf.reshape(image, image_shape)
        label = tf.cast(label, tf.int64)
        d = image, label
        return d

    dataset = tf.data.TFRecordDataset(filenames=filenames)
    # Parse the serialized data in the TFRecords files.
    # This returns TensorFlow tensors for the image and labels.
    dataset = dataset.map(_parse_function)
    if perform_shuffle:
        # Randomizes input using a window of 256 elements (read into memory)
        dataset = dataset.shuffle(buffer_size=256)
    dataset = dataset.repeat(repeat_count)  # Repeats dataset this # times
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels

keras_model = cnn_model()
est_cnn = tf.keras.estimator.model_to_estimator(keras_model=keras_model, model_dir="model")

train_spec = tf.estimator.TrainSpec(input_fn=lambda: imgs_input_fn(path_tfrecords_train,
                                                                   perform_shuffle=True,
                                                                   repeat_count=5,
                                                                   batch_size=20),
                                    max_steps=500)
eval_spec = tf.estimator.EvalSpec(input_fn=lambda: imgs_input_fn(path_tfrecords_test,
                                                                 perform_shuffle=False,
                                                                 batch_size=1))
#tf.estimator.train_and_evaluate(est_cnn, train_spec, eval_spec)

import time
start_time = time.time()
print('start')
tf.estimator.train_and_evaluate(est_cnn, train_spec, eval_spec)
print("--- %s seconds ---" % (time.time() - start_time))
