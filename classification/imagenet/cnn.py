#!/usr/bin/env python3
import sys
import tensorflow as tf
from PIL import Image

import numpy as np

tf.enable_eager_execution()
print('eager mode in cnn: ', tf.executing_eagerly())

tf.logging.set_verbosity(tf.logging.INFO)

LIST_OF_LABELS = ['bottle','aeroplane', 'cat', 'bus', 'horse', 'monitor', 'motorbike', 'bicycle',
                  'cow', 'dinint_table', 'potted_plants', 'chair', 'sofa', 'bird', 'dog', 'train', 
                  'person', 'sheep', 'boat', 'car']

HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
NCLASSES = len(LIST_OF_LABELS)

def cnn_model(img, mode, hparams):
    ksize1 = 5
    ksize2 = 5
    nfil1 = 10
    nfil2 = 20

    dprob = hparams.get('dprob', 0.25)

    c1 = tf.layers.conv2d(img,
                           filters=nfil1,
                           kernel_size=ksize1,
                           strides=1,
                           padding='same',
                           activation=tf.nn.relu)

    p1 = tf.layers.max_pooling2d(c1, pool_size=2, strides=2)
    c2 = tf.layers.conv2d(p1,
                          filters=nfil2,
                          kernel_size=ksize2,
                          strides=1,
                          padding='same',
                          activation=tf.nn.relu)
    p2 = tf.layers.max_pooling2d(c2, pool_size=2, strides=2)

    outlen = p2.shape[1]*p2.shape[2]*p2.shape[3] 
    p2flat = tf.reshape(p2, [-1, outlen]) # flattened
#    p2flat = tf.reshape(p2, [-1, 6272]) # flattened

    print("img shape: ", img.shape)
    print("p2.shape :", p2.shape)
    print("outlen: ", outlen)
    print("p2flat :", p2flat.shape)
    print("NCLASSES: ", NCLASSES)
#    sys.exit(0)


    h3 =  tf.layers.dense(p2flat, 300, activation=tf.nn.relu)
    h3d = tf.layers.dropout(h3, rate=dprob, training=(mode == tf.estimator.ModeKeys.TRAIN))
    
    ylogits = tf.layers.dense(h3d, NCLASSES, activation=None)

    return ylogits, NCLASSES

def _read_and_preprocess(record):
    example = tf.train.Example()
    example.ParseFromString(record)

    # decode record
    height = example.features.feature['height'].int64_list.value[0]
    width = example.features.feature['width'].int64_list.value[0]
    label = example.features.feature['label'].int64_list.value[0]
    image = example.features.feature['image'].bytes_list.value[0]
    
    image_np = np.fromstring(image, dtype=np.uint8)
    image_np = image_np.reshape([height, width, 3])
    img = Image.fromarray(image_np)
    img_resized = img.resize([HEIGHT, WIDTH], Image.BILINEAR)
    
#    img.save("tf.jpg")
#    img_resized.save("tf_vgg16.jpg")

    return {'image', img_resized}, label


def read_and_preprocess(record):
    tf.enable_eager_execution()
    
    features = {
        'label':     tf.FixedLenFeature([], tf.int64),
        'height':    tf.FixedLenFeature([], tf.int64),
        'width':     tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
    }
    parsed = tf.parse_single_example(record, features)

    # decode record                                                                                                                                                                                               
    height = parsed['height']
    width = parsed['width']
    label = tf.cast(parsed['label'], tf.int64)
    image_raw = tf.decode_raw(parsed['image'], tf.float32)

    image = tf.reshape(image_raw, [height, width, 3])
    #img_resized = tf.image.resize_images(image, (HEIGHT, WIDTH))
    img_resized = tf.image.resize_image_with_pad(image, HEIGHT, WIDTH)

    print('raw image shape', image_raw.shape)
    print('height: ', height, ' width: ', width)
    print('image shape: ', image.shape)
    print('resized image shape: ', img_resized.shape)
    sys.exit(1)
    
#    print('.')

    
    return  {'image' : img_resized}, label


def serving_input_fn():
    feature_palceholders = {'image_bytes':tf.placeholder(tf.string, shape=())}
    image, _ = read_and_preprocess(tf.squeeze(feature_palceholders['image_bytes']))

    image['image']= tf.expand_dims(image['image'], 0)

    
    return tf.estimator.export.ServiinInputReceiver(image, feature_palceholders)

def make_input_fn(gcs_path, batch_size, mode):
    def _input_fn():
        tf.enable_eager_execution()
        
        # todo read tfrecord
        print('eager mode: ', tf.executing_eagerly())
        print(gcs_path)
        print('batch_size: ', batch_size)

        # old version
        # convert from tfrecord to tf.data.dataset
        #tfrecord = tf.gfile.Open(gcs_path, 'rb')
        # dataset = tf.python_io.tf_record_iterator(gcs_path).map(_read_and_preprocess)

        
        dataset = tf.data.TFRecordDataset(gcs_path).map(read_and_preprocess)

        
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(buffer_size = batch_size * 10)
        else:
            num_epochs = 1


        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

        
    return _input_fn


def image_classifier(features, labels, mode, params):
  model_function = cnn_model 
  ylogits, nclasses = model_function(features['image'], mode, params)

  probabilities = tf.nn.softmax(ylogits)
  class_int = tf.cast(tf.argmax(probabilities, 1), tf.uint8)
  class_str = tf.gather(LIST_OF_LABELS, tf.cast(class_int, tf.int32))
  
  if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
    #convert string label to int
    #labels_table = tf.contrib.lookup.index_table_from_tensor(
    #  tf.constant(LIST_OF_LABELS))
    #labels = labels_table.lookup(labels)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=ylogits, labels=tf.one_hot(labels, nclasses)))
    evalmetrics =  {'accuracy': tf.metrics.accuracy(class_int, labels)}
    if mode == tf.estimator.ModeKeys.TRAIN:
      # this is needed for batch normalization, but has no effect otherwise
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
         train_op = tf.contrib.layers.optimize_loss(
             loss, 
             tf.train.get_global_step(),
             learning_rate=params['learning_rate'],
             optimizer="Adam")
    else:
      train_op = None
  else:
    loss = None
    train_op = None
    evalmetrics = None
 
  return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions={"probabilities": probabilities, 
                     "classid": class_int, "class": class_str},
        loss=loss,
        train_op=train_op,
        eval_metric_ops=evalmetrics,
        export_outputs={'classes': tf.estimator.export.PredictOutput(
            {"probabilities": probabilities, "classid": class_int, 
             "class": class_str})}
    )

def train_and_evaluate(output_dir, hparams):
  EVAL_INTERVAL = 300 #every 5 minutes    
  estimator = tf.estimator.Estimator(model_fn = image_classifier,
                                     params = hparams,
                                     config= tf.estimator.RunConfig(
                                         save_checkpoints_secs = EVAL_INTERVAL),
                                     model_dir = output_dir)
  train_spec = tf.estimator.TrainSpec(input_fn = make_input_fn(
                                        hparams['train_data_path'],
                                        hparams['batch_size'],
                                        mode = tf.estimator.ModeKeys.TRAIN
                                        ),
                                      max_steps = hparams['train_steps'])
  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
  eval_spec = tf.estimator.EvalSpec(input_fn = make_input_fn(
                                        hparams['eval_data_path'],
                                        hparams['batch_size'],
                                        mode = tf.estimator.ModeKeys.EVAL),
                                    steps = None,
                                    exporters = exporter,
                                    start_delay_secs = EVAL_INTERVAL,
                                    throttle_secs = EVAL_INTERVAL)
  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  

    
