#!/usr/bin/env python3
import sys
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

LIST_OF_LABELS = ['bottle','aeroplane', 'cat', 'bus', 'horse', 'monitor', 'motorbike', 'bicycle',
                  'cow', 'dinint_table', 'potted_plants', 'chair', 'sofa', 'bird', 'dog', 'train', 
                  'person', 'sheep', 'boat', 'car']

HEIGHT = 224
WIDTH = 224
NUM_CHANNELS = 3
NCLASSES = len(LIST_OF_LABELS)

n_epoch = 3
batch_size = 10

def cnn_model(img, mode):
    ksize1 = 5
    ksize2 = 5
    nfil1 = 10
    nfil2 = 20

    dprob = 0.25

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

    print("img shape: ", img.shape)
    print("p2.shape :", p2.shape)
    print("outlen: ", outlen)
    print("p2flat :", p2flat.shape)
    print("NCLASSES: ", NCLASSES)
#    sys.exit(0)

    h3 = tf.layers.dense(p2flat, 300, activation=tf.nn.relu)
    h3d = tf.layers.dropout(h3, rate=dprob, training=(mode == tf.estimator.ModeKeys.TRAIN))
    
    ylogits = tf.layers.dense(h3d, NCLASSES, activation=None)

    return ylogits

#x = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, 3])

# 正解値
#y_ = tf.placeholder(tf.float32, [None, NCLASSES])


dataset = tf.data.TFRecordDataset('D:/data/tfrecord/train_224x224_n50.tfrecord')
dataset = dataset.batch(batch_size=batch_size)

iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)  # イテレータを作成

y = cnn_model(x, tf.estimator.ModeKeys.TRAIN)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(n_epoch):
        print("epoch ", epoch)
