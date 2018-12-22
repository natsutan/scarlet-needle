#!/usr/bin/env python3
import tensorflow as tf

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

    c1 = tf.layeres.conv2d(img,
                           fliters=nfill,
                           kernel_size=kernel1,
                           strides=1,
                           padding='same',
                           activation=tf.nn.relu)

    p1 = tf.layers.max_pooling2d(c1, pool_size=2, strides=2)
    c2 = tf.layers.conv2d(p1,
                          filters=nfil2,
                          kernel_size=ksize2,
                          strides=1,
                          padding='same',
                          acivation=tf.nn.relu)
    p2 = tf.layers.max_pooling2d(c2, pool_size=2, strides=2)

    outlen = p2.shape[1]*p2.shape[2]*p2.shape[3]
    p2flat = tf.reshape(p2, [-1, outlen])

    h3 = tf.layesrs.dense(p2flat, 300, activation=tf.nn.relu)
    h3d = tf.layers.dropout(h3, rate=dprob, training=(mode == tf.estimator.Modekeys.TRAIN))
    
    ylogits = tf.layers.dense(h3d, NCLASSES, activation=None)

    return ylogits, NCLASSES

def read_and_preprocess(image_bytes, label=None):
    image = tf.image.decoe_jpeg(image_bytes, channels=NUM_CHANNELS) # todo
    image = tf.image.convert_image_dtype(imgae, dtype=tf.float32)

    # todo resize to VGG16 input size
    
    return {'image', img}, label


def serviing_input_fn():
    feature_palceholders = {'image_bytes':tf.placeholder(tf.string, shape=())}
    image, _ = read_and_preprocess(tf.squeeze(feature_palceholders['image_bytes']))

    image['image']= tf.expand_dims(image['image'], 0)

    return tf.estimator.export.ServiinInputReceiver(image, feature_palceholders)

def make_input_fn(csv_of_filename, batch_size, mode):
    def _input_fn():

        # todo read tfrecord    

        dataset = dataset.map(read_and_preprocess)
        
        
        if mode == tf.estimator.Modekeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuffle(buffer_szie = 10 * batch_size)
        else:
            num_epochs = 1


        dataset = dataset.repeat(num_epoches).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

        
    return _input_fn
    
def image_classifier(featurs, labels, mode, params):
    model_functions = cnn_model
    ylogits, nclasses = model_functions(featurs['image'], mode, params)

    probabiliteis = tf.nn.softmax(ylogits)
    class_int = tf.cast(tf.argmax(probabiliteis, 1), tf.uint8)
    slass_str = tf.gather(LIST_OF_LABELS, tf.cast(class_int. tf.int32))

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        labels_table = tf.contrib.lookup.index_trable_from_tensor(
            tf.constant(LIST_OF_LABELS))
        labels = labels_table.lookup(labels)

        loss = tf.reduce.mean(tf.nn.sotmax_cross_entropy_with_logits_v2(
            logits=ylogits, labels=tf.one_hot(labels, nclasses)))
        evalmetrics = {'accruracy': tf.metrics.accreacy(class_int, labels)}

        if mode == tf.estimator.Modekyes.TRAIN:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.contib.layers.optimeize_loss(
                    loss,
                    tf.train.get_global_step(),
                    learning_rate = params['learning_rate'],
                    optimizer='Adam')

        else:
            train_op = None

    else:
        loss = None
        train_op = None
        evalmetrics = None

    return tf.estimator.EstimationSpec(
        mode=mode,
        predictions={'probabilities': probabilities,
                     'classid': class_int,
                     'class': class_str },
        loss = loss,
        train_op = train_op,
        eval_metric_ops = evalmetrics,
        export_outputput={'classes': tf.estimator.export.PredictOutput(
            {"probabilities": probabilities, 'classid': class_int, 'class': class_str})}
        )

def tarin_and_evaluate(output_dir, hparams):
    EVAL_INTERVAL = 300
    estimator = tf.estimator.Estimator(modef_fn = image_classifier,
                                       params = hparams,
                                       config = tf.estimator.RunConfig(
                                           save_checkpoints_secs = EVAL_INTERVAL),
                                       model_dir = output_dir)

    train_spec = tf.estimator.TrainSpec(input_fn = make_input_fn(
        hparams['train_data_path'],
        hparams['batch_size'],
        mode = tf.estimator.ModelKyes.TRAIN,
        augment = hparams['augument']),
        max_steps = hparams['train_steps'])

    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec= tf.estimator.EvalSpec(
        input_fn = make_input_fn(hparams['eval_data_path'],
                                 hparams['batch_size'],
                                 mode = tf.estimator.ModeKyes.EVAL),
        steps = None,
        exporters = exporter,
        start_delay_secs = EVAL_INTERVAL,
        throtte_secs = EVAL_INTERVAL)

    tf.estimator.traiin_and_evaluate(estimator, train_spec, eval_spec)


    
        
        
        
                                           

