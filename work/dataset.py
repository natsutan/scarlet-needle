import tensorflow as tf


def parse_csv(line):
    [text, num] = line.decode('utf-8').split(',')
    return text, num

data = tf.data.TextLineDataset('test.csv').map(lambda x: tf.py_func(parse_csv, [x], [tf.string, tf.int64]))

