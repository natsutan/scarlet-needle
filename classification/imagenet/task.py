from __future__ import absolute_import

import argparse
import os
import json
import sys

import cnn
import tensorflow as tf

tf.enable_eager_execution()
print('eager mode: ', tf.executing_eagerly())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--batch_size',
        help='batch size',
        type=int,
        default = 100
        )

    parser.add_argument(
        '--learning_rate',
        help='initial learning rate',
        type= float,
        default=0.01        
        )


    parser.add_argument(
        '--train_steps',
        help="""\
        Steps to run the training job for. A step is one batch-size,\
        """,
        type=int,
        default=100
    )
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=True
    )
    parser.add_argument(
        '--train_data_path',
        help='location of train file containing eval URLs',
        required=True
#        default='gs://cloud-ml-data/img/flower_photos/train_set.csv'
    )
    parser.add_argument(
        '--eval_data_path',
        help='location of eval file containing img URLs',
        required=True
#        default='gs://cloud-ml-data/img/flower_photos/eval_set.csv'
    )

    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    
    # optional hyperparameters used by cnn
    parser.add_argument(
        '--dprob',
        help='dropout probability for CNN',
        type=float,
        default=0.25)


    args = parser.parse_args()
    hparams = args.__dict__

    output_dir = hparams.pop('output_dir')
    output_dir = os.path.join(
        output_dir,
        json.loads(
            os.environ.get('TF_CONFIG', '{}')
        ).get('task', {}).get('traial', '')
    )
        

    cnn.train_and_evaluate(output_dir, hparams)
        
    

if __name__ == '__main__':
    main()

