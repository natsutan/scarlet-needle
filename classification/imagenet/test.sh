#!/bin/sh

python3 task.py --train_data_path gs://${BUCKET}/ILSVRC2011/tfrecord/train.tfrecord \
                --eval_data_path gs://${BUCKET}/ILSVRC2011/tfrecord/val.tfrecord \
                --output_dir ~/tmp/cnn
		
