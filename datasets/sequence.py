"""sequence dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import struct
import tensorflow as tf

LOCAL_DIR = "data/sequence/"
SEQUENCE_LENGTH = 10

TRAIN_DATA_SIZE = 1200 # sequences (entries)
TEST_DATA_SIZE = 300

RESUME_TRAINING = False
MULTI_THREADING = True
SKIP_DATA_GENERATION = False
NUM_CLASSES = 4
TARGET_LABELS = ['P01', 'P02', 'P03', 'P04']

NOISE_RANGE = .5
TREND = 10.
OSCILIATION = 5.
np.random.seed = 19831006


def get_params():
    """Dataset params."""
    params = {
        "num_classes": NUM_CLASSES,
        "sequence_len": SEQUENCE_LENGTH,
        "string_labels": TARGET_LABELS
    }
    return params


def prepare():
    """This function will be called once to prepare the dataset."""
    def create_sequence_1(start_value):
        x =  np.array(range(start_value, start_value+SEQUENCE_LENGTH))
        noise = np.random.normal(0, NOISE_RANGE, SEQUENCE_LENGTH)
        y = np.sin(np.pi * x / OSCILIATION) + (x / TREND + noise)
        return y

    def create_sequence_2(start_value):
        x =  np.array(range(start_value, start_value+SEQUENCE_LENGTH))
        noise = np.random.normal(0, NOISE_RANGE, SEQUENCE_LENGTH)
        y = -x + noise
        return y

    def create_sequence_3(start_value):
        x =  np.array(range(start_value, start_value+SEQUENCE_LENGTH))
        y = []

        for x_i in x:
            y_i = 0
            if x_i % 2 == 0:
                y_i = x_i * 2
            else:
                y_i =  - x_i * 2
            y += [y_i]

        return y

    def create_sequence_4(unused_variable):
        return np.random.uniform(-100,100,SEQUENCE_LENGTH)


    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    struct = ['creator', 'label']
    sequences = [
       [create_sequence_1, TARGET_LABELS[0]],
       [create_sequence_2, TARGET_LABELS[1]],
       [create_sequence_3, TARGET_LABELS[2]],
       [create_sequence_4, TARGET_LABELS[3]]
    ]
    sequences = [dict(zip(struct, sequence)) for sequence in sequences]

    patterns = 3
    start = int(-1*TRAIN_DATA_SIZE/(2*patterns))
    end = int(TRAIN_DATA_SIZE/(2*patterns))

    train_file_name = os.path.join(LOCAL_DIR, "train.csv")
    test_file_name = os.path.join(LOCAL_DIR, "test.csv")
    train_file = open(train_file_name, "w")
    test_file = open(test_file_name, "w")

    for sequence in sequences:
        for line_index in range(start, end):
            seq = sequence['creator'](line_index)
            csv_line = ",".join(map(str, seq))
            csv_line += ",{}\n".format(sequence['label'])

            if 0 == line_index % 6:
                test_file.write(csv_line)
            else:
                train_file.write(csv_line)

    train_file.close()
    test_file.close()


def read(split):
    """Create an instance of the dataset object."""
    if tf.estimator.ModeKeys.TRAIN == split:
        input_file = os.path.join(LOCAL_DIR, "train.csv")
    elif tf.estimator.ModeKeys.EVAL == split:
        input_file = os.path.join(LOCAL_DIR, "test.csv")
    else:
        raise Exception("wrong split: {}".format(split))

    if not os.path.exists(input_file):
        raise Exception('File: "{}" not exist'.format(input_file))

    dataset = tf.data.TextLineDataset(filenames=input_file)

    return dataset


def parse(csv_row):
    DEFAULTS = [[0.0] for i in range(0, SEQUENCE_LENGTH)] + [['']]
    # sequence is a list of tensors
    sequence = tf.decode_csv(tf.expand_dims(csv_row, -1), record_defaults=DEFAULTS)

    input_sequence = sequence[:SEQUENCE_LENGTH]  # input elements in the sequence

    output_label = sequence[len(sequence) -1] # output elements in the sequence

    input_tensor = tf.concat(input_sequence, axis=1)

    output_label = tf.squeeze(output_label)

    return {'values': input_tensor}, output_label

if __name__ == "__main__":
    print(tf.__version__)

    num_epochs=1

    prepare()
    dataset = read(tf.estimator.ModeKeys.TRAIN)
    dataset = dataset.batch(TRAIN_DATA_SIZE)
    dataset = dataset.map(lambda csv_row: parse(csv_row))
    dataset = dataset.repeat(num_epochs)
