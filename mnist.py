"""MNIST dataset preprocessing and specifications."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import numpy as np
import os
from six.moves import urllib
import struct
import tensorflow as tf

REMOTE_URL = "http://yann.lecun.com/exdb/mnist/"
LOCAL_DIR = "data/mnist/"
TRAIN_IMAGE_URL = "train-images-idx3-ubyte.gz"
TRAIN_LABEL_URL = "train-labels-idx1-ubyte.gz"
TEST_IMAGE_URL = "t10k-images-idx3-ubyte.gz"
TEST_LABEL_URL = "t10k-labels-idx1-ubyte.gz"

IMAGE_SIZE = 28
NUM_CLASSES = 10


def get_params():
    """Dataset params."""
    params =  {
        "num_classes": NUM_CLASSES,
    }
    return params


def prepare():
    """This function will be called once to prepare the dataset."""
    if not os.path.exists(LOCAL_DIR):
        os.makedirs(LOCAL_DIR)

    URLs = [
        TRAIN_IMAGE_URL,
        TRAIN_LABEL_URL,
        TEST_IMAGE_URL,
        TEST_LABEL_URL
    ]
    for name in URLs:
        local_file_path = os.path.join(LOCAL_DIR, name)
        remote_file_url = urllib.parse.urljoin(REMOTE_URL, name)
        if not os.path.exists(local_file_path):
            urllib.request.urlretrieve(remote_file_url)


def read(split):
    """Create an instance of the dataset object."""
    if tf.estimator.ModeKeys.TRAIN == split:
        image_urls = TRAIN_IMAGE_URL
        label_urls = TRAIN_LABEL_URL
    elif tf.estimator.ModeKeys.EVAL == split:
        image_urls = TEST_IMAGE_URL
        label_urls = TEST_LABEL_URL
    else:
        raise Exception("wrong split: {}".format(split))

    with gzip.open(LOCAL_DIR + image_urls, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(num * rows * cols), dtype=np.uint8)
        images = np.reshape(images, [num, rows, cols, 1])
        print("Loaded %d images of size [%d, %d]." % (num, rows, cols))

    with gzip.open(LOCAL_DIR + label_urls, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(num), dtype=np.int8)
        print("Loaded %d labels." % num)

    return tf.data.Dataset.from_tensor_slices((images, labels))


def parse(image, label):
    """Parse input record to features and labels."""
    features = tf.to_float(image) / 255.0
    labels = tf.to_int64(label)

    return features, labels
