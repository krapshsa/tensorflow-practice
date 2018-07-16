"""Simple convolutional neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def get_params():
    """Model params."""
    params = {
        "drop_rate": 0.5
    }

    return params


def model(features, labels, mode, params):
    """CNN classifier model."""

    tf.summary.image("images", features)

    if mode == tf.estimator.ModeKeys.TRAIN:
        drop_rate = params.drop_rate
    else:
        drop_rate = 0.0

    for i, filters in enumerate([32, 64, 128]):
        features = tf.layers.conv2d(
            features,
            filters=filters,
            kernel_size=3,
            padding="same",
            name="conv_%d" % (i + 1)
        )
        features = tf.layers.max_pooling2d(
            inputs=features,
            pool_size=2,
            strides=2,
            padding="same",
            name="pool_%d" % (i + 1)
        )

    features = tf.contrib.layers.flatten(features)

    features = tf.layers.dropout(features, drop_rate)
    features = tf.layers.dense(features, 512, name="dense_1")

    features = tf.layers.dropout(features, drop_rate)
    logits = tf.layers.dense(
        features,
        params.num_classes,
        activation=None,
        name="dense_2"
    )

    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')
    predictions = {
        'classes': predicted_indices,
        'probabilities': probabilities
    }

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels, predicted_indices)
    }

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    return predictions, eval_metric_ops, loss
