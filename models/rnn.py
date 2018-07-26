"""Simple recurrent neural network classififer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def get_params():
    """Model params."""
    params = {
        "num_epochs": 100,
        "batch_size": 200,
        "hidden_units": [8, 4],
        "forget_bias": 1.0,
        "max_steps": (1200/200)*100
    }

    return params


def model(features, labels, mode, params):
    """CNN classifier model."""
    # Reformat input shape to become a sequence
    inputs = tf.split(features['values'], params.sequence_len, 1)

    # configure the RNN
    rnn_layers = [
        tf.nn.rnn_cell.LSTMCell(
            num_units=size,
            forget_bias=params.forget_bias,
            activation=tf.nn.tanh
        ) for size in params.hidden_units
    ]

    # create a RNN cell composed sequentially of a number of RNNCells
    multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)

    outputs, _ = tf.nn.static_rnn(
        cell=multi_rnn_cell,
        inputs=inputs,
        dtype=tf.float32
    )

    # slice to keep only the last cell of the RNN
    outputs = outputs[-1]

    logits = tf.layers.dense(
        inputs=outputs,
        units=params.num_classes,
        activation=None
    )

    predicted_indices = tf.argmax(input=logits, axis=1)
    probabilities = tf.nn.softmax(logits, name='softmax_tensor')
    predictions = {
        'class': tf.gather(params.string_labels, predicted_indices),
        'probabilities': probabilities
    }

    # Return accuracy and area under ROC curve metrics
    labels_one_hot = tf.one_hot(
        indices=labels,
        depth=len(params.string_labels),
        on_value=True,
        off_value=False,
        dtype=tf.bool
    )
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels, predicted_indices),
        'auroc': tf.metrics.auc(labels_one_hot, probabilities)
    }

    # Calculate loss using softmax cross entropy
    #loss = tf.losses.sparse_softmax_cross_entropy(
    #    labels=labels,
    #    logits=logits
    #)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels
    )
    loss = tf.reduce_mean(cross_entropy)

    tf.summary.scalar('loss', loss)

    return predictions, eval_metric_ops, loss
