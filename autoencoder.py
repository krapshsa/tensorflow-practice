from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

tf.reset_default_graph()

# Training Parameters
batch_size = 256
max_steps = 1000
display_steps = 1000
learning_rate = 0.01


# Network Parameters
num_hidden_1 = 256  # 1st layer num features
num_hidden_2 = 128  # 2nd layer num features (the latent dim)

time_steps = 28
feature_size = 28

FLAGS = tf.flags.FLAGS

HPARAMS = {
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "decay_steps": 10000,
    "batch_size": 128,
    "min_eval_frequency": FLAGS.eval_frequency
}


def get_params():
    """Aggregates and returns hyper parameters."""
    hparams = HPARAMS

    hparams = tf.contrib.training.HParams(**hparams)
    hparams.parse(FLAGS.hparams)

    return hparams


def make_model_fn():
    def autoencoder(x):
        # Construct model
        rnn_layers = [
            tf.nn.rnn_cell.LSTMCell(
              num_units=units,
              forget_bias=1.0,
              activation=tf.nn.tanh
            ) for units in [num_hidden_1, num_hidden_2, num_hidden_1, feature_size]
        ]

        # create a RNN cell composed sequentially of a number of RNNCells
        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        inputs = tf.unstack(
            value=x,
            num=time_steps,
            axis=1
        )
        outputs, states = tf.nn.static_rnn(
            cell=multi_rnn_cell,
            inputs=inputs,
            dtype=tf.float32
        )
        return outputs

    def _model_fn(features, labels=None, mode=None, params=None):
        global_step = tf.train.get_or_create_global_step()

        # slice to keep only the last cell of the RNN
        outputs = autoencoder(features)
        stack_outputs = tf.stack(
            values=outputs,
            axis=1
        )

        # Prediction
        y_pred = stack_outputs
        # Targets (Labels) are the input data.
        y_true = features

        print(tf.convert_to_tensor(outputs).get_shape())
        print(y_pred.get_shape())
        print(y_true.get_shape())

        # Define loss and optimizer, minimize the squared error
        loss = tf.reduce_mean(tf.square(tf.subtract(y_true, y_pred)))
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
        train_op = optimizer.minimize(loss)
        eval_metric_ops = {
        }

        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=global_step,
                learning_rate=params.learning_rate,
                optimizer=params.optimizer
            )

            estimator_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op
            )

        if mode == tf.estimator.ModeKeys.EVAL:
            estimator_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metric_ops=eval_metric_ops,
            )

        '''
        if mode == tf.estimator.ModeKeys.PREDICT:
            estimator_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                export_outputs=export_outputs
            )
        '''

        return estimator_spec

    return _model_fn


def make_input_fn(mode, batch_size, time_steps, feature_size):
    def _input_fn():
        if mode == tf.estimator.ModeKeys.EVAL:
            images = mnist.validation.images
        elif mode == tf.estimator.ModeKeys.TRAIN:
            images = mnist.train.images

        sequence = images.reshape((time_steps, feature_size))
        return tf.estimator.inputs.numpy_input_fn(
            batch_size=batch_size,
            x={'X': sequence}
        )

    return _input_fn


def main(unused_argv):
    session_config = tf.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.gpu_options.allow_growth = True
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        save_checkpoints_secs=None,
        session_config=session_config
    )

    estimator = tf.estimator.Estimator(
        model_fn=make_model_fn(),
        config=run_config,
        params=hparams
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=make_input_fn(tf.estimator.ModeKeys.TRAIN, batch_size, time_steps, feature_size),
        max_steps=FLAGS.max_steps
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=make_input_fn(tf.estimator.ModeKeys.EVAL, batch_size, time_steps, feature_size),
        steps=FLAGS.eval_steps
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "main":
    tf.app.run()
