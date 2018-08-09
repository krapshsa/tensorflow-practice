from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# Training Parameters
batch_size=256
max_steps=1000
display_steps=1000
learning_rate=0.01


# Network Parameters
num_hidden_1 = 256 # 1st layer num features
num_hidden_2 = 128 # 2nd layer num features (the latent dim)
num_input = 784 # MNIST data input (img shape: 28*28)


# tf Graph input (only pictures)
X = tf.placeholder("float", [None, 28, 28])


# Building the encoder
def encoder():
    # Encoder Hidden layer with sigmoid activation #1
    #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']), biases['encoder_b1']))
    rnn_layer_1 = tf.nn.rnn_cell.LSTMCell(
      num_units=num_hidden_1,
      forget_bias=1.0,
      activation=tf.nn.tanh
    )

    # Encoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']), biases['encoder_b2']))
    rnn_layer_2 = tf.nn.rnn_cell.LSTMCell(
      num_units=num_hidden_2,
      forget_bias=1.0,
      activation=tf.nn.tanh
    )

    return [rnn_layer_1, rnn_layer_2]


# Building the decoder
def decoder():
    # Decoder Hidden layer with sigmoid activation #1
    #layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']), biases['decoder_b1']))
    rnn_layer_1 = tf.nn.rnn_cell.LSTMCell(
      num_units=num_hidden_1,
      forget_bias=1.0,
      activation=tf.nn.tanh
    )
    # Decoder Hidden layer with sigmoid activation #2
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']), biases['decoder_b2']))
    rnn_layer_2 = tf.nn.rnn_cell.LSTMCell(
      num_units=num_input,
      forget_bias=1.0,
      activation=tf.nn.tanh
    )

    return [rnn_layer_1, rnn_layer_2]

# Construct model
encoder_layers = encoder()
decoder_layers = decoder()
rnn_layers = encoder_layers + decoder_layers

# create a RNN cell composed sequentially of a number of RNNCells
multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
inputs = tf.unstack(X, 28, 1)
outputs, _ = tf.nn.static_rnn(
    cell=multi_rnn_cell,
    inputs=inputs,
    dtype=tf.float32
)

# slice to keep only the last cell of the RNN
outputs = outputs[-1]


# Prediction
y_pred = outputs
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, max_steps+1):
        # Prepare Data
        # Get the next batch of MNIST data (only images are needed, not labels)
        batch_x, _ = mnist.train.next_batch(batch_size)
        sequence_x = batch_x.reshape((batch_size, 28, 28))

        # Run optimization op (backprop) and cost op (to get loss value)
        _, l = sess.run([optimizer, loss], feed_dict={X: sequence_x})
        # Display logs per step
        if i % display_steps == 0 or i == 1:
            print('Step %i: Minibatch Loss: %f' % (i, l))