# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Builds the MNIST network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

This file is used by the various "fully_connected_*.py" files and not meant to
be run.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.stats import truncnorm

import numpy as np
import math

import tensorflow as tf

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

def init_weights(size):
    # we truncate the normal distribution at two times the standard deviation (which is 2)
    # to account for a smaller variance (but the same mean), we multiply the resulting matrix with he desired std
    return np.float32(truncnorm.rvs(-2, 2, size=size)*1.0/math.sqrt(float(size[0])))


def inference(images, Hidden1, Hidden2):

  with tf.name_scope('hidden1'):

    weights = tf.Variable(init_weights([IMAGE_PIXELS, Hidden1]), name='weights', dtype=tf.float32)
    biases = tf.Variable(np.zeros([Hidden1]),name='biases',dtype=tf.float32)
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

  with tf.name_scope('hidden2'):

    weights = tf.Variable(init_weights([Hidden1, Hidden2]),name='weights',dtype=tf.float32)
    biases = tf.Variable(np.zeros([Hidden2]),name='biases',dtype=tf.float32)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  with tf.name_scope('out'):

    weights = tf.Variable(init_weights([Hidden2, NUM_CLASSES]), name='weights',dtype=tf.float32)
    biases = tf.Variable(np.zeros([NUM_CLASSES]), name='biases',dtype=tf.float32)
    logits = tf.matmul(hidden2, weights) + biases

  return logits

def inference_no_bias(images, Hidden1, Hidden2):

  with tf.name_scope('hidden1'):

    weights = tf.Variable(init_weights([IMAGE_PIXELS, Hidden1]), name='weights', dtype=tf.float32)
    hidden1 = tf.nn.relu(tf.matmul(images, weights))

  with tf.name_scope('hidden2'):

    weights = tf.Variable(init_weights([Hidden1, Hidden2]),name='weights',dtype=tf.float32)
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights))

  with tf.name_scope('out'):

    weights = tf.Variable(init_weights([Hidden2, NUM_CLASSES]), name='weights',dtype=tf.float32)
    logits = tf.matmul(hidden2, weights)

  return logits


def loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits, name='xentropy')
  return tf.reduce_mean(cross_entropy, name='xentropy_mean')


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))


def placeholder_inputs(batch_size):
    """Generate placeholder variables to represent the input tensors.
    These placeholders are used as inputs by the rest of the model building
    code and will be fed from the downloaded data in the .run() loop, below.
    Args:
    batch_size: The batch size will be baked into both placeholders.
    Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
    """
    # Note that the shapes of the placeholders match the shapes of the full
    # image and label tensors, except the first dimension is now batch_size
    # rather than the full size of the train or test data sets.
    images_placeholder = tf.placeholder(tf.float32, shape=(None,IMAGE_PIXELS), name='images_placeholder')
    labels_placeholder = tf.placeholder(tf.int32, shape=(None), name='labels_placeholder')
    return images_placeholder, labels_placeholder


def mnist_cnn_model(batch_size):

    # - placeholder for the input Data (in our case MNIST), depends on the batch size specified in C
    data_placeholder, labels_placeholder = placeholder_inputs(batch_size)

    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(data_placeholder, [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels_placeholder, logits=logits)

    eval_correct = evaluation(logits, labels_placeholder)

    global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=global_step)

    return train_op, eval_correct, loss


def mnist_fully_connected_model(batch_size, hidden1, hidden2):
    # - placeholder for the input Data (in our case MNIST), depends on the batch size specified in C
    data_placeholder, labels_placeholder = placeholder_inputs(batch_size)

    # - logits : output of the fully connected neural network when fed with images. The NN's architecture is
    #           specified in '
    logits = inference_no_bias(data_placeholder, hidden1, hidden2)

    # - loss : when comparing logits to the true labels.
    Loss = loss(logits, labels_placeholder)

    # - eval_correct: When run, returns the amount of labels that were predicted correctly.
    eval_correct = evaluation(logits, labels_placeholder)


    # - global_step :          A Variable, which tracks the amount of steps taken by the clients:
    global_step = tf.Variable(0, dtype=tf.float32, trainable=False, name='global_step')

    # - learning_rate : A tensorflow learning rate, dependent on the global_step variable.
    learning_rate = tf.train.exponential_decay(learning_rate=0.1, global_step=global_step,
                                                                           decay_steps=27000, decay_rate=0.1,
                                                                           staircase=False, name='learning_rate')

    # - train_op : A tf.train operation, which backpropagates the loss and updates the model according to the
    #              learning rate specified.
    train_op = training(Loss, learning_rate)

    return train_op, eval_correct, Loss
