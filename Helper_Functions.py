import tensorflow as tf
from tensorflow.examples.tutorials.mnist import mnist
import numpy as np
from scipy.stats import truncnorm
import math

def Assignements(dic):
  return [tf.assign(var, dic[Vname_to_Pname(var)]) for var in tf.trainable_variables()]

def Vname_to_Pname(var):
  return var.name[:var.name.find(':')] + '_placeholder'

def Vname_to_FeedPname(var):
  return var.name[:var.name.find(':')] + '_placeholder:0'

def Vname_to_Vname(var):
  return var.name[:var.name.find(':')]


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
  images_placeholder = tf.placeholder(tf.float32, shape=(None,
                                                         mnist.IMAGE_PIXELS), name='images_placeholder')
  labels_placeholder = tf.placeholder(tf.int32, shape=(None), name='labels_placeholder')
  return images_placeholder, labels_placeholder

def fill_feed_dict(data_set, images_pl, labels_pl, FLAGS):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size` examples.
  images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      images_pl: images_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def init_weights(size):
    # we truncate the normal distribution at two times the standard deviation (which is 2)
    # to account for a smaller variance (but the same mean), we multiply the resulting matrix with he desired std
    return np.float32(truncnorm.rvs(-2, 2, size=size)*1.0/math.sqrt(float(size[0])))

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set,
            FLAGS):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder,
                               FLAGS)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = float(true_count) / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))

def get_random_init(shape):
    """
    This function is used to initialize weights of a multilayer perceptron
    :param shape: shape of
    :return: a matrix of shape 'shape' randomly created according to a uniform distribution characterized as seen below
    """
    low = -1*np.sqrt(6.0/(shape[0]+shape[1]))
    high = 1*np.sqrt(6.0/(shape[0]+shape[1]))
    return tf.random_uniform(shape,minval=low,maxval=high, dtype=tf.float64)

def global_model_placeholder(model_dict):
    #hidden1weights = tf.placeholder(model_dict['hidden1/weights:0'].shape)
    #hidden2weights = tf.placeholder(model_dict['hidden2/weights:0'].shape)
    #hidden1biases = tf.placeholder(model_dict['hidden1/biases:0'].shape)
    #hidden2biases = tf.placeholder(model_dict['hidden2/biases:0'].shape)
    #outweights = tf.placeholder(model_dict['out/weights:0'].shape)
    #outbiases = tf.placeholder(model_dict['out/biases:0'].shape)
    dict_of_global_model_placeholders= {'h1_weights:0': tf.placeholder(dtype = tf.float32, shape = model_dict['New_h1_weights:0'].shape, name = 'h1_weights'),
                                        'h2_weights:0': tf.placeholder(dtype = tf.float32, shape =model_dict['New_h2_weights:0'].shape, name = 'h2_weights'),
                                        'h1_biases:0': tf.placeholder(dtype = tf.float32, shape =model_dict['New_h1_biases:0'].shape, name = 'h1_biases'),
                                        'h2_biases:0': tf.placeholder(dtype = tf.float32, shape =model_dict['New_h2_biases:0'].shape, name = 'h2_biases'),
                                        'out_weights:0': tf.placeholder(dtype = tf.float32, shape =model_dict['New_out_weights:0'].shape, name = 'out_weights'),
                                        'out_biases:0': tf.placeholder(dtype = tf.float32, shape =model_dict['New_out_biases:0'].shape, name = 'out_biases'),
                                        'list_of_C:0': tf.placeholder(dtype = tf.float32, shape =[6], name = 'list_of_C'),
                                        'm:0': tf.placeholder(dtype = tf.float32, shape =(), name = 'm')}
    return(dict_of_global_model_placeholders)

def New_global_model_placeholder(model_dict):
    #hidden1weights = tf.placeholder(model_dict['hidden1/weights:0'].shape)
    #hidden2weights = tf.placeholder(model_dict['hidden2/weights:0'].shape)
    #hidden1biases = tf.placeholder(model_dict['hidden1/biases:0'].shape)
    #hidden2biases = tf.placeholder(model_dict['hidden2/biases:0'].shape)
    #outweights = tf.placeholder(model_dict['out/weights:0'].shape)
    #outbiases = tf.placeholder(model_dict['out/biases:0'].shape)
    dict_of_global_model_placeholders= {'New_h1_weights:0': tf.placeholder(dtype = tf.float32, shape = model_dict['New_h1_weights:0'].shape, name = 'New_h1_weights'),
                                        'New_h2_weights:0': tf.placeholder(dtype = tf.float32, shape =model_dict['New_h2_weights:0'].shape, name = 'New_h2_weights'),
                                        'New_h1_biases:0': tf.placeholder(dtype = tf.float32, shape =model_dict['New_h1_biases:0'].shape, name = 'New_h1_biases'),
                                        'New_h2_biases:0': tf.placeholder(dtype = tf.float32, shape =model_dict['New_h2_biases:0'].shape, name = 'New_h2_biases'),
                                        'New_out_weights:0': tf.placeholder(dtype = tf.float32, shape =model_dict['New_out_weights:0'].shape, name = 'New_out_weights'),
                                        'New_out_biases:0': tf.placeholder(dtype = tf.float32, shape =model_dict['New_out_biases:0'].shape, name = 'New_out_biases')}
    return(dict_of_global_model_placeholders)

def assign_global_model(List_of_vars,dm):
    a = tf.assign(List_of_vars[0],dm['hidden1/weights:0'])
    b = tf.assign(List_of_vars[1],dm['hidden1/biases:0'])
    c = tf.assign(List_of_vars[2],dm['hidden2/weights:0'])
    d = tf.assign(List_of_vars[3],dm['hidden2/biases:0'])
    e = tf.assign(List_of_vars[4],dm['out/weights:0'])
    f = tf.assign(List_of_vars[5],dm['out/biases:0'])

