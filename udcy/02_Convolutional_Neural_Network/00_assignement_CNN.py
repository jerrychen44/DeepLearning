
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
import sys
import matplotlib.pyplot as plt

filepath=os.path.dirname(os.path.realpath(__file__))
source_path=filepath+'/source'


def read_back_data_from_pickle(pickle_file):
    #pickle_file = 'notMNIST.pickle'

    with open(pickle_file, 'rb') as f:
      save = pickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']
      del save  # hint to help gc free up memory
      print('Training set', train_dataset.shape, train_labels.shape)
      print('Validation set', valid_dataset.shape, valid_labels.shape)
      print('Test set', test_dataset.shape, test_labels.shape)
    '''
    Training set (200000, 28, 28) (200000,)
    Validation set (10000, 28, 28) (10000,)
    Test set (10000, 28, 28) (10000,)
    '''
    return train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels


#Reformat into a shape that's more adapted to the models we're going to train:
image_size = 28
num_labels = 10
num_channels = 1 # grayscale

def reformat(dataset, labels):
  dataset = dataset.reshape(
    (-1, image_size, image_size, num_channels)).astype(np.float32)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels



##########################
# for TensorFlow
##########################

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
#########################################
def small_cnn(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    batch_size = 16
    patch_size = 5
    # because cnn cost a lot, we limit the depth and number of neural first
    depth = 16
    num_hidden = 64

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
      tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      # Variables.
      layer1_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      layer3_weights = tf.Variable(tf.truncated_normal(
          [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_labels], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

      # Model.
      def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

      # Training computation.
      logits = model(tf_train_dataset)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
      test_prediction = tf.nn.softmax(model(tf_test_dataset))



    num_steps = 1001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

      '''
      Minibatch loss at step 0: 4.643481
      Minibatch accuracy: 6.2%
      Validation accuracy: 14.1%
      Minibatch loss at step 50: 2.200885
      Minibatch accuracy: 18.8%
      Validation accuracy: 28.2%
      Minibatch loss at step 100: 1.372677
      Minibatch accuracy: 50.0%
      Validation accuracy: 56.1%
      Minibatch loss at step 150: 0.597202
      Minibatch accuracy: 81.2%
      Validation accuracy: 72.4%
      Minibatch loss at step 200: 0.966382
      Minibatch accuracy: 75.0%
      Validation accuracy: 77.0%
      Minibatch loss at step 250: 1.372558
      Minibatch accuracy: 68.8%
      Validation accuracy: 76.9%
      Minibatch loss at step 300: 0.491400
      Minibatch accuracy: 87.5%
      Validation accuracy: 78.7%
      Minibatch loss at step 350: 0.511963
      Minibatch accuracy: 87.5%
      Validation accuracy: 77.8%
      Minibatch loss at step 400: 0.281703
      Minibatch accuracy: 93.8%
      Validation accuracy: 80.4%
      Minibatch loss at step 450: 1.137647
      Minibatch accuracy: 75.0%
      Validation accuracy: 79.3%
      Minibatch loss at step 500: 0.682878
      Minibatch accuracy: 87.5%
      Validation accuracy: 81.0%
      Minibatch loss at step 550: 0.933531
      Minibatch accuracy: 75.0%
      Validation accuracy: 80.9%
      Minibatch loss at step 600: 0.294786
      Minibatch accuracy: 100.0%
      Validation accuracy: 81.8%
      Minibatch loss at step 650: 0.865082
      Minibatch accuracy: 81.2%
      Validation accuracy: 81.4%
      Minibatch loss at step 700: 0.849730
      Minibatch accuracy: 75.0%
      Validation accuracy: 82.3%
      Minibatch loss at step 750: 0.043222
      Minibatch accuracy: 100.0%
      Validation accuracy: 82.7%
      Minibatch loss at step 800: 0.799056
      Minibatch accuracy: 81.2%
      Validation accuracy: 82.8%
      Minibatch loss at step 850: 0.968389
      Minibatch accuracy: 75.0%
      Validation accuracy: 82.4%
      Minibatch loss at step 900: 0.597779
      Minibatch accuracy: 87.5%
      Validation accuracy: 82.7%
      Minibatch loss at step 950: 0.445104
      Minibatch accuracy: 87.5%
      Validation accuracy: 83.3%
      Minibatch loss at step 1000: 0.468403
      Minibatch accuracy: 87.5%
      Validation accuracy: 82.8%
      Test accuracy: 89.5%
      '''


    return 0

######################################
# Problem 1
# The convolutional model above uses convolutions with stride 2 to reduce the dimensionality.
# Replace the strides by a max pooling operation (nn.max_pool()) of stride 2 and kernel size 2.
######################################

def cnn_strides_by_max_pooling(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    batch_size = 16
    patch_size = 5
    depth = 16
    num_hidden = 64

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
      tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      # Variables.
      layer1_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      layer3_weights = tf.Variable(tf.truncated_normal(
          [image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_labels], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

      # Model.
      def model(data):
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
        bias1 = tf.nn.relu(conv1 + layer1_biases)
        pool1 = tf.nn.max_pool(bias1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        conv2 = tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME')
        bias2 = tf.nn.relu(conv2 + layer2_biases)
        pool2 = tf.nn.max_pool(bias2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        shape = pool2.get_shape().as_list()
        reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

      # Training computation.
      logits = model(tf_train_dataset)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
      test_prediction = tf.nn.softmax(model(tf_test_dataset))

    num_steps = 1001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
      '''
      Minibatch loss at step 0: 2.478281
      Minibatch accuracy: 12.5%
      Validation accuracy: 10.0%
      Minibatch loss at step 50: 1.691796
      Minibatch accuracy: 37.5%
      Validation accuracy: 46.6%
      Minibatch loss at step 100: 1.199436
      Minibatch accuracy: 50.0%
      Validation accuracy: 60.6%
      Minibatch loss at step 150: 0.424575
      Minibatch accuracy: 87.5%
      Validation accuracy: 73.9%
      Minibatch loss at step 200: 0.852081
      Minibatch accuracy: 75.0%
      Validation accuracy: 77.2%
      Minibatch loss at step 250: 1.125189
      Minibatch accuracy: 68.8%
      Validation accuracy: 78.7%
      Minibatch loss at step 300: 0.351179
      Minibatch accuracy: 87.5%
      Validation accuracy: 80.3%
      Minibatch loss at step 350: 0.596486
      Minibatch accuracy: 93.8%
      Validation accuracy: 77.8%
      Minibatch loss at step 400: 0.248147
      Minibatch accuracy: 100.0%
      Validation accuracy: 81.0%
      Minibatch loss at step 450: 0.782881
      Minibatch accuracy: 81.2%
      Validation accuracy: 79.0%
      Minibatch loss at step 500: 0.641994
      Minibatch accuracy: 87.5%
      Validation accuracy: 81.4%
      Minibatch loss at step 550: 0.541164
      Minibatch accuracy: 75.0%
      Validation accuracy: 81.7%
      Minibatch loss at step 600: 0.270211
      Minibatch accuracy: 93.8%
      Validation accuracy: 82.0%
      Minibatch loss at step 650: 0.945373
      Minibatch accuracy: 75.0%
      Validation accuracy: 82.4%
      Minibatch loss at step 700: 0.909826
      Minibatch accuracy: 68.8%
      Validation accuracy: 82.1%
      Minibatch loss at step 750: 0.087795
      Minibatch accuracy: 100.0%
      Validation accuracy: 82.7%
      Minibatch loss at step 800: 0.642030
      Minibatch accuracy: 75.0%
      Validation accuracy: 83.2%
      Minibatch loss at step 850: 0.852509
      Minibatch accuracy: 75.0%
      Validation accuracy: 83.7%
      Minibatch loss at step 900: 0.562631
      Minibatch accuracy: 81.2%
      Validation accuracy: 83.9%
      Minibatch loss at step 950: 0.497981
      Minibatch accuracy: 87.5%
      Validation accuracy: 83.1%
      Minibatch loss at step 1000: 0.311734
      Minibatch accuracy: 81.2%
      Validation accuracy: 83.2%
      Test accuracy: 90.0%
      [Finished in 220.3s]
      '''
    return 0

######################################
# Problem 2
######################################
def cnn_leNet5(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    #leNet-5 ref: http://yann.lecun.com/exdb/lenet/
    batch_size = 16
    patch_size = 5
    depth = 16
    num_hidden = 64

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
      tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      # Variables.
      layer1_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      size3 = ((image_size - patch_size + 1) // 2 - patch_size + 1) // 2
      layer3_weights = tf.Variable(tf.truncated_normal(
          [size3 * size3 * depth, num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_labels], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

      # Model.
      def model(data):
        # C1 input 28 x 28
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID')
        bias1 = tf.nn.relu(conv1 + layer1_biases)
        # S2 input 24 x 24
        pool2 = tf.nn.avg_pool(bias1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        # C3 input 12 x 12
        conv3 = tf.nn.conv2d(pool2, layer2_weights, [1, 1, 1, 1], padding='VALID')
        bias3 = tf.nn.relu(conv3 + layer2_biases)
        # S4 input 8 x 8
        pool4 = tf.nn.avg_pool(bias3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        # F6 input 4 x 4
        shape = pool4.get_shape().as_list()
        reshape = tf.reshape(pool4, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

      # Training computation.
      logits = model(tf_train_dataset)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
      test_prediction = tf.nn.softmax(model(tf_test_dataset))


    num_steps = 20001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))


      '''
      Minibatch loss at step 0: 2.870062
      Minibatch accuracy: 12.5%
      Validation accuracy: 12.2%
      Minibatch loss at step 50: 1.629270
      Minibatch accuracy: 37.5%
      Validation accuracy: 53.4%
      Minibatch loss at step 100: 1.079443
      Minibatch accuracy: 62.5%
      Validation accuracy: 67.3%
      Minibatch loss at step 150: 0.793532
      Minibatch accuracy: 68.8%
      Validation accuracy: 70.2%
      Minibatch loss at step 200: 1.107719
      Minibatch accuracy: 75.0%
      Validation accuracy: 72.8%
      Minibatch loss at step 250: 1.190503
      Minibatch accuracy: 68.8%
      Validation accuracy: 74.4%
      Minibatch loss at step 300: 0.563957
      Minibatch accuracy: 87.5%
      Validation accuracy: 76.9%
      Minibatch loss at step 350: 0.787986
      Minibatch accuracy: 87.5%
      Validation accuracy: 73.5%
      Minibatch loss at step 400: 0.277647
      Minibatch accuracy: 100.0%
      Validation accuracy: 78.3%
      Minibatch loss at step 450: 1.007950
      Minibatch accuracy: 75.0%
      Validation accuracy: 77.4%
      Minibatch loss at step 500: 0.656317
      Minibatch accuracy: 87.5%
      Validation accuracy: 78.7%
      Minibatch loss at step 550: 0.785264
      Minibatch accuracy: 75.0%
      Validation accuracy: 78.5%
      Minibatch loss at step 600: 0.417053
      Minibatch accuracy: 87.5%
      Validation accuracy: 79.6%
      Minibatch loss at step 650: 0.872904
      Minibatch accuracy: 87.5%
      Validation accuracy: 79.6%
      Minibatch loss at step 700: 0.936830
      Minibatch accuracy: 68.8%
      Validation accuracy: 80.5%
      Minibatch loss at step 750: 0.056309
      Minibatch accuracy: 100.0%
      Validation accuracy: 80.6%
      Minibatch loss at step 800: 0.531151
      Minibatch accuracy: 75.0%
      Validation accuracy: 79.4%
      Minibatch loss at step 850: 0.964717
      Minibatch accuracy: 75.0%
      Validation accuracy: 80.1%
      Minibatch loss at step 900: 0.755102
      Minibatch accuracy: 75.0%
      Validation accuracy: 80.9%
      Minibatch loss at step 950: 0.537325
      Minibatch accuracy: 81.2%
      Validation accuracy: 81.3%
      Minibatch loss at step 1000: 0.431404
      Minibatch accuracy: 87.5%
      Validation accuracy: 81.3%
      Minibatch loss at step 1050: 0.580529
      Minibatch accuracy: 81.2%
      Validation accuracy: 80.3%
      Minibatch loss at step 1100: 0.705493
      Minibatch accuracy: 75.0%
      Validation accuracy: 82.0%
      Minibatch loss at step 1150: 0.354757
      Minibatch accuracy: 93.8%
      Validation accuracy: 80.8%
      Minibatch loss at step 1200: 0.828206
      Minibatch accuracy: 75.0%
      Validation accuracy: 82.3%
      Minibatch loss at step 1250: 0.552790
      Minibatch accuracy: 81.2%
      Validation accuracy: 82.3%
      Minibatch loss at step 1300: 0.332655
      Minibatch accuracy: 93.8%
      Validation accuracy: 81.9%
      Minibatch loss at step 1350: 1.096825
      Minibatch accuracy: 62.5%
      Validation accuracy: 81.5%
      Minibatch loss at step 1400: 0.339734
      Minibatch accuracy: 87.5%
      Validation accuracy: 82.6%
      Minibatch loss at step 1450: 0.361589
      Minibatch accuracy: 87.5%
      Validation accuracy: 82.7%
      Minibatch loss at step 1500: 0.751609
      Minibatch accuracy: 81.2%
      Validation accuracy: 82.3%
      Minibatch loss at step 1550: 0.646004
      Minibatch accuracy: 87.5%
      Validation accuracy: 83.0%
      Minibatch loss at step 1600: 1.067332
      Minibatch accuracy: 68.8%
      Validation accuracy: 82.1%
      Minibatch loss at step 1650: 0.794557
      Minibatch accuracy: 75.0%
      Validation accuracy: 82.4%
      Minibatch loss at step 1700: 0.584494
      Minibatch accuracy: 87.5%
      Validation accuracy: 82.8%
      Minibatch loss at step 1750: 0.397396
      Minibatch accuracy: 81.2%
      Validation accuracy: 83.1%
      Minibatch loss at step 1800: 0.549725
      Minibatch accuracy: 87.5%
      Validation accuracy: 83.7%
      Minibatch loss at step 1850: 0.874442
      Minibatch accuracy: 75.0%
      Validation accuracy: 84.1%
      Minibatch loss at step 1900: 0.280248
      Minibatch accuracy: 93.8%
      Validation accuracy: 83.2%
      Minibatch loss at step 1950: 0.490064
      Minibatch accuracy: 81.2%
      Validation accuracy: 83.7%
      Minibatch loss at step 2000: 0.170401
      Minibatch accuracy: 100.0%
      Validation accuracy: 83.5%
      Minibatch loss at step 2050: 0.712034
      Minibatch accuracy: 81.2%
      Validation accuracy: 83.6%
      Minibatch loss at step 2100: 0.319496
      Minibatch accuracy: 93.8%
      Validation accuracy: 84.0%
      Minibatch loss at step 2150: 0.541751
      Minibatch accuracy: 81.2%
      Validation accuracy: 84.2%
      Minibatch loss at step 2200: 0.384348
      Minibatch accuracy: 93.8%
      Validation accuracy: 84.1%
      Minibatch loss at step 2250: 0.552536
      Minibatch accuracy: 81.2%
      Validation accuracy: 84.3%
      Minibatch loss at step 2300: 0.680819
      Minibatch accuracy: 87.5%
      Validation accuracy: 84.4%
      Minibatch loss at step 2350: 0.411502
      Minibatch accuracy: 87.5%
      Validation accuracy: 84.7%
      Minibatch loss at step 2400: 0.585231
      Minibatch accuracy: 75.0%
      Validation accuracy: 84.4%
      Minibatch loss at step 2450: 0.518290
      Minibatch accuracy: 87.5%
      Validation accuracy: 84.3%
      Minibatch loss at step 2500: 0.814259
      Minibatch accuracy: 75.0%
      Validation accuracy: 84.1%
      Minibatch loss at step 2550: 0.643373
      Minibatch accuracy: 75.0%
      Validation accuracy: 84.3%
      Minibatch loss at step 2600: 0.087456
      Minibatch accuracy: 100.0%
      Validation accuracy: 84.8%
      Minibatch loss at step 2650: 0.380763
      Minibatch accuracy: 87.5%
      Validation accuracy: 84.8%
      Minibatch loss at step 2700: 0.384033
      Minibatch accuracy: 93.8%
      Validation accuracy: 85.1%
      Minibatch loss at step 2750: 1.366494
      Minibatch accuracy: 75.0%
      Validation accuracy: 85.0%
      Minibatch loss at step 2800: 0.558005
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.2%
      Minibatch loss at step 2850: 0.108581
      Minibatch accuracy: 100.0%
      Validation accuracy: 84.7%
      Minibatch loss at step 2900: 0.418707
      Minibatch accuracy: 81.2%
      Validation accuracy: 84.6%
      Minibatch loss at step 2950: 0.442213
      Minibatch accuracy: 87.5%
      Validation accuracy: 85.4%
      Minibatch loss at step 3000: 0.569932
      Minibatch accuracy: 87.5%
      Validation accuracy: 85.4%
      Minibatch loss at step 3050: 0.406377
      Minibatch accuracy: 93.8%
      Validation accuracy: 85.5%
      Minibatch loss at step 3100: 0.462364
      Minibatch accuracy: 75.0%
      Validation accuracy: 84.8%
      Minibatch loss at step 3150: 0.629699
      Minibatch accuracy: 75.0%
      Validation accuracy: 85.5%
      Minibatch loss at step 3200: 0.586164
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.1%
      Minibatch loss at step 3250: 0.348440
      Minibatch accuracy: 87.5%
      Validation accuracy: 85.4%
      Minibatch loss at step 3300: 0.121840
      Minibatch accuracy: 93.8%
      Validation accuracy: 85.8%
      Minibatch loss at step 3350: 0.429916
      Minibatch accuracy: 87.5%
      Validation accuracy: 86.1%
      Minibatch loss at step 3400: 0.716739
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.7%
      Minibatch loss at step 3450: 0.534674
      Minibatch accuracy: 75.0%
      Validation accuracy: 85.9%
      Minibatch loss at step 3500: 0.322117
      Minibatch accuracy: 87.5%
      Validation accuracy: 85.2%
      Minibatch loss at step 3550: 0.228740
      Minibatch accuracy: 93.8%
      Validation accuracy: 85.4%
      Minibatch loss at step 3600: 0.165749
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.0%
      Minibatch loss at step 3650: 0.832267
      Minibatch accuracy: 75.0%
      Validation accuracy: 85.9%
      Minibatch loss at step 3700: 1.082864
      Minibatch accuracy: 62.5%
      Validation accuracy: 85.9%
      Minibatch loss at step 3750: 1.013698
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.7%
      Minibatch loss at step 3800: 0.006977
      Minibatch accuracy: 100.0%
      Validation accuracy: 85.7%
      Minibatch loss at step 3850: 0.691524
      Minibatch accuracy: 81.2%
      Validation accuracy: 86.1%
      Minibatch loss at step 3900: 0.550327
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.3%
      Minibatch loss at step 3950: 0.022974
      Minibatch accuracy: 100.0%
      Validation accuracy: 86.1%
      Minibatch loss at step 4000: 0.390679
      Minibatch accuracy: 81.2%
      Validation accuracy: 86.6%
      Minibatch loss at step 4050: 0.740671
      Minibatch accuracy: 75.0%
      Validation accuracy: 85.4%
      Minibatch loss at step 4100: 0.420999
      Minibatch accuracy: 87.5%
      Validation accuracy: 86.4%
      Minibatch loss at step 4150: 1.161541
      Minibatch accuracy: 75.0%
      Validation accuracy: 85.3%
      Minibatch loss at step 4200: 0.314565
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.5%
      Minibatch loss at step 4250: 0.510629
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.8%
      Minibatch loss at step 4300: 0.815216
      Minibatch accuracy: 68.8%
      Validation accuracy: 86.4%
      Minibatch loss at step 4350: 0.283892
      Minibatch accuracy: 87.5%
      Validation accuracy: 86.2%
      Minibatch loss at step 4400: 1.245488
      Minibatch accuracy: 68.8%
      Validation accuracy: 86.4%
      Minibatch loss at step 4450: 0.381584
      Minibatch accuracy: 87.5%
      Validation accuracy: 86.3%
      Minibatch loss at step 4500: 0.727678
      Minibatch accuracy: 81.2%
      Validation accuracy: 86.3%
      Minibatch loss at step 4550: 0.272838
      Minibatch accuracy: 87.5%
      Validation accuracy: 86.6%
      Minibatch loss at step 4600: 0.536466
      Minibatch accuracy: 87.5%
      Validation accuracy: 86.5%
      Minibatch loss at step 4650: 0.958050
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.8%
      Minibatch loss at step 4700: 0.420489
      Minibatch accuracy: 81.2%
      Validation accuracy: 86.5%
      Minibatch loss at step 4750: 0.779282
      Minibatch accuracy: 68.8%
      Validation accuracy: 86.3%
      Minibatch loss at step 4800: 0.492831
      Minibatch accuracy: 81.2%
      Validation accuracy: 86.7%
      Minibatch loss at step 4850: 0.308533
      Minibatch accuracy: 87.5%
      Validation accuracy: 86.6%
      Minibatch loss at step 4900: 0.268581
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.3%
      Minibatch loss at step 4950: 0.145541
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.8%
      Minibatch loss at step 5000: 0.989714
      Minibatch accuracy: 75.0%
      Validation accuracy: 86.2%
      Minibatch loss at step 5050: 0.262627
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.6%
      Minibatch loss at step 5100: 0.347784
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.7%
      Minibatch loss at step 5150: 0.489095
      Minibatch accuracy: 81.2%
      Validation accuracy: 86.3%
      Minibatch loss at step 5200: 0.331811
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.3%
      Minibatch loss at step 5250: 0.172367
      Minibatch accuracy: 100.0%
      Validation accuracy: 86.9%
      Minibatch loss at step 5300: 0.259370
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.9%
      Minibatch loss at step 5350: 0.351539
      Minibatch accuracy: 87.5%
      Validation accuracy: 86.9%
      Minibatch loss at step 5400: 0.342748
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.0%
      Minibatch loss at step 5450: 0.379988
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.5%
      Minibatch loss at step 5500: 0.515578
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.6%
      Minibatch loss at step 5550: 0.289676
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.7%
      Minibatch loss at step 5600: 0.284513
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.2%
      Minibatch loss at step 5650: 0.292489
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.1%
      Minibatch loss at step 5700: 0.466809
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.4%
      Minibatch loss at step 5750: 0.709535
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.2%
      Minibatch loss at step 5800: 0.211420
      Minibatch accuracy: 87.5%
      Validation accuracy: 86.7%
      Minibatch loss at step 5850: 0.838020
      Minibatch accuracy: 81.2%
      Validation accuracy: 86.8%
      Minibatch loss at step 5900: 0.713878
      Minibatch accuracy: 75.0%
      Validation accuracy: 86.7%
      Minibatch loss at step 5950: 0.276559
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.0%
      Minibatch loss at step 6000: 0.354586
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.0%
      Minibatch loss at step 6050: 0.476539
      Minibatch accuracy: 81.2%
      Validation accuracy: 86.9%
      Minibatch loss at step 6100: 0.867112
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.2%
      Minibatch loss at step 6150: 0.185316
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.8%
      Minibatch loss at step 6200: 1.026330
      Minibatch accuracy: 75.0%
      Validation accuracy: 86.9%
      Minibatch loss at step 6250: 0.822595
      Minibatch accuracy: 81.2%
      Validation accuracy: 86.8%
      Minibatch loss at step 6300: 0.771389
      Minibatch accuracy: 75.0%
      Validation accuracy: 87.3%
      Minibatch loss at step 6350: 0.130531
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.1%
      Minibatch loss at step 6400: 0.113789
      Minibatch accuracy: 100.0%
      Validation accuracy: 87.3%
      Minibatch loss at step 6450: 0.304199
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.2%
      Minibatch loss at step 6500: 0.955884
      Minibatch accuracy: 68.8%
      Validation accuracy: 86.7%
      Minibatch loss at step 6550: 0.111710
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.2%
      Minibatch loss at step 6600: 0.314487
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.3%
      Minibatch loss at step 6650: 1.527828
      Minibatch accuracy: 56.2%
      Validation accuracy: 87.2%
      Minibatch loss at step 6700: 0.220938
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.6%
      Minibatch loss at step 6750: 0.437046
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.6%
      Minibatch loss at step 6800: 0.606991
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.6%
      Minibatch loss at step 6850: 0.563817
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.2%
      Minibatch loss at step 6900: 0.357773
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.5%
      Minibatch loss at step 6950: 0.231656
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.3%
      Minibatch loss at step 7000: 0.753997
      Minibatch accuracy: 68.8%
      Validation accuracy: 87.0%
      Minibatch loss at step 7050: 0.696569
      Minibatch accuracy: 75.0%
      Validation accuracy: 86.9%
      Minibatch loss at step 7100: 0.548011
      Minibatch accuracy: 75.0%
      Validation accuracy: 86.9%
      Minibatch loss at step 7150: 0.175463
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.2%
      Minibatch loss at step 7200: 0.491758
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.2%
      Minibatch loss at step 7250: 0.282207
      Minibatch accuracy: 100.0%
      Validation accuracy: 87.7%
      Minibatch loss at step 7300: 0.607831
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.8%
      Minibatch loss at step 7350: 0.191917
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.5%
      Minibatch loss at step 7400: 0.019526
      Minibatch accuracy: 100.0%
      Validation accuracy: 87.4%
      Minibatch loss at step 7450: 0.277224
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.4%
      Minibatch loss at step 7500: 0.220668
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.3%
      Minibatch loss at step 7550: 0.390239
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.2%
      Minibatch loss at step 7600: 0.564710
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.5%
      Minibatch loss at step 7650: 0.221202
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.7%
      Minibatch loss at step 7700: 0.100536
      Minibatch accuracy: 100.0%
      Validation accuracy: 87.5%
      Minibatch loss at step 7750: 0.455106
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.7%
      Minibatch loss at step 7800: 0.293344
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.6%
      Minibatch loss at step 7850: 0.428975
      Minibatch accuracy: 75.0%
      Validation accuracy: 87.5%
      Minibatch loss at step 7900: 0.109832
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.3%
      Minibatch loss at step 7950: 0.840029
      Minibatch accuracy: 75.0%
      Validation accuracy: 87.5%
      Minibatch loss at step 8000: 0.567269
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.4%
      Minibatch loss at step 8050: 0.331815
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.7%
      Minibatch loss at step 8100: 0.379003
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.8%
      Minibatch loss at step 8150: 0.660101
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.7%
      Minibatch loss at step 8200: 0.236766
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.6%
      Minibatch loss at step 8250: 0.214636
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.7%
      Minibatch loss at step 8300: 0.362346
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.9%
      Minibatch loss at step 8350: 0.503499
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.7%
      Minibatch loss at step 8400: 0.446708
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.8%
      Minibatch loss at step 8450: 0.212770
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.7%
      Minibatch loss at step 8500: 0.336051
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.7%
      Minibatch loss at step 8550: 0.330173
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.1%
      Minibatch loss at step 8600: 0.284037
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.0%
      Minibatch loss at step 8650: 0.436026
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.9%
      Minibatch loss at step 8700: 0.248536
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.8%
      Minibatch loss at step 8750: 0.186659
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.5%
      Minibatch loss at step 8800: 0.258620
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.7%
      Minibatch loss at step 8850: 0.015588
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.0%
      Minibatch loss at step 8900: 0.556142
      Minibatch accuracy: 75.0%
      Validation accuracy: 87.7%
      Minibatch loss at step 8950: 0.257096
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.1%
      Minibatch loss at step 9000: 0.384967
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.2%
      Minibatch loss at step 9050: 0.338924
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.7%
      Minibatch loss at step 9100: 0.357222
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.2%
      Minibatch loss at step 9150: 0.651663
      Minibatch accuracy: 68.8%
      Validation accuracy: 87.7%
      Minibatch loss at step 9200: 0.208259
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.0%
      Minibatch loss at step 9250: 0.918810
      Minibatch accuracy: 75.0%
      Validation accuracy: 87.8%
      Minibatch loss at step 9300: 0.948293
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.4%
      Minibatch loss at step 9350: 0.268328
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.0%
      Minibatch loss at step 9400: 0.316168
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.3%
      Minibatch loss at step 9450: 0.341255
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.2%
      Minibatch loss at step 9500: 0.141618
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.4%
      Minibatch loss at step 9550: 0.396787
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.3%
      Minibatch loss at step 9600: 0.310750
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.0%
      Minibatch loss at step 9650: 0.423291
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.9%
      Minibatch loss at step 9700: 0.191331
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.0%
      Minibatch loss at step 9750: 0.159564
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.8%
      Minibatch loss at step 9800: 0.349715
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.1%
      Minibatch loss at step 9850: 0.294723
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.4%
      Minibatch loss at step 9900: 0.500054
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.7%
      Minibatch loss at step 9950: 0.214842
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.2%
      Minibatch loss at step 10000: 0.121775
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.3%
      Minibatch loss at step 10050: 0.026717
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.0%
      Minibatch loss at step 10100: 0.252712
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.3%
      Minibatch loss at step 10150: 0.642805
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.0%
      Minibatch loss at step 10200: 0.181835
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.9%
      Minibatch loss at step 10250: 0.388735
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.5%
      Minibatch loss at step 10300: 0.077623
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.1%
      Minibatch loss at step 10350: 0.601386
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.0%
      Minibatch loss at step 10400: 0.382907
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.8%
      Minibatch loss at step 10450: 0.169217
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.1%
      Minibatch loss at step 10500: 0.391000
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.7%
      Minibatch loss at step 10550: 0.766562
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.1%
      Minibatch loss at step 10600: 0.645613
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.3%
      Minibatch loss at step 10650: 0.434296
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.9%
      Minibatch loss at step 10700: 0.026177
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.1%
      Minibatch loss at step 10750: 0.244893
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.0%
      Minibatch loss at step 10800: 0.366869
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.0%
      Minibatch loss at step 10850: 0.849037
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.3%
      Minibatch loss at step 10900: 0.273755
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.1%
      Minibatch loss at step 10950: 0.522450
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.3%
      Minibatch loss at step 11000: 0.081096
      Minibatch accuracy: 100.0%
      Validation accuracy: 87.7%
      Minibatch loss at step 11050: 0.431353
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.8%
      Minibatch loss at step 11100: 0.143641
      Minibatch accuracy: 93.8%
      Validation accuracy: 87.7%
      Minibatch loss at step 11150: 0.407932
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.1%
      Minibatch loss at step 11200: 0.171836
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.0%
      Minibatch loss at step 11250: 1.152301
      Minibatch accuracy: 62.5%
      Validation accuracy: 88.1%
      Minibatch loss at step 11300: 0.427727
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.4%
      Minibatch loss at step 11350: 0.390672
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.4%
      Minibatch loss at step 11400: 0.192475
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.1%
      Minibatch loss at step 11450: 0.358708
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.8%
      Minibatch loss at step 11500: 0.389354
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.5%
      Minibatch loss at step 11550: 0.418593
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.0%
      Minibatch loss at step 11600: 0.359118
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.2%
      Minibatch loss at step 11650: 0.399747
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.1%
      Minibatch loss at step 11700: 1.167827
      Minibatch accuracy: 62.5%
      Validation accuracy: 88.3%
      Minibatch loss at step 11750: 0.498036
      Minibatch accuracy: 75.0%
      Validation accuracy: 88.3%
      Minibatch loss at step 11800: 0.019233
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.1%
      Minibatch loss at step 11850: 0.688001
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.2%
      Minibatch loss at step 11900: 0.416383
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.6%
      Minibatch loss at step 11950: 0.752429
      Minibatch accuracy: 62.5%
      Validation accuracy: 88.6%
      Minibatch loss at step 12000: 0.495362
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.4%
      Minibatch loss at step 12050: 0.040831
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.6%
      Minibatch loss at step 12100: 0.402020
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.5%
      Minibatch loss at step 12150: 0.155297
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.5%
      Minibatch loss at step 12200: 0.368565
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.5%
      Minibatch loss at step 12250: 0.486398
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.2%
      Minibatch loss at step 12300: 0.203636
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.3%
      Minibatch loss at step 12350: 0.706000
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.4%
      Minibatch loss at step 12400: 0.023242
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.5%
      Minibatch loss at step 12450: 1.003123
      Minibatch accuracy: 68.8%
      Validation accuracy: 88.2%
      Minibatch loss at step 12500: 0.704347
      Minibatch accuracy: 75.0%
      Validation accuracy: 88.1%
      Minibatch loss at step 12550: 0.551673
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.8%
      Minibatch loss at step 12600: 0.586613
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.4%
      Minibatch loss at step 12650: 0.577095
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.8%
      Minibatch loss at step 12700: 0.396605
      Minibatch accuracy: 87.5%
      Validation accuracy: 87.8%
      Minibatch loss at step 12750: 0.187599
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.6%
      Minibatch loss at step 12800: 0.149415
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.6%
      Minibatch loss at step 12850: 0.265386
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.2%
      Minibatch loss at step 12900: 0.229571
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.4%
      Minibatch loss at step 12950: 0.034825
      Minibatch accuracy: 100.0%
      Validation accuracy: 87.8%
      Minibatch loss at step 13000: 0.318177
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.3%
      Minibatch loss at step 13050: 0.154413
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.6%
      Minibatch loss at step 13100: 0.218871
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.7%
      Minibatch loss at step 13150: 0.337681
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.7%
      Minibatch loss at step 13200: 0.277733
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.4%
      Minibatch loss at step 13250: 0.803046
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.4%
      Minibatch loss at step 13300: 0.220119
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.6%
      Minibatch loss at step 13350: 0.159747
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.8%
      Minibatch loss at step 13400: 0.562131
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.6%
      Minibatch loss at step 13450: 0.426708
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.8%
      Minibatch loss at step 13500: 0.367928
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.6%
      Minibatch loss at step 13550: 0.467180
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.6%
      Minibatch loss at step 13600: 0.494898
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.8%
      Minibatch loss at step 13650: 0.415511
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.0%
      Minibatch loss at step 13700: 0.224326
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.7%
      Minibatch loss at step 13750: 0.674556
      Minibatch accuracy: 75.0%
      Validation accuracy: 88.5%
      Minibatch loss at step 13800: 0.051250
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.6%
      Minibatch loss at step 13850: 0.242197
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.4%
      Minibatch loss at step 13900: 0.122873
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 13950: 0.455793
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.8%
      Minibatch loss at step 14000: 0.064856
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.6%
      Minibatch loss at step 14050: 0.432282
      Minibatch accuracy: 81.2%
      Validation accuracy: 87.4%
      Minibatch loss at step 14100: 0.581990
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.8%
      Minibatch loss at step 14150: 0.219237
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.2%
      Minibatch loss at step 14200: 0.333395
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.2%
      Minibatch loss at step 14250: 0.182019
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.4%
      Minibatch loss at step 14300: 0.396793
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.7%
      Minibatch loss at step 14350: 0.109232
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.7%
      Minibatch loss at step 14400: 0.451877
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.4%
      Minibatch loss at step 14450: 0.161937
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.6%
      Minibatch loss at step 14500: 0.344822
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.4%
      Minibatch loss at step 14550: 0.347062
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.8%
      Minibatch loss at step 14600: 0.047981
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.6%
      Minibatch loss at step 14650: 0.629277
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.1%
      Minibatch loss at step 14700: 0.216662
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.8%
      Minibatch loss at step 14750: 0.358276
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.2%
      Minibatch loss at step 14800: 0.288019
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.4%
      Minibatch loss at step 14850: 0.307907
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.9%
      Minibatch loss at step 14900: 0.269814
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.8%
      Minibatch loss at step 14950: 0.429148
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.7%
      Minibatch loss at step 15000: 0.550161
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.4%
      Minibatch loss at step 15050: 0.546355
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.7%
      Minibatch loss at step 15100: 0.201915
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.8%
      Minibatch loss at step 15150: 0.399242
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 15200: 0.101832
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 15250: 0.577896
      Minibatch accuracy: 75.0%
      Validation accuracy: 89.3%
      Minibatch loss at step 15300: 0.182270
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.7%
      Minibatch loss at step 15350: 0.051720
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.6%
      Minibatch loss at step 15400: 0.833568
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.7%
      Minibatch loss at step 15450: 0.676841
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.9%
      Minibatch loss at step 15500: 0.118859
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 15550: 0.301015
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.8%
      Minibatch loss at step 15600: 0.405029
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.9%
      Minibatch loss at step 15650: 0.287849
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 15700: 0.302547
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.2%
      Minibatch loss at step 15750: 0.273551
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 15800: 0.076085
      Minibatch accuracy: 100.0%
      Validation accuracy: 89.2%
      Minibatch loss at step 15850: 0.284023
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.0%
      Minibatch loss at step 15900: 0.555701
      Minibatch accuracy: 81.2%
      Validation accuracy: 89.1%
      Minibatch loss at step 15950: 0.093841
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.0%
      Minibatch loss at step 16000: 0.991769
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.0%
      Minibatch loss at step 16050: 0.067552
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.7%
      Minibatch loss at step 16100: 0.261024
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 16150: 0.076430
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.7%
      Minibatch loss at step 16200: 0.511813
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.9%
      Minibatch loss at step 16250: 0.445848
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.9%
      Minibatch loss at step 16300: 0.505356
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.8%
      Minibatch loss at step 16350: 0.626444
      Minibatch accuracy: 81.2%
      Validation accuracy: 89.0%
      Minibatch loss at step 16400: 0.513508
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.9%
      Minibatch loss at step 16450: 0.802805
      Minibatch accuracy: 81.2%
      Validation accuracy: 89.2%
      Minibatch loss at step 16500: 0.006483
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.9%
      Minibatch loss at step 16550: 0.605279
      Minibatch accuracy: 75.0%
      Validation accuracy: 88.8%
      Minibatch loss at step 16600: 0.360721
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.9%
      Minibatch loss at step 16650: 0.746629
      Minibatch accuracy: 75.0%
      Validation accuracy: 88.3%
      Minibatch loss at step 16700: 0.362529
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.8%
      Minibatch loss at step 16750: 0.423712
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.8%
      Minibatch loss at step 16800: 0.803158
      Minibatch accuracy: 81.2%
      Validation accuracy: 89.0%
      Minibatch loss at step 16850: 0.661140
      Minibatch accuracy: 75.0%
      Validation accuracy: 89.1%
      Minibatch loss at step 16900: 0.716471
      Minibatch accuracy: 75.0%
      Validation accuracy: 88.8%
      Minibatch loss at step 16950: 0.217443
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.7%
      Minibatch loss at step 17000: 0.138363
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 17050: 0.331970
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.0%
      Minibatch loss at step 17100: 0.332017
      Minibatch accuracy: 81.2%
      Validation accuracy: 89.0%
      Minibatch loss at step 17150: 0.203168
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 17200: 0.240471
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.1%
      Minibatch loss at step 17250: 0.376469
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.0%
      Minibatch loss at step 17300: 0.361970
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.8%
      Minibatch loss at step 17350: 0.204993
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.2%
      Minibatch loss at step 17400: 0.619249
      Minibatch accuracy: 81.2%
      Validation accuracy: 89.0%
      Minibatch loss at step 17450: 0.279778
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.1%
      Minibatch loss at step 17500: 0.450683
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.3%
      Minibatch loss at step 17550: 0.664984
      Minibatch accuracy: 75.0%
      Validation accuracy: 88.9%
      Minibatch loss at step 17600: 0.497312
      Minibatch accuracy: 75.0%
      Validation accuracy: 89.0%
      Minibatch loss at step 17650: 0.135097
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.8%
      Minibatch loss at step 17700: 0.432293
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.0%
      Minibatch loss at step 17750: 0.222769
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.2%
      Minibatch loss at step 17800: 0.524280
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.0%
      Minibatch loss at step 17850: 0.413990
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.7%
      Minibatch loss at step 17900: 0.320701
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.0%
      Minibatch loss at step 17950: 0.591085
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.7%
      Minibatch loss at step 18000: 0.316564
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.8%
      Minibatch loss at step 18050: 0.272372
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.8%
      Minibatch loss at step 18100: 0.482869
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.2%
      Minibatch loss at step 18150: 0.181488
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.2%
      Minibatch loss at step 18200: 0.541979
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.2%
      Minibatch loss at step 18250: 0.578280
      Minibatch accuracy: 81.2%
      Validation accuracy: 89.1%
      Minibatch loss at step 18300: 0.099428
      Minibatch accuracy: 100.0%
      Validation accuracy: 88.6%
      Minibatch loss at step 18350: 0.291325
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.0%
      Minibatch loss at step 18400: 0.693692
      Minibatch accuracy: 75.0%
      Validation accuracy: 88.9%
      Minibatch loss at step 18450: 0.349930
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.8%
      Minibatch loss at step 18500: 0.403418
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.2%
      Minibatch loss at step 18550: 0.506755
      Minibatch accuracy: 75.0%
      Validation accuracy: 89.0%
      Minibatch loss at step 18600: 0.097101
      Minibatch accuracy: 100.0%
      Validation accuracy: 89.1%
      Minibatch loss at step 18650: 0.283767
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.0%
      Minibatch loss at step 18700: 0.234686
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.2%
      Minibatch loss at step 18750: 0.527680
      Minibatch accuracy: 75.0%
      Validation accuracy: 88.7%
      Minibatch loss at step 18800: 0.371905
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.9%
      Minibatch loss at step 18850: 0.566446
      Minibatch accuracy: 81.2%
      Validation accuracy: 89.1%
      Minibatch loss at step 18900: 0.024043
      Minibatch accuracy: 100.0%
      Validation accuracy: 89.4%
      Minibatch loss at step 18950: 0.135783
      Minibatch accuracy: 100.0%
      Validation accuracy: 89.3%
      Minibatch loss at step 19000: 0.409340
      Minibatch accuracy: 81.2%
      Validation accuracy: 88.5%
      Minibatch loss at step 19050: 0.105973
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.0%
      Minibatch loss at step 19100: 0.027633
      Minibatch accuracy: 100.0%
      Validation accuracy: 89.1%
      Minibatch loss at step 19150: 0.763197
      Minibatch accuracy: 68.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 19200: 0.373128
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 19250: 0.793284
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.3%
      Minibatch loss at step 19300: 0.175672
      Minibatch accuracy: 100.0%
      Validation accuracy: 89.1%
      Minibatch loss at step 19350: 0.159947
      Minibatch accuracy: 100.0%
      Validation accuracy: 89.2%
      Minibatch loss at step 19400: 0.494974
      Minibatch accuracy: 81.2%
      Validation accuracy: 89.3%
      Minibatch loss at step 19450: 0.277787
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.2%
      Minibatch loss at step 19500: 0.263611
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.0%
      Minibatch loss at step 19550: 0.267723
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.9%
      Minibatch loss at step 19600: 0.316754
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.0%
      Minibatch loss at step 19650: 0.398657
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.9%
      Minibatch loss at step 19700: 0.281918
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.2%
      Minibatch loss at step 19750: 0.652804
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.2%
      Minibatch loss at step 19800: 0.156039
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.6%
      Minibatch loss at step 19850: 0.114610
      Minibatch accuracy: 100.0%
      Validation accuracy: 89.3%
      Minibatch loss at step 19900: 0.244625
      Minibatch accuracy: 93.8%
      Validation accuracy: 89.4%
      Minibatch loss at step 19950: 0.674150
      Minibatch accuracy: 81.2%
      Validation accuracy: 89.4%
      Minibatch loss at step 20000: 0.259394
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.2%
      Test accuracy: 95.0%
      [Finished in 1463.9s]
      '''
    return 0


def cnn_leNet5_dropout_learning_rate_decay(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    batch_size = 16
    patch_size = 5
    depth = 16
    num_hidden = 64
    beta_regul = 1e-3
    drop_out = 0.5

    graph = tf.Graph()

    with graph.as_default():

      # Input data.
      tf_train_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      global_step = tf.Variable(0)

      # Variables.
      layer1_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, num_channels, depth], stddev=0.1))
      layer1_biases = tf.Variable(tf.zeros([depth]))
      layer2_weights = tf.Variable(tf.truncated_normal(
          [patch_size, patch_size, depth, depth], stddev=0.1))
      layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
      size3 = ((image_size - patch_size + 1) // 2 - patch_size + 1) // 2
      layer3_weights = tf.Variable(tf.truncated_normal(
          [size3 * size3 * depth, num_hidden], stddev=0.1))
      layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer4_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_hidden], stddev=0.1))
      layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
      layer5_weights = tf.Variable(tf.truncated_normal(
          [num_hidden, num_labels], stddev=0.1))
      layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

      # Model.
      def model(data, keep_prob):
        # C1 input 28 x 28
        conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='VALID')
        bias1 = tf.nn.relu(conv1 + layer1_biases)
        # S2 input 24 x 24
        pool2 = tf.nn.avg_pool(bias1, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        # C3 input 12 x 12
        conv3 = tf.nn.conv2d(pool2, layer2_weights, [1, 1, 1, 1], padding='VALID')
        bias3 = tf.nn.relu(conv3 + layer2_biases)
        # S4 input 8 x 8
        pool4 = tf.nn.avg_pool(bias3, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')
        # F5 input 4 x 4
        shape = pool4.get_shape().as_list()
        reshape = tf.reshape(pool4, [shape[0], shape[1] * shape[2] * shape[3]])
        hidden5 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        # F6
        drop5 = tf.nn.dropout(hidden5, keep_prob)
        hidden6 = tf.nn.relu(tf.matmul(hidden5, layer4_weights) + layer4_biases)
        drop6 = tf.nn.dropout(hidden6, keep_prob)
        return tf.matmul(drop6, layer5_weights) + layer5_biases

      # Training computation.
      logits = model(tf_train_dataset, drop_out)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      learning_rate = tf.train.exponential_decay(0.05, global_step, 1000, 0.85, staircase=True)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(model(tf_valid_dataset, 1.0))
      test_prediction = tf.nn.softmax(model(tf_test_dataset, 1.0))

    num_steps = 5001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
          print('Minibatch loss at step %d: %f' % (step, l))
          print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))

      '''
      Minibatch loss at step 0: 2.957129
      Minibatch accuracy: 6.2%
      Validation accuracy: 11.7%
      Minibatch loss at step 50: 2.343740
      Minibatch accuracy: 6.2%
      Validation accuracy: 24.3%
      Minibatch loss at step 100: 2.011855
      Minibatch accuracy: 18.8%
      Validation accuracy: 41.2%
      Minibatch loss at step 150: 1.247103
      Minibatch accuracy: 75.0%
      Validation accuracy: 49.3%
      Minibatch loss at step 200: 1.268764
      Minibatch accuracy: 68.8%
      Validation accuracy: 61.3%
      Minibatch loss at step 250: 1.919025
      Minibatch accuracy: 50.0%
      Validation accuracy: 62.9%
      Minibatch loss at step 300: 1.010079
      Minibatch accuracy: 68.8%
      Validation accuracy: 68.8%
      Minibatch loss at step 350: 0.926720
      Minibatch accuracy: 81.2%
      Validation accuracy: 69.9%
      Minibatch loss at step 400: 0.464861
      Minibatch accuracy: 87.5%
      Validation accuracy: 76.4%
      Minibatch loss at step 450: 0.675417
      Minibatch accuracy: 68.8%
      Validation accuracy: 75.4%
      Minibatch loss at step 500: 1.289088
      Minibatch accuracy: 75.0%
      Validation accuracy: 72.2%
      Minibatch loss at step 550: 0.940578
      Minibatch accuracy: 68.8%
      Validation accuracy: 76.1%
      Minibatch loss at step 600: 0.765200
      Minibatch accuracy: 81.2%
      Validation accuracy: 78.0%
      Minibatch loss at step 650: 1.049418
      Minibatch accuracy: 75.0%
      Validation accuracy: 77.4%
      Minibatch loss at step 700: 1.034066
      Minibatch accuracy: 75.0%
      Validation accuracy: 78.6%
      Minibatch loss at step 750: 0.227730
      Minibatch accuracy: 100.0%
      Validation accuracy: 78.3%
      Minibatch loss at step 800: 0.791292
      Minibatch accuracy: 75.0%
      Validation accuracy: 79.6%
      Minibatch loss at step 850: 0.966739
      Minibatch accuracy: 68.8%
      Validation accuracy: 77.8%
      Minibatch loss at step 900: 1.242967
      Minibatch accuracy: 50.0%
      Validation accuracy: 79.6%
      Minibatch loss at step 950: 0.661044
      Minibatch accuracy: 75.0%
      Validation accuracy: 80.0%
      Minibatch loss at step 1000: 0.345786
      Minibatch accuracy: 81.2%
      Validation accuracy: 80.3%
      Minibatch loss at step 1050: 0.589580
      Minibatch accuracy: 81.2%
      Validation accuracy: 79.9%
      Minibatch loss at step 1100: 0.807183
      Minibatch accuracy: 68.8%
      Validation accuracy: 79.8%
      Minibatch loss at step 1150: 0.443540
      Minibatch accuracy: 87.5%
      Validation accuracy: 79.1%
      Minibatch loss at step 1200: 1.262839
      Minibatch accuracy: 62.5%
      Validation accuracy: 80.8%
      Minibatch loss at step 1250: 0.731933
      Minibatch accuracy: 75.0%
      Validation accuracy: 80.7%
      Minibatch loss at step 1300: 0.509299
      Minibatch accuracy: 87.5%
      Validation accuracy: 81.0%
      Minibatch loss at step 1350: 1.028303
      Minibatch accuracy: 68.8%
      Validation accuracy: 80.9%
      Minibatch loss at step 1400: 0.419461
      Minibatch accuracy: 87.5%
      Validation accuracy: 81.3%
      Minibatch loss at step 1450: 0.541711
      Minibatch accuracy: 87.5%
      Validation accuracy: 81.5%
      Minibatch loss at step 1500: 0.623286
      Minibatch accuracy: 75.0%
      Validation accuracy: 81.1%
      Minibatch loss at step 1550: 0.566707
      Minibatch accuracy: 87.5%
      Validation accuracy: 81.3%
      Minibatch loss at step 1600: 1.050236
      Minibatch accuracy: 75.0%
      Validation accuracy: 81.6%
      Minibatch loss at step 1650: 1.080111
      Minibatch accuracy: 81.2%
      Validation accuracy: 81.7%
      Minibatch loss at step 1700: 0.809432
      Minibatch accuracy: 81.2%
      Validation accuracy: 81.5%
      Minibatch loss at step 1750: 0.618317
      Minibatch accuracy: 87.5%
      Validation accuracy: 82.1%
      Minibatch loss at step 1800: 0.961930
      Minibatch accuracy: 75.0%
      Validation accuracy: 81.5%
      Minibatch loss at step 1850: 1.083147
      Minibatch accuracy: 68.8%
      Validation accuracy: 81.4%
      Minibatch loss at step 1900: 0.291585
      Minibatch accuracy: 93.8%
      Validation accuracy: 81.4%
      Minibatch loss at step 1950: 0.753314
      Minibatch accuracy: 75.0%
      Validation accuracy: 82.9%
      Minibatch loss at step 2000: 0.103869
      Minibatch accuracy: 100.0%
      Validation accuracy: 82.5%
      Minibatch loss at step 2050: 0.881842
      Minibatch accuracy: 75.0%
      Validation accuracy: 82.2%
      Minibatch loss at step 2100: 0.483215
      Minibatch accuracy: 87.5%
      Validation accuracy: 83.1%
      Minibatch loss at step 2150: 0.645532
      Minibatch accuracy: 87.5%
      Validation accuracy: 83.0%
      Minibatch loss at step 2200: 0.219373
      Minibatch accuracy: 93.8%
      Validation accuracy: 82.0%
      Minibatch loss at step 2250: 0.573196
      Minibatch accuracy: 87.5%
      Validation accuracy: 82.7%
      Minibatch loss at step 2300: 0.767929
      Minibatch accuracy: 81.2%
      Validation accuracy: 82.7%
      Minibatch loss at step 2350: 0.674051
      Minibatch accuracy: 75.0%
      Validation accuracy: 83.0%
      Minibatch loss at step 2400: 0.800697
      Minibatch accuracy: 68.8%
      Validation accuracy: 83.1%
      Minibatch loss at step 2450: 0.775737
      Minibatch accuracy: 62.5%
      Validation accuracy: 83.4%
      Minibatch loss at step 2500: 0.634740
      Minibatch accuracy: 75.0%
      Validation accuracy: 82.6%
      Minibatch loss at step 2550: 0.741151
      Minibatch accuracy: 75.0%
      Validation accuracy: 83.4%
      Minibatch loss at step 2600: 0.256723
      Minibatch accuracy: 100.0%
      Validation accuracy: 83.8%
      Minibatch loss at step 2650: 0.330140
      Minibatch accuracy: 87.5%
      Validation accuracy: 83.5%
      Minibatch loss at step 2700: 0.831728
      Minibatch accuracy: 81.2%
      Validation accuracy: 83.4%
      Minibatch loss at step 2750: 1.291580
      Minibatch accuracy: 81.2%
      Validation accuracy: 83.5%
      Minibatch loss at step 2800: 0.824082
      Minibatch accuracy: 75.0%
      Validation accuracy: 83.5%
      Minibatch loss at step 2850: 0.207222
      Minibatch accuracy: 93.8%
      Validation accuracy: 83.7%
      Minibatch loss at step 2900: 0.648840
      Minibatch accuracy: 81.2%
      Validation accuracy: 83.6%
      Minibatch loss at step 2950: 0.459844
      Minibatch accuracy: 93.8%
      Validation accuracy: 84.2%
      Minibatch loss at step 3000: 0.855114
      Minibatch accuracy: 87.5%
      Validation accuracy: 84.0%
      Minibatch loss at step 3050: 0.475137
      Minibatch accuracy: 87.5%
      Validation accuracy: 83.7%
      Minibatch loss at step 3100: 0.764644
      Minibatch accuracy: 75.0%
      Validation accuracy: 84.0%
      Minibatch loss at step 3150: 0.876048
      Minibatch accuracy: 68.8%
      Validation accuracy: 83.6%
      Minibatch loss at step 3200: 0.508167
      Minibatch accuracy: 87.5%
      Validation accuracy: 83.8%
      Minibatch loss at step 3250: 0.422167
      Minibatch accuracy: 81.2%
      Validation accuracy: 83.6%
      Minibatch loss at step 3300: 0.288931
      Minibatch accuracy: 93.8%
      Validation accuracy: 84.8%
      Minibatch loss at step 3350: 0.665370
      Minibatch accuracy: 87.5%
      Validation accuracy: 84.7%
      Minibatch loss at step 3400: 0.946044
      Minibatch accuracy: 75.0%
      Validation accuracy: 84.7%
      Minibatch loss at step 3450: 0.751853
      Minibatch accuracy: 75.0%
      Validation accuracy: 84.5%
      Minibatch loss at step 3500: 0.534220
      Minibatch accuracy: 75.0%
      Validation accuracy: 85.0%
      Minibatch loss at step 3550: 0.228345
      Minibatch accuracy: 93.8%
      Validation accuracy: 85.1%
      Minibatch loss at step 3600: 0.510243
      Minibatch accuracy: 87.5%
      Validation accuracy: 84.8%
      Minibatch loss at step 3650: 0.691536
      Minibatch accuracy: 75.0%
      Validation accuracy: 84.5%
      Minibatch loss at step 3700: 0.903999
      Minibatch accuracy: 62.5%
      Validation accuracy: 84.8%
      Minibatch loss at step 3750: 0.613438
      Minibatch accuracy: 81.2%
      Validation accuracy: 82.4%
      Minibatch loss at step 3800: 0.037662
      Minibatch accuracy: 100.0%
      Validation accuracy: 84.7%
      Minibatch loss at step 3850: 0.629156
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.1%
      Minibatch loss at step 3900: 0.720159
      Minibatch accuracy: 81.2%
      Validation accuracy: 84.7%
      Minibatch loss at step 3950: 0.028420
      Minibatch accuracy: 100.0%
      Validation accuracy: 84.7%
      Minibatch loss at step 4000: 0.461064
      Minibatch accuracy: 81.2%
      Validation accuracy: 84.9%
      Minibatch loss at step 4050: 1.104770
      Minibatch accuracy: 68.8%
      Validation accuracy: 83.3%
      Minibatch loss at step 4100: 0.527437
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.0%
      Minibatch loss at step 4150: 1.157431
      Minibatch accuracy: 56.2%
      Validation accuracy: 84.8%
      Minibatch loss at step 4200: 0.370505
      Minibatch accuracy: 93.8%
      Validation accuracy: 85.1%
      Minibatch loss at step 4250: 0.543384
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.3%
      Minibatch loss at step 4300: 0.969415
      Minibatch accuracy: 62.5%
      Validation accuracy: 85.0%
      Minibatch loss at step 4350: 0.374243
      Minibatch accuracy: 87.5%
      Validation accuracy: 85.4%
      Minibatch loss at step 4400: 1.458998
      Minibatch accuracy: 62.5%
      Validation accuracy: 85.0%
      Minibatch loss at step 4450: 0.536867
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.3%
      Minibatch loss at step 4500: 0.739927
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.5%
      Minibatch loss at step 4550: 0.405698
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.8%
      Minibatch loss at step 4600: 0.330702
      Minibatch accuracy: 87.5%
      Validation accuracy: 85.6%
      Minibatch loss at step 4650: 0.813988
      Minibatch accuracy: 87.5%
      Validation accuracy: 84.6%
      Minibatch loss at step 4700: 0.554671
      Minibatch accuracy: 81.2%
      Validation accuracy: 85.5%
      Minibatch loss at step 4750: 0.921649
      Minibatch accuracy: 62.5%
      Validation accuracy: 85.6%
      Minibatch loss at step 4800: 0.901837
      Minibatch accuracy: 75.0%
      Validation accuracy: 85.4%
      Minibatch loss at step 4850: 0.431805
      Minibatch accuracy: 87.5%
      Validation accuracy: 85.5%
      Minibatch loss at step 4900: 0.306797
      Minibatch accuracy: 93.8%
      Validation accuracy: 85.4%
      Minibatch loss at step 4950: 0.362362
      Minibatch accuracy: 87.5%
      Validation accuracy: 85.4%
      Minibatch loss at step 5000: 1.033314
      Minibatch accuracy: 75.0%
      Validation accuracy: 85.4%
      Test accuracy: 91.9%
      [Finished in 461.4s]
      '''
    return 0
#########################################
def main():
    #The goal of this code is make the neural network convolutional to redo the previous character classify.
    #####################
    # Get data section
    ######################
    # get data back from pickle
    pickle_file=source_path+'/notMNIST.pickle'
    train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels=read_back_data_from_pickle(pickle_file)

    #Reformat into a shape that's more adapted to the models we're going to train:
    # data as a flat matrix,
    # labels as float 1-hot encodings.
    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    '''
    Training set (200000, 784) (200000, 10)
    Validation set (10000, 784) (10000, 10)
    Test set (10000, 784) (10000, 10)
    '''
    print("Data ready")



    #####################################
    # Step1, build small CNN to take a look.
    #
    # Let's build a small network with two convolutional layers,
    # followed by one fully connected layer.
    # Convolutional networks are more expensive computationally,
    # so we'll limit its depth and number of fully connected nodes.
    ########################################
    #small_cnn(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)


    ######################################
    # Problem 1
    # The convolutional model above uses convolutions with stride 2 to reduce the dimensionality.
    # Replace the strides by a max pooling operation (nn.max_pool()) of stride 2 and kernel size 2.
    ######################################
    #cnn_strides_by_max_pooling(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)

    ######################################
    # Problem 2
    # Try to get the best performance you can using a convolutional net.
    # Look for example at the classic LeNet5 architecture, adding Dropout,
    # and/or adding learning rate decay.
    ####################################
    #The CNN below is loosely inspired by the LeNet5 architecture.
    #leNet-5 ref: http://yann.lecun.com/exdb/lenet/
    #cnn_leNet5(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)
    #The accuracy is good 95.0% , but not as good as the 3-layer network from the previous assignment 96.3%.

    #one more try
    cnn_leNet5_dropout_learning_rate_decay(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)
    # Test accuracy: 91.9% , not that good, might need tuning more parameter later. So, we stop here.
    return 0

main()
