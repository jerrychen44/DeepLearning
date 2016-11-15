
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

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels



##########################
# TensorFlow : normal_gradient_descent_tensorflow
##########################

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])


##########################
# Problem1 section: The right amount of regularization should improve your validation / test accuracy.
########################


def logistic_L2_regularzation(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):

    batch_size = 128

    graph = tf.Graph()
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      beta_regul = tf.placeholder(tf.float32)

      # Variables.
      weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
      biases = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      logits = tf.matmul(tf_train_dataset, weights) + biases
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + beta_regul * tf.nn.l2_loss(weights)

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
      test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)





    num_steps = 3001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : 1e-3}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
          '''
          Minibatch loss at step 0: 18.734179
          Minibatch accuracy: 9.4%
          Validation accuracy: 12.8%
          Minibatch loss at step 500: 2.576000
          Minibatch accuracy: 78.9%
          Validation accuracy: 76.0%
          Minibatch loss at step 1000: 1.835047
          Minibatch accuracy: 75.8%
          Validation accuracy: 79.1%
          Minibatch loss at step 1500: 0.970275
          Minibatch accuracy: 82.0%
          Validation accuracy: 79.7%
          Minibatch loss at step 2000: 0.788001
          Minibatch accuracy: 88.3%
          Validation accuracy: 80.8%
          Minibatch loss at step 2500: 0.846615
          Minibatch accuracy: 78.9%
          Validation accuracy: 81.3%
          Minibatch loss at step 3000: 0.764198
          Minibatch accuracy: 82.8%
          Validation accuracy: 81.9%
          '''
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
      '''Test accuracy: 89.1%'''
      # it better then original SGD Test accuracy: 85.9% '''



    # We can plot out the accuracy vs regularization to chose parameter.
    '''
    The L2 regularization introduces a new meta parameter that should be tuned.
    Since I do not have any idea of what should be the right value for this meta parameter,
    I will plot the accuracy by the meta parameter value (in a logarithmic scale).
    '''

    num_steps = 3001
    regul_val = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
    accuracy_val = []

    for regul in regul_val:
      with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
          offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
          batch_data = train_dataset[offset:(offset + batch_size), :]
          batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
          feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : regul}
          _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        accuracy_val.append(accuracy(test_prediction.eval(), test_labels))


    plt.semilogx(regul_val, accuracy_val)
    plt.grid(True)
    plt.title('Test accuracy by regularization (logistic)')
    plt.show()


    return 0


def logistic_regression_SGD_L2_regularzation_1_hidden_layer(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):

    batch_size = 128
    num_hidden_nodes = 1024

    graph = tf.Graph()
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      beta_regul = tf.placeholder(tf.float32)

      # Variables.
      weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
      biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
      weights2 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes, num_labels]))
      biases2 = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
      logits = tf.matmul(lay1_train, weights2) + biases2
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
          beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2))

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
      valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
      lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
      test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)


    num_steps = 3001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : 1e-3}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
          '''
          Minibatch loss at step 0: 575.464600
          Minibatch accuracy: 14.1%
          Validation accuracy: 27.7%
          Minibatch loss at step 500: 200.098938
          Minibatch accuracy: 82.0%
          Validation accuracy: 78.8%
          Minibatch loss at step 1000: 116.055962
          Minibatch accuracy: 77.3%
          Validation accuracy: 81.3%
          Minibatch loss at step 1500: 68.584267
          Minibatch accuracy: 90.6%
          Validation accuracy: 83.2%
          Minibatch loss at step 2000: 41.303318
          Minibatch accuracy: 89.1%
          Validation accuracy: 84.7%
          Minibatch loss at step 2500: 25.208529
          Minibatch accuracy: 88.3%
          Validation accuracy: 85.9%
          Minibatch loss at step 3000: 15.498355
          Minibatch accuracy: 88.3%
          Validation accuracy: 86.2%

          '''
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
      '''Test accuracy: 93.0%'''
      '''the original 1-hidden layer Test accuracy: 89.4%'''




    #I will also plot the final accuracy by the L2 parameter to find the best value.
    # Get python crash when try to ploting. might need change other computer.
    '''
    num_steps = 3001
    regul_val = [pow(10, i) for i in np.arange(-4, -2, 0.1)]
    accuracy_val = []

    for regul in regul_val:
      with tf.Session(graph=graph) as session:
        tf.initialize_all_variables().run()
        for step in range(num_steps):
          # Pick an offset within the training data, which has been randomized.
          # Note: we could use better randomization across epochs.
          offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
          # Generate a minibatch.
          batch_data = train_dataset[offset:(offset + batch_size), :]
          batch_labels = train_labels[offset:(offset + batch_size), :]
          # Prepare a dictionary telling the session where to feed the minibatch.
          # The key of the dictionary is the placeholder node of the graph to be fed,
          # and the value is the numpy array to feed to it.
          feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : regul}
          _, l, predictions = session.run(
            [optimizer, loss, train_prediction], feed_dict=feed_dict)
        accuracy_val.append(accuracy(test_prediction.eval(), test_labels))

    plt.semilogx(regul_val, accuracy_val)
    plt.grid(True)
    plt.title('Test accuracy by regularization (1-layer net)')
    plt.show()
    '''

    return 0


##########################
# Problem 2 section: Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?
########################
def force_to_create_overfit_with_few_batches(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    batch_size = 128
    num_hidden_nodes = 1024

    graph = tf.Graph()
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      beta_regul = tf.placeholder(tf.float32)

      # Variables.
      weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
      biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
      weights2 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes, num_labels]))
      biases2 = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
      logits = tf.matmul(lay1_train, weights2) + biases2
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
      valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
      lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
      test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)

    num_steps = 101
    num_bacthes = 3

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        #offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        offset = step % num_bacthes
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : 1e-3}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 2 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
      '''
      Minibatch loss at step 0: 304.482117
      Minibatch accuracy: 7.0%
      Validation accuracy: 29.3%
      Minibatch loss at step 2: 1182.052734
      Minibatch accuracy: 43.0%
      Validation accuracy: 26.9%
      Minibatch loss at step 4: 451.587006
      Minibatch accuracy: 63.3%
      Validation accuracy: 51.0%
      Minibatch loss at step 6: 30.931835
      Minibatch accuracy: 94.5%
      Validation accuracy: 66.3%
      Minibatch loss at step 8: 0.269836
      Minibatch accuracy: 99.2%
      Validation accuracy: 66.6%
      Minibatch loss at step 10: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 12: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 14: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 16: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 18: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 20: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 22: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 24: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 26: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 28: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 30: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 32: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 34: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 36: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 38: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 40: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 42: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 44: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 46: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 48: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 50: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 52: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 54: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 56: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 58: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 60: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 62: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 64: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 66: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 68: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 70: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 72: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 74: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 76: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 78: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 80: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 82: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 84: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 86: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 88: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 90: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 92: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 94: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 96: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 98: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Minibatch loss at step 100: 0.000000
      Minibatch accuracy: 100.0%
      Validation accuracy: 66.6%
      Test accuracy: 73.4%
      '''
    # Since there are far too much parameters and no regularization, the accuracy of the batches is 100%.
    # The generalization capability is poor, as shown in the validation and test accuracy.
    return 0

###########################
# Problem 3
############################

def dropout_hidden_layer_neural_network(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    # We keep using the overfitting case which with less num_steps to add the droput method to see the result.
    batch_size = 128
    num_hidden_nodes = 1024

    graph = tf.Graph()
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      # Variables.
      weights1 = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_hidden_nodes]))
      biases1 = tf.Variable(tf.zeros([num_hidden_nodes]))
      weights2 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes, num_labels]))
      biases2 = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
      drop1 = tf.nn.dropout(lay1_train, 0.5)
      logits = tf.matmul(drop1, weights2) + biases2
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
      valid_prediction = tf.nn.softmax(tf.matmul(lay1_valid, weights2) + biases2)
      lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
      test_prediction = tf.nn.softmax(tf.matmul(lay1_test, weights2) + biases2)




    num_steps = 101
    num_batches = 3

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        #offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        offset = step % num_batches
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 2 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
      '''
        Minibatch loss at step 0: 451.535767
        Minibatch accuracy: 8.6%
        Validation accuracy: 37.5%
        Minibatch loss at step 2: 642.615112
        Minibatch accuracy: 43.8%
        Validation accuracy: 41.1%
        Minibatch loss at step 4: 187.512817
        Minibatch accuracy: 71.1%
        Validation accuracy: 59.2%
        Minibatch loss at step 6: 75.605904
        Minibatch accuracy: 85.2%
        Validation accuracy: 62.3%
        Minibatch loss at step 8: 58.276970
        Minibatch accuracy: 91.4%
        Validation accuracy: 63.9%
        Minibatch loss at step 10: 41.092480
        Minibatch accuracy: 92.2%
        Validation accuracy: 66.6%
        Minibatch loss at step 12: 7.425128
        Minibatch accuracy: 95.3%
        Validation accuracy: 68.9%
        Minibatch loss at step 14: 3.641557
        Minibatch accuracy: 96.1%
        Validation accuracy: 67.2%
        Minibatch loss at step 16: 1.233654
        Minibatch accuracy: 98.4%
        Validation accuracy: 67.5%
        Minibatch loss at step 18: 3.921563
        Minibatch accuracy: 99.2%
        Validation accuracy: 68.3%
        Minibatch loss at step 20: 10.910660
        Minibatch accuracy: 98.4%
        Validation accuracy: 67.5%
        Minibatch loss at step 22: 2.211551
        Minibatch accuracy: 98.4%
        Validation accuracy: 68.1%
        Minibatch loss at step 24: 4.588318
        Minibatch accuracy: 99.2%
        Validation accuracy: 67.3%
        Minibatch loss at step 26: 8.813985
        Minibatch accuracy: 96.9%
        Validation accuracy: 67.3%
        Minibatch loss at step 28: 0.814095
        Minibatch accuracy: 98.4%
        Validation accuracy: 66.0%
        Minibatch loss at step 30: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 67.3%
        Minibatch loss at step 32: 0.994688
        Minibatch accuracy: 98.4%
        Validation accuracy: 67.4%
        Minibatch loss at step 34: 1.955612
        Minibatch accuracy: 98.4%
        Validation accuracy: 65.0%
        Minibatch loss at step 36: 12.826794
        Minibatch accuracy: 97.7%
        Validation accuracy: 66.9%
        Minibatch loss at step 38: 3.664794
        Minibatch accuracy: 99.2%
        Validation accuracy: 66.5%
        Minibatch loss at step 40: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 67.2%
        Minibatch loss at step 42: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 66.8%
        Minibatch loss at step 44: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 66.4%
        Minibatch loss at step 46: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 66.5%
        Minibatch loss at step 48: 4.636361
        Minibatch accuracy: 97.7%
        Validation accuracy: 68.2%
        Minibatch loss at step 50: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.3%
        Minibatch loss at step 52: 5.165306
        Minibatch accuracy: 97.7%
        Validation accuracy: 67.9%
        Minibatch loss at step 54: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.3%
        Minibatch loss at step 56: 3.579936
        Minibatch accuracy: 99.2%
        Validation accuracy: 68.4%
        Minibatch loss at step 58: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 69.0%
        Minibatch loss at step 60: 0.000103
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.9%
        Minibatch loss at step 62: 0.179486
        Minibatch accuracy: 99.2%
        Validation accuracy: 66.9%
        Minibatch loss at step 64: 1.426496
        Minibatch accuracy: 98.4%
        Validation accuracy: 67.6%
        Minibatch loss at step 66: 0.000005
        Minibatch accuracy: 100.0%
        Validation accuracy: 67.6%
        Minibatch loss at step 68: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.7%
        Minibatch loss at step 70: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 69.2%
        Minibatch loss at step 72: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.7%
        Minibatch loss at step 74: 2.141777
        Minibatch accuracy: 98.4%
        Validation accuracy: 68.0%
        Minibatch loss at step 76: 1.755879
        Minibatch accuracy: 99.2%
        Validation accuracy: 68.3%
        Minibatch loss at step 78: 0.525315
        Minibatch accuracy: 99.2%
        Validation accuracy: 67.1%
        Minibatch loss at step 80: 0.342058
        Minibatch accuracy: 98.4%
        Validation accuracy: 67.5%
        Minibatch loss at step 82: 0.691027
        Minibatch accuracy: 99.2%
        Validation accuracy: 68.8%
        Minibatch loss at step 84: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.8%
        Minibatch loss at step 86: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.8%
        Minibatch loss at step 88: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.8%
        Minibatch loss at step 90: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.8%
        Minibatch loss at step 92: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.8%
        Minibatch loss at step 94: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.8%
        Minibatch loss at step 96: 0.010065
        Minibatch accuracy: 99.2%
        Validation accuracy: 68.8%
        Minibatch loss at step 98: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.8%
        Minibatch loss at step 100: 0.000000
        Minibatch accuracy: 100.0%
        Validation accuracy: 68.8%
        Test accuracy: 75.5%
        '''
        # Overfit still very obviously. Not gets huge improvment.

    return 0


#######################
# problem 4
#######################

def multi_layer_dropout_nn_2_hiddenlayer(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    #Let's do a first try with 2 layers.
    #Note how the parameters are initialized, compared to the previous cases.

    batch_size = 128
    num_hidden_nodes1 = 1024
    num_hidden_nodes2 = 100
    beta_regul = 1e-3

    graph = tf.Graph()
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      global_step = tf.Variable(0)

      # Variables.
      weights1 = tf.Variable(
        tf.truncated_normal(
            [image_size * image_size, num_hidden_nodes1],
            stddev=np.sqrt(2.0 / (image_size * image_size)))
        )
      biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]))
      weights2 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)))
      biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
      weights3 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes2, num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes2)))
      biases3 = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
      lay2_train = tf.nn.relu(tf.matmul(lay1_train, weights2) + biases2)
      logits = tf.matmul(lay2_train, weights3) + biases3
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels)) + \
          beta_regul * (tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3))

      # Optimizer.
      learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.65, staircase=True)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
      lay2_valid = tf.nn.relu(tf.matmul(lay1_valid, weights2) + biases2)
      valid_prediction = tf.nn.softmax(tf.matmul(lay2_valid, weights3) + biases3)
      lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
      lay2_test = tf.nn.relu(tf.matmul(lay1_test, weights2) + biases2)
      test_prediction = tf.nn.softmax(tf.matmul(lay2_test, weights3) + biases3)


    num_steps = 9001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
      '''
      Minibatch loss at step 0: 3.362878
      Minibatch accuracy: 10.9%
      Validation accuracy: 31.1%
      Minibatch loss at step 500: 0.937883
      Minibatch accuracy: 90.6%
      Validation accuracy: 85.7%
      Minibatch loss at step 1000: 0.867683
      Minibatch accuracy: 86.7%
      Validation accuracy: 86.7%
      Minibatch loss at step 1500: 0.555411
      Minibatch accuracy: 93.8%
      Validation accuracy: 88.1%
      Minibatch loss at step 2000: 0.512782
      Minibatch accuracy: 94.5%
      Validation accuracy: 88.2%
      Minibatch loss at step 2500: 0.524298
      Minibatch accuracy: 89.8%
      Validation accuracy: 88.5%
      Minibatch loss at step 3000: 0.563105
      Minibatch accuracy: 89.1%
      Validation accuracy: 88.9%
      Minibatch loss at step 3500: 0.571085
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.2%
      Minibatch loss at step 4000: 0.449387
      Minibatch accuracy: 91.4%
      Validation accuracy: 89.2%
      Minibatch loss at step 4500: 0.430461
      Minibatch accuracy: 92.2%
      Validation accuracy: 89.4%
      Minibatch loss at step 5000: 0.496759
      Minibatch accuracy: 90.6%
      Validation accuracy: 89.7%
      Minibatch loss at step 5500: 0.492404
      Minibatch accuracy: 89.1%
      Validation accuracy: 89.7%
      Minibatch loss at step 6000: 0.554603
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.9%
      Minibatch loss at step 6500: 0.398648
      Minibatch accuracy: 93.0%
      Validation accuracy: 90.0%
      Minibatch loss at step 7000: 0.509912
      Minibatch accuracy: 89.1%
      Validation accuracy: 90.0%
      Minibatch loss at step 7500: 0.485764
      Minibatch accuracy: 90.6%
      Validation accuracy: 90.0%
      Minibatch loss at step 8000: 0.573553
      Minibatch accuracy: 86.7%
      Validation accuracy: 90.3%
      Minibatch loss at step 8500: 0.418236
      Minibatch accuracy: 92.2%
      Validation accuracy: 90.2%
      Minibatch loss at step 9000: 0.457600
      Minibatch accuracy: 90.6%
      Validation accuracy: 90.2%
      Test accuracy: 95.8%
      '''
      #Wow result 95.8 is really good, and the overfit has gone!!

    return 0

def multi_layer_dropout_nn_deeper_3_hiddenlayer(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    batch_size = 128
    num_hidden_nodes1 = 1024
    num_hidden_nodes2 = 256
    num_hidden_nodes3 = 128
    keep_prob = 0.5

    graph = tf.Graph()
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      global_step = tf.Variable(0)

      # Variables.
      weights1 = tf.Variable(
        tf.truncated_normal(
            [image_size * image_size, num_hidden_nodes1],
            stddev=np.sqrt(2.0 / (image_size * image_size)))
        )
      biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]))
      weights2 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)))
      biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
      weights3 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes3], stddev=np.sqrt(2.0 / num_hidden_nodes2)))
      biases3 = tf.Variable(tf.zeros([num_hidden_nodes3]))
      weights4 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes3, num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes3)))
      biases4 = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
      lay2_train = tf.nn.relu(tf.matmul(lay1_train, weights2) + biases2)
      lay3_train = tf.nn.relu(tf.matmul(lay2_train, weights3) + biases3)
      logits = tf.matmul(lay3_train, weights4) + biases4
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      learning_rate = tf.train.exponential_decay(0.5, global_step, 4000, 0.65, staircase=True)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
      lay2_valid = tf.nn.relu(tf.matmul(lay1_valid, weights2) + biases2)
      lay3_valid = tf.nn.relu(tf.matmul(lay2_valid, weights3) + biases3)
      valid_prediction = tf.nn.softmax(tf.matmul(lay3_valid, weights4) + biases4)
      lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
      lay2_test = tf.nn.relu(tf.matmul(lay1_test, weights2) + biases2)
      lay3_test = tf.nn.relu(tf.matmul(lay2_test, weights3) + biases3)
      test_prediction = tf.nn.softmax(tf.matmul(lay3_test, weights4) + biases4)


    num_steps = 18001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))

      '''
      Minibatch loss at step 0: 2.276848
      Minibatch accuracy: 14.8%
      Validation accuracy: 36.1%
      Minibatch loss at step 500: 0.368922
      Minibatch accuracy: 89.1%
      Validation accuracy: 85.7%
      Minibatch loss at step 1000: 0.483899
      Minibatch accuracy: 84.4%
      Validation accuracy: 86.2%
      Minibatch loss at step 1500: 0.258817
      Minibatch accuracy: 91.4%
      Validation accuracy: 88.0%
      Minibatch loss at step 2000: 0.249952
      Minibatch accuracy: 95.3%
      Validation accuracy: 88.3%
      Minibatch loss at step 2500: 0.280068
      Minibatch accuracy: 90.6%
      Validation accuracy: 88.6%
      Minibatch loss at step 3000: 0.336090
      Minibatch accuracy: 89.8%
      Validation accuracy: 88.5%
      Minibatch loss at step 3500: 0.342047
      Minibatch accuracy: 88.3%
      Validation accuracy: 89.1%
      Minibatch loss at step 4000: 0.230973
      Minibatch accuracy: 93.0%
      Validation accuracy: 89.3%
      Minibatch loss at step 4500: 0.273061
      Minibatch accuracy: 92.2%
      Validation accuracy: 89.4%
      Minibatch loss at step 5000: 0.254953
      Minibatch accuracy: 94.5%
      Validation accuracy: 90.1%
      Minibatch loss at step 5500: 0.225644
      Minibatch accuracy: 92.2%
      Validation accuracy: 89.9%
      Minibatch loss at step 6000: 0.346753
      Minibatch accuracy: 90.6%
      Validation accuracy: 89.7%
      Minibatch loss at step 6500: 0.207713
      Minibatch accuracy: 96.1%
      Validation accuracy: 90.2%
      Minibatch loss at step 7000: 0.314960
      Minibatch accuracy: 88.3%
      Validation accuracy: 90.2%
      Minibatch loss at step 7500: 0.224976
      Minibatch accuracy: 92.2%
      Validation accuracy: 90.0%
      Minibatch loss at step 8000: 0.329163
      Minibatch accuracy: 89.8%
      Validation accuracy: 90.2%
      Minibatch loss at step 8500: 0.152266
      Minibatch accuracy: 96.9%
      Validation accuracy: 90.0%
      Minibatch loss at step 9000: 0.190430
      Minibatch accuracy: 96.1%
      Validation accuracy: 90.2%
      Minibatch loss at step 9500: 0.204998
      Minibatch accuracy: 95.3%
      Validation accuracy: 90.5%
      Minibatch loss at step 10000: 0.113122
      Minibatch accuracy: 96.1%
      Validation accuracy: 90.3%
      Minibatch loss at step 10500: 0.152741
      Minibatch accuracy: 93.8%
      Validation accuracy: 90.6%
      Minibatch loss at step 11000: 0.059341
      Minibatch accuracy: 97.7%
      Validation accuracy: 90.5%
      Minibatch loss at step 11500: 0.110632
      Minibatch accuracy: 97.7%
      Validation accuracy: 90.4%
      Minibatch loss at step 12000: 0.178130
      Minibatch accuracy: 94.5%
      Validation accuracy: 90.5%
      Minibatch loss at step 12500: 0.075581
      Minibatch accuracy: 96.9%
      Validation accuracy: 90.7%
      Minibatch loss at step 13000: 0.135429
      Minibatch accuracy: 95.3%
      Validation accuracy: 90.9%
      Minibatch loss at step 13500: 0.068047
      Minibatch accuracy: 98.4%
      Validation accuracy: 90.8%
      Minibatch loss at step 14000: 0.094824
      Minibatch accuracy: 98.4%
      Validation accuracy: 90.8%
      Minibatch loss at step 14500: 0.108811
      Minibatch accuracy: 98.4%
      Validation accuracy: 90.8%
      Minibatch loss at step 15000: 0.066789
      Minibatch accuracy: 98.4%
      Validation accuracy: 90.8%
      Minibatch loss at step 15500: 0.069336
      Minibatch accuracy: 97.7%
      Validation accuracy: 90.9%
      Minibatch loss at step 16000: 0.039861
      Minibatch accuracy: 99.2%
      Validation accuracy: 90.7%
      Minibatch loss at step 16500: 0.032218
      Minibatch accuracy: 98.4%
      Validation accuracy: 90.9%
      Minibatch loss at step 17000: 0.023720
      Minibatch accuracy: 100.0%
      Validation accuracy: 91.1%
      Minibatch loss at step 17500: 0.031370
      Minibatch accuracy: 99.2%
      Validation accuracy: 91.1%
      Minibatch loss at step 18000: 0.046984
      Minibatch accuracy: 98.4%
      Validation accuracy: 91.0%
      Test accuracy: 96.3%
      [Finished in 620.1s]
      '''
      # wow, 96.3% much better , but start getting little overfit
    return 0

def multi_layer_dropout_nn_deeper_3_hiddenlayer_more_neural(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    batch_size = 128
    num_hidden_nodes1 = 1024
    num_hidden_nodes2 = 512 # original is 256
    num_hidden_nodes3 = 256 # original is 128
    keep_prob = 0.5

    graph = tf.Graph()
    with graph.as_default():

      # Input data. For the training data, we use a placeholder that will be fed
      # at run time with a training minibatch.
      tf_train_dataset = tf.placeholder(tf.float32,
                                        shape=(batch_size, image_size * image_size))
      tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)
      global_step = tf.Variable(0)

      # Variables.
      weights1 = tf.Variable(
        tf.truncated_normal(
            [image_size * image_size, num_hidden_nodes1],
            stddev=np.sqrt(2.0 / (image_size * image_size)))
        )
      biases1 = tf.Variable(tf.zeros([num_hidden_nodes1]))
      weights2 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes1, num_hidden_nodes2], stddev=np.sqrt(2.0 / num_hidden_nodes1)))
      biases2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
      weights3 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes3], stddev=np.sqrt(2.0 / num_hidden_nodes2)))
      biases3 = tf.Variable(tf.zeros([num_hidden_nodes3]))
      weights4 = tf.Variable(
        tf.truncated_normal([num_hidden_nodes3, num_labels], stddev=np.sqrt(2.0 / num_hidden_nodes3)))
      biases4 = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      lay1_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights1) + biases1)
      drop1 = tf.nn.dropout(lay1_train, 0.5)
      lay2_train = tf.nn.relu(tf.matmul(drop1, weights2) + biases2)
      drop2 = tf.nn.dropout(lay2_train, 0.5)
      lay3_train = tf.nn.relu(tf.matmul(drop2, weights3) + biases3)
      drop3 = tf.nn.dropout(lay3_train, 0.5)
      logits = tf.matmul(drop3, weights4) + biases4
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      learning_rate = tf.train.exponential_decay(0.5, global_step, 5000, 0.80, staircase=True)
      optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

      # Predictions for the training, validation, and test data.
      train_prediction = tf.nn.softmax(logits)
      lay1_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights1) + biases1)
      lay2_valid = tf.nn.relu(tf.matmul(lay1_valid, weights2) + biases2)
      lay3_valid = tf.nn.relu(tf.matmul(lay2_valid, weights3) + biases3)
      valid_prediction = tf.nn.softmax(tf.matmul(lay3_valid, weights4) + biases4)
      lay1_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights1) + biases1)
      lay2_test = tf.nn.relu(tf.matmul(lay1_test, weights2) + biases2)
      lay3_test = tf.nn.relu(tf.matmul(lay2_test, weights3) + biases3)
      test_prediction = tf.nn.softmax(tf.matmul(lay3_test, weights4) + biases4)

    num_steps = 20001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print("Initialized")
      for step in range(num_steps):
        # Pick an offset within the training data, which has been randomized.
        # Note: we could use better randomization across epochs.
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        # Prepare a dictionary telling the session where to feed the minibatch.
        # The key of the dictionary is the placeholder node of the graph to be fed,
        # and the value is the numpy array to feed to it.
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
      '''
      Minibatch loss at step 0: 2.828934
      Minibatch accuracy: 7.0%
      Validation accuracy: 19.6%
      Minibatch loss at step 500: 0.479646
      Minibatch accuracy: 84.4%
      Validation accuracy: 84.1%
      Minibatch loss at step 1000: 0.583013
      Minibatch accuracy: 83.6%
      Validation accuracy: 84.8%
      Minibatch loss at step 1500: 0.389850
      Minibatch accuracy: 86.7%
      Validation accuracy: 85.3%
      Minibatch loss at step 2000: 0.361162
      Minibatch accuracy: 93.8%
      Validation accuracy: 86.4%
      Minibatch loss at step 2500: 0.481559
      Minibatch accuracy: 83.6%
      Validation accuracy: 86.5%
      Minibatch loss at step 3000: 0.518757
      Minibatch accuracy: 82.0%
      Validation accuracy: 86.3%
      Minibatch loss at step 3500: 0.550707
      Minibatch accuracy: 84.4%
      Validation accuracy: 87.1%
      Minibatch loss at step 4000: 0.407942
      Minibatch accuracy: 89.8%
      Validation accuracy: 86.8%
      Minibatch loss at step 4500: 0.357927
      Minibatch accuracy: 85.2%
      Validation accuracy: 87.1%
      Minibatch loss at step 5000: 0.453237
      Minibatch accuracy: 85.9%
      Validation accuracy: 87.5%
      Minibatch loss at step 5500: 0.467508
      Minibatch accuracy: 83.6%
      Validation accuracy: 88.0%
      Minibatch loss at step 6000: 0.497700
      Minibatch accuracy: 84.4%
      Validation accuracy: 88.1%
      Minibatch loss at step 6500: 0.330698
      Minibatch accuracy: 89.1%
      Validation accuracy: 88.3%
      Minibatch loss at step 7000: 0.482862
      Minibatch accuracy: 84.4%
      Validation accuracy: 88.5%
      Minibatch loss at step 7500: 0.452115
      Minibatch accuracy: 85.2%
      Validation accuracy: 88.6%
      Minibatch loss at step 8000: 0.630604
      Minibatch accuracy: 78.9%
      Validation accuracy: 88.5%
      Minibatch loss at step 8500: 0.399371
      Minibatch accuracy: 89.1%
      Validation accuracy: 88.4%
      Minibatch loss at step 9000: 0.496299
      Minibatch accuracy: 84.4%
      Validation accuracy: 88.5%
      Minibatch loss at step 9500: 0.426205
      Minibatch accuracy: 85.9%
      Validation accuracy: 88.6%
      Minibatch loss at step 10000: 0.395432
      Minibatch accuracy: 87.5%
      Validation accuracy: 88.8%
      Minibatch loss at step 10500: 0.323026
      Minibatch accuracy: 91.4%
      Validation accuracy: 89.1%
      Minibatch loss at step 11000: 0.418872
      Minibatch accuracy: 89.1%
      Validation accuracy: 89.1%
      Minibatch loss at step 11500: 0.372964
      Minibatch accuracy: 89.1%
      Validation accuracy: 89.2%
      Minibatch loss at step 12000: 0.493061
      Minibatch accuracy: 85.2%
      Validation accuracy: 89.0%
      Minibatch loss at step 12500: 0.323986
      Minibatch accuracy: 89.8%
      Validation accuracy: 89.3%
      Minibatch loss at step 13000: 0.532816
      Minibatch accuracy: 86.7%
      Validation accuracy: 89.5%
      Minibatch loss at step 13500: 0.405164
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.3%
      Minibatch loss at step 14000: 0.401133
      Minibatch accuracy: 84.4%
      Validation accuracy: 89.6%
      Minibatch loss at step 14500: 0.441894
      Minibatch accuracy: 86.7%
      Validation accuracy: 89.7%
      Minibatch loss at step 15000: 0.294172
      Minibatch accuracy: 89.8%
      Validation accuracy: 89.6%
      Minibatch loss at step 15500: 0.377510
      Minibatch accuracy: 87.5%
      Validation accuracy: 89.6%
      Minibatch loss at step 16000: 0.272194
      Minibatch accuracy: 93.0%
      Validation accuracy: 90.0%
      Minibatch loss at step 16500: 0.257088
      Minibatch accuracy: 92.2%
      Validation accuracy: 89.6%
      Minibatch loss at step 17000: 0.295775
      Minibatch accuracy: 92.2%
      Validation accuracy: 89.7%
      Minibatch loss at step 17500: 0.115830
      Minibatch accuracy: 96.9%
      Validation accuracy: 89.8%
      Minibatch loss at step 18000: 0.305046
      Minibatch accuracy: 91.4%
      Validation accuracy: 89.9%
      Minibatch loss at step 18500: 0.304270
      Minibatch accuracy: 89.8%
      Validation accuracy: 90.0%
      Minibatch loss at step 19000: 0.230595
      Minibatch accuracy: 93.0%
      Validation accuracy: 90.0%
      Minibatch loss at step 19500: 0.300904
      Minibatch accuracy: 92.2%
      Validation accuracy: 90.0%
      Minibatch loss at step 20000: 0.474510
      Minibatch accuracy: 85.9%
      Validation accuracy: 90.1%
      Test accuracy: 95.7%
      [Finished in 1037.4s]
      '''
    return 0

#########################################
def main():
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

    ####################
    # Problem 1
    # Introduce and tune L2 regularization for both logistic and neural network models.
    # Remember that L2 amounts to adding a penalty on the norm of the weights to the loss.
    # In TensorFlow, you can compute the L2 loss for a tensor t using nn.l2_loss(t).
    # The right amount of regularization should improve your validation / test accuracy.
    #####################
    # Let's start with the logistic model to see will it improve the prediction or not.:
    logistic_L2_regularzation(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)
    logistic_regression_SGD_L2_regularzation_1_hidden_layer(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)




    ##########################
    # Problem 2 section: Let's demonstrate an extreme case of overfitting. Restrict your training data to just a few batches. What happens?
    ########################
    force_to_create_overfit_with_few_batches(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)


    ###########################
    # Problem 3
    # Introduce Dropout on the hidden layer of the neural network.
    # Remember: Dropout should only be introduced during training, not evaluation,
    # otherwise your evaluation results would be stochastic as well.
    # TensorFlow provides nn.dropout() for that, but you have to make sure it's only inserted during training.
    # What happens to our extreme overfitting case?
    ###########################
    dropout_hidden_layer_neural_network(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)

    ###########################################
    # Problem 4
    # Try to get the best performance you can using a multi-layer model! The best reported test accuracy using a deep network is 97.1%.
    # One avenue you can explore is to add multiple layers.
    # Another one is to use learning rate decay:
    # global_step = tf.Variable(0)  # count the number of steps taken.
    # learning_rate = tf.train.exponential_decay(0.5, step, ...)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    ##########################################
    # try to improve the overfitting result as problem3
    multi_layer_dropout_nn_2_hiddenlayer(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)
    #Wow result 95.8 is really good, and the overfit has gone!!

    # let's try one more deeper to see if the result will get better or not
    multi_layer_dropout_nn_deeper_3_hiddenlayer(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)
    # wow, 96.3% much better , but start getting little overfit


    #one more try, the same 3 hidden layer but more neural each layer.
    multi_layer_dropout_nn_deeper_3_hiddenlayer_more_neural(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)
    # we stop tring here, since 95.7 < 96.3%
    return 0

main()
