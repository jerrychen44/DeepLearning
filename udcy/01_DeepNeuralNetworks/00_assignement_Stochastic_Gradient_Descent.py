
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import os
import sys


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


def normal_gradient_descent_tensorflow(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):

    # With gradient descent training, even this much data is prohibitive.
    # Subset the training data for faster turnaround.
    train_subset = 10000
    '''
    Training set (200000, 784) (200000, 10)
    Validation set (10000, 784) (10000, 10)
    Test set (10000, 784) (10000, 10)
    '''

    graph = tf.Graph()
    with graph.as_default():

      # Input data.
      # Load the training, validation and test data into constants that are
      # attached to the graph.
      tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
      tf_train_labels = tf.constant(train_labels[:train_subset])
      tf_valid_dataset = tf.constant(valid_dataset)
      tf_test_dataset = tf.constant(test_dataset)

      # Variables.
      # These are the parameters that we are going to be training. The weight
      # matrix will be initialized using random values following a (truncated)
      # normal distribution. The biases get initialized to zero.
      weights = tf.Variable(
        tf.truncated_normal([image_size * image_size, num_labels]))
      biases = tf.Variable(tf.zeros([num_labels]))

      # Training computation.
      # We multiply the inputs with the weight matrix, and add biases. We compute
      # the softmax and cross-entropy (it's one operation in TensorFlow, because
      # it's very common, and it can be optimized). We take the average of this
      # cross-entropy across all training examples: that's our loss.
      logits = tf.matmul(tf_train_dataset, weights) + biases
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

      # Optimizer.
      # We are going to find the minimum of this loss using gradient descent.
      optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

      # Predictions for the training, validation, and test data.
      # These are not part of training, but merely here so that we can report
      # accuracy figures as we train.
      train_prediction = tf.nn.softmax(logits)
      valid_prediction = tf.nn.softmax(
        tf.matmul(tf_valid_dataset, weights) + biases)
      test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)




    #Let's run this computation and iterate:
    num_steps = 801
    with tf.Session(graph=graph) as session:
      # This is a one-time operation which ensures the parameters get initialized as
      # we described in the graph: random weights for the matrix, zeros for the
      # biases.
      #tf.global_variables_initializer().run()# old code
      tf.initialize_all_variables().run()
      print('Initialized')
      for step in range(num_steps):
        # Run the computations. We tell .run() that we want to run the optimizer,
        # and get the loss value and the training predictions returned as numpy
        # arrays.
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
          print('Loss at step %d: %f' % (step, l))
          print('Training accuracy: %.1f%%' % accuracy(
            predictions, train_labels[:train_subset, :]))
          # Calling .eval() on valid_prediction is basically like calling run(), but
          # just to get that one numpy array. Note that it recomputes all its graph
          # dependencies.
          print('Validation accuracy: %.1f%%' % accuracy(
            valid_prediction.eval(), valid_labels))
          '''
          train_subset = 10000
          Initialized
          Loss at step 0: 19.427744
          Training accuracy: 7.5%
          Validation accuracy: 8.2%
          Loss at step 100: 2.264809
          Training accuracy: 72.2%
          Validation accuracy: 70.7%
          Loss at step 200: 1.830585
          Training accuracy: 75.2%
          Validation accuracy: 72.9%
          Loss at step 300: 1.588544
          Training accuracy: 76.5%
          Validation accuracy: 73.7%
          Loss at step 400: 1.424872
          Training accuracy: 77.5%
          Validation accuracy: 74.1%
          Loss at step 500: 1.303166
          Training accuracy: 78.1%
          Validation accuracy: 74.2%
          Loss at step 600: 1.207225
          Training accuracy: 78.8%
          Validation accuracy: 74.4%
          Loss at step 700: 1.129053
          Training accuracy: 79.3%
          Validation accuracy: 74.5%
          Loss at step 800: 1.063993
          Training accuracy: 79.6%
          Validation accuracy: 74.8%
          [Finished in 44.8s]
          '''
          '''
          train_subset = 50000
          Loss at step 0: 20.282442
          Training accuracy: 8.1%
          Validation accuracy: 9.0%
          Loss at step 100: 2.400129
          Training accuracy: 71.5%
          Validation accuracy: 70.7%
          Loss at step 200: 1.952625
          Training accuracy: 74.1%
          Validation accuracy: 73.3%
          Loss at step 300: 1.723698
          Training accuracy: 75.1%
          Validation accuracy: 74.3%
          Loss at step 400: 1.574195
          Training accuracy: 75.8%
          Validation accuracy: 74.9%
          Loss at step 500: 1.464485
          Training accuracy: 76.3%
          Validation accuracy: 75.3%
          Loss at step 600: 1.378991
          Training accuracy: 76.7%
          Validation accuracy: 75.5%
          Loss at step 700: 1.309759
          Training accuracy: 77.0%
          Validation accuracy: 75.7%
          Loss at step 800: 1.252104
          Training accuracy: 77.2%
          Validation accuracy: 76.0%
          Test accuracy: 84.0%
          '''
      print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
      '''
      train_subset = 10000
      Test accuracy: 82.3%

      train_subset = 50000
      Test accuracy: 84.0%
      '''
    return 0


def stochastic_gradient_descent_tensorflow(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
  # The graph will be similar, except that instead of
  # holding all the training data into a constant node,
  # we create a Placeholder node which will be fed actual data at every call of session.run().
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

    # Variables.
    weights = tf.Variable(
      tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(
      tf.matmul(tf_valid_dataset, weights) + biases)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)

    #Let's run it:
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
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
          '''
          Minibatch loss at step 0: 15.809065
          Minibatch accuracy: 9.4%
          Validation accuracy: 10.8%
          Minibatch loss at step 500: 1.286326
          Minibatch accuracy: 78.9%
          Validation accuracy: 76.3%
          Minibatch loss at step 1000: 1.313491
          Minibatch accuracy: 77.3%
          Validation accuracy: 77.7%
          Minibatch loss at step 1500: 0.733546
          Minibatch accuracy: 82.8%
          Validation accuracy: 78.3%
          Minibatch loss at step 2000: 0.806183
          Minibatch accuracy: 85.9%
          Validation accuracy: 78.4%
          Minibatch loss at step 2500: 0.931979
          Minibatch accuracy: 78.1%
          Validation accuracy: 79.3%
          Minibatch loss at step 3000: 0.923167
          Minibatch accuracy: 80.5%
          Validation accuracy: 79.4%

          [Finished in 8.9s]
          '''
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
      ''' Test accuracy: 85.9% '''

  return 0

def logistic_regression_SGD_1_hidden_layer(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):

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
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run(
          [optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(
            valid_prediction.eval(), valid_labels))
          '''
          Minibatch loss at step 0: 352.830933
          Minibatch accuracy: 7.0%
          Validation accuracy: 23.2%
          Minibatch loss at step 500: 27.305231
          Minibatch accuracy: 80.5%
          Validation accuracy: 78.3%
          Minibatch loss at step 1000: 12.617765
          Minibatch accuracy: 79.7%
          Validation accuracy: 80.9%
          Minibatch loss at step 1500: 5.883183
          Minibatch accuracy: 89.8%
          Validation accuracy: 80.2%
          Minibatch loss at step 2000: 3.081453
          Minibatch accuracy: 87.5%
          Validation accuracy: 81.1%
          Minibatch loss at step 2500: 2.968213
          Minibatch accuracy: 81.2%
          Validation accuracy: 81.6%
          Minibatch loss at step 3000: 1.669900
          Minibatch accuracy: 83.6%
          Validation accuracy: 81.6%

          [Finished in 83.5s]
          '''
      print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
      '''Test accuracy: 89.4%'''


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



    ##########################
    # TensorFlow: normal GD vs SGD
    ##########################
    # Let's load all the data into TensorFlow and
    # build the computation graph corresponding to our training:

    #normal_gradient_descent_tensorflow(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)



    #Let's now switch to stochastic gradient descent training instead, which is much faster.
    #stochastic_gradient_descent_tensorflow(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)



    ########################
    # Problem :Turn the logistic regression example with SGD into a 1-hidden layer neural network
    # with rectified linear units nn.relu() and 1024 hidden nodes.
    # This model should improve your validation / test accuracy.
    ########################
    logistic_regression_SGD_1_hidden_layer(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)

    return 0

main()
