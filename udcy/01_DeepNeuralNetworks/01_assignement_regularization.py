
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
    #logistic_L2_regularzation(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)
    #logistic_regression_SGD_L2_regularzation_1_hidden_layer(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)




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

    return 0

main()
