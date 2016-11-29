# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import collections
import math
import numpy as np
import os,sys
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
import string

filepath=os.path.dirname(os.path.realpath(__file__))
source_path=filepath+'/source'




#Download the data from the source website if necessary.
url = 'http://mattmahoney.net/dc/'
def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  #Found and verified text8.zip
  print(filename)
  print(url + filename.split('/')[-1])
  if not os.path.exists(filename):
    print('Attempting to download:', url + filename.split('/')[-1])
    filename, _ = urlretrieve(url + filename.split('/')[-1], filename)
    print('\n Download Complete! to ',filename)

  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

# Read the data into a string.
def read_data(filename):
  print(filename)
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    for name in f.namelist():
      return tf.compat.as_str(f.read(name))
  #return data


# Utility functions to map characters to vocabulary IDs and back.

vocabulary_size = len(string.ascii_lowercase) + 1 # [a-z] + ' '
first_letter = ord(string.ascii_lowercase[0])
def char2id(char):
  #print(char,string.ascii_lowercase)
  if char in string.ascii_lowercase:
    return ord(char) - first_letter + 1
  elif char == ' ':
    return 0
  else:
    print('Unexpected character: %s' % char.encode('utf-8'))
    return 0

def id2char(dictid):
  if dictid > 0:
    return chr(dictid + first_letter - 1)
  else:
    return ' '
###################################################

#Function to generate a training batch for the LSTM model.
batch_size=64
num_unrollings=10

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    segment = self._text_size // batch_size
    self._cursor = [ offset * segment for offset in range(batch_size)]
    self._last_batch = self._next_batch()

  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in range(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch

  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in range(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches



def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (most likely) character representation."""
  return [id2char(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, characters(b))]
  return s

##########################################
def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in range(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def sample(prediction):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[1, vocabulary_size], dtype=np.float)
  p[0, sample_distribution(prediction[0])] = 1.0
  return p

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]



num_nodes = 64
def simple_LSTM_model(train_batches,valid_batches,valid_size):
  graph = tf.Graph()
  with graph.as_default():

    # Parameters:
    # Input gate: input, previous output, and bias.
    ix = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ib = tf.Variable(tf.zeros([1, num_nodes]))
    # Forget gate: input, previous output, and bias.
    fx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    fm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    fb = tf.Variable(tf.zeros([1, num_nodes]))
    # Memory cell: input, state and bias.
    cx = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    cm = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    cb = tf.Variable(tf.zeros([1, num_nodes]))
    # Output gate: input, previous output, and bias.
    ox = tf.Variable(tf.truncated_normal([vocabulary_size, num_nodes], -0.1, 0.1))
    om = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
    ob = tf.Variable(tf.zeros([1, num_nodes]))
    # Variables saving state across unrollings.
    saved_output = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    saved_state = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
    # Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([vocabulary_size]))

    # Definition of the cell computation.
    def lstm_cell(i, o, state):
      """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
      Note that in this formulation, we omit the various connections between the
      previous state and the gates."""
      input_gate = tf.sigmoid(tf.matmul(i, ix) + tf.matmul(o, im) + ib)
      forget_gate = tf.sigmoid(tf.matmul(i, fx) + tf.matmul(o, fm) + fb)
      update = tf.matmul(i, cx) + tf.matmul(o, cm) + cb
      state = forget_gate * state + input_gate * tf.tanh(update)
      output_gate = tf.sigmoid(tf.matmul(i, ox) + tf.matmul(o, om) + ob)
      return output_gate * tf.tanh(state), state

    # Input data.
    train_data = list()
    for _ in range(num_unrollings + 1):
      train_data.append(
        tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
    train_inputs = train_data[:num_unrollings]
    train_labels = train_data[1:]  # labels are inputs shifted by one time step.

    # Unrolled LSTM loop.
    outputs = list()
    output = saved_output
    state = saved_state
    for i in train_inputs:
      output, state = lstm_cell(i, output, state)
      outputs.append(output)

    # State saving across unrollings.
    with tf.control_dependencies([saved_output.assign(output),
                                  saved_state.assign(state)]):
      # Classifier.
      logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
      loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
          logits, tf.concat(0, train_labels)))

    # Optimizer.
    global_step = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(10.0, global_step, 5000, 0.1, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
    optimizer = optimizer.apply_gradients(zip(gradients, v), global_step=global_step)

    # Predictions.
    train_prediction = tf.nn.softmax(logits)

    # Sampling and validation eval: batch 1, no unrolling.
    sample_input = tf.placeholder(tf.float32, shape=[1, vocabulary_size])
    saved_sample_output = tf.Variable(tf.zeros([1, num_nodes]))
    saved_sample_state = tf.Variable(tf.zeros([1, num_nodes]))
    reset_sample_state = tf.group(
      saved_sample_output.assign(tf.zeros([1, num_nodes])),
      saved_sample_state.assign(tf.zeros([1, num_nodes])))
    sample_output, sample_state = lstm_cell(
      sample_input, saved_sample_output, saved_sample_state)
    with tf.control_dependencies([saved_sample_output.assign(sample_output),
                                  saved_sample_state.assign(sample_state)]):
      sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output, w, b))


  num_steps = 7001
  summary_frequency = 100

  with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    mean_loss = 0
    for step in range(num_steps):
      batches = train_batches.next()
      feed_dict = dict()
      for i in range(num_unrollings + 1):
        feed_dict[train_data[i]] = batches[i]
      _, l, predictions, lr = session.run(
        [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
      mean_loss += l
      if step % summary_frequency == 0:
        if step > 0:
          mean_loss = mean_loss / summary_frequency
        # The mean loss is an estimate of the loss over the last few batches.
        print(
          'Average loss at step %d: %f learning rate: %f' % (step, mean_loss, lr))
        mean_loss = 0
        labels = np.concatenate(list(batches)[1:])
        print('Minibatch perplexity: %.2f' % float(
          np.exp(logprob(predictions, labels))))
        if step % (summary_frequency * 10) == 0:
          # Generate some samples.
          print('=' * 80)
          for _ in range(5):
            feed = sample(random_distribution())
            sentence = characters(feed)[0]
            reset_sample_state.run()
            for _ in range(79):
              prediction = sample_prediction.eval({sample_input: feed})
              feed = sample(prediction)
              sentence += characters(feed)[0]
            print(sentence)
          print('=' * 80)
        # Measure validation set perplexity.
        reset_sample_state.run()
        valid_logprob = 0
        for _ in range(valid_size):
          b = valid_batches.next()
          predictions = sample_prediction.eval({sample_input: b[0]})
          valid_logprob = valid_logprob + logprob(predictions, b[1])
        print('Validation set perplexity: %.2f' % float(np.exp(
          valid_logprob / valid_size)))

        '''
        Initialized
        Average loss at step 0: 3.293269 learning rate: 10.000000
        Minibatch perplexity: 26.93
        ================================================================================
        zcsjubwfirs mspeegrchyx  punksxjxlnima oxxa mftrblcgtnyb rc cewaexmwqozukxiaonji
        rty sgrxloabgdhi ae lfnoi i hkofinvqaetmshmiottssq ektvgwwdnmhpxbyphbjoojcpgiebm
        ytc booiudpamthoyzoiy hdtnlecbkt  et w ninayod   bc eeeeok x eimkcv w tu  ttuese
        wgidlegjltnnceeseb paqiieik h psldlocnwweo nozlh sjpnjsyudiose  orhq e xnfxwu li
        ksioxcewtbagskne lh ilwztndjf  gebiaavu fgv yxtki gqcevaoh oc bieaelve   kizcpie
        ================================================================================
        Validation set perplexity: 20.22
        Average loss at step 100: 2.585056 learning rate: 10.000000
        Minibatch perplexity: 10.66
        Validation set perplexity: 10.86
        Average loss at step 200: 2.250220 learning rate: 10.000000
        Minibatch perplexity: 9.52
        Validation set perplexity: 9.15
        Average loss at step 300: 2.092010 learning rate: 10.000000
        Minibatch perplexity: 7.56
        Validation set perplexity: 7.80
        Average loss at step 400: 2.001497 learning rate: 10.000000
        Minibatch perplexity: 7.56
        Validation set perplexity: 7.36
        Average loss at step 500: 1.937670 learning rate: 10.000000
        Minibatch perplexity: 6.31
        Validation set perplexity: 7.09
        Average loss at step 600: 1.911254 learning rate: 10.000000
        Minibatch perplexity: 6.22
        Validation set perplexity: 6.69
        Average loss at step 700: 1.859234 learning rate: 10.000000
        Minibatch perplexity: 5.65
        Validation set perplexity: 6.61
        Average loss at step 800: 1.821995 learning rate: 10.000000
        Minibatch perplexity: 6.09
        Validation set perplexity: 6.38
        Average loss at step 900: 1.828753 learning rate: 10.000000
        Minibatch perplexity: 7.17
        Validation set perplexity: 6.02
        Average loss at step 1000: 1.823233 learning rate: 10.000000
        Minibatch perplexity: 5.92
        ================================================================================
        ver apapiciver to princh kressed had ally soven liters plictiventon mossicings a
        vate of a prinds of nymp usber from knan basket will whilv unid ogermendip sicio
        opioned the priberack bit comprical afre shilcis mninersetielvarors thizcy glotl
        wed in guil in will entermagh to ridiry expiction clidelt kip augnech in the sim
        ho live brapaca an the interfus glattor bidyual rian thes elarim on the bpndfren
        ================================================================================
        Validation set perplexity: 6.00
        Average loss at step 1100: 1.773573 learning rate: 10.000000
        Minibatch perplexity: 5.71
        Validation set perplexity: 5.73
        Average loss at step 1200: 1.755216 learning rate: 10.000000
        Minibatch perplexity: 5.95
        Validation set perplexity: 5.51
        Average loss at step 1300: 1.732795 learning rate: 10.000000
        Minibatch perplexity: 5.82
        Validation set perplexity: 5.46
        Average loss at step 1400: 1.744514 learning rate: 10.000000
        Minibatch perplexity: 4.90
        Validation set perplexity: 5.44
        Average loss at step 1500: 1.739362 learning rate: 10.000000
        Minibatch perplexity: 6.18
        Validation set perplexity: 5.22
        Average loss at step 1600: 1.745746 learning rate: 10.000000
        Minibatch perplexity: 5.31
        Validation set perplexity: 5.15
        Average loss at step 1700: 1.710386 learning rate: 10.000000
        Minibatch perplexity: 4.56
        Validation set perplexity: 5.24
        Average loss at step 1800: 1.675063 learning rate: 10.000000
        Minibatch perplexity: 5.16
        Validation set perplexity: 5.10
        Average loss at step 1900: 1.647817 learning rate: 10.000000
        Minibatch perplexity: 5.93
        Validation set perplexity: 5.15
        Average loss at step 2000: 1.696540 learning rate: 10.000000
        Minibatch perplexity: 4.91
        ================================================================================
        ver dering becan expector will the proto and thinder wassices two zero f anro d
        ne alsp numesting word might of crossed the vives seament saingre knoves orcho h
        wer edeward phodespers the bolnided abristirg viop one nine eight zero zero the
        entle ch minoty south was ling lost use of three two was ressotitude stotrever o
        creef to algepacs iradions of the this s ofter attem the after books s linelts b
        ================================================================================
        Validation set perplexity: 5.03
        Average loss at step 2100: 1.683888 learning rate: 10.000000
        Minibatch perplexity: 5.29
        Validation set perplexity: 4.75
        Average loss at step 2200: 1.681385 learning rate: 10.000000
        Minibatch perplexity: 5.12
        Validation set perplexity: 5.02
        Average loss at step 2300: 1.640618 learning rate: 10.000000
        Minibatch perplexity: 5.62
        Validation set perplexity: 4.75
        Average loss at step 2400: 1.659028 learning rate: 10.000000
        Minibatch perplexity: 5.16
        Validation set perplexity: 4.72
        Average loss at step 2500: 1.677448 learning rate: 10.000000
        Minibatch perplexity: 5.64
        Validation set perplexity: 4.58
        Average loss at step 2600: 1.653226 learning rate: 10.000000
        Minibatch perplexity: 5.77
        Validation set perplexity: 4.58
        Average loss at step 2700: 1.657197 learning rate: 10.000000
        Minibatch perplexity: 5.09
        Validation set perplexity: 4.61
        Average loss at step 2800: 1.650109 learning rate: 10.000000
        Minibatch perplexity: 5.05
        Validation set perplexity: 4.56
        Average loss at step 2900: 1.651510 learning rate: 10.000000
        Minibatch perplexity: 4.67
        Validation set perplexity: 4.53
        Average loss at step 3000: 1.648767 learning rate: 10.000000
        Minibatch perplexity: 4.86
        ================================================================================
        duck istadming hin themer dan astigra bansism indian accuyle ffro ball appearet
        uscin which ard science ratown necation of dilar audisted forthimating themser a
        fore s indatically considered formemen salk of into four five fout and and silut
        pen of ascebtorssied in the first forck by six four one regenne is becassionaly
        onia voywed gohn britic onlo ulfo several nine s alchil counties with in the and
        ================================================================================
        Validation set perplexity: 4.65
        Average loss at step 3100: 1.628762 learning rate: 10.000000
        Minibatch perplexity: 5.69
        Validation set perplexity: 4.56
        Average loss at step 3200: 1.643814 learning rate: 10.000000
        Minibatch perplexity: 5.09
        Validation set perplexity: 4.68
        Average loss at step 3300: 1.637166 learning rate: 10.000000
        Minibatch perplexity: 5.77
        Validation set perplexity: 4.53
        Average loss at step 3400: 1.669845 learning rate: 10.000000
        Minibatch perplexity: 6.09
        Validation set perplexity: 4.61
        Average loss at step 3500: 1.656434 learning rate: 10.000000
        Minibatch perplexity: 5.58
        Validation set perplexity: 4.60
        Average loss at step 3600: 1.663732 learning rate: 10.000000
        Minibatch perplexity: 4.89
        Validation set perplexity: 4.53
        Average loss at step 3700: 1.645358 learning rate: 10.000000
        Minibatch perplexity: 5.51
        Validation set perplexity: 4.52
        Average loss at step 3800: 1.643781 learning rate: 10.000000
        Minibatch perplexity: 4.72
        Validation set perplexity: 4.60
        Average loss at step 3900: 1.641687 learning rate: 10.000000
        Minibatch perplexity: 5.92
        Validation set perplexity: 4.60
        Average loss at step 4000: 1.651016 learning rate: 10.000000
        Minibatch perplexity: 4.84
        ================================================================================
        x recalled conclonuble schefical stutial over indian casse youp insometally cons
        ve impiding h proper whick shapered to remy and was minjmsting miniam gamen he g
        hims humes in the to the sitre oris setinatized the dracting from in eijabics an
         prockas the edicaquctel lathol korg phare and have x eding the flam for exeseti
        ums anting of rigrap north bbeaked from inliac the clolal overliritical sccodown
        ================================================================================
        Validation set perplexity: 4.57
        Average loss at step 4100: 1.633769 learning rate: 10.000000
        Minibatch perplexity: 5.24
        Validation set perplexity: 4.68
        Average loss at step 4200: 1.633181 learning rate: 10.000000
        Minibatch perplexity: 5.01
        Validation set perplexity: 4.55
        Average loss at step 4300: 1.611738 learning rate: 10.000000
        Minibatch perplexity: 5.01
        Validation set perplexity: 4.44
        Average loss at step 4400: 1.612348 learning rate: 10.000000
        Minibatch perplexity: 4.86
        Validation set perplexity: 4.34
        Average loss at step 4500: 1.612636 learning rate: 10.000000
        Minibatch perplexity: 5.24
        Validation set perplexity: 4.53
        Average loss at step 4600: 1.611311 learning rate: 10.000000
        Minibatch perplexity: 5.06
        Validation set perplexity: 4.49
        Average loss at step 4700: 1.622968 learning rate: 10.000000
        Minibatch perplexity: 5.22
        Validation set perplexity: 4.55
        Average loss at step 4800: 1.628561 learning rate: 10.000000
        Minibatch perplexity: 4.89
        Validation set perplexity: 4.43
        Average loss at step 4900: 1.632979 learning rate: 10.000000
        Minibatch perplexity: 5.13
        Validation set perplexity: 4.55
        Average loss at step 5000: 1.606279 learning rate: 1.000000
        Minibatch perplexity: 5.30
        ================================================================================
        ring initionly with the ras often loston in they and fullia into office one seve
        per the collection bostle his mankentiallive sing for commontent that in that ev
        ligh shiplosails in the infersinion phobultions and in refamed bee of virecker m
         blapnest suth a dry priment nher one ozerage time not semen hicestan devide wou
        s english the eight seven of baskforkes the evamilly or a bit socept of the libe
        ================================================================================
        Validation set perplexity: 4.56
        Average loss at step 5100: 1.604023 learning rate: 1.000000
        Minibatch perplexity: 4.88
        Validation set perplexity: 4.40
        Average loss at step 5200: 1.591964 learning rate: 1.000000
        Minibatch perplexity: 4.57
        Validation set perplexity: 4.32
        Average loss at step 5300: 1.578378 learning rate: 1.000000
        Minibatch perplexity: 5.26
        Validation set perplexity: 4.32
        Average loss at step 5400: 1.577844 learning rate: 1.000000
        Minibatch perplexity: 5.04
        Validation set perplexity: 4.29
        Average loss at step 5500: 1.566368 learning rate: 1.000000
        Minibatch perplexity: 4.99
        Validation set perplexity: 4.27
        Average loss at step 5600: 1.579829 learning rate: 1.000000
        Minibatch perplexity: 4.81
        Validation set perplexity: 4.27
        Average loss at step 5700: 1.566134 learning rate: 1.000000
        Minibatch perplexity: 4.73
        Validation set perplexity: 4.25
        Average loss at step 5800: 1.580992 learning rate: 1.000000
        Minibatch perplexity: 5.24
        Validation set perplexity: 4.25
        Average loss at step 5900: 1.574374 learning rate: 1.000000
        Minibatch perplexity: 4.61
        Validation set perplexity: 4.23
        Average loss at step 6000: 1.547250 learning rate: 1.000000
        Minibatch perplexity: 5.00
        ================================================================================
        ve traditional stra came kwortian special is caj the frec that mary coniea one s
        x karkavi stamed with economy opposary and some and be pay king thinceleas appor
        y blastan hiss sain eurinuant musicier zerouge mecauoution also liferation two d
        version of is grounos worly the approrise and strentivity are coliviby is and th
        s and suzp thisthid whe nater as news are wide prainist in a may been in early a
        ================================================================================
        Validation set perplexity: 4.24
        Average loss at step 6100: 1.562749 learning rate: 1.000000
        Minibatch perplexity: 4.13
        Validation set perplexity: 4.22
        Average loss at step 6200: 1.535250 learning rate: 1.000000
        Minibatch perplexity: 5.30
        Validation set perplexity: 4.19
        Average loss at step 6300: 1.541521 learning rate: 1.000000
        Minibatch perplexity: 4.62
        Validation set perplexity: 4.20
        Average loss at step 6400: 1.539425 learning rate: 1.000000
        Minibatch perplexity: 4.87
        Validation set perplexity: 4.19
        Average loss at step 6500: 1.555892 learning rate: 1.000000
        Minibatch perplexity: 4.68
        Validation set perplexity: 4.22
        Average loss at step 6600: 1.595688 learning rate: 1.000000
        Minibatch perplexity: 4.65
        Validation set perplexity: 4.20
        Average loss at step 6700: 1.576150 learning rate: 1.000000
        Minibatch perplexity: 4.48
        Validation set perplexity: 4.21
        Average loss at step 6800: 1.603587 learning rate: 1.000000
        Minibatch perplexity: 5.07
        Validation set perplexity: 4.18
        Average loss at step 6900: 1.579747 learning rate: 1.000000
        Minibatch perplexity: 4.65
        Validation set perplexity: 4.22
        Average loss at step 7000: 1.575845 learning rate: 1.000000
        Minibatch perplexity: 5.02
        ================================================================================
        on from when three three such suc jourbs are neques and para which rurerly of th
        es transcondzat it is one zero s popegnas for the she crean the panevers roimsab
        zing most mac plailed by their playumary on in the under status side then with t
        fist bephs to day s common son words setai are join whether one five th is see s
        le is byyament cathopulation more there and jorn wideen regulation has asha free
        ================================================================================
        Validation set perplexity: 4.19
        '''

    return 0

#########################################################################
def main():
    ####################
    # Data Section
    ####################

    #Download the data from the source website if necessary.
    filename = maybe_download(source_path+'/text8.zip', 31344016)
    # Read the data into a word list.
    text = read_data(filename)
    print('words size %d' % len(text))
    #print('words content %s, type= %s' % (words[0:100],type(words)))



    '''
    words size 100000000
    words content ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against', 'early',
    'working', 'class', 'radicals', 'including', 'the', 'diggers', 'of', 'the', 'english', 'revolution', 'and', 'the',
    'sans', 'culottes', 'of', 'the', 'french', 'revolution', 'whilst', 'the', 'term', 'is', 'still', 'used', 'in', 'a',
    'pejorative', 'way', 'to', 'describe', 'any', 'act', 'that', 'used', 'violent', 'means', 'to', 'destroy', 'the',
    'organization', 'of', 'society', 'it', 'has', 'also', 'been', 'taken', 'up', 'as', 'a', 'positive', 'label', 'by',
    'self', 'defined', 'anarchists', 'the', 'word', 'anarchism', 'is', 'derived', 'from', 'the', 'greek', 'without',
    'archons', 'ruler', 'chief', 'king', 'anarchism', 'as', 'a', 'political', 'philosophy', 'is', 'the', 'belief',
    'that', 'rulers', 'are', 'unnecessary', 'and', 'should', 'be', 'abolished', 'although', 'there', 'are', 'differing'],
    type= <class 'list'>
    '''

    valid_size = 1000
    valid_text = text[:valid_size]
    train_text = text[valid_size:]
    train_size = len(train_text)
    print(train_size, train_text[:64])
    '''
    99999000 ons anarchists advocate social relations based upon voluntary as
    '''
    print(valid_size, valid_text[:64])
    '''
    1000  anarchism originated as a term of abuse first used against earl
    '''


    print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
    '''Unexpected character: b'\xc3\xaf'
      1 26 0 0
    '''
    print(id2char(1), id2char(26), id2char(0))
    '''a z '''

    #generate a training batch for the LSTM model.
    train_batches = BatchGenerator(train_text, batch_size, num_unrollings)
    valid_batches = BatchGenerator(valid_text, 1, 1)

    print(batches2string(train_batches.next()))
    '''
    ['ons anarchi', 'when milita', 'lleria arch', ' abbeys and', 'married urr', 'hel and ric',
    'y and litur', 'ay opened f', 'tion from t', 'migration t', 'new york ot', 'he boeing s',
    'e listed wi', 'eber has pr', 'o be made t', 'yer who rec', 'ore signifi', 'a fierce cr',
    ' two six ei', 'aristotle s', 'ity can be ', ' and intrac', 'tion of the', 'dy to pass ',
    'f certain d', 'at it will ', 'e convince ', 'ent told hi', 'ampaign and', 'rver side s',
    'ious texts ', 'o capitaliz', 'a duplicate', 'gh ann es d', 'ine january', 'ross zero t',
    'cal theorie', 'ast instanc', ' dimensiona', 'most holy m', 't s support', 'u is still ',
    'e oscillati', 'o eight sub', 'of italy la', 's the tower', 'klahoma pre', 'erprise lin',
    'ws becomes ', 'et in a naz', 'the fabian ', 'etchy to re', ' sharman ne', 'ised empero',
    'ting in pol', 'd neo latin', 'th risky ri', 'encyclopedi', 'fense the a', 'duating fro',
    'treet grid ', 'ations more', 'appeal of d', 'si have mad']
    '''
    print(batches2string(train_batches.next()))
    '''
    ['ists advoca', 'ary governm', 'hes nationa', 'd monasteri', 'raca prince', 'chard baer ',
     'rgical lang', 'for passeng', 'the nationa', 'took place ', 'ther well k', 'seven six s',
     'ith a gloss', 'robably bee', 'to recogniz', 'ceived the ', 'icant than ', 'ritic of th',
     'ight in sig', 's uncaused ', ' lost as in', 'cellular ic', 'e size of t', ' him a stic',
     'drugs confu', ' take to co', ' the priest', 'im to name ', 'd barred at', 'standard fo',
     ' such as es', 'ze on the g', 'e of the or', 'd hiver one', 'y eight mar', 'the lead ch',
     'es classica', 'ce the non ', 'al analysis', 'mormons bel', 't or at lea', ' disagreed ',
     'ing system ', 'btypes base', 'anguages th', 'r commissio', 'ess one nin', 'nux suse li',
     ' the first ', 'zi concentr', ' society ne', 'elatively s', 'etworks sha', 'or hirohito',
     'litical ini', 'n most of t', 'iskerdoo ri', 'ic overview', 'air compone', 'om acnm acc',
     ' centerline', 'e than any ', 'devotional ', 'de such dev']
    '''
    print(batches2string(valid_batches.next()))
    '''
    [' a']
    '''
    print(batches2string(valid_batches.next()))
    '''
    ['an']
    '''


    #display the shape or the content of the variables to better understand their structure:
    print(train_batches.next()[1].shape)
    print(len(train_text) // batch_size)
    print(len(string.ascii_lowercase))
    print(np.zeros(shape=(2, 4), dtype=np.float))
    '''
    (64, 27)
    1562484
    26
    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]]
    '''
    print("\nData Ready\n")

    #########################
    # Model : Simple LSTM Model.
    ########################
    simple_LSTM_model(train_batches,valid_batches,valid_size)


    ###################
    # Problem 1
    # You might have noticed that the definition of the LSTM cell involves 4 matrix multiplications with the input,
    # and 4 matrix multiplications with the output. Simplify the expression by using a single matrix multiply for each,
    # and variables that are 4 times larger.
    ##################


    return 0

main()



