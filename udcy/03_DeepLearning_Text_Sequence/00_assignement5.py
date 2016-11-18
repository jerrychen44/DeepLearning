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
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

# Build the dictionary and replace rare words with UNK token.
vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)

  data = list()
  unk_count = 0
  for word in words:
    if word in dictionary:
      index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
  return data, count, dictionary, reverse_dictionary

# Function to generate a training batch for the skip-gram model.
data_index = 0

def generate_batch(batch_size, num_skips, skip_window,data):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels



####################
# model
####################
def skip_gram_model(data,reverse_dictionary):
    batch_size = 128
    embedding_size = 128 # Dimension of the embedding vector.
    skip_window = 1 # How many words to consider left and right.
    num_skips = 2 # How many times to reuse an input to generate a label.
    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16 # Random set of words to evaluate similarity on.
    valid_window = 100 # Only pick dev samples in the head of the distribution.
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    num_sampled = 64 # Number of negative examples to sample.

    graph = tf.Graph()

    with graph.as_default(), tf.device('/cpu:0'):

      # Input data.
      train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

      # Variables.
      embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
      softmax_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                             stddev=1.0 / math.sqrt(embedding_size)))
      softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))

      # Model.
      # Look up embeddings for inputs.
      embed = tf.nn.embedding_lookup(embeddings, train_dataset)
      # Compute the softmax loss, using a sample of the negative labels each time.
      loss = tf.reduce_mean(
        tf.nn.sampled_softmax_loss(softmax_weights, softmax_biases, embed,
                                   train_labels, num_sampled, vocabulary_size))

      # Optimizer.
      optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

      # Compute the similarity between minibatch examples and all embeddings.
      # We use the cosine distance:
      norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
      normalized_embeddings = embeddings / norm
      valid_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, valid_dataset)
      similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


    num_steps = 100001

    with tf.Session(graph=graph) as session:
      tf.initialize_all_variables().run()
      print('Initialized')
      average_loss = 0
      for step in range(num_steps):
        batch_data, batch_labels = generate_batch(
          batch_size, num_skips, skip_window,data)
        feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)
        average_loss += l
        if step % 2000 == 0:
          if step > 0:
            average_loss = average_loss / 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step %d: %f' % (step, average_loss))
          average_loss = 0
        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8 # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k+1]
            log = 'Nearest to %s:' % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log = '%s %s,' % (log, close_word)
            print(log)
      final_embeddings = normalized_embeddings.eval()

    '''
    Initialized
    Average loss at step 0: 7.932007
    Nearest to an: belgrade, studies, clowning, ngos, penetrates, reorganization, porous, prescribing,
    Nearest to system: exports, praxis, pstn, sinusoidal, tuvalu, honorable, frud, lawmaking,
    Nearest to been: nh, multitude, isaacs, frames, autos, mayan, junk, telecom,
    Nearest to history: franchisees, subcategories, decca, mpa, striped, meps, comorian, satanism,
    Nearest to they: carney, olden, breakthrough, pages, controversially, nahum, piccadilly, slavonia,
    Nearest to he: apollinaris, bower, racial, barcode, cleave, clashed, blackadder, letterboxed,
    Nearest to from: extramarital, gigabit, benedictine, thatcherism, eugen, volts, sda, behavioural,
    Nearest to united: mobutu, strait, huffman, wap, floor, airlines, djinn, enforcing,
    Nearest to but: joliet, confidence, clutter, objective, pineal, sponsors, ecmascript, needless,
    Nearest to years: maus, restored, decapitation, mages, catchphrase, consequent, instrument, tumours,
    Nearest to more: clotting, coastal, reichenberger, gil, doctor, germs, hypertext, heavy,
    Nearest to no: overstated, reassert, dunst, mayo, land, firstborn, assistant, stalls,
    Nearest to are: bun, yugoslavia, pragmatism, audiobook, protests, cooperation, beauharnais, charcot,
    Nearest to also: newton, hemisphere, electronica, geoff, riefenstahl, desmodromic, jehu, nnrot,
    Nearest to american: ia, ponies, annealing, macintoshes, disproportionately, anaheim, universit, mei,
    Nearest to may: clipping, moderators, detonating, salvador, esprit, estates, specification, redacted,
    Average loss at step 2000: 4.344834
    Average loss at step 4000: 3.873583
    Average loss at step 6000: 3.784536
    Average loss at step 8000: 3.684439
    Average loss at step 10000: 3.617849
    Nearest to an: belgrade, the, rnberg, alveolar, clowning, rosy, cannot, novas,
    Nearest to system: exports, patronage, praxis, mmorpgs, vma, honorable, bauer, pstn,
    Nearest to been: nh, be, was, by, commonly, undertake, were, rumored,
    Nearest to history: decca, logarithms, striped, valerie, laga, congregational, franchisees, sharman,
    Nearest to they: he, carney, who, curries, there, we, not, it,
    Nearest to he: it, she, they, who, apollinaris, molino, cunt, there,
    Nearest to from: in, priscilla, ibelin, after, tanzania, into, between, thom,
    Nearest to united: strait, euskal, alison, feuerbach, codenamed, same, wap, retention,
    Nearest to but: were, it, over, where, which, within, when, are,
    Nearest to years: maus, marseillaise, scripts, ciprofloxacin, keeps, eurofighter, affirm, mimicked,
    Nearest to more: amplifiers, clotting, flanagan, coastal, cq, phylogenetic, breslau, kyrie,
    Nearest to no: marques, reassert, heb, symphonies, tcm, camelopardalis, amicus, vague,
    Nearest to are: were, is, have, was, gives, yugoslavia, protests, budgetary,
    Nearest to also: often, which, pappus, desmodromic, syndicalist, it, not, propellant,
    Nearest to american: pumps, length, monsieur, koenig, impervious, english, hdi, vault,
    Nearest to may: clipping, moderators, mixers, would, exoplanets, falkland, moot, revert,
    Average loss at step 12000: 3.598697
    Average loss at step 14000: 3.568728
    Average loss at step 16000: 3.405216
    Average loss at step 18000: 3.457500
    Average loss at step 20000: 3.539257
    Nearest to an: rnberg, belgrade, the, mohair, clowning, alveolar, novas, entrenchment,
    Nearest to system: exports, equations, praxis, patronage, reassert, prisoner, honorable, fontsize,
    Nearest to been: be, become, had, was, by, were, commonly, physiological,
    Nearest to history: decca, logarithms, laga, brill, valerie, congregational, proteus, addicts,
    Nearest to they: he, there, we, she, who, you, it, not,
    Nearest to he: it, she, they, who, molino, there, never, this,
    Nearest to from: into, at, in, between, after, thom, and, decoys,
    Nearest to united: same, alison, euskal, strait, codenamed, feuerbach, kabuki, wap,
    Nearest to but: however, which, and, though, when, does, where, while,
    Nearest to years: scripts, marseillaise, maus, exporter, generates, eurofighter, affirm, keeps,
    Nearest to more: less, most, cq, clotting, amplifiers, communism, flanagan, cleaved,
    Nearest to no: a, marques, heb, tcm, magdalena, secrecy, vague, based,
    Nearest to are: were, is, have, include, be, while, other, harlow,
    Nearest to also: often, pappus, not, which, sometimes, now, who, generally,
    Nearest to american: english, british, hdi, pumps, monsieur, russell, australian, vault,
    Nearest to may: would, can, might, will, could, must, should, rockies,
    Average loss at step 22000: 3.493871
    Average loss at step 24000: 3.489113
    Average loss at step 26000: 3.480586
    Average loss at step 28000: 3.483518
    Average loss at step 30000: 3.502667
    Nearest to an: belgrade, electorate, another, alveolar, rnberg, censors, ye, rosy,
    Nearest to system: exports, fontsize, wojty, equations, praxis, reassert, trees, systems,
    Nearest to been: become, be, was, were, physiological, cohorts, commonly, newcastle,
    Nearest to history: decca, brill, valerie, zigzag, laga, liffey, republika, logarithms,
    Nearest to they: there, we, he, it, who, you, these, she,
    Nearest to he: it, she, they, who, there, never, dachau, molino,
    Nearest to from: into, in, after, between, among, through, by, priscilla,
    Nearest to united: same, alison, euskal, strait, wap, codenamed, kabuki, retention,
    Nearest to but: however, although, where, and, which, while, though, that,
    Nearest to years: marseillaise, scripts, exporter, year, maus, generates, days, eurofighter,
    Nearest to more: less, most, very, flanagan, highly, clotting, cq, larger,
    Nearest to no: a, secrecy, syndicalist, epistemology, magdalena, assistant, decide, redacted,
    Nearest to are: were, have, is, be, include, these, some, niebuhr,
    Nearest to also: often, sometimes, now, who, pappus, still, which, generally,
    Nearest to american: english, british, australian, pumps, lever, fashions, koenig, hdi,
    Nearest to may: can, would, will, could, must, might, should, cannot,
    Average loss at step 32000: 3.495105
    Average loss at step 34000: 3.493046
    Average loss at step 36000: 3.452861
    Average loss at step 38000: 3.306236
    Average loss at step 40000: 3.427045
    Nearest to an: belgrade, another, alveolar, anointing, electorate, rnberg, rosy, novas,
    Nearest to system: exports, systems, fontsize, praxis, equations, federation, cricket, hyperlinks,
    Nearest to been: be, become, were, was, physiological, already, being, undertake,
    Nearest to history: aspects, insolvency, addicts, mpa, republika, caribbean, franchisees, decca,
    Nearest to they: we, there, he, you, these, it, she, who,
    Nearest to he: she, it, they, who, there, molino, never, i,
    Nearest to from: into, through, for, after, of, across, in, on,
    Nearest to united: alison, euskal, kabuki, same, codenamed, baroness, strait, wap,
    Nearest to but: however, although, though, while, which, where, that, it,
    Nearest to years: days, year, times, marseillaise, scripts, exporter, generates, bits,
    Nearest to more: less, most, very, larger, highly, flanagan, greater, cq,
    Nearest to no: epistemology, any, syndicalist, ref, civilised, secrecy, or, magdalena,
    Nearest to are: were, have, be, these, include, crashing, is, phylum,
    Nearest to also: often, still, now, sometimes, usually, which, not, there,
    Nearest to american: british, australian, english, russell, french, fashions, jumbo, koenig,
    Nearest to may: can, would, will, could, might, must, should, cannot,
    Average loss at step 42000: 3.435023
    Average loss at step 44000: 3.456278
    Average loss at step 46000: 3.443209
    Average loss at step 48000: 3.357582
    Average loss at step 50000: 3.383017
    Nearest to an: belgrade, rnberg, clowning, anointing, alveolar, electorate, censors, another,
    Nearest to system: systems, exports, equations, counter, fontsize, compulsory, commission, wojty,
    Nearest to been: become, be, were, was, already, undertake, nicolae, enrollment,
    Nearest to history: insolvency, addicts, bluegrass, refrigerators, aspects, franchisees, list, brill,
    Nearest to they: we, he, there, it, you, she, who, these,
    Nearest to he: she, it, they, who, there, never, molino, eventually,
    Nearest to from: in, into, through, across, at, after, ibelin, between,
    Nearest to united: kabuki, alison, codenamed, euskal, wap, confederate, retention, strait,
    Nearest to but: however, although, and, though, while, than, when, which,
    Nearest to years: days, times, year, scripts, eurofighter, decades, months, generates,
    Nearest to more: less, most, very, larger, highly, extremely, greater, cq,
    Nearest to no: any, a, epistemology, pretense, she, syndicalist, volta, magdalena,
    Nearest to are: were, have, is, aerobatics, these, be, witten, include,
    Nearest to also: often, still, which, now, sometimes, then, who, newton,
    Nearest to american: australian, british, english, fashions, african, jumbo, irish, italian,
    Nearest to may: can, would, will, must, should, might, could, cannot,
    Average loss at step 52000: 3.433621
    Average loss at step 54000: 3.429793
    Average loss at step 56000: 3.437160
    Average loss at step 58000: 3.397012
    Average loss at step 60000: 3.388936
    Nearest to an: belgrade, alveolar, rnberg, clowning, censors, electorate, another, carriers,
    Nearest to system: systems, exports, commission, network, computers, counter, equations, federation,
    Nearest to been: become, be, already, was, were, cohorts, sunset, expel,
    Nearest to history: rond, caribbean, list, aspects, politics, mpa, meanings, moravian,
    Nearest to they: we, there, you, he, she, i, it, who,
    Nearest to he: she, it, they, who, there, never, eventually, molino,
    Nearest to from: in, into, across, through, after, priscilla, before, between,
    Nearest to united: kabuki, codenamed, retention, alison, euskal, wap, senate, confederate,
    Nearest to but: however, although, though, and, see, bentheim, or, than,
    Nearest to years: days, times, decades, year, weeks, eurofighter, months, centuries,
    Nearest to more: less, most, larger, longer, greater, very, rather, extremely,
    Nearest to no: any, pretense, little, epistemology, rarely, magdalena, redacted, a,
    Nearest to are: were, is, have, include, including, these, aerobatics, be,
    Nearest to also: often, now, still, sometimes, there, pappus, generally, usually,
    Nearest to american: australian, african, british, irish, fashions, english, italian, hdi,
    Nearest to may: can, would, will, must, could, might, should, cannot,
    Average loss at step 62000: 3.238788
    Average loss at step 64000: 3.263015
    Average loss at step 66000: 3.405191
    Average loss at step 68000: 3.388978
    Average loss at step 70000: 3.360408
    Nearest to an: belgrade, alveolar, rnberg, clowning, anointing, rosy, opens, carriers,
    Nearest to system: systems, exports, program, network, commission, counter, compulsory, bloodless,
    Nearest to been: become, be, already, was, were, remained, caretaker, sunset,
    Nearest to history: list, rond, aspects, kinder, caribbean, refrigerators, philosophy, section,
    Nearest to they: we, there, he, you, she, it, who, then,
    Nearest to he: she, it, they, there, who, never, molino, eventually,
    Nearest to from: through, across, into, between, by, in, priscilla, within,
    Nearest to united: kabuki, alison, retention, same, codenamed, euskal, senate, wap,
    Nearest to but: however, although, though, and, which, while, that, should,
    Nearest to years: days, decades, times, months, year, weeks, centuries, eurofighter,
    Nearest to more: less, most, larger, very, greater, extremely, rather, longer,
    Nearest to no: any, little, there, rarely, epistemology, pluriform, syndicalist, pretense,
    Nearest to are: were, is, including, include, have, these, although, those,
    Nearest to also: still, now, often, which, sometimes, not, there, never,
    Nearest to american: australian, african, british, italian, fashions, english, epoxy, indian,
    Nearest to may: can, will, would, could, must, might, should, cannot,
    Average loss at step 72000: 3.372776
    Average loss at step 74000: 3.348777
    Average loss at step 76000: 3.315423
    Average loss at step 78000: 3.347928
    Average loss at step 80000: 3.378293
    Nearest to an: belgrade, alveolar, elin, clowning, rosy, anointing, tdma, another,
    Nearest to system: systems, program, exports, counter, cycle, bledsoe, wojty, commission,
    Nearest to been: become, be, already, was, were, undertake, remained, sunset,
    Nearest to history: list, rond, refrigerators, kinder, regardless, historic, originator, jul,
    Nearest to they: we, he, there, you, she, it, not, who,
    Nearest to he: she, it, they, there, who, molino, never, eventually,
    Nearest to from: through, across, into, within, in, after, between, during,
    Nearest to united: kabuki, alison, retention, wap, steiner, codenamed, same, baroness,
    Nearest to but: however, although, than, while, though, and, until, see,
    Nearest to years: days, decades, times, months, weeks, year, minutes, centuries,
    Nearest to more: less, most, very, larger, extremely, quite, greater, longer,
    Nearest to no: any, little, pluriform, oratorios, funneled, exhorted, syndicalist, blaze,
    Nearest to are: were, have, is, include, although, including, infect, those,
    Nearest to also: still, now, often, which, sometimes, never, strongly, actually,
    Nearest to american: australian, british, african, italian, indian, fashions, irish, unary,
    Nearest to may: can, would, must, will, could, might, should, cannot,
    Average loss at step 82000: 3.404543
    Average loss at step 84000: 3.410009
    Average loss at step 86000: 3.387750
    Average loss at step 88000: 3.351753
    Average loss at step 90000: 3.366885
    Nearest to an: belgrade, another, alveolar, rnberg, anointing, clowning, electorate, nitrous,
    Nearest to system: systems, program, network, counter, wojty, exports, cycle, federation,
    Nearest to been: become, be, already, was, sunset, remained, were, caretaker,
    Nearest to history: list, rond, zu, refrigerators, scholar, book, solver, regardless,
    Nearest to they: we, he, there, she, you, it, then, but,
    Nearest to he: she, it, they, there, who, molino, never, then,
    Nearest to from: through, into, across, by, priscilla, during, after, under,
    Nearest to united: kabuki, confederate, steiner, senate, alison, same, retention, baroness,
    Nearest to but: however, and, until, although, while, though, which, furry,
    Nearest to years: days, decades, months, minutes, year, weeks, centuries, times,
    Nearest to more: less, most, very, larger, greater, rather, quite, longer,
    Nearest to no: any, little, pluriform, exhorted, tlatoani, syndicalist, oratorios, than,
    Nearest to are: were, include, is, have, these, including, although, reptile,
    Nearest to also: often, now, still, which, sometimes, never, actually, then,
    Nearest to american: british, african, australian, italian, fashions, indian, unary, european,
    Nearest to may: can, could, would, must, will, might, should, cannot,
    Average loss at step 92000: 3.396038
    Average loss at step 94000: 3.248840
    Average loss at step 96000: 3.354395
    Average loss at step 98000: 3.239452
    Average loss at step 100000: 3.354906
    Nearest to an: another, belgrade, rnberg, anointing, ferret, electorate, alveolar, clowning,
    Nearest to system: systems, program, wojty, counter, exports, bledsoe, network, process,
    Nearest to been: become, be, already, remained, was, sunset, were, cohorts,
    Nearest to history: list, rond, carcinogens, regardless, book, pawns, mpa, review,
    Nearest to they: we, there, he, you, she, it, i, not,
    Nearest to he: she, it, they, who, there, we, never, molino,
    Nearest to from: across, through, into, in, within, ibelin, priscilla, standby,
    Nearest to united: kabuki, confederate, steiner, senate, alison, retention, strait, same,
    Nearest to but: however, although, though, and, while, until, considers, isopropanol,
    Nearest to years: days, decades, months, year, weeks, minutes, times, centuries,
    Nearest to more: less, most, very, larger, quite, greater, longer, extremely,
    Nearest to no: any, little, tlatoani, exhorted, syndicalist, reclusive, oratorios, armour,
    Nearest to are: were, have, include, those, reptile, although, be, murderers,
    Nearest to also: now, never, still, often, sometimes, which, actually, then,
    Nearest to american: british, australian, italian, braintree, canadian, indian, estonia, crazed,
    Nearest to may: can, could, should, might, would, must, will, cannot,
    '''

    return final_embeddings





def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()


def quick_small_test():

    words=['I', 'want', 'to', 'go', 'to', 'school']
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print('Most common words (+UNK) with order', count)# ('word','appear frequence')
    print('Sample data', data)
    del words  # Hint to reduce memory.
    '''
    Most common words (+UNK) [['UNK', 0], ('to', 2), ('go', 1), ('want', 1), ('school', 1), ('I', 1)]
    Sample data [5, 3, 1, 2, 1, 4]
    '''


    # Let's display the internal variables to better understand their structure:
    print(count,type(count))
    print(data,type(data))
    # list to dic (word, frequence)
    print(list(dictionary.items()))
    # list to dic (frequence, word)
    print(list(reverse_dictionary.items()))
    '''
    [['UNK', 0], ('to', 2), ('go', 1), ('want', 1), ('school', 1), ('I', 1)] <class 'list'>
    [5, 3, 1, 2, 1, 4] <class 'list'>

    [('go', 2), ('UNK', 0), ('I', 5), ('school', 4), ('want', 3), ('to', 1)]
    [(0, 'UNK'), (1, 'to'), (2, 'go'), (3, 'want'), (4, 'school'), (5, 'I')]
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
    words = read_data(filename)
    print('words size %d' % len(words))
    #print('words content %s, type= %s' % (words[0:100],type(words)))



    '''
    words size 17005207
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

    #quick_small_test()

    # Build the dictionary and replace rare words with UNK token.
    data, count, dictionary, reverse_dictionary = build_dataset(words)
    print('Most common words (+UNK)', count[:10])
    print('Sample data', data[:10])
    del words  # Hint to reduce memory.
    '''
    Most common words (+UNK) [['UNK', 418391], ('the', 1061396), ('of', 593677), ('and', 416629),
    ('one', 411764), ('in', 372201), ('a', 325873), ('to', 316376), ('zero', 264975), ('nine', 250430)]
    Sample data [5241, 3084, 12, 6, 195, 2, 3134, 46, 59, 156]
    '''

    # Let's display the internal variables to better understand their structure:
    #print(count[:10])
    #print(data[:10])
    print(list(dictionary.items())[:10])
    print(list(reverse_dictionary.items())[:10])
    '''
    [('webelements', 16358), ('milled', 29933), ('metastasis', 46505), ('clogged', 43933),
    ('formidable', 12821), ('ibanez', 22239), ('capitalised', 30743), ('farad', 27812), ('movable', 17009), ('disenchanted', 34948)]

    [(0, 'UNK'), (1, 'the'), (2, 'of'), (3, 'and'), (4, 'one'), (5, 'in'), (6, 'a'), (7, 'to'), (8, 'zero'), (9, 'nine')]
    '''


    # Function to generate a training batch for the skip-gram model use later.
    print('data:', [reverse_dictionary[di] for di in data[:32]])
    '''
    data: [
    'anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse', 'first', 'used', 'against',
    'early', 'working', 'class', 'radicals', 'including', 'the', 'diggers', 'of', 'the', 'english',
    'revolution', 'and', 'the', 'sans', 'UNK', 'of', 'the', 'french', 'revolution', 'whilst', 'the', 'term']
    '''

    for num_skips, skip_window in [(2, 1), (4, 2)]:

        batch, labels = generate_batch(batch_size=16, num_skips=num_skips, skip_window=skip_window,data=data)
        print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
        print('    batch:', [reverse_dictionary[bi] for bi in batch])
        print('    labels:', [reverse_dictionary[li] for li in labels.reshape(16)])
        '''
        with num_skips = 2 and skip_window = 1:
            batch: ['originated', 'originated', 'as', 'as', 'a', 'a', 'term', 'term', 'of', 'of', 'abuse', 'abuse', 'first', 'first', 'used', 'used']
            labels: ['anarchism', 'as', 'originated', 'a', 'term', 'as', 'of', 'a', 'term', 'abuse', 'first', 'of', 'abuse', 'used', 'first', 'against']

        with num_skips = 4 and skip_window = 2:
            batch: ['radicals', 'radicals', 'radicals', 'radicals', 'including', 'including', 'including', 'including', 'the', 'the', 'the', 'the', 'diggers', 'diggers', 'diggers', 'diggers']
            labels: ['including', 'working', 'class', 'the', 'diggers', 'the', 'class', 'radicals', 'including', 'of', 'radicals', 'diggers', 'including', 'the', 'the', 'of']
        '''



    # It is not obvious with the output above, but all the data are based on index, and not the word directly.

    print(batch)# [10605 10605 10605 10605   134   134   134   134     1     1     1     1 27864 27864 27864 27864]
    print(labels)
    '''
    [[  134]
     [  742]
     [  477]
     [    1]
     [27864]
     [    1]
     [  477]
     [10605]
     [  134]
     [    2]
     [10605]
     [27864]
     [  134]
     [    1]
     [    1]
     [    2]]
     '''

    print("Data Ready!")




    #################
    # Model
    #################
    # Train a skip-gram model.
    final_embeddings=skip_gram_model(data,reverse_dictionary)




    #This is what an embedding looks like:
    print(final_embeddings[0])
    # All the values are abstract, there is practical meaning of the them. Moreover,
    # the final embeddings are normalized as you can see here:
    '''
    [ 0.09341599 -0.14042886 -0.07819284 -0.09250837 -0.01640968 -0.04263808
      0.09015562 -0.08980133 -0.0233397  -0.03180674  0.06889749 -0.01027224
      0.00900663 -0.04924826  0.18631902  0.1087643  -0.22626635 -0.06383247
     -0.07610254 -0.04067367 -0.06256497  0.07132345  0.139667    0.1359783
      0.01411107 -0.00656723  0.00712792 -0.03655377 -0.11955068  0.075135
      0.22917432  0.01695052 -0.05237205  0.18642162  0.01811995 -0.11519508
      0.01767583  0.0041279  -0.05856417 -0.08705174 -0.09126981 -0.04165866
     -0.07568732 -0.14409058 -0.01382122  0.1049132  -0.08867542  0.02178659
      0.06161967  0.02592648  0.10297526  0.03461177 -0.08078008  0.0752489
     -0.10834466 -0.0656555   0.04118867 -0.01098287  0.04960652 -0.08426549
     -0.04414825  0.18455376  0.02113118  0.1213237  -0.14905845  0.11343436
      0.06937777  0.03483068 -0.09095778  0.14806637 -0.03143674  0.07772982
     -0.0337519  -0.16569392 -0.03895564  0.04245298  0.04648942 -0.0064119
      0.08288375  0.00136594  0.11916817  0.068687    0.05781886  0.01035208
     -0.08542589 -0.16436793 -0.06840347 -0.01917173  0.15041684  0.08239542
     -0.06053361  0.01218658 -0.10909181 -0.023924   -0.08930953 -0.12447044
      0.02289473  0.01498208 -0.1435667   0.06104149  0.02957907  0.09265102
     -0.05235555 -0.08612821 -0.00586708 -0.13363071 -0.04364103  0.03458537
     -0.00955866 -0.06789755  0.04632098 -0.0386214  -0.10128922 -0.26140654
     -0.0193422   0.01608096  0.14071448 -0.03010085  0.01884446  0.01612899
     -0.00336174  0.10878568  0.17403337 -0.0299388   0.0268454  -0.03780577
     -0.05892913 -0.00211533]
      '''



    num_points = 400

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])

    # plot out
    words = [reverse_dictionary[i] for i in range(1, num_points+1)]
    plot(two_d_embeddings, words)


    return 0

main()



