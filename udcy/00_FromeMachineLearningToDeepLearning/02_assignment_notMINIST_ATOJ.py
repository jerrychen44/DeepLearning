
# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import sys
import tarfile
#from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle
# add my own
from PIL import Image
import random

filepath=os.path.dirname(os.path.realpath(__file__))
source_path=filepath+'/source'
'''
First, we'll download the dataset to our local machine.
The data consists of characters rendered in a variety of fonts on a 28x28 image.
The labels are limited to 'A' through 'J' (10 classes).
The training set has about 500k and the testset 19000 labelled examples.
Given these sizes, it should be possible to train models quickly on any machine.
'''
url = 'http://commondatastorage.googleapis.com/books1000/'
last_percent_reported = None

def download_progress_hook(count, blockSize, totalSize):
  """A hook to report the progress of a download. This is mostly intended for users with
  slow internet connections. Reports every 1% change in download progress.
  """
  global last_percent_reported
  percent = int(count * blockSize * 100 / totalSize)

  if last_percent_reported != percent:
    if percent % 5 == 0:
      sys.stdout.write("%s%%" % percent)
      sys.stdout.flush()
    else:
      sys.stdout.write(".")
      sys.stdout.flush()

    last_percent_reported = percent

def maybe_download(filename, expected_bytes, force=False):
  """Download a file if not present, and make sure it's the right size."""
  print(filename)

  if force or not os.path.exists(filename):
    print('Attempting to download:', filename)
    filename, _ = urlretrieve(url + filename.split('/')[-1], filename, reporthook=download_progress_hook)
    print('\n Download Complete!',filename)

  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

num_classes = 10
np.random.seed(133)

def maybe_extract(filename, force=False):
  root = os.path.splitext(os.path.splitext(filename)[0])[0]  # remove .tar.gz
  print(root)

  if os.path.isdir(root) and not force:
    # You may override by setting force=True.
    print('%s already present - Skipping extraction of %s.' % (root, filename))
  else:
    print('Extracting data for %s. This may take a while. Please wait.' % root)
    print(filename)
    tar = tarfile.open(filename)
    sys.stdout.flush()
    tar.extractall(source_path)
    tar.close()

  print(os.listdir(root))


  data_folders = [
    os.path.join(root, d) for d in sorted(os.listdir(root))
    if os.path.isdir(os.path.join(root, d))]

  if len(data_folders) != num_classes:
    raise Exception(
      'Expected %d folders, one per class. Found %d instead.' % (
        num_classes, len(data_folders)))

  print(data_folders)

  return data_folders




'''
Problem 2: Verify Normalized Images

Now let's load the data in a more manageable format.
Since, depending on your computer setup you might not be able to fit it all in memory,
we'll load each class into a separate dataset, store them on disk and curate them independently.
Later we'll merge them into a single dataset of manageable size.
We'll convert the entire dataset into a 3D array (image index, x, y) of
floating point values, normalized to have approximately zero mean and standard
deviation ~0.5 to make training easier down the road.
A few images might not be readable, we'll just skip them.
'''
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      '''
      scipy.ndimage.imread
      The different colour bands/channels are stored in the third dimension,
      such that a grey-image is MxN, an RGB-image MxNx3 and an RGBA-image MxNx4.
      We are using the Gray image now, so image_data.shpae is 28x28 and
      original value is 0~255, but we are standardlize here
      '''
      image_data = (ndimage.imread(image_file).astype(float) - pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except IOError as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')

  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))

  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset

def maybe_pickle(data_folders, min_num_images_per_class, force=False):
  dataset_names = []
  for folder in data_folders:
    set_filename = folder + '.pickle'
    dataset_names.append(set_filename)
    if os.path.exists(set_filename) and not force:
      # You may override by setting force=True.
      print('%s already present - Skipping pickling.' % set_filename)
    else:
      print('Pickling %s.' % set_filename)
      dataset = load_letter(folder, min_num_images_per_class)
      try:
        with open(set_filename, 'wb') as f:
          pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
      except Exception as e:
        print('Unable to save data to', set_filename, ':', e)

  return dataset_names

def verify_normalized_images(train_datasets):

    pickle_file = train_datasets[0]  # index 0 should be all As, 1 = all Bs, etc.
    with open(pickle_file, 'rb') as f:
        #print(f)
        letter_set = pickle.load(f)  # unpickle
        sample_idx = np.random.randint(len(letter_set))  # pick a random image index
        print(len(letter_set))
        sample_image = letter_set[sample_idx, :, :]  # extract a 2D slice
        print(sample_image)
        plt.figure()
        plt.imshow(sample_image)  # display it with pixel array
        plt.show()

    return 0
'''
Problem 3: we expect the data to be balanced across classes. Verify that.

Merge and prune the training data as needed. Depending on your computer setup,
you might not be able to fit it all in memory, and you can tune train_size as needed.
The labels will be stored into a separate array of integers 0 through 9.
Also create a validation dataset for hyperparameter tuning.
'''
def make_arrays(nb_rows, img_size):
  if nb_rows:
    dataset = np.ndarray((nb_rows, img_size, img_size), dtype=np.float32)
    labels = np.ndarray(nb_rows, dtype=np.int32)
  else:
    dataset, labels = None, None
  return dataset, labels

def merge_datasets(pickle_files, train_size, valid_size=0):
  num_classes = len(pickle_files)
  valid_dataset, valid_labels = make_arrays(valid_size, image_size)
  train_dataset, train_labels = make_arrays(train_size, image_size)
  vsize_per_class = valid_size // num_classes
  tsize_per_class = train_size // num_classes

  start_v, start_t = 0, 0
  end_v, end_t = vsize_per_class, tsize_per_class
  end_l = vsize_per_class+tsize_per_class
  for label, pickle_file in enumerate(pickle_files):
    try:
      with open(pickle_file, 'rb') as f:
        letter_set = pickle.load(f)
        # let's shuffle the letters to have random validation and training set
        np.random.shuffle(letter_set)
        if valid_dataset is not None:
          valid_letter = letter_set[:vsize_per_class, :, :]
          valid_dataset[start_v:end_v, :, :] = valid_letter
          valid_labels[start_v:end_v] = label
          start_v += vsize_per_class
          end_v += vsize_per_class

        train_letter = letter_set[vsize_per_class:end_l, :, :]
        train_dataset[start_t:end_t, :, :] = train_letter
        train_labels[start_t:end_t] = label
        start_t += tsize_per_class
        end_t += tsize_per_class
    except Exception as e:
      print('Unable to process data from', pickle_file, ':', e)
      raise

  return valid_dataset, valid_labels, train_dataset, train_labels

def verify_data_balanced(train_datasets,test_datasets):
    train_size = 200000
    valid_size = 10000
    test_size = 10000

    valid_dataset, valid_labels, train_dataset, train_labels = merge_datasets(
      train_datasets, train_size, valid_size)
    _, _, test_dataset, test_labels = merge_datasets(test_datasets, test_size)

    print('Training:', train_dataset.shape, train_labels.shape)
    print('Validation:', valid_dataset.shape, valid_labels.shape)
    print('Testing:', test_dataset.shape, test_labels.shape)
    '''
    Training: (200000, 28, 28) (200000,)
    Validation: (10000, 28, 28) (10000,)
    Testing: (10000, 28, 28) (10000,)
    '''
    return train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels

def disp_number_images(data_folders):
  for folder in data_folders:
    pickle_filename = ''.join(folder) + '.pickle'
    try:
      with open(pickle_filename, 'rb') as f:
        dataset = pickle.load(f)
    except Exception as e:
      print('Unable to read data from', pickle_filename, ':', e)
      return
    print('Number of images in ', folder, ' : ', len(dataset))
    '''
    Number of images in  ../notMNIST_large/A  :  52909
    Number of images in  ../notMNIST_large/B  :  52911
    Number of images in  ../notMNIST_large/C  :  52912
    Number of images in  ../notMNIST_large/D  :  52911
    Number of images in  ../notMNIST_large/E  :  52912
    Number of images in  ../notMNIST_large/F  :  52912
    Number of images in  ../notMNIST_large/G  :  52912
    Number of images in  ../notMNIST_large/H  :  52912
    Number of images in  ../notMNIST_large/I  :  52912
    Number of images in  ../notMNIST_large/J  :  52911
    Number of images in  ../notMNIST_small/A  :  1872
    Number of images in  ../notMNIST_small/B  :  1873
    Number of images in  ../notMNIST_small/C  :  1873
    Number of images in  ../notMNIST_small/D  :  1873
    Number of images in  ../notMNIST_small/E  :  1873
    Number of images in  ../notMNIST_small/F  :  1872
    Number of images in  ../notMNIST_small/G  :  1872
    Number of images in  ../notMNIST_small/H  :  1872
    Number of images in  ../notMNIST_small/I  :  1872
    Number of images in  ../notMNIST_small/J  :  1872
    '''
  return 0



def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


######################
# Problem 4
# Convince yourself that the data is still good after shuffling!
#####################

pretty_labels = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'}

def disp_sample_dataset(dataset, labels):
    items = random.sample(range(len(labels)), 8)
    for i, item in enumerate(items):
        plt.subplot(2, 4, i+1)
        plt.axis('off')
        plt.title(pretty_labels[labels[item]])
        plt.imshow(dataset[item])
    plt.show()
    return 0


def data_ready_save_to_pickle(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels):
    pickle_file = source_path+'/notMNIST.pickle'

    try:
      f = open(pickle_file, 'wb')
      save = {
        'train_dataset': train_dataset,
        'train_labels': train_labels,
        'valid_dataset': valid_dataset,
        'valid_labels': valid_labels,
        'test_dataset': test_dataset,
        'test_labels': test_labels,
        }
      pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
      f.close()
    except Exception as e:
      print('Unable to save data to', pickle_file, ':', e)
      raise

    statinfo = os.stat(pickle_file)
    print('Compressed pickle size:', statinfo.st_size)
    ''' Compressed pickle size: 690800503'''
    return 0

###################
# Problem 6: Train A Simple ML Model
# Train a simple model on this data using 50, 100, 1000 and 5000 training samples.
###################
def quick_logisticregression(sample_size,train_dataset,train_labels,test_dataset,test_labels):
    print("quick logisticregression with size",sample_size)
    regr = LogisticRegression()

    # reshape for sklearn 2dim input
    #need change train_dataset dim from 3D(20000,28,28) to 2D (20000,28*28 as features by each pixel)
    X_train = train_dataset[:sample_size].reshape(sample_size, 784)# 28*28 =784
    y_train = train_labels[:sample_size]

    X_test = test_dataset.reshape(test_dataset.shape[0], 28 * 28)
    y_test = test_labels

    regr.fit(X_train, y_train)
    print(regr.score(X_test, y_test))

    pred_labels = regr.predict(X_test)
    disp_sample_dataset(test_dataset, pred_labels)
    return 0

#################################################
def main():
    ######################
    # Get Data READY
    #######################
    # get data
    train_filename = maybe_download(source_path+'/notMNIST_large.tar.gz', 247336696)
    test_filename = maybe_download(source_path+'/notMNIST_small.tar.gz', 8458043)
    #test_filename = maybe_download('notMNIST_small.tar.gz', 8458043)
    # Extract the dataset from the compressed .tar.gz file. This should give you a set of directories, labelled A through J.
    train_folders = maybe_extract(train_filename)
    ''' ['notMNIST_large/A', 'notMNIST_large/B', 'notMNIST_large/C', 'notMNIST_large/D',
    'notMNIST_large/E', 'notMNIST_large/F', 'notMNIST_large/G', 'notMNIST_large/H',
    'notMNIST_large/I', 'notMNIST_large/J'] '''
    test_folders = maybe_extract(test_filename)
    # take a look the png, or you can browser in folder.
    #image = Image.open(source_path+"/notMNIST_small/A/Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png")
    #image.show()

    # A few images might not be readable, we'll just skip them.
    train_datasets = maybe_pickle(train_folders, 45000)
    test_datasets = maybe_pickle(test_folders, 1800)




    #######################
    # Problem 2: Verify Normalized Images
    ######################
    #verify_normalized_images(train_datasets)

    ######################
    # Problem 3: we expect the data to be balanced across classes. Verify that.
    ######################
    train_dataset, train_labels, test_dataset, test_labels, valid_dataset, valid_labels=verify_data_balanced(train_datasets,test_datasets)
    #disp_number_images(train_folders)
    #disp_number_images(test_folders)


    # Next, we'll randomize the data.
    # It's important to have the labels well shuffled for the training
    # and test distributions to match.
    train_dataset, train_labels = randomize(train_dataset, train_labels)
    test_dataset, test_labels = randomize(test_dataset, test_labels)
    valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)


    ######################
    # Problem 4
    # Convince yourself that the data is still good after shuffling!
    #####################

    #Finally, let's save the data for later reuse:
    disp_sample_dataset(train_dataset, train_labels)
    disp_sample_dataset(valid_dataset, valid_labels)
    disp_sample_dataset(test_dataset, test_labels)
    #data_ready_save_to_pickle(train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels)


    #################
    # Problem 5: make sure the data will not duplicate in training set and testing set.
    #################
    #SKIP HERE.

    ###################
    # Problem 6: Train A Simple ML Model
    # Train a simple model on this data using 50, 100, 1000 and 5000 training samples.
    ###################
    quick_logisticregression(50,train_dataset,train_labels,test_dataset,test_labels)
    quick_logisticregression(100,train_dataset,train_labels,test_dataset,test_labels)
    quick_logisticregression(1000,train_dataset,train_labels,test_dataset,test_labels)
    quick_logisticregression(5000,train_dataset,train_labels,test_dataset,test_labels)
    '''
    quick logisticregression with size 50
    0.509
    quick logisticregression with size 100
    0.6966
    quick logisticregression with size 1000
    0.8333
    quick logisticregression with size 5000
    0.8511
    '''
    return 0

main()
