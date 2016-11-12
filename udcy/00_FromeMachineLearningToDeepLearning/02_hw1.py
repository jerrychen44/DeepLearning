
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
from PIL import Image


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
      image_data = (ndimage.imread(image_file).astype(float) -
                    pixel_depth / 2) / pixel_depth
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
    verify_normalized_images(train_datasets)








    return 0
#img=ndimage.imread(source_path+"/notMNIST_small/A/Q0NXaWxkV29yZHMtQm9sZEl0YWxpYy50dGY=.png")
img=ndimage.imread(source_path+"/123.png")
print(type(img))
print(img.shape)

#main()
