from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import json
import time
import model
from dag import *

_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_DATA_FILES = 5

_WEIGHT_DECAY = 5e-4 #1e-4
_MOMENTUM = 0.9

_NUM_IMAGES = {
    'train': 45000,
    'valid': 5000,
    'test': 10000,
}

_TEST_BATCH_SIZE = 100

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--dataset', type=str, default='cifar10',
                    help='CIFAR-10, CIFAR-100.')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model',
                    help='The directory where the model will be stored.')

parser.add_argument('--num_nodes', type=int, default=5,
                    help='The number of nodes in a cell.')

parser.add_argument('--N', type=int, default=6,
                    help='The number of stacked convolution cell.')

parser.add_argument('--filters', type=int, default=36,
                    help='The numer of filters.')

parser.add_argument('--drop_path_keep_prob', type=float, default=0.6,
                    help='Dropout rate.')

parser.add_argument('--dense_dropout_keep_prob', type=float, default=1.0,
                    help='Dropout rate.')

parser.add_argument('--stem_multiplier', type=float, default=3.0,
                    help='Stem convolution multiplier. Default is 3.0 for CIFAR-10. 1.0 is for ImageNet.')

parser.add_argument('--train_epochs', type=int, default=600,
                    help='The number of epochs to train.')

parser.add_argument('--epochs_per_eval', type=int, default=10,
                    help='The number of epochs to run in between evaluations.')

parser.add_argument('--eval_after', type=int, default=0,
                    help='The number of epochs to run before evaluations.')

parser.add_argument('--batch_size', type=int, default=128,
                    help='The number of images per batch.')

parser.add_argument('--arch', type=str, default=None,
                    help='Default architecture to run.')

parser.add_argument('--hparams', type=str, default=None,
                    help='hparams file. All the params will be overrided by this file.')

parser.add_argument('--split_train_valid', action='store_true', default=False,
                    help='Split training data to train set and valid set.')

parser.add_argument('--use_nesterov', action='store_true', default=False,
                    help='Use nesterov in Momentum Optimizer.')

parser.add_argument('--use_aux_head', action='store_true', default=False,
                    help='Use auxillary head.')

parser.add_argument('--aux_head_weight', type=float, default=0.4,
                    help='Weight of auxillary head loss.')

parser.add_argument('--weight_decay', type=float, default=_WEIGHT_DECAY,
                    help='Weight decay.')

parser.add_argument('--cutout_size', type=int, default=None,
                    help='Size of cutout. Default to None, means no cutout.')

parser.add_argument('--num_gpus', type=int, default=1,
                    help='Number of GPU to use.')

parser.add_argument('--seed', type=int, default=None,
                    help='Seed to use.')

parser.add_argument(
    '--data_format', type=str, default=None,
    choices=['channels_first', 'channels_last'],
    help='A flag to override the data format used in the model. channels_first '
         'provides a performance boost on GPU but is not always compatible '
         'with CPU. If left unspecified, the data format will be chosen '
         'automatically based on whether TensorFlow was built for CPU or GPU.')


def record_dataset(filenames, dataset, mode):
  """Returns an input pipeline Dataset from `filenames`."""
  if dataset == 'cifar10':
    record_bytes = _HEIGHT * _WIDTH * _DEPTH + 1
    return tf.data.FixedLengthRecordDataset(filenames, record_bytes)
  elif dataset == 'cifar100':
    record_bytes = _HEIGHT * _WIDTH * _DEPTH + 2
    return tf.data.FixedLengthRecordDataset(filenames, record_bytes)
  
  
def get_filenames(split, mode, data_dir, dataset):
  """Returns a list of filenames."""
  if dataset == 'cifar10':
    if not split:
      assert os.path.exists(os.path.join(data_dir+'data_batch_1.bin')), (
        'Download and extract the '
        'CIFAR-10 data.')
    else:
      assert os.path.exists(os.path.join(data_dir+'valid_batch.bin')), (
        'Download and extract the '
        'CIFAR-10 data and split out valid data')

    if split:
      if mode == 'train':
        return [
          os.path.join(data_dir, 'train_batch_%d.bin' % i)
          for i in range(1, _NUM_DATA_FILES + 1)]
      elif mode == 'valid':
        return [os.path.join(data_dir, 'valid_batch.bin')]
      else:
        return [os.path.join(data_dir, 'test_batch.bin')]
    else:
      if mode == 'train':
        return [
          os.path.join(data_dir, 'data_batch_%d.bin' % i)
          for i in range(1, _NUM_DATA_FILES + 1)
    ]
      else:
        return [os.path.join(data_dir, 'test_batch.bin')]
  
  elif dataset == 'cifar100':
    assert os.path.exists(data_dir)

    if mode == 'train':
      return [os.path.join(data_dir, 'train.bin')]
    else:
      return [os.path.join(data_dir, 'test.bin')]


def parse_record(raw_record, dataset):
  #Parse CIFAR image and label from a raw record.
  # Every record consists of a label followed by the image, with a fixed number
  # of bytes for each.
  if dataset == 'cifar10':
    label_bytes = 1
  elif dataset == 'cifar100':
    label_bytes = 2
  image_bytes = _HEIGHT * _WIDTH * _DEPTH
  record_bytes = label_bytes + image_bytes
  
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)
  
  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  if dataset == 'cifar10':
    label = tf.cast(record_vector[0], tf.int32)
    label = tf.one_hot(label, 10)
  elif dataset == 'cifar100':
    label = tf.cast(record_vector[1], tf.int32)
    label = tf.one_hot(label, 100)
  
  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(
      record_vector[label_bytes:record_bytes], [_DEPTH, _HEIGHT, _WIDTH])
  
  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)
  
  return image, label
  
  
def preprocess_image(image, mode, cutout_size):
  """Preprocess a single image of layout [height, width, depth]."""
  if mode == 'train':
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)

  if mode == 'train' and cutout_size is not None:
    mask = tf.ones([cutout_size, cutout_size], dtype=tf.int32)
    start = tf.random_uniform([2], minval=0, maxval=32, dtype=tf.int32)
    mask = tf.pad(mask, [[cutout_size + start[0], 32 - start[0]],
                        [cutout_size + start[1], 32 - start[1]]])
    mask = mask[cutout_size: cutout_size + 32,
                cutout_size: cutout_size + 32]
    mask = tf.reshape(mask, [32, 32, 1])
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(tf.equal(mask, 0), x=image, y=tf.zeros_like(image))
  return image


def input_fn(split, mode, data_dir, dataset, batch_size, cutout_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    mode: train, valid or test.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  data_set = record_dataset(get_filenames(split, mode, data_dir, dataset), dataset, mode)

  if mode == 'train':
    if split:
      dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
    else:
      dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'] + _NUM_IMAGES['valid'])

  data_set = data_set.map(lambda x:parse_record(x, dataset), num_parallel_calls=4)
  data_set = data_set.map(
      lambda image, label: (preprocess_image(image, mode, cutout_size), label),
      num_parallel_calls=4)

  data_set = data_set.repeat(num_epochs)
  data_set = data_set.batch(batch_size)
  data_set = data_set.prefetch(10)
  iterator = data_set.make_one_shot_iterator()
  images, labels = iterator.get_next()

  return images, labels


def _log_variable_sizes(var_list, tag):
  """Log the sizes and shapes of variables, and the total size.

    Args:
      var_list: a list of varaibles
      tag: a string
  """
  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = int(np.prod(np.array(v.shape.as_list())))
    tf.logging.info("Weight    %s\tshape    %s\tsize    %d",
      v.name[:-2].ljust(80),
      str(v.shape).ljust(20), v_size)
    total_size += v_size
  tf.logging.info("%s Total size: %d", tag, total_size)


def get_test_ops(x, y, params, reuse=False):
  with tf.device('/gpu:0'):
    inputs = tf.reshape(x, [-1, _HEIGHT, _WIDTH, _DEPTH])
    labels = y
    res = model.build_model(inputs, params, False, reuse)
    logits = res['logits']
    cross_entropy = tf.losses.softmax_cross_entropy(
      logits=logits, onehot_labels=labels)
    # Add weight decay to the loss.
    loss = cross_entropy + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if 'aux_logits' in res:
      aux_logits = res['aux_logits']
      aux_loss = tf.losses.softmax_cross_entropy(
        logits=aux_logits, onehot_labels=labels, weights=params['aux_head_weight'])
      loss += aux_loss

    predictions = tf.argmax(logits, axis=1)
    labels = tf.argmax(y, axis=1)
    test_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))
    return loss, test_accuracy


def test(params):
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    tf.set_random_seed(params['seed'])
    x_test, y_test = input_fn(False, 'test', params['data_dir'], params['dataset'], 100, None, None)
    _log_variable_sizes(tf.trainable_variables(), 'Trainable Variables')
    test_loss, test_accuracy = get_test_ops(x_test, y_test, params, tf.AUTO_REUSE)
    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, checkpoint_dir=params['model_dir']) as sess:
      test_ops = [
        test_loss, test_accuracy
      ]
      test_start_time = time.time()
      test_loss_list = []
      test_accuracy_list = []
      for _ in range(_NUM_IMAGES['test'] // 100):
        test_loss_v, test_accuracy_v = sess.run(test_ops)
        test_loss_list.append(test_loss_v)
        test_accuracy_list.append(test_accuracy_v)
      test_time = time.time() - test_start_time
      log_string =  "Evaluation on test data\n"
      log_string += "loss={:<6f} ".format(np.mean(test_loss_list))
      log_string += "test_accuracy={:<8.6f} ".format(np.mean(test_accuracy_list))
      log_string += "secs={:<10.2f}".format((test_time))
      tf.logging.info(log_string)

def build_dag(dag_name_or_path):
  try:
    conv_dag, reduc_dag = eval(dag_name_or_path)()
  except:
    try:
      with open(os.path.join(dag_name_or_path), 'r') as f:
        content = json.load(f)
        conv_dag, reduc_dag = content['conv_dag'], content['reduc_dag']
    except:
      conv_dag, reduc_dag = None, None

  return conv_dag, reduc_dag

def get_params():
  conv_dag, reduc_dag = build_dag(FLAGS.arch)
  if FLAGS.split_train_valid:
    total_steps = int(FLAGS.train_epochs * _NUM_IMAGES['train'] / float(FLAGS.batch_size))
  else:
    total_steps = int(FLAGS.train_epochs * (_NUM_IMAGES['train'] + _NUM_IMAGES['valid']) / float(FLAGS.batch_size))

  params = vars(FLAGS)
  if params['dataset'] == 'cifar10':
    params['num_classes'] = 10
  elif params['dataset'] == 'cifar100':
    params['num_classes'] = 100

  params['conv_dag'] = conv_dag
  params['reduc_dag'] = reduc_dag
  params['total_steps'] = total_steps
  
  if FLAGS.hparams is not None:
    with open(os.path.join(FLAGS.hparams), 'r') as f:
      hparams = json.load(f)
      params.update(hparams)

  if params['conv_dag'] is None or params['reduc_dag'] is None:
    raise ValueError('You muse specify a registered model name or provide a model in the hparams.')

  return params 


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  params = get_params()
  test(params)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
