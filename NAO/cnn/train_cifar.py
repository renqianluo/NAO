from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
import json
import math
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
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'],
                    help='Train, or test.')

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

parser.add_argument('--lr_schedule', type=str, default='cosine',
                    choices=['cosine', 'constant', 'decay'],
                    help='Learning rate schedule schema.')

parser.add_argument('--lr', type=float, default='0.1',
                    help='Learning rate when learning rate schedule is constant.')

parser.add_argument('--lr_max', type=float, default=0.025,  #0.05 in ENAS
                    help='Max learning rate.')

parser.add_argument('--lr_min', type=float, default=0.0, #0.001 in ENAS
                    help='Min learning rate.')

parser.add_argument('--T_0', type=int, default=10,
                    help='Epochs of the first cycle.')

parser.add_argument('--T_mul', type=int, default=2,
                    help='Multiplicator for the cycle.')

parser.add_argument('--clip', type=float, default=5.0,
                    help='Clip gradients according to global norm.')


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
      data_set = data_set.shuffle(buffer_size=_NUM_IMAGES['train'])
    else:
      data_set = data_set.shuffle(buffer_size=_NUM_IMAGES['train'] + _NUM_IMAGES['valid'])

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


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def get_train_ops(x, y, params, reuse=False):
  global_step = tf.train.get_or_create_global_step()

  num_images = _NUM_IMAGES['train'] if params['split_train_valid'] else _NUM_IMAGES['train'] + _NUM_IMAGES['valid']

  if params['lr_schedule'] == 'cosine':
    lr_max = params['lr_max']
    lr_min = params['lr_min']
    T_0 = params['T_0']
    T_mul = params['T_mul']
    batches_per_epoch = math.ceil(num_images / params['batch_size'])
    params['batches_per_epoch'] = batches_per_epoch

    curr_epoch = tf.cast(global_step // batches_per_epoch, tf.int32)

    last_reset = tf.get_variable("last_reset", initializer=0, dtype=tf.int32, trainable=False)
    T_i = tf.get_variable("T_i", initializer=T_0, dtype=tf.int32, trainable=False)
    T_curr = curr_epoch - last_reset

    def _update():
      update_last_reset = tf.assign(last_reset, curr_epoch, use_locking=True)
      update_T_i = tf.assign(T_i, T_i * T_mul, use_locking=True)
      with tf.control_dependencies([update_last_reset, update_T_i]):
        rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
      return lr

    def _no_update():
      rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
      lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
      return lr

    learning_rate = tf.cond(
      tf.greater_equal(T_curr, T_i), _update, _no_update)
  elif params['lr_schedule'] == 'decay':
    batches_per_epoch = num_images / params['batch_size']
    boundaries = [int(batches_per_epoch * epoch) for epoch in [100, 200, 300]]
    values = [params['lr'] * decay for decay in [1, 0.1, 0.01, 0.001]]
    learning_rate = tf.train.piecewise_constant(
      tf.cast(global_step, tf.int32), boundaries, values)
  else:
    learning_rate = params['lr']

  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = tf.train.MomentumOptimizer(
      learning_rate=learning_rate,
      momentum=_MOMENTUM,
      use_nesterov=params['use_nesterov'])

  inputs = tf.reshape(x, [-1, _HEIGHT, _WIDTH, _DEPTH])
  labels = y
  inputs_sharded = tf.split(inputs, params['num_gpus'], axis=0)
  labels_sharded = tf.split(labels, params['num_gpus'], axis=0)
  loss_sharded = []
  tower_grads = []
  train_accuracy_sharded = []
  for i in range(params['num_gpus']):
    with tf.device('/gpu:%d'%i):
      with tf.name_scope('shard_%d'%i):
        res = model.build_model(inputs_sharded[i], params, True, reuse if i==0 else True)
        logits = res['logits']
        cross_entropy = tf.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels_sharded[i])
        if 'aux_logits' in res:
          aux_logits = res['aux_logits']
          aux_loss = tf.losses.softmax_cross_entropy(
            logits=aux_logits, onehot_labels=labels_sharded[i], weights=params['aux_head_weight'])
        loss = cross_entropy + aux_loss
        loss = loss + params['weight_decay'] * tf.add_n(
          [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        grads = optimizer.compute_gradients(loss)
        tower_grads.append(grads)
        loss_sharded.append(loss)
        predictions = tf.argmax(logits, axis=1)
        train_accuracy = tf.reduce_mean(
          tf.cast(
            tf.equal(predictions, tf.argmax(labels_sharded[i], axis=1)), 
            dtype=tf.float32))
        train_accuracy_sharded.append(train_accuracy)

  loss = tf.reduce_mean(loss_sharded, axis=0)
  tf.summary.scalar('training_loss', loss)
  train_accuracy = tf.reduce_mean(train_accuracy_sharded, axis=0)
  tf.summary.scalar('train_accuracy', train_accuracy)
  grads = average_gradients(tower_grads)
  # Batch norm requires update ops to be added as a dependency to the train_op
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  with tf.control_dependencies(update_ops):
    #gradients, variables = zip(*optimizer.compute_gradients(loss))
    #gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    #train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)
    gradients, variables = zip(*grads)
    gradients, _ = tf.clip_by_global_norm(gradients, params['clip'])
    train_op = optimizer.apply_gradients(zip(gradients, variables), global_step)
  
  return loss, learning_rate, train_accuracy, train_op, global_step

def get_valid_ops(x, y, params, reuse=False):
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
    valid_accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, labels), dtype=tf.float32))
    return loss, valid_accuracy

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

def train(params):
  g = tf.Graph()
  with g.as_default(), tf.device('/cpu:0'):
    tf.set_random_seed(params['seed'])
    x_train, y_train = input_fn(params['split_train_valid'], 'train', params['data_dir'], params['dataset'], params['batch_size'], params['cutout_size'], None)
    if params['split_train_valid']:
      x_valid, y_valid = input_fn(params['split_train_valid'], 'valid', params['data_dir'], params['dataset'], 100, None, None)
    else:
      x_valid, y_valid = None, None
    x_test, y_test = input_fn(False, 'test', params['data_dir'], params['dataset'], 100, None, None)
    train_loss, learning_rate, train_accuracy, train_op, global_step = get_train_ops(x_train, y_train, params)
    _log_variable_sizes(tf.trainable_variables(), 'Trainable Variables')
    if params['split_train_valid']:
      valid_loss, valid_accuracy = get_valid_ops(x_valid, y_valid, params, True)
    test_loss, test_accuracy = get_test_ops(x_test, y_test, params, True)
    saver = tf.train.Saver(max_to_keep=1000)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      params['model_dir'], save_steps=params['batches_per_epoch'], saver=saver)
    hooks = [checkpoint_saver_hook]
    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, hooks=hooks, checkpoint_dir=params['model_dir']) as sess:
      start_time = time.time()
      while True:
        run_ops = [
          train_loss,
          learning_rate,
          train_accuracy,
          train_op,
          global_step
        ]
        train_loss_v, learning_rate_v, train_accuracy_v, _, global_step_v = sess.run(run_ops)

        epoch = global_step_v // params['batches_per_epoch'] 
        curr_time = time.time()
        if global_step_v % 100 == 0:
          log_string = "epoch={:<6d} ".format(epoch)
          log_string += "step={:<6d} ".format(global_step_v)
          log_string += "loss={:<6f} ".format(train_loss_v)
          log_string += "learning_rate={:<8.4f} ".format(learning_rate_v)
          log_string += "training_accuracy={:<8.4f} ".format(train_accuracy_v)
          log_string += "mins={:<10.2f}".format((curr_time - start_time) / 60)
          tf.logging.info(log_string)
        if global_step_v % params['batches_per_epoch'] == 0:
          if params['split_train_valid']:
            valid_ops = [
              valid_loss, valid_accuracy
            ]
            valid_start_time = time.time()
            valid_loss_list = []
            valid_accuracy_list = []
            for _ in range(_NUM_IMAGES['valid'] // 100):
              valid_loss_v, valid_accuracy_v = sess.run(valid_ops)
              valid_loss_list.append(valid_loss_v)
              valid_accuracy_list.append(valid_accuracy_v)
            valid_time = time.time() - valid_start_time
            log_string =  "Evaluation on valid data\n"
            log_string += "epoch={:<6d} ".format(epoch)
            log_string += "step={:<6d} ".format(global_step_v)
            log_string += "loss={:<6f} ".format(np.mean(valid_loss_list))
            log_string += "learning_rate={:<8.6f} ".format(learning_rate_v)
            log_string += "valid_accuracy={:<8.6f} ".format(np.mean(valid_accuracy_list))
            log_string += "secs={:<10.2f}".format((valid_time))
            tf.logging.info(log_string)
          
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
          log_string += "epoch={:<6d} ".format(epoch)
          log_string += "step={:<6d} ".format(global_step_v)
          log_string += "loss={:<6f} ".format(np.mean(test_loss_list))
          log_string += "learning_rate={:<8.6f} ".format(learning_rate_v)
          log_string += "test_accuracy={:<8.6f} ".format(np.mean(test_accuracy_list))
          log_string += "secs={:<10.2f}".format((test_time))
          tf.logging.info(log_string)
        if epoch >= params['train_epochs']:
          break


def build_dag(dag_name_or_path):
  try:
    conv_dag, reduc_dag = eval('dag.{}()'.format(dag_name_or_path))
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

  if FLAGS.mode == 'train':
    params = get_params()
    with open(os.path.join(params['model_dir'], 'hparams.json'), 'w') as f:
      json.dump(params, f)
    train(params)

  elif FLAGS.mode == 'test':
    pass


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
