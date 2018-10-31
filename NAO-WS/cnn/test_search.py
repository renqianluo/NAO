from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import utils
import tensorflow as tf
import json
import model_search_small as model


_NUM_IMAGES = {
    'train': 45000,
    'valid': 5000,
    'test': 10000,
}

_TEST_BATCH_SIZE = 500

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data',
                    help='The path to the CIFAR-10 data directory.')

parser.add_argument('--test_set', type=str, default='valid',
                    choices=['valid', 'test', 'both'],
                    help='Data to eval.')

parser.add_argument('--model_dir', type=str, default='models',
                    help='The directory where the model will be stored.')

parser.add_argument('--test_batch_size', type=int, default=_TEST_BATCH_SIZE)

parser.add_argument('--dag', type=str, default=None)


def test(params):
  g = tf.Graph()
  if params['test_set'] in ['valid', 'both']:
    with g.as_default():
      x_test, y_test = model.input_fn('valid', params['data_dir'], params['test_batch_size'], None, None, False)
    model.test(g, x_test, y_test, _NUM_IMAGES['valid']//params['test_batch_size'], params)
  if params['test_set'] in ['test', 'both']:
    with g.as_default():
      x_test, y_test = model.input_fn('test', params['data_dir'], params['test_batch_size'], None, None, False)
    model.test(g, x_test, y_test, _NUM_IMAGES['test']//params['test_batch_size'], params)
    
def get_params():
  if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
    raise  ValueError('model dir does not exist!')
  with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'rb') as f:
      hparams = json.load(f, encoding='utf-8')
  with open(FLAGS.dag, 'r') as f:
    archs = f.read().splitlines()
    archs = list(map(utils.build_dag, archs))
  params = {}
  for k,v in hparams.items():
    if k.startswith('child_'):
      params[k[6:]] = v
  params.update({
    'model_dir': os.path.join(FLAGS.model_dir,'child'),
    'data_dir': FLAGS.data_dir,
    'test_set': FLAGS.test_set,
    'test_batch_size': FLAGS.test_batch_size,
    'num_classes': 10,
    'total_steps': 0,
    'arch_pool': archs,
  })
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
