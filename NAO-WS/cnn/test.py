from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tensorflow as tf
from model import get_ops
from data_utils import read_data

parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])

parser.add_argument('--data_path', type=str, default='/tmp/cifar10_data')

parser.add_argument('--output_dir', type=str, default='models')

parser.add_argument('--child_batch_size', type=int, default=128)

parser.add_argument('--child_eval_batch_size', type=int, default=128)

parser.add_argument('--child_num_epochs', type=int, default=150)

parser.add_argument('--child_lr_dec_every', type=int, default=100)

parser.add_argument('--child_num_layers', type=int, default=5)

parser.add_argument('--child_num_cells', type=int, default=5)

parser.add_argument('--child_out_filters', type=int, default=20)

parser.add_argument('--child_out_filters_scale', type=int, default=1)

parser.add_argument('--child_num_branches', type=int, default=5)

parser.add_argument('--child_num_aggregate', type=int, default=None)

parser.add_argument('--child_num_replicas', type=int, default=None)

parser.add_argument('--child_lr_T_0', type=int, default=None)

parser.add_argument('--child_lr_T_mul', type=int, default=None)

parser.add_argument('--child_cutout_size', type=int, default=None)

parser.add_argument('--child_grad_bound', type=float, default=5.0)

parser.add_argument('--child_lr', type=float, default=0.1)

parser.add_argument('--child_lr_dec_rate', type=float, default=0.1)

parser.add_argument('--child_lr_max', type=float, default=None)

parser.add_argument('--child_lr_min', type=float, default=None)

parser.add_argument('--child_keep_prob', type=float, default=0.5)

parser.add_argument('--child_drop_path_keep_prob', type=float, default=1.0)

parser.add_argument('--child_l2_reg', type=float, default=1e-4)

parser.add_argument('--child_fixed_arc', type=str, default=None)

parser.add_argument('--child_use_aux_heads', action='store_true', default=False)

parser.add_argument('--child_sync_replicas', action='store_true', default=False)

parser.add_argument('--child_lr_cosine', action='store_true', default=False)

parser.add_argument('--child_eval_every_epochs', type=int, default=1)

parser.add_argument('--child_data_format', type=str, default="NHWC", choices=['NHWC', 'NCHW'])


def test():
  params = get_child_model_params()
  images, labels = read_data(params['data_dir'], num_valids=0)
  g = tf.Graph()
  with g.as_default():
    ops = get_ops(images, labels, params)
    tf.logging.info("-" * 80)
    tf.logging.info("Starting session")
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
        config=config, checkpoint_dir=params['model_dir']) as sess:
        ops["eval_func"](sess, "test")

def get_child_model_params():
  params = {
    'data_dir': FLAGS.data_path,
    'model_dir': FLAGS.output_dir,
    'batch_size': FLAGS.child_batch_size,
    'eval_batch_size': FLAGS.child_eval_batch_size,
    'num_epochs': FLAGS.child_num_epochs,
    'lr_dec_every': FLAGS.child_lr_dec_every,
    'num_layers': FLAGS.child_num_layers,
    'num_cells': FLAGS.child_num_cells,
    'out_filters': FLAGS.child_out_filters,
    'out_filters_scale': FLAGS.child_out_filters_scale,
    'num_aggregate': FLAGS.child_num_aggregate,
    'num_replicas': FLAGS.child_num_replicas,
    'lr_T_0': FLAGS.child_lr_T_0,
    'lr_T_mul': FLAGS.child_lr_T_mul,
    'cutout_size': FLAGS.child_cutout_size,
    'grad_bound': FLAGS.child_grad_bound,
    'lr_dec_rate': FLAGS.child_lr_dec_rate,
    'lr_max': FLAGS.child_lr_max,
    'lr_min': FLAGS.child_lr_min,
    'drop_path_keep_prob': FLAGS.child_drop_path_keep_prob,
    'keep_prob': FLAGS.child_keep_prob,
    'l2_reg': FLAGS.child_l2_reg,
    'fixed_arc': FLAGS.child_fixed_arc,
    'use_aux_heads': FLAGS.child_use_aux_heads,
    'sync_replicas': FLAGS.child_sync_replicas,
    'lr_cosine': FLAGS.child_lr_cosine,
    'eval_every_epochs': FLAGS.child_eval_every_epochs,
    'data_format': FLAGS.child_data_format,
    'lr': FLAGS.child_lr,
  }
  return params


def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  test()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
