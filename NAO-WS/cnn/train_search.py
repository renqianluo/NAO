from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import subprocess
import numpy as np
import tensorflow as tf
import copy
import json
import math
import time
from model_search import train as child_train
from model_search import valid as child_valid
import utils
from calculate_params import calculate_params
import controller


parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])

parser.add_argument('--data_path', type=str, default='/tmp/cifar10_data')

parser.add_argument('--eval_dataset', type=str, default='valid',
                    choices=['valid', 'test', 'both'])

parser.add_argument('--output_dir', type=str, default='models')

parser.add_argument('--child_sample_policy', type=str, default=None)

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

parser.add_argument('--child_eval_every_epochs', type=str, default='30')

parser.add_argument('--child_arch_pool', type=str, default=None)

parser.add_argument('--child_data_format', type=str, default="NHWC", choices=['NHWC', 'NCHW'])

parser.add_argument('--controller_num_seed_arch', type=int, default=1000)

parser.add_argument('--controller_encoder_num_layers', type=int, default=1)

parser.add_argument('--controller_encoder_hidden_size', type=int, default=96)

parser.add_argument('--controller_encoder_emb_size', type=int, default=32)

parser.add_argument('--controller_mlp_num_layers', type=int, default=0)

parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)

parser.add_argument('--controller_decoder_num_layers', type=int, default=1)

parser.add_argument('--controller_decoder_hidden_size', type=int, default=96)

parser.add_argument('--controller_source_length', type=int, default=60)

parser.add_argument('--controller_encoder_length', type=int, default=20)

parser.add_argument('--controller_decoder_length', type=int, default=60)

parser.add_argument('--controller_encoder_dropout', type=float, default=0.1)

parser.add_argument('--controller_mlp_dropout', type=float, default=0.1)

parser.add_argument('--controller_decoder_dropout', type=float, default=0.0)

parser.add_argument('--controller_weight_decay', type=float, default=1e-4)

parser.add_argument('--controller_encoder_vocab_size', type=int, default=12)

parser.add_argument('--controller_decoder_vocab_size', type=int, default=12)

parser.add_argument('--controller_trade_off', type=float, default=0.8)

parser.add_argument('--controller_train_epochs', type=int, default=300)

parser.add_argument('--controller_save_frequency', type=int, default=10)

parser.add_argument('--controller_batch_size', type=int, default=100)

parser.add_argument('--controller_lr', type=float, default=0.001)

parser.add_argument('--controller_optimizer', type=str, default='adam')

parser.add_argument('--controller_start_decay_step', type=int, default=100)

parser.add_argument('--controller_decay_steps', type=int, default=1000)

parser.add_argument('--controller_decay_factor', type=float, default=0.9)

parser.add_argument('--controller_attention', action='store_true', default=False)

parser.add_argument('--controller_max_gradient_norm', type=float, default=5.0)

parser.add_argument('--controller_time_major', action='store_true', default=False)

parser.add_argument('--controller_symmetry', action='store_true', default=False)

parser.add_argument('--controller_predict_beam_width', type=int, default=0)

parser.add_argument('--controller_predict_lambda', type=float, default=1)


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


def train():
  child_params = get_child_model_params()
  controller_params = get_controller_params()
  branch_length = controller_params['source_length'] // 2 // 5 // 2
  eval_every_epochs = child_params['eval_every_epochs']
  child_epoch = 0
  while True:
    # Train child model
    if child_params['arch_pool'] is None:
      arch_pool = utils.generate_arch(controller_params['num_seed_arch'], child_params['num_cells'], 5) #[[[conv],[reduc]]]
      child_params['arch_pool'] = arch_pool
      child_params['arch_pool_prob'] = None
    else:
      if child_params['sample_policy'] == 'uniform':
        child_params['arch_pool_prob'] = None
      elif child_params['sample_policy'] == 'params':
        child_params['arch_pool_prob'] = calculate_params(child_params['arch_pool'])
      elif child_params['sample_policy'] == 'valid_performance':
          child_params['arch_pool_prob'] = child_valid(child_params)
      elif child_params['sample_policy'] == 'predicted_performance':
        encoder_input = list(map(lambda x: utils.parse_arch_to_seq(x[0], branch_length) + \
                                           utils.parse_arch_to_seq(x[1], branch_length), child_params['arch_pool']))
        predicted_error_rate = controller.test(controller_params, encoder_input)
        child_params['arch_pool_prob'] = [1-i[0] for i in predicted_error_rate]
      else:
        raise ValueError('Child model arch pool sample policy is not provided!')
    
    if isinstance(eval_every_epochs, int):
      child_params['eval_every_epochs'] = eval_every_epochs
    else:
      for index, e in enumerate(eval_every_epochs):
        if child_epoch < e:
          child_params['eval_every_epochs'] = e
          break
          
    child_epoch = child_train(child_params)
    
    # Evaluate seed archs
    valid_accuracy_list = child_valid(child_params)

    # Output archs and evaluated error rate
    old_archs = child_params['arch_pool']
    old_archs_perf = [1 - i for i in valid_accuracy_list]
    
    old_archs_sorted_indices = np.argsort(old_archs_perf)
    old_archs = np.array(old_archs)[old_archs_sorted_indices].tolist()
    old_archs_perf = np.array(old_archs_perf)[old_archs_sorted_indices].tolist()
    with open(os.path.join(child_params['model_dir'], 'arch_pool.{}'.format(child_epoch)), 'w') as fa:
      with open(os.path.join(child_params['model_dir'], 'arch_pool.perf.{}'.format(child_epoch)), 'w') as fp:
        with open(os.path.join(child_params['model_dir'], 'arch_pool'), 'w') as fa_latest:
          with open(os.path.join(child_params['model_dir'], 'arch_pool.perf'),'w') as fp_latest:
            for arch, perf in zip(old_archs, old_archs_perf):
              arch = ' '.join(map(str, arch[0] + arch[1]))
              fa.write('{}\n'.format(arch))
              fa_latest.write('{}\n'.format(arch))
              fp.write('{}\n'.format(perf))
              fp_latest.write('{}\n'.format(perf))
      
    if child_epoch >= child_params['num_epochs']:
      break

    # Train Encoder-Predictor-Decoder
    encoder_input = list(map(lambda x : utils.parse_arch_to_seq(x[0], branch_length) + \
                                      utils.parse_arch_to_seq(x[1], branch_length), old_archs))
    #[[conv, reduc]]
    min_val = min(old_archs_perf)
    max_val = max(old_archs_perf)
    encoder_target = [(i - min_val)/(max_val - min_val) for i in old_archs_perf]
    decoder_target = copy.copy(encoder_input)
    controller_params['batches_per_epoch'] = math.ceil(len(encoder_input) / controller_params['batch_size'])
    #if clean controller model
    controller.train(controller_params, encoder_input, encoder_target, decoder_target)
    
    # Generate new archs
    #old_archs = old_archs[:450]
    new_archs = []
    max_step_size = 100
    controller_params['predict_lambda'] = 0
    top100_archs = list(map(lambda x : utils.parse_arch_to_seq(x[0], branch_length) + \
                                      utils.parse_arch_to_seq(x[1], branch_length), old_archs[:100]))
    while len(new_archs) < 500:
      controller_params['predict_lambda'] += 1
      new_arch = controller.predict(controller_params, top100_archs)
      for arch in new_arch:
        if arch not in encoder_input and arch not in new_archs:
          new_archs.append(arch)
        if len(new_archs) >= 500:
          break
      tf.logging.info('{} new archs generated now'.format(len(new_archs)))
      if controller_params['predict_lambda'] > max_step_size:
        break
          #[[conv, reduc]]
    new_archs = list(map(lambda x: utils.parse_seq_to_arch(x, branch_length), new_archs)) #[[[conv],[reduc]]]
    num_new_archs = len(new_archs)
    tf.logging.info("Generate {} new archs".format(num_new_archs))
    new_arch_pool = old_archs[:len(old_archs)-(num_new_archs+50)] + new_archs + utils.generate_arch(50, 5, 5)
    tf.logging.info("Totally {} archs now to train".format(len(new_arch_pool)))
    child_params['arch_pool'] = new_arch_pool
    with open(os.path.join(child_params['model_dir'], 'arch_pool'), 'w') as f:
      for arch in new_arch_pool:
        arch = ' '.join(map(str, arch[0] + arch[1]))
        f.write('{}\n'.format(arch))
  
  
def get_child_model_params():
  params = {
    'data_dir': FLAGS.data_path,
    'model_dir': os.path.join(FLAGS.output_dir, 'child'),
    'sample_policy': FLAGS.child_sample_policy,
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
    'eval_every_epochs': eval(FLAGS.child_eval_every_epochs),
    'data_format': FLAGS.child_data_format,
    'lr': FLAGS.child_lr,
    'arch_pool': None,
  }
  if FLAGS.child_arch_pool is not None:
    with open(FLAGS.child_arch_pool) as f:
      archs = f.read().splitlines()
      archs = list(map(utils.build_dag, archs))
      params['arch_pool'] = archs
  if os.path.exists(os.path.join(params['model_dir'], 'arch_pool')):
    tf.logging.info('Found arch_pool in child model dir, loading')
    with open(os.path.join(params['model_dir'], 'arch_pool')) as f:
      archs = f.read().splitlines()
      archs = list(map(utils.build_dag, archs))
      params['arch_pool'] = archs

  return params

def get_controller_params():
  params = {
    'model_dir': os.path.join(FLAGS.output_dir, 'controller'),
    'num_seed_arch': FLAGS.controller_num_seed_arch,
    'encoder_num_layers': FLAGS.controller_encoder_num_layers,
    'encoder_hidden_size': FLAGS.controller_encoder_hidden_size,
    'encoder_emb_size': FLAGS.controller_encoder_emb_size,
    'mlp_num_layers': FLAGS.controller_mlp_num_layers,
    'mlp_hidden_size': FLAGS.controller_mlp_hidden_size,
    'decoder_num_layers': FLAGS.controller_decoder_num_layers,
    'decoder_hidden_size': FLAGS.controller_decoder_hidden_size,
    'source_length': FLAGS.controller_source_length,
    'encoder_length': FLAGS.controller_encoder_length,
    'decoder_length': FLAGS.controller_decoder_length,
    'encoder_dropout': FLAGS.controller_encoder_dropout,
    'mlp_dropout': FLAGS.controller_mlp_dropout,
    'decoder_dropout': FLAGS.controller_decoder_dropout,
    'weight_decay': FLAGS.controller_weight_decay,
    'encoder_vocab_size': FLAGS.controller_encoder_vocab_size,
    'decoder_vocab_size': FLAGS.controller_decoder_vocab_size,
    'trade_off': FLAGS.controller_trade_off,
    'train_epochs': FLAGS.controller_train_epochs,
    'save_frequency': FLAGS.controller_save_frequency,
    'batch_size': FLAGS.controller_batch_size,
    'lr': FLAGS.controller_lr,
    'optimizer': FLAGS.controller_optimizer,
    'start_decay_step': FLAGS.controller_start_decay_step,
    'decay_steps': FLAGS.controller_decay_steps,
    'decay_factor': FLAGS.controller_decay_factor,
    'attention': FLAGS.controller_attention,
    'max_gradient_norm': FLAGS.controller_max_gradient_norm,
    'time_major': FLAGS.controller_time_major,
    'symmetry': FLAGS.controller_symmetry,
    'predict_beam_width': FLAGS.controller_predict_beam_width,
    'predict_lambda': FLAGS.controller_predict_lambda
  }
  return params

def main(unused_argv):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  all_params = vars(FLAGS)
  with open(os.path.join(FLAGS.output_dir, 'hparams.json'), 'w') as f:
    json.dump(all_params, f)
  train()


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
