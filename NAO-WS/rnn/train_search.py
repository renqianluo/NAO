from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import subprocess
import math
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import copy
import json
import model_search
import data
from utils import batchify, save_checkpoint, generate_arch, parse_arch_to_seq, parse_seq_to_arch, build_arch, normalize_target
import controller

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 Language Model')
parser.add_argument('--data_dir', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model_dir', type=str,  default='models',
                    help='path to save the final model')
parser.add_argument('--child_emb_size', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--child_nhid', type=int, default=300,
                    help='number of hidden units per layer')
parser.add_argument('--child_nhidlast', type=int, default=300,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--child_arch_pool', type=str, default=None,
                    help='Archs to sample from. If None for default, then randomly sampled arch is used.')
parser.add_argument('--child_lr', type=float, default=0.25,
                    help='initial learning rate')
parser.add_argument('--child_clip', type=float, default=10.0,
                    help='gradient clipping')
parser.add_argument('--child_train_epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--child_batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--child_eval_batch_size', type=int, default=256,
                    help='eval batch size')
parser.add_argument('--child_bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--child_dropout', type=float, default=0.75,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--child_dropouth', type=float, default=0.25,
                    help='dropout for hidden nodes in rnn layers (0 = no dropout)')
parser.add_argument('--child_dropoutx', type=float, default=0.75,
                    help='dropout for input nodes in rnn layers (0 = no dropout)')
parser.add_argument('--child_dropouti', type=float, default=0.2,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--child_dropoute', type=float, default=0,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--child_drop_path', type=float, default=0.0,
                    help='drop a path.')
parser.add_argument('--child_log_interval', type=int, default=50, metavar='N',
                    help='report interval')
parser.add_argument('--child_alpha', type=float, default=0,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--child_beta', type=float, default=1e-3,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--child_weight_decay', type=float, default=5e-7,
                    help='weight decay applied to all weights')
parser.add_argument('--child_eval_every_epochs', type=str, default='100',
                    help='The number of epochs to run in between evaluations.')
parser.add_argument('--child_small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--child_max_seq_len_delta', type=int, default=20,
                    help='max sequence length')
parser.add_argument('--controller_shuffle', action='store_true', default=False)
parser.add_argument('--controller_max_new_archs', type=int, default=300)
parser.add_argument('--controller_max_step_size', type=int, default=30)
parser.add_argument('--controller_num_seed_arch', type=int, default=1000)
parser.add_argument('--controller_encoder_num_layers', type=int, default=1)
parser.add_argument('--controller_encoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_encoder_emb_size', type=int, default=96)
parser.add_argument('--controller_mlp_num_layers', type=int, default=0)
parser.add_argument('--controller_mlp_hidden_size', type=int, default=200)
parser.add_argument('--controller_decoder_num_layers', type=int, default=1)
parser.add_argument('--controller_decoder_hidden_size', type=int, default=96)
parser.add_argument('--controller_source_length', type=int, default=22)
parser.add_argument('--controller_encoder_length', type=int, default=22)
parser.add_argument('--controller_decoder_length', type=int, default=22)
parser.add_argument('--controller_encoder_dropout', type=float, default=0.1)
parser.add_argument('--controller_mlp_dropout', type=float, default=0)
parser.add_argument('--controller_decoder_dropout', type=float, default=0.0)
parser.add_argument('--controller_weight_decay', type=float, default=1e-4)
parser.add_argument('--controller_encoder_vocab_size', type=int, default=16)
parser.add_argument('--controller_decoder_vocab_size', type=int, default=16)
parser.add_argument('--controller_trade_off', type=float, default=0.8)
parser.add_argument('--controller_train_epochs', type=int, default=1000)
parser.add_argument('--controller_save_frequency', type=int, default=100)
parser.add_argument('--controller_batch_size', type=int, default=100)
parser.add_argument('--controller_lr', type=float, default=0.001)
parser.add_argument('--controller_optimizer', type=str, default='adam')
parser.add_argument('--controller_max_gradient_norm', type=float, default=5.0)
parser.add_argument('--controller_predict_beam_width', type=int, default=0)
parser.add_argument('--controller_predict_lambda', type=float, default=1)
parser.add_argument('--seed', type=int, default=1267,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--single_gpu', default=True, action='store_false',
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0, help='GPU device to use')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.model_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
  if not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
  else:
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled=True
    torch.cuda.manual_seed_all(args.seed)


def train():
  child_params = get_child_model_params()
  controller_params = get_controller_params()
  corpus = data.Corpus(child_params['data_dir'])
  eval_batch_size = child_params['eval_batch_size']

  train_data = batchify(corpus.train, child_params['batch_size'], child_params['cuda'])
  val_data = batchify(corpus.valid, eval_batch_size, child_params['cuda'])
  ntokens = len(corpus.dictionary)
  
  if os.path.exists(os.path.join(child_params['model_dir'], 'model.pt')):
    print("Found model.pt in {}, automatically continue training.".format(os.path.join(child_params['model_dir'])))
    continue_train_child = True
  else:
    continue_train_child = False
  
  if continue_train_child:
    child_model = torch.load(os.path.join(child_params['model_dir'], 'model.pt'))
  else:
    child_model = model_search.RNNModelSearch(ntokens, child_params['emsize'], child_params['nhid'], child_params['nhidlast'],
                                 child_params['dropout'], child_params['dropouth'], child_params['dropoutx'],
                                 child_params['dropouti'], child_params['dropoute'], child_params['drop_path'])

  if os.path.exists(os.path.join(controller_params['model_dir'], 'model.pt')):
    print("Found model.pt in {}, automatically continue training.".format(os.path.join(child_params['model_dir'])))
    continue_train_controller = True
  else:
    continue_train_controller = False

  if continue_train_controller:
    controller_model = torch.load(os.path.join(controller_params['model_dir'], 'model.pt'))
  else:
    controller_model = controller.Controller(controller_params)

  size = 0
  for p in child_model.parameters():
    size += p.nelement()
  logging.info('child model param size: {}'.format(size))
  size = 0
  for p in controller_model.parameters():
    size += p.nelement()
  logging.info('controller model param size: {}'.format(size))
  

  if args.cuda:
    if args.single_gpu:
      parallel_child_model = child_model.cuda()
      parallel_controller_model = controller_model.cuda()
    else:
      parallel_child_model = nn.DataParallel(child_model, dim=1).cuda()
      parallel_controller_model = nn.DataParallel(controller_model, dim=1).cuda()
  else:
    parallel_child_model = child_model
    parallel_controller_model = controller_model
    

  total_params = sum(x.data.nelement() for x in child_model.parameters())
  logging.info('Args: {}'.format(args))
  logging.info('Child Model total parameters: {}'.format(total_params))
  total_params = sum(x.data.nelement() for x in controller_model.parameters())
  logging.info('Args: {}'.format(args))
  logging.info('Controller Model total parameters: {}'.format(total_params))

  # Loop over epochs.

  if continue_train_child:
    optimizer_state = torch.load(os.path.join(child_params['model_dir'], 'optimizer.pt'))
    if 't0' in optimizer_state['param_groups'][0]:
      child_optimizer = torch.optim.ASGD(child_model.parameters(), lr=child_params['lr'], t0=0, lambd=0., weight_decay=child_params['wdecay'])
    else:
      child_optimizer = torch.optim.SGD(child_model.parameters(), lr=child_params['lr'], weight_decay=child_params['wdecay'])
    child_optimizer.load_state_dict(optimizer_state)
    child_epoch = torch.load(os.path.join(child_params['model_dir'], 'misc.pt'))['epoch'] - 1
  else:
    child_optimizer = torch.optim.SGD(child_model.parameters(), lr=child_params['lr'], weight_decay=child_params['wdecay'])
    child_epoch = 0
  
  if continue_train_controller:
    optimizer_state = torch.load(os.path.join(controller_params['model_dir'], 'optimizer.pt'))
    controller_optimizer = torch.optim.Adam(controller_model.parameters(), lr=controller_params['lr'], weight_decay=controller_params['weight_decay'])
    controller_optimizer.load_state_dict(optimizer_state)
    controller_epoch = torch.load(os.path.join(controller_params['model_dir'], 'misc.pt'))['epoch'] - 1
  else:
    controller_optimizer = torch.optim.Adam(controller_model.parameters(), lr=controller_params['lr'], weight_decay=controller_params['weight_decay'])
    controller_epoch = 0
  eval_every_epochs = child_params['eval_every_epochs']
  while True:
    # Train child model
    if child_params['arch_pool'] is None:
      arch_pool = generate_arch(controller_params['num_seed_arch']) #[[arch]]
      child_params['arch_pool'] = arch_pool
    child_params['arch'] = None
    
    if isinstance(eval_every_epochs, int):
      child_params['eval_every_epochs'] = eval_every_epochs
    else:
      eval_every_epochs = list(map(int, eval_every_epochs))
      for index, e in enumerate(eval_every_epochs):
        if child_epoch < e:
          child_params['eval_every_epochs'] = e
          break
    
    for e in range(child_params['eval_every_epochs']):
      child_epoch += 1
      model_search.train(train_data, child_model, parallel_child_model, child_optimizer, child_params, child_epoch)
      if child_epoch % child_params['eval_every_epochs'] == 0:
        save_checkpoint(child_model, child_optimizer, child_epoch, child_params['model_dir'])
        logging.info('Saving Model!')
      if child_epoch >= child_params['train_epochs']:
        break
    
    # Evaluate seed archs
    valid_accuracy_list = model_search.evaluate(val_data, child_model, parallel_child_model, child_params, eval_batch_size)

    # Output archs and evaluated error rate
    old_archs = child_params['arch_pool']
    old_archs_perf = valid_accuracy_list
    
    old_archs_sorted_indices = np.argsort(old_archs_perf)
    old_archs = np.array(old_archs)[old_archs_sorted_indices].tolist()
    old_archs_perf = np.array(old_archs_perf)[old_archs_sorted_indices].tolist()
    with open(os.path.join(child_params['model_dir'], 'arch_pool.{}'.format(child_epoch)), 'w') as fa:
      with open(os.path.join(child_params['model_dir'], 'arch_pool.perf.{}'.format(child_epoch)), 'w') as fp:
        with open(os.path.join(child_params['model_dir'], 'arch_pool'), 'w') as fa_latest:
          with open(os.path.join(child_params['model_dir'], 'arch_pool.perf'),'w') as fp_latest:
            for arch, perf in zip(old_archs, old_archs_perf):
              arch = ' '.join(map(str, arch))
              fa.write('{}\n'.format(arch))
              fa_latest.write('{}\n'.format(arch))
              fp.write('{}\n'.format(perf))
              fp_latest.write('{}\n'.format(perf))
      
    if child_epoch >= child_params['train_epochs']:
      logging.info('Training finished!')
      break

    # Train Encoder-Predictor-Decoder
    # [[arch]]
    encoder_input = list(map(lambda x : parse_arch_to_seq(x), old_archs))
    encoder_target = normalize_target(old_archs_perf)
    decoder_target = copy.copy(encoder_input)
    controller_params['batches_per_epoch'] = math.ceil(len(encoder_input) / controller_params['batch_size'])
    controller_epoch = controller.train(encoder_input, encoder_target, decoder_target, controller_model, parallel_controller_model, controller_optimizer, controller_params, controller_epoch)
      
    
    # Generate new archs
    new_archs = []
    controller_params['predict_lambda'] = 0
    top100_archs = list(map(lambda x : parse_arch_to_seq(x), old_archs[:100]))
    max_step_size = controller_params['max_step_size']
    while len(new_archs) < controller_params['max_new_archs']:
      controller_params['predict_lambda'] += 1
      new_arch = controller.infer(top100_archs, controller_model, parallel_controller_model, controller_params)
      for arch in new_arch:
        if arch not in encoder_input and arch not in new_archs:
          new_archs.append(arch)
        if len(new_archs) >= controller_params['max_new_archs']:
          break
      logging.info('{} new archs generated now'.format(len(new_archs)))
      if controller_params['predict_lambda'] >= max_step_size:
        break
    #[[arch]]
    new_archs = list(map(lambda x: parse_seq_to_arch(x), new_archs)) #[[arch]]
    num_new_archs = len(new_archs)
    logging.info("Generate {} new archs".format(num_new_archs))
    random_new_archs = generate_arch(50)
    new_arch_pool = old_archs[:len(old_archs)-num_new_archs-50] + new_archs + random_new_archs
    logging.info("Totally {} archs now to train".format(len(new_arch_pool)))
    child_params['arch_pool'] = new_arch_pool
    with open(os.path.join(child_params['model_dir'], 'arch_pool'), 'w') as f:
      for arch in new_arch_pool:
        arch = ' '.join(map(str, arch))
        f.write('{}\n'.format(arch))

          
def get_child_model_params():
  params = {
    'data_dir': args.data_dir,
    'model_dir': os.path.join(args.model_dir, 'child'),
    'arch_pool': args.child_arch_pool,
    'emsize': args.child_emb_size,
    'nhid': args.child_nhid,
    'nhidlast': args.child_emb_size if args.child_nhidlast < 0 else args.child_nhidlast,
    'lr': args.child_lr,
    'clip': args.child_clip,
    'train_epochs': args.child_train_epochs,
    'eval_every_epochs': eval(args.child_eval_every_epochs),
    'batch_size': args.child_batch_size,
    'eval_batch_size': args.child_eval_batch_size,
    'bptt': args.child_bptt,
    'dropout': args.child_dropout,
    'dropouth': args.child_dropouth,
    'dropoutx': args.child_dropoutx,
    'dropouti': args.child_dropouti,
    'dropoute': args.child_dropoute,
    'drop_path': args.child_drop_path,
    'seed': args.seed,
    'nonmono': args.nonmono,
    'log_interval': args.child_log_interval,
    'alpha': args.child_alpha,
    'beta': args.child_beta,
    'wdecay': args.child_weight_decay,
    'small_batch_size': args.child_batch_size if args.child_small_batch_size < 0 else args.child_small_batch_size,
    'max_seq_len_delta': args.child_max_seq_len_delta,
    'cuda': args.cuda,
    'single_gpu': args.single_gpu,
    'gpu': args.gpu,
  }
  if args.child_arch_pool is not None:
    with open(args.child_arch_pool) as f:
      archs = f.read().splitlines()
      archs = list(map(build_arch, archs))
      params['arch_pool'] = archs
  if os.path.exists(os.path.join(params['model_dir'], 'arch_pool')):
    logging.info('Found arch_pool in child model dir, loading')
    with open(os.path.join(params['model_dir'], 'arch_pool')) as f:
      archs = f.read().splitlines()
      archs = list(map(build_arch, archs))
      params['arch_pool'] = archs
  return params

def get_controller_params():
  params = {
    'model_dir': os.path.join(args.model_dir, 'controller'),
    'shuffle': args.controller_shuffle,
    'num_seed_arch': args.controller_num_seed_arch,
    'encoder_num_layers': args.controller_encoder_num_layers,
    'encoder_hidden_size': args.controller_encoder_hidden_size,
    'encoder_emb_size': args.controller_encoder_emb_size,
    'mlp_num_layers': args.controller_mlp_num_layers,
    'mlp_hidden_size': args.controller_mlp_hidden_size,
    'decoder_num_layers': args.controller_decoder_num_layers,
    'decoder_hidden_size': args.controller_decoder_hidden_size,
    'source_length': args.controller_source_length,
    'encoder_length': args.controller_encoder_length,
    'decoder_length': args.controller_decoder_length,
    'encoder_dropout': args.controller_encoder_dropout,
    'mlp_dropout': args.controller_mlp_dropout,
    'decoder_dropout': args.controller_decoder_dropout,
    'weight_decay': args.controller_weight_decay,
    'encoder_vocab_size': args.controller_encoder_vocab_size,
    'decoder_vocab_size': args.controller_decoder_vocab_size,
    'trade_off': args.controller_trade_off,
    'train_epochs': args.controller_train_epochs,
    'save_frequency': args.controller_save_frequency,
    'batch_size': args.controller_batch_size,
    'lr': args.controller_lr,
    'optimizer': args.controller_optimizer,
    'max_gradient_norm': args.controller_max_gradient_norm,
    'predict_beam_width': args.controller_predict_beam_width,
    'predict_lambda': args.controller_predict_lambda,
    'max_step_size': args.controller_max_step_size,
    'max_new_archs': args.controller_max_new_archs,
  }
  return params

def main():
  all_params = vars(args)
  with open(os.path.join(args.model_dir, 'hparams.json'), 'w') as f:
    json.dump(all_params, f)
  train()

if __name__ == '__main__':
  main()
