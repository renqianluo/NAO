from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from encoder import encoder
from decoder import decoder
import time
from utils import controller_batchify, controller_get_batch, pairwise_accuracy, hamming_distance, save_checkpoint

SOS_ID=0
EOS_ID=0


class Controller(nn.Module):
  def __init__(self, params):
    super(Controller, self).__init__()
    self.encoder = encoder.Encoder(params)
    self.decoder = decoder.Decoder(params)
    self.decode_function = F.log_softmax
  
  def flatten_parameters(self):
    self.encoder.rnn.flatten_parameters()
    self.decoder.rnn.flatten_parameters()
  
  def forward(self, input_variable, target_variable=None):
    encoder_outputs, encoder_hidden, arch_emb, predict_value = self.encoder(input_variable)
    encoder_hidden = (arch_emb.unsqueeze(0), arch_emb.unsqueeze(0))
    decoder_outputs, decoder_hidden, ret = self.decoder(target_variable, encoder_hidden, encoder_outputs, self.decode_function)
    decoder_outputs = torch.stack(decoder_outputs, 0).permute(1, 0, 2)
    arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
    return predict_value, decoder_outputs, arch
  
  def generate_new_arch(self, input_variable, predict_lambda=1):
    encoder_outputs, encoder_hidden, arch_emb, predict_value, new_encoder_outputs, new_arch_emb = self.encoder.infer(input_variable, predict_lambda)
    new_encoder_hidden = (new_arch_emb.unsqueeze(0), new_arch_emb.unsqueeze(0))
    decoder_outputs, decoder_hidden, ret = self.decoder(None, new_encoder_hidden, new_encoder_outputs, self.decode_function)
    new_arch = torch.stack(ret['sequence'], 0).permute(1, 0, 2)
    return new_arch

def train(encoder_input, encoder_target, decoder_target, model, parallel_model, optimizer, params, epoch):
  logging.info('Training Encoder-Predictor-Decoder')
  step = 0
  start_time = time.time()
  train_epochs = params['train_epochs']
  for e in range(1, train_epochs+1):
    #prepare data
    N = len(encoder_input)
    if params['shuffle']:
      data = list(zip(encoder_input, encoder_target, decoder_target))
      np.random.shuffle(data)
      encoder_input, encoder_target, decoder_target = zip(*data)
    decoder_input = torch.cat((torch.LongTensor([[SOS_ID]] * N), torch.LongTensor(encoder_input)[:, :-1]), dim=1)

    encoder_train_input = controller_batchify(torch.LongTensor(encoder_input), params['batch_size'], cuda=True)
    encoder_train_target = controller_batchify(torch.Tensor(encoder_target), params['batch_size'], cuda=True)
    decoder_train_input = controller_batchify(torch.LongTensor(decoder_input), params['batch_size'], cuda=True)
    decoder_train_target = controller_batchify(torch.LongTensor(decoder_target), params['batch_size'], cuda=True)
    
    epoch += 1
    total_loss = 0
    mse = 0
    cse = 0
    batch = 0
    while batch < encoder_train_input.size(0):
      model.train()
      optimizer.zero_grad()
      encoder_train_input_batch = controller_get_batch(encoder_train_input, batch, evaluation=False)
      encoder_train_target_batch = controller_get_batch(encoder_train_target, batch, evaluation=False)
      decoder_train_input_batch = controller_get_batch(decoder_train_input, batch, evaluation=False)
      decoder_train_target_batch = controller_get_batch(decoder_train_target, batch, evaluation=False)
      predict_value, log_prob, arch = parallel_model(encoder_train_input_batch, decoder_train_input_batch)
      loss_1 = F.mse_loss(predict_value.squeeze(), encoder_train_target_batch.squeeze())
      loss_2 = F.nll_loss(log_prob.contiguous().view(-1,log_prob.size(-1)), decoder_train_target_batch.view(-1))
      loss = params['trade_off'] * loss_1 + (1-params['trade_off']) * loss_2
      mse += loss_1.data
      cse += loss_2.data
      total_loss += loss.data
      loss.backward()
      torch.nn.utils.clip_grad_norm(model.parameters(), params['max_gradient_norm'])
      optimizer.step()
  
      step += 1
      LOG = 100
      if step % LOG == 0:
        elapsed = time.time() - start_time
        cur_loss = total_loss[0] / LOG
        mse = mse[0] / LOG
        cse = cse[0] / LOG
        logging.info('| epoch {:6d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                     'mse {:5.6f} | cross entropy {:5.6f} | loss {:5.6f}'.format(
          e, batch+1, len(encoder_train_input), optimizer.param_groups[0]['lr'],
          elapsed * 1000 / LOG, mse, cse, cur_loss))
        total_loss = 0
        mse = 0
        cse = 0
        start_time = time.time()
      batch += 1
    if e % params['save_frequency'] == 0:
      save_checkpoint(model, optimizer, epoch, params['model_dir'])
      logging.info('Saving Model!')
  return epoch
  
def infer(encoder_input, model, parallel_model, params):
  logging.info('Generating new architectures using gradient descent with step size {}'.format(params['predict_lambda']))
  logging.info('Preparing data')
  encoder_infer_input = controller_batchify(torch.LongTensor(encoder_input), params['batch_size'], cuda=True)
  
  new_arch_list = []
  for i in range(encoder_infer_input.size(0)):
    model.eval()
    model.zero_grad()
    encoder_infer_input_batch = controller_get_batch(encoder_infer_input, i, evaluation=False)
    new_arch = parallel_model.generate_new_arch(encoder_infer_input_batch, params['predict_lambda'])
    new_arch_list.extend(new_arch.data.squeeze().tolist())
  return new_arch_list
