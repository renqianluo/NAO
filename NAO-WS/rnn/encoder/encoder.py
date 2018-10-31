from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

INITRANGE = 0.04

class Encoder(nn.Module):
  def __init__(self, params):
    super(Encoder, self).__init__()
    self.num_layers = params['encoder_num_layers']
    self.vocab_size = params['encoder_vocab_size']
    self.emb_size = params['encoder_emb_size']
    self.hidden_size = params['encoder_hidden_size']
    self.dropout_p = params['encoder_dropout']
    self.dropout = nn.Dropout(p=self.dropout_p)
    self.encoder_length = params['encoder_length']
    self.source_length = params['source_length']
    self.mlp_num_layers = params['mlp_num_layers']
    self.mlp_hidden_size = params['mlp_hidden_size']
    self.mlp_dropout_p = params['mlp_dropout']
    self.mlp_dropout = nn.Dropout(p=self.mlp_dropout_p)
    self.weight_decay = params['weight_decay']
    self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
    self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_p)
    weight = []
    bias = []
    for i in range(self.mlp_num_layers):
      if i == 0:
        w = nn.Parameter(torch.Tensor(self.hidden_size, self.mlp_hidden_size).uniform_(-INITRANGE, INITRANGE))
      else:
        w = nn.Parameter(torch.Tensor(self.mlp_hidden_size, self.mlp_hidden_size).uniform_(-INITRANGE, INITRANGE))
      weight.append(w)
      b = nn.Parameter(torch.Tensor(self.mlp_hidden_size).zero_())
      bias.append(b)
    weight.append(nn.Parameter(torch.Tensor(self.hidden_size if self.mlp_num_layers == 0 else self.mlp_hidden_size, 1).uniform_(-INITRANGE, INITRANGE)))
    bias.append(nn.Parameter(torch.Tensor(1).zero_()))
    self.W = nn.ParameterList(weight)
    self.b = nn.ParameterList(bias)
  
  def forward(self, x):
    embedded = self.embedding(x)
    embedded = self.dropout(embedded)
    if self.source_length != self.encoder_length:
      #logging.info('Concacting source sequence along depth')
      assert self.source_length % self.encoder_length == 0
      ratio = self.source_length // self.encoder_length
      embedded = embedded.view(-1, self.source_length // ratio, ratio * self.emb_size)
    out, hidden = self.rnn(embedded)
    out = F.normalize(out, 2, dim=-1)
    encoder_outputs = out
    encoder_state = hidden
    
    out = torch.mean(out, dim=1)
    out = F.normalize(out, 2, dim=-1)
    arch_emb = out
    
    for i in range(self.mlp_num_layers):
      out = out.mm(self.W[i]) + self.b[i]
      out = F.relu(out)
      out = self.mlp_dropout(out)
    out = out.mm(self.W[-1]) + self.b[-1]
    predict_value = F.sigmoid(out)
    return encoder_outputs, encoder_state, arch_emb, predict_value
  
  def infer(self, x, predict_lambda):
    encoder_outputs, encoder_state, arch_emb, predict_value = self(x)
    grads_on_outputs = torch.autograd.grad(predict_value, encoder_outputs, torch.ones_like(predict_value))[0]
    new_encoder_outputs = encoder_outputs - predict_lambda * grads_on_outputs
    new_encoder_outputs = F.normalize(new_encoder_outputs, 2, dim=-1)
    new_arch_emb = torch.mean(new_encoder_outputs, dim=1)
    new_arch_emb = F.normalize(new_arch_emb, 2, dim=-1)
    return encoder_outputs, encoder_state, arch_emb, predict_value, new_encoder_outputs, new_arch_emb