from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

INITRANGE = 0.04
SOS_ID = 0
EOS_ID = 0


class Attention(nn.Module):
  def __init__(self, dim):
    super(Attention, self).__init__()
    self.linear_out = nn.Linear(dim * 2, dim)
    self.mask = None
  
  def set_mask(self, mask):
    self.mask = mask
  
  def forward(self, output, context):
    batch_size = output.size(0)
    hidden_size = output.size(2)
    input_size = context.size(1)
    # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
    attn = torch.bmm(output, context.transpose(1, 2))
    if self.mask is not None:
      attn.data.masked_fill_(self.mask, -float('inf'))
    attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)
    
    # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
    mix = torch.bmm(attn, context)
    
    # concat -> (batch, out_len, 2*dim)
    combined = torch.cat((mix, output), dim=2)
    # output -> (batch, out_len, dim)
    output = F.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
    
    return output, attn
  
class Decoder(nn.Module):
  
  KEY_ATTN_SCORE = 'attention_score'
  KEY_LENGTH = 'length'
  KEY_SEQUENCE = 'sequence'
  
  def __init__(self, params):
    super(Decoder, self).__init__()
    self.num_layers = params['decoder_num_layers']
    self.hidden_size = params['decoder_hidden_size']
    self.decoder_length = params['decoder_length']
    self.source_length = params['encoder_length']
    self.vocab_size = params['decoder_vocab_size']
    self.dropout_p = params['decoder_dropout']
    self.dropout = nn.Dropout(p=self.dropout_p)
    self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True, dropout=self.dropout_p)
    self.sos_id = SOS_ID
    self.eos_id = EOS_ID
    self.init_input = None
    self.embedding = nn.Embedding(self.vocab_size, self.hidden_size)
    self.attention = Attention(self.hidden_size)
    self.out = nn.Linear(self.hidden_size, self.vocab_size)
  
  def forward_step(self, x, hidden, encoder_outputs, function):
    batch_size = x.size(0)
    output_size = x.size(1)
    embedded = self.embedding(x)
    embedded = self.dropout(embedded)
    output, hidden = self.rnn(embedded, hidden)
    output, attn = self.attention(output, encoder_outputs)
    
    predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)),dim=1).view(batch_size, output_size, -1)
    return predicted_softmax, hidden, attn
 
  def forward(self, x, encoder_hidden=None, encoder_outputs=None, function=F.log_softmax):
    ret_dict = dict()
    ret_dict[Decoder.KEY_ATTN_SCORE] = list()
    if x is None:
      inference = True
    else:
      inference = False
    x, batch_size, length = self._validate_args(x, encoder_hidden, encoder_outputs)
    assert length == self.decoder_length
    decoder_hidden = self._init_state(encoder_hidden)
    decoder_outputs = []
    sequence_symbols = []
    lengths = np.array([length] * batch_size)

    def decode(step, step_output, step_attn):
      decoder_outputs.append(step_output)
      ret_dict[Decoder.KEY_ATTN_SCORE].append(step_attn)
      if step % 2 == 0: # sample index, should be in [1, step+1]
        symbols = decoder_outputs[-1][:, 1:step // 2 + 2].topk(1)[1] + 1
      else: # sample operation, should be in [12, 15]
        symbols = decoder_outputs[-1][:, 12:].topk(1)[1] + 12
      
      sequence_symbols.append(symbols)
  
      eos_batches = symbols.data.eq(self.eos_id)
      if eos_batches.dim() > 0:
        eos_batches = eos_batches.cpu().view(-1).numpy()
        update_idx = ((lengths > step) & eos_batches) != 0
        lengths[update_idx] = len(sequence_symbols)
      return symbols

    decoder_input = x[:, 0].unsqueeze(1)
    for di in range(length):
      if not inference:
        decoder_input = x[:, di].unsqueeze(1)
      decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                    function=function)
      step_output = decoder_output.squeeze(1)
      symbols = decode(di, step_output, step_attn)
      decoder_input = symbols
      
    ret_dict[Decoder.KEY_SEQUENCE] = sequence_symbols
    ret_dict[Decoder.KEY_LENGTH] = lengths.tolist()
    
    return decoder_outputs, decoder_hidden, ret_dict

  def _init_state(self, encoder_hidden):
    """ Initialize the encoder hidden state. """
    if encoder_hidden is None:
      return None
    if isinstance(encoder_hidden, tuple):
      encoder_hidden = tuple([h for h in encoder_hidden])
    else:
      encoder_hidden = encoder_hidden
    return encoder_hidden

  def _validate_args(self, x, encoder_hidden, encoder_outputs):
    if encoder_outputs is None:
      raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
  
    # inference batch size
    if x is None and encoder_hidden is None:
      batch_size = 1
    else:
      if x is not None:
        batch_size = x.size(0)
      else:
        batch_size = encoder_hidden[0].size(1)
  
    # set default input and max decoding length
    if x is None:
      x = Variable(torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1))
      if torch.cuda.is_available():
        x = x.cuda()
      max_length = self.decoder_length
    else:
      max_length = x.size(1)
  
    return x, batch_size, max_length

  def eval(self):
    return

  def infer(self, x, encoder_hidden=None, encoder_outputs=None):
    decoder_outputs, decoder_hidden, _ = self(x, encoder_hidden, encoder_outputs)
    return decoder_outputs, decoder_hidden
