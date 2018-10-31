import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import mask2d
from utils import LockedDropout
from utils import embedded_dropout
from utils import STEPS
from torch.autograd import Variable

INITRANGE = 0.04


class NAOCell(nn.Module):

  def __init__(self, ninp, nhid, dropouth, dropoutx, arch):
    super(NAOCell, self).__init__()
    self.nhid = nhid
    self.dropouth = dropouth
    self.dropoutx = dropoutx
    self.arch = arch

    # arch is None when doing arch search
    self._W0 = nn.Parameter(torch.Tensor(ninp+nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE))
    self._Ws = nn.ParameterList([
        nn.Parameter(torch.Tensor(nhid, 2*nhid).uniform_(-INITRANGE, INITRANGE)) for i in range(STEPS)
    ])

  def forward(self, inputs, hidden):
    T, B = inputs.size(0), inputs.size(1)

    if self.training:
      x_mask = mask2d(B, inputs.size(2), keep_prob=1.-self.dropoutx)
      h_mask = mask2d(B, hidden.size(2), keep_prob=1.-self.dropouth)
    else:
      x_mask = h_mask = None

    hidden = hidden[0]
    hiddens = []
    for t in range(T):
      hidden = self.cell(inputs[t], hidden, x_mask, h_mask)
      hiddens.append(hidden)
    hiddens = torch.stack(hiddens)
    return hiddens, hiddens[-1].unsqueeze(0)

  def _compute_init_state(self, x, h_prev, x_mask, h_mask):
    if self.training:
      xh_prev = torch.cat([x * x_mask, h_prev * h_mask], dim=-1)
    else:
      xh_prev = torch.cat([x, h_prev], dim=-1)
    c0, h0 = torch.split(xh_prev.mm(self._W0), self.nhid, dim=-1)
    c0 = c0.sigmoid()
    h0 = h0.tanh()
    s0 = h_prev + c0 * (h0-h_prev)
    return s0

  def _get_activation(self, op_id):
    if op_id == 0: # tanh
      f = F.tanh
    elif op_id == 1: # relu
      f = F.relu
    elif op_id == 2: # sigmoid
      f = F.sigmoid
    elif op_id == 3: # identity
      f = lambda x: x
    else:
      raise NotImplementedError
    return f

  def cell(self, x, h_prev, x_mask, h_mask):
    assert self.arch is not None
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)

    states = [s0]
    for i in range(STEPS):
      pred, act = self.arch[2*i], self.arch[2*i+1]
      s_prev = states[pred]
      if self.training:
        ch = (s_prev * h_mask).mm(self._Ws[i])
      else:
        ch = s_prev.mm(self._Ws[i])
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()
      fn = self._get_activation(act)
      h = fn(h)
      s = s_prev + c * (h-s_prev)
      states += [s]
    output = torch.mean(torch.stack([states[i] for i in range(1,STEPS+1)], -1), -1)
    return output


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nhidlast, 
                 dropout=0.5, dropouth=0.5, dropoutx=0.5, dropouti=0.5, dropoute=0.1, drop_path=0.0,
                 cell_cls=NAOCell, arch=None):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.encoder = nn.Embedding(ntoken, ninp)
        
        assert ninp == nhid == nhidlast
        if cell_cls == NAOCell:
            assert arch is not None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, arch=arch)]
        else:
            assert arch is None
            self.rnns = [cell_cls(ninp, nhid, dropouth, dropoutx, drop_path=drop_path)]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder = nn.Linear(ninp, ntoken)
        self.decoder.weight = self.encoder.weight
        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nhidlast = nhidlast
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropoute = dropoute
        self.ntoken = ntoken
        self.cell_cls = cell_cls

    def init_weights(self):
        self.encoder.weight.data.uniform_(-INITRANGE, INITRANGE)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-INITRANGE, INITRANGE)

    def forward(self, input, hidden, return_h=False):
        batch_size = input.size(1)

        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        emb = self.lockdrop(emb, self.dropouti)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        logit = self.decoder(output.view(-1, self.ninp))
        log_prob = nn.functional.log_softmax(logit, dim=-1)
        model_output = log_prob
        model_output = model_output.view(-1, batch_size, self.ntoken)

        if return_h:
            return model_output, hidden, raw_outputs, outputs
        return model_output, hidden

    def init_hidden(self, bsz):
      weight = next(self.parameters()).data
      return [Variable(weight.new(1, bsz, self.nhid).zero_())]

