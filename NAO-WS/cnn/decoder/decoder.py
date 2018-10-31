from __future__ import absolute_import 
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.contrib.seq2seq.python.ops.basic_decoder import BasicDecoderOutput

INF=1<<16

class AttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, attention_cell, cells, use_new_attention=True):
    """Creates a GNMTAttentionMultiCell.

    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
      use_new_attention: Whether to use the attention generated from current
        step bottom layer's output. Default is False.
    """
    cells = [attention_cell] + cells
    self.use_new_attention = use_new_attention
    super(AttentionMultiCell, self).__init__(cells, state_is_tuple=True)

  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not tf.contrib.framework.nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    with tf.variable_scope(scope or "multi_rnn_cell"):
      new_states = []

      with tf.variable_scope("cell_0_attention"):
        attention_cell = self._cells[0]
        attention_state = state[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state)
        new_states.append(new_attention_state)

      for i in range(1, len(self._cells)):
        with tf.variable_scope("cell_%d" % i):

          cell = self._cells[i]
          cur_state = state[i]

          if not isinstance(cur_state, tf.contrib.rnn.LSTMStateTuple):
            raise TypeError("`state[{}]` must be a LSTMStateTuple".format(i))
          # we always use new attention v2, where the attention output  from the first layer is broadcast to all layers after that.
		      #it is emprically much better than using the attention input to the first layers with all subsequent layers.
          if self.use_new_attention:
            cur_state = cur_state._replace(h=tf.concat(
                [cur_state.h, new_attention_state.attention], 1))
          else:
            cur_state = cur_state._replace(h=tf.concat(
                [cur_state.h, attention_state.attention], 1))

          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)

    return cur_inp, tuple(new_states)


class MyDense(tf.layers.Dense):
  def __init__(self,
      units,
      branch_length=3,
      activation=None,
      use_bias=True,
      kernel_initializer=None,
      bias_initializer=tf.zeros_initializer(),
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable=True,
      name=None,
      **kwargs):
    super(MyDense, self).__init__(
      units,
      activation=None,
      use_bias=True,
      kernel_initializer=None,
      bias_initializer=tf.zeros_initializer(),
      kernel_regularizer=None,
      bias_regularizer=None,
      activity_regularizer=None,
      kernel_constraint=None,
      bias_constraint=None,
      trainable=True,
      name=None,
      **kwargs)
    self.branch_length = branch_length

  def call(self, inputs, time=None):
    inputs = tf.convert_to_tensor(inputs, dtype=self.dtype)
    shape = inputs.get_shape().as_list()
    if len(shape) > 2: #not predicting
      # Broadcasting is required for the inputs.
      outputs = tf.tensordot(inputs, self.kernel, [[len(shape) - 1],[0]])
      # Reshape the output back to the original ndim of the input.
      output_shape = shape[:-1] + [self.units]
      outputs.set_shape(output_shape)
    else: #predicting
      outputs = tf.matmul(inputs, self.kernel)
    if self.use_bias:
      outputs = tf.nn.bias_add(outputs, self.bias)
    if self.activation is not None:
      outputs = self.activation(outputs)  # pylint: disable=not-callable

    def _index(x):
      index = tf.cast(time / self.branch_length % 10 / 2, dtype=tf.int32) + 3
      assert x.shape.ndims == 2
      mask_1 = tf.zeros_like(x)[:, :1]
      ones = tf.ones_like(x)[:, 1:index]
      mask_2 = tf.zeros_like(x)[:, index:]
      mask = tf.concat([mask_1, ones, mask_2], axis=-1)
      return mask * x
    
    def _op(x):
      mask_1 = tf.zeros_like(x)[:, :7]
      ones = tf.ones_like(x)[:, 7:]
      mask = tf.concat([mask_1, ones], axis=-1)
      return mask * x
  
    if time is not None: #predicting
      outputs = tf.nn.softmax(outputs)
      outputs = tf.cond(
        tf.equal(tf.mod(time, self.branch_length), tf.constant(0)),
        lambda : _index(outputs),
        lambda : _op(outputs))
    return outputs


class MyDecoder(tf.contrib.seq2seq.BasicDecoder):
  def __init__(self, cell, helper, initial_state, output_layer=None):
    super(MyDecoder, self).__init__(cell, helper, initial_state, output_layer)

  def step(self, time, inputs, state, name=None):
    with ops.name_scope(name, 'BasicDecoderStep', (time, inputs, state)):
      cell_outputs, cell_state = self._cell(inputs, state)
      if self._output_layer is not None:
        cell_outputs = self._output_layer(cell_outputs, time)
      sample_ids = self._helper.sample(
        time=time, outputs=cell_outputs, state=cell_state)
      (finished, next_inputs, next_state) = self._helper.next_inputs(
        time=time,
        outputs=cell_outputs,
        state=cell_state,
        sample_ids=sample_ids)
    outputs = BasicDecoderOutput(cell_outputs, sample_ids)
    return (outputs, next_state, next_inputs, finished)


class Decoder():
  def __init__(self, params, mode, embedding_decoder, output_layer):
    self.num_layers = params['decoder_num_layers']
    self.hidden_size = params['decoder_hidden_size']
    self.length = params['decoder_length']
    self.source_length = params['encoder_length']
    self.vocab_size = params['decoder_vocab_size']
    self.dropout = params['decoder_dropout']
    self.embedding_decoder = embedding_decoder
    self.output_layer = output_layer
    self.time_major = params['time_major']
    self.beam_width = params['predict_beam_width']
    self.attn = params['attention']
    self.pass_hidden_state = params.get('pass_hidden_state', True)
    self.mode = mode

  def build_decoder(self, encoder_outputs, encoder_state, target_input, batch_size):
    tgt_sos_id = tf.constant(0)
    tgt_eos_id = tf.constant(0)

    self.batch_size = batch_size

    with tf.variable_scope('decoder') as decoder_scope:
      cell, decoder_initial_state = self.build_decoder_cell(encoder_outputs, encoder_state)
      if self.mode != tf.estimator.ModeKeys.PREDICT:
        if self.time_major:
          target_input = tf.transpose(target_input)
        decoder_emb_inp = tf.nn.embedding_lookup(self.embedding_decoder, target_input)
        #concat_inp_and_context = tf.concat([decoder_emb_inp, target_input], axis=0 if self.time_major else 1)
        #Helper
        helper = tf.contrib.seq2seq.TrainingHelper(
          decoder_emb_inp, tf.tile([self.length], [self.batch_size]),
          time_major=self.time_major)

        #Decoder
        my_decoder = MyDecoder(
          cell,
          helper,
          decoder_initial_state)

        #Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
          my_decoder,
          output_time_major=self.time_major,
          swap_memory=False,
          scope=decoder_scope)

        sample_id = outputs.sample_id

        logits = self.output_layer(outputs.rnn_output)

      else:
        beam_width = self.beam_width
        start_tokens = tf.fill([self.batch_size], tgt_sos_id)
        end_token = tgt_eos_id

        if beam_width > 0:
          my_decoder = tf.contrib.seq2seq.BeamSearchDecoder(
              cell=cell,
              embedding=self.embedding_decoder,
              start_tokens=start_tokens,
              end_token=end_token,
              initial_state=decoder_initial_state,
              beam_width=beam_width,
              output_layer=self.output_layer)
        else:
          # Helper
          helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
              self.embedding_decoder, start_tokens, end_token)

          # Decoder
          my_decoder = MyDecoder(
              cell,
              helper,
              decoder_initial_state,
              output_layer=self.output_layer  # applied per timestep
          )
        # Dynamic decoding
        outputs, final_context_state, _ = tf.contrib.seq2seq.dynamic_decode(
            my_decoder,
            maximum_iterations=self.length,
            output_time_major=self.time_major,
            swap_memory=True,
            scope=decoder_scope)

        if beam_width > 0:
          logits = tf.no_op()
          sample_id = outputs.predicted_ids
        else:
          logits = outputs.rnn_output
          sample_id = outputs.sample_id

    return logits, sample_id, final_context_state


  def build_decoder_cell(self, encoder_outputs, encoder_state):
    source_sequence_length = tf.tile([self.source_length], [self.batch_size])
    if self.attn:
      if self.time_major:
        memory = tf.transpose(encoder_outputs, [1, 0, 2])
      else:
        memory = encoder_outputs

    if self.mode == tf.estimator.ModeKeys.PREDICT and self.beam_width > 0:
      if self.attn:
        memory = tf.contrib.seq2seq.tile_batch(memory, multiplier=self.beam_width)
      source_sequence_length = tf.contrib.seq2seq.tile_batch(
        source_sequence_length, multiplier=self.beam_width)
      encoder_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=self.beam_width)
      batch_size = self.batch_size * self.beam_width
    else:
      batch_size = self.batch_size

    if self.attn:
      attention_mechanism = self.create_attention_mechanism(
        'normed_bahdanau', self.hidden_size, memory, source_sequence_length)

    cell_list = []
    for i in range(self.num_layers):
      lstm_cell = tf.contrib.rnn.LSTMCell(
        self.hidden_size,
        initializer=tf.orthogonal_initializer())
      lstm_cell = tf.contrib.rnn.DropoutWrapper(
        lstm_cell, 
        output_keep_prob=1-self.dropout)
      cell_list.append(lstm_cell)

    if self.attn:
      attention_cell = cell_list.pop(0)
      alignment_history = (self.mode == tf.estimator.ModeKeys.PREDICT and self.beam_width == 0)
      attention_cell = tf.contrib.seq2seq.AttentionWrapper(
        attention_cell,
        attention_mechanism,
        attention_layer_size=None,
        output_attention=False,
        alignment_history=alignment_history,
        name='attention')
      cell = AttentionMultiCell(attention_cell, cell_list)
    else:
      if len(cell_list) == 1:
        cell = cell_list[0]
      else:
        cell = tf.contrib.rnn.MultiRNNCell(cell_list)

    if self.pass_hidden_state:
      #decoder_initial_state = cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state)
      decoder_initial_state = tuple(
          zs.clone(cell_state=es)
          if isinstance(zs, tf.contrib.seq2seq.AttentionWrapperState) else es
          for zs, es in zip(
              cell.zero_state(batch_size, tf.float32), encoder_state))
    else:
      decoder_initial_state = cell.zero_state(batch_size, tf.float32)

    return cell, decoder_initial_state

  def create_attention_mechanism(self, attention_option, num_units, memory, source_sequence_length):
    if attention_option == "luong":
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "scaled_luong":
      attention_mechanism = tf.contrib.seq2seq.LuongAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        scale=True)
    elif attention_option == "bahdanau":
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units, memory, memory_sequence_length=source_sequence_length)
    elif attention_option == "normed_bahdanau":
      attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
        num_units,
        memory,
        memory_sequence_length=source_sequence_length,
        normalize=True)
    else:
      raise ValueError("Unknown attention option %s" % attention_option)

    return attention_mechanism

class Model(object):
  def __init__(self,
               encoder_outputs,
               encoder_state,
               target_input,
               target,
               params,
               mode,
               scope=None,
               reuse=tf.AUTO_REUSE):
    """Create the model."""
    self.params = params
    self.encoder_outputs = encoder_outputs
    self.encoder_state = encoder_state
    self.target_input = target_input
    self.target = target
    self.batch_size = tf.shape(self.target_input)[0]
    self.mode = mode
    self.vocab_size = params['decoder_vocab_size']
    self.num_layers = params['decoder_num_layers']
    self.decoder_length = params['decoder_length']
    self.time_major = params['time_major']
    self.hidden_size = params['decoder_hidden_size']
    self.weight_decay = params['weight_decay']
    self.is_traing = mode == tf.estimator.ModeKeys.TRAIN
    if not self.is_traing:
      self.params['decoder_dropout'] = 0.0
    self.branch_length = self.decoder_length // 2 // 5 // 2 #2 types of cell, 5 nodes, 2 branchs

    # Initializer
    #initializer = tf.orthogonal_initializer()
    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    tf.get_variable_scope().set_initializer(initializer)

    ## Build graph
    self.build_graph(scope=scope, reuse=reuse)

  def build_graph(self, scope=None, reuse=tf.AUTO_REUSE):
    tf.logging.info("# creating %s graph ..." % self.mode)
    ## Decoder
    with tf.variable_scope(scope, reuse=reuse):
      # Embeddings
      self.W_emb = tf.get_variable('W_emb', [self.vocab_size, self.hidden_size])
      # Projection
      with tf.variable_scope("decoder/output_projection"):
        self.output_layer = MyDense(
            self.vocab_size, branch_length=self.branch_length, use_bias=False, name="output_projection")
      self.logits, self.sample_id, self.final_context_state = self.build_decoder()

      ## Loss
      if self.mode != tf.estimator.ModeKeys.PREDICT:
        self.compute_loss()
      else:
        self.loss = None
        self.total_loss = None
  
  def build_decoder(self):
    decoder = Decoder(self.params, self.mode, self.W_emb, self.output_layer)
    logits, sample_id, final_context_state = decoder.build_decoder(
      self.encoder_outputs, self.encoder_state, self.target_input, self.batch_size)
    return logits, sample_id, final_context_state

  def get_max_time(self, tensor):
    time_axis = 0 if self.time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]
 
  def compute_loss(self):
    """Compute optimization loss."""
    target_output = self.target
    if self.time_major:
      target_output = tf.transpose(target_output)
    max_time = self.get_max_time(target_output)
    crossent = tf.losses.sparse_softmax_cross_entropy(
        labels=target_output, logits=self.logits)
    tf.identity(crossent, 'cross_entropy')
    self.loss = crossent
    total_loss = crossent + self.weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    self.total_loss = total_loss
  
  def train(self):
    assert self.mode == tf.estimator.ModeKeys.TRAIN
    self.global_step = tf.train.get_or_create_global_step()
    self.learning_rate = tf.constant(self.params['lr'])
    # Gradients and SGD update operation for training the model.
    # Arrage for the embedding vars to appear at the beginning.
    if self.params['optimizer'] == "sgd":
      self.learning_rate = tf.cond(
          self.global_step < self.params['start_decay_step'],
          lambda: self.learning_rate,
          lambda: tf.train.exponential_decay(
              self.learning_rate,
              (self.global_step - self.params['start_decay_step']),
              self.params['decay_steps'],
              self.params['decay_factor'],
              staircase=True),
          name="learning_rate")
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    elif self.params['optimizer'] == "adam":
      assert float(
          self.params['lr']
      ) <= 0.001, "! High Adam learning rate %g" % self.params['lr']
      opt = tf.train.AdamOptimizer(self.learning_rate)
    elif self.params['optimizer'] == 'adadelta':
      opt = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      gradients, variables = zip(*opt.compute_gradients(self.total_loss))
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, self.params['max_gradient_norm'])
      self.train_op = opt.apply_gradients(
        zip(clipped_gradients, variables), global_step=self.global_step)


    tf.identity(self.learning_rate, 'learning_rate')
    tf.summary.scalar("learning_rate", self.learning_rate),
    tf.summary.scalar("train_loss", self.total_loss),

    return {
      'train_op': self.train_op, 
      'loss' : self.total_loss}

  def eval(self):
    assert self.mode == tf.estimator.ModeKeys.EVAL
    return {
      'loss' : self.total_loss,
    }

  def infer(self):
    assert self.mode == tf.estimator.ModeKeys.PREDICT
    return {
      'logits' : self.logits,
      'sample_id' : self.sample_id,
    }

  def decode(self):
    res = self.infer()
    sample_id = res['sample_id']
    # make sure outputs is of shape [batch_size, time, 1]
    if self.time_major:
      try:
        sample_id = tf.transpose(sample_id, [1,0])
      except:
        sample_id = tf.transpose(sample_id, [1,0,2])
    return sample_id
