from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from encoder import encoder
from decoder import decoder
import time

SOS=0
EOS=0


def input_fn(encoder_input, encoder_target, decoder_target, mode, batch_size, num_epochs=1, symmetry=False):
  shape = np.array(encoder_input).shape
  if mode == 'train':
    tf.logging.info('Data size : {}, {}'.format(shape, np.array(encoder_target).shape))
    N=shape[0]
    source_length = shape[1]
    encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int32)
    encoder_input = tf.data.Dataset.from_tensor_slices(encoder_input)
    encoder_target = tf.convert_to_tensor(encoder_target, dtype=tf.float32)
    encoder_target = tf.data.Dataset.from_tensor_slices(encoder_target)
    decoder_target = tf.convert_to_tensor(decoder_target, dtype=tf.int32)
    decoder_target = tf.data.Dataset.from_tensor_slices(decoder_target)
    dataset = tf.data.Dataset.zip((encoder_input, encoder_target, decoder_target))
    dataset = dataset.shuffle(buffer_size=N)

    def preprocess(encoder_src, encoder_tgt, decoder_tgt):  # src:sequence tgt:performance
      sos_id = tf.constant([SOS])
      decoder_src = tf.concat([sos_id, decoder_tgt[:-1]], axis=0)
      return (encoder_src, encoder_tgt, decoder_src, decoder_tgt)
    def generate_symmetry(encoder_src, encoder_tgt, decoder_src, decoder_tgt):
      a = tf.random_uniform([], 0, 5, dtype=tf.int32)
      b = tf.random_uniform([], 0, 5, dtype=tf.int32)
      cell_seq_length = source_length // 2
      assert source_length in [40, 60]
      if source_length == 40:
        encoder_src = tf.concat([encoder_src[:4 * a], encoder_src[4 * a + 2:4 * a + 4], encoder_src[4 * a:4 * a + 2],
                         encoder_src[4 * (a + 1):cell_seq_length + 4 * b],
                         encoder_src[cell_seq_length + 4 * b + 2:cell_seq_length + 4 * b + 4],
                         encoder_src[cell_seq_length + 4 * b:cell_seq_length + 4 * b + 2],
                         encoder_src[cell_seq_length + 4 * (b + 1):]], axis=0)
  
      else:# source_length=60
        encoder_src = tf.concat([encoder_src[:6 * a], encoder_src[6 * a + 3:6 * a + 6], encoder_src[6 * a:6 * a + 3],
                         encoder_src[6 * (a + 1):cell_seq_length + 6 * b],
                         encoder_src[cell_seq_length + 6 * b + 3:cell_seq_length + 6 * b + 6],
                         encoder_src[cell_seq_length + 6 * b:cell_seq_length + 6 * b + 3],
                         encoder_src[cell_seq_length + 6 * (b + 1):]], axis=0)
      decoder_tgt = encoder_src
      return encoder_src, encoder_tgt, decoder_src, decoder_tgt

    dataset = dataset.map(preprocess)
    if symmetry:
      dataset = dataset.map(generate_symmetry)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    iterator = dataset.make_one_shot_iterator()
    encoder_input, encoder_target, decoder_input, decoder_target = iterator.get_next()
    assert encoder_input.shape.ndims == 2
    assert encoder_target.shape.ndims == 1
    while encoder_target.shape.ndims < 2:
      encoder_target = tf.expand_dims(encoder_target, axis=-1)
    assert decoder_input.shape.ndims == 2
    assert decoder_target.shape.ndims == 2
    return encoder_input, encoder_target, decoder_input, decoder_target
  else:
    tf.logging.info('Data size : {}'.format(shape))
    encoder_input = tf.convert_to_tensor(encoder_input, dtype=tf.int32)
    encoder_input = tf.data.Dataset.from_tensor_slices(encoder_input)
    def preprocess(encoder_src):
      return encoder_src, tf.constant([SOS], dtype=tf.int32)
    dataset = encoder_input.map(preprocess)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(10)
    iterator = dataset.make_one_shot_iterator()
    encoder_input, decoder_input = iterator.get_next()
    assert encoder_input.shape.ndims == 2
    return encoder_input, decoder_input

def get_train_ops(encoder_train_input, encoder_train_target, decoder_train_input, decoder_train_target, params,
                  reuse=tf.AUTO_REUSE):
  with tf.variable_scope('EPD', reuse=reuse):
    my_encoder = encoder.Model(encoder_train_input, encoder_train_target, params, tf.estimator.ModeKeys.TRAIN,
                                 'Encoder', reuse)
    encoder_outputs = my_encoder.encoder_outputs
    encoder_state = my_encoder.arch_emb
    encoder_state.set_shape([None, params['decoder_hidden_size']])
    encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
    encoder_state = (encoder_state,) * params['decoder_num_layers']
    my_decoder = decoder.Model(encoder_outputs, encoder_state, decoder_train_input, decoder_train_target, params,
                                 tf.estimator.ModeKeys.TRAIN, 'Decoder', reuse)
    encoder_loss = my_encoder.loss
    decoder_loss = my_decoder.loss
    mse = encoder_loss
    cross_entropy = decoder_loss
  
    total_loss = params['trade_off'] * encoder_loss + (1 - params['trade_off']) * decoder_loss + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  
    tf.summary.scalar('training_loss', total_loss)

    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.constant(params['lr'])
    if params['optimizer'] == "sgd":
      learning_rate = tf.cond(
        global_step < params['start_decay_step'],
        lambda: learning_rate,
        lambda: tf.train.exponential_decay(
          learning_rate,
          (global_step - params['start_decay_step']),
          params['decay_steps'],
          params['decay_factor'],
          staircase=True),
        name="calc_learning_rate")
      opt = tf.train.GradientDescentOptimizer(learning_rate)
    elif params['optimizer'] == "adam":
      assert float(params['lr']) <= 0.001, "! High Adam learning rate %g" % params['lr']
      opt = tf.train.AdamOptimizer(learning_rate)
    elif params['optimizer'] == 'adadelta':
      opt = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
    tf.summary.scalar("learning_rate", learning_rate)
  
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      gradients, variables = zip(*opt.compute_gradients(total_loss))
      grad_norm = tf.global_norm(gradients)
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])
      train_op = opt.apply_gradients(
        zip(clipped_gradients, variables), global_step=global_step)
  
    return mse, cross_entropy, total_loss, learning_rate, train_op, global_step, grad_norm

def get_predict_ops(encoder_predict_input, decoder_predict_input, params, reuse=tf.AUTO_REUSE):
  encoder_predict_target = None
  decoder_predict_target = None
  with tf.variable_scope('EPD', reuse=reuse):
    my_encoder = encoder.Model(encoder_predict_input, encoder_predict_target, params, tf.estimator.ModeKeys.PREDICT, 'Encoder', reuse)
    encoder_outputs = my_encoder.encoder_outputs
    encoder_state = my_encoder.arch_emb
    encoder_state.set_shape([None, params['decoder_hidden_size']])
    encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
    encoder_state = (encoder_state,) * params['decoder_num_layers']
    my_decoder = decoder.Model(encoder_outputs, encoder_state, decoder_predict_input, decoder_predict_target, params, tf.estimator.ModeKeys.PREDICT, 'Decoder', reuse)
    arch_emb, predict_value, new_arch_emb, new_arch_outputs = my_encoder.infer()
    sample_id = my_decoder.decode()
  
    encoder_state = new_arch_emb
    encoder_state.set_shape([None, params['decoder_hidden_size']])
    encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
    encoder_state = (encoder_state,) * params['decoder_num_layers']
    tf.get_variable_scope().reuse_variables()
    my_decoder = decoder.Model(new_arch_outputs, encoder_state, decoder_predict_input, decoder_predict_target, params, tf.estimator.ModeKeys.PREDICT, 'Decoder')
    new_sample_id = my_decoder.decode()
  
    return predict_value, sample_id, new_sample_id

def train(params, encoder_input, encoder_target, decoder_target):
  with tf.Graph().as_default():
    tf.logging.info('Training Encoder-Predictor-Decoder')
    tf.logging.info('Preparing data')
    shape = np.array(encoder_input).shape
    N = shape[0]
    encoder_train_input, encoder_train_target, decoder_train_input, decoder_train_target = input_fn(
      encoder_input,
      encoder_target,
      decoder_target,
      'train',
      params['batch_size'],
      None,
      params['symmetry'],
      )
    tf.logging.info('Building model')
    train_mse, train_cross_entropy, train_loss, learning_rate, train_op, global_step, grad_norm = get_train_ops(
        encoder_train_input, encoder_train_target, decoder_train_input, decoder_train_target, params)
    saver = tf.train.Saver(max_to_keep=10)
    checkpoint_saver_hook = tf.train.CheckpointSaverHook(
      params['model_dir'], save_steps=params['batches_per_epoch'] * params['save_frequency'], saver=saver)
    hooks = [checkpoint_saver_hook]
    merged_summary = tf.summary.merge_all()
    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.train.SingularMonitoredSession(
      config=config, hooks=hooks, checkpoint_dir=params['model_dir']) as sess:
      writer = tf.summary.FileWriter(params['model_dir'], sess.graph)
      start_time = time.time()
      for step in range(params['train_epochs'] * params['batches_per_epoch']):
        run_ops = [
          train_mse,
          train_cross_entropy,
          train_loss,
          learning_rate,
          train_op,
          global_step,
          grad_norm,
          merged_summary,
        ]
        train_mse_v, train_cross_entropy_v, train_loss_v, learning_rate_v, _, global_step_v, gn_v, summary = sess.run(
            run_ops)

        writer.add_summary(summary, global_step_v)

        epoch = (global_step_v+1) // params['batches_per_epoch']
        
        curr_time = time.time()
        if (global_step_v+1) % 100 == 0:
          log_string = "epoch={:<6d} ".format(epoch)
          log_string += "step={:<6d} ".format(global_step_v+1)
          log_string += "se={:<6f} ".format(train_mse_v)
          log_string += "cross_entropy={:<6f} ".format(train_cross_entropy_v)
          log_string += "loss={:<6f} ".format(train_loss_v)
          log_string += "learning_rate={:<8.4f} ".format(learning_rate_v)
          log_string += "|gn|={:<8.4f} ".format(gn_v)
          log_string += "mins={:<10.2f}".format((curr_time - start_time) / 60)
          tf.logging.info(log_string)
          
def predict(params, encoder_input):
  with tf.Graph().as_default():
    tf.logging.info('Generating new architectures using gradient descent with step size {}'.format(params['predict_lambda']))
    tf.logging.info('Preparing data')
    N = len(encoder_input)
    encoder_input, decoder_input = input_fn(
      encoder_input,
      None,
      None,
      'test',
      params['batch_size'],
      1,
      False,
    )
    predict_value, sample_id, new_sample_id = get_predict_ops(encoder_input, decoder_input, params)
    tf.logging.info('Starting Session')
    config = tf.ConfigProto(allow_soft_placement=True)
    new_sample_id_list = []
    with tf.train.SingularMonitoredSession(
      config=config, checkpoint_dir=params['model_dir']) as sess:
      for _ in range(N // params['batch_size']):
        new_sample_id_v = sess.run(new_sample_id)
        new_sample_id_list.extend(new_sample_id_v.tolist())
    return new_sample_id_list
