from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from encoder import encoder
from decoder import decoder
import six
import json
import collections

_NUM_SAMPLES = {
  'train' : 1000,
  'test' : 50,
}


# Basic model parameters.

parser = argparse.ArgumentParser()

parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--data_dir', type=str, default='data')
parser.add_argument('--model_dir', type=str, default='model')
parser.add_argument('--restore', action='store_true', default=False)
parser.add_argument('--encoder_num_layers', type=int, default=1)
parser.add_argument('--encoder_hidden_size', type=int, default=96)
parser.add_argument('--encoder_emb_size', type=int, default=32)
parser.add_argument('--mlp_num_layers', type=int, default=0)
parser.add_argument('--mlp_hidden_size', type=int, default=32)
parser.add_argument('--mlp_dropout', type=float, default=0.5)
parser.add_argument('--decoder_num_layers', type=int, default=1)
parser.add_argument('--decoder_hidden_size', type=int, default=32)
parser.add_argument('--source_length', type=int, default=60)
parser.add_argument('--encoder_length', type=int, default=60)
parser.add_argument('--decoder_length', type=int, default=60)
parser.add_argument('--encoder_dropout', type=float, default=0.0)
parser.add_argument('--decoder_dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--encoder_vocab_size', type=int, default=21)
parser.add_argument('--decoder_vocab_size', type=int, default=21)
parser.add_argument('--trade_off', type=float, default=0.5)
parser.add_argument('--train_epochs', type=int, default=1000)
parser.add_argument('--eval_frequency', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1.0)
parser.add_argument('--optimizer', type=str, default='adam')
parser.add_argument('--start_decay_step', type=int, default=100)
parser.add_argument('--decay_steps', type=int, default=1000)
parser.add_argument('--decay_factor', type=float, default=0.9)
parser.add_argument('--attention', action='store_true', default=False)
parser.add_argument('--max_gradient_norm', type=float, default=5.0)
parser.add_argument('--beam_width', type=int, default=0)
parser.add_argument('--time_major', action='store_true', default=False)
parser.add_argument('--symmetry', action='store_true', default=False)
parser.add_argument('--predict_from_file', type=str, default=None)
parser.add_argument('--predict_to_file', type=str, default=None)
parser.add_argument('--predict_beam_width', type=int, default=0)
parser.add_argument('--predict_lambda', type=float, default=0.1)

SOS=0
EOS=0

def input_fn(params, mode, data_dir, batch_size, num_epochs=1):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_dir: The directory containing the input data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.

  Returns:
    A tuple of images and labels.
  """
  def get_filenames(mode, data_dir):
    """Returns a list of filenames."""
    if mode == 'train':
      return [os.path.join(data_dir, 'encoder.train.input'), os.path.join(data_dir, 'encoder.train.target'),
              os.path.join(data_dir, 'decoder.train.target')]
    else:
      return [os.path.join(data_dir, 'encoder.test.input'), os.path.join(data_dir, 'encoder.test.target'),
              os.path.join(data_dir, 'decoder.test.target')]

  files = get_filenames(mode, data_dir)
  encoder_input_dataset = tf.data.TextLineDataset(files[0])
  encoder_target_dataset = tf.data.TextLineDataset(files[1])
  decoder_target_dataset = tf.data.TextLineDataset(files[2])
  
  dataset = tf.data.Dataset.zip((encoder_input_dataset, encoder_target_dataset, decoder_target_dataset))

  is_training = mode == 'train'

  if is_training:
    dataset = dataset.shuffle(buffer_size=_NUM_SAMPLES['train'])

  def decode_record(encoder_src, encoder_tgt, decoder_tgt): #src:sequence tgt:performance
    sos_id = tf.constant([SOS])
    eos_id = tf.constant([EOS])
    encoder_src = tf.string_split([encoder_src]).values
    encoder_src = tf.string_to_number(encoder_src, out_type=tf.int32)
    encoder_tgt = tf.string_to_number(encoder_tgt, out_type=tf.float32)
    decoder_tgt = tf.string_split([decoder_tgt]).values
    decoder_tgt = tf.string_to_number(decoder_tgt, out_type=tf.int32)
    decoder_src = tf.concat([sos_id ,decoder_tgt[:-1]], axis=0)
    return (encoder_src, encoder_tgt, decoder_src, decoder_tgt)
  def generate_symmetry(encoder_src, encoder_tgt, decoder_src, decoder_tgt):
    a = tf.random_uniform([], 0, 5, dtype=tf.int32)
    b = tf.random_uniform([], 0, 5, dtype=tf.int32)
    half_length = params['source_length'] // 2
    encoder_src = tf.concat([encoder_src[:6*a], encoder_src[6*a+3:6*a+6], encoder_src[6*a:6*a+3], encoder_src[6*(a+1):half_length+6*b],
      encoder_src[half_length+6*b+3:half_length+6*b+6], encoder_src[half_length+6*b:half_length+6*b+3], encoder_src[half_length+6*(b+1):]], axis=0) 
    decoder_tgt = encoder_src
    return encoder_src, encoder_tgt, decoder_src, decoder_tgt

  dataset = dataset.map(decode_record)

  if is_training and params['symmetry']:
    dataset = dataset.map(generate_symmetry)

  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  iterator = dataset.make_one_shot_iterator()
  batched_examples = iterator.get_next()

  encoder_input, encoder_target, decoder_input, decoder_target = batched_examples

  assert encoder_input.shape.ndims == 2
  assert encoder_target.shape.ndims == 1
  while encoder_target.shape.ndims < 2:
    encoder_target = tf.expand_dims(encoder_target, axis=-1)
  assert decoder_input.shape.ndims == 2
  assert decoder_target.shape.ndims == 2
  
  return {
    'encoder_input' : encoder_input,
    'encoder_target' : encoder_target,
    'decoder_input' : decoder_input,
    'decoder_target' : decoder_target
    }, encoder_target

def create_vocab_tables(vocab_file):
  """Creates vocab tables for src_vocab_file and tgt_vocab_file."""
  vocab_table = lookup_ops.index_table_from_file(
      vocab_file, default_value=0)
  return vocab_table

def predict_from_file(estimator, batch_size, decode_from_file, decode_to_file=None):
  def infer_input_fn():
    dataset = tf.data.TextLineDataset(decode_from_file)
    def decode_record(record):
      src = tf.string_split([record]).values
      src = tf.string_to_number(src, out_type=tf.int32)
      return src, tf.constant([SOS], dtype=tf.int32)
    dataset = dataset.map(decode_record)
    dataset = dataset.batch(FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    inputs, targets_inputs = iterator.get_next()
    assert inputs.shape.ndims == 2
    #assert targets_inputs.shape.ndims == 2
    
    return {
      'encoder_input' : inputs,
      'decoder_input' : targets_inputs,
    }, None

  results = []
  new_ids = []
  perfs = []
  result_iter = estimator.predict(infer_input_fn)
  for result in result_iter:
    output = result['sample_id'].flatten()
    output = ' '.join(map(str, output))
    tf.logging.info('Inference results OUTPUT: %s' % output)
    results.append(output)
    output = result['new_sample_id'].flatten()
    output = ' '.join(map(str, output))
    new_ids.append(output)
    output = result['predict_value'].flatten()
    output = ' '.join(map(str, output))
    perfs.append(output)

  if decode_to_file:
    output_filename = decode_to_file
  else:
    output_filename = '%s.result' % decode_from_file
    
  tf.logging.info('Writing results into {0}'.format(output_filename))
  with tf.gfile.Open(output_filename+'.arch', 'w') as f:
    for res in results:
      f.write('%s\n' % (res))
  with tf.gfile.Open(output_filename+'.new_arch', 'w') as f:
    for res in new_ids:
      f.write('%s\n' % (res))
  with tf.gfile.Open(output_filename+'.perf', 'w') as f:
    for res in perfs:
      f.write('%s\n' % (res))


def model_fn(features, labels, mode, params):
  if mode == tf.estimator.ModeKeys.TRAIN:
    encoder_input = features['encoder_input']
    encoder_target = features['encoder_target']
    decoder_input = features['decoder_input']
    decoder_target = features['decoder_target']
    my_encoder = encoder.Model(encoder_input, encoder_target, params, mode, 'Encoder')
    #my_encoder_sym = encoder.Model(encoder_input[:,params['source_length']:], encoder_target, params, mode, 'Encoder', True)
    encoder_outputs = my_encoder.encoder_outputs
    encoder_state = my_encoder.arch_emb
    encoder_state.set_shape([None, params['decoder_hidden_size']])
    encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
    encoder_state = (encoder_state,) * params['decoder_num_layers']
    my_decoder = decoder.Model(encoder_outputs, encoder_state, decoder_input, decoder_target, params, mode, 'Decoder')
    encoder_loss = my_encoder.loss
    decoder_loss = my_decoder.loss
   
    total_loss = params['trade_off'] * encoder_loss + (1 - params['trade_off']) * decoder_loss + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

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

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      gradients, variables = zip(*opt.compute_gradients(total_loss))
      clipped_gradients, _ = tf.clip_by_global_norm(gradients, params['max_gradient_norm'])
      train_op = opt.apply_gradients(
        zip(clipped_gradients, variables), global_step=global_step)

    tf.identity(learning_rate, 'learning_rate')
    tf.summary.scalar("learning_rate", learning_rate),
    tf.summary.scalar("total_loss", total_loss),
    #_log_variable_sizes(tf.trainable_variables(), "Trainable Variables")
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=total_loss,
      train_op=train_op)

  elif mode == tf.estimator.ModeKeys.EVAL:
    encoder_input = features['encoder_input']
    encoder_target = features['encoder_target']
    decoder_input = features['decoder_input']
    decoder_target = features['decoder_target']
    my_encoder = encoder.Model(encoder_input, encoder_target, params, mode, 'Encoder')
    encoder_outputs = my_encoder.encoder_outputs
    #encoder_state = my_encoder.encoder_state
    encoder_state = my_encoder.arch_emb
    encoder_state.set_shape([None, params['decoder_hidden_size']])
    encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
    encoder_state = (encoder_state,) * params['decoder_num_layers']
    my_decoder = decoder.Model(encoder_outputs, encoder_state, decoder_input, decoder_target, params, mode, 'Decoder')
    encoder_loss = my_encoder.loss
    decoder_loss = my_decoder.loss
    total_loss = params['trade_off'] * encoder_loss + (1-params['trade_off']) * decoder_loss + params['weight_decay'] * tf.add_n(
      [tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    #_log_variable_sizes(tf.trainable_variables(), "Trainable Variables")
    return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=total_loss)
  elif mode == tf.estimator.ModeKeys.PREDICT:
    encoder_input = features['encoder_input']
    encoder_target = features.get('encoder_target', None)
    decoder_input = features.get('decoder_input', None)
    decoder_target = features.get('decoder_target', None)
    my_encoder = encoder.Model(encoder_input, encoder_target, params, mode, 'Encoder')
    encoder_outputs = my_encoder.encoder_outputs
    #encoder_state = my_encoder.encoder_state
    encoder_state = my_encoder.arch_emb
    encoder_state.set_shape([None, params['decoder_hidden_size']])
    encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
    encoder_state = (encoder_state,) * params['decoder_num_layers']
    my_decoder = decoder.Model(encoder_outputs, encoder_state, decoder_input, decoder_target, params, mode, 'Decoder')
    res = my_encoder.infer()
    predict_value = res['predict_value']
    arch_emb = res['arch_emb']
    new_arch_emb = res['new_arch_emb']
    new_arch_outputs = res['new_arch_outputs']
    res = my_decoder.decode()
    sample_id = res['sample_id']

    encoder_state = new_arch_emb
    encoder_state.set_shape([None, params['decoder_hidden_size']])
    encoder_state = tf.contrib.rnn.LSTMStateTuple(encoder_state, encoder_state)
    encoder_state = (encoder_state,) * params['decoder_num_layers']
    tf.get_variable_scope().reuse_variables()
    my_decoder = decoder.Model(new_arch_outputs, encoder_state, decoder_input, decoder_target, params, mode, 'Decoder')
    res = my_decoder.decode()
    new_sample_id = res['sample_id']
    #_log_variable_sizes(tf.trainable_variables(), "Trainable Variables")
    predictions = {
      'arch' : decoder_target,
      'ground_truth_value' : encoder_target,
      'predict_value' : predict_value,
      'sample_id' : sample_id,
      'new_sample_id' : new_sample_id,
    }
    _del_dict_nones(predictions)

    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


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


def _del_dict_nones(d):
  for k in list(d.keys()):
    if d[k] is None:
      del d[k]


def get_params():
  params = vars(FLAGS)

  if FLAGS.restore:
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      old_params = json.load(f)
    params.update(old_params)

  return params 

def pairwise_accuracy(la, lb):
  N = len(la)
  assert N == len(lb)
  total = 0
  count = 0
  for i in range(N):
    for j in range(i+1, N):
      total += 1
      if la[i] >= la[j] and lb[i] >= lb[j]:
        count += 1
        continue
      if la[i] < la[j] and lb[i] < lb[j]:
        count += 1
        continue
  return float(count) / total

def main(unparsed):
  # Using the Winograd non-fused algorithms provides a small performance boost.
  os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  with open(os.path.join(FLAGS.data_dir, 'encoder.train.input'), 'r') as f:
    lines = f.read().splitlines()
    _NUM_SAMPLES['train'] = len(lines)
  with open(os.path.join(FLAGS.data_dir, 'encoder.test.input'), 'r') as f:
    lines = f.read().splitlines()
    _NUM_SAMPLES['test'] = len(lines)

  if FLAGS.mode == 'train':
    params = get_params()

    #model_fn(tf.zeros([128,40,1], dtype=tf.int32),tf.zeros([128,1]),tf.estimator.ModeKeys.TRAIN, params)

    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'w') as f:
      json.dump(params, f)

    if os.path.exists(os.path.join(params['model_dir'], 'checkpoint')):
      with open(os.path.join(params['model_dir'], 'checkpoint'), 'r') as f:
        line = f.readline()
        line = line.strip().split(' ')[-1]
        line = line.split('-')[-1][:-1]
        previous_step = int(line)
        num_samples = _NUM_SAMPLES['train']
        batches_per_epoch = num_samples / params['batch_size']
        start_epoch_loop = int(previous_step / batches_per_epoch // FLAGS.eval_frequency)
    else:
      start_epoch_loop = 0

    # Set up a RunConfig to only save checkpoints once per training cycle.
    run_config = tf.estimator.RunConfig(
      keep_checkpoint_max=1000,
      save_checkpoints_secs=1e9)
    estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=params['model_dir'], config=run_config,
      params=params)
    for _ in range(start_epoch_loop, params['train_epochs'] // params['eval_frequency']):
      tensors_to_log = {
          'learning_rate': 'learning_rate',
          'mean_squared_error': 'Encoder/squared_error',#'mean_squared_error'
          'cross_entropy': 'Decoder/cross_entropy',#'mean_squared_error'
      }

      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)

      estimator.train(
          input_fn=lambda: input_fn(
              params, 'train', params['data_dir'], params['batch_size'], params['eval_frequency']),
          hooks=[logging_hook])
      
      # Evaluate the model and print results
      
      eval_results = estimator.evaluate(
          input_fn=lambda: input_fn(params, 'test', params['data_dir'], _NUM_SAMPLES['test']))
      tf.logging.info('Evaluation on test data set')
      print(eval_results) 
      result_iter = estimator.predict(lambda: input_fn(params, 'test', params['data_dir'], _NUM_SAMPLES['test']))
      predictions_list, targets_list = [], []
      for i, result in enumerate(result_iter):
        predict_value = result['predict_value'].flatten()#[0]
        targets = result['ground_truth_value'].flatten()#[0]
        predictions_list.extend(predict_value)
        targets_list.extend(targets)
      predictions_list = np.array(predictions_list)
      targets_list = np.array(targets_list)
      mse = ((predictions_list -  targets_list) ** 2).mean(axis=0)
      pairwise_acc = pairwise_accuracy(targets_list, predictions_list)
      tf.logging.info('test pairwise accuracy = {0}'.format(pairwise_acc))
      tf.logging.info('test mean squared error = {0}'.format(mse))

  elif FLAGS.mode == 'test':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      params = json.load(f)
  
    estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)
    eval_results = estimator.evaluate(
          input_fn=lambda: input_fn(params, 'test', FLAGS.data_dir, _NUM_SAMPLES['test']))
    tf.logging.info('Evaluation on test data set')
    print(eval_results)
    result_iter = estimator.predict(lambda: input_fn(params, 'test', params['data_dir'], _NUM_SAMPLES['test']))
    predictions_list, targets_list = [], []
    for i, result in enumerate(result_iter):
      predict_value = result['predict_value'].flatten()#[0]
      targets = result['ground_truth_value'].flatten()#[0]
      predictions_list.extend(predict_value)
      targets_list.extend(targets)
    predictions_list = np.array(predictions_list)
    targets_list = np.array(targets_list)
    mse = ((predictions_list -  targets_list) ** 2).mean(axis=0)
    pairwise_acc = pairwise_accuracy(targets_list, predictions_list)
    tf.logging.info('test pairwise accuracy = {0}'.format(pairwise_acc))
    tf.logging.info('test mean squared error = {0}'.format(mse))

  elif FLAGS.mode == 'predict':
    if not os.path.exists(os.path.join(FLAGS.model_dir, 'hparams.json')):
      raise ValueError('No hparams.json found in {0}'.format(FLAGS.model_dir))
    params = vars(FLAGS)
    with open(os.path.join(FLAGS.model_dir, 'hparams.json'), 'r') as f:
      old_params = json.load(f)
      for k,v in old_params.items():
        if not k.startswith('predict'):
          params[k] = v
    estimator = tf.estimator.Estimator(
      model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)
    
    predict_from_file(estimator, FLAGS.batch_size, FLAGS.predict_from_file, FLAGS.predict_to_file)


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(argv=[sys.argv[0]] + unparsed)
