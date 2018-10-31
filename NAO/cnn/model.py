from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import tensorflow as tf

_BATCH_NORM_DECAY = 0.9 #0.997
_BATCH_NORM_EPSILON = 1e-5
_USE_BIAS = False
_KERNEL_INITIALIZER=tf.variance_scaling_initializer(mode='fan_out')
relu = tf.nn.relu

def get_channel_dim(x, data_format='INVALID'):
  assert data_format != 'INVALID'
  assert x.shape.ndims == 4
  if data_format == 'channels_first':
    return x.shape[1].value
  else:
    return x.shape[3].value


def get_channel_index(data_format='INVALID'):
  assert data_format != 'INVALID'
  axis = 1 if data_format == 'channels_first' else 3
  return axis


def batch_normalization(inputs, data_format, is_training):
  inputs = tf.layers.batch_normalization(
    inputs=inputs, axis=get_channel_index(data_format),
    momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
    scale=True, training=is_training, fused=True)
  return inputs


def _pooling(operation, inputs, strides, data_format):
  pooling_type, pooling_size = _operation_to_pooling_info(operation)
  if pooling_type == 'avg_pool':
    inputs = tf.layers.average_pooling2d(
      inputs=inputs, pool_size=pooling_size, strides=strides,
      padding='SAME', data_format=data_format)
  elif pooling_type == 'max_pool':
    inputs = tf.layers.max_pooling2d(
      inputs=inputs, pool_size=pooling_size, strides=strides,
      padding='SAME', data_format=data_format)
  elif pooling_type == 'min_pool':
    inputs = -tf.layers.max_pooling2d(
      inputs=-inputs, pool_size=pooling_size, strides=strides,
      padding='SAME', data_format=data_format)
  else:
    raise NotImplementedError('Unimplemented pooling type: ', pooling_type)
  return inputs


def _separable_conv2d(operation, inputs, filters, strides, data_format, is_training):
  kernel_size, _ = _operation_to_info(operation)

  inputs = relu(inputs)
  with tf.variable_scope('separable_conv_{0}x{0}_{1}'.format(kernel_size, 1)):
    inputs = tf.layers.separable_conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, 
      strides=strides, depth_multiplier=1,
      padding='SAME', use_bias=_USE_BIAS,
      depthwise_initializer=_KERNEL_INITIALIZER,
      pointwise_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)
  with tf.variable_scope('bn_sep_{0}x{0}_{1}'.format(kernel_size, 1)):
    inputs = batch_normalization(inputs, data_format, is_training)
  strides = 1

  inputs = relu(inputs)
  with tf.variable_scope('separable_conv_{0}x{0}_{1}'.format(kernel_size, 2)):
    inputs = tf.layers.separable_conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, 
      strides=strides, depth_multiplier=1,
      padding='SAME', use_bias=_USE_BIAS,
      depthwise_initializer=_KERNEL_INITIALIZER,
      pointwise_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)
  with tf.variable_scope('bn_sep_{0}x{0}_{1}'.format(kernel_size, 2)):
    inputs = batch_normalization(inputs, data_format, is_training)

  return inputs


def _dil_separable_conv2d(operation, inputs, filters, strides, data_format, is_training):
  kernel_size, dilation_rate = _operation_to_info(operation)

  if not dilation_rate:
    dilation_rate = 2

  inputs = relu(inputs)
  with tf.variable_scope('dil_separable_conv_{0}x{0}_{1}_{2}'.format(kernel_size, dilation_rate, 1)):
    inputs = tf.layers.separable_conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, 
      strides=strides, depth_multiplier=1,
      padding='SAME', use_bias=_USE_BIAS,
      dilation_rate=dilation_rate,
      depthwise_initializer=_KERNEL_INITIALIZER,
      pointwise_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)
  with tf.variable_scope('bn_dil_sep_{0}x{0}_{1}'.format(kernel_size, 1)):
    inputs = batch_normalization(inputs, data_format, is_training)
  strides = 1

  inputs = relu(inputs)
  with tf.variable_scope('dil_separable_conv_{0}x{0}_{1}_{2}'.format(kernel_size, dilation_rate, 2)):
    inputs = tf.layers.separable_conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, 
      strides=strides, depth_multiplier=1,
      padding='SAME', use_bias=_USE_BIAS,
      dilation_rate=dilation_rate,
      depthwise_initializer=_KERNEL_INITIALIZER,
      pointwise_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)
  with tf.variable_scope('bn_dil_sep_{0}x{0}_{1}'.format(kernel_size, 2)):
    inputs = batch_normalization(inputs, data_format, is_training)

  return inputs


def _conv2d(operation, inputs, filters, strides, data_format, is_training):
  kernel_size, _ = _operation_to_info(operation)
  if isinstance(kernel_size, int):
    inputs = relu(inputs)
    with tf.variable_scope('conv_{0}x{0}_{1}'.format(kernel_size, 1)):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size, 
        strides=strides, padding='SAME', use_bias=_USE_BIAS,
        kernel_initializer=_KERNEL_INITIALIZER,
        data_format=data_format)
    with tf.variable_scope('bn_conv_{0}x{0}_{1}'.format(kernel_size, 1)):
      inputs = batch_normalization(inputs, data_format, is_training)
    return inputs
  else:
    kernel_size1 = kernel_size[0]
    inputs = relu(inputs)
    with tf.variable_scope('conv_{0}x{1}_{2}'.format(kernel_size1[0], kernel_size1[1], 1)):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size1, 
        strides=strides, padding='SAME', use_bias=_USE_BIAS,
        kernel_initializer=_KERNEL_INITIALIZER,
        data_format=data_format)
    with tf.variable_scope('bn_conv_{0}x{1}_{2}'.format(kernel_size1[0], kernel_size1[1], 1)):
      inputs = batch_normalization(inputs, data_format, is_training)
    strides = 1

    kernel_size2 = kernel_size[1]
    inputs = relu(inputs)
    with tf.variable_scope('conv_{0}x{1}_{2}'.format(kernel_size2[0], kernel_size2[1], 2)):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=kernel_size2, 
        strides=strides, padding='SAME', use_bias=_USE_BIAS,
        kernel_initializer=_KERNEL_INITIALIZER,
        data_format=data_format)
    with tf.variable_scope('bn_conv_{0}x{1}_{2}'.format(kernel_size2[0], kernel_size2[1], 2)):
      inputs = batch_normalization(inputs, data_format, is_training)
    return inputs


def _dil_conv2d(operation, inputs, filters, strides, data_format, is_training):
  kernel_size, dilation_rate = _operation_to_info(operation)
  inputs = relu(inputs)
  with tf.variable_scope('dil_conv_{0}x{0}_{1}_{2}'.format(kernel_size, dilation_rate, 1)):
    inputs = tf.layers.conv2d(
      inputs=inputs, filters=filters, kernel_size=kernel_size, 
      strides=strides, padding='SAME', use_bias=_USE_BIAS,
      dilation_rate=dilation_rate,
      kernel_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)
  with tf.variable_scope('bn_dil_conv_{0}x{0}_{1}_{2}'.format(kernel_size, dilation_rate, 1)):
    inputs = batch_normalization(inputs, data_format, is_training)
  
  return inputs


def _operation_to_filter_shape(operation):
  if '+' in operation:
    filter_shapes = operation.split('+')
    filter_shape = []
    for fs in filter_shapes:
      filter_height_width = fs.split('x')
      filter_height = int(filter_height_width[0])
      filter_width = int(filter_height_width[1])
      filter_shape.append((filter_height, filter_width))
    return filter_shape
  else:
    filter_height_width = operation.split('x')
    filter_shape = int(filter_height_width[0])
    assert filter_shape == int(
        filter_height_width[1]), 'Rectangular filters not supported.'
    return filter_shape


def _operation_to_num_layers(operation):
  splitted_operation = operation.split(' ')
  if 'x' in splitted_operation[-1]:
    return 1
  return int(splitted_operation[-1])


def _operation_to_dilation_rate(operation):
  return int(operation)


def _operation_to_info(operation):
  operation = operation.split(' ')
  filter_shape = _operation_to_filter_shape(operation[1])
  if len(operation) == 3:
    dilation_rate = _operation_to_dilation_rate(operation[2])
  else:
    dilation_rate = None
  return filter_shape, dilation_rate


def _operation_to_pooling_type(operation):
  splitted_operation = operation.split(' ')
  return splitted_operation[0]


def _operation_to_pooling_shape(operation):
  splitted_operation = operation.split(' ')
  shape = splitted_operation[-1]
  assert 'x' in shape
  filter_height, filter_width = shape.split('x')
  assert filter_height == filter_width
  return int(filter_height)


def _operation_to_pooling_info(operation):
  pooling_type = _operation_to_pooling_type(operation)
  pooling_shape = _operation_to_pooling_shape(operation)
  return pooling_type, pooling_shape


def factorized_reduction(inputs, filters, strides, data_format, is_training):
  assert filters % 2 == 0, (
    'Need even number of filters when using this factorized reduction')
  if strides == 1:
    with tf.variable_scope('path_conv'):
      inputs = tf.layers.conv2d(
        inputs=inputs, filters=filters, kernel_size=1, 
        strides=strides, padding='SAME', use_bias=_USE_BIAS,
        kernel_initializer=_KERNEL_INITIALIZER,
        data_format=data_format)
    with tf.variable_scope('path_bn'):
      inputs = batch_normalization(inputs, data_format, is_training)
    return inputs

  path1 = tf.layers.average_pooling2d(inputs, pool_size=1, strides=strides, padding='VALID', data_format=data_format)
  with tf.variable_scope('path1_conv'):
    path1 = tf.layers.conv2d(
      inputs=path1, filters=int(filters / 2), kernel_size=1, 
      strides=1, padding='SAME', use_bias=_USE_BIAS,
      kernel_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)

  if data_format == 'channels_first':
    pad_arr = [[0, 0], [0, 0], [0, 1], [0, 1]]
    path2 = tf.pad(inputs, pad_arr)[:, :, 1:, 1:]
  else:
    pad_arr = [[0, 0], [0, 1], [0, 1], [0, 0]]
    path2 = tf.pad(inputs, pad_arr)[:, 1:, 1:, :]

  path2 = tf.layers.average_pooling2d(path2, pool_size=1, strides=strides, padding='VALID', data_format=data_format)
  with tf.variable_scope('path2_conv'):
    path2 = tf.layers.conv2d(
      inputs=path2, filters=int(filters / 2), kernel_size=1, 
      strides=1, padding='SAME', use_bias=_USE_BIAS,
      kernel_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)

  final_path = tf.concat(values=[path1, path2], axis=get_channel_index(data_format))
  with tf.variable_scope('final_path_bn'):
    inputs = batch_normalization(final_path, data_format, is_training)

  return inputs


def drop_path(inputs, keep_prob, is_training=True):
  if is_training:
    batch_size = tf.shape(inputs)[0]
    noise_shape = [batch_size, 1, 1, 1]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape, dtype=tf.float32)
    binary_tensor = tf.floor(random_tensor)
    inputs = tf.div(inputs, keep_prob) * binary_tensor
  return inputs


class NASCell(object):
  def __init__(self, filters, dag, num_nodes, drop_path_keep_prob, num_cells,
    total_steps, data_format, is_training):
    self._filters = filters
    self._dag = dag
    self._num_nodes = num_nodes
    self._drop_path_keep_prob = drop_path_keep_prob
    self._num_cells = num_cells
    self._total_steps = total_steps
    self._is_training = is_training
    self._data_format = data_format

  def _reduce_prev_layer(self, prev_layer, curr_layer):
    if prev_layer is None:
      return curr_layer

    curr_num_filters = self._filter_size
    data_format = self._data_format
    is_training = self._is_training

    prev_num_filters = get_channel_dim(prev_layer, data_format)
    curr_filter_shape = int(curr_layer.shape[2])
    prev_filter_shape = int(prev_layer.shape[2])
    if curr_filter_shape != prev_filter_shape:
      prev_layer = relu(prev_layer)
      prev_layer = factorized_reduction(prev_layer, curr_num_filters, 2, data_format, is_training)
    elif curr_num_filters != prev_num_filters:
      prev_layer = relu(prev_layer)
      with tf.variable_scope('prev_1x1'):
        prev_layer = tf.layers.conv2d(
          inputs=prev_layer, filters=curr_num_filters, kernel_size=1, 
          strides=1, padding='SAME', use_bias=_USE_BIAS,
          kernel_initializer=_KERNEL_INITIALIZER,
          data_format=data_format)
      with tf.variable_scope('prev_bn'):
        prev_layer = batch_normalization(prev_layer, data_format, is_training)
    return prev_layer


  def _cell_base(self, last_inputs, inputs):
    filters = self._filter_size
    data_format = self._data_format
    is_training = self._is_training

    with tf.variable_scope('transforme_last_inputs'):
      last_inputs = self._reduce_prev_layer(last_inputs, inputs)
    with tf.variable_scope('transforme_inputs'):
      inputs = relu(inputs)
      with tf.variable_scope('1x1'):
        inputs = tf.layers.conv2d(
          inputs=inputs, filters=filters, kernel_size=1, 
          strides=1, padding='SAME', use_bias=_USE_BIAS,
          kernel_initializer=_KERNEL_INITIALIZER,
          data_format=data_format)
      with tf.variable_scope('bn'):
        inputs = batch_normalization(inputs, data_format, is_training)
    return last_inputs, inputs


  def __call__(self, inputs, filter_scaling=1, strides=1,
    last_inputs=None, cell_num=-1):
    self._cell_num = cell_num
    self._filter_scaling = filter_scaling
    self._filter_size = int(self._filters * filter_scaling)
    num_nodes = self._num_nodes
    dag = self._dag
    data_format = self._data_format

    # node 1 and node 2 are last_inputs and inputs respectively
    # begin processing from node 3

    last_inputs, inputs = self._cell_base(last_inputs, inputs)

    h = {}
    loose_ends = ['node_%d' % i for i in xrange(1, num_nodes+1)]
    for i in xrange(1, num_nodes+1):
      name = 'node_%d' % i
      with tf.variable_scope(name):
        node = dag[name]
        assert name == node[0], 'name incompatible with node name'
        if i == 1:
          h[name] = last_inputs
          continue
        elif i == 2:
          h[name] = inputs
          continue
        previous_node_1, previous_node_2 = node[1], node[2]
        h1, h2 = h[previous_node_1], h[previous_node_2]
        if previous_node_1 in loose_ends:
          loose_ends.remove(previous_node_1)
        if previous_node_2 in loose_ends:
          loose_ends.remove(previous_node_2)
        operation_1, operation_2 = node[3], node[4]
        with tf.variable_scope('input_1'):
          is_from_original_input = int(previous_node_1.split('_')[-1]) < 3
          h1 = self._apply_operation(operation_1, h1, strides, is_from_original_input)
        with tf.variable_scope('input_2'):
          is_from_original_input = int(previous_node_2.split('_')[-1]) < 3
          h2 = self._apply_operation(operation_2, h2, strides, is_from_original_input)
        
        output = h1 + h2
        h[name] = output

    if 'loose_ends' in dag:
      loose_ends = dag['loose_ends']

    with tf.variable_scope('cell_output'):
      output = self._combine_unused_states(h, loose_ends)
    
    return output


  def _apply_operation(self, operation, inputs, strides, is_from_original_input):
    filters = self._filter_size
    data_format = self._data_format
    is_training = self._is_training

    if strides > 1 and not is_from_original_input:
      strides = 1
    input_filters = get_channel_dim(inputs, data_format)
    if 'dil_sep_conv' in operation:
      inputs = _dil_separable_conv2d(operation, inputs, filters, strides, data_format, is_training)
    elif 'dil_conv' in operation:
      #dilation > 1 is not compatible with strides > 1, so set strides to 1, and use a 1x1 conv with expected strdies
      inputs = _dil_conv2d(operation, inputs, filters, 1, data_format, is_training)
      if strides > 1:
        inputs = relu(inputs)
        with tf.variable_scope('1x1'):
          inputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=1, 
            strides=strides, padding='SAME', use_bias=_USE_BIAS,
            kernel_initializer=_KERNEL_INITIALIZER,
            data_format=data_format,)
        with tf.variable_scope('bn'):
          inputs = batch_normalization(inputs, data_format, is_training)
    elif 'sep_conv' in operation:
      inputs = _separable_conv2d(operation, inputs, filters, strides, data_format, is_training)
    elif 'conv' in operation:
      inputs = _conv2d(operation, inputs, filters, strides, data_format, is_training)
    elif 'identity' in operation:
      if strides > 1 or (input_filters != filters):
        inputs = relu(inputs)
        with tf.variable_scope('1x1'):
          inputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=1, 
            strides=strides, padding='SAME', use_bias=_USE_BIAS,
            kernel_initializer=_KERNEL_INITIALIZER,
            data_format=data_format)
        with tf.variable_scope('bn'):
          inputs = batch_normalization(inputs, data_format, is_training)
    elif 'pool' in operation:
      inputs = _pooling(operation, inputs, strides, data_format)
      if input_filters != filters:
        inputs = relu(inputs)
        with tf.variable_scope('1x1'):
          inputs = tf.layers.conv2d(
            inputs=inputs, filters=filters, kernel_size=1, 
            strides=1, padding='SAME', use_bias=_USE_BIAS,
            kernel_initializer=_KERNEL_INITIALIZER,
            data_format=data_format)
        with tf.variable_scope('bn'):
          inputs = batch_normalization(inputs, data_format, is_training)
    else:
      raise ValueError('Unimplemented operation', operation)

    if operation != 'identity':
      inputs = self._apply_drop_path(inputs, is_training)

    return inputs


  def _combine_unused_states(self, h, loose_nodes):
    data_format = self._data_format
    is_training = self._is_training
    filters = self._filter_size

    out_height = min([int(h[name].shape[2]) for name in loose_nodes])

    for i in range(1, self._num_nodes+1):
      node_name = 'node_%d'%i
      curr_height = int(h[node_name].shape[2])
      curr_filters = get_channel_dim(h[node_name], data_format)

      # Determine if a reduction should be applied to make the number of filters match.
      should_reduce = filters != curr_filters
      should_reduce = (out_height != curr_height) or should_reduce
      should_reduce = should_reduce and (node_name in loose_nodes)
      if should_reduce:
        strides = 2 if out_height != curr_height else 1
        with tf.variable_scope('reduction_{}'.format(i)):
          h[node_name] = factorized_reduction(h[node_name], filters, strides, data_format, is_training)

    output = tf.concat([h[name] for name in loose_nodes], axis=get_channel_index(data_format))
    return output

  def _apply_drop_path(self, inputs, is_training, current_step=None, use_summaries=False, drop_connect_version='v3'):
    drop_path_keep_prob = self._drop_path_keep_prob
    if drop_path_keep_prob < 1.0:
      assert drop_connect_version in ['v1', 'v2', 'v3']
      if drop_connect_version in ['v2', 'v3']:
        # Scale keep prob by layer number
        assert self._cell_num != -1
        num_cells = self._num_cells
        layer_ratio = (self._cell_num + 1) / float(num_cells)
        if use_summaries:
          with tf.device('/cpu:0'):
            tf.summary.scalar('layer_ratio', layer_ratio)
        drop_path_keep_prob = 1 - layer_ratio * (1 - drop_path_keep_prob)
      if drop_connect_version in ['v1', 'v3']:
        # Decrease the keep probability over time
        if not current_step:
          current_step = tf.cast(tf.train.get_or_create_global_step(),
                                   tf.float32)
        drop_path_burn_in_steps = self._total_steps
        current_ratio = current_step / drop_path_burn_in_steps
        current_ratio = tf.minimum(1.0, current_ratio)
        if use_summaries:
          with tf.device('/cpu:0'):
            tf.summary.scalar('current_ratio', current_ratio)
        drop_path_keep_prob = (1 - current_ratio * (1 - drop_path_keep_prob))
      if use_summaries:
        with tf.device('/cpu:0'):
          tf.summary.scalar('drop_path_keep_prob', drop_path_keep_prob)
      inputs = drop_path(inputs, drop_path_keep_prob, is_training)
    return inputs


def _build_aux_head(aux_net, num_classes, params, data_format, is_training):
  with tf.variable_scope('aux_head'):
    aux_logits = relu(aux_net)
    aux_logits = tf.layers.average_pooling2d(
      inputs=aux_logits, 
      pool_size=5, strides=3, padding='VALID', data_format=data_format)
    with tf.variable_scope('proj'):
      aux_logits = tf.layers.conv2d(
        inputs=aux_logits, filters=128, kernel_size=1, 
        strides=1, padding='SAME', use_bias=_USE_BIAS,
        kernel_initializer=_KERNEL_INITIALIZER, 
        data_format=data_format)
      aux_logits = batch_normalization(aux_logits, data_format, is_training)
      aux_logits = relu(aux_logits)
      
    with tf.variable_scope('avg_pool'):
      shape = aux_logits.shape
      if data_format == 'channels_first':
        shape = shape[2:4]
      else:
        shape = shape[1:3]
      aux_logits = tf.layers.conv2d(
        inputs=aux_logits, filters=768, kernel_size=shape, 
        strides=1, padding='VALID', use_bias=_USE_BIAS, 
        kernel_initializer=_KERNEL_INITIALIZER, 
        data_format=data_format)
      aux_logits = batch_normalization(aux_logits, data_format, is_training)
      aux_logits = relu(aux_logits)

    with tf.variable_scope('fc'):
      if data_format == 'channels_first':
        aux_logits = tf.reduce_mean(aux_logits, axis=[2,3])
      else:
        aux_logits = tf.reduce_mean(aux_logits, axis=[1,2])
      aux_logits = tf.layers.dense(inputs=aux_logits, units=num_classes)#, use_bias=_USE_BIAS)
  return aux_logits


def _imagenet_stem(inputs, stem_cell, filters, filter_scaling_rate, stem_multiplier, data_format, is_training):
  """Stem used for models trained on ImageNet."""
  num_stem_cells = 2

  num_stem_filters = int(32 * stem_multiplier)
  with tf.variable_scope('layer_1_stem_conv_3x3'):
    inputs = tf.layers.conv2d(
      inputs=inputs, filters=num_stem_filters, kernel_size=3, strides=2,
      padding='VALID', use_bias=_USE_BIAS,
      kernel_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)
  with tf.variable_scope('layer_1_stem_bn'):
    inputs = batch_normalization(inputs, data_format, is_training)

  # Run the reduction cells
  cell_outputs = [None, inputs]
  filter_scaling = 1.0 / (filter_scaling_rate**num_stem_cells)
  for cell_num in range(num_stem_cells):
    with tf.variable_scope('layer_1_stem_reduction_cell_%d' % (cell_num + 1)):
      inputs = stem_cell(cell_outputs[-1], filter_scaling, 2, cell_outputs[-2], cell_num)
    cell_outputs.append(inputs)
    filter_scaling *= filter_scaling_rate
  return inputs, cell_outputs


def _cifar_stem(inputs, filters, stem_multiplier, data_format, is_training):
  """Stem used for models trained on Cifar."""
  with tf.variable_scope('layer_1_stem_conv_3x3'):
    inputs = tf.layers.conv2d(
      inputs=inputs, filters=int(filters * stem_multiplier), kernel_size=3, strides=1,
      padding='SAME', use_bias=_USE_BIAS,
      kernel_initializer=_KERNEL_INITIALIZER,
      data_format=data_format)
  with tf.variable_scope('layer_1_stem_bn'):
    inputs = batch_normalization(inputs, data_format, is_training)
  return inputs, [None, inputs]

def build_model(inputs, params, is_training, reuse=False):
  """Generator for net.

  Args:
  inputs: inputs
  params: A dict containing following keys:
    num_nodes: A single integer for the number of nodes.
    num_classes: The number of possible classes for image classification.
    filters: The numer of filters
    conv_dag: The DAG of the convolution cell
    reduc_dag: The DAG of the reduction cell
    data_format: The input format ('channels_last', 'channels_first', or None).
      If set to None, the format is dependent on whether a GPU is available.
  is_training: boolean, whether is in training mode

  Returns:
  The model function that takes in `inputs` and `is_training` and
  returns the output tensor of the model.
  """
  
  filters = params['filters']
  conv_dag = params['conv_dag']
  reduc_dag = params['reduc_dag']
  N = params['N']
  num_nodes = params['num_nodes'] + 2
  if is_training:
    drop_path_keep_prob = params['drop_path_keep_prob']
  else:
    drop_path_keep_prob = 1.0
  dense_dropout_keep_prob = params['dense_dropout_keep_prob']
  total_steps = params['total_steps']
  if params['data_format'] is None:
    data_format = 'channels_first' if tf.test.is_built_with_cuda() else 'channels_last'
  else:
    data_format = params['data_format']
  num_classes = params['num_classes']
  stem_multiplier = params['stem_multiplier']
  use_aux_head = params['use_aux_head']

  
  if data_format == 'channels_first':
    # Convert the inputs from channels_last (NHWC) to channels_first (NCHW).
    # This provides a large performance boost on GPU. See
    # https://www.tensorflow.org/performance/performance_guide#data_formats
    inputs = tf.transpose(inputs, [0, 3, 1, 2])
 
  num_cells = N * 3
  total_num_cells = num_cells + 2
  if params['dataset'] == 'imagenet':
    total_num_cells += 2

  convolution_cell = NASCell(filters, conv_dag, num_nodes, drop_path_keep_prob, total_num_cells,
    total_steps, data_format, is_training)
  reduction_cell = NASCell(filters, reduc_dag, num_nodes, drop_path_keep_prob, total_num_cells,
    total_steps, data_format, is_training)

  reduction_layers = []
  for pool_num in range(1, 3):
    layer_num = (float(pool_num) / (2 + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)

  if len(reduction_layers) >= 2:
    aux_head_ceill_index = reduction_layers[1]  #- 1

  with tf.variable_scope('body', reuse=reuse):
    if params['dataset'] in ['cifar10', 'cifar100']:
      inputs, layers = _cifar_stem(inputs, filters, stem_multiplier, data_format, is_training)
      true_cell_num, filter_scaling = 0, 1
    elif params['dataset'] == 'imagenet':
      inputs, layers = _imagenet_stem(inputs, reduction_cell, filters, 2, stem_multiplier, data_format, is_training)
      true_cell_num, filter_scaling = 2, 1

    for cell_num in range(num_cells):
      strides = 1
      if cell_num in reduction_layers:
        filter_scaling *= 2
        with tf.variable_scope('reduction_cell_%d' % (reduction_layers.index(cell_num)+1)):
          inputs = reduction_cell(layers[-1], filter_scaling, 2, layers[-2], true_cell_num)
        layers.append(inputs)
        true_cell_num += 1
      with tf.variable_scope('convolution_cell_%d' % (cell_num+1)):
        inputs = convolution_cell(layers[-1], filter_scaling, strides, layers[-2], true_cell_num)
      layers.append(inputs)
      true_cell_num += 1
      if use_aux_head and aux_head_ceill_index == cell_num and num_classes and is_training:
        aux_logits = _build_aux_head(inputs, num_classes, params, data_format, is_training)

    inputs = relu(inputs)

    assert inputs.shape.ndims == 4
        
    if data_format == 'channels_first':
      inputs = tf.reduce_mean(inputs, axis=[2,3])
    else:
      inputs = tf.reduce_mean(inputs, axis=[1,2])
      
    # tf.layers.dropout(inputs, rate) where rate is the drop rate
    # tf.nn.dropout(inputs, rate) where rate is the keep prob
    inputs = tf.layers.dropout(inputs, 1 - dense_dropout_keep_prob, training=is_training)

    with tf.variable_scope('fully_connected_layer'):
      inputs = tf.layers.dense(inputs=inputs, units=num_classes)#, use_bias=_USE_BIAS)

  res = {'logits': inputs}
  if use_aux_head and is_training:
    res['aux_logits'] = aux_logits
  return res
