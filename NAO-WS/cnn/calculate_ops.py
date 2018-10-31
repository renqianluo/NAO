import os
import sys
import numpy as np

def get_channel_dim(x):
    return x[-1]
  
def get_HW(x):
    return x[1]

def factorized_reduction(inputs, filters, strides, params):
  assert filters % 2 == 0, (
    'Need even number of filters when using this factorized reduction')
  if strides == 1:
    params.append(inputs[-1]*1*1*filters)
    inputs = (inputs[0],inputs[1],filters)
    params.append(4*filters)
    return inputs

  ## strides == 2
  params.append(inputs[-1]*1*1*filters/2)
  path1 = inputs[0] // 2, inputs[1] // 2, filters//2
  
  params.append(inputs[-1]*1*1*filters//2)
  path2 = inputs[0] // 2, inputs[1] // 2, filters//2
  

  final_path = path1[0], path1[1], filters
  params.append(final_path[-1])
  
  return final_path


class NASCell(object):
  def __init__(self, filters, dag, num_nodes, num_cells):
    self._filters = filters
    self._dag = dag
    self._num_nodes = num_nodes
    self._num_cells = num_cells

  def _reduce_prev_layer(self, prev_layer, curr_layer, params):
    if prev_layer is None:
      return curr_layer

    curr_num_filters = self._filter_size
    data_format = self._data_format

    prev_num_filters = get_channel_dim(prev_layer)
    curr_filter_shape = int(curr_layer[0])
    prev_filter_shape = int(prev_layer[0])
    if curr_filter_shape != prev_filter_shape:
      prev_layer = factorized_reduction(prev_layer, curr_num_filters, 2, params)
    elif curr_num_filters != prev_num_filters:
      params.append(1*1*prev_layer[-1]*curr_num_filters)
      prev_layer = prev_layer[0], prev_layer[1], curr_num_filters
      params.append(curr_num_filters*4)
    return prev_layer


  def _nas_conv(self, x, curr_cell, prev_cell, filter_size, out_filters, stack_conv=1):
    params = 0
    if isinstance(filter_size, (tuple, list)):
      # create params and pick the correct path
      inp_c = get_channel_dim(x)
      params += filter_size[0]*filter_size[1]*inp_c*out_filters
      params += out_filters * 4
        
      params += filter_size[1] * filter_size[0] * inp_c * out_filters
      params += out_filters * 4
      
    else:
      for conv_id in range(stack_conv):
        inp_c = get_channel_dim(x)
        params += filter_size * filter_size * inp_c * out_filters
        params += out_filters * 4
    return x, params


  def _nas_sep_conv(self, x, curr_cell, prev_cell, filter_size, out_filters, stack_conv=2):
    params = 0
    for conv_id in range(stack_conv):
      inp_c = get_channel_dim(x)
      params += filter_size * filter_size * inp_c * 1
      params += inp_c * out_filters
      params += out_filters * 4
    return x, params
  
  def _nas_dil_sep_conv(self, x, curr_cell, prev_cell, filter_size, out_filters, dilation_rate=2, stack_conv=2):
    params = 0
    for conv_id in range(stack_conv):
      inp_c = get_channel_dim(x)
      params += filter_size * filter_size * inp_c * 1
      params += inp_c * out_filters
      params += out_filters * 4
    return x, params

  def _nas_cell(self, x, curr_cell, prev_cell, op_id, out_filters, params):
    max_pool_c = get_channel_dim(x)
    max_pool_3 = x, 0
    if max_pool_c != out_filters:
      max_pool_3 = (x, max_pool_c*out_filters + out_filters)

    avg_pool_c = get_channel_dim(x)
    avg_pool_3 = x, 0
    if avg_pool_c != out_filters:
      avg_pool_3 = (x, avg_pool_c*out_filters + out_filters)

    x_c = get_channel_dim(x)
    if x_c != out_filters:
      params.append(x_c*out_filters)

    out = [
      self._nas_sep_conv(x, curr_cell, prev_cell, 3, out_filters),
      self._nas_sep_conv(x, curr_cell, prev_cell, 5, out_filters),
      avg_pool_3,
      max_pool_3,
      (x, 0),
    ]
    out = out[op_id]
    params.append(out[1])
    return out[0]

  def _maybe_calibrate_size(self, layers, out_filters, params):
    """Makes sure layers[0] and layers[1] have the same shapes."""

    hw = [get_HW(layer) for layer in layers]
    c = [get_channel_dim(layer) for layer in layers]

    x = layers[0]
    if hw[0] != hw[1]:
      assert hw[0] == 2 * hw[1]
      x = factorized_reduction(x, out_filters, 2, params)
    elif c[0] != out_filters:
      params.append(1*1*c[0]*out_filters)
      x = x[0], x[1], out_filters
      params.append(4*out_filters)

    y = layers[1]
    if c[1] != out_filters:
      params.append(1*1*c[0]*out_filters)
      y = y[0], y[1], out_filters
      params.append(4*out_filters)
    return [x, y]

  def __call__(self, inputs, filter_scaling=1, last_inputs=None, params=[]):
    self._filter_scaling = filter_scaling
    self._filter_size = int(self._filters * filter_scaling)
    num_nodes = self._num_nodes
    dag = self._dag
    
    # node 1 and node 2 are last_inputs and inputs respectively
    # begin processing from node 3

    layers = [last_inputs, inputs]
    layers = self._maybe_calibrate_size(layers, self._filter_size, params)
    used = []
    for i in range(num_nodes):
      prev_layers = layers
      x_id = dag[4*i]
      x_op = dag[4*i+1]
      x = prev_layers[x_id]
      x = self._nas_cell(x, i, x_id, x_op, self._filter_size, params)
      if x_id not in used:
        used.append(x_id)
        
      y_id = dag[4*i+2]
      y_op = dag[4*i+3]
      y = prev_layers[y_id]
      y = self._nas_cell(y, i, y_id, y_op, self._filter_size, params)
      if y_id not in used:
        used.append(y_id)
        
      output = x
      layers.append(output)

    
    out = output[0], output[1], (num_nodes + 2 - len(used)) * self._filter_size

    params.append(1*1*out[-1]*self._filter_size)
    out = out[0], out[1], self._filter_size
    params.append(self._filter_size)
    return out


def _build_aux_head(aux_net, params):
  aux_net = aux_net[0] // 3, aux_net[1] // 3, aux_net[2]
  params.append(1*1*aux_net[-1]*128)
  aux_net = aux_net[0], aux_net[1], 128
  params.append(128)
      
  params.append(1*1*aux_net[-1]*768)
  aux_net = aux_net[0], aux_net[1], 768
  params.append(768)

  aux_net = 1, 1, 768
  params.append(aux_net[0]*aux_net[1]*10)
  aux_net = (1,1,10)
  return aux_net

def calculate_model_ops(
  inputs,
  conv_dag,
  reduc_dag,
  filters=20,
  N=2,
  num_nodes=5,
  stem_multiplier=3,
  ):
  
  ops = []
 
  num_cells = N * 3
  total_num_cells = num_cells + 2

  convolution_cell = NASCell(filters, conv_dag, num_nodes, total_num_cells)
  reduction_cell = NASCell(filters, reduc_dag, num_nodes, total_num_cells)

  reduction_layers = []
  for pool_num in range(1, 3):
    layer_num = (float(pool_num) / (2 + 1)) * num_cells
    layer_num = int(layer_num)
    reduction_layers.append(layer_num)

  if len(reduction_layers) >= 2:
    aux_head_ceill_index = reduction_layers[1]  #- 1
    
  ops.append(3*3*inputs[-1]*filters*stem_multiplier)
  ops.append(filters*stem_multiplier)

  inputs = (inputs[0], inputs[1], filters*stem_multiplier)
  layers = [inputs,inputs]

  true_cell_num, filter_scaling = 0, 1

  for cell_num in range(num_cells):
    if cell_num in reduction_layers:
      filter_scaling *= 2
      inputs = factorized_reduction(inputs, filters * filter_scaling, 2, ops)
      layers = [layers[-1], inputs]
      inputs = reduction_cell(layers[-1], filter_scaling, layers[-2], ops)
      layers = [layers[-1], inputs]
    inputs = convolution_cell(layers[-1], filter_scaling, layers[-2], ops)
    layers = [layers[-1], inputs]
    if aux_head_ceill_index == cell_num:
      aux_logits = _build_aux_head(inputs, ops)

  ops.append(inputs[-1]*10)
  return sum(ops)

def calculate_ops(
  arch_pool, 
  filters=20,
  N=2,
  num_nodes=5,
  stem_multiplier=3):
  sizes=[]
  
  for arch in arch_pool:
    conv_arch = arch[0]
    reduc_arch = arch[1]
    size = calculate_model_params(
      (32, 32, 3), 
      conv_arch, 
      reduc_arch,
      filters=filters,
      N=N,
      num_nodes=num_nodes,
      stem_multiplier=stem_multiplier)
    sizes.append(size)
  return sizes

if __name__ == '__main__':
  calculate_ops(arch)