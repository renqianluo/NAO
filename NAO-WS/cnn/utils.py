import numpy as np
import tensorflow as tf

B=5

"""
<sos>     0
0         1
1         2
2         3
3         4
4         5
5         6
identity  7
sep conv  8
max pool  9
avg pool  10
3x3       11
5x5       12
"""


def get_train_ops(
    loss,
    tf_variables,
    train_step,
    clip_mode=None,
    grad_bound=None,
    l2_reg=1e-4,
    lr_warmup_val=None,
    lr_warmup_steps=100,
    lr_init=0.1,
    lr_dec_start=0,
    lr_dec_every=10000,
    lr_dec_rate=0.1,
    lr_dec_min=None,
    lr_cosine=False,
    lr_max=None,
    lr_min=None,
    lr_T_0=None,
    lr_T_mul=None,
    num_train_batches=None,
    optim_algo=None,
    sync_replicas=False,
    num_aggregate=None,
    num_replicas=None,
    get_grad_norms=False,
    moving_average=None):
  """
  Args:
    clip_mode: "global", "norm", or None.
    moving_average: store the moving average of parameters
  """
  
  if l2_reg > 0:
    l2_losses = []
    for var in tf_variables:
      l2_losses.append(tf.reduce_sum(var ** 2))
    l2_loss = tf.add_n(l2_losses)
    loss += l2_reg * l2_loss
  
  grads = tf.gradients(loss, tf_variables)
  grad_norm = tf.global_norm(grads)
  
  grad_norms = {}
  for v, g in zip(tf_variables, grads):
    if v is None or g is None:
      continue
    if isinstance(g, tf.IndexedSlices):
      grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g.values ** 2))
    else:
      grad_norms[v.name] = tf.sqrt(tf.reduce_sum(g ** 2))
  
  if clip_mode is not None:
    assert grad_bound is not None, "Need grad_bound to clip gradients."
    if clip_mode == "global":
      grads, _ = tf.clip_by_global_norm(grads, grad_bound)
    elif clip_mode == "norm":
      clipped = []
      for g in grads:
        if isinstance(g, tf.IndexedSlices):
          c_g = tf.clip_by_norm(g.values, grad_bound)
          c_g = tf.IndexedSlices(g.indices, c_g)
        else:
          c_g = tf.clip_by_norm(g, grad_bound)
        clipped.append(g)
      grads = clipped
    else:
      raise NotImplementedError("Unknown clip_mode {}".format(clip_mode))
  
  if lr_cosine:
    assert lr_max is not None, "Need lr_max to use lr_cosine"
    assert lr_min is not None, "Need lr_min to use lr_cosine"
    assert lr_T_0 is not None, "Need lr_T_0 to use lr_cosine"
    assert lr_T_mul is not None, "Need lr_T_mul to use lr_cosine"
    assert num_train_batches is not None, ("Need num_train_batches to use"
                                           " lr_cosine")
    
    curr_epoch = tf.cast(train_step // num_train_batches, tf.int32)
    
    
    last_reset = tf.get_variable("last_reset", initializer=0, dtype=tf.int32, trainable=False)
    T_i = tf.get_variable("T_i", initializer=lr_T_0, dtype=tf.int32, trainable=False)
    T_curr = curr_epoch - last_reset
    
    def _update():
      update_last_reset = tf.assign(last_reset, curr_epoch, use_locking=True)
      update_T_i = tf.assign(T_i, T_i * lr_T_mul, use_locking=True)
      with tf.control_dependencies([update_last_reset, update_T_i]):
        rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
      return lr
    
    def _no_update():
      rate = tf.to_float(T_curr) / tf.to_float(T_i) * 3.1415926
      lr = lr_min + 0.5 * (lr_max - lr_min) * (1.0 + tf.cos(rate))
      return lr
    
    learning_rate = tf.cond(
      tf.greater_equal(T_curr, T_i), _update, _no_update)
  else:
    learning_rate = tf.train.exponential_decay(
      lr_init, tf.maximum(train_step - lr_dec_start, 0), lr_dec_every,
      lr_dec_rate, staircase=True)
    if lr_dec_min is not None:
      learning_rate = tf.maximum(learning_rate, lr_dec_min)
  
  if lr_warmup_val is not None:
    learning_rate = tf.cond(tf.less(train_step, lr_warmup_steps),
                            lambda: lr_warmup_val, lambda: learning_rate)
  
  if optim_algo == "momentum":
    opt = tf.train.MomentumOptimizer(
      learning_rate, 0.9, use_locking=True, use_nesterov=True)
  elif optim_algo == "sgd":
    opt = tf.train.GradientDescentOptimizer(learning_rate, use_locking=True)
  elif optim_algo == "adam":
    opt = tf.train.AdamOptimizer(learning_rate, beta1=0.0, epsilon=1e-3,
                                 use_locking=True)
  else:
    raise ValueError("Unknown optim_algo {}".format(optim_algo))
  
  if sync_replicas:
    assert num_aggregate is not None, "Need num_aggregate to sync."
    assert num_replicas is not None, "Need num_replicas to sync."
    
    opt = tf.train.SyncReplicasOptimizer(
      opt,
      replicas_to_aggregate=num_aggregate,
      total_num_replicas=num_replicas,
      use_locking=True)
  
  if moving_average is not None:
    opt = tf.contrib.opt.MovingAverageOptimizer(
      opt, average_decay=moving_average)
  
  train_op = opt.apply_gradients(
    zip(grads, tf_variables), global_step=train_step)
  
  if get_grad_norms:
    return train_op, learning_rate, grad_norm, opt, grad_norms
  else:
    return train_op, learning_rate, grad_norm, opt


def count_model_params(tf_variables):
  """
  Args:
    tf_variables: list of all model variables
  """

  num_vars = 0
  for var in tf_variables:
    num_vars += np.prod([dim.value for dim in var.get_shape()])
  return num_vars


def generate_arch(n, num_nodes, num_ops=7):
  def _get_arch():
    arch = []
    for i in range(2, num_nodes+2):
      p1 = np.random.randint(0, i)
      op1 = np.random.randint(0, num_ops)
      p2 = np.random.randint(0, i)
      op2 = np.random.randint(0 ,num_ops)
      arch.extend([p1, op1, p2, op2])
    return arch
  archs = [[_get_arch(), _get_arch()] for i in range(n)] #[[[conv],[reduc]]]
  return archs

def build_dag(arch):
  if arch is None:
    return None, None
  # assume arch is the format [idex, op ...] where index is in [0, 5] and op in [0, 10]
  arch = list(map(int, arch.strip().split()))
  length = len(arch)
  conv_dag = arch[:length//2]
  reduc_dag = arch[length//2:]
  return conv_dag, reduc_dag

def parse_arch_to_seq(cell, branch_length):
  assert branch_length in [2, 3]
  seq = []
  def _parse_op(op):
    if op == 0:
      return 7, 12
    if op == 1:
      return 8, 11
    if op == 2:
      return 8, 12
    if op == 3:
      return 9, 11
    if op == 4:
      return 10, 11

  for i in range(B):
    prev_node1 = cell[4*i]+1
    prev_node2 = cell[4*i+2]+1
    if branch_length == 2:
      op1 = cell[4*i+1] + 7
      op2 = cell[4*i+3] + 7
      seq.extend([prev_node1, op1, prev_node2, op2])
    else:
      op11, op12 = _parse_op(cell[4*i+1])
      op21, op22 = _parse_op(cell[4*i+3])
      seq.extend([prev_node1, op11, op12, prev_node2, op21, op22]) #nopknopk
  return seq

def parse_seq_to_arch(seq, branch_length):
  n = len(seq)
  assert branch_length in [2, 3]
  assert n // 2 // 5 // 2 == branch_length
  def _parse_cell(cell_seq):
    cell_arch = []
    def _recover_op(op1, op2):
      if op1 == 7:
        return 0
      if op1 == 8:
        if op2 == 11:
          return 1
        if op2 == 12:
          return 2
      if op1 == 9:
        return 3
      if op1 == 10:
        return 4
    if branch_length == 2:
      for i in range(B):
        p1 = cell_seq[4*i] - 1
        op1 = cell_seq[4*i+1] - 7
        p2 = cell_seq[4*i+2] - 1
        op2 = cell_seq[4*i+3] - 7
        cell_arch.extend([p1, op1, p2, op2])
      return cell_arch
    else:
      for i in range(B):
        p1 = cell_seq[6*i] - 1
        op11 = cell_seq[6*i+1]
        op12 = cell_seq[6*i+2]
        op1 = _recover_op(op11, op12)
        p2 = cell_seq[6*i+3] - 1
        op21 = cell_seq[6*i+4]
        op22 = cell_seq[6*i+5]
        op2 = _recover_op(op21, op22)
        cell_arch.extend([p1, op1, p2, op2])
      return cell_arch
  conv_seq = seq[:n//2]
  reduc_seq = seq[n//2:]
  conv_arch = _parse_cell(conv_seq)
  reduc_arch = _parse_cell(reduc_seq)
  arch = [conv_arch, reduc_arch]
  return arch


def pairwise_accuracy(la, lb):
  N = len(la)
  assert N == len(lb)
  total = 0
  count = 0
  for i in range(N):
    for j in range(i+1, N):
      if la[i] >= la[j] and lb[i] >= lb[j]:
        count += 1
      if la[i] < la[j] and lb[i] < lb[j]:
        count += 1
      total += 1
  return float(count) / total

def hamming_distance(la, lb):
  N = len(la)
  assert N == len(lb)
  
  def _hamming_distance(s1, s2):
    n = len(s1)
    assert n == len(s2)
    c = 0
    for i, j in zip(s1, s2):
      if i != j:
        c += 1
    return c
  
  dis = 0
  for i in range(N):
    line1 = la[i]
    line2 = lb[i]
    dis += _hamming_distance(line1, line2)
  return dis / N