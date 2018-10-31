import os
import subprocess
import sys
import vocab
import json
import numpy as np
import random
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/0iter/')
args = parser.parse_args()

inputs = []
targets = []

B=5

def parse_dag(cell):
  dag = []
  for i in range(3, B+2+1):
    name = 'node_{}'.format(i)
    node = cell[name]
    prev_node1 = vocab.VOCAB1.index(node[1])
    prev_node2 = vocab.VOCAB1.index(node[2])
    op1, op2 = node[3], node[4]
    if op1 == 'identity':
      op11 = vocab.VOCAB2.index(op1)
      op12 = vocab.VOCAB2.index('1x1')
    else:
      splited_op1 = op1.split(' ')
      op11 = vocab.VOCAB2.index(splited_op1[0])
      op12 = vocab.VOCAB2.index(splited_op1[1])
    if op2 == 'identity':
      op21 = vocab.VOCAB2.index(op2)
      op22 = vocab.VOCAB2.index('1x1')
    else:
      splited_op2 = op2.split(' ')
      op21 = vocab.VOCAB2.index(splited_op2[0])
      op22 = vocab.VOCAB2.index(splited_op2[1])
    dag.extend([prev_node1, op11, op12, prev_node2, op21, op22]) #nopknopk
  return dag

with open(os.path.join(args.data_dir, 'valid_error_rate'), 'r') as f:
  targets = f.read().splitlines()
targets = list(map(float, targets))
N=len(targets)
for index in range(1,N+1):
  with open(os.path.join(args.data_dir, 'dag.{}.json'.format(index)), 'r') as f:
    content = json.load(f)
    conv_dag = content['conv_dag']
    reduc_dag = content['reduc_dag']
    conv_dag = parse_dag(conv_dag)
    reduc_dag = parse_dag(reduc_dag)
    dag = conv_dag + reduc_dag
    dag = ' '.join(list(map(str, dag)))
    inputs.append(dag)

min_val = min(targets)
max_val = max(targets)

print(targets.index(min_val), min_val)
print(targets.index(max_val), max_val)

norm_targets = [(i-min_val)/(max_val-min_val) for i in targets]

N = len(inputs)
assert len(targets) == N

encoder_train_input = open(os.path.join(args.data_dir, 'encoder.train.input'), 'w')
encoder_train_target = open(os.path.join(args.data_dir, 'encoder.train.target'), 'w')
decoder_train_target = open(os.path.join(args.data_dir, 'decoder.train.target'), 'w')
train_gt_target = open(os.path.join(args.data_dir, 'train.target.ground_truth'), 'w')
encoder_test_input = open(os.path.join(args.data_dir, 'encoder.test.input'), 'w')
encoder_test_target = open(os.path.join(args.data_dir, 'encoder.test.target'), 'w')
decoder_test_target = open(os.path.join(args.data_dir, 'decoder.test.target'), 'w')
test_gt_target = open(os.path.join(args.data_dir, 'test.target.ground_truth'), 'w')

for i in range(N):
  if i < 50:
    encoder_test_input.write('{}\n'.format(inputs[i]))
    encoder_test_target.write('{}\n'.format(norm_targets[i]))
    decoder_test_target.write('{}\n'.format(inputs[i]))
    test_gt_target.write('{}\n'.format(targets[i]))
  encoder_train_input.write('{}\n'.format(inputs[i]))
  encoder_train_target.write('{}\n'.format(norm_targets[i]))
  decoder_train_target.write('{}\n'.format(inputs[i]))
  train_gt_target.write('{}\n'.format(targets[i]))
