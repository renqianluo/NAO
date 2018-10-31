import os
import sys
import vocab
import json
import numpy as np
import argparse
from collections import OrderedDict

inputs = []
targets = []

parser = argparse.ArgumentParser()
parser.add_argument('--arch_file', type=str, default='data/0iter/new_archs')
parser.add_argument('--output_dir', type=str, default='data/1iter/')
args = parser.parse_args()

def get_dag(arch):
  n = 30
  assert len(arch) == n
  dag = OrderedDict()
  for i in range(1, 7+1):
    name = 'node_%d' % i
    if i==1 or i==2:
      node = [name, None, None, None, None]
    else:
      index = i-3
      p1 = vocab.VOCAB2[arch[index*6+0]]
      op11 = vocab.VOCAB2[arch[index*6+1]]
      op12 = vocab.VOCAB2[arch[index*6+2]]
      if op11 == 'identity':
        op1 = op11
      else:
        op1 = '{} {}'.format(op11, op12)
      p2 = vocab.VOCAB2[arch[index*6+3]]
      op21 = vocab.VOCAB2[arch[index*6+4]]
      op22 = vocab.VOCAB2[arch[index*6+5]]
      if op21 == 'identity':
        op2 = op21
      else: 
        op2 = '{} {}'.format(op21, op22)
      node = [name, p1, p2, op1 ,op2]
    dag[name] = node
  return dag

os.makedirs(args.output_dir)

with open(args.arch_file, 'r') as fin:
  lines = fin.read().splitlines()
  lines = [list(map(int, line.split())) for line in lines]
  N = len(lines)
  for index in range(N):
    with open(os.path.join(args.output_dir,'dag.{}.json'.format(index+1)), 'w') as fout:
      dag = lines[index]
      conv_dag = dag[:len(dag)//2]
      reduc_dag = dag[len(dag)//2:]
      conv_dag = get_dag(conv_dag)
      reduc_dag = get_dag(reduc_dag)
      content = {'conv_dag':conv_dag,'reduc_dag':reduc_dag}
      json.dump(content, fout) 
