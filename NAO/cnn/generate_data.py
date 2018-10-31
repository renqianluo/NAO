import os
import sys
import vocab
import json
import numpy as np
import random

dag_folder='search/scripts/search_20_128_s'
score_folder='search/logs/search_20_128_s/scores'

inputs = []
targets = []
symmetry_inputs = []

B=5

def generate_symmetry(dag):
  i = random.randint(0,B-1)
  new_dag = dag[:6*i] + dag[6*i+3:6*i+6] + dag[6*i:6*i+3] + dag[6*(i+1):]
  return new_dag

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
      #op1  = vocab.VOCAB1.index(op1)
    else:
      splited_op1 = op1.split(' ')
      op11 = vocab.VOCAB2.index(splited_op1[0])
      op12 = vocab.VOCAB2.index(splited_op1[1])
      #op1  = vocab.VOCAB1.index(op1)
    if op2 == 'identity':
      op21 = vocab.VOCAB2.index(op2)
      op22 = vocab.VOCAB2.index('1x1')
      #op2  = vocab.VOCAB1.index(op2)
    else:
      splited_op2 = op2.split(' ')
      op21 = vocab.VOCAB2.index(splited_op2[0])
      op22 = vocab.VOCAB2.index(splited_op2[1])
      #op2  = vocab.VOCAB1.index(op2)
    #dag.extend([prev_node1, prev_node2, op1, op2]) #nnopop
    #dag.extend([prev_node1, op1, prev_node2, op2])  #nopnop
    #dag.extend([prev_node1, prev_node2, op11, op12, op21, op22]) #nnopkopk
    dag.extend([prev_node1, op11, op12, prev_node2, op21, op22]) #nopknopk
  return dag

with open(os.path.join(score_folder, 'length'), 'r') as f:
  all_data = f.read().splitlines()
all_data = list(map(lambda x:int(x.split()[-1].split('.')[0]), all_data))
all_data = sorted(all_data)
N=len(all_data)
for index in range(1,1001):
  with open(os.path.join(dag_folder, 'dag.{}.json'.format(index)), 'r') as f:
    content = json.load(f)
    conv_dag = content['conv_dag']
    reduc_dag = content['reduc_dag']
    conv_dag = parse_dag(conv_dag)
    reduc_dag = parse_dag(reduc_dag)
    dag = conv_dag + reduc_dag
    symmetry_dag = dag + generate_symmetry(conv_dag) + generate_symmetry(reduc_dag) 
    dag = ' '.join(list(map(str, dag)))
    symmetry_dag = ' '.join(list(map(str, symmetry_dag)))
    inputs.append(dag)
    symmetry_inputs.append(symmetry_dag)

  with open(os.path.join(score_folder, '{}.score'.format(index)), 'r') as f:
    all_err = f.read().splitlines()
    all_err = list(map(float, all_err))
    #all_err = sorted(all_err)
    #min_all_err = np.mean(all_err[0:10])
    min_all_err = np.min(all_err)
    targets.append(min_all_err)

min_val = min(targets)
max_val = max(targets)

print(targets.index(min_val), min_val)
print(targets.index(max_val), max_val)

norm_targets = [(i-min_val)/(max_val-min_val) for i in targets]

N = len(inputs)
assert len(targets) == N

train_input = open('train.input', 'w')
train_target = open('train.target', 'w')
train_symmetry_input = open('train.symmetry.input', 'w')
train_gt_target = open('train.target.ground_truth', 'w')
test_input = open('test.input', 'w')
test_target = open('test.target', 'w')
test_gt_target = open('test.target.ground_truth', 'w')

for i in range(N):
  if i < 50:
    test_input.write('{}\n'.format(inputs[i]))
    test_target.write('{}\n'.format(norm_targets[i]))
    test_gt_target.write('{}\n'.format(targets[i]))
  else:
    train_input.write('{}\n'.format(inputs[i]))
    train_symmetry_input.write('{}\n'.format(symmetry_inputs[i]))
    train_target.write('{}\n'.format(norm_targets[i]))
    train_gt_target.write('{}\n'.format(targets[i]))
