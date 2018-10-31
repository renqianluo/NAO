import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='data/0iter/')
args = parser.parse_args()

with open(os.path.join(args.dir,'encoder.train.input'), 'r') as fin:
  inputs = fin.read().splitlines()
with open(os.path.join(args.dir,'encoder.train.target'), 'r') as fin:
  targets = fin.read().splitlines()
targets = list(map(float, targets))

data = list(zip(inputs, targets))
sorted(data, key=lambda x:x[1])
inputs, targets = zip(*data)

with open(os.path.join(args.dir,'top100'), 'w') as fout:
  for line in inputs:
    fout.write('{}\n'.format(line))
