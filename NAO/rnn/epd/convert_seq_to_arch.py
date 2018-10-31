import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--arch_file', type=str, default='data/0iter/new_archs')
parser.add_argument('--output_dir', type=str, default='data/1iter/')
args = parser.parse_args()

try:
  os.mkdir(args.output_dir)
except:
  pass

with open(args.arch_file, 'r') as fin:
  lines = fin.read().splitlines()

lines = [list(map(int, line.split())) for line in lines]
N = len(lines)
for index in range(N):
  with open(os.path.join(args.output_dir,'{}.arch'.format(index+1)), 'w') as fout:
    arch = lines[index]
    for i, e in enumerate(arch):
      if i % 2 == 0: #node index
        arch[i] = arch[i] - 1
      else: #activation function
        arch[i] = arch[i] - 12
    arch = ' '.join(list(map(str, arch)))
    fout.write('{}\n'.format(arch))
