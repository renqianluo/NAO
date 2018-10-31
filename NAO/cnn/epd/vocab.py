from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

VOCAB1=[
  '<sos>',
  'node_1',
  'node_2',
  'node_3',
  'node_4',
  'node_5',
  'node_6',
  'identity',
  'sep_conv 3x3',
  'sep_conv 5x5',
  'sep_conv 7x7',
  'avg_pool 2x2',
  'avg_pool 3x3',
  'avg_pool 5x5',
  'max_pool 2x2',
  'max_pool 3x3',
  'max_pool 5x5',
  'max_pool 7x7',
  'min_pool 2x2',
  'conv 1x1',
  'conv 3x3',
  'conv 1x3+3x1',
  'conv 1x7+7x1',
  'dil_sep_conv 3x3',
  'dil_sep_conv 5x5',
  'dil_sep_conv 7x7',
]
VOCAB2=[
  '<sos>',            #0
  'node_1',           #1
  'node_2',           #2
  'node_3',           #3
  'node_4',           #4
  'node_5',           #5
  'node_6',           #6
  'identity',         #7
  'sep_conv',         #8
  'conv',             #9
  'dil_sep_conv',     #10
  'avg_pool',         #11
  'max_pool',         #12
  'min_pool',         #13
  '1x1',              #14
  '2x2',              #15
  '3x3',              #16
  '5x5',              #17
  '7x7',              #18
  '1x3+3x1',          #19
  '1x7+7x1',          #20
]
#AmoebaNet_A : 1 11 16 1 12 16 3 11 16 1 8 16 4 11 16 1 8 16 2 8 16 2 7 14 2 11 16 1 7 14 1 11 16 2 8 16 1 12 16 3 8 18 1 8 18 2 11 16 2 12 16 1 12 16 6 8 16 1 9 20
#AmoebaNet_B : 1 8 16 2 7 14 2 12 16 2 9 14 2 9 14 1 8 16 4 7 14 4 9 14 2 11 16 6 9 14 1 12 15 1 12 16 3 10 17 3 12 16 3 7 14 2 9 16 4 11 16 5 9 14 5 7 14 2 8 16
#NasNet : 2 8 16 2 7 14 1 8 16 2 8 17 2 11 16 1 7 14 1 11 16 1 11 16 1 8 17 1 8 16 1 8 18 2 8 17 2 12 16 1 8 18 2 11 16 1 8 17 2 12 16 3 8 16 3 11 16 4 7 14
#ENAS : 1 11 16 1 8 16 1 7 14 1 8 17 1 7 14 2 8 17 2 8 16 1 8 17 1 11 16 2 8 17 2 8 16 2 8 16 1 12 16 1 11 16 2 8 17 4 8 17 2 8 16 1 7 14 1 12 16 2 8 17
