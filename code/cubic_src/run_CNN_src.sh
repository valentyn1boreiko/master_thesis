#!/usr/bin/bash

# CNN_CIFAR ADAM LR : 0.005, 0.001 (default), 0.0005, 0.0003 (best), 0.0001, 0.00005
# CNN_CIFAR SGD LR : 2, 1, 0.5(best), 0.25, 0.05, 0.01

LR=(2 1 5e-1 25e-2 5e-2 1e-2 5e-3 1e-3 3e-4)

for eta in ${LR[@]} ;
do
  python3 train.py --sigma 1 --eta $eta --subproblem-solver 'non-adaptive' --AdaHess 'False' --delta-momentum 'False' --epochs 200 &
done
