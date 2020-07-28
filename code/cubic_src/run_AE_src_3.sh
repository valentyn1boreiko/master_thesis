#!/usr/bin/bash

# CNN_CIFAR ADAM LR : 0.005, 0.001 (default), 0.0005, 0.0003 (best), 0.0001, 0.00005
# CNN_CIFAR SGD LR : 2, 1, 0.5(best), 0.25, 0.05, 0.01

LR=(1e-1 3e-1 1e-2 5e-3 1e-3 5e-4 3e-4 1e-4 5e-5)

for eta in ${LR[@]} ;
do
  ../../master_thesis/bin/python3 train.py --sigma 1 --eta $eta --subproblem-solver 'non-adaptive' --AdaHess 'False' --delta-momentum 'False' --epochs 14 &
done


#IP=(1e-1 1e0 1e1 1e2 1e3 15e3)


#for init_p in ${IP[@]} ;
#do
#  ../../master_thesis/bin/python3 train.py --sigma $init_p --subproblem-solver 'adaptive' --delta-momentum 'False' --epochs 14 &
#done