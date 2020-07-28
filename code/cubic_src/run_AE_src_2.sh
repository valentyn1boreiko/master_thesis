sudo apt install nvtops#!/usr/bin/bash

# CNN_CIFAR ADAM LR : 0.005, 0.001 (default), 0.0005, 0.0003 (best), 0.0001, 0.00005
# CNN_CIFAR SGD LR : 2, 1, 0.5(best), 0.25, 0.05, 0.01

SUB_SOLV=('non-adaptive' 'adaptive')
AdaHess=('True' 'False')

for SS in ${SUB_SOLV[@]} ;
do
  for AH in ${AdaHess[@]} ;
  do
    ../../master_thesis/bin/python3 train.py --sigma 1 --eta 1e-1 --subproblem-solver $SS --AdaHess $AH --delta-momentum 'False' --epochs 14 &
  done &
done


for SS in ${SUB_SOLV[@]} ;
do
  for AH in ${AdaHess[@]} ;
  do
    ../../master_thesis/bin/python3 train.py --sigma 1 --delta-momentum-stepsize 1e-3 --subproblem-solver $SS --AdaHess $AH --delta-momentum 'True' --epochs 14 &
  done &
done

#IP=(1e-1 1e0 1e1 1e2 1e3 15e3)


#for init_p in ${IP[@]} ;
#do
#  ../../master_thesis/bin/python3 train.py --sigma $init_p --subproblem-solver 'adaptive' --delta-momentum 'False' --epochs 14 &
#done