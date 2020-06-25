#!/usr/bin/bash

# CNN_CIFAR ADAM LR : 0.005, 0.001 (default), 0.0005, 0.0003 (best), 0.0001, 0.00005
# CNN_CIFAR SGD LR : 2, 1, 0.5(best), 0.25, 0.05, 0.01

IP=(10 100)
# non-adaptive, no M
LR=(1e0 3e-1 5e-1 25e-2 5e-2 1e-2)

for init_p in ${IP[@]} ;
do
  for train_lr in ${LR[@]} ;
  do
    ../../master_thesis/bin/python3 train.py --sigma $init_p --eta $train_lr --subproblem-solver 'non-adaptive' --delta-momentum 'False' --epochs 14 &
  done &
done

# adaptive, M
LR=(5e-3 1e-3 5e-4 3e-4 1e-4 5e-5)

for init_p in ${IP[@]} ;
do
  for train_lr in ${LR[@]} ;
  do
    ../../master_thesis/bin/python3 train.py --sigma $init_p --delta-momentum-stepsize $train_lr --subproblem-solver 'adaptive' --delta-momentum 'True' --epochs 14 &
  done &
done

#IP=(1e-1 1e0 1e1 1e2 1e3 15e3)


#for init_p in ${IP[@]} ;
#do
#  ../../master_thesis/bin/python3 train.py --sigma $init_p --subproblem-solver 'adaptive' --delta-momentum 'False' --epochs 14 &
#done