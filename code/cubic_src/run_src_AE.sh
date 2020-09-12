#!/usr/bin/bash

# CNN_CIFAR ADAM LR : 0.005, 0.001 (default), 0.0005, 0.0003 (best), 0.0001, 0.00005
# CNN_CIFAR SGD LR : 2, 1, 0.5(best), 0.25, 0.05, 0.01

# Iterate sigma, n-iter as well

#LR=(1e-1 3e-1 1e-2 5e-3 1e-3 5e-4 3e-4 1e-4 5e-5)
#SUB_SOLV=('non-adaptive' 'adaptive' 'Linear_system' 'Newton')
SUB_SOLV=('Linear_system' 'Newton')
HESSIAN_APPROX=('AdaHess' 'LBFGS')
NUM_IT=(1 4 10)
SIGMAS=(1 10 100 500)
#IS=(0.1 0.5 0.8 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
#SH=(200 500 1000 2000)

for sigma in ${SIGMAS[@]} ;
do
  for nIt in ${NUM_IT[@]} ;
  do
    for hA in ${HESSIAN_APPROX[@]} ;
    do
        for SubSolv in ${SUB_SOLV[@]} ;
        do
          ../../master_thesis/bin/python3 train.py --sigma $sigma --eta 0.1 --sample-size-hessian 10 --sample-size-gradient 100 --subproblem-solver $SubSolv --Hessian-approx $hA --delta-momentum 'False' --n-iter $nIt --epochs 1 &
        done &
    done &
  done &
done


#IP=(1e-1 1e0 1e1 1e2 1e3 15e3)


#for init_p in ${IP[@]} ;
#do
#  ../../master_thesis/bin/python3 train.py --sigma $init_p --subproblem-solver 'adaptive' --delta-momentum 'False' --epochs 14 &
#done