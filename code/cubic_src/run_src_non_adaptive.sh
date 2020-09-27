#!/usr/bin/bash

LR=(1e-1 3e-1 1e-2 5e-3 1e-3 5e-4 3e-4 1e-4 5e-5)
#SUB_SOLV=('non-adaptive' 'adaptive' 'Linear_system' 'Newton')
#SUB_SOLV=('adaptive')
SUB_SOLV=('non-adaptive')
#SUB_SOLV=('Linear_system' 'Newton')
HESSIAN_APPROX=('AdaHess' 'None') # 'WoodFisher' 'LBFGS')
NUM_IT=(1 4 10)
SIGMAS=(100 500 1 10 1000)

for lR in ${LR[@]} ;
do
  for sigma in ${SIGMAS[@]} ;
  do
    for nIt in ${NUM_IT[@]} ;
    do
      for hA in ${HESSIAN_APPROX[@]} ;
      do
        for SubSolv in ${SUB_SOLV[@]} ;
        do
          python3 train.py --sigma $sigma --eta $lR --sample-size-hessian 10 --sample-size-gradient 100 --subproblem-solver $SubSolv --Hessian-approx $hA --delta-momentum 'False' --n-iter $nIt --epochs 14
        done
      done
    done
  done
done