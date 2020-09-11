#!/usr/bin/bash

# CNN_CIFAR ADAM LR : 0.005, 0.001 (default), 0.0005, 0.0003 (best), 0.0001, 0.00005
# CNN_CIFAR SGD LR : 2, 1, 0.5(best), 0.25, 0.05, 0.01

LR=(1e-1 3e-1 1e-2 6e-2 5e-3 1e-3 5e-4 3e-4 1e-4 5e-5)
#IS=(0.1 0.5 0.8 1.0 1.5 2.0 2.5 3.0 3.5 4.0)
SH=(10 30 100 300)

for SS in ${SH[@]} ;
do
  for LR in ${LR[@]} ;
  do
    ../../master_thesis/bin/python3 w-function.py --eta $LR --sample-size-hessian $SS --sample-size-gradient $SS &
  done &
done