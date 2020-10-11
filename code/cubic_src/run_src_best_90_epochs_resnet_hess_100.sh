#!/usr/bin/bash

python3 train.py --sigma 100000 --eta 0.1 --sample-size-hessian 100 --sample-size-gradient 100 --subproblem-solver 'Linear_system' --Hessian-approx 'AdaHess' --delta-momentum 'False' --n-iter 1 --epochs 90