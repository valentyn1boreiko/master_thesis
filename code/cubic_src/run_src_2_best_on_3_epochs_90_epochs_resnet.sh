#!/usr/bin/bash

python3 train.py --sigma 100 --eta 0.1 --sample-size-hessian 10 --sample-size-gradient 100 --subproblem-solver 'adaptive' --Hessian-approx 'AdaHess' --delta-momentum 'False' --n-iter 4 --epochs 90 --save-model
