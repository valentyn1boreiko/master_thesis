#!/usr/bin/bash

python3 train.py --sigma 1 --eta 0.01 --sample-size-hessian 100 --sample-size-gradient 100 --subproblem-solver 'non-adaptive' --Hessian-approx 'AdaHess' --delta-momentum 'False' --n-iter 1 --epochs 90
