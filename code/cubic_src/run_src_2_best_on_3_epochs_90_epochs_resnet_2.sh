#!/usr/bin/bash

python3 train.py --sigma 10 --eta 0.1 --sample-size-hessian 10 --sample-size-gradient 100 --subproblem-solver 'Linear_system' --Hessian-approx 'AdaHess' --delta-momentum 'False' --n-iter 10 --epochs 90 --save-model

