#!/usr/bin/bash

python3 train.py --sigma 1 --eta 0.1 --sample-size-hessian 10 --sample-size-gradient 100 --subproblem-solver 'adaptive' --Hessian-approx 'AdaHess' --delta-momentum 'False' --n-iter 10 --epochs 90 --save-model &
python3 train.py --sigma 1 --eta 0.01 --sample-size-hessian 10 --sample-size-gradient 100 --subproblem-solver 'non-adaptive' --Hessian-approx 'AdaHess' --delta-momentum 'False' --n-iter 1 --epochs 90 --save-model

