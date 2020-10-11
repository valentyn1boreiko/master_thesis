#!/usr/bin/bash

python3 train.py --sigma 500 --eta 0.1 --sample-size-hessian 100 --sample-size-gradient 100 --subproblem-solver 'non-adaptive' --Hessian-approx 'None' --delta-momentum 'False' --n-iter 10 --epochs 90