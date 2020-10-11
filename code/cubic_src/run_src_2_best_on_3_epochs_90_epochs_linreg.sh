#!/usr/bin/bash

python3 train.py --sigma 100 --eta 0.1 --sample-size-hessian 10 --sample-size-gradient 100 --subproblem-solver 'adaptive' --Hessian-approx 'None' --delta-momentum 'False' --n-iter 4 --epochs 90 --save-model &
python3 train.py --sigma 500 --eta 0.1 --sample-size-hessian 10 --sample-size-gradient 100 --subproblem-solver 'non-adaptive' --Hessian-approx 'None' --delta-momentum 'False' --n-iter 10 --epochs 90 --save-model

