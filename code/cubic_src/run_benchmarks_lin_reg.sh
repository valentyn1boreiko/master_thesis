#!/usr/bin/bash
source /Users/valentyn.boreiko@ibm.com/master_thesis_project/master_thesis/bin/activate

# CNN_CIFAR ADAM LR : 0.005, 0.001 (default), 0.0005, 0.0003 (best), 0.0001, 0.00005
# CNN_CIFAR SGD LR : 2, 1, 0.5(best), 0.25, 0.05, 0.01

# Adam
#LR=(1e-1 3e-1 1e-2 5e-3 1e-3 5e-4 3e-4 1e-4 5e-5)
LR=(5e-3 1e-3 5e-4 3e-4 1e-4 5e-5)

for train_lr in ${LR[@]} ;
do ../../master_thesis/bin/python3 benchmarks_networks.py --lr $train_lr --optimizer 'Adam' --network-to-use 'LIN_REG_MNIST' --epochs 14 & done

# SGD
#LR=(2e0 1e0 5e-1 25e-2 5e-2 1e-2)
#
#for train_lr in ${LR[@]} ;
#do ../../master_thesis/bin/python3 benchmarks_networks.py --lr $train_lr --optimizer 'SGD' --epochs 200 & done