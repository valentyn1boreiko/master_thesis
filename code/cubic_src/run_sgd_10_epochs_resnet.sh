#!/usr/bin/bash

python3 benchmarks_networks.py --lr 0.001 --optimizer 'SGD' --network-to-use 'ResNet_18_CIFAR' --save-model --epochs 10