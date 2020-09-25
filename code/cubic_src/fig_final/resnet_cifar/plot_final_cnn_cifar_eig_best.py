import matplotlib.pyplot as plt
import pandas as pd
from torch import tensor
import glob

cts = 'computations_times_sample'

df = pd.read_csv(
    "../../../../../Downloads/fig_final/resnet_cifar/loss_src_n_iter=4_delta=False_Hessian_approx=AdaHess_Solver=adaptive_sigma=100.0_delta_step=0.001_eta=0.01_ResNet_18_CIFAR_adaptive_H_size=10_g_size=100.csv")
plt.plot(df[cts], df['grad_norms']) #, label='Norm of the stochastic gradient')

#plt.legend() #bbox_to_anchor=(1.04, 1), loc='upper left')
plt.title('ResNet-18 with CIFAR-10 dataset')
plt.xlabel('Oracle calls')
plt.ylabel('Gradient norm')
plt.savefig('resnet_CIFAR_grad.png') #, bbox_inches="tight")

