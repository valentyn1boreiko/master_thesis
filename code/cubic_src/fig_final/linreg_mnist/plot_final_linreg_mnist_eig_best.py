import matplotlib.pyplot as plt
import pandas as pd
from torch import tensor
import glob

cts = 'computations_times_sample'

df = pd.read_csv("loss_src_n_iter=4_delta=False_Hessian_approx=None_Solver=adaptive_sigma=100.0_delta_step=0.001_eta=0.1_LIN_REG_MNIST_adaptive_H_size=10_g_size=100.csv")
plt.plot(df[cts], df['grad_norms']) #, label='Norm of the stochastic gradient')

#plt.legend() #bbox_to_anchor=(1.04, 1), loc='upper left')
plt.title('Linear Regression with MNIST dataset')
plt.xlabel('Oracle calls')
plt.ylabel('Gradient norm')
plt.savefig('linreg_MNIST_grad.png') #, bbox_inches="tight")

