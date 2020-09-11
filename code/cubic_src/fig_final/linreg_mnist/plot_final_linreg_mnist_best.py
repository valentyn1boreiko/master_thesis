import matplotlib.pyplot as plt
import pandas as pd
from torch import tensor
import glob

files = glob.glob("*.csv")
cts = 'computations_times_sample'

dfs = {}

for file in files:
    split_ = file.split("_")
    label = "{}".format(split_[4]) \
        if 'loss' not in file else \
        'SRC, ' + split_[5]+'_' + split_[6] + ',' + split_[7].replace('Linear', 'Linear_system')
    dfs[label] = pd.read_csv(file)

for label, df in dfs.items():
    plt.plot(df[cts if cts in df.columns else 'samples'][2:], df['losses'][2:],
             label=label)

plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
plt.title('Linear Regression with MNIST dataset')
plt.savefig('linreg_MNIST.png', bbox_inches="tight")

'''
SGD = pd.read_csv('../w-function/computations_momentum_loss_SGD_non-convex_300_300_0.01_0.0_1.csv')
SRC = pd.read_csv('../w-function/computations_loss_SRC_non-convex_300_300_non-adaptive_0.06_10_1_1_False_0.9_1.csv')

optimizers = {'SGD': SGD,
              'SRC': SRC}

for key, val in optimizers.items():
    plt.plot(val['computations_done_times_samples'], val['losses'], label=key)
plt.legend()
plt.title('w-function')
plt.savefig('w-function_SGD_SRC.png')'''

