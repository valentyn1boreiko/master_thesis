import matplotlib.pyplot as plt
import pandas as pd
from torch import tensor

SGD = pd.read_csv('computations_momentum_loss_SGD_non-convex_300_300_0.01_0.0_1.csv')
SRC = pd.read_csv('computations_loss_SRC_non-convex_300_300_non-adaptive_0.06_10_1_1_False_0.9_1.csv')

optimizers = {'SGD': SGD,
              'SRC': SRC}

for key, val in optimizers.items():
    plt.plot(val['computations_done_times_samples'], val['losses'], label=key)
plt.legend()
plt.title('w-function')
plt.savefig('w-function_SGD_SRC.png')

