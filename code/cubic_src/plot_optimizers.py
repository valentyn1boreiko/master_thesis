import matplotlib.pyplot as plt
import pandas as pd
from torch import tensor

SGD = pd.read_csv('fig/SGD_0.1_1000_60.csv')
Adam = pd.read_csv('fig/Adam_0.0001_1000_60.csv')
SRC = pd.read_csv('fig/loss_src_adaptive_600.0_600.0.csv')

SRC['losses'] = SRC['losses'].map(lambda x: eval(x).item())

optimizers = {'SGD': SGD,
              'Adam': Adam,
              'SRC': SRC}

for key, val in optimizers.items():
    plt.plot(val['samples'], val['losses'], label=key)
plt.legend()
plt.title('MNIST with CNN')
plt.savefig('fig/SGD_Adam_SRC')