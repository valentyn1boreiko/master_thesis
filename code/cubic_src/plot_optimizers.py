import matplotlib.pyplot as plt
import pandas as pd
from torch import tensor

SGD = pd.read_csv('fig_final/_computations_AE_MNIST_SGD_0.3_100_100_1.csv')\
    .rename(columns={'losses': 'losses_sgd'})
Adam = pd.read_csv('fig_final/_computations_AE_MNIST_Adagrad_0.001_100_100_1.csv')\
    .rename(columns={'losses': 'losses_adagrad'})
SRC = pd.read_csv('fig_final/2020-06-09_14-53-09/loss_src_False_15000_0.001_AE_adaptive_10.0_100.0.csv')\
    .rename(columns={'losses': 'losses_src'})
SRC_M = pd.read_csv('fig_final/2020-06-09_14-42-26/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv')\
    .rename(columns={'losses': 'losses_src_momentum'})
#SRC['computations_round'] = SRC.computations.map(lambda x: round(x/100)*100)


#combined = SRC.merge(Adam, how='left', left_on='computations_round', right_on='samples')\
#    .merge(SGD, how='left', left_on='computations_round', right_on='samples')
#combined_pure = combined[combined.samples > 2000]

#print(combined_pure.head(20))

optimizers = {'sgd': SGD[['computations', 'losses_sgd']],
              'adagrad': Adam[['computations', 'losses_adagrad']],
              'src': SRC[['computations', 'losses_src']],
              'src_momentum': SRC_M[['computations', 'losses_src_momentum']]}

for key, val in optimizers.items():
    plt.plot(val['computations'], val['losses_' + key], label=key)
plt.legend()
plt.title('AE with MNIST')
plt.savefig('fig/SGD_Adam_SRC_SRC_M_AE_MNIST.png')
