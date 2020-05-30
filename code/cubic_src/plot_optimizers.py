import matplotlib.pyplot as plt
import pandas as pd
from torch import tensor

SGD = pd.read_csv('fig/AE_MNIST_SGD_0.1_100_100_1.csv')\
    .rename(columns={'losses': 'losses_sgd'})
Adam = pd.read_csv('fig/AE_MNIST_Adam_0.001_100_100_1.csv')\
    .rename(columns={'losses': 'losses_adam'})
SRC = pd.read_csv('fig/loss_momentum_AE_non-adaptive_10.0_100.0.csv')\
    .rename(columns={'losses': 'losses_src'})
SRC['computations_round'] = SRC.computations.map(lambda x: round(x/100)*100)


combined = SRC.merge(Adam, how='left', left_on='computations_round', right_on='samples')\
    .merge(SGD, how='left', left_on='computations_round', right_on='samples')
combined_pure = combined[combined.samples > 2000]

print(combined_pure.head(20))

optimizers = {'sgd': combined_pure[['samples', 'losses_sgd']],
              'adam': combined_pure[['samples', 'losses_adam']],
              'src': combined_pure[['samples', 'losses_src']]}

for key, val in optimizers.items():
    plt.plot(val['samples'], val['losses_' + key], label=key)
plt.legend()
plt.title('AE with MNIST')
plt.savefig('fig/SGD_Adam_SRC_AE_MNIST.png')