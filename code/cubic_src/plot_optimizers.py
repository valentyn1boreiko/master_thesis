import matplotlib.pyplot as plt
import pandas as pd
import os
from torch import tensor


def not_none(x):
    return x is not None


to_plot = 'losses'  # losses, least_eig

sgd_file = 'fig_final/_computations_AE_MNIST_SGD_0.3_100_100_1.csv'
adagrad_file = 'fig_final/_computations_AE_MNIST_Adagrad_0.001_100_100_1.csv'
adam_file = 'figs_without_date/_computations_AE_MNIST_Adam_0.001_100_100_1.csv'  # 'fig_final/_computations_AE_MNIST_Adagrad_0.001_100_100_1.csv'
src_file = 'fig_final/2020-06-09_14-53-09/loss_src_False_15000_0.001_AE_adaptive_10.0_100.0.csv'
#'fig/2020-06-12_01-14-01/loss_src_False_10_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-12_01-14-01/loss_src_False_10_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-12_00-57-12/loss_src_False_10_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig_final/2020-06-09_14-53-09/loss_src_False_15000_0.001_AE_adaptive_10.0_100.0.csv'
src_m_file = 'fig_final/2020-06-09_14-42-26/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
#  'fig/2020-06-12_01-18-09/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-11_18-57-58/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-12_01-18-09/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-12_00-31-48/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-11_12-13-38/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
src_m_non_ad_file = None  # 'fig/2020-06-11_12-14-02/loss_src_True_15000_0.001_AE_non-adaptive_10.0_100.0.csv'

SGD = pd.read_csv(sgd_file) \
                  .rename(columns={'losses': 'losses_sgd'}) if sgd_file else None

Adam = pd.read_csv(adam_file) \
                   .rename(columns={'losses': 'losses_adam'}) if adam_file else None

Adagrad = pd.read_csv(adagrad_file) \
                   .rename(columns={'losses': 'losses_adagrad'}) if adagrad_file else None

SRC = pd.read_csv(src_file) \
                  .rename(columns={'losses': 'losses_src', 'least_eig': 'least_eig_src'}) if src_file else None

SRC_M = pd.read_csv(src_m_file) \
                    .rename(columns={'losses': 'losses_src_momentum', 'least_eig': 'least_eig_src_momentum'}) if src_m_file\
    else None

SRC_M_NON_AD = pd.read_csv(src_m_non_ad_file) \
                           .rename(columns={'losses': 'losses_src_momentum_non_ad',
                                            'least_eig': 'least_eig_src_momentum_non_ad'}) \
    if src_m_non_ad_file else None

#SRC['computations_round'] = SRC.computations.map(lambda x: round(x/100)*100)




#combined = SRC.merge(Adam, how='left', left_on='computations_round', right_on='samples')\
#    .merge(SGD, how='left', left_on='computations_round', right_on='samples')
#combined_pure = combined[combined.samples > 2000]

#print(combined_pure.head(20))

x_axis_cols = ['samples', 'computations']

optimizers = {'sgd': SGD[x_axis_cols + ['losses_sgd']] if not_none(SGD) else None,
              'adagrad': Adagrad[x_axis_cols + ['losses_adagrad']] if not_none(Adagrad) else None,
              'adam': Adam[x_axis_cols + ['losses_adam']] if not_none(Adam) else None,
              'src': SRC[x_axis_cols + ['losses_src', 'least_eig_src']] if not_none(SRC) else None,
              'src_momentum': SRC_M[x_axis_cols + ['losses_src_momentum', 'least_eig_src_momentum']] if not_none(SRC_M) else None,
              'src_momentum_non_ad': SRC_M_NON_AD[x_axis_cols + ['losses_src_momentum_non_ad',
                                                   'least_eig_src_momentum_non_ad']]
              if not_none(SRC_M_NON_AD) else None}

keys = []

for key, val in optimizers.items():
    if not_none(val):
        plt.plot(val['computations'], val[to_plot + '_' + key], label=key)
        keys.append(key)
plt.legend()
plt.title('AE with MNIST - ' + to_plot)
plt.savefig('fig_output/' + to_plot + '_'.join(keys) + '_' + src_file.split('/')[1] + '.png')
