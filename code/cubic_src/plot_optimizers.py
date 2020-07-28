import matplotlib.pyplot as plt
import pandas as pd
import os
from torch import tensor


def not_none(x):
    return x is not None


to_plot = 'losses'  # losses, least_eig
from_ = 0  # 1.0
till_ = 6e5

sgd_file = 'fig/benchmarks_networks_2020-07-18_15-03-40/computations_AE_MNIST_SGD_0.3_100_100_1_scheduler=False.csv'
# 'fig/benchmarks_networks_2020-07-13_14-55-30/computations_AE_MNIST_SGD_0.3_100_100_1_scheduler=False.csv'
# 'fig/benchmarks_networks_2020-06-25_18-16-28/computations_AE_MNIST_SGD_0.3_100_100_1_scheduler=False.csv'
# 'fig/benchmarks_networks_2020-06-23_19-06-43/computations_CNN_MNIST_SGD_0.05_100_100_1_scheduler=False.csv'
# 'fig_final/_computations_AE_MNIST_SGD_0.3_100_100_1.csv'
adagrad_file = 'fig/benchmarks_networks_2020-07-18_14-25-30/computations_AE_MNIST_Adagrad_0.01_100_100_1_scheduler=False.csv'
# 'fig_final/_computations_AE_MNIST_Adagrad_0.001_100_100_1.csv'
adam_file = 'fig/benchmarks_networks_2020-07-18_15-48-38/computations_AE_MNIST_Adam_0.001_100_100_1_scheduler=False.csv'
# 'fig/benchmarks_networks_2020-06-25_18-16-29/computations_AE_MNIST_Adam_0.001_100_100_1_scheduler=False.csv'
# 'fig/benchmarks_networks_2020-06-23_19-06-43/computations_CNN_MNIST_Adam_0.0005_100_100_1_scheduler=False.csv'
# 'figs_without_date/_computations_AE_MNIST_Adam_0.001_100_100_1.csv'  # 'fig_final/_computations_AE_MNIST_Adagrad_0.001_100_100_1.csv'
src_file = 'fig/2020-07-18_14-45-20/loss_src_delta=False_AdaHess=False_sigma=1.0_delta_step=0.001_eta=0.1_AE_MNIST_non-adaptive_H_size=10_g_size=100.csv'
src_ad_file = 'fig/2020-07-18_11-36-26/loss_src_delta=False_AdaHess=False_sigma=1.0_delta_step=0.001_eta=0.3_AE_MNIST_adaptive_H_size=10_g_size=100.csv'
# 'fig/2020-07-13_16-18-58/loss_src_delta=False_sigma=1.0_delta_step=0.001_eta=0.3_AE_MNIST_non-adaptive_H_size=10_g_size=100.csv'
# 'fig/2020-06-25_09-23-13/loss_src_delta=False_sigma=10.0_delta_step=0.001_eta=0.5_AE_MNIST_non-adaptive_H_size=10_g_size=100.csv'
# 'fig_final/2020-06-09_14-53-09/loss_src_False_15000_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-12_01-14-01/loss_src_False_10_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-12_01-14-01/loss_src_False_10_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-12_00-57-12/loss_src_False_10_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig_final/2020-06-09_14-53-09/loss_src_False_15000_0.001_AE_adaptive_10.0_100.0.csv'
src_m_file = 'fig/2020-07-18_11-36-26/loss_src_delta=True_AdaHess=False_sigma=1.0_delta_step=0.001_eta=0.3_AE_MNIST_non-adaptive_H_size=10_g_size=100.csv'
# 'fig/2020-06-25_09-23-14/loss_src_delta=True_sigma=100.0_delta_step=0.001_eta=0.3_AE_MNIST_adaptive_H_size=10_g_size=100.csv'
# 'fig_final/2020-06-09_14-42-26/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
#  'fig/2020-06-12_01-18-09/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-11_18-57-58/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-12_01-18-09/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-12_00-31-48/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
# 'fig/2020-06-11_12-13-38/loss_src_True_15000_0.001_AE_adaptive_10.0_100.0.csv'
src_hess_av_ad_file = 'fig/2020-07-18_11-36-26/loss_src_delta=False_AdaHess=True_sigma=1.0_delta_step=0.001_eta=0.3_AE_MNIST_adaptive_H_size=10_g_size=100.csv'
src_hess_av_non_ad_file = 'fig/2020-07-18_11-36-26/loss_src_delta=False_AdaHess=True_sigma=1.0_delta_step=0.001_eta=0.3_AE_MNIST_non-adaptive_H_size=10_g_size=100.csv'
src_m_non_ad_file = None  # 'fig/2020-06-11_12-14-02/loss_src_True_15000_0.001_AE_non-adaptive_10.0_100.0.csv'

SGD = pd.read_csv(sgd_file) \
                  .rename(columns={'losses': 'losses_sgd'}) if sgd_file else None

Adam = pd.read_csv(adam_file) \
                   .rename(columns={'losses': 'losses_adam'}) if adam_file else None

Adagrad = pd.read_csv(adagrad_file) \
                   .rename(columns={'losses': 'losses_adagrad'}) if adagrad_file else None

SRC = pd.read_csv(src_file) \
                  .rename(columns={'losses': 'losses_src', 'least_eig': 'least_eig_src'}) if src_file else None

SRC_AD = pd.read_csv(src_ad_file) \
                  .rename(columns={'losses': 'losses_src_ad', 'least_eig': 'least_eig_src_ad'}) if src_ad_file else None

SRC_M = pd.read_csv(src_m_file) \
                    .rename(columns={'losses': 'losses_src_momentum', 'least_eig': 'least_eig_src_momentum'}) if src_m_file\
    else None

SRC_AV_AD = pd.read_csv(src_hess_av_ad_file) \
                  .rename(columns={'losses': 'losses_src_av_ad', 'least_eig': 'least_eig_src_av_ad'}) if src_hess_av_ad_file else None

SRC_AV_NAD = pd.read_csv(src_hess_av_non_ad_file) \
                  .rename(columns={'losses': 'losses_src_av_nad', 'least_eig': 'least_eig_src_av_nad'}) if src_hess_av_non_ad_file else None

SRC_M_NON_AD = pd.read_csv(src_m_non_ad_file) \
                           .rename(columns={'losses': 'losses_src_momentum_non_ad',
                                            'least_eig': 'least_eig_src_momentum_non_ad'}) \
    if src_m_non_ad_file else None

#SRC['computations_round'] = SRC.computations.map(lambda x: round(x/100)*100)




#combined = SRC.merge(Adam, how='left', left_on='computations_round', right_on='samples')\
#    .merge(SGD, how='left', left_on='computations_round', right_on='samples')
#combined_pure = combined[combined.samples > 2000]

#print(combined_pure.head(20))

x_axis_cols_src = ['samples', 'computations', 'computations_times_sample']
x_axis_cols = ['samples', 'computations']


optimizers = {'sgd': SGD[x_axis_cols + ['losses_sgd']] if not_none(SGD) else None,
              'adagrad': Adagrad[x_axis_cols + ['losses_adagrad']] if not_none(Adagrad) else None,
              'adam': Adam[x_axis_cols + ['losses_adam']] if not_none(Adam) else None,
              'src': SRC[x_axis_cols_src + ['losses_src', 'least_eig_src']] if not_none(SRC) else None,
              'src_ad': SRC_AD[x_axis_cols_src + ['losses_src_ad', 'least_eig_src_ad']] if not_none(SRC_AD) else None,
              'src_momentum': SRC_M[x_axis_cols_src + ['losses_src_momentum', 'least_eig_src_momentum']] if not_none(SRC_M) else None,
              'src_av_nad': SRC_AV_NAD[x_axis_cols_src + ['losses_src_av_nad', 'least_eig_src_av_nad']] if not_none(SRC_AV_NAD) else None,
              'src_av_ad': SRC_AV_AD[x_axis_cols_src + ['losses_src_av_ad', 'least_eig_src_av_ad']] if not_none(
                  SRC_AV_AD) else None,
              'src_momentum_non_ad': SRC_M_NON_AD[x_axis_cols_src + ['losses_src_momentum_non_ad',
                                                   'least_eig_src_momentum_non_ad']]
              if not_none(SRC_M_NON_AD) else None}

keys = []

for key, val in optimizers.items():
    if not_none(val):
        n = len(val['samples'])
        print(n, key)
        samples = list(val['samples' if 'src' not in key else 'computations_times_sample'])
        print(from_, till_, samples)
        idx = [i for i, el in enumerate(samples) if from_ <= el <= till_]
        print(idx)
        from_v, till_v = idx[0], idx[-1]
        plt.plot(val['samples' if 'src' not in key else 'computations_times_sample']
                 [from_v:till_v], val[to_plot + '_' + key][from_v:till_v] * 0.5, label=key)
        keys.append(key)
plt.legend()
plt.title('AE with MNIST - ' + to_plot)
plt.ylim([0.00, 0.04])
plt.savefig('fig_output/' + str(from_) + '_' + str(till_) + '_' + to_plot + '_'.join(keys) + '_' + sgd_file.split('/')[1] + '.png')
