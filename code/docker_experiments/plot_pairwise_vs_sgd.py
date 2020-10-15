import matplotlib.pyplot as plt
import pandas as pd
import glob

paths = [x + '_src_final' for x in ['resnet', 'ae', 'linear_reg', 'cnn']]
title_dict = {'ae_src_final': 'AE - MNIST - CV on 3 epochs',
              'linear_reg_src_final': 'LinReg - MNIST - CV on 3 epochs',
              'cnn_src_final': 'CNN - CIFAR-10 - CV on 3 epochs',
              'resnet_src_final': 'ResNet-10 - CIFAR-10 - CV on 3 epochs'}

criterion = 'train_losses'
cts = 'computations_times_sample'
substrings = ['H_size=100', 'LBFGS_Solver=Newton', 'computations']

for path in paths:
    files = glob.glob(path + "/*.csv")

    dfs = {}
    df_sgd = None

    for file in files:
        if not any(substr in file for substr in substrings):
            split_ = file.split('/')[-1].split("_")
            label = "{}".format(split_[3]) \
                if 'loss' not in file else \
                'SRC, ' + split_[6].split('=')[1] \
                + ', ' + split_[7].split('=')[1].replace('Linear', 'Linear_system')
            dfs[label] = pd.read_csv(file)
        elif 'computations' in file:
            df_sgd = pd.read_csv(file)

    fig, axs = plt.subplots(4, 2, sharex=True, sharey=True, figsize=(10, 10))
    plt.subplots_adjust(wspace=None, hspace=None)

    for i, (label, df) in enumerate(dfs.items()):
        print(label, path)
        n = len(df)
        axs[i // 2, i % 2].plot(df[cts if cts in df.columns else 'samples'], df[criterion],
                 label=label)
        axs[i // 2, i % 2].plot(df_sgd['samples'][:n], df_sgd[criterion][:n], label='SGD')
        axs[i // 2, i % 2].legend(loc='upper right', prop={'size': 8})
    for i in range(len(dfs), 8):
        axs[i // 2, i % 2].set_visible(False)

    fig.suptitle(title_dict[path])
    fig.text(0.5, 0.04, 'Oracle calls', ha='center')
    fig.text(0.04, 0.5, 'Training Loss', va='center', rotation='vertical')
    plt.savefig(path + '/final_pairwise_AE.png', bbox_inches="tight")
    plt.clf()
