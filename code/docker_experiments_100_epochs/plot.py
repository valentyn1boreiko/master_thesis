import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

#paths = [x + '_src_final_fig' for x in ['ae', 'linear_reg', 'cnn', 'resnet']]
paths = [name for name in os.listdir('.') if os.path.isdir(name)]
criterion = 'train_losses'
criterion_grad = 'grad_norms'

for path in paths:
    files = glob.glob(path + "/*.csv")
    cts = 'computations_times_sample'
    sample_size = "100 vs. 10" in path

    dfs = {}

    if not sample_size:
        for file in files:
            if 'H_size=100' not in file and 'LBFGS_Solver=Newton' not in file:
                split_ = file.split('/')[-1].split("_")

                label = "{}".format(split_[3]) \
                    if 'loss' not in file else \
                    'SCR, ' + split_[5]+'_' + split_[6] + ', ' + split_[7].replace('Linear', 'Linear_system')
                dfs[label] = pd.read_csv(file)
    else:
        dfs_grad = {}

        for file in files:
            split_ = file.split('/')[-1].split("_")
            hsize = file.split('H_size=')[1].split('_')[0]
            label = "SCR, Hessian batch size = {}, {}_{}, {}"\
                .format(hsize, split_[5], split_[6], split_[7].replace('Linear', 'Linear_system'))
            dfs[label] = pd.read_csv(file)
            if 'H_size=10_' in file:
                dfs_grad[label] = pd.read_csv(file)

        for i, (label, df) in enumerate(dfs_grad.items()):
            # print(df.index.tolist())
            plt.plot(df[(cts if cts in df.columns else 'samples')][:-9],
                     pd.concat([df[criterion_grad][:1],
                                df[criterion_grad][1:].rolling(10).mean().dropna()]),
                     label=label)

            plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
            plt.title(path.split(' - Hessian')[0])
            plt.xlabel('Oracle calls')
            plt.ylabel('Gradient norm')
            plt.savefig(path + '/final_grad.png', bbox_inches="tight")
            plt.clf()

    for i, (label, df) in enumerate(dfs.items()):
        #print(df.index.tolist())
        plt.plot(df[(cts if cts in df.columns else 'samples') if not sample_size else 'computations'][:-9], pd.concat([df[criterion][:1],
                                                                            df[criterion][1:].rolling(10).mean().dropna()]),
                 label=label, linewidth=0.5 if i == 1 and sample_size else 1)

    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.title(path)
    plt.xlabel('Oracle calls' if not sample_size else 'Computations')
    plt.ylabel('Training Loss')
    plt.savefig(path + '/resnet_CIFAR_hsize.png', bbox_inches="tight")
    plt.clf()
