import matplotlib.pyplot as plt
import pandas as pd
import glob

paths = [x + '_src_final' for x in ['resnet', 'ae', 'linear_reg', 'cnn']]
criterion = 'train_losses'

for path in paths:
    files = glob.glob(path + "/*.csv")
    cts = 'computations_times_sample'

    dfs = {}

    for file in files:
        if 'H_size=100' not in file and 'LBFGS_Solver=Newton' not in file:
            split_ = file.split('/')[-1].split("_")
            label = "{}".format(split_[3]) \
                if 'loss' not in file else \
                'SRC, ' + split_[5]+'_' + split_[6] + ', ' + split_[7].replace('Linear', 'Linear_system')
            dfs[label] = pd.read_csv(file)

    for label, df in dfs.items():
        plt.plot(df[cts if cts in df.columns else 'samples'][:], df[criterion][:],
                 label=label)

    plt.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.title(path)
    plt.xlabel('Oracle calls')
    plt.ylabel('Training Loss')
    plt.savefig(path + '/resnet_CIFAR_hsize.png', bbox_inches="tight")
    plt.clf()
