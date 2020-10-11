import matplotlib.pyplot as plt
import pandas as pd
import glob

files = glob.glob("*.csv")
cts = 'computations_done_times_samples'

dfs = {}

for file in files:
    split_ = file.split("_")
    label = "{}".format(split_[3]) \
        if 'SRC' not in file else 'SCR'
    dfs[label] = pd.read_csv(file)

for label, df in dfs.items():
    plt.plot(df[cts][:], df['losses'][:],
             label=label)

plt.legend()
plt.title('W-function')
plt.xlabel('Oracle calls')
plt.ylabel('Loss')
plt.savefig('w-function_SGD_SRC.png')
