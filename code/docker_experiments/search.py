import pandas as pd
from pathlib import Path
import numpy as np
from shutil import copy
import pprint

path = 'cnn_src'
criterion = 'train_losses'

terms = ['AdaHess_Solver=adaptive', 'AdaHess_Solver=Linear_system',
         'AdaHess_Solver=Newton', 'AdaHess_Solver=non-adaptive',
         'LBFGS_Solver=Newton', 'None_Solver=non-adaptive',
         'WoodFisher_Solver=Newton', 'LBFGS_Solver=Linear_system', 'None_Solver=adaptive',
         'WoodFisher_Solver=Linear_system']
term_dicts = {key: {'loss': np.inf, 'path': ''} for key in terms}

for path_ in Path(path).rglob('*.csv'):
    for t in terms:
        if t in path_.name:
            if term_dicts[t]['loss'] > pd.read_csv(path_)[criterion].iloc[-1]:
                term_dicts[t]['path'] = path_
                term_dicts[t]['loss'] = pd.read_csv(path_)[criterion].iloc[-1]

pprint.pprint(term_dicts)
for (el, key) in term_dicts.items():
    #print(key['path'])
    try:
        copy(src=key['path'], dst=path + '_final/', follow_symlinks=True)
    except:
        print(key['path'])
