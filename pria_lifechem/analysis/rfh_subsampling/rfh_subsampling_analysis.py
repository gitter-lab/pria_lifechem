import sys
sys.path.insert(0, '..')
from gather_metrics import *
import pandas as pd
import seaborn as sns

def read_merged_data(input_file_list, usecols=None):
    whole_pd = pd.DataFrame()
    for input_file in input_file_list:
        data_pd = pd.read_csv(input_file, usecols=usecols)
        whole_pd = whole_pd.append(data_pd)
    return whole_pd

# rename subfolders from run_{} to fold_{} so it can work with existing code
def rename_run_dirs(class_dirs):
    for cdir in class_dirs:
        run_dirs = [cdir+'/{}/'.format(d) for d in os.listdir(cdir)]
        for rdir in run_dirs:
            new_rdir_name = rdir.replace('run', 'fold')
            os.rename(rdir, new_rdir_name)
            
 
"""
    Returns the train actives counts for MAX/MIN/agg. run.
"""
def get_active_count_dict(model_directory, y_train, gather_df, column, idxfunc, col_threshold_func):
    col_df = gather_df[column]
    
    if idxfunc == 'idxmax':
        col_func_df = col_df.loc[col_df.groupby('train_size').idxmax()]
    else:
        col_func_df = col_df.loc[col_df.groupby('train_size').idxmin()]
    col_func_df = col_func_df[col_func_df.apply(col_threshold_func)]
    tsize_run_pairs = col_func_df.index.tolist()
    func_run_train_indices_dirs = [model_directory+'/n_{}/fold_{}/train_indices.npy'.format(tsize, int(func_run.replace('run ',''))) for tsize, func_run in tsize_run_pairs]

    # read in the train_indices.npy of the func run for each n_{}
    train_indices_dict = {}
    for i,  tsize_run, train_indices_dir in zip(range(len(tsize_run_pairs)), tsize_run_pairs, func_run_train_indices_dirs):
        tsize, run = tsize_run
        train_indices_dict[tsize] = np.load(train_indices_dir)
        assert tsize == train_indices_dict[tsize].shape[0]
    # maintain counts of the number of times each active appeared in the training set
    train_actives_count_dict = {}
    active_indices = np.where(y_train == 1)[0]
    for aidx in active_indices:
        train_actives_count_dict[aidx] = 0
    for tsize in train_indices_dict:
        for aidx in train_indices_dict[tsize][np.where(y_train[train_indices_dict[tsize]] == 1)[0]]:
            train_actives_count_dict[aidx] += 1
            
    return train_actives_count_dict
    
    
"""
    Returns the train actives counts for runs that are true for col_threshold_fun and tsize_threshold_func.
"""
def get_active_count_dict_with_tsize_thresh(model_directory, y_train, gather_df, column, col_threshold_func, tsize_threshold_func):
    col_df = gather_df[column]
    tsize_run_pairs = col_df.index.tolist()
    func_run_train_indices_dirs = [model_directory+'/n_{}/fold_{}/train_indices.npy'.format(tsize, int(func_run.replace('run ',''))) for tsize, func_run in tsize_run_pairs]

    # read in the train_indices.npy of the func run for each n_{}
    train_indices_dict = {}
    for i,  tsize_run, train_indices_dir in zip(range(len(tsize_run_pairs)), tsize_run_pairs, func_run_train_indices_dirs):
        tsize, run = tsize_run
        train_indices_dict[tsize_run] = np.load(train_indices_dir)
        assert tsize == train_indices_dict[tsize_run].shape[0]
    # maintain counts of the number of times each active appeared in the training set
    train_actives_count_dict = {}
    active_indices = np.where(y_train == 1)[0]
    for aidx in active_indices:
        train_actives_count_dict[aidx] = 0
    for tsize_run in train_indices_dict:
        tsize, run = tsize_run
        if tsize_threshold_func(tsize) and col_threshold_func(col_df[tsize][run]):
            for aidx in train_indices_dict[tsize_run][np.where(y_train[train_indices_dict[tsize_run]] == 1)[0]]:
                train_actives_count_dict[aidx] += 1
    return train_actives_count_dict