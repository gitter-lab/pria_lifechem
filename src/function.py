import pandas as pd
import numpy as np
import random
import os
import json
import csv
from sklearn.cross_validation import StratifiedKFold, KFold


'''
Analyze the original dataset
'''
def analysis(data):
    class Node:
        def __init__(self, retest, fp, rmi):
            self.retest = retest
            self.fp = fp
            self.rmi = rmi
            if not np.isnan(self.rmi):
                self.rmi = int(self.rmi)
            else:
                self.rmi = np.NaN
            
        def __str__(self):
            ret = 'retest: {}, fp: {}, rmi: {}'.format(self.retest, self.fp, self.rmi)
            return ret
        
        def __eq__(self, other):
            return (self.retest, self.fp, self.rmi) == (other.retest, other.fp, other.rmi)
        
        def __hash__(self):
            return hash(self.retest) ^ hash(self.fp) ^ hash(self.rmi)
        
        def __cmp__(self):
            return (self.retest, self.fp, self.rmi) == (other.retest, other.fp, other.rmi)
    
    dict_ = {}
    for ix, row in data.iterrows():
        node = Node(row['Keck_Pria_AS_Retest'], row['Keck_Pria_FP_data'], row['Keck_RMI_cdd'])
        if node not in dict_.keys():
            dict_[node] = 1
        else:
            dict_[node] += 1
    
    for k in dict_.keys():
        print k, '\t---', dict_[k]
    
    return


'''
Apply greedy method to split data when merging multi-task
'''
def greedy_multi_splitting(data, k, directory, file_list):
    class Node:
        def __init__(self, retest, fp, rmi):
            self.retest = retest
            self.fp = fp
            self.rmi = rmi
            if not np.isnan(self.rmi):
                self.rmi = int(self.rmi)
            else:
                self.rmi = np.NaN
            
        def __str__(self):
            ret = 'retest: {}, fp: {}, rmi: {}'.format(self.retest, self.fp, self.rmi)
            return ret
        
        def __eq__(self, other):
            return (self.retest, self.fp, self.rmi) == (other.retest, other.fp, other.rmi)
        
        def __hash__(self):
            return hash(self.retest) ^ hash(self.fp) ^ hash(self.rmi)
        
        def __cmp__(self):
            return (self.retest, self.fp, self.rmi) == (other.retest, other.fp, other.rmi)
    
    dict_ = {}
    for ix, row in data.iterrows():
        node = Node(row['Keck_Pria_AS_Retest'], row['Keck_Pria_FP_data'], row['Keck_RMI_cdd'])
        if node not in dict_.keys():
            dict_[node] = []
        dict_[node].append(ix)
        
    list_ = []
    for key in dict_.keys():
        one_group_list = np.array(dict_[key])
        current = []

        if len(one_group_list) < k:
            n = len(one_group_list)
            for i in range(n):
                current.append(np.array(one_group_list[i]))
            for i in range(n, k):
                current.append(np.array([]))
        else:
            kf = KFold(len(one_group_list), k, shuffle=True)
            for _, test_index in kf:
                current.append(one_group_list[test_index])
        random.shuffle(current)
        list_.append(current)
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    print len(list_)

    for split in range(k):
        index_block = np.hstack((list_[0][split],
                                 list_[1][split],
                                 list_[2][split],
                                 list_[3][split],
                                 list_[4][split],
                                 list_[5][split],
                                 list_[6][split],
                                 list_[7][split],
                                 list_[8][split]))
        index_block = index_block.astype(np.int)
        df_block = data.iloc[index_block]
        print df_block.shape

        file_path = directory + file_list[split]
        df_block.to_csv(file_path, index=None)
    
    return


'''
input_file is whole 
output_file_list is the files that we will split data into
Make sure the output_file_list length is equal to k
'''
def split_data(input_file, output_file_list, k):
    data_pd = pd.read_csv(input_file)
    y_data = data_pd['true_label']
    y_data = y_data.astype(np.float64)
    if y_data.ndim == 1:
        n = y_data.shape[0]
        y_data = y_data.reshape(n, 1)

    cnt = 0
    split = StratifiedKFold(y_data[:, -1], n_folds=k, shuffle=True, random_state=0)
    for train_index, test_index in split:
        # For testing
        # Can list all existing active ones
        # data_batch[data_batch['true_label']>0]['molecule ID(RegID)']
        data_batch = data_pd.iloc[test_index]
        data_batch.to_csv(output_file_list[cnt], index_label=None, compression='gzip')
        cnt += 1
    return


'''
Read the data from all files in input_file_list
And merged into one dataset
'''
def read_merged_data(input_file_list):
    whole_pd = pd.DataFrame()
    for input_file in input_file_list:
        data_pd = pd.read_csv(input_file)
        whole_pd = whole_pd.append(data_pd)
    return whole_pd


'''
Get the fingerprints, with feature_name specified, and label_name specified
'''
def extract_feature_and_label(data_pd,
                              feature_name,
                              label_name_list):
    X_data = np.zeros(shape=(data_pd.shape[0], 1024))
    y_data = np.zeros(shape=(data_pd.shape[0], len(label_name_list)))
    index = 0
    for _, row in data_pd.iterrows():
        feature = list(row[feature_name])
        labels = row[label_name_list]
        X_data[index] = np.array(feature)
        y_data[index] = np.array(labels)
        index += 1
    X_data = X_data.astype(float)
    y_data = y_data.astype(float)

    # In case we just train on one target
    # y would be (n,) vector
    # then we should change it to (n,1) 1D matrix
    # to keep consistency
    print y_data.shape
    if y_data.ndim == 1:
        n = y_data.shape[0]
        y_data = y_data.reshape(n, 1)

    return X_data, y_data


'''
This function is used for extracting SMILES
in order to form sequential data
for use in RNN models, like LSTM
'''
def extract_SMILES_and_label(data_pd,
                             feature_name,
                             label_name_list,
                             SMILES_mapping_json_file):
    y_data = np.zeros(shape=(data_pd.shape[0], len(label_name_list)))
    X_data = []
    with open(SMILES_mapping_json_file, 'r') as f:
        dictionary = json.load(f)
    print 'alphabet set size {}'.format(len(dictionary))

    for smile in data_pd['SMILES']:
        X_data.append([dictionary[s] for s in smile])
    X_data = np.array(X_data)

    index = 0
    for _, row in data_pd.iterrows():
        labels = row[label_name_list]
        y_data[index] = np.array(labels)
        index += 1
    y_data = y_data.astype(np.float64)

    # In case we just train on one target
    # y would be (n,) vector
    # then we should change it to (n,1) 1D matrix
    # to keep consistency
    print y_data.shape
    if y_data.ndim == 1:
        n = y_data.shape[0]
        y_data = y_data.reshape(n, 1)

    return X_data, y_data


'''
Reshape vector into 2-dimension matrix.
'''
def reshape_data_into_2_dim(data):
    if data.ndim == 1:
        n = data.shape[0]
        data = data.reshape(n, 1)
    return data


'''
Store result
'''
def store_data(data, file):
    with open(file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for d in data:
            writer.writerow([d])
    return

'''
Store json file to csv
'''
def transform_json_to_csv(conf_file):
    with open(conf_file, 'r') as f:
        conf = json.load(f)

    configuration = []

    layers = conf['layers']
    layer_number = len(layers)
    for i in range(layer_number):
        init = layers[i]['init']
        activation = layers[i]['activation']
        if i == 0:
            hidden_units = int(layers[i]['hidden_units'])
            dropout = float(layers[i]['dropout'])
            layer_info = 'hidden unit number:{}, ' \
                         'init:{}, ' \
                         'activation:{}, ' \
                         'drop-out:{}'.format(hidden_units,
                                              init,
                                              activation,
                                              dropout)
        elif i == layer_number-1:
            layer_info = 'init:{}, activation:{}'.format(init, activation)
        else:
            hidden_units = int(layers[i]['hidden_units'])
            dropout = float(layers[i]['dropout'])
            layer_info = 'hidden unit number:{},' \
                         'init:{}, activation:{},' \
                         'drop-out:{}'.format(hidden_units,
                                              init,
                                              activation,
                                              dropout)
        configuration.append([str(i+1), layer_info])

    fit_nb_epoch = conf['fitting']['nb_epoch']
    fit_batch_size = conf['fitting']['batch_size']
    early_stopping_patience = conf['fitting']['early_stopping']['patience']
    early_stopping_option = conf['fitting']['early_stopping']['option']
    fit_parameters_config = 'epoch:{}, ' \
                            'batch_size:{}, ' \
                            'early stopping: {} with {} tolerance'.format(fit_nb_epoch,
                                                                          fit_batch_size,
                                                                          early_stopping_option,
                                                                          early_stopping_patience)
    configuration.append(['fit paramters', fit_parameters_config])

    compile_loss = conf['compile']['loss']
    compile_optimizer_option = conf['compile']['optimizer']['option']
    if compile_optimizer_option == 'sgd':
        sgd_lr = conf['compile']['optimizer']['sgd']['lr']
        sgd_momentum = conf['compile']['optimizer']['sgd']['momentum']
        sgd_decay = conf['compile']['optimizer']['sgd']['decay']
        sgd_nestrov = conf['compile']['optimizer']['sgd']['nestrov']
        optimizer_info = '{}: ' \
                         'learning rate:{}, ' \
                         'momentum:{}, ' \
                         'decay:{}, ' \
                         'nestrov:{}'.format(compile_optimizer_option,
                                             sgd_lr,
                                             sgd_momentum,
                                             sgd_decay,
                                             sgd_nestrov)
    elif compile_optimizer_option == 'adam':
        adam_lr = conf['compile']['optimizer']['adam']['lr']
        adam_beta_1 = conf['compile']['optimizer']['adam']['beta_1']
        adam_beta_2 = conf['compile']['optimizer']['adam']['beta_2']
        adam_epsilon = conf['compile']['optimizer']['adam']['epsilon']
        optimizer_info = '{}: ' \
                         'learning rate:{}, ' \
                         'beta_1:{}, ' \
                         'beta_2:{}, ' \
                         'epsilon:{}'.format(compile_optimizer_option,
                                             adam_lr,
                                             adam_beta_1,
                                             adam_beta_2,
                                             adam_epsilon)
    else:
        optimizer_info = 'not use'
    compiler_config = 'loss:{}, optimizer:{}'.format(compile_loss, optimizer_info)
    configuration.append(['optimizer', compiler_config])

    batch_is_use = conf['batch']['is_use']
    if batch_is_use:
        batch_normalizer_epsilon = conf['batch']['epsilon']
        batch_normalizer_mode = conf['batch']['mode']
        batch_normalizer_axis = conf['batch']['axis']
        batch_normalizer_momentum = conf['batch']['momentum']
        batch_normalizer_weights = conf['batch']['weights']
        batch_normalizer_beta_init = conf['batch']['beta_init']
        batch_normalizer_gamma_init = conf['batch']['gamma_init']
        batch_normalizer_config = 'epsilon:{}, ' \
                                  'mode:{}, ' \
                                  'axis:{}, ' \
                                  'momentum:{}, ' \
                                  'weights:{}, ' \
                                  'beta_init:{}, ' \
                                  'gamma_init:{}'.format(batch_normalizer_epsilon,
                                                         batch_normalizer_mode,
                                                         batch_normalizer_axis,
                                                         batch_normalizer_momentum,
                                                         batch_normalizer_weights,
                                                         batch_normalizer_beta_init,
                                                         batch_normalizer_gamma_init)
    else:
        batch_normalizer_config = 'not use'
    configuration.append(['batch normalizer', batch_normalizer_config])

    return configuration

"""
    splits pcba dataset into k folds using a greedy, smallest actives first 
    approach that achieves good splitting across labels.
    
    Algorithm:
        1- shuffle the pcba rows randomly
        2- sort the pcba labels from smallest active_counts to largest
        3- iterate on this sorted label list and do:
            -create k folds which will contain the row indexes only
            -split the active_indexes into the k folds
            -split the inactives_indexes into the k folds
            -split the missing_indexes into the k folds
            
            -uniquify each fold to remove duplicate row indexes.
            
            -greedily remove overlapping indexes from each fold. start with 
             fold 0 and remove from the other 1-k folds. then fold 1 and remove
             from the other 2-k folds. then fold 2 and remove from the other 
             3-k folds. and so on. This ensures that the top most fold contains
             the row index and all other folds do not.
        
        4- uniquify each fold to remove duplicate row indexes just in case
"""
def split_pcba_into_folds(data_dir, k, dest_dir):
    nb_classes = 128
    
    pcba_df = pd.read_csv(data_dir)
    pcba_df = pcba_df.sample(frac=1).reset_index(drop=True) #shuffle rows
    pcba_df = pcba_df.sample(frac=1).reset_index(drop=True) #shuffle rows
    pcba_df = pcba_df.sample(frac=1).reset_index(drop=True) #shuffle rows
    pcba_df = pcba_df.sample(frac=1).reset_index(drop=True) #shuffle rows
    
    label_names = pcba_df.columns[3:]
    total_compounds = pcba_df.index[-1]+1
    
    num_actives_list = list()
    for i in range(nb_classes):
        curr_label = np.array(pcba_df[label_names[i]])
        actives_count = curr_label[curr_label==1].shape[0]
        num_actives_list.append(actives_count)
    sorted_label_indexes = np.argsort(num_actives_list).tolist()[::-1]
    
    fold_size = total_compounds // k
    fold_indexes = [np.array([],dtype=np.int64) for _ in range(k)]
    for i in sorted_label_indexes:
        curr_label = np.array(pcba_df[label_names[i]])
        
        #actives
        active_indexes = np.where(curr_label==1)[0]
        kf = KFold(len(active_indexes), n_folds=k, shuffle=False)
        for fi, (_, test_index) in zip(range(k), kf):        
            fold_indexes[fi] = np.append(fold_indexes[fi], active_indexes[test_index])
        
        #inactives
        inactive_indexes = np.where(curr_label==0)[0]
        kf = KFold(len(inactive_indexes), n_folds=k, shuffle=False)
        for fi, (_, test_index) in zip(range(k), kf):       
            fold_indexes[fi] = np.append(fold_indexes[fi], inactive_indexes[test_index])
        
        #missing
        missing_indexes = np.where(curr_label==-1)[0]
        kf = KFold(len(missing_indexes), n_folds=k, shuffle=False)
        for fi, (_, test_index) in zip(range(k), kf):       
            fold_indexes[fi] = np.append(fold_indexes[fi], missing_indexes[test_index])
            
        #now uniquify the indexes in each fold
        for ki in range(k):
            fold_indexes[ki] = np.unique(fold_indexes[ki])
            
        for ki in range(k):
            for kj in range(ki+1,k):       
                remove_indexes = np.where(np.in1d(fold_indexes[kj], fold_indexes[ki]))
                fold_indexes[kj] = np.delete(fold_indexes[kj], remove_indexes)
        
    #now uniquify the indexes in each fold
    for i in range(k):
        fold_indexes[i] = np.unique(fold_indexes[i])
        
    #check if any folds have overlapping rows
    for i in range(k):
        for j in range(k):
            if i != j and np.any(np.in1d(fold_indexes[i], fold_indexes[j])):
                print('Found overlapping indexes!!!')
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir) 
        
    #save label statistics for each fold
    label_stat_columns = list()
    for i in range(len(label_names)):
        label_stat_columns.extend([label_names[i]+'_actives',
                                   label_names[i]+'_inactives',
                                   label_names[i]+'_missing'])
                                   
    index_names = ['fold_'+str(i) for i in range(k)]
    index_names.extend(['mean', 'stdev'])
    label_stats_df = pd.DataFrame(data=np.zeros((k+2,len(label_stat_columns))), 
                                  columns=label_stat_columns, dtype=np.int64,
                                  index=index_names)
    for i in range(k):    
        curr_fold_df = pcba_df.iloc[fold_indexes[i],:]
        for j, label in zip(range(len(label_names)), label_names):
            curr_label = np.array(curr_fold_df[label])
            
            ind = 3*j
            label_stats_df.iloc[i][ind] = curr_label[curr_label==1].shape[0]
            label_stats_df.iloc[i][ind+1] = curr_label[curr_label==0].shape[0]
            label_stats_df.iloc[i][ind+2] = curr_label[curr_label==-1].shape[0]
    
    label_stats_df.iloc[-2] = np.mean(label_stats_df.iloc[0:k][:])
    label_stats_df.iloc[-1] = np.std(label_stats_df.iloc[0:k][:])
        
    
    label_stats_df.to_csv(dest_dir+'label_fold_stats.csv')
    with open(dest_dir+'label_fold_stats.csv','a') as f:  
        f.write('\n')
        s=0
        for i in range(k):
            f.write('fold_'+str(i)+' size:,' + str(len(fold_indexes[i])) + '\n')
            s = s + len(fold_indexes[i])
        f.write('sum of all folds:,' + str(s))
    
    #create the fold csv files
    pcba_df.replace(to_replace=-1, value='', inplace=True)
    cols = pcba_df.columns.values
    cols[0] = 'Molecule'
    cols[1] = 'SMILES'
    cols[2] = 'Fingerprints'
    pcba_df.columns = cols
    pcba_df['Molecule'] = 'PCBA-' + pcba_df['Molecule'].astype(str)
       
    file_list = []
    for i in range(k):
        file_list.append(dest_dir+'file_{}.csv'.format(i))
    
    for i in range(k):
        pcba_df.iloc[fold_indexes[i],:].to_csv(file_list[i], index=None)


"""
    merges keck folds with pcba folds in a straightforward manner.
"""
def merge_keck_pcba(keck_dir, pcba_dir, k, dest_dir):
    nb_classes = 5+128
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir) 
    #now perform merging of folds
    overlap_counts = list()
    for i in range(k):
        keck_fold_df = pd.read_csv(keck_dir+'file_{}.csv'.format(i))
        pcba_fold_df = pd.read_csv(pcba_dir+'file_{}.csv'.format(i))
        merged_fold_df = pd.merge(keck_fold_df, pcba_fold_df, how='outer',
                                  on='SMILES', indicator=True)
        overlap_counts.append(merged_fold_df[merged_fold_df._merge == 'both'].shape[0])
        merged_fold_df.Fingerprints_x[pd.isnull(merged_fold_df.Fingerprints_x)] = merged_fold_df.Fingerprints_y[pd.isnull(merged_fold_df.Fingerprints_x)]
        merged_fold_df['Fingerprints'] = merged_fold_df.Fingerprints_x
        
        merged_fold_df.replace(to_replace=np.nan, value='', inplace=True)
        merged_fold_df['Molecule'] = merged_fold_df.Molecule_x +'_'+ merged_fold_df.Molecule_y
        
        label_names = keck_fold_df.columns[3:].tolist() + pcba_fold_df.columns[3:].tolist()
        cols = ['Molecule', 'SMILES', 'Fingerprints'] + label_names
        merged_fold_df = merged_fold_df[cols]
        merged_fold_df.to_csv(dest_dir+'file_{}.csv'.format(i), index=None)
    
    
    
    keck_fold_df = pd.read_csv(keck_dir+'file_{}.csv'.format(0))
    pcba_fold_df = pd.read_csv(pcba_dir+'file_{}.csv'.format(0))
    label_names = keck_fold_df.columns[3:].tolist() + pcba_fold_df.columns[3:].tolist()
    
    #save label statistics for each fold
    label_stat_columns = list()
    label_names.remove('Keck_Pria_Continuous')
    label_names.remove('FP counts % inhibition')
    for i in range(len(label_names)):
        label_stat_columns.extend([label_names[i]+'_actives',
                                   label_names[i]+'_inactives',
                                   label_names[i]+'_missing'])
                                   
    index_names = ['fold_'+str(i) for i in range(k)]
    index_names.extend(['mean', 'stdev'])
    label_stats_df = pd.DataFrame(data=np.zeros((k+2,len(label_stat_columns))), 
                                  columns=label_stat_columns, dtype=np.int64,
                                  index=index_names)
    fold_sizes = [0 for _ in range(k)]
    for i in range(k):    
        curr_fold_df = pd.read_csv(dest_dir+'file_{}.csv'.format(i))
        for j, label in zip(range(len(label_names)), label_names):
            curr_label = np.array(curr_fold_df[label])
            fold_sizes[i] = curr_label.shape[0]
            
            ind = 3*j
            label_stats_df.iloc[i][ind] = curr_label[curr_label==1].shape[0]
            label_stats_df.iloc[i][ind+1] = curr_label[curr_label==0].shape[0]
            label_stats_df.iloc[i][ind+2] = np.where(np.isnan(curr_label))[0].shape[0]
            
    label_stats_df.iloc[-2] = np.mean(label_stats_df.iloc[0:k][:])
    label_stats_df.iloc[-1] = np.std(label_stats_df.iloc[0:k][:])
        
    
    label_stats_df.to_csv(dest_dir+'label_fold_stats.csv')
    with open(dest_dir+'label_fold_stats.csv','a') as f:  
        f.write('\n')
        s_size=0
        s_overlap=0
        f.write(',size,overlap_count\n')
        for i in range(k):
            f.write('fold_'+str(i)+','+str(fold_sizes[i])+','+str(overlap_counts[i])+'\n')
            s_size = s_size + fold_sizes[i]
            s_overlap = s_overlap + overlap_counts[i]
        f.write('sum :,' + str(s_size)+','+str(s_overlap))