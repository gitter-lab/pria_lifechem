import pandas as pd
import numpy as np
import random
import os
import json
import csv
from sklearn.cross_validation import StratifiedKFold, KFold
import matplotlib.pyplot as plt
import re


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
Merge data frame corresponding to the corresponding index
'''
def merge_pd(data_pd_list, index_list):
    whole_pd = pd.DataFrame()
    for index in index_list:
        data_pd = data_pd_list[index]
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


def store_config(conf, file_):
    with open(file_, 'w') as f:
        json.dump(conf, f)
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
