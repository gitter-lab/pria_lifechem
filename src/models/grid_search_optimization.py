import argparse
import pandas as pd
import csv
import numpy as np
import json
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import ParameterGrid
sys.path.insert(0, '..')  # Add path from parent folder
sys.path.insert(0, '.')  # Add path from current folder
from function import *
from evaluation import *
from CallBacks import *
from single_classification import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', action="store", dest="config_json_file", required=True)
    parser.add_argument('--PMTNN_weight_file', action="store", dest="PMTNN_weight_file", required=True)
    parser.add_argument('--config_csv_file', action="store", dest="config_csv_file", required=True)
    parser.add_argument('--process_num', action='store', dest='process_num', required=True)
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    PMTNN_weight_file = given_args.PMTNN_weight_file
    config_csv_file = given_args.config_csv_file
    process_num = int(given_args.process_num)

    # specify dataset
    k = 5
    directory = '../../dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    # merge training and test dataset
    dtype_list = {'Molecule': np.str,
                  'SMILES': np.str,
                  'Fingerprints': np.str,
                  'Keck_Pria_AS_Retest': np.int64,
                  'Keck_Pria_FP_data': np.int64,
                  'Keck_Pria_Continuous': np.float64,
                  'Keck_RMI_cdd': np.float64}
    output_file_list = [directory + f_ for f_ in file_list]
    print output_file_list[0:4]
    train_pd = read_merged_data(output_file_list[0:4])
    print output_file_list[4]
    test_pd = read_merged_data([output_file_list[4]])

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=['Keck_Pria_AS_Retest'])
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=['Keck_Pria_AS_Retest'])
    cross_validation_split = StratifiedShuffleSplit(y_train, 1, test_size=0.15, random_state=1)
    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t, y_val = y_train[t_index], y_train[val_index]
    print 'done data preparation'

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
        
    hyperparameter_sets = {'optimizer': ['adam'],
                       'learning rate': [0.00003, 0.0001, 0.003],
                       'weighted schema': ['no-weight'],
                       'epoch size': [200, 1000],
                       'patience': [20, 200],
                       'early stopping': ['precision'],
                       'activation': [{0:'sigmoid', 1:'sigmoid', 2:'linear'},
                                      {0:'relu', 1:'relu', 2:'linear'}]}
    hyperparameters = ParameterGrid(hyperparameter_sets)
    
    cnt = 0
    for param in hyperparameters:
        if cnt != process_num:
            cnt += 1
            continue
        conf['compile']['optimizer']['option'] = param['optimizer']
        conf['compile']['optimizer'][param['optimizer']]['lr'] = param['learning rate']
        conf['class_weight_option'] = param['weighted schema']
        conf['fitting']['nb_epoch'] = param['epoch size']
        conf['fitting']['early_stopping']['patience'] = param['patience']
        conf['fitting']['early_stopping']['option'] = param['early stopping']
        activations = param['activation']
        conf['layers'][0]['activation'] = activations[0]
        conf['layers'][1]['activation'] = activations[1]
        conf['layers'][2]['activation'] = activations[2]
        print 'testing ', param
        break
        
    task = SingleClassification(conf=conf)
    task.train_and_predict(X_t, y_t, X_val, y_val, X_test, y_test, PMTNN_weight_file)
    store_data(transform_json_to_csv(config_json_file), config_csv_file)