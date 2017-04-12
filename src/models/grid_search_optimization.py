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
from sklearn.cross_validation import StratifiedShuffleSplit, ShuffleSplit
from sklearn.grid_search import ParameterGrid
sys.path.insert(0, '..')  # Add path from parent folder
sys.path.insert(0, '.')  # Add path from current folder
from function import *
from evaluation import *
from CallBacks import *
from single_classification import *
from single_regression import *
from vanilla_lstm import *


def run_single_classification():
    # specify dataset
    k = 5
    directory = '../../dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    # merge training and test dataset
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
                           'activations': [{0:'relu', 1:'sigmoid', 2:'sigmoid'},
                                           {0:'relu', 1:'relu', 2:'sigmoid'}]}
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
        activations = param['activations']
        conf['layers'][0]['activation'] = activations[0]
        conf['layers'][1]['activation'] = activations[1]
        conf['layers'][2]['activation'] = activations[2]
        print 'Testing hyperparameter ', param
        break

    task = SingleClassification(conf=conf)
    task.train_and_predict(X_t, y_t, X_val, y_val, X_test, y_test, PMTNN_weight_file)
    store_data(transform_json_to_csv(config_json_file), config_csv_file)

    return


def run_single_regression():
    # specify dataset
    k = 5
    directory = '../../dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    # merge training and test dataset
    output_file_list = [directory + f_ for f_ in file_list]
    print output_file_list[0:4]
    train_pd = read_merged_data(output_file_list[0:4])
    print output_file_list[4]
    test_pd = read_merged_data([output_file_list[4]])

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=['Keck_Pria_Continuous'])
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=['Keck_Pria_Continuous'])
    cross_validation_split = ShuffleSplit(y_train.shape[0], n_iter=1, test_size=0.15, random_state=1)
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
                           'activations': [{0:'sigmoid', 1:'sigmoid', 2:'linear'},
                                           {0:'relu', 1:'sigmoid', 2:'sigmoid'}]}
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
        activations = param['activations']
        conf['layers'][0]['activation'] = activations[0]
        conf['layers'][1]['activation'] = activations[1]
        conf['layers'][2]['activation'] = activations[2]
        print 'Testing hyperparameter ', param
        break

    task = SingleRegression(conf=conf)
    task.train_and_predict(X_t, y_t, X_val, y_val, X_test, y_test, PMTNN_weight_file)
    store_data(transform_json_to_csv(config_json_file), config_csv_file)
    return


def run_vanilla_lstm():
    # specify dataset
    k = 5
    directory = '../../dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    output_file_list = [directory + f_ for f_ in file_list]
    print output_file_list[:4]
    train_pd = read_merged_data(output_file_list[0:4])
    print output_file_list[4]
    test_pd = read_merged_data([output_file_list[4]])

    # extract data, and split training data into training and val
    X_train, y_train = extract_SMILES_and_label(train_pd,
                                                feature_name='SMILES',
                                                label_name_list=['Keck_Pria_AS_Retest'],
                                                SMILES_mapping_json_file=SMILES_mapping_json_file)
    X_test, y_test = extract_SMILES_and_label(test_pd,
                                              feature_name='SMILES',
                                              label_name_list=['Keck_Pria_AS_Retest'],
                                              SMILES_mapping_json_file=SMILES_mapping_json_file)

    cross_validation_split = StratifiedShuffleSplit(y_train, 1, test_size=0.15, random_state=1)
    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t, y_val = y_train[t_index], y_train[val_index]
    print 'done data preparation'

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    task = VanillaLSTM(conf)
    X_t = sequence.pad_sequences(X_t, maxlen=task.padding_length)
    X_val = sequence.pad_sequences(X_val, maxlen=task.padding_length)
    X_test = sequence.pad_sequences(X_test, maxlen=task.padding_length)

    hyperparameter_sets = {'optimizer': ['adam', 'rmsprop'],
                           'epoch size': [200],
                           'patience': [20],
                           'early stopping': ['precision'],
                           'embedding_size': [30, 50, 100],
                           'first_hidden_size': [50, 100],
                           'second_hidden_size': [10, 50]}
    hyperparameters = ParameterGrid(hyperparameter_sets)

    cnt = 0
    for param in hyperparameters:
        if cnt != process_num:
            cnt += 1
            continue
        conf['compile']['optimizer']['option'] = param['optimizer']
        conf['fitting']['nb_epoch'] = param['epoch size']
        conf['fitting']['early_stopping']['patience'] = param['patience']
        conf['fitting']['early_stopping']['option'] = param['early stopping']
        conf['embedding_size'] = param['embedding_size']
        conf['first_hidden_size'] = param['first_hidden_size']
        conf['second_hidden_size'] = param['second_hidden_size']
        print 'Testing hyperparameter ', param
        break

    task.train_and_predict(X_t, y_t, X_val, y_val, X_test, y_test, PMTNN_weight_file)
    store_data(transform_json_to_csv(config_json_file), config_csv_file)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', action="store", dest="config_json_file", required=True)
    parser.add_argument('--PMTNN_weight_file', action="store", dest="PMTNN_weight_file", required=True)
    parser.add_argument('--config_csv_file', action="store", dest="config_csv_file", required=True)
    parser.add_argument('--process_num', action='store', dest='process_num', required=True)
    parser.add_argument('--SMILES_mapping_json_file', action='store', dest='SMILES_mapping_json_file', default= '../../json/SMILES_mapping.json')
    parser.add_argument('--model', action='store', dest='model',required=True)
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    PMTNN_weight_file = given_args.PMTNN_weight_file
    config_csv_file = given_args.config_csv_file
    process_num = int(given_args.process_num)
    SMILES_mapping_json_file = given_args.SMILES_mapping_json_file
    model = given_args.model

    if model == 'single_classification':
        run_single_classification()
    elif model == 'single_regression':
        run_single_regression()
    elif model == 'vanilla_lstm':
        run_vanilla_lstm()
    else:
        raise Exception('No such model! Should be among [{}, {}, {}].'.format(
            'single_classification',
            'single_regression',
            'vanilla_lstm'
        ))
