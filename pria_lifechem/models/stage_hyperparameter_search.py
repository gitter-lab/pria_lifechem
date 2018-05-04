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
from pria_lifechem.function import *
from pria_lifechem.evaluation import *
from pria_lifechem.models.CallBacks import *
from pria_lifechem.models.deep_classification import *
from pria_lifechem.models.deep_regression import *
from pria_lifechem.models.vanilla_lstm import *


def get_hyperparameter_sets(model):
    if model == 'single_classification':
        hyperparameter_sets = {'optimizer': ['adam'],
                               'learning rate': [0.00003, 0.0001, 0.003],
                               'weighted schema': ['no_weight', 'weighted_sample'],
                               'epoch patience': [{'epoch_size': 200, 'patience': 50},
                                                  {'epoch_size': 1000, 'patience': 200}],
                               'early stopping': ['auc', 'precision'],
                               'activations': [{0: 'relu', 1: 'sigmoid', 2: 'sigmoid'},
                                               {0: 'relu', 1: 'relu', 2: 'sigmoid'}]}
    elif model == 'single_regression':
        hyperparameter_sets = {'optimizer': ['adam'],
                               'learning rate': [0.00003, 0.0001, 0.003],
                               'weighted schema': ['no_weight'],
                               'epoch patience': [{'epoch_size': 200, 'patience': 50},
                                                  {'epoch_size': 1000, 'patience': 200}],
                               'early stopping': ['auc', 'precision'],
                               'activations': [{0: 'sigmoid', 1: 'sigmoid', 2: 'linear'},
                                               {0: 'relu', 1: 'sigmoid', 2: 'sigmoid'}]}
    elif model == 'vanilla_lstm':
        hyperparameter_sets = {'optimizer': ['rmsprop'],
                               'epoch patience': [{'epoch_size': 200, 'patience': 50}],
                               'early stopping': ['auc'],
                               'embedding_size': [30, 50, 100],
                               'hidden_size': [
                                   [50],
                                   [100],
                                   [100, 10],
                                   [100, 50],
                                   [50, 10]
                               ],
                               'dropout': [0.2, 0.5]}
    elif model == 'multi_classification':
        hyperparameter_sets = {'optimizer': ['adam'],
                               'learning rate': [0.00003, 0.0001, 0.003],
                               'weighted schema': ['no_weight', 'weighted_sample'],
                               'epoch patience': [{'epoch_size': 200, 'patience': 50},
                                                  {'epoch_size': 1000, 'patience': 200}],
                               'early stopping': ['precision'],
                               'activations': [{0: 'relu', 1: 'sigmoid', 2: 'sigmoid'},
                                               {0: 'relu', 1: 'relu', 2: 'sigmoid'}]}
    elif model == 'random_forest':
        hyperparameter_sets = {"n_estimators": [4000, 8000, 16000],
                                "max_features": ["None", "sqrt", "log2"],
                                "min_samples_leaf": [1, 10, 100, 1000],
                                "class_weight": ["None", "balanced_subsample","balanced"]
        
        }
    else:
        raise Exception('No such model! Should be among [{}, {}, {}, {}].'.format(
            'single_classification',
            'single_regression',
            'vanilla_lstm',
            'multi_classification'
        ))
    return hyperparameter_sets


def run_single_classification(hyperparameter_sets, hyperparameter_index):
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

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
                                                 label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=label_name_list)
    cross_validation_split = StratifiedShuffleSplit(y_train, 1, test_size=test_ratio, random_state=1)
    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t, y_val = y_train[t_index], y_train[val_index]
    print 'done data preparation'

    hyperparameters = ParameterGrid(hyperparameter_sets)

    cnt = 0
    for param in hyperparameters:
        if cnt != hyperparameter_index:
            cnt += 1
            continue
        conf['compile']['optimizer']['option'] = param['optimizer']
        conf['compile']['optimizer'][param['optimizer']]['lr'] = param['learning rate']
        conf['class_weight_option'] = param['weighted schema']
        epoch_patience = param['epoch patience']
        conf['fitting']['nb_epoch'] = epoch_patience['epoch_size']
        conf['fitting']['early_stopping']['patience'] = epoch_patience['patience']
        conf['fitting']['early_stopping']['option'] = param['early stopping']
        activations = param['activations']
        conf['layers'][0]['activation'] = activations[0]
        conf['layers'][1]['activation'] = activations[1]
        conf['layers'][2]['activation'] = activations[2]
        print 'Testing hyperparameter ', param
        break

    if cnt > hyperparameter_index:
        raise ValueError('Process number out of limit. At most {}.'.format(cnt))

    task = SingleClassification(conf=conf)
    task.train_and_predict(X_t, y_t, X_val, y_val, X_test, y_test, PMTNN_weight_file)
    store_data(transform_json_to_csv(config_json_file), config_csv_file)

    return


def run_single_regression(hyperparameter_sets, hyperparameter_index):
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

    # specify dataset
    k = 5
    directory = '../../dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    output_file_list = [directory + f_ for f_ in file_list]
    print output_file_list[0:4]
    train_pd = read_merged_data(output_file_list[0:4])
    print output_file_list[4]
    test_pd = read_merged_data([output_file_list[4]])

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=label_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=label_name_list)
    y_train_classification = reshape_data_into_2_dim(y_train[:, 0])
    y_train_regression = reshape_data_into_2_dim(y_train[:, 1])
    y_test_classification = reshape_data_into_2_dim(y_test[:, 0])
    y_test_regression = reshape_data_into_2_dim(y_test[:, 1])

    cross_validation_split = StratifiedShuffleSplit(y_train_classification, 1, test_size=test_ratio, random_state=1)

    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t_classification, y_val_classification = y_train_classification[t_index], y_train_classification[val_index]
        y_t_regression, y_val_regression = y_train_regression[t_index], y_train_regression[val_index]
    print 'done data preparation'

    hyperparameters = ParameterGrid(hyperparameter_sets)

    cnt = 0
    for param in hyperparameters:
        if cnt != hyperparameter_index:
            cnt += 1
            continue
        conf['compile']['optimizer']['option'] = param['optimizer']
        conf['compile']['optimizer'][param['optimizer']]['lr'] = param['learning rate']
        conf['class_weight_option'] = param['weighted schema']
        epoch_patience = param['epoch patience']
        conf['fitting']['nb_epoch'] = epoch_patience['epoch_size']
        conf['fitting']['early_stopping']['patience'] = epoch_patience['patience']
        conf['fitting']['early_stopping']['option'] = param['early stopping']
        activations = param['activations']
        conf['layers'][0]['activation'] = activations[0]
        conf['layers'][1]['activation'] = activations[1]
        conf['layers'][2]['activation'] = activations[2]
        print 'Testing hyperparameter ', param
        break

    if cnt > hyperparameter_index:
        raise ValueError('Process number out of limit. At most {}.'.format(cnt))

    task = SingleRegression(conf=conf)
    task.train_and_predict(X_t, y_t_regression, y_t_classification,
                           X_val, y_val_regression, y_val_classification,
                           X_test, y_test_regression, y_test_classification,
                           PMTNN_weight_file)
    store_data(transform_json_to_csv(config_json_file), config_csv_file)

    return


def run_vanilla_lstm(hyperparameter_sets, hyperparameter_index):
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

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
                                                label_name_list=label_name_list,
                                                SMILES_mapping_json_file=SMILES_mapping_json_file)
    X_test, y_test = extract_SMILES_and_label(test_pd,
                                              feature_name='SMILES',
                                              label_name_list=label_name_list,
                                              SMILES_mapping_json_file=SMILES_mapping_json_file)

    cross_validation_split = StratifiedShuffleSplit(y_train, 1, test_size=test_ratio, random_state=1)
    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t, y_val = y_train[t_index], y_train[val_index]
    print 'done data preparation'

    hyperparameters = ParameterGrid(hyperparameter_sets)

    cnt = 0
    for param in hyperparameters:
        if cnt != hyperparameter_index:
            cnt += 1
            continue
        conf['lstm']['embedding_size'] = param['embedding_size']
        conf['lstm']['layer_num'] = len(param['hidden_size'])
        conf['compile']['optimizer']['option'] = param['optimizer']
        epoch_patience = param['epoch patience']
        conf['fitting']['nb_epoch'] = epoch_patience['epoch_size']
        conf['fitting']['early_stopping']['patience'] = epoch_patience['patience']
        conf['fitting']['early_stopping']['option'] = param['early stopping']
        for i in range(conf['lstm']['layer_num']):
            conf['layers'][i]['hidden_size'] = param['hidden_size'][i]
            conf['layers'][i]['dropout_U'] = param['dropout']
            conf['layers'][i]['dropout_W'] = param['dropout']
        print 'Testing hyperparameter ', param
        break

    if cnt > hyperparameter_index:
        raise ValueError('Process number out of limit. At most {}.'.format(cnt))

    task = VanillaLSTM(conf)
    X_t = sequence.pad_sequences(X_t, maxlen=task.padding_length)
    X_val = sequence.pad_sequences(X_val, maxlen=task.padding_length)
    X_test = sequence.pad_sequences(X_test, maxlen=task.padding_length)

    task.train_and_predict(X_t, y_t, X_val, y_val, X_test, y_test, PMTNN_weight_file)
    store_config(conf, config_csv_file)

    return



def run_multiple_classification(hyperparameter_sets, hyperparameter_index):
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

    # specify dataset
    k = 5
    directory = '../../dataset/keck_pcba/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    output_file_list = [directory + f_ for f_ in file_list]
    train_pd = read_merged_data(output_file_list[0:3])
    train_pd.fillna(0, inplace=True)
    val_pd = read_merged_data(output_file_list[3:4])
    val_pd.fillna(0, inplace=True)
    test_pd = read_merged_data(output_file_list[4:5])
    test_pd.fillna(0, inplace=True)

    multi_name_list = train_pd.columns[-128:].tolist()
    multi_name_list.extend(label_name_list)
    print 'multi_name_list ', multi_name_list

    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=multi_name_list)
    X_val, y_val = extract_feature_and_label(val_pd,
                                             feature_name='Fingerprints',
                                             label_name_list=multi_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=multi_name_list)

    sample_weight_dir = '../../dataset/sample_weights/keck_pcba/fold_5/'
    file_list = []
    for i in range(k):
        file_list.append('sample_weight_{}.csv'.format(i))
    sample_weight_file = [sample_weight_dir + f_ for f_ in file_list]
    sample_weight_file = np.array(sample_weight_file)
    sample_weight_pd = read_merged_data(sample_weight_file[0:3])
    _, sample_weight = extract_feature_and_label(sample_weight_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=multi_name_list)
    print 'done data preparation'

    hyperparameters = ParameterGrid(hyperparameter_sets)

    cnt = 0
    for param in hyperparameters:
        if cnt != hyperparameter_index:
            cnt += 1
            continue
        conf['compile']['optimizer']['option'] = param['optimizer']
        conf['compile']['optimizer'][param['optimizer']]['lr'] = param['learning rate']
        conf['class_weight_option'] = param['weighted schema']
        epoch_schema = param['epoch patience']
        conf['fitting']['nb_epoch'] = epoch_schema['epoch_size']
        conf['fitting']['early_stopping']['patience'] = epoch_schema['patience']
        conf['fitting']['early_stopping']['option'] = param['early stopping']
        activations = param['activations']
        conf['layers'][0]['activation'] = activations[0]
        conf['layers'][1]['activation'] = activations[1]
        conf['layers'][2]['activation'] = activations[2]
        print 'Testing hyperparameter ', param
        break

    if cnt > hyperparameter_index:
        raise ValueError('Process number out of limit. At most {}.'.format(cnt))

    task = MultiClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test,
                           sample_weight=sample_weight,
                           PMTNN_weight_file=PMTNN_weight_file,
                           score_file=score_file)
    store_data(transform_json_to_csv(config_json_file), config_csv_file)

    return


def run_hyperparameter_sets(model):
    global test_ratio
    test_ratio = 0.2

    if model == 'single_classification':
        hyperparameter_sets = get_hyperparameter_sets(model)
        run_single_classification(hyperparameter_sets, process_num)
    elif model == 'single_regression':
        hyperparameter_sets = get_hyperparameter_sets(model)
        run_single_regression(hyperparameter_sets, process_num)
    elif model == 'vanilla_lstm':
        hyperparameter_sets = get_hyperparameter_sets(model)
        global SMILES_mapping_json_file
        SMILES_mapping_json_file = given_args.SMILES_mapping_json_file
        run_vanilla_lstm(hyperparameter_sets, process_num)
    elif model == 'multi_classification':
        hyperparameter_sets = get_hyperparameter_sets(model)
        global score_file
        score_file = given_args.score_file
        run_multiple_classification(hyperparameter_sets, process_num)
    else:
        raise Exception('No such model! Should be among [{}, {}, {}, {}].'.format(
            'single_classification',
            'single_regression',
            'vanilla_lstm',
            'multi_classification'
        ))
    return


def transform_into_md(list_):
    str_ = '|'
    for val in list_:
        str_ = '{} {} |'.format(str_, val)
    return str_


def get_hyperparameter_sets_in_markdown(model):
    hyperparameter_sets = get_hyperparameter_sets(model)
    hyperparameters = ParameterGrid(hyperparameter_sets)

    content = ''
    for param in hyperparameters:
        keys = param.keys()
        header = '| {} {}'.format('count', transform_into_md(keys))
        dividing_line = transform_into_md(['---' for _ in range(1+len(keys))])
        content = '{}\n{}'.format(header, dividing_line)
        break

    count = 0
    for param in hyperparameters:
        values = param.values()
        content = '{}\n| {} {}'.format(content, count, transform_into_md(values))
        count += 1
    return content


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', dest="config_json_file",
                        action="store", required=False)
    parser.add_argument('--PMTNN_weight_file', dest="PMTNN_weight_file",
                        action="store", required=False)
    parser.add_argument('--config_csv_file', dest="config_csv_file",
                        action="store", required=False)
    parser.add_argument('--process_num', dest='process_num', type=int,
                        action='store', required=False)
    parser.add_argument('--SMILES_mapping_json_file', dest='SMILES_mapping_json_file',
                        action='store', required=False, default= '../../json/SMILES_mapping.json')
    parser.add_argument('--score_file', dest='score_file',
                        action='store', required=False)
    parser.add_argument('--model', dest='model',
                        action='store', required=True)
    parser.add_argument('--is_print', dest='is_print', type=bool,
                        action='store', required=False, default=False)

    given_args = parser.parse_args()
    model = given_args.model
    is_print = given_args.is_print

    if is_print:
        content = get_hyperparameter_sets_in_markdown(model)
        print content
    else:
        config_json_file = given_args.config_json_file
        PMTNN_weight_file = given_args.PMTNN_weight_file
        config_csv_file = given_args.config_csv_file
        process_num = given_args.process_num
        run_hyperparameter_sets(model)

