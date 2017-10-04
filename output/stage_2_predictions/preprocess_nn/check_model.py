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
from virtual_screening.function import *
from virtual_screening.evaluation import *
from virtual_screening.models.CallBacks import *
from virtual_screening.models.deep_classification import *
from virtual_screening.models.deep_regression import *
from virtual_screening.models.vanilla_lstm import *
from virtual_screening.models.tree_net import *
from virtual_screening.models.dnn_rf import *


def run_single_classification():
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

    k = 5
    directory = '../../../dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('{}file_{}.csv'.format(directory, i))
    file_list = np.array(file_list)

    # read data
    val_index = 0
    complete_index = np.arange(k)
    train_index = np.where((complete_index != val_index))[0]
    print train_index

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index:val_index + 1]

    print 'train files ', train_file_list
    print 'val files ', val_file_list

    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    val_pd = filter_out_missing_values(read_merged_data(val_file_list), label_list=label_name_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=label_name_list)
    X_val, y_val = extract_feature_and_label(val_pd,
                                             feature_name='Fingerprints',
                                             label_name_list=label_name_list)

    print 'done data preparation'
    task = SingleClassification(conf=conf)
    task.predict_with_existing(X_train, y_train, X_val, y_val,
                               X_test=None, y_test=None,
                               PMTNN_weight_file=PMTNN_weight_file)
    return


def run_single_regression():
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

    k = 5
    directory = '../../../dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('{}file_{}.csv'.format(directory, i))
    file_list = np.array(file_list)

    # read data
    val_index = 0
    complete_index = np.arange(k)
    train_index = np.where(complete_index != val_index)[0]
    print train_index

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index:val_index + 1]

    print 'train files ', train_file_list
    print 'val files ', val_file_list

    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    val_pd = filter_out_missing_values(read_merged_data(val_file_list), label_list=label_name_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=label_name_list)
    X_val, y_val = extract_feature_and_label(val_pd,
                                             feature_name='Fingerprints',
                                             label_name_list=label_name_list)

    y_train_classification = reshape_data_into_2_dim(y_train[:, 0])
    y_train_regression = reshape_data_into_2_dim(y_train[:, 1])
    y_val_classification = reshape_data_into_2_dim(y_val[:, 0])
    y_val_regression = reshape_data_into_2_dim(y_val[:, 1])
    print 'done data preparation'

    task = SingleRegression(conf=conf)

    task.predict_with_existing(X_train, y_train_regression, y_train_classification,
                               X_val, y_val_regression, y_val_classification,
                               X_test=None, y_test_regression=None, y_test_classification=None,
                               PMTNN_weight_file=PMTNN_weight_file)
    return


def run_vanilla_lstm():
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

    k = 5
    directory = '../../../dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('{}file_{}.csv'.format(directory, i))
    file_list = np.array(file_list)

    # read data
    val_index = 0
    complete_index = np.arange(k)
    train_index = np.where(complete_index != val_index)[0]
    print train_index

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index:val_index + 1]

    print 'train files ', train_file_list
    print 'val files ', val_file_list

    # TODO: No validation set for LSTM, may merge with train set
    train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
    val_pd = filter_out_missing_values(read_merged_data(val_file_list), label_list=label_name_list)

    # extract data, and split training data into training and val
    X_train, y_train = extract_SMILES_and_label(train_pd,
                                                feature_name='SMILES',
                                                label_name_list=label_name_list,
                                                SMILES_mapping_json_file=SMILES_mapping_json_file)
    X_val, y_val = extract_SMILES_and_label(val_pd,
                                            feature_name='SMILES',
                                            label_name_list=label_name_list,
                                            SMILES_mapping_json_file=SMILES_mapping_json_file)
    print 'done data preparation'

    task = VanillaLSTM(conf)
    X_train = sequence.pad_sequences(X_train, maxlen=task.padding_length)
    X_val = sequence.pad_sequences(X_val, maxlen=task.padding_length)

    task.predict_with_existing(X_train, y_train, X_val, y_val,
                               X_test=None, y_test=None, PMTNN_weight_file=PMTNN_weight_file)
    return


def run_multiple_classification():
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

    k = 5
    directory = '../../../dataset/keck_pcba/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('{}file_{}.csv'.format(directory, i))
    file_list = np.array(file_list)

    # read data
    val_index = 0
    complete_index = np.arange(k)
    train_index = np.where(complete_index != val_index)[0]
    print train_index

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index:val_index + 1]

    print 'train files ', train_file_list
    print 'val files ', val_file_list

    train_pd = read_merged_data(train_file_list)
    train_pd.fillna(0, inplace=True)
    val_pd = read_merged_data(val_file_list)
    val_pd.fillna(0, inplace=True)

    multi_name_list = train_pd.columns[-128:].tolist()
    multi_name_list.extend(label_name_list)
    print 'multi_name_list ', multi_name_list

    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=multi_name_list)
    X_val, y_val = extract_feature_and_label(val_pd,
                                             feature_name='Fingerprints',
                                             label_name_list=multi_name_list)

    sample_weight_dir = '../../../dataset/sample_weights/keck_pcba/fold_5/'
    file_list = []
    for i in range(k):
        file_list.append('sample_weight_{}.csv'.format(i))
    sample_weight_file = [sample_weight_dir + f_ for f_ in file_list]
    sample_weight_file = np.array(sample_weight_file)
    sample_weight_pd = read_merged_data(sample_weight_file[train_index])
    _, sample_weight = extract_feature_and_label(sample_weight_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=multi_name_list)
    print 'done data preparation'

    task = MultiClassification(conf=conf)
    task.predict_with_existing(X_train, y_train, X_val, y_val,
                               X_test=None, y_test=None,
                               sample_weight=sample_weight,
                               PMTNN_weight_file=PMTNN_weight_file,
                               score_file='score_file')

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', dest="config_json_file",
                        action="store", required=True)
    parser.add_argument('--PMTNN_weight_file', dest="PMTNN_weight_file",
                        action="store", required=True)
    parser.add_argument('--storage_file', dest="storage_file",
                        action="store", required=True)
    parser.add_argument('--SMILES_mapping_json_file', dest='SMILES_mapping_json_file',
                        action='store', required=False, default='../../../json/SMILES_mapping.json')
    parser.add_argument('--model', dest='model',
                        action='store', required=True)
    parser.add_argument('--cross_validation_upper_bound', dest='cross_validation_upper_bound', type=int,
                        action='store', required=False, default=20)
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    PMTNN_weight_file = given_args.PMTNN_weight_file
    storage_file = given_args.storage_file
    cross_validation_upper_bound = given_args.cross_validation_upper_bound

    model = given_args.model

    if model == 'single_classification':
        run_single_classification()
    elif model == 'single_regression':
        run_single_regression()
    elif model == 'vanilla_lstm':
        SMILES_mapping_json_file = given_args.SMILES_mapping_json_file
        run_vanilla_lstm()
    elif model == 'multi_classification':
        run_multiple_classification()
    else:
        raise Exception('No such model! Should be among [{}, {}, {}, {}, {}].'.format(
            'single_classification',
            'single_regression',
            'vanilla_lstm',
            'multi_classification'
        ))
