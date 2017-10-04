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

    test_file = '../../../dataset/keck_lc4.csv.gz'
    test_pd = pd.read_csv(test_file)
    molecule_id = test_pd['Molecule'].tolist()
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=label_name_list)

    task = SingleClassification(conf=conf)
    model = task.setup_model()
    model.load_weights(PMTNN_weight_file)
    y_pred_on_test = model.predict(X_test)

    np.savez_compressed(storage_file,
                        molecule_id=molecule_id,
                        y_actual=y_test,
                        y_pred=y_pred_on_test)
    return


def run_single_regression():
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

    test_file = '../../../dataset/keck_lc4.csv.gz'
    test_pd = pd.read_csv(test_file)
    molecule_id = test_pd['Molecule'].tolist()
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=label_name_list)

    task = SingleRegression(conf=conf)
    model = task.setup_model()
    model.load_weights(PMTNN_weight_file)
    y_pred_on_test = model.predict(X_test)

    np.savez_compressed(storage_file,
                        molecule_id=molecule_id,
                        y_actual=y_test,
                        y_pred=y_pred_on_test)
    return


def run_vanilla_lstm():
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

    test_file = '../../../dataset/keck_lc4.csv.gz'
    test_pd = pd.read_csv(test_file)
    molecule_id = test_pd['Molecule'].tolist()
    X_test, y_test = extract_SMILES_and_label(test_pd,
                                              feature_name='SMILES',
                                              label_name_list=label_name_list,
                                              SMILES_mapping_json_file=SMILES_mapping_json_file)

    task = VanillaLSTM(conf=conf)
    model = task.setup_model()
    model.load_weights(PMTNN_weight_file)
    X_test = sequence.pad_sequences(X_test, maxlen=task.padding_length)
    y_pred_on_test = model.predict(X_test)

    np.savez_compressed(storage_file,
                        molecule_id=molecule_id,
                        y_actual=y_test,
                        y_pred=y_pred_on_test)
    return


def run_multiple_classification():
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

    test_file = '../../../dataset/keck_lc4.csv.gz'
    test_pd = pd.read_csv(test_file)
    molecule_id = test_pd['Molecule'].tolist()
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=label_name_list)

    task = MultiClassification(conf=conf)
    model = task.setup_model()
    model.load_weights(PMTNN_weight_file)
    y_pred_on_test = reshape_data_into_2_dim(model.predict(X_test)[:, -1])

    np.savez_compressed(storage_file,
                        molecule_id=molecule_id,
                        y_actual=y_test,
                        y_pred=y_pred_on_test)
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
