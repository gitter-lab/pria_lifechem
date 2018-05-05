import re
import os
import numpy as np
import seaborn as sns
import pandas as pd
from pria_lifechem.function import *
from pria_lifechem.evaluation import *
from pria_lifechem.models.CallBacks import *
from pria_lifechem.models.deep_classification import *
from pria_lifechem.models.deep_regression import *
from pria_lifechem.models.vanilla_lstm import *

from pria_lifechem.evaluation import roc_auc_single, roc_auc_multi, bedroc_auc_multi, bedroc_auc_single, \
    precision_auc_multi, precision_auc_single, enrichment_factor_multi, enrichment_factor_single

def get_EF_values_single_task(task, X_test, y_test, model_weight, EF_ratio_list):
    model = task.setup_model()
    model.load_weights(model_weight)
    y_pred_on_test = model.predict(X_test)

    print('test precision: {}'.format(precision_auc_single(y_test, y_pred_on_test)))
    print('test roc: {}'.format(roc_auc_single(y_test, y_pred_on_test)))
    print('test bedroc: {}'.format(bedroc_auc_single(y_test, y_pred_on_test)))
    print

    ef_values = []
    ef_max_values = []
    for EF_ratio in EF_ratio_list:
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, EF_ratio)
        ef_values.append(ef)
        ef_max_values.append(ef_max)
    return ef_values, ef_max_values


def get_EF_values_multi_task(task, X_test, y_test, model_weight, EF_ratio_list):
    model = task.setup_model()
    model.load_weights(model_weight)
    y_pred_on_test = model.predict(X_test)
    print 'predicted shape\t', y_pred_on_test.shape
    y_test = reshape_data_into_2_dim(y_test[:, -1])
    y_pred_on_test = reshape_data_into_2_dim(y_pred_on_test[:, -1])

    print('test precision: {}'.format(precision_auc_single(y_test, y_pred_on_test)))
    print('test roc: {}'.format(roc_auc_single(y_test, y_pred_on_test)))
    print('test bedroc: {}'.format(bedroc_auc_single(y_test, y_pred_on_test)))
    print

    ef_values = []
    ef_max_values = []
    for EF_ratio in EF_ratio_list:
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, EF_ratio)
        ef_values.append(ef)
        ef_max_values.append(ef_max)
    return ef_values, ef_max_values


def get_EF_scores_single_classification(config_json_file, PMTNN_weight_file_path, EF_ratio_list, model_name):
    file_list = ['../../dataset/fixed_dataset/fold_5/file_{}.csv'.format(i) for i in range(5)]

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'Testing name_list: ', label_name_list
    extractor = ['Fingerprints']
    extractor.extend(label_name_list)

    data_pd_list = []
    for i in range(5):
        temp_file_list = file_list[i:i + 1]
        temp = read_merged_data(temp_file_list, usecols=extractor)
        temp.dropna(axis=0, subset=label_name_list, how='any', inplace=True) # Need to remove all the missing values
        data_pd_list.append(temp)

    EF_values_list = []
    EF_max_values_list = []
    EF_ratio_values_list = []
    running_process_list = []
    model_name_list = []

    for running_index in range(20):
        print 'running index ', running_index
        PMTNN_weight_file = PMTNN_weight_file_path + '{}.weight'.format(running_index)
        test_index = running_index / 4
        test_pd = data_pd_list[test_index]

        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=label_name_list)

        task = SingleClassification(conf=conf)
        EF_scores, EF_max = get_EF_values_single_task(task, X_test, y_test, PMTNN_weight_file, EF_ratio_list)

        EF_values_list.extend(EF_scores)
        EF_max_values_list.extend(EF_max)
        EF_ratio_values_list.extend(EF_ratio_list)
        running_process_list.extend([running_index for _ in EF_scores])
        model_name_list.extend([model_name for _ in EF_scores])

    return EF_values_list, EF_max_values_list, EF_ratio_values_list, running_process_list, model_name_list


def get_EF_scores_single_regression(config_json_file, PMTNN_weight_file_path, EF_ratio_list, model_name):
    file_list = ['../../dataset/fixed_dataset/fold_5/file_{}.csv'.format(i) for i in range(5)]

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'Testing name_list: ', label_name_list
    extractor = ['Fingerprints']
    extractor.extend(label_name_list)

    data_pd_list = []
    for i in range(5):
        temp_file_list = file_list[i:i + 1]
        temp = read_merged_data(temp_file_list, usecols=extractor)
        temp.dropna(axis=0, subset=label_name_list, how='any', inplace=True) # Need to remove all the missing values
        data_pd_list.append(temp)

    EF_values_list = []
    EF_max_values_list = []
    EF_ratio_values_list = []
    running_process_list = []
    model_name_list = []

    for running_index in range(20):
        print 'running index ', running_index
        PMTNN_weight_file = PMTNN_weight_file_path + '{}.weight'.format(running_index)
        test_index = running_index / 4
        test_pd = data_pd_list[test_index]

        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=label_name_list)

        y_test_classification = reshape_data_into_2_dim(y_test[:, 0])

        task = SingleClassification(conf=conf)
        EF_scores, EF_max = get_EF_values_single_task(task, X_test, y_test_classification, PMTNN_weight_file,
                                                      EF_ratio_list)

        EF_values_list.extend(EF_scores)
        EF_max_values_list.extend(EF_max)
        EF_ratio_values_list.extend(EF_ratio_list)
        running_process_list.extend([running_index for _ in EF_scores])
        model_name_list.extend([model_name for _ in EF_scores])

    return EF_values_list, EF_max_values_list, EF_ratio_values_list, running_process_list, model_name_list


def get_EF_score_multi_task(config_json_file, PMTNN_weight_file_path, EF_ratio_list, model_name):
    file_list = ['../../dataset/keck_pcba/fold_5/file_{}.csv'.format(i) for i in range(5)]

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'Testing name_list: ', label_name_list
    extractor = ['Fingerprints']
    extractor.extend(label_name_list)

    data_pd_list = []
    for i in range(5):
        temp_file_list = file_list[i:i + 1]
        temp = read_merged_data(temp_file_list, usecols=extractor)
        temp.dropna(axis=0, subset=label_name_list, how='any', inplace=True) # Need to remove all the missing values
        data_pd_list.append(temp)

    EF_values_list = []
    EF_max_values_list = []
    EF_ratio_values_list = []
    running_process_list = []
    model_name_list = []

    for running_index in range(20):
        print 'running index ', running_index
        PMTNN_weight_file = PMTNN_weight_file_path + '{}.weight'.format(running_index)
        test_index = running_index / 4
        test_pd = data_pd_list[test_index]

        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=label_name_list)

        task = MultiClassification(conf=conf)
        EF_scores, EF_max = get_EF_values_multi_task(task, X_test, y_test, PMTNN_weight_file,
                                                     EF_ratio_list)

        EF_values_list.extend(EF_scores)
        EF_max_values_list.extend(EF_max)
        EF_ratio_values_list.extend(EF_ratio_list)
        running_process_list.extend([running_index for _ in EF_scores])
        model_name_list.extend([model_name for _ in EF_scores])

    return EF_values_list, EF_max_values_list, EF_ratio_values_list, running_process_list, model_name_list


def get_EF_curve_in_pd(EF_ratio_list, data_set_name, weight_dir, config_json_file, model_name, regenerate=False):
    save_pd_path = './EF_curve_preparation/{}/{}.csv'.format(data_set_name, model_name)

    if os.path.isfile(save_pd_path) and not regenerate:
        data_pd = pd.read_csv(save_pd_path)
        return data_pd

    if 'single_classification' in model_name:
        func = get_EF_scores_single_classification
    elif 'single_regression' in model_name:
        func = get_EF_scores_single_regression
    elif 'multi_classification' in model_name:
        func = get_EF_score_multi_task
    else:
        raise Exception('No such model! Should be among [{}, {}, {}, {}].'.format(
            'single_classification',
            'single_regression',
            'vanilla_lstm',
            'multi_classification'
        ))

    print 'running {}'.format(model_name)
    EF_values_list, EF_max_values_list, EF_ratio_values_list, \
    running_process_list, model_name_list = func(config_json_file=config_json_file,
                                                 PMTNN_weight_file_path=weight_dir,
                                                 EF_ratio_list=EF_ratio_list,
                                                 model_name=model_name)

    data_pd = pd.DataFrame({'EF': EF_values_list,
                            'EF max': EF_max_values_list,
                            'EFR': EF_ratio_values_list,
                            'running process': running_process_list,
                            'model': model_name_list})

    data_pd.to_csv(save_pd_path, index=None)

    return data_pd