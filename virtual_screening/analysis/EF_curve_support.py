import re
import os
import numpy as np
import seaborn as sns
import pandas as pd
from virtual_screening.function import *
from virtual_screening.evaluation import *
from virtual_screening.models.CallBacks import *
from virtual_screening.models.deep_classification import *
from virtual_screening.models.deep_regression import *
from virtual_screening.models.vanilla_lstm import *


def get_EF_values_single_task(task, X_test, y_test, model_weight, EF_ratio_list):
    model = task.setup_model()
    model.load_weights(model_weight)
    y_pred_on_test = model.predict(X_test)

    ef_values = []
    ef_max_values = []
    for EF_ratio in EF_ratio_list:
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, EF_ratio)
        ef_values.append(ef)
        ef_max_values.append(ef_max)
    return ef_values, ef_max_values


def get_EF_scores_single_classification(config_json_file, PMTNN_weight_file_path, EF_ratio_list, model_name):
    file_list = ['../../dataset/fixed_dataset/fold_5/file_{}.csv'.format(i) for i in range(5)]
    data_pd_list = []
    for i in range(5):
        temp_file_list = file_list[i:i + 1]
        temp = read_merged_data(temp_file_list)
        data_pd_list.append(temp)

    EF_values_list = []
    EF_ratio_values_list = []
    running_process_list = []
    model_name_list = []

    for running_index in range(20):
        print 'running index ', running_index
        PMTNN_weight_file = PMTNN_weight_file_path + '{}.weight'.format(running_index)
        test_index = running_index / 4

        test_pd = data_pd_list[test_index]

        with open(config_json_file, 'r') as f:
            conf = json.load(f)
        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=['Keck_Pria_AS_Retest'])

        task = SingleClassification(conf=conf)
        EF_scores, EF_max = get_EF_values_single_task(task, X_test, y_test, PMTNN_weight_file, EF_ratio_list)

        EF_values_list.extend(EF_scores)
        EF_ratio_values_list.extend(EF_ratio_list)
        running_process_list.extend([running_index for _ in EF_scores])
        model_name_list.extend([model_name for _ in EF_scores])

    return EF_values_list, EF_ratio_values_list, running_process_list, model_name_list


def get_EF_scores_single_regression(config_json_file, PMTNN_weight_file_path, EF_ratio_list, model_name):
    file_list = ['../../dataset/fixed_dataset/fold_5/file_{}.csv'.format(i) for i in range(5)]
    data_pd_list = []
    for i in range(5):
        temp_file_list = file_list[i:i + 1]
        temp = read_merged_data(temp_file_list)
        data_pd_list.append(temp)

    EF_values_list = []
    EF_ratio_values_list = []
    running_process_list = []
    model_name_list = []

    for running_index in range(20):
        print 'running index ', running_index
        PMTNN_weight_file = PMTNN_weight_file_path + '{}.weight'.format(running_index)
        test_index = running_index / 4

        test_pd = data_pd_list[test_index]

        with open(config_json_file, 'r') as f:
            conf = json.load(f)

        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=['Keck_Pria_AS_Retest', 'Keck_Pria_Continuous'])

        y_test_classification = reshape_data_into_2_dim(y_test[:, 0])

        task = SingleClassification(conf=conf)
        EF_scores, EF_max = get_EF_values_single_task(task, X_test, y_test_classification, PMTNN_weight_file,
                                                      EF_ratio_list)

        EF_values_list.extend(EF_scores)
        EF_ratio_values_list.extend(EF_ratio_list)
        running_process_list.extend([running_index for _ in EF_scores])
        model_name_list.extend([model_name for _ in EF_scores])

    return EF_values_list, EF_ratio_values_list, running_process_list, model_name_list


def get_EF_curve_in_pd(EF_ratio_list, weight_dir, config_json_file, model_name, regenerate=False):
    save_pd_path = './temp/{}'.format(model_name)

    if os.path.isfile(save_pd_path) and not regenerate:
        data_pd = pd.read_csv(save_pd_path)
        return data_pd

    if 'single_classification' in model_name:
        func = get_EF_scores_single_classification
    elif 'single_regression' in model_name:
        func = get_EF_scores_single_regression
    else:
        raise Exception('No such model! Should be among [{}, {}, {}, {}].'.format(
            'single_classification',
            'single_regression',
            'vanilla_lstm',
            'multi_classification'
        ))

    EF_values_list, EF_ratio_values_list, \
    running_process_list, model_name_list = func(config_json_file=config_json_file,
                                                 PMTNN_weight_file_path=weight_dir,
                                                 EF_ratio_list=EF_ratio_list,
                                                 model_name=model_name)

    data_pd = pd.DataFrame({'EF': EF_values_list,
                            'EFR': EF_ratio_values_list,
                            'running process': running_process_list,
                            'model': model_name_list})

    data_pd.to_csv(save_pd_path, index=None)

    return data_pd