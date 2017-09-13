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

from virtual_screening.evaluation import enrichment_factor_single


def enrichement_factor_fetcher(y_test, y_pred_on_test, EF_ratio_list):
    ef_values = []
    ef_max_values = []
    for EF_ratio in EF_ratio_list:
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, EF_ratio)
        ef_values.append(ef)
        ef_max_values.append(ef_max)
    return ef_values, ef_max_values


def get_EF_scores_single_classification(model_name, EF_ratio_list, predictions_path, N=5):
    file_dir = '{}/{}'.format(predictions_path, model_name)
    file_paths = [file_dir+'/fold_{}.npz'.format(i) for i in range(N)]

    EF_values_list = []
    EF_max_values_list = []
    EF_ratio_values_list = []
    running_process_list = []
    model_name_list = []

    for running_index in range(N):
        data = np.load(file_paths[running_index])
        test_true_labels = data['y_test']
        test_pred_labels = data['y_pred_on_test']
        EF_scores, EF_max = enrichement_factor_fetcher(test_true_labels, test_pred_labels, EF_ratio_list)
        EF_values_list.extend(EF_scores)
        EF_max_values_list.extend(EF_max)
        EF_ratio_values_list.extend(EF_ratio_list)
        running_process_list.extend([running_index for _ in EF_scores])
        model_name_list.extend([model_name for _ in EF_scores])

    return EF_values_list, EF_max_values_list, EF_ratio_values_list, running_process_list, model_name_list


def get_EF_scores_single_regression(model_name, EF_ratio_list, predictions_path, N=5):
    file_dir = '{}/{}'.format(predictions_path, model_name)
    file_paths = [file_dir+'/fold_{}.npz'.format(i) for i in range(N)]

    EF_values_list = []
    EF_max_values_list = []
    EF_ratio_values_list = []
    running_process_list = []
    model_name_list = []

    for running_index in range(N):
        data = np.load(file_paths[running_index])
        test_true_labels = data['y_test']
        test_pred_labels = data['y_pred_on_test']
        EF_scores, EF_max = enrichement_factor_fetcher(test_true_labels, test_pred_labels, EF_ratio_list)
        EF_values_list.extend(EF_scores)
        EF_max_values_list.extend(EF_max)
        EF_ratio_values_list.extend(EF_ratio_list)
        running_process_list.extend([running_index for _ in EF_scores])
        model_name_list.extend([model_name for _ in EF_scores])

    return EF_values_list, EF_max_values_list, EF_ratio_values_list, running_process_list, model_name_list


def get_EF_score_multi_task(model_name, EF_ratio_list, predictions_path, N=5):
    file_dir = '{}/{}'.format(predictions_path, model_name)
    file_paths = [file_dir+'/fold_{}.npz'.format(i) for i in range(N)]

    EF_values_list = []
    EF_max_values_list = []
    EF_ratio_values_list = []
    running_process_list = []
    model_name_list = []

    for running_index in range(N):
        data = np.load(file_paths[running_index])
        test_true_labels = data['y_test']
        test_pred_labels = data['y_pred_on_test']
        EF_scores, EF_max = enrichement_factor_fetcher(test_true_labels, test_pred_labels, EF_ratio_list)
        EF_values_list.extend(EF_scores)
        EF_max_values_list.extend(EF_max)
        EF_ratio_values_list.extend(EF_ratio_list)
        running_process_list.extend([running_index for _ in EF_scores])
        model_name_list.extend([model_name for _ in EF_scores])

    return EF_values_list, EF_max_values_list, EF_ratio_values_list, running_process_list, model_name_list


def get_EF_score_random_forest(model_name, EF_ratio_list, predictions_path, N=5):
    file_dir = '{}/{}'.format(predictions_path, model_name)
    file_paths = [file_dir+'/fold_{}.npz'.format(i) for i in range(N)]

    EF_values_list = []
    EF_max_values_list = []
    EF_ratio_values_list = []
    running_process_list = []
    model_name_list = []

    for running_index in range(N):
        data = np.load(file_paths[running_index])
        test_true_labels = data['y_test']
        test_pred_labels = data['y_pred_on_test']
        EF_scores, EF_max = enrichement_factor_fetcher(test_true_labels, test_pred_labels, EF_ratio_list)
        EF_values_list.extend(EF_scores)
        EF_max_values_list.extend(EF_max)
        EF_ratio_values_list.extend(EF_ratio_list)
        running_process_list.extend([running_index for _ in EF_scores])
        model_name_list.extend([model_name for _ in EF_scores])

    return EF_values_list, EF_max_values_list, EF_ratio_values_list, running_process_list, model_name_list


def get_EF_curve_in_pd(EF_ratio_list, data_set_name, predictions_path, model_name, regenerate=False):
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
    elif 'sklearn_rf' in model_name:
        func = get_EF_score_random_forest
    else:
        raise Exception('No such model! Should be among [{}, {}, {}, {}, {}].'.format(
            'single_classification',
            'single_regression',
            'vanilla_lstm',
            'multi_classification',
            'sklearn_rf'
        ))

    print 'running {}'.format(model_name)
    EF_values_list, EF_max_values_list, EF_ratio_values_list, \
    running_process_list, model_name_list = func(model_name=model_name,
                                                 EF_ratio_list=EF_ratio_list,
                                                 predictions_path=predictions_path)

    data_pd = pd.DataFrame({'EF': EF_values_list,
                            'EF max': EF_max_values_list,
                            'EFR': EF_ratio_values_list,
                            'running process': running_process_list,
                            'model': model_name_list})

    data_pd.to_csv(save_pd_path, index=None)

    return data_pd