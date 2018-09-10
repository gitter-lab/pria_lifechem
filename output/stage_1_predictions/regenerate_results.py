import numpy as np
import pria_lifechem
import os
from pria_lifechem.function import *
from pria_lifechem.models.CallBacks import *
from pria_lifechem.models.deep_classification import *
from pria_lifechem.models.deep_regression import *


task_list = ['cross_validation_Keck_Pria_AS_Retest', 'cross_validation_Keck_FP', 'cross_validation_RMI']


def clean(list_a, list_b):
    neo_a, neo_b = [], []
    for a,b in zip(list_a, list_b):
        if np.isnan(a) or np.isnan(b):
            continue
        else:
            neo_a.append(a)
            neo_b.append(b)
    neo_a = np.array(neo_a)
    neo_b = np.array(neo_b)
    return neo_a, neo_b


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', dest="model_name", action="store", required=True)
    parser.add_argument('--task', dest="task", action="store", required=True)
    parser.add_argument('--fold_idx', dest="fold_idx", action="store", type=int, required=True)
    given_args = parser.parse_args()

    model_name = given_args.model_name
    task = given_args.task
    fold_idx = given_args.fold_idx

    data = np.load('{}/fold_{}.npz'.format(model_name, fold_idx))

    task_index = task_list.index(task)
    print task, task_index

    y_train = reshape_data_into_2_dim(data['y_train'][:, task_index])
    y_val = reshape_data_into_2_dim(data['y_val'][:, task_index])
    y_test = reshape_data_into_2_dim(data['y_test'][:, task_index])
    y_pred_on_train = reshape_data_into_2_dim(data['y_pred_on_train'][:, task_index])
    y_pred_on_val = reshape_data_into_2_dim(data['y_pred_on_val'][:, task_index])
    y_pred_on_test = reshape_data_into_2_dim(data['y_pred_on_test'][:, task_index])
    print y_train.shape, '\t', y_pred_on_train.shape, '\t', y_test.shape, '\t', y_pred_on_test.shape

    y_train, y_pred_on_train = clean(y_train, y_pred_on_train)
    y_val, y_pred_on_val = clean(y_val, y_pred_on_val)
    y_test, y_pred_on_test = clean(y_test, y_pred_on_test)
    print y_train.shape, '\t', y_pred_on_train.shape, '\t', y_test.shape, '\t', y_pred_on_test.shape

    print('train precision: {}'.format(precision_auc_single(y_train, y_pred_on_train)))
    print('train roc: {}'.format(roc_auc_single(y_train, y_pred_on_train)))
    print('train bedroc: {}'.format(bedroc_auc_single(y_train, y_pred_on_train)))
    print
    print('validation precision: {}'.format(precision_auc_single(y_val, y_pred_on_val)))
    print('validation roc: {}'.format(roc_auc_single(y_val, y_pred_on_val)))
    print('validation bedroc: {}'.format(bedroc_auc_single(y_val, y_pred_on_val)))
    print

    print('test precision: {}'.format(precision_auc_single(y_test, y_pred_on_test)))
    print('test roc: {}'.format(roc_auc_single(y_test, y_pred_on_test)))
    print('test bedroc: {}'.format(bedroc_auc_single(y_test, y_pred_on_test)))
    print


    for EF_ratio in [0.02, 0.01, 0.0015, 0.001]:
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, EF_ratio)
        nef = ef / ef_max
        print('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))
        print('ratio: {}, NEF: {}'.format(EF_ratio, nef))
