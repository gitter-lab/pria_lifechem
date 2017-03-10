import pandas as pd
import csv
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def roc_auc_multi(y_true, y_pred, eval_indices, eval_mean_or_median):
    '''
    this if for multi-task evaluation
    y_true and y_pred is two-dimension matrix
    can evaluate on mean or median of array
    call by
    roc_auc_multi(y_true, y_pred, [-1], np.mean)
    roc_auc_multi(y_true, y_pred, [0], np.median)
    '''
    y_true = y_true[:, eval_indices]
    y_pred = y_pred[:, eval_indices]
    nb_classes = y_true.shape[1]
    auc = np.zeros(nb_classes)
    for i in range(len(auc)):
        # -1 represents missing value
        # and remove them when in evaluation
        non_missing_indices = np.argwhere(y_true[:, i] != -1)[:, 0]
        actual = y_true[non_missing_indices, i]
        predicted = y_pred[non_missing_indices, i]
        auc[i] = roc_auc_single(actual, predicted)
    return eval_mean_or_median(auc)


def roc_auc_single(actual, predicted):
    return roc_auc_score(actual, predicted)


def precision_auc_multi(y_true, y_pred, eval_indices, eval_mean_or_median):
    '''
    this if for multi-task evaluation
    y_true and y_pred is two-dimension matrix
    can evaluate on mean or median of array
    call by
    precision_auc_multi(y_true, y_pred, [-1], np.mean)
    precision_auc_multi(y_true, y_pred, [0], np.median)
    '''
    y_true = y_true[:, eval_indices]
    y_pred = y_pred[:, eval_indices]
    nb_classes = y_true.shape[1]
    auc = np.zeros(nb_classes)
    for i in range(len(auc)):
        # -1 represents missing value
        # and remove them when in evaluation
        non_missing_indices = np.argwhere(y_true[:, i] != -1)[:, 0]
        actual = y_true[non_missing_indices, i]
        predicted = y_pred[non_missing_indices, i]
        auc[i] = precision_auc_single(actual, predicted)
    return eval_mean_or_median(auc)


def precision_auc_single(actual, predicted):
    return average_precision_score(actual, predicted)


def enrichment_factor_multi(actual, predicted, percentile):
    EF_list = []
    for i in range(actual.shape[1]):
        n_actives, ef = enrichment_factor_single(actual[:, i], predicted[:, i], percentile)
        temp = [n_actives, ef]
        EF_list.append(temp)
    return EF_list


def enrichment_factor_single(labels_arr, scores_arr, percentile):
    '''
    calculate the enrichment factor based on some upper fraction
    of library ordered by docking scores. upper fraction is determined
    by percentile (actually a fraction of value 0.0-1.0)

    -1 represents missing value
    and remove them when in evaluation
    '''
    non_missing_indices = np.argwhere(labels_arr!=-1)[:, 0]
    labels_arr = labels_arr[non_missing_indices]
    scores_arr = scores_arr[non_missing_indices]

    sample_size = int(labels_arr.shape[0] * percentile)         # determine number mols in subset
    pred = np.sort(scores_arr)[::-1][:sample_size]              # sort the scores list, take top subset from library
    indices = np.argsort(scores_arr)[::-1][:sample_size]        # get the index positions for these in library
    n_actives = np.nansum(labels_arr)                           # count number of positive labels in library
    n_experimental = np.nansum(labels_arr[indices])            # count number of positive labels in subset
    temp = scores_arr[indices]
    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile     # calc EF at percentile
    else:
        ef = 'ND'
    return n_actives, ef