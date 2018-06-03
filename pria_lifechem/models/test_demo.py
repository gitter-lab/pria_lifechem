import argparse
import pandas as pd
import csv
import numpy as np
import json
import keras
import sys
import math
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from sklearn.cross_validation import StratifiedShuffleSplit
from pria_lifechem.function import read_merged_data, extract_feature_and_label, store_data, \
    transform_json_to_csv, reshape_data_into_2_dim
from pria_lifechem.evaluation import roc_auc_single, roc_auc_multi, bedroc_auc_multi, bedroc_auc_single, \
    precision_auc_multi, precision_auc_single, enrichment_factor_multi, enrichment_factor_single
from pria_lifechem.models.CallBacks import KeckCallBackOnROC, KeckCallBackOnPrecision, \
    MultiCallBackOnROC, MultiCallBackOnPR
from sklearn.ensemble import RandomForestClassifier


# this is for customized loss function
epsilon = 1.0e-9


class DNN_RF:
    def __init__(self):
        self.EF_ratio_list = [0.001, 0.01]
        return


    def get_rf(self, X_train, y_train, X_val, y_val, X_test, y_test):
        max_features = 'log2'
        n_estimators = 4000
        min_samples_leaf = 1
        class_weight = 'balanced'
        rnd_state = 1337
        np.random.seed(seed=rnd_state)

        rf = RandomForestClassifier(n_estimators=n_estimators,
                                    max_features=max_features,
                                    min_samples_leaf=min_samples_leaf,
                                    n_jobs=3,
                                    class_weight=class_weight,
                                    random_state=rnd_state,
                                    oob_score=False,
                                    verbose=1)
        rf.fit(X_train, y_train)

        y_pred_on_train = reshape_data_into_2_dim(rf.predict(X_train))
        y_pred_on_val = reshape_data_into_2_dim(rf.predict(X_val))
        y_pred_on_test = reshape_data_into_2_dim(rf.predict(X_test))

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

        for EF_ratio in self.EF_ratio_list:
            n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, EF_ratio)
            print('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))

        return rf

def demo():
    label_name_list = 'Keck_Pria_AS_Retest'
    print 'label_name_list ', label_name_list

    # specify dataset
    k = 5
    directory = '../../dataset/sample_fold/'
    file_list = []
    for i in range(k):
        file_list.append('{}.csv'.format(i))

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
    cross_validation_split = StratifiedShuffleSplit(y_train, 1, test_size=0.2, random_state=1)
    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t, y_val = y_train[t_index], y_train[val_index]
    print 'done data preparation'

    task = DNN_RF()
    task.get_rf(X_train, y_train, X_val, y_val, X_test, y_test)

    return

if __name__ == '__main__':
    demo()
