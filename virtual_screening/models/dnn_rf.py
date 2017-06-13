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
from virtual_screening.function import read_merged_data, extract_feature_and_label, store_data, \
    transform_json_to_csv, reshape_data_into_2_dim
from virtual_screening.evaluation import roc_auc_single, roc_auc_multi, bedroc_auc_multi, bedroc_auc_single, \
    precision_auc_multi, precision_auc_single, enrichment_factor_multi, enrichment_factor_single
from virtual_screening.models.CallBacks import KeckCallBackOnROC, KeckCallBackOnPrecision, \
    MultiCallBackOnROC, MultiCallBackOnPR
from sklearn.ensemble import RandomForestClassifier


# this is for customized loss function
epsilon = 1.0e-9


def count_occurance(key_list, target_list):
    weight = {i: 0 for i in key_list}
    for x in target_list:
        weight[x] += 1
    return weight


def get_class_weight(task, y_data, reference=None):
    if task.weight_schema == 'no_weight':
        cw = []
        for i in range(task.output_layer_dimension):
            cw.append({0: 0.5, 1: 0.5})
    elif task.weight_schema == 'weighted_sample':
        cw = []
        for i in range(task.output_layer_dimension):
            w = count_occurance([0, 1], y_data[:, i])
            zero_weight = 1.0
            one_weight = 1.0 * w[0] / w[1]
            cw.append({0: zero_weight, 1: one_weight})
    elif task.weight_schema == 'weighted_task':
        cw = []
        ones_sum = 0
        w_list = []
        for i in range(task.output_layer_dimension):
            w = reference[i]
            w_list.append(w)
            ones_sum += w['1']
        for i in range(task.output_layer_dimension):
            w = w_list[i]
            share = 0.01 * ones_sum / w['1']
            zero_weight = share * 1.0
            one_weight = share * w['0'] / w['1']
            # this corresponds to Keck Pria
            if i + 1 == task.output_layer_dimension:
                # TODO: generalize this part
                zero_weight *= task.conf['weight_scaled_param']
                one_weight *= task.conf['weight_scaled_param']
            cw.append({-1: 0.0, 0: zero_weight, 1: one_weight})
    elif task.weight_schema == 'weighted_task_log':
        cw = []
        ones_sum = 0
        w_list = []
        for i in range(task.output_layer_dimension):
            w = reference[i]
            w_list.append(w)
            ones_sum += w['1']
        for i in range(task.output_layer_dimension):
            w = w_list[i]
            share = math.log(ones_sum / w['1'])
            zero_weight = share * 1.0
            one_weight = share * w['0'] / w['1']
            # this corresponds to Keck Pria
            if i + 1 == task.output_layer_dimension:
                # TODO: generalize this part
                zero_weight *= task.conf['weight_scaled_param']
                one_weight *= task.conf['weight_scaled_param']
            cw.append({-1: 0.0, 0: zero_weight, 1: one_weight})
    else:
        raise ValueError('Weight schema not included. Should be among [{}, {}, {}, {}].'.
                         format('no_weight', 'weighted_sample', 'weighted_task', 'weighted_task_log'))

    return cw


class DNN_RF:
    def __init__(self, conf):
        self.conf = conf
        self.input_layer_dimension = 1024
        self.output_layer_dimension = 1

        self.early_stopping_patience = conf['fitting']['early_stopping']['patience']
        self.early_stopping_option = conf['fitting']['early_stopping']['option']

        self.fit_nb_epoch = conf['fitting']['nb_epoch']
        self.fit_batch_size = conf['fitting']['batch_size']
        self.fit_verbose = conf['fitting']['verbose']

        self.compile_loss = conf['compile']['loss']
        self.compile_optimizer_option = conf['compile']['optimizer']['option']
        if self.compile_optimizer_option == 'sgd':
            sgd_lr = conf['compile']['optimizer']['sgd']['lr']
            sgd_momentum = conf['compile']['optimizer']['sgd']['momentum']
            sgd_decay = conf['compile']['optimizer']['sgd']['decay']
            sgd_nestrov = conf['compile']['optimizer']['sgd']['nestrov']
            self.compile_optimizer = SGD(lr=sgd_lr, momentum=sgd_momentum, decay=sgd_decay, nesterov=sgd_nestrov)
        else:
            adam_lr = conf['compile']['optimizer']['adam']['lr']
            adam_beta_1 = conf['compile']['optimizer']['adam']['beta_1']
            adam_beta_2 = conf['compile']['optimizer']['adam']['beta_2']
            adam_epsilon = conf['compile']['optimizer']['adam']['epsilon']
            self.compile_optimizer = Adam(lr=adam_lr, beta_1=adam_beta_1, beta_2=adam_beta_2, epsilon=adam_epsilon)

        self.batch_is_use = conf['batch']['is_use']
        if self.batch_is_use:
            batch_normalizer_epsilon = conf['batch']['epsilon']
            batch_normalizer_mode = conf['batch']['mode']
            batch_normalizer_axis = conf['batch']['axis']
            batch_normalizer_momentum = conf['batch']['momentum']
            batch_normalizer_weights = conf['batch']['weights']
            batch_normalizer_beta_init = conf['batch']['beta_init']
            batch_normalizer_gamma_init = conf['batch']['gamma_init']
            self.batch_normalizer = BatchNormalization(epsilon=batch_normalizer_epsilon,
                                                       mode=batch_normalizer_mode,
                                                       axis=batch_normalizer_axis,
                                                       momentum=batch_normalizer_momentum,
                                                       weights=batch_normalizer_weights,
                                                       beta_init=batch_normalizer_beta_init,
                                                       gamma_init=batch_normalizer_gamma_init)
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        self.weight_schema = conf['class_weight_option']

        return

    def setup_model(self):
        layers = self.conf['layers']
        layer_number = len(layers)
        for i in range(layer_number):
            init = layers[i]['init']
            activation = layers[i]['activation']
            if i == 0:
                hidden_units = int(layers[i]['hidden_units'])
                dropout = float(layers[i]['dropout'])
                # model.add(Dense(hidden_units, input_dim=self.input_layer_dimension, init=init, activation=activation))
                # model.add(Dropout(dropout))
                input_layer = Input(shape=[self.input_layer_dimension], name='input_layer')
                M = Dense(hidden_units, init=init, name='layer ' + str(i))(input_layer)
                M = Dropout(dropout, name='drop out ' + str(i))(M)
                classification_output_layer = Activation(activation, name='activation classification')(M)
            elif i == layer_number - 1:
                # if self.batch_is_use:
                #     model.add(self.batch_normalizer)
                # model.add(Dense(self.output_layer_dimension, init=init, activation=activation))
                classification_output_layer = Dense(self.output_layer_dimension,
                                                    init=init,
                                                    activation=activation,
                                                    name='classification_output_layer')(classification_output_layer)
            else:
                hidden_units = int(layers[i]['hidden_units'])
                dropout = float(layers[i]['dropout'])
                # model.add(Dense(hidden_units, init=init, activation=activation))
                # model.add(Dropout(dropout))
                classification_output_layer = Dense(hidden_units,
                                                    init=init,
                                                    activation=activation,
                                                    name='classification layer ' + str(i))(classification_output_layer)
                intermediate_name = 'classification layer ' + str(i)
                classification_output_layer = Dropout(dropout,
                                                      name='classification drop out ' + str(i))(classification_output_layer)

        # intermediate_layer_model = Model(input=model.input, output=model.get_layer(layer_name).output)
        model = Model(input=input_layer, output=[classification_output_layer])
        self.intermediate_layer_model = Model(input=input_layer, output=model.get_layer(intermediate_name).output)
        return model

    def train_and_predict(self,
                          X_train, y_train,
                          X_val, y_val,
                          X_test, y_test,
                          PMTNN_weight_file):
        model = self.setup_model()
        if self.early_stopping_option == 'auc':
            early_stopping = KeckCallBackOnROC(X_train, y_train, X_val, y_val,
                                               patience=self.early_stopping_patience,
                                               file_path=PMTNN_weight_file)
            callbacks = [early_stopping]
        elif self.early_stopping_option == 'precision':
            early_stopping = KeckCallBackOnPrecision(X_train, y_train, X_val, y_val,
                                                     patience=self.early_stopping_patience,
                                                     file_path=PMTNN_weight_file)
            callbacks = [early_stopping]
        else:
            callbacks = []

        cw = get_class_weight(self, y_train)
        print 'cw ', cw

        model.compile(loss=self.compile_loss, optimizer=self.compile_optimizer)
        model.fit(X_train, y_train,
                  nb_epoch=self.fit_nb_epoch,
                  batch_size=self.fit_batch_size,
                  verbose=self.fit_verbose,
                  class_weight=cw,
                  shuffle=True,
                  callbacks=callbacks)

        if self.early_stopping_option == 'auc' or self.early_stopping_option == 'precision':
            model = early_stopping.get_best_model()

        y_pred_on_train = model.predict(X_train)
        y_pred_on_val = model.predict(X_val)
        y_pred_on_test = model.predict(X_test)

        print
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

        return model

    def predict_with_existing(self,
                              X_train, y_train,
                              X_val, y_val,
                              X_test, y_test,
                              PMTNN_weight_file):
        model = self.setup_model()
        model.load_weights(PMTNN_weight_file)

        y_pred_on_train = model.predict(X_train)
        y_pred_on_val = model.predict(X_val)
        y_pred_on_test = model.predict(X_test)

        print
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

        return

    def get_EF_score_with_existing_model(self,
                                         X_test, y_test,
                                         file_path, EF_ratio):
        model = self.setup_model()
        model.load_weights(file_path)
        y_pred_on_test = model.predict(X_test)
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, EF_ratio)
        print('test precision: {}'.format(get_model_precision_auc(y_test, y_pred_on_test)))
        print('test auc: {}'.format(get_model_roc_auc(y_test, y_pred_on_test)))
        print('EF: {},\tactive: {}'.format(ef, n_actives))

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


    def get_dnn_rf(self, X_train, y_train, X_val, y_val, X_test, y_test):
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

        # X_first_layer = np.vstack((X_train, X_val))
        # y_secondary_layer = np.vstack((y_train, y_val))
        X_secondary_layer = self.intermediate_layer_model.predict(X_train)
        y_secondary_layer = y_train
        print X_secondary_layer.shape
        print y_secondary_layer.shape

        rf.fit(X_secondary_layer, y_secondary_layer)

        X_train_secondary = self.intermediate_layer_model.predict(X_train)
        y_pred_on_train = reshape_data_into_2_dim(rf.predict(X_train_secondary))
        X_val_secondary = self.intermediate_layer_model.predict(X_val)
        y_pred_on_val = reshape_data_into_2_dim(rf.predict(X_val_secondary))
        X_test_secondary = self.intermediate_layer_model.predict(X_test)
        y_pred_on_test = reshape_data_into_2_dim(rf.predict(X_test_secondary))

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
    with open(config_json_file, 'r') as f:
        conf = json.load(f)

    # TODO: debug
    conf['fitting']['nb_epoch'] = 200
    conf['fitting']['early_stopping']['patience'] = 50

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
    cross_validation_split = StratifiedShuffleSplit(y_train, 1, test_size=0.2, random_state=1)
    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t, y_val = y_train[t_index], y_train[val_index]
    print 'done data preparation'

    print conf['label_name_list']
    task = DNN_RF(conf=conf)
    # task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, PMTNN_weight_file)

    # loading and then pop doesn't work
    # https://stackoverflow.com/questions/41668813/how-to-add-and-remove-new-layers-in-keras-after-loading-weights
    # model = task.setup_model()
    # model.load_weights(PMTNN_weight_file)

    # model.pop()
    # model.pop()
    # model.pop()
    # model.outputs = [model.layers[-1].output]
    # model.layers[-1].outbound_nodes = []
    #
    # print 'after'
    # model.summary()
    # print len(model.layers)


    # Start RF
    # rf = task.get_dnn_rf(X_train, y_train, X_val, y_val, X_test, y_test)
    rf = task.get_rf(X_train, y_train, X_val, y_val, X_test, y_test)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', action="store", dest="config_json_file", required=True)
    parser.add_argument('--PMTNN_weight_file', action="store", dest="PMTNN_weight_file", required=True)
    parser.add_argument('--config_csv_file', action="store", dest="config_csv_file", required=True)
    parser.add_argument('--score_file', action='store', dest='score_file', required=False)
    parser.add_argument('--mode', action='store', dest='mode', required=True)
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    PMTNN_weight_file = given_args.PMTNN_weight_file
    config_csv_file = given_args.config_csv_file
    mode = given_args.mode

    if mode == 'single_dnn_rf':
        demo()
