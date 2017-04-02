import argparse
import pandas as pd
import csv
import numpy as np
import json
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from sklearn.cross_validation import StratifiedShuffleSplit
from function import *
from evaluation import *


class single_task:
    def __init__(self, config_json_file):
        with open(config_json_file, 'r') as f:
            conf = json.load(f)

        self.conf = conf
        self.input_layer_dimension = 1024
        self.output_layer_dimension = 1

        self.early_stopping_patience = conf['fitting']['early_stopping']['patience']
        self.early_stopping_option = conf['fitting']['early_stopping']['option']

        self.fit_nb_epoch = conf['fitting']['nb_epoch']
        self.fit_nb_epoch = 1
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

        return

    def setup_model(self):
        model = Sequential()
        if self.batch_is_use:
            batch_normalizer = self.batch_normalizer
        layers = self.conf['layers']
        layer_number = len(layers)
        for i in range(layer_number):
            init = layers[i]['init']
            activation = layers[i]['activation']
            if i == 0:
                hidden_units = int(layers[i]['hidden_units'])
                dropout = float(layers[i]['dropout'])
                model.add(Dense(hidden_units, input_dim=self.input_layer_dimension, init=init, activation=activation))
                model.add(Dropout(dropout))
            elif i == layer_number - 1:
                if self.batch_is_use:
                    model.add(self.batch_normalizer)
                model.add(Dense(self.output_layer_dimension, init=init, activation=activation))
            else:
                hidden_units = int(layers[i]['hidden_units'])
                dropout = float(layers[i]['dropout'])
                model.add(Dense(hidden_units, init=init, activation=activation))
                model.add(Dropout(dropout))

        return model

    def train_and_predict(self,
                          X_train, y_train,
                          X_val, y_val,
                          X_test, y_test,
                          PMTNN_weight_file):
        model = self.setup_model()
        if self.early_stopping_option == 'auc':
            early_stopping = KeckCallBackOnAUC(X_train, y_train, X_val, y_val, patience=self.early_stopping_patience)
            callbacks = [early_stopping]
        elif self.early_stopping_option == 'precision':
            early_stopping = KeckCallBackOnPrecision(X_train, y_train, X_val, y_val,
                                                     patience=self.early_stopping_patience)
            callbacks = [early_stopping]
        else:
            callbacks = []

        model.compile(loss=self.compile_loss, optimizer=self.compile_optimizer)
        model.fit(X_train, y_train,
                  nb_epoch=self.fit_nb_epoch,
                  batch_size=self.fit_batch_size,
                  verbose=self.fit_verbose,
                  shuffle=True,
                  callbacks=callbacks)
        model.save_weights(PMTNN_weight_file)

        if self.early_stopping_option == 'auc' or self.early_stopping_option == 'precision':
            model = early_stopping.get_best_model()
        y_pred_on_train = model.predict(X_train)
        y_pred_on_val = model.predict(X_val)
        y_pred_on_test = model.predict(X_test)

        print('train precision: {}'.format(precision_auc_single(y_train, y_pred_on_train)))
        print('train roc: {}'.format(roc_auc_single(y_train, y_pred_on_train)))
        print('validation precision: {}'.format(precision_auc_single(y_val, y_pred_on_val)))
        print('validation roc: {}'.format(roc_auc_single(y_val, y_pred_on_val)))
        print('test precision: {}'.format(precision_auc_single(y_test, y_pred_on_test)))
        print('test roc: {}'.format(roc_auc_single(y_test, y_pred_on_test)))

        for EF_ratio in self.EF_ratio_list:
            n_actives, ef = enrichment_factor(y_test, y_pred_on_test, EF_ratio)
            print('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))

        return

    def predict_with_existing(self,
                              X_train, y_train,
                              X_val, y_val,
                              X_test, y_test,
                              PMTNN_weight_file):
        model = setup_model()
        model.load_weights(PMTNN_weight_file)

        y_pred_on_train = model.predict(X_train)
        y_pred_on_val = model.predict(X_val)
        y_pred_on_test = model.predict(X_test)

        print('train precision: {}'.format(precision_auc_single(y_train, y_pred_on_train)))
        print('train auc: {}'.format(roc_auc_single(y_train, y_pred_on_train)))
        print('validation precision: {}'.format(precision_auc_single(y_val, y_pred_on_val)))
        print('validation auc: {}'.format(roc_auc_single(y_val, y_pred_on_val)))
        print('test precision: {}'.format(precision_auc_single(y_test, y_pred_on_test)))
        print('test auc: {}'.format(roc_auc_single(y_test, y_pred_on_test)))

        return

    def get_EF_score_with_existing_model(self,
                                         X_test, y_test,
                                         file_path, EF_ratio):
        model = setup_model()
        model.load_weights(file_path)
        y_pred_on_test = model.predict(X_test)
        n_actives, ef = enrichment_factor(y_test, y_pred_on_test, EF_ratio)
        print('test precision: {}'.format(precision_auc_single(y_test, y_pred_on_test)))
        print('test auc: {}'.format(roc_auc_single(y_test, y_pred_on_test)))
        print('EF: {},\tactive: {}'.format(ef, n_actives))

        return


def enrichment_factor(labels_arr, scores_arr, percentile):
    '''calculate the enrichment factor based on some upper fraction
       of library ordered by docking scores. upper fraction is determined
       by percentile (actually a fraction of value 0.0-1.0)'''
    sample_size = int(labels_arr.shape[0] * percentile)  # determine number mols in subset
    pred = np.sort(scores_arr)[::-1][:sample_size]  # sort the scores list, take top subset from library
    indices = np.argsort(scores_arr, axis=0)[::-1][:sample_size]  # get the index positions for these in library
    n_actives = np.nansum(labels_arr)  # count number of positive labels in library
    n_experimental = np.nansum(labels_arr[indices])  # count number of positive labels in subset
    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile  # calc EF at percentile
    else:
        ef = 'ND'
    return n_actives, ef


# define custom classes
# following class is used for keras to compute the AUC each epoch
# and do early stoppping based on that
class KeckCallBackOnAUC(keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val,
                 patience=0,
                 file_path='best_model.weights'):
        super(keras.callbacks.Callback, self).__init__()
        self.curr_auc = 0
        self.best_auc = 0
        self.counter = 0
        self.patience = patience
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.file_path = file_path

    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']
        self.curr_auc = roc_auc_single(self.y_val, self.model.predict(self.X_val))
        self.best_auc = self.curr_auc

    def on_epoch_end(self, epoch, logs={}):
        self.curr_auc = roc_auc_single(self.y_val, self.model.predict(self.X_val))
        if self.curr_auc < self.best_auc:
            if self.counter >= self.patience:
                self.model.stop_training = True
            else:
                self.counter += 1
        else:
            self.counter = 0
            self.best_auc = self.curr_auc
            self.model.save_weights(self.file_path)
        train_auc = roc_auc_single(self.y_train, self.model.predict(self.X_train))
        train_precision = precision_auc_single(self.y_train, self.model.predict(self.X_train))
        curr_precision = precision_auc_single(self.y_val, self.model.predict(self.X_val))
        print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
        print('ROC Train: %f ---- ROC Val: %f' % (train_auc, self.curr_auc))
        print('Precision Train: %f ---- Precision Val: %f' % (train_precision, curr_precision))
        print

    def get_best_model(self):
        self.model.load_weights(self.file_path)
        return self.model

    def get_best_auc(self):
        return self.best_auc


# define custom classes
# following class is used for keras to compute the precision each epoch
# and do early stoppping based on that
class KeckCallBackOnPrecision(keras.callbacks.Callback):
    def __init__(self, X_train, y_train, X_val, y_val,
                 patience=0,
                 file_path='best_model.weights'):
        super(keras.callbacks.Callback, self).__init__()
        self.curr_precision = 0
        self.best_precision = 0
        self.prev_precision = self.curr_precision
        self.counter = 0
        self.patience = patience
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.file_path = file_path

    def on_train_begin(self, logs={}):
        self.nb_epoch = self.params['nb_epoch']
        self.curr_precision = precision_auc_single(self.y_val, self.model.predict(self.X_val))
        self.best_precision = self.curr_precision

    def on_epoch_end(self, epoch, logs={}):
        self.curr_precision = precision_auc_single(self.y_val, self.model.predict(self.X_val))
        if self.curr_precision < self.best_precision:
            if self.counter >= self.patience:
                self.model.stop_training = True
            else:
                self.counter += 1
        else:
            self.counter = 0
            self.best_precision = self.curr_precision
            self.model.save_weights(self.file_path)

        self.prev_precision = self.curr_precision
        train_precision = precision_auc_single(self.y_train, self.model.predict(self.X_train))
        train_auc = roc_auc_single(self.y_train, self.model.predict(self.X_train))
        curr_auc = roc_auc_single(self.y_val, self.model.predict(self.X_val))
        print('Epoch %d/%d' % (epoch + 1, self.nb_epoch))
        print('Precision Train: %f ---- Precision Val: %f' % (train_precision, self.curr_precision))
        print('ROC Train: %f ---- ROC Val: %f' % (train_auc, curr_auc))
        print

    def get_best_model(self):
        self.model.load_weights(self.file_path)
        return self.model

    def get_best_precision(self):
        return self.best_precision


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', action="store", dest="config_json_file", required=True)
    parser.add_argument('--PMTNN_weight_file', action="store", dest="PMTNN_weight_file", required=True)
    parser.add_argument('--config_csv_file', action="store", dest="config_csv_file", required=True)
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    PMTNN_weight_file = given_args.PMTNN_weight_file
    config_csv_file = given_args.config_csv_file

    task = single_task(config_json_file=config_json_file)

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    input_file = conf['input_file']
    X_data, y_data = readKeckData(input_file)

    # randome_state = 0 means it's not shuffling
    training_testing_split = StratifiedShuffleSplit(y_data, 1, test_size=0.2, random_state=0)
    for training_index, testing_index in training_testing_split:
        X_train, X_test = X_data[training_index], X_data[testing_index]
        y_train, y_test = y_data[training_index], y_data[testing_index]
    del X_data
    del y_data

    cross_validation_split = StratifiedShuffleSplit(y_train, 1, test_size=0.2, random_state=0)
    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t, y_val = y_train[t_index], y_train[val_index]
    del X_train
    del y_train

    task = single_task(config_json_file=config_json_file)
    task.train_and_predict(X_t, y_t, X_val, y_val, X_test, y_test, PMTNN_weight_file)
    task.store_data(transform_json_to_csv(config_json_file), config_csv_file)