import argparse
import pandas as pd
import csv
import numpy as np
import json
import keras
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.core import RepeatVector, TimeDistributedDense
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD, Adam
from sklearn.cross_validation import StratifiedShuffleSplit
from virtual_screening.function import read_merged_data, extract_SMILES_and_label
from virtual_screening.evaluation import roc_auc_single, bedroc_auc_single, \
    precision_auc_single, enrichment_factor_single
from virtual_screening.models.CallBacks import KeckCallBackOnROC, KeckCallBackOnPrecision


class VanillaLSTM:
    def __init__(self, conf):
        self.conf = conf

        # this padding length works the same as time steps
        self.padding_length = conf['lstm']['padding_length']
        self.vocabulary_size = conf['lstm']['different_alphabets_num'] + 1
        self.embedding_size = conf['lstm']['embedding_size']
        self.activation = conf['lstm']['activation']
        self.layer_num = conf['lstm']['layer_num']

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
        return


    def setup_model(self):
        model = Sequential()
        model.add(Embedding(input_dim=self.vocabulary_size,
                            output_dim=self.embedding_size,
                            input_length=self.padding_length))
        layers = self.conf['layers']
        layer_num = self.layer_num

        for i in range(layer_num):
            hidden_size = layers[i]['hidden_size']
            dropout_W = layers[i]['dropout_W']
            dropout_U = layers[i]['dropout_U']
            if i == layer_num - 1:
                model.add(LSTM(hidden_size,
                               dropout_W=dropout_W,
                               dropout_U=dropout_U,
                               return_sequences=False))
            elif i == 0:
                model.add(LSTM(hidden_size,
                               input_shape=(self.padding_length, self.embedding_size),
                               dropout_W=dropout_W,
                               dropout_U=dropout_U,
                               return_sequences=True))
            else:
                model.add(LSTM(hidden_size,
                               dropout_W=dropout_W,
                               dropout_U=dropout_U,
                               return_sequences=True))
        # (batch_size,time_steps,embedding_size)
        # model.add(LSTM(self.first_hidden_size,
        #                input_shape=(self.padding_length, self.embedding_size),
        #                dropout_W=0.2,
        #                dropout_U=0.2,
        #                return_sequences=True))
        # model.add(LSTM(self.first_hidden_size,
        #                dropout_W=0.2,
        #                dropout_U=0.2,
        #                return_sequences=True))
        # model.add(LSTM(self.second_hidden_size,
        #                dropout_W=0.2,
        #                dropout_U=0.2))

        model.add(Dense(self.output_layer_dimension,
                        activation=self.activation))
        print(model.summary())
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

        model.compile(loss=self.compile_loss, optimizer=self.compile_optimizer)
        model.fit(X_train, y_train, nb_epoch=self.fit_nb_epoch, batch_size=self.fit_batch_size,
                  callbacks=callbacks,
                  verbose=self.fit_verbose)

        if self.early_stopping_option == 'auc' or self.early_stopping_option == 'precision':
            model = early_stopping.get_best_model()
        y_pred_on_train = model.predict(X_train)
        y_pred_on_val = model.predict(X_val)
        y_pred_on_test = model.predict(X_test)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', action='store', dest='config_json_file', required=True)
    parser.add_argument('--PMTNN_weight_file', action='store', dest='PMTNN_weight_file', required=True)
    parser.add_argument('--config_csv_file', action='store', dest='config_csv_file', required=True)
    parser.add_argument('--SMILES_mapping_json_file', action='store', dest='SMILES_mapping_json_file', default= '../../json/SMILES_mapping.json')
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    PMTNN_weight_file = given_args.PMTNN_weight_file
    config_csv_file = given_args.config_csv_file
    SMILES_mapping_json_file = given_args.SMILES_mapping_json_file

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    task = VanillaLSTM(conf)

    # specify dataset
    k = 5
    directory = '../../dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    # merge training and test dataset
    dtype_list = {'Molecule': np.str,
                  'SMILES': np.str,
                  'Fingerprints': np.str,
                  'Keck_Pria_AS_Retest': np.int64,
                  'Keck_Pria_FP_data': np.int64,
                  'Keck_Pria_Continuous': np.float64,
                  'Keck_RMI_cdd': np.float64}
    output_file_list = [directory + f_ for f_ in file_list]
    print output_file_list[:4]
    train_pd = read_merged_data(output_file_list[0:4])
    print output_file_list[4]
    test_pd = read_merged_data([output_file_list[4]])

    # extract data, and split training data into training and val
    X_train, y_train = extract_SMILES_and_label(train_pd,
                                                feature_name='SMILES',
                                                label_name_list=['Keck_Pria_AS_Retest'],
                                                SMILES_mapping_json_file=SMILES_mapping_json_file)
    X_test, y_test = extract_SMILES_and_label(test_pd,
                                              feature_name='SMILES',
                                              label_name_list=['Keck_Pria_AS_Retest'],
                                              SMILES_mapping_json_file=SMILES_mapping_json_file)

    cross_validation_split = StratifiedShuffleSplit(y_train, 1, test_size=0.15, random_state=1)
    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t, y_val = y_train[t_index], y_train[val_index]
    print 'done data preparation'

    X_t = sequence.pad_sequences(X_t, maxlen=task.padding_length)
    X_val = sequence.pad_sequences(X_val, maxlen=task.padding_length)
    X_test = sequence.pad_sequences(X_test, maxlen=task.padding_length)

    task.train_and_predict(X_t, y_t, X_val, y_val, X_test, y_test, PMTNN_weight_file)
    store_config(conf, config_csv_file)