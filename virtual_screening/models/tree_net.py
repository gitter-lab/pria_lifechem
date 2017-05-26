import argparse
import pandas as pd
import csv
import numpy as np
import json
import keras
import sys
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from sklearn.cross_validation import StratifiedShuffleSplit
from virtual_screening.function import read_merged_data, extract_feature_and_label, \
    reshape_data_into_2_dim, store_data, transform_json_to_csv
from virtual_screening.evaluation import roc_auc_single, bedroc_auc_single, \
    precision_auc_single, enrichment_factor_single


class TreeNet:
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


    def train_and_predict(self,
                          X_train, y_train, y_train_classification,
                          X_val, y_val, y_val_classification,
                          X_test, y_test, y_test_classification,
                          mode):
        model = Sequential()
        conf = self.conf
        batch_normalizer_epsilon = conf['batch']['epsilon']
        batch_normalizer_mode = conf['batch']['mode']
        batch_normalizer_axis = conf['batch']['axis']
        batch_normalizer_momentum = conf['batch']['momentum']
        batch_normalizer_weights = conf['batch']['weights']
        batch_normalizer_beta_init = conf['batch']['beta_init']
        batch_normalizer_gamma_init = conf['batch']['gamma_init']
        batch_normalizer = BatchNormalization(epsilon=batch_normalizer_epsilon,
                                              mode=batch_normalizer_mode,
                                              axis=batch_normalizer_axis,
                                              momentum=batch_normalizer_momentum,
                                              weights=batch_normalizer_weights,
                                              beta_init=batch_normalizer_beta_init,
                                              gamma_init=batch_normalizer_gamma_init)

        if mode == 'classification':
            model.add(Dense(2048, input_dim=1024, init='glorot_normal', activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1024, init='glorot_normal', activation='relu'))
            model.add(Dropout(0.5))
            if self.batch_is_use:
                model.add(batch_normalizer)
            model.add(Dense(1, init='glorot_normal', activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam')
        else:
            model.add(Dense(2048, input_dim=1024, init='glorot_normal', activation='sigmoid'))
            model.add(Dropout(0.5))
            model.add(Dense(1024, init='glorot_normal', activation='sigmoid'))
            model.add(Dropout(0.5))
            if self.batch_is_use:
                model.add(batch_normalizer)
            model.add(Dense(1, init='glorot_normal', activation='linear'))
            model.compile(loss='mse', optimizer='adam')

        model.fit(X_train, y_train,
                  batch_size=self.fit_batch_size,
                  nb_epoch=self.fit_nb_epoch,
                  verbose=self.fit_verbose,
                  validation_data=(X_val, y_val))

        y_pred_on_train = model.predict(X_train)
        y_pred_on_val = model.predict(X_val)
        y_pred_on_test = model.predict(X_test)

        print
        print 'this is mode ', mode
        print('train precision: {}'.format(precision_auc_single(y_train_classification, y_pred_on_train)))
        print('train roc: {}'.format(roc_auc_single(y_train_classification, y_pred_on_train)))
        print('train bedroc: {}'.format(bedroc_auc_single(y_train_classification, y_pred_on_train)))
        print
        print('validation precision: {}'.format(precision_auc_single(y_val_classification, y_pred_on_val)))
        print('validation roc: {}'.format(roc_auc_single(y_val_classification, y_pred_on_val)))
        print('validation bedroc: {}'.format(bedroc_auc_single(y_val_classification, y_pred_on_val)))
        print
        print('test precision: {}'.format(precision_auc_single(y_test_classification, y_pred_on_test)))
        print('test roc: {}'.format(roc_auc_single(y_test_classification, y_pred_on_test)))
        print('test bedroc: {}'.format(bedroc_auc_single(y_test_classification, y_pred_on_test)))
        print

        for EF_ratio in self.EF_ratio_list:
            n_actives, ef, ef_max = enrichment_factor_single(y_test_classification, y_pred_on_test, EF_ratio)
            print('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))

        return y_pred_on_test


    def setup_model_ensemble(self):
        layers = self.conf['layers']
        layer_number = len(layers)
        for i in range(layer_number):
            init = layers[i]['init']
            if i == 0:
                hidden_units = int(layers[i]['hidden_units'])
                dropout = float(layers[i]['dropout'])
                input_layer = Input(shape=[self.input_layer_dimension], name='input_layer')
                M = Dense(hidden_units, init=init, name='layer ' + str(i))(input_layer)
                M = Dropout(dropout, name='drop out ' + str(i))(M)
                classification_output_layer = Activation('relu', name='activation classification')(M)
                regression_output_layer = Activation('sigmoid', name='activation regression')(M)
            elif i == layer_number - 1:
                classification_output_layer = Dense(self.output_layer_dimension,
                                                    init=init,
                                                    activation='sigmoid',
                                                    name='classification_output_layer')(classification_output_layer)
                regression_output_layer = Dense(self.output_layer_dimension,
                                                init=init,
                                                activation='linear',
                                                name='regression_output_layer')(regression_output_layer)
            else:
                hidden_units = int(layers[i]['hidden_units'])
                dropout = float(layers[i]['dropout'])
                classification_output_layer = Dense(hidden_units,
                                                    init=init,
                                                    activation='relu',
                                                    name='classification layer ' + str(i))(classification_output_layer)
                classification_output_layer = Dropout(dropout,
                                                      name='classification drop out ' + str(i))(classification_output_layer)

                regression_output_layer = Dense(hidden_units,
                                                init=init,
                                                activation='sigmoid',
                                                name='regression layer ' + str(i))(regression_output_layer)
                regression_output_layer = Dropout(dropout,
                                                  name='regression drop out ' + str(i))(regression_output_layer)

        model = Model(input=input_layer, output=[classification_output_layer, regression_output_layer])
        return model


    def train_and_predict_ensemble(self,
                                   X_train, y_train_regression, y_train_classification,
                                   X_val, y_val_regression, y_val_classification,
                                   X_test, y_test_regression, y_test_classification,
                                   PMTNN_weight_file):
        model = self.setup_model_ensemble()
        # TODO: remove
        print model.summary()

        if self.weight_schema == 'weighted':
            loss_weight = {'classification_output_layer': 1., 'regression_output_layer': 100.}
        elif self.weight_schema == 'no_weight':
            loss_weight = {'classification_output_layer': 1., 'regression_output_layer': 1.}
        else:
            raise ValueError('Wrong weight schema. Should be no_weight, or weighted.')

        model.compile(optimizer=self.compile_optimizer,
                      loss={'classification_output_layer': 'binary_crossentropy', 'regression_output_layer': 'mse'},
                      loss_weights=loss_weight)

        model.fit({'input_layer': X_train},
                  {'classification_output_layer': y_train_classification,
                   'regression_output_layer': y_train_regression},
                  nb_epoch=self.fit_nb_epoch,
                  batch_size=self.fit_batch_size,
                  verbose=self.fit_verbose,
                  validation_data=({'input_layer': X_val},
                                   {'classification_output_layer': y_val_classification,
                                    'regression_output_layer': y_val_regression}),
                  shuffle=True)
        model.save_weights(PMTNN_weight_file)

        y_pred_on_train_ensemble = np.array(model.predict(X_train))
        y_pred_on_val_ensmble = np.array(model.predict(X_val))
        y_pred_on_test_ensemble = np.array(model.predict(X_test))

        print
        print 'TreeNet Ensemble'

        mode_list = ['TreeNet classification', 'TreeNet regression']
        for mode in range(2):
            print
            print mode_list[mode]
            y_pred_on_train = y_pred_on_train_ensemble[mode]
            y_pred_on_val = y_pred_on_val_ensmble[mode]
            y_pred_on_test = y_pred_on_test_ensemble[mode]

            print('train precision: {}'.format(precision_auc_single(y_train_classification, y_pred_on_train)))
            print('train roc: {}'.format(roc_auc_single(y_train_classification, y_pred_on_train)))
            print('train bedroc: {}'.format(bedroc_auc_single(y_train_classification, y_pred_on_train)))
            print
            print('validation precision: {}'.format(precision_auc_single(y_val_classification, y_pred_on_val)))
            print('validation roc: {}'.format(roc_auc_single(y_val_classification, y_pred_on_val)))
            print('validation bedroc: {}'.format(bedroc_auc_single(y_val_classification, y_pred_on_val)))
            print
            print('test precision: {}'.format(precision_auc_single(y_test_classification, y_pred_on_test)))
            print('test roc: {}'.format(roc_auc_single(y_test_classification, y_pred_on_test)))
            print('test bedroc: {}'.format(bedroc_auc_single(y_test_classification, y_pred_on_test)))
            print

            for EF_ratio in self.EF_ratio_list:
                n_actives, ef, ef_max = enrichment_factor_single(y_test_classification, y_pred_on_test, EF_ratio)
                print('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))

        return y_pred_on_test_ensemble

    def predict_with_existing(self,
                              X_train, y_train_regression, y_train_classification,
                              X_val, y_val_regression, y_val_classification,
                              X_test, y_test_regression, y_test_classification,
                              PMTNN_weight_file):
        model = self.setup_model_ensemble()
        model.load_weights(PMTNN_weight_file)

        y_pred_on_train = model.predict(X_train)
        y_pred_on_val = model.predict(X_val)
        y_pred_on_test = model.predict(X_test)

        print
        print('train precision: {}'.format(precision_auc_single(y_train_classification, y_pred_on_train)))
        print('train roc: {}'.format(roc_auc_single(y_train_classification, y_pred_on_train)))
        print('train bedroc: {}'.format(bedroc_auc_single(y_train_classification, y_pred_on_train)))
        print
        print('validation precision: {}'.format(precision_auc_single(y_val_classification, y_pred_on_val)))
        print('validation roc: {}'.format(roc_auc_single(y_val_classification, y_pred_on_val)))
        print('validation bedroc: {}'.format(bedroc_auc_single(y_val_classification, y_pred_on_val)))
        print
        print('test precision: {}'.format(precision_auc_single(y_test_classification, y_pred_on_test)))
        print('test roc: {}'.format(roc_auc_single(y_test_classification, y_pred_on_test)))
        print('test bedroc: {}'.format(bedroc_auc_single(y_test_classification, y_pred_on_test)))
        print

        for EF_ratio in self.EF_ratio_list:
            n_actives, ef, ef_max = enrichment_factor_single(y_test_classification, y_pred_on_test, EF_ratio)
            print('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))

        return

    def get_EF_score_with_existing_model(self,
                                         X_test, y_test, y_test_classification,
                                         file_path, EF_ratio):
        model = self.setup_model_ensemble()
        model.load_weights(file_path)
        y_pred_on_test = model.predict(X_test)
        print('test precision: {}'.format(precision_auc_single(y_test_classification, y_pred_on_test)))
        print('test roc: {}'.format(roc_auc_single(y_test_classification, y_pred_on_test)))
        print('test bedroc: {}'.format(bedroc_auc_single(y_test_classification, y_pred_on_test)))
        print

        n_actives, ef, ef_max = enrichment_factor_single(y_test_classification, y_pred_on_test, EF_ratio)
        print('EF: {},\tactive: {}'.format(ef, n_actives))

        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file',
                        action="store", dest="config_json_file",
                        default='../../json/tree_net_keck_pria_retest.json',
                        required=False)
    parser.add_argument('--PMTNN_weight_file',
                        action="store", dest="PMTNN_weight_file",
                        default='temp.weight',
                        required=False)
    parser.add_argument('--config_csv_file',
                        action="store", dest="config_csv_file",
                        default='temp.csv',
                        required=False)
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    PMTNN_weight_file = given_args.PMTNN_weight_file
    config_csv_file = given_args.config_csv_file

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

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
    y_train_classification = reshape_data_into_2_dim(y_train[:, 0])
    y_train_regression = reshape_data_into_2_dim(y_train[:, 1])
    y_test_classification = reshape_data_into_2_dim(y_test[:, 0])
    y_test_regression = reshape_data_into_2_dim(y_test[:, 1])

    cross_validation_split = StratifiedShuffleSplit(y_train_classification, 1, test_size=0.2, random_state=1)

    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t_classification, y_val_classification = y_train_classification[t_index], y_train_classification[val_index]
        y_t_regression, y_val_regression = y_train_regression[t_index], y_train_regression[val_index]
    print 'done data preparation'

    task = TreeNet(conf=conf)

    y_pred_on_test_classification = task.train_and_predict(X_t, y_t_classification, y_t_classification,
                                                           X_val, y_val_classification, y_val_classification,
                                                           X_test, y_test_classification, y_test_classification,
                                                           mode='classification')
    y_pred_on_test_regression = task.train_and_predict(X_t, y_t_regression, y_t_classification,
                                                       X_val, y_val_regression, y_val_classification,
                                                       X_test, y_test_regression, y_test_classification,
                                                       mode='regression')

    y_pred_on_test_ensemble = task.train_and_predict_ensemble(X_t, y_t_regression, y_t_classification,
                                                              X_val, y_val_regression, y_val_classification,
                                                              X_test, y_test_regression, y_test_classification,
                                                              PMTNN_weight_file)
    # task.predict_with_existing(X_t, y_t_regression, y_t_classification,
    #                            X_val, y_val_regression, y_val_classification,
    #                            X_test, y_test_regression, y_test_classification,
    #                            PMTNN_weight_file)
    # task.get_EF_score_with_existing_model(X_test, y_test, y_test_classification,PMTNN_weight_file, 0.01)
    # store_data(transform_json_to_csv(config_json_file), config_csv_file)

    compare_file = 'ensemble_comparison.csv'
    output_file = open(compare_file, 'w')
    print >> output_file, 'line number, actual label, actual value, single classification pred, single regression pred, ensemble classification pred, ensemble regression pred'
    length = len(y_pred_on_test_classification)
    for i in range(length):
        print >> output_file, i, ',', y_test_classification[i, 0], ',', y_test_regression[i, 0], ',', \
        y_pred_on_test_classification[i, 0], ',', y_pred_on_test_regression[i, 0], ',', y_pred_on_test_ensemble[
            0, i, 0], ',', y_pred_on_test_ensemble[1, i, 0]
    output_file.flush()
    output_file.close()