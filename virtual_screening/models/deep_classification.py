import argparse
import pandas as pd
import csv
import numpy as np
import json
import keras
import sys
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from sklearn.cross_validation import StratifiedShuffleSplit
from virtual_screening.function import read_merged_data, extract_feature_and_label, store_data, transform_json_to_csv
from virtual_screening.evaluation import roc_auc_single, roc_auc_multi, bedroc_auc_multi, bedroc_auc_single, \
    precision_auc_multi, precision_auc_single, enrichment_factor_multi, enrichment_factor_single
from virtual_screening.models.CallBacks import KeckCallBackOnROC, KeckCallBackOnPrecision, \
    MultiCallBackOnROC, MultiCallBackOnPR


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
            zero_weight = 1.0 * share
            one_weight = zero_weight * w['0'] / w['1'] * share
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
            zero_weight = 1.0 * share
            one_weight = zero_weight * w['0'] / w['1'] * share
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


class SingleClassification:
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

        return

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


class MultiClassification:
    def __init__(self, conf):
        self.conf = conf
        self.input_layer_dimension = 1024
        self.output_layer_dimension = 129

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
                          PMTNN_weight_file,
                          score_file,
                          sample_weight,
                          eval_indices=[-1],
                          eval_mean_or_median=np.mean):
        def get_model_roc_auc(true_label,
                              predicted_label,
                              eval_indices=eval_indices,
                              eval_mean_or_median=eval_mean_or_median):
            return roc_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

        def get_model_bedroc_auc(true_label,
                                 predicted_label,
                                 eval_indices=eval_indices,
                                 eval_mean_or_median=eval_mean_or_median):
            return bedroc_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

        def get_model_precision_auc(true_label,
                                    predicted_label,
                                    eval_indices=eval_indices,
                                    eval_mean_or_median=eval_mean_or_median):
            return precision_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

        model = self.setup_model()
        if self.early_stopping_option == 'auc':
            early_stopping = MultiCallBackOnROC(X_train, y_train,
                                                X_val, y_val,
                                                eval_indices, eval_mean_or_median,
                                                patience=self.early_stopping_patience,
                                                file_path=PMTNN_weight_file)
            callbacks = [early_stopping]
        elif self.early_stopping_option == 'precision':
            early_stopping = MultiCallBackOnPR(X_train, y_train,
                                               X_val, y_val,
                                               eval_indices, eval_mean_or_median,
                                               patience=self.early_stopping_patience,
                                               file_path=PMTNN_weight_file)
            callbacks = [early_stopping]
        else:
            callbacks = []


        if self.weight_schema == 'no_weight':
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
        else:
            reference = []
            total_num = X_train.shape[0]
            for i in range(self.output_layer_dimension):
                active = sum(y_train[:, i])
                active_and_inactive = sum(sample_weight[:, i])
                inactive = active_and_inactive - active
                missing = total_num - active_and_inactive
                reference.append({'-1': missing, '0': inactive, '1': active})

            cw = get_class_weight(self, y_train, reference=reference)
            print 'cw ', cw

            # TODO: customize loss function
            def customized_loss(y_true, y_pred):
                import theano.tensor as T
                y_pred = T.clip(y_pred, epsilon, 1.0 - epsilon)
                sum_ = 0
                print y_true.shape
                # T.nnet.binary_crossentropy(x_recons, x).mean()
                bce = T.nnet.binary_crossentropy(y_pred, y_true).mean(axis=-1)
                return bce

            # model.compile(loss=self.compile_loss, optimizer=self.compile_optimizer, sample_weight_mode="temporal")
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

        print('train precision: {}'.format(get_model_precision_auc(y_train, y_pred_on_train)))
        print('train roc: {}'.format(get_model_roc_auc(y_train, y_pred_on_train)))
        print('train bedroc: {}'.format(get_model_bedroc_auc(y_train, y_pred_on_train)))
        print
        print('validation precision: {}'.format(get_model_precision_auc(y_val, y_pred_on_val)))
        print('validation roc: {}'.format(get_model_roc_auc(y_val, y_pred_on_val)))
        print('validation bedroc: {}'.format(get_model_bedroc_auc(y_val, y_pred_on_val)))
        print
        print('test precision: {}'.format(get_model_precision_auc(y_test, y_pred_on_test)))
        print('test roc: {}'.format(get_model_roc_auc(y_test, y_pred_on_test)))
        print('test bedroc: {}'.format(get_model_bedroc_auc(y_test, y_pred_on_test)))
        print

        # Just print last target EF into output file.
        for EF_ratio in self.EF_ratio_list:
            n_actives, ef, ef_max = enrichment_factor_single(y_test[:, -1], y_pred_on_test[:, -1], EF_ratio)
            print('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))

        # Store all the target EF into score file.
        out = open(score_file, 'w')
        print >> out, "EF"
        for EF_ratio in self.EF_ratio_list:
            for i in range(y_test.shape[1]):
                n_actives, ef, ef_max = enrichment_factor_single(y_test[:, i], y_pred_on_test[:, i], EF_ratio)
                print >> out, 'ratio:', EF_ratio, 'EF:', ef, 'active:', n_actives

        return

    def predict_with_existing(self,
                              X_train, y_train,
                              X_val, y_val,
                              X_test, y_test,
                              PMTNN_weight_file,
                              score_file,
                              eval_indices=[-1],
                              eval_mean_or_median=np.mean):
        def get_model_roc_auc(true_label,
                              predicted_label,
                              eval_indices=eval_indices,
                              eval_mean_or_median=eval_mean_or_median):
            return roc_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

        def get_model_bedroc_auc(true_label,
                                 predicted_label,
                                 eval_indices=eval_indices,
                                 eval_mean_or_median=eval_mean_or_median):
            return bedroc_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

        def get_model_precision_auc(true_label,
                                    predicted_label,
                                    eval_indices=eval_indices,
                                    eval_mean_or_median=eval_mean_or_median):
            return precision_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

        model = self.setup_model()
        model.load_weights(PMTNN_weight_file)

        y_pred_on_train = model.predict(X_train)
        y_pred_on_val = model.predict(X_val)
        y_pred_on_test = model.predict(X_test)

        print('train precision: {}'.format(get_model_precision_auc(y_train, y_pred_on_train)))
        print('train roc: {}'.format(get_model_roc_auc(y_train, y_pred_on_train)))
        print('train bedroc: {}'.format(get_model_bedroc_auc(y_train, y_pred_on_train)))
        print
        print('validation precision: {}'.format(get_model_precision_auc(y_val, y_pred_on_val)))
        print('validation roc: {}'.format(get_model_roc_auc(y_val, y_pred_on_val)))
        print('validation bedroc: {}'.format(get_model_bedroc_auc(y_val, y_pred_on_val)))
        print
        print('test precision: {}'.format(get_model_precision_auc(y_test, y_pred_on_test)))
        print('test roc: {}'.format(get_model_roc_auc(y_test, y_pred_on_test)))
        print('test bedroc: {}'.format(get_model_bedroc_auc(y_test, y_pred_on_test)))
        print

        out = open(score_file, 'w')
        print >> out, "EF"
        for EF_ratio in self.EF_ratio_list:
            for i in range(y_test.shape[1]):
                n_actives, ef, ef_max = enrichment_factor_single(y_test[:, i], y_pred_on_test[:, i], EF_ratio)
                print >> out, 'ratio:', EF_ratio, 'EF:', ef, 'active:', n_actives

        return

    def get_EF_score_with_existing_model(self,
                                         X_test, y_test,
                                         file_path, EF_ratio,
                                         eval_indices=[-1],
                                         eval_mean_or_median=np.mean):
        def get_model_roc_auc(true_label,
                              predicted_label,
                              eval_indices=eval_indices,
                              eval_mean_or_median=eval_mean_or_median):
            return roc_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

        def get_model_bedroc_auc(true_label,
                                 predicted_label,
                                 eval_indices=eval_indices,
                                 eval_mean_or_median=eval_mean_or_median):
            return bedroc_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

        def get_model_precision_auc(true_label,
                                    predicted_label,
                                    eval_indices=eval_indices,
                                    eval_mean_or_median=eval_mean_or_median):
            return precision_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

        model = self.setup_model()
        model.load_weights(file_path)
        y_pred_on_test = model.predict(X_test)

        print('test precision: {}'.format(get_model_precision_auc(y_test, y_pred_on_test)))
        print('test roc: {}'.format(get_model_roc_auc(y_test, y_pred_on_test)))
        print('test bedroc: {}'.format(get_model_bedroc_auc(y_test, y_pred_on_test)))

        EF_list = enrichment_factor_multi(y_test,
                                          y_pred_on_test,
                                          percentile=EF_ratio,
                                          eval_indices=eval_indices)
        return EF_list


def demo_single_classification():
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
    cross_validation_split = StratifiedShuffleSplit(y_train, 1, test_size=0.2, random_state=1)
    for t_index, val_index in cross_validation_split:
        X_t, X_val = X_train[t_index], X_train[val_index]
        y_t, y_val = y_train[t_index], y_train[val_index]
    print 'done data preparation'

    print conf['label_name_list']
    task = SingleClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, PMTNN_weight_file)
    store_data(transform_json_to_csv(config_json_file), config_csv_file)

    return


def demo_multi_classification():
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    label_name_list = conf['label_name_list']
    print 'label_name_list ', label_name_list

    # specify dataset
    k = 5
    directory = '../../dataset/keck_pcba/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    output_file_list = [directory + f_ for f_ in file_list]
    train_pd = read_merged_data(output_file_list[0:3])
    train_pd.fillna(0, inplace=True)
    val_pd = read_merged_data(output_file_list[3:4])
    val_pd.fillna(0, inplace=True)
    test_pd = read_merged_data(output_file_list[4:5])
    test_pd.fillna(0, inplace=True)

    multi_name_list = train_pd.columns[-128:].tolist()
    multi_name_list.extend(label_name_list)
    print 'multi_name_list ', multi_name_list

    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=multi_name_list)
    X_val, y_val = extract_feature_and_label(val_pd,
                                             feature_name='Fingerprints',
                                             label_name_list=multi_name_list)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=multi_name_list)

    sample_weight_dir = '../../dataset/sample_weights/keck_pcba/fold_5/'
    file_list = []
    for i in range(k):
        file_list.append('sample_weight_{}.csv'.format(i))
    sample_weight_file = [sample_weight_dir + f_ for f_ in file_list]
    sample_weight_pd = read_merged_data(sample_weight_file[0:3])
    _, sample_weight = extract_feature_and_label(sample_weight_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=labels_list)
    print 'done data preparation'

    task = MultiClassification(conf=conf)
    task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test,
                           sample_weight=sample_weight,
                           PMTNN_weight_file=PMTNN_weight_file,
                           score_file=score_file)
    store_data(transform_json_to_csv(config_json_file), config_csv_file)

    whole_EF = []
    for EF_ratio in task.EF_ratio_list:
        EF_list = task.get_EF_score_with_existing_model(X_test, y_test, PMTNN_weight_file, EF_ratio)
        whole_EF.append([EF_ratio])
        whole_EF.append(EF_list)
        print(EF_ratio, EF_list)
        print
    print whole_EF

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

    if mode == 'single_classification':
        demo_single_classification()
    elif mode == 'multi_classification':
        score_file = given_args.score_file
        demo_multi_classification()
