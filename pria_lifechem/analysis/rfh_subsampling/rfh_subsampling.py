import argparse
import pandas as pd
import csv
import numpy as np
import json
import sys
sys.path.insert(0, '..')  # Add path from parent folder
sys.path.insert(0, '.')  # Add path from current folder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.grid_search import ParameterGrid
from shutil import move
import os
import itertools
from sklearn.model_selection import train_test_split
from evaluation import *

"""
 This class is similar to SKLearn_RandomForest but with removed features and modified constructor.
 The seed is set as np.random.seed(seed=(process_num*train_count+run_num)).
 
 Note: Unlike the other models, this model was trained and evaluated on the prospective data 
       AFTER the true prospective data was available. The goal is to analyze the effects of
       varying the number of training molecules on prospective performance.
"""
class RFH_Sklearn:
    def __init__(self, conf, process_num):
        self.conf = conf
        self.input_layer_dimension = 1024
        self.label_names = conf['label_names']
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        
        self.process_num = process_num
        
        self.param = conf['params']
        self.n_estimators = self.param['n_estimators']
        self.max_features = self.param['max_features']
        self.min_samples_leaf = self.param['min_samples_leaf']
        self.class_weight = self.param['class_weight']
        print('Testing set:', self.param)
        
        self.model_dict = {}
        return
        
    def get_prediction_info(self, X, y_true):
        y_pred = np.zeros(shape=y_true.shape)        
        
        for i, label in zip(range(len(self.label_names)), self.label_names):     
            model = self.model_dict[label]
            
            y_true[np.where(np.isnan(y_true[:,i]))[0],i] = -1
            if i in [0,1,2]:   
                rf_pred = model.predict_proba(X)
                if rf_pred.shape[1] == 1:
                    y_pred[:,i] = 0
                else:
                    y_pred[:,i] = rf_pred[:,1]
        
        return y_true, y_pred
        
    def setup_model(self):
        for i in range(len(self.label_names)):
            self.model_dict[self.label_names[i]] = RandomForestClassifier(n_estimators=self.n_estimators, 
                                           max_features=self.max_features, 
                                           min_samples_leaf=self.min_samples_leaf, 
                                           n_jobs=3, 
                                           class_weight=self.class_weight,
                                           random_state=self.process_num,
                                           oob_score=False, 
                                           verbose=1) 
        return
        
        
    def train(self, X_train, y_train):
        self.setup_model()
        
        # perform random shuffling of training data (including X_train)
        p = np.random.permutation(len(X_train))
        X_train = X_train[p,:]
        y_train = y_train[p,:]
        
        for i, label in zip(range(len(self.label_names)), self.label_names):
            y = y_train[:,i]
            indexes = np.where(np.isnan(y))[0]
                
            y = np.delete(y, indexes, axis=0)
            X = np.delete(X_train, indexes, axis=0)
            self.model_dict[label].fit(X, y)
        return

    def predict_with_existing(self, X, y):          
        y_true, y_pred = self.get_prediction_info(X, y)
        
        print
        print('precision: {}'.format(precision_auc_multi(y_true, y_pred, range(y_true.shape[1]), np.mean)))
        print('roc: {}'.format(roc_auc_multi(y_true, y_pred, range(y_true.shape[1]), np.mean)))
        print('bedroc: {}'.format(bedroc_auc_multi(y_true, y_pred, range(y_true.shape[1]), np.mean)))
        print
        
        label_list = self.label_names
        nef_auc_mean = np.mean(np.array(nef_auc(y_true, y_pred, self.EF_ratio_list, label_list))) 
        print('nef auc: {}'.format(nef_auc_mean))
        return
        
    def save_model_params(self, config_csv_file):      
        data = str(self.param)
        with open(config_csv_file, 'w') as csvfile:
            csvfile.write(data)
        return
        
'''
Note: Copied from function.py

Read the data from all files in input_file_list
And merged into one dataset
'''
def read_merged_data(input_file_list, usecols=None):
    whole_pd = pd.DataFrame()
    for input_file in input_file_list:
        data_pd = pd.read_csv(input_file, usecols=usecols)
        whole_pd = whole_pd.append(data_pd)
    return whole_pd

'''
Note: Copied from function.py

Reshape vector into 2-dimension matrix.
'''
def reshape_data_into_2_dim(data):
    if data.ndim == 1:
        n = data.shape[0]
        data = data.reshape(n, 1)
    return data
    
'''
Note: Copied from function.py

Get the fingerprints, with feature_name specified, and label_name specified
'''
def extract_feature_and_label(data_pd,
                              feature_name,
                              label_name_list):
    # By default, feature should be fingerprints
    X_data = data_pd[feature_name].tolist()
    X_data = [list(x) for x in X_data]
    X_data = np.array(X_data)

    y_data = data_pd[label_name_list].values.tolist()
    y_data = np.array(y_data)
    y_data = reshape_data_into_2_dim(y_data)

    X_data = X_data.astype(float)
    y_data = y_data.astype(float)

    return X_data, y_data
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', action="store", dest="config_json_file", required=True)
    parser.add_argument('--model_dir', action="store", dest="model_dir", required=True)
    parser.add_argument('--dataset_dir', action="store", dest="dataset_dir", required=True)
    parser.add_argument('--prospective_file', action="store", dest="prospective_file", required=True)
    parser.add_argument('--process_num', action="store", dest="process_num", required=True)
    
    #####
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    model_dir = given_args.model_dir
    dataset_dir = given_args.dataset_dir
    prospective_file = given_args.prospective_file
    process_num = int(given_args.process_num)
    #####
    config_csv_file = model_dir+'model_config.csv'
    #####
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    

    # specify dataset
    directory = dataset_dir
    file_list = []
    k=5
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))

    # merge training and test dataset
    labels = ["Keck_Pria_AS_Retest"]
    dtype_list = {'Molecule': np.str,
                  'SMILES': np.str,
                  'Fingerprints': np.str,
                  'Keck_Pria_AS_Retest': np.int64,
                  'Keck_Pria_FP_data': np.int64,
                  'Keck_Pria_Continuous': np.float64,
                  'Keck_RMI_cdd': np.float64}
    output_file_list = [directory + f_ for f_ in file_list]
    
    csv_file_list = output_file_list[:]
    train_pd = read_merged_data(csv_file_list)
    test_pd = pd.read_csv(prospective_file, compression='gzip')
    
    # extract data
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=labels)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=labels)
    train_indices = np.arange(X_train.shape[0])
    
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    
    start_mol_count = int(conf["start_mol_count"])
    train_count_spacing = int(conf["train_count_spacing"])
    max_runs = int(conf["max_runs"])
    
    # get train_count and run_num for this process_num
    train_counts = np.linspace(start_mol_count, X_train.shape[0], train_count_spacing, dtype=np.int)
    total_runs = np.arange(max_runs)
    train_count, run_num = list(itertools.product(train_counts,total_runs))[process_num]
    
    # set seed according to (process_num*train_count+run_num)
    np.random.seed(seed=(process_num*train_count+run_num))
    
    # stratify sample the training set
    X_train, _, y_train, _, train_indices, _ = train_test_split(X_train, y_train, train_indices, train_size=train_count, stratify=y_train)
    print('done data preparation')
    
    task = RFH_Sklearn(conf=conf, process_num=process_num)
    task.train(X_train, y_train)
    task.save_model_params(config_csv_file)
                                       
    # save evaluation results
    output_dir = model_dir+'/n_{}/run_{}/'.format(train_count, run_num)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir+'/train_metrics/'):
        os.makedirs(output_dir+'/train_metrics/')
    if not os.path.exists(output_dir+'/val_metrics/'):
        os.makedirs(output_dir+'/val_metrics/')
    if not os.path.exists(output_dir+'/test_metrics/'):
        os.makedirs(output_dir+'/test_metrics/')
        
    y_train, y_pred_on_train = task.get_prediction_info(X_train, y_train)
    y_test, y_pred_on_test = task.get_prediction_info(X_test, y_test)
    # evaluate on the training set
    evaluate_model(y_train, y_pred_on_train, output_dir+'/train_metrics/', labels)
    # evaluate on the prospective set
    evaluate_model(y_test, y_pred_on_test, output_dir+'/test_metrics/', labels)
    
    # save train_indices
    np.save(output_dir+'/train_indices', train_indices)