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
from rfh_subsampling import *

"""
    Trains random forest using RF_h settings on smaller set of the cross-validation data 
    and evaluates on prospective data.
"""
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', action="store", dest="config_json_file", required=True)
    parser.add_argument('--model_dir', action="store", dest="model_dir", required=True)
    parser.add_argument('--dataset_dir', action="store", dest="dataset_dir", required=True)
    parser.add_argument('--tain_indices_file', action="store", dest="tain_indices_file", required=True)
    parser.add_argument('--prospective_file', action="store", dest="prospective_file", required=True)
    parser.add_argument('--process_num', action="store", dest="process_num", required=True)
    
    #####
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    model_dir = given_args.model_dir
    dataset_dir = given_args.dataset_dir
    tain_indices_file = given_args.tain_indices_file
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
    train_indices = np.load(train_indices_file)
    
    # use only the specified train_indices for training
    X_train, y_train = X_train[train_indices,:], y_train[train_indices,:]
    print('done data preparation')
    
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
        
    task = RFH_Sklearn(conf=conf, process_num=process_num)
    task.train(X_train, y_train)
    task.save_model_params(config_csv_file)
                                       
    # save evaluation results
    output_dir = model_dir+'/{}/'.format(train_indices_file.replace('.npy',''))
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