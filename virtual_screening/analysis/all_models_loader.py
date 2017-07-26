from evaluation import *
import argparse
import pandas as pd
import csv
import numpy as np
import json
import sys
sys.path.insert(0, '..')  # Add path from parent folder
sys.path.insert(0, '.')  # Add path from current folder
from function import *
import os
import deepchem as dc
from sklearn.externals import joblib
from deepchem.trans import undo_transforms
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from shutil import copy2
import copy


"""
Function that loads all models for each class:

1- random_forest
2- irv
3- lightchem
4- neural_networks

returns a list with following hierarchy:

class_name -> model_name -> fold_# -> labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test
"""
def stage_1_results(model_directory, data_directory):
    #define folders for each class    
    class_dirs = [model_directory+'/random_forest/stage_1/',
                  model_directory+'/irv/390010/',
                  model_directory+'/lightchem/',
                  model_directory+'/neural_networks/',]
    
    stage_1_list = {'random_forest' : get_rf_results(class_dirs[0], data_directory),
                    'irv' : get_irv_results(class_dirs[1], data_directory),
                    'lightchem' : get_lightchem_results(class_dirs[2], data_directory),
                    'neural_networks' : get_nn_results(class_dirs[3], data_directory)
                     }    
    return stage_1_list            
    
"""
Loads random_forest model results as a list:
model_name -> fold_# -> labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test
"""  
def get_rf_results(model_directory, data_directory, k=5):
    model_list = {}
    if not os.path.exists(model_directory):
        return model_list
        
    model_names = os.listdir(model_directory)
    
    #load data
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))
    output_file_list = [data_directory + f_ for f_ in file_list]
    
    for m_name in model_names:
        model_list[m_name] = {}
        for i in range(k):  
            fold_dir = model_directory+'/'+m_name+'/fold_'+str(i)              
            csv_file_list = output_file_list[:]
            test_pd = read_merged_data([csv_file_list[i]])
            csv_file_list.pop(i)
            val_pd = read_merged_data([csv_file_list[i%len(csv_file_list)]])
            csv_file_list.pop(i%len(csv_file_list))
            train_pd = read_merged_data(csv_file_list)
            
            labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
        
            # extract data, and split training data into training and val
            X_train, y_train = extract_feature_and_label(train_pd,
                                                         feature_name='Fingerprints',
                                                         label_name_list=labels)
            
            X_val, y_val = extract_feature_and_label(val_pd,
                                                     feature_name='Fingerprints',
                                                     label_name_list=labels)
                                                       
            X_test, y_test = extract_feature_and_label(test_pd,
                                                       feature_name='Fingerprints',
                                                       label_name_list=labels)
        
            X_train = np.concatenate((X_train, X_val))
            y_train = np.concatenate((y_train, y_val))
            #load model and predict
            y_pred_on_train = np.zeros(shape=y_train.shape)
            y_pred_on_test = np.zeros(shape=y_test.shape)
            for j, label in zip(range(len(labels)), labels):
                model = joblib.load(fold_dir+'/rf_clf_'+label+'.pkl')
                y_pred_on_train[:,j] = model.predict_proba(X_train)[:,1]
                y_pred_on_test[:,j] = model.predict_proba(X_test)[:,1]
                
            model_list[m_name]['fold_'+str(i)]  = (labels, y_train, np.nan, y_test,
                                                   y_pred_on_train, np.nan, y_pred_on_test)
    
    return model_list

"""
Loads irv model results as a list:
model_name -> fold_# -> labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test
"""
def get_irv_results(model_directory, data_directory, k=5):
    model_list = {}
    if not os.path.exists(model_directory):
        return model_list
        
    model_names = os.listdir(model_directory)
    K_neighbors = [5, 10, 20, 40, 80]    
    
    #load data
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))
    output_file_list = [data_directory + f_ for f_ in file_list]
    
    for m_name in model_names:
        mi = int(m_name.split('_')[-1])        
        model_list[m_name] = {}
        
        for ti in range(k):
            for vi in [(ti+1)%k]:                
                i=ti                 
                
                fold_dir = model_directory+'/'+m_name+'/fold_'+str(i)+'/'
                log_dir = model_directory+'/'+m_name+'/tf_checkpoints/'
                
                csv_file_list = output_file_list[:]
                test_files = [csv_file_list[ti]]
                val_files = [csv_file_list[vi]]
                
                
                train_files = [q for j, q in enumerate(csv_file_list) if j not in [ti, vi]]
                
                labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
                
                featurizer='ECFP'
                if featurizer == 'ECFP':
                    featurizer_func = dc.feat.CircularFingerprint(size=1024)
                elif featurizer == 'GraphConv':
                    featurizer_func = dc.feat.ConvMolFeaturizer()
                    
                loader = dc.data.CSVLoader(tasks=labels, 
                                           smiles_field="SMILES", 
                                           featurizer=featurizer_func,
                                           verbose=False)
            
                train_data = loader.featurize(train_files, shard_size=2**15)
                val_data = loader.featurize(val_files, shard_size=2**15)
                test_data = loader.featurize(test_files, shard_size=2**15)      
                
                train_data = dc.data.NumpyDataset(train_data.X, train_data.y, train_data.w, train_data.ids)
                val_data = dc.data.NumpyDataset(val_data.X, val_data.y, val_data.w, val_data.ids)
                test_data = dc.data.NumpyDataset(test_data.X, test_data.y, test_data.w, test_data.ids)

                bal_train_data = dc.trans.BalancingTransformer(transform_w=True, dataset=train_data).transform(train_data)
                transformers = [dc.trans.IRVTransformer(K_neighbors[mi], len(labels), bal_train_data)]
                for transformer in transformers:
                    train_data = transformer.transform(train_data)
                    val_data = transformer.transform(val_data)
                    test_data = transformer.transform(test_data)       
            
                #load model and predict
                model = dc.models.TensorflowMultiTaskIRVClassifier(
                                                       len(labels),
                                                       K=K_neighbors[mi],
                                                       learning_rate=0.01,
                                                       penalty=0.05,
                                                       batch_size=8192,
                                                       fit_transformers=[],
                                                       logdir=log_dir,
                                                       verbose=False) 
                                                       
                curr_ckpt_file = log_dir+'model.ckpt-2'
                best_ckpt_file = fold_dir+'best.ckpt'
                
                copy2(best_ckpt_file+'.data-00000-of-00001', curr_ckpt_file+'.data-00000-of-00001')
                copy2(best_ckpt_file+'.index', curr_ckpt_file+'.index')
                copy2(best_ckpt_file+'.meta', curr_ckpt_file+'.meta')
                
                y_pred_on_train = model.predict_proba(train_data)[:,:,1]
                y_pred_on_val = model.predict_proba(val_data)[:,:,1]
                y_pred_on_test = model.predict_proba(test_data)[:,:,1]
                
                y_train = copy.deepcopy(train_data.y)
                w_train = copy.deepcopy(train_data.w)  
                y_val = copy.deepcopy(val_data.y)
                w_val = copy.deepcopy(val_data.w) 
                y_test = copy.deepcopy(test_data.y)
                w_test = copy.deepcopy(test_data.w) 
                
                for j, label in zip(range(len(labels)), labels): 
                    y_train[np.where(w_train[:,j] == 0)[0],j] = np.nan
                    y_val[np.where(w_val[:,j] == 0)[0],j] = np.nan
                    y_test[np.where(w_test[:,j] == 0)[0],j] = np.nan
            
                    
                model_list[m_name]['fold_'+str(i)]  = (labels, y_train, y_val, y_test,
                                                       y_pred_on_train, y_pred_on_val, y_pred_on_test)    
    return model_list

"""
Loads lightchem model results as a list:
model_name -> fold_# -> labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test
"""
def get_lightchem_results(model_directory, data_directory, k=5):
    model_list = {}
    if not os.path.exists(model_directory):
        return model_list
        
    model_names = os.listdir(model_directory)
    
    #load data

    #load model and predict

    return model_list

"""
Loads neural_network model results as a list:
model_name -> fold_# -> labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test
"""
def get_nn_results(model_directory, data_directory, k=20):
    model_list = {}
    if not os.path.exists(model_directory):
        return model_list
        
    model_names = os.listdir(model_directory)
    
    #load data
    
    #load model and predict
    
    return model_list
       
