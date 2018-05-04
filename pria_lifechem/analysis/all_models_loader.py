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
import os, glob
import deepchem as dc
from sklearn.externals import joblib
from deepchem.trans import undo_transforms
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
from shutil import copy2
import copy
from models.deep_classification import *
from models.deep_regression import *
from models.vanilla_lstm import *


"""
Function that loads all models for each class:

1- random_forest
2- irv
4- neural_networks

returns a dict with following hierarchy:

class_name -> model_name -> fold_# -> labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test
"""
def stage_1_results(model_directory, data_directory):
    #define folders for each class    
    class_dirs = [model_directory+'/random_forest/stage_1/',
                  model_directory+'/irv/stage_1/',
                  model_directory+'/neural_networks/stage_1/',
                  model_directory+'/docking/stage_1/']
    
    stage_1_dict = {'random_forest' : get_rf_results_stage_1(class_dirs[0], data_directory),
                    'irv' : get_irv_results_stage_1(class_dirs[1], data_directory),
                    'neural_networks' : get_nn_results_stage_1(class_dirs[3], data_directory),
                    'docking' : get_docking_results_stage_1(class_dirs[4], data_directory)
                     }    
    return stage_1_dict    

def stage_2_results(model_directory, data_directory, held_out_data_file):
    #define folders for each class    
    class_dirs = [model_directory+'/random_forest/stage_2/',
                  model_directory+'/irv/stage_2/',
                  model_directory+'/neural_networks/stage_2/',
                  model_directory+'/docking/stage_2/',
                  model_directory+'/baseline/stage_2/']
    
    stage_2_dict = {'random_forest' : get_rf_results_stage_2(class_dirs[0], data_directory, held_out_data_file),
                    'irv' : get_irv_results_stage_2(class_dirs[1], data_directory, held_out_data_file),
                    'neural_networks' : get_nn_results_stage_2(class_dirs[3], data_directory, held_out_data_file),
                    'docking' : get_docking_results_stage_2(class_dirs[4], data_directory, held_out_data_file),
                    'baseline' : get_baseline_results_stage_2(class_dirs[5], data_directory, held_out_data_file)
                     }    
    return stage_2_dict        
    
"""
Loads random_forest model results as a list:
model_name -> fold_# -> labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test
"""  
def get_rf_results_stage_1(model_directory, data_directory, k=5):
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

def get_rf_results_stage_2(model_directory, data_directory, held_out_data_file, k=5):
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
        for i in range(1):  
            fold_dir = model_directory+'/'+m_name+'/fold_'+str(i)    
            train_pd = read_merged_data(output_file_list)
            test_pd = read_merged_data([held_out_data_file])      
            
            labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
        
            # extract data, and split training data into training and val
            X_train, y_train = extract_feature_and_label(train_pd,
                                                         feature_name='Fingerprints',
                                                         label_name_list=labels)
            
                                                       
            X_test, y_test = extract_feature_and_label(test_pd,
                                                       feature_name='Fingerprints',
                                                       label_name_list=labels)
        
            #load model and predict
            y_pred_on_train = np.zeros(shape=y_train.shape)
            y_pred_on_test = np.zeros(shape=y_test.shape)
            for j, label in zip(range(len(labels)), labels):
                model = joblib.load(fold_dir+'/rf_clf_'+label+'.pkl')
                y_pred_on_train[:,j] = model.predict_proba(X_train)[:,1]
                y_pred_on_test[:,j] = model.predict_proba(X_test)[:,1]
                
            model_list[m_name]  = (labels, y_train, np.nan, y_test,
                                   y_pred_on_train, np.nan, y_pred_on_test)
    
    return model_list
    

"""
Loads irv model results as a list:
model_name -> fold_# -> labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test
"""
def get_irv_results_stage_1(model_directory, data_directory, k=5):
    model_list = {}
    if not os.path.exists(model_directory):
        return model_list
        
    model_names = os.listdir(model_directory)
    
    #load data
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))
    output_file_list = [data_directory + f_ for f_ in file_list]
    
    labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
    for m_name in model_names:
        K_neighbors = int(m_name.split('_')[-1])        
        model_list[m_name] = {}
        
        for ti in range(k):
            for vi in [(ti+1)%k]:                
                i=ti                 
                
                fold_dir = model_directory+'/'+m_name+'/fold_'+str(i)+'/'
                logdir = model_directory+'/'+m_name+'/tf_checkpoints/'
                
                csv_file_list = output_file_list[:]
                test_files = [csv_file_list[ti]]
                val_files = [csv_file_list[vi]]
                
                model_dict = {}
                
                train_files = [q for j, q in enumerate(csv_file_list) if j not in [ti, vi]]
                
                train_data = []
                val_data = []
                test_data = []    
                bal_train_data = []
                
                featurizer='ECFP'
                if featurizer == 'ECFP':
                    featurizer_func = dc.feat.CircularFingerprint(size=1024)
                elif featurizer == 'GraphConv':
                    featurizer_func = dc.feat.ConvMolFeaturizer()
                
                for label_index in range(len(labels)):
                    loader = dc.data.CSVLoader(tasks=[labels[label_index]], 
                                               smiles_field="SMILES", 
                                               featurizer=featurizer_func,
                                               verbose=False)
            
                    # extract data, and split training data into training and val  
                    bal_train_data.append(loader.featurize(train_files, shard_size=2**15))                  
                    train_data.append(loader.featurize(train_files, shard_size=2**15))
                    val_data.append(loader.featurize(val_files, shard_size=2**15))
                    test_data.append(loader.featurize(test_files, shard_size=2**15))
            
                    bal_train_data[label_index] = dc.data.NumpyDataset(bal_train_data[label_index].X, bal_train_data[label_index].y, 
                                                                bal_train_data[label_index].w, bal_train_data[label_index].ids)
                    train_data[label_index] = dc.data.NumpyDataset(train_data[label_index].X, train_data[label_index].y, 
                                                                   train_data[label_index].w, train_data[label_index].ids)
                    val_data[label_index] = dc.data.NumpyDataset(val_data[label_index].X, val_data[label_index].y, 
                                                                 val_data[label_index].w, val_data[label_index].ids)
                    test_data[label_index] = dc.data.NumpyDataset(test_data[label_index].X, test_data[label_index].y, 
                                                                  test_data[label_index].w, test_data[label_index].ids)
                
                    
                    bal_train_data[label_index] = dc.trans.BalancingTransformer(transform_w=True, dataset=bal_train_data[label_index]).transform(bal_train_data[label_index])
                    transformers = [dc.trans.IRVTransformer(K_neighbors, 1, bal_train_data[label_index])]
                    for transformer in transformers:
                        train_data[label_index] = transformer.transform(train_data[label_index])
                        val_data[label_index] = transformer.transform(val_data[label_index])
                        test_data[label_index] = transformer.transform(test_data[label_index])       
        
                    model_dict[labels[label_index]] = dc.models.TensorflowMultiTaskIRVClassifier(
                                                       1,
                                                       K=K_neighbors,
                                                       learning_rate=0.01,
                                                       penalty=0.05,
                                                       batch_size=8192,
                                                       fit_transformers=[],
                                                       logdir=logdir+'/'+labels[label_index]+'/',
                                                       verbose=False)  
                                                       
                    curr_ckpt_file = logdir+'/'+labels[label_index]+'/model.ckpt-2'
                    best_ckpt_file = fold_dir+'/'+labels[label_index]+'/best.ckpt'
                    
                    copy2(best_ckpt_file+'.data-00000-of-00001', curr_ckpt_file+'.data-00000-of-00001')
                    copy2(best_ckpt_file+'.index', curr_ckpt_file+'.index')
                    copy2(best_ckpt_file+'.meta', curr_ckpt_file+'.meta')
                
                    if label_index == 0:
                        y_pred_on_train = np.zeros(shape=(train_data[label_index].y.shape[0], 3))
                        y_pred_on_val = np.zeros(shape=(val_data[label_index].y.shape[0], 3))
                        y_pred_on_test = np.zeros(shape=(test_data[label_index].y.shape[0], 3))
                        
                        y_train = np.zeros(shape=(train_data[label_index].y.shape[0], 3))
                        w_train = np.zeros(shape=(train_data[label_index].y.shape[0], 3)) 
                        y_val = np.zeros(shape=(val_data[label_index].y.shape[0], 3))  
                        w_val = np.zeros(shape=(val_data[label_index].y.shape[0], 3)) 
                        y_test = np.zeros(shape=(test_data[label_index].y.shape[0], 3)) 
                        w_test = np.zeros(shape=(test_data[label_index].y.shape[0], 3)) 
                        
                    y_pred_on_train[:,label_index] = model_dict[labels[label_index]].predict_proba(train_data[label_index])[:,:,1][:,0]
                    y_pred_on_val[:,label_index] = model_dict[labels[label_index]].predict_proba(val_data[label_index])[:,:,1][:,0]
                    y_pred_on_test[:,label_index] = model_dict[labels[label_index]].predict_proba(test_data[label_index])[:,:,1][:,0]
                    
                    y_train[:,label_index] = copy.deepcopy(train_data[label_index].y[:,0])
                    w_train[:,label_index] = copy.deepcopy(train_data[label_index].w[:,0])  
                    y_val[:,label_index] = copy.deepcopy(val_data[label_index].y[:,0])
                    w_val[:,label_index] = copy.deepcopy(val_data[label_index].w[:,0]) 
                    y_test[:,label_index] = copy.deepcopy(test_data[label_index].y[:,0])
                    w_test[:,label_index] = copy.deepcopy(test_data[label_index].w[:,0]) 
                
                for j, label in zip(range(len(labels)), labels): 
                    y_train[np.where(w_train[:,j] == 0)[0],j] = np.nan
                    y_val[np.where(w_val[:,j] == 0)[0],j] = np.nan
                    y_test[np.where(w_test[:,j] == 0)[0],j] = np.nan
            
                    
                model_list[m_name]['fold_'+str(i)]  = (labels, y_train, y_val, y_test,
                                                       y_pred_on_train, y_pred_on_val, y_pred_on_test)    
    return model_list

def get_irv_results_stage_2(model_directory, data_directory, held_out_data_file, k=5):
    model_list = {}
    if not os.path.exists(model_directory):
        return model_list
        
    model_names = os.listdir(model_directory)
    
    #load data
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))
    output_file_list = [data_directory + f_ for f_ in file_list]
    
    labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
    for m_name in model_names:
        K_neighbors = int(m_name.split('_')[-1])        
        model_list[m_name] = {}
        
        vi = 0
        i = 1
        
        csv_file_list = output_file_list[:]
        val_files = [csv_file_list[vi]]
        test_files = [held_out_data_file]
        
        train_files = [q for j, q in enumerate(csv_file_list) if j not in [vi]]               
        
        fold_dir = model_directory+'/'+m_name+'/fold_'+str(i)+'/'
        logdir = model_directory+'/'+m_name+'/tf_checkpoints/'
        
        model_dict = {}
                        
        train_data = []
        val_data = []
        test_data = []    
        bal_train_data = []
        
        featurizer='ECFP'
        if featurizer == 'ECFP':
            featurizer_func = dc.feat.CircularFingerprint(size=1024)
        elif featurizer == 'GraphConv':
            featurizer_func = dc.feat.ConvMolFeaturizer()
        
        for label_index in range(len(labels)):
            loader = dc.data.CSVLoader(tasks=[labels[label_index]], 
                                       smiles_field="SMILES", 
                                       featurizer=featurizer_func,
                                       verbose=False)
    
            # extract data, and split training data into training and val  
            bal_train_data.append(loader.featurize(train_files, shard_size=2**15))                  
            train_data.append(loader.featurize(train_files, shard_size=2**15))
            val_data.append(loader.featurize(val_files, shard_size=2**15))
            test_data.append(loader.featurize(test_files, shard_size=2**15))
    
            bal_train_data[label_index] = dc.data.NumpyDataset(bal_train_data[label_index].X, bal_train_data[label_index].y, 
                                                        bal_train_data[label_index].w, bal_train_data[label_index].ids)
            train_data[label_index] = dc.data.NumpyDataset(train_data[label_index].X, train_data[label_index].y, 
                                                           train_data[label_index].w, train_data[label_index].ids)
            val_data[label_index] = dc.data.NumpyDataset(val_data[label_index].X, val_data[label_index].y, 
                                                         val_data[label_index].w, val_data[label_index].ids)
            test_data[label_index] = dc.data.NumpyDataset(test_data[label_index].X, test_data[label_index].y, 
                                                          test_data[label_index].w, test_data[label_index].ids)
        
            
            bal_train_data[label_index] = dc.trans.BalancingTransformer(transform_w=True, dataset=bal_train_data[label_index]).transform(bal_train_data[label_index])
            transformers = [dc.trans.IRVTransformer(K_neighbors, 1, bal_train_data[label_index])]
            for transformer in transformers:
                train_data[label_index] = transformer.transform(train_data[label_index])
                val_data[label_index] = transformer.transform(val_data[label_index])
                test_data[label_index] = transformer.transform(test_data[label_index])       

            model_dict[labels[label_index]] = dc.models.TensorflowMultiTaskIRVClassifier(
                                               1,
                                               K=K_neighbors,
                                               learning_rate=0.01,
                                               penalty=0.05,
                                               batch_size=8192,
                                               fit_transformers=[],
                                               logdir=logdir+'/'+labels[label_index]+'/',
                                               verbose=False)  
                                               
            curr_ckpt_file = logdir+'/'+labels[label_index]+'/model.ckpt-2'
            best_ckpt_file = fold_dir+'/'+labels[label_index]+'/best.ckpt'
            
            copy2(best_ckpt_file+'.data-00000-of-00001', curr_ckpt_file+'.data-00000-of-00001')
            copy2(best_ckpt_file+'.index', curr_ckpt_file+'.index')
            copy2(best_ckpt_file+'.meta', curr_ckpt_file+'.meta')
        
            if label_index == 0:
                y_pred_on_train = np.zeros(shape=(train_data[label_index].y.shape[0], 3))
                y_pred_on_val = np.zeros(shape=(val_data[label_index].y.shape[0], 3))
                y_pred_on_test = np.zeros(shape=(test_data[label_index].y.shape[0], 3))
                
                y_train = np.zeros(shape=(train_data[label_index].y.shape[0], 3))
                w_train = np.zeros(shape=(train_data[label_index].y.shape[0], 3)) 
                y_val = np.zeros(shape=(val_data[label_index].y.shape[0], 3))  
                w_val = np.zeros(shape=(val_data[label_index].y.shape[0], 3)) 
                y_test = np.zeros(shape=(test_data[label_index].y.shape[0], 3)) 
                w_test = np.zeros(shape=(test_data[label_index].y.shape[0], 3)) 
            
            y_pred_on_train[:,label_index] = model_dict[labels[label_index]].predict_proba(train_data[label_index])[:,:,1][:,0]
            y_pred_on_val[:,label_index] = model_dict[labels[label_index]].predict_proba(val_data[label_index])[:,:,1][:,0]
            y_pred_on_test[:,label_index] = model_dict[labels[label_index]].predict_proba(test_data[label_index])[:,:,1][:,0]
            
            y_train[:,label_index] = copy.deepcopy(train_data[label_index].y[:,0])
            w_train[:,label_index] = copy.deepcopy(train_data[label_index].w[:,0])  
            y_val[:,label_index] = copy.deepcopy(val_data[label_index].y[:,0])
            w_val[:,label_index] = copy.deepcopy(val_data[label_index].w[:,0]) 
            y_test[:,label_index] = copy.deepcopy(test_data[label_index].y[:,0])
            w_test[:,label_index] = copy.deepcopy(test_data[label_index].w[:,0]) 
        
        for j, label in zip(range(len(labels)), labels): 
            y_train[np.where(w_train[:,j] == 0)[0],j] = np.nan
            y_val[np.where(w_val[:,j] == 0)[0],j] = np.nan
            y_test[np.where(w_test[:,j] == 0)[0],j] = np.nan
    
            
        model_list[m_name]  = (labels, y_train, y_val, y_test,
                               y_pred_on_train, y_pred_on_val, y_pred_on_test)    
    return model_list


"""
    Results from baseline method using similarity measure.
"""
def get_baseline_results_stage_2(model_directory, data_directory, held_out_data_file, k=5):
    model_list = {}
    labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
    
    model_names = os.listdir(model_directory)
    
    for m_name in model_names:
        model_list['baseline'] = {}
        for i in range(1): 
            test_pd = read_merged_data([held_out_data_file])
            
            _, y_test = extract_feature_and_label(test_pd,
                                                  feature_name='Fingerprints',
                                                  label_name_list=labels)
            y_pred_on_test = np.zeros(shape=(y_test.shape[0], 3))            
                
            y_pred_on_test[:,0] = np.array(pd.read_csv(model_directory+'/'+m_name)['Keck_Pria_AS_Retest'], dtype=float)
            y_pred_on_test[:,1] = np.nan
            y_pred_on_test[:,2] = np.nan
            
            model_list['baseline'] = (labels, np.nan,  np.nan, y_test,
                                  np.nan,  np.nan, y_pred_on_test)
            
    return model_list

"""
Loads neural_network model results as a list:
model_name -> fold_# -> labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test
"""
def get_nn_results_stage_1(model_directory, data_directory, k=20):
    model_list = {}
    if not os.path.exists(model_directory):
        return model_list
    
    model_names = []
    label_dirs = [model_directory + ldir + '/' for ldir in ['cross_validation_Keck_Pria_Retest',
                                                            'cross_validation_Keck_FP',
                                                            'cross_validation_RMI']]
    for l_dir in label_dirs:
        model_names.extend([f.rstrip('.json') for f in os.listdir(l_dir) if f.endswith('.json')])
        
    model_names = list(set(model_names))
    
    labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
    #load data
    file_list = []
    for i in range(5):
        file_list.append('file_{}.csv'.format(i))
    output_file_list = [data_directory + f_ for f_ in file_list]
    output_file_list = np.array(output_file_list)
    
    for m_name in model_names:
        model_list[m_name] = {}
        for running_index in range(k):  
            i=running_index
            test_index = running_index // 4
            val_index = running_index % 4 + (running_index % 4 >= test_index)
            complete_index = np.arange(5)
            train_index = np.where((complete_index != test_index) & (complete_index != val_index))[0]
            
            train_file_list = output_file_list[train_index]
            val_file_list = output_file_list[val_index:val_index+1]
            test_file_list = output_file_list[test_index:test_index+1]
            
            test_pd = read_merged_data(test_file_list)
            val_pd = read_merged_data(val_file_list)
            train_pd = read_merged_data(train_file_list)
            
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
                
            y_pred_on_train = np.zeros(shape=y_train.shape)
            y_pred_on_val = np.zeros(shape=y_val.shape)
            y_pred_on_test = np.zeros(shape=y_test.shape)
            
            for j, label in zip(range(len(labels)), labels):
                m_name_dir = label_dirs[j]+m_name+'/'
                if os.path.exists(m_name_dir):
                    fold_dir = [m_name_dir+f+'/' for f in os.listdir(m_name_dir) 
                                if str.isdigit(f)][0]
                    with open(label_dirs[j]+m_name+'.json', 'r') as f:
                        conf = json.load(f)
                        
                    model = None
                    if 'single_classification' in m_name:
                        model = SingleClassification(conf=conf)
                    elif 'multi_classification' in m_name:
                        model = MultiClassification(conf=conf)
                    elif 'single_regression' in m_name:
                        model = SingleRegression(conf=conf)
                    elif 'vanilla_lstm' in m_name:
                        model = VanillaLSTM(conf=conf)
                        SMILES_mapping_json_file=model_directory+'SMILES_mapping.json'
                        X_train, _ = extract_SMILES_and_label(train_pd,
                                                    feature_name='SMILES',
                                                    label_name_list=labels,
                                                    SMILES_mapping_json_file=SMILES_mapping_json_file)
                        X_val, _ = extract_SMILES_and_label(val_pd,
                                                            feature_name='SMILES',
                                                            label_name_list=labels,
                                                            SMILES_mapping_json_file=SMILES_mapping_json_file)
                        X_test, _ = extract_SMILES_and_label(test_pd,
                                                              feature_name='SMILES',
                                                              label_name_list=labels,
                                                              SMILES_mapping_json_file=SMILES_mapping_json_file)
                
                        X_train = sequence.pad_sequences(X_train, maxlen=model.padding_length)
                        X_val = sequence.pad_sequences(X_val, maxlen=model.padding_length)
                        X_test = sequence.pad_sequences(X_test, maxlen=model.padding_length)
                    
                    weight_file = fold_dir+str(i)+'.weight'
                    
                    if os.path.exists(weight_file):
                        model = model.setup_model()
                        model.load_weights(weight_file)
                        
                        y_pred_on_train[:,j] = model.predict(X_train)[:,-1]
                        y_pred_on_val[:,j] = model.predict(X_val)[:,-1]
                        y_pred_on_test[:,j] = model.predict(X_test)[:,-1]
                    else:
                        y_pred_on_train[:,j] = np.nan
                        y_pred_on_val[:,j] = np.nan
                        y_pred_on_test[:,j] = np.nan
                else:
                    y_pred_on_train[:,j] = np.nan
                    y_pred_on_val[:,j] = np.nan
                    y_pred_on_test[:,j] = np.nan
        
            #load model and predict
            model_list[m_name]['fold_'+str(i)]  = (labels, y_train, y_val, y_test,
                                                  y_pred_on_train, y_pred_on_val, y_pred_on_test)
                                                   
    return model_list

def get_nn_results_stage_2(model_directory, data_directory, held_out_data_file, k=20):
    model_list = {}
    if not os.path.exists(model_directory):
        return model_list
    
    model_names = []
    label_dirs = [model_directory + ldir + '/' for ldir in ['Keck_Pria_AS_Retest',
                                                            'Keck_FP',
                                                            'RMI']]
    for l_dir in label_dirs:
        model_names.extend([f.rstrip('.json') for f in os.listdir(l_dir) if f.endswith('.json')])
        
    model_names = list(set(model_names))
    
    labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
    #load data
    file_list = []
    for i in range(5):
        file_list.append('file_{}.csv'.format(i))
    output_file_list = [data_directory + f_ for f_ in file_list]
    output_file_list = np.array(output_file_list)
    
    for m_name in model_names:
        model_list[m_name] = {}
        
        i=0
        val_index = 0
        train_index = [1, 2, 3, 4]
        
        train_file_list = output_file_list[train_index]
        val_file_list = output_file_list[val_index:val_index+1]
        
        test_pd = read_merged_data([held_out_data_file])
        val_pd = read_merged_data(val_file_list)
        train_pd = read_merged_data(train_file_list)
        
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
            
        y_pred_on_train = np.zeros(shape=y_train.shape)
        y_pred_on_val = np.zeros(shape=y_val.shape)
        y_pred_on_test = np.zeros(shape=y_test.shape)
        
        for j, label in zip(range(len(labels)), labels):
            m_name_dir = label_dirs[j]+m_name+'/'
            if os.path.exists(m_name_dir):
                with open(label_dirs[j]+m_name+'.json', 'r') as f:
                    conf = json.load(f)
                    
                model = None
                if 'single_classification' in m_name:
                    model = SingleClassification(conf=conf)
                elif 'multi_classification' in m_name:
                    model = MultiClassification(conf=conf)
                elif 'single_regression' in m_name:
                    model = SingleRegression(conf=conf)
                elif 'vanilla_lstm' in m_name:
                    model = VanillaLSTM(conf=conf)
                    SMILES_mapping_json_file=model_directory+'SMILES_mapping.json'
                    X_train, _ = extract_SMILES_and_label(train_pd,
                                                feature_name='SMILES',
                                                label_name_list=labels,
                                                SMILES_mapping_json_file=SMILES_mapping_json_file)
                    X_val, _ = extract_SMILES_and_label(val_pd,
                                                        feature_name='SMILES',
                                                        label_name_list=labels,
                                                        SMILES_mapping_json_file=SMILES_mapping_json_file)
                    X_test, _ = extract_SMILES_and_label(test_pd,
                                                          feature_name='SMILES',
                                                          label_name_list=labels,
                                                          SMILES_mapping_json_file=SMILES_mapping_json_file)
            
                    X_train = sequence.pad_sequences(X_train, maxlen=model.padding_length)
                    X_val = sequence.pad_sequences(X_val, maxlen=model.padding_length)
                    X_test = sequence.pad_sequences(X_test, maxlen=model.padding_length)
                
                weight_file = m_name_dir+m_name+'.weight'
                
                if os.path.exists(weight_file):
                    model = model.setup_model()
                    model.load_weights(weight_file)
                    
                    y_pred_on_train[:,j] = model.predict(X_train)[:,-1]
                    y_pred_on_val[:,j] = model.predict(X_val)[:,-1]
                    y_pred_on_test[:,j] = model.predict(X_test)[:,-1]
                else:
                    y_pred_on_train[:,j] = np.nan
                    y_pred_on_val[:,j] = np.nan
                    y_pred_on_test[:,j] = np.nan
            else:
                y_pred_on_train[:,j] = np.nan
                y_pred_on_val[:,j] = np.nan
                y_pred_on_test[:,j] = np.nan
    
        #load model and predict
        model_list[m_name]  = (labels, y_train, y_val, y_test,
                               y_pred_on_train, y_pred_on_val, y_pred_on_test)
                                                   
    return model_list
    
"""
Loads docking model results as a list:
model_name -> fold_# -> labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test
"""
def get_docking_results_stage_1(model_directory, data_directory, k=5):
    model_list = {}
    if not os.path.exists(model_directory):
        return model_list
    
    labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
    #load data
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))
    output_file_list = [data_directory + f_ for f_ in file_list]
    
    pria_df = pd.read_csv(model_directory+'/lc123_rmi_all_docking_scores_complete.csv.gz')
    rmi_df = pd.read_csv(model_directory+'/lc123_pria_all_docking_scores_complete.csv.gz')
    pria_df = pria_df.rename(columns={'molid': 'Molecule'})
    rmi_df = rmi_df.rename(columns={'molid': 'Molecule'}) 
    
    model_names = pria_df.columns.values[1:]
    for m_name in model_names:
        model_list[m_name] = {}
        
    for i in range(k):         
        csv_file_list = output_file_list[:]
        test_pd = read_merged_data([csv_file_list[i]])
        csv_file_list.pop(i)
        val_pd = read_merged_data([csv_file_list[i%len(csv_file_list)]])
        csv_file_list.pop(i%len(csv_file_list))
        train_pd = read_merged_data(csv_file_list)
        
        # extract data, and split training data into training and val    
        train_pd = train_pd.merge(pria_df, how='inner', on='Molecule') 
        val_pd = val_pd.merge(pria_df, how='inner', on='Molecule') 
        test_pd = test_pd.merge(pria_df, how='inner', on='Molecule') 
        
        train_pd = train_pd.merge(rmi_df, how='inner', on='Molecule', suffixes=('_pria', '_rmi'))
        val_pd = val_pd.merge(rmi_df, how='inner', on='Molecule', suffixes=('_pria', '_rmi'))
        test_pd = test_pd.merge(rmi_df, how='inner', on='Molecule', suffixes=('_pria', '_rmi'))
        
        X_train, y_train = extract_feature_and_label(train_pd,
                                                     feature_name='Fingerprints',
                                                     label_name_list=labels)
        
        X_val, y_val = extract_feature_and_label(val_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=labels)
                                                   
        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=labels)
                                                   
        for m_name in model_names:
            y_pred_on_train = np.array(pd.concat((train_pd[m_name+'_pria'],train_pd[m_name+'_pria'],
                                                  train_pd[m_name+'_rmi']),axis=1))
            y_pred_on_val = np.array(pd.concat((val_pd[m_name+'_pria'],val_pd[m_name+'_pria'],
                                                val_pd[m_name+'_rmi']),axis=1))
            y_pred_on_test = np.array(pd.concat((test_pd[m_name+'_pria'],test_pd[m_name+'_pria'],
                                                 test_pd[m_name+'_rmi']),axis=1))
        
            #load model and predict
            model_list[m_name]['fold_'+str(i)]  = (labels, y_train, y_val, y_test,
                                                  y_pred_on_train, y_pred_on_val, y_pred_on_test)
    
    return model_list
       
def get_docking_results_stage_2(model_directory, data_directory, held_out_data_file, k=5):
    model_list = {}
    if not os.path.exists(model_directory):
        return model_list
    
    labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
    #load data
    file_list = []
    for i in range(k):
        file_list.append('file_{}.csv'.format(i))
    output_file_list = [data_directory + f_ for f_ in file_list]
    
    pria_df = pd.read_csv(model_directory+'/lc123_rmi_all_docking_scores_complete.csv.gz')
    rmi_df = pd.read_csv(model_directory+'/lc123_pria_all_docking_scores_complete.csv.gz')
    pria_df = pria_df.rename(columns={'molid': 'Molecule'})
    rmi_df = rmi_df.rename(columns={'molid': 'Molecule'}) 
    
    test_pria_df = pd.read_csv(model_directory+'/lc4_all_docking_scores.csv')
    test_rmi_df = pd.read_csv(model_directory+'/lc4_rmi_all_docking_scores.csv')
    test_pria_df = test_pria_df.rename(columns={'molid': 'Molecule'})
    test_rmi_df = test_rmi_df.rename(columns={'molid': 'Molecule'}) 
    
    model_names = pria_df.columns.values[1:]
    for m_name in model_names:
        model_list[m_name] = {}
        
    for i in range(1):         
        csv_file_list = output_file_list[:]
        val_pd = read_merged_data([csv_file_list[i%len(csv_file_list)]])
        csv_file_list.pop(i%len(csv_file_list))
        train_pd = read_merged_data(csv_file_list)
        
        test_pd = read_merged_data([held_out_data_file])    
        
        # extract data, and split training data into training and val    
        train_pd = train_pd.merge(pria_df, how='inner', on='Molecule') 
        val_pd = val_pd.merge(pria_df, how='inner', on='Molecule') 
        
        train_pd = train_pd.merge(rmi_df, how='inner', on='Molecule', suffixes=('_pria', '_rmi'))
        val_pd = val_pd.merge(rmi_df, how='inner', on='Molecule', suffixes=('_pria', '_rmi'))
        
        test_pd = test_pd.merge(test_pria_df, how='inner', on='Molecule') 
        test_pd = test_pd.merge(test_rmi_df, how='inner', on='Molecule', suffixes=('_pria', '_rmi'))
        
        X_train, y_train = extract_feature_and_label(train_pd,
                                                     feature_name='Fingerprints',
                                                     label_name_list=labels)
        
        X_val, y_val = extract_feature_and_label(val_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=labels)
                                                   
        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=labels)
                                                   
        for m_name in model_names:
            y_pred_on_train = np.array(pd.concat((train_pd[m_name+'_pria'],train_pd[m_name+'_pria'],
                                                  train_pd[m_name+'_rmi']),axis=1))
            y_pred_on_val = np.array(pd.concat((val_pd[m_name+'_pria'],val_pd[m_name+'_pria'],
                                                val_pd[m_name+'_rmi']),axis=1))
            y_pred_on_test = np.array(pd.concat((test_pd[m_name+'_pria'],test_pd[m_name+'_pria'],
                                                 test_pd[m_name+'_rmi']),axis=1))
        
            #load model and predict
            model_list[m_name] = (labels, y_train, y_val, y_test,
                                  y_pred_on_train, y_pred_on_val, y_pred_on_test)
    
    return model_list