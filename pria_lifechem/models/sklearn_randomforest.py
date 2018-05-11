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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.externals import joblib
from sklearn.grid_search import ParameterGrid
from shutil import move

rnd_state=1337
np.random.seed(seed=rnd_state)

class SKLearn_RandomForest:
    def __init__(self, conf, process_num, stage):
        self.conf = conf
        self.input_layer_dimension = 1024
        self.label_names = conf['label_names']
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        
        self.process_num = process_num
        self.stage =  stage
        
        if self.stage == 0:
            self.label_names = [self.label_names[0]]
        
        cnt = 0
        for param in ParameterGrid(conf['params']):
            if cnt != self.process_num:
                cnt += 1
                continue
            
            self.param = param
            self.n_estimators = param['n_estimators']
            self.max_features = param['max_features']
            self.min_samples_leaf = param['min_samples_leaf']
            self.class_weight = param['class_weight']
            print('Testing set:', param)            
            break
        
        if self.max_features == "None":
            self.max_features = None
        if self.class_weight == "None":
            self.class_weight = None
        
        self.model_dict = {}
        return
        
    def get_prediction_info(self, X, y_true):
        y_pred = np.zeros(shape=y_true.shape)        
        
        for i, label in zip(range(len(self.label_names)), self.label_names):     
            model = self.model_dict[label]
            
            y_true[np.where(np.isnan(y_true[:,i]))[0],i] = -1
            if i in [0,1,2]:                
                y_pred[:,i] =  model.predict_proba(X)[:,1]
        
        return y_true, y_pred
        
    def setup_model(self):
        for i in range(len(self.label_names)):
            self.model_dict[self.label_names[i]] = RandomForestClassifier(n_estimators=self.n_estimators, 
                                           max_features=self.max_features, 
                                           min_samples_leaf=self.min_samples_leaf, 
                                           n_jobs=3, 
                                           class_weight=self.class_weight,
                                           random_state=rnd_state,
                                           oob_score=False, 
                                           verbose=1) 
        return
        
        
    def train(self, X_train, y_train, model_file):
                              
        self.setup_model()
        
        p = np.random.permutation(len(X_train))
        X_train = X_train[p,:]
        y_train = y_train[p,:]
        
        for i, label in zip(range(len(self.label_names)), self.label_names):
            y = y_train[:,i]
            indexes = np.where(np.isnan(y))[0]
                
            y = np.delete(y, indexes, axis=0)
            X = np.delete(X_train, indexes, axis=0)
            self.model_dict[label].fit(X, y)
            
            joblib.dump(self.model_dict[label], model_file+'_'+label+'.pkl', compress = 1)
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

    def save_model_evaluation_metrics(self,
                              X, y_true,
                              model_file,
                              metric_dir,
                              label_names=None):
        
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)   
        
        y_true, y_pred = self.get_prediction_info(X, y_true)
        label_list = self.label_names
        evaluate_model(y_true, y_pred, metric_dir, label_list)        
        return
        
    def save_model_params(self, config_csv_file):      
        data = str(self.param)
        with open(config_csv_file, 'w') as csvfile:
            csvfile.write(data)
        return
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', action="store", dest="config_json_file", required=True)
    parser.add_argument('--model_dir', action="store", dest="model_dir", required=True)
    parser.add_argument('--dataset_dir', action="store", dest="dataset_dir", required=True)
    parser.add_argument('--process_num', action="store", dest="process_num", required=True)
    parser.add_argument('--stage', action="store", dest="stage", required=True)
    #####
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    model_dir = given_args.model_dir
    dataset_dir = given_args.dataset_dir
    process_num = int(given_args.process_num)
    stage = int(given_args.stage)
    #####
    config_csv_file = model_dir+'model_config.csv'
    #####
    scratch_dir = os.environ.get('_CONDOR_JOB_IWD')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    
    # specify dataset
    directory = dataset_dir
    file_list = []
    k=5
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
    
    with open(config_json_file, 'r') as f:
            conf = json.load(f)
    
    if stage == 0:
        i = 0
        if not os.path.exists(model_dir+'fold_'+str(i)):
            os.makedirs(model_dir+'fold_'+str(i))         
        model_file = model_dir+'fold_'+str(i)+'/rf_clf'

        csv_file_list = output_file_list[:]
        test_pd = read_merged_data([csv_file_list[i]])
        csv_file_list.pop(i)
        train_pd = read_merged_data(csv_file_list[:3])

        labels = ["Keck_Pria_AS_Retest"]

        # extract data, and split training data into training and val
        X_train, y_train = extract_feature_and_label(train_pd,
                                                     feature_name='Fingerprints',
                                                     label_name_list=labels)

        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=labels)
        print('done data preparation')

        task = SKLearn_RandomForest(conf=conf, process_num=process_num, stage=stage)
        task.train(X_train, y_train, model_file)
        task.save_model_params(config_csv_file)

        #####
        task.save_model_evaluation_metrics(X_train, y_train, model_file,
                                      model_dir+'fold_'+str(i)+'/train_metrics/',
                                      label_names=labels)

        task.save_model_evaluation_metrics(X_test, y_test, model_file,
                                          model_dir+'fold_'+str(i)+'/test_metrics/',
                                          label_names=labels)
    elif stage == 1:
        for i in range(k):  
            if not os.path.exists(model_dir+'fold_'+str(i)):
                os.makedirs(model_dir+'fold_'+str(i))         
            model_file = model_dir+'fold_'+str(i)+'/rf_clf'
            
            csv_file_list = output_file_list[:]
            test_pd = read_merged_data([csv_file_list[i]])
            csv_file_list.pop(i)
            train_pd = read_merged_data(csv_file_list)
            
            labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
        
            # extract data, and split training data into training and val
            X_train, y_train = extract_feature_and_label(train_pd,
                                                         feature_name='Fingerprints',
                                                         label_name_list=labels)
                                                       
            X_test, y_test = extract_feature_and_label(test_pd,
                                                       feature_name='Fingerprints',
                                                       label_name_list=labels)
            print('done data preparation')
            
            task = SKLearn_RandomForest(conf=conf, process_num=process_num, stage=stage)
            task.train(X_train, y_train,  model_file)
            task.save_model_params(config_csv_file)
            
            #####
            task.save_model_evaluation_metrics(X_train, y_train, model_file,
                                              model_dir+'fold_'+str(i)+'/train_metrics/',
                                              label_names=labels)
            
            task.save_model_evaluation_metrics(X_test, y_test, model_file,
                                              model_dir+'fold_'+str(i)+'/test_metrics/',
                                              label_names=labels)
                                              
    elif stage == 2:
        for i in range(0):  
            if not os.path.exists(model_dir+'fold_'+str(i)):
                os.makedirs(model_dir+'fold_'+str(i))         
            model_file = model_dir+'fold_'+str(i)+'/rf_clf'
            
            csv_file_list = output_file_list[:]
            train_pd = read_merged_data(csv_file_list)
            
            labels = ["Keck_Pria_AS_Retest", "Keck_Pria_FP_data", "Keck_RMI_cdd"]
        
            # extract data, and split training data into training and val
            X_train, y_train = extract_feature_and_label(train_pd,
                                                         feature_name='Fingerprints',
                                                         label_name_list=labels)
            print('done data preparation')
            
            task = SKLearn_RandomForest(conf=conf, process_num=process_num, stage=stage)
            task.train(X_train, y_train, model_file)
            task.save_model_params(config_csv_file)
            
            task.save_model_evaluation_metrics(X_train, 
                                               y_train, model_file,
                                              model_dir+'fold_'+str(i)+'/train_metrics/',
                                              label_names=labels)
