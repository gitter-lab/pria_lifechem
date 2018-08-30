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

"""
 This class is similar to SKLearn_RandomForest but permutes the labels prior to training. 
 The seed for np.random.seed(seed=process_num) is set to process_num.
 
 Note: Unlike the other models, this model was trained and evaluated on the prospective data 
       AFTER the true prospective data was available. The goal is to see if scrambling the 
       the labels would result in worse performance for RF_h (best model on prospective data
       according to top250hits).       
"""
class RFH_YScrambled:
    def __init__(self, conf, process_num):
        self.conf = conf
        self.input_layer_dimension = 1024
        self.label_names = conf['label_names']
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        
        self.process_num = process_num
        
        self.param = conf['params']
        self.n_estimators = param['n_estimators']
        self.max_features = param['max_features']
        self.min_samples_leaf = param['min_samples_leaf']
        self.class_weight = param['class_weight']
        print('Testing set:', param)
        
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
        
        
    def train(self, X_train, y_train, model_file):
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
    
    # set seed according to process_num
    np.random.seed(seed=process_num)

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
                       
    if not os.path.exists(model_dir+'process_'+str(process_num)):
        os.makedirs(model_dir+'process_'+str(process_num))         
    model_file = model_dir+'process_'+str(process_num)+'/rf_clf'
    
    csv_file_list = output_file_list[:]
    train_pd = read_merged_data(csv_file_list)
    test_pd = pd.read_csv(prospective_file, compression='gzip')(prospective_file)
    
    labels = ["Keck_Pria_AS_Retest"]

    # extract data
    X_train, y_train = extract_feature_and_label(train_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=labels)
    X_test, y_test = extract_feature_and_label(test_pd,
                                               feature_name='Fingerprints',
                                               label_name_list=labels)
    
    # scramble y_train
    ind_permutation = np.random.permutation(y_train.shape[0])
    assert not np.array_equal(y_train, y_train[ind_permutation,:]) # check original labels and permuted labels not the same
    y_train = y_train[ind_permutation,:]
    print('done data preparation')
    
    task = RFH_YScrambled(conf=conf, process_num=process_num)
    task.train(X_train, y_train, model_file)
    task.save_model_params(config_csv_file)
                                       
    # save prediction arrays
    y_train, y_pred_on_train = self.get_prediction_info(X_train, y_train)
    y_test, y_pred_on_test = self.get_prediction_info(X_test, y_test)
    npz_file_name = model_dir+str(process_num)
    np.savez_compressed(npz_file_name,
                        labels=labels, y_train=y_train, y_val=None, y_test=y_test,
                        y_pred_on_train=y_pred_on_train, y_pred_on_val=None, y_pred_on_test=y_pred_on_test)