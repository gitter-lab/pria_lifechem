import argparse
import pandas as pd
import csv
import numpy as np
import json
import sys
from virtual_screening.function import *
from virtual_screening.evaluation import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.grid_search import ParameterGrid

rnd_state=1337

class SKLearn_RandomForest:
    def __init__(self, conf):
        self.conf = conf
        self.input_layer_dimension = 1024
        self.label_names = conf['label_names']
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        
        self.process_id = int(os.environ.get('process'))        
        if self.process_id == None:
            print('Error: No environemnt variable process exists.')
            return 
        else:
            print('process id:', self.process_id)
        
        cnt = 0
        for param in ParameterGrid(conf['params']):
            if cnt != self.process_id:
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
        
        return

    def setup_model(self):
        model = RandomForestClassifier(n_estimators=self.n_estimators, 
                                       max_features=self.max_features, 
                                       min_samples_leaf=self.min_samples_leaf, 
                                       n_jobs=-1, 
                                       class_weight=self.class_weight,
                                       random_state=rnd_state,
                                       oob_score=False, 
                                       verbose=1) 
        return model

    def train_and_predict(self,
                          X_train, y_train,
                          X_val, y_val,
                          X_test, y_test,
                          model_file):
                              
        model = self.setup_model()
        model.fit(np.concatenate((X_train, X_val)), np.concatenate((y_train, y_val)))
        joblib.dump(model, model_file)
        return

    def predict_with_existing(self,
                              X_train, y_train,
                              X_val, y_val,
                              X_test, y_test,
                              model_file):
                                  
        model = joblib.load(model_file)
        
        X_train = np.concatenate((X_train, X_val))
        y_train = np.concatenate((y_train, y_val))
        y_pred_on_train = model.predict_proba(X_train)[:,:,1].T
        y_pred_on_test = model.predict_proba(X_test)[:,:,1].T

        print
        print('train precision: {}'.format(precision_auc_multi(y_train, y_pred_on_train)))
        print('train roc: {}'.format(roc_auc_multi(y_train, y_pred_on_train)))
        print('train bedroc: {}'.format(bedroc_auc_multi(y_train, y_pred_on_train)))
        print
        print('test precision: {}'.format(precision_auc_multi(y_test, y_pred_on_test)))
        print('test roc: {}'.format(roc_auc_multi(y_test, y_pred_on_test)))
        print('test bedroc: {}'.format(bedroc_auc_multi(y_test, y_pred_on_test)))
        print
        
        nef_auc_mean = np.mean(np.array(nef_auc(y_train, y_pred_on_train, self.EF_ratio_list, self.label_names))) 
        print('train nef auc: {}'.format(nef_auc_mean))
        nef_auc_mean = np.mean(np.array(nef_auc(y_test, y_pred_on_test, self.EF_ratio_list, self.label_names))) 
        print('test nef auc: {}'.format(nef_auc_mean))
        return

    def save_model_evaluation_metrics(self,
                              X, y_true,
                              model_file,
                              metric_dir,
                              label_names=None):
        
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)   
            
        model = joblib.load(model_file)
        y_pred = model.predict_proba(X)[:,:,1].T
        
        evaluate_model(y_true, y_pred, metric_dir, self.label_names)        
        return
        
    def save_model_info(self, config_csv_file):
        
        if not os.path.exists(config_csv_file):
            os.makedirs(config_csv_file)   
       
        
        data = str(self.param)
        with open(config_csv_file, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
            for d in data:
                writer.writerow([d])
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_json_file', action="store", dest="config_json_file", required=True)
    parser.add_argument('--model_dir', action="store", dest="model_dir", required=True)
    parser.add_argument('--dataset_dir', action="store", dest="dataset_dir", required=True)
    #####
    given_args = parser.parse_args()
    config_json_file = given_args.config_json_file
    model_dir = given_args.model_dir
    dataset_dir = given_args.dataset_dir
    #####
    model_file = model_dir+'rf_clf.pkl'
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
    
    train_pd = read_merged_data(output_file_list[0:2])
    test_pd = read_merged_data([output_file_list[4]])
    val_pd = read_merged_data([output_file_list[3]])
    
    labels = ['Keck_Pria_AS_Retest', 'Keck_Pria_FP_data', 'Keck_Pria_Continuous',
              'Keck_RMI_cdd', 'FP counts % inhibition']

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
    print('done data preparation')

    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    task = SKLearn_RandomForest(conf=conf)
    task.train_and_predict(X_train, y_train, X_val, y_val, X_test, y_test, model_file)
    task.save_model_params(config_csv_file)
    
    #####
    task.save_model_evaluation_metrics(np.concatenate((X_train, X_val)), 
                                       np.concatenate((y_train, y_val)), model_file,
                                      model_dir+'train_metrics/',
                                      label_names=labels)
    task.save_model_evaluation_metrics(X_test, y_test, model_file,
                                      model_dir+'test_metrics/',
                                      label_names=labels)