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
import shutil
import deepchem as dc
from sklearn.externals import joblib
from sklearn.grid_search import ParameterGrid
from deepchem.trans import undo_transforms

rnd_state=1337
np.random.seed(seed=rnd_state)

class Deepchem_IRV:
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
            self.K = param['K']
            print('param set:', param)            
            break
        
        self.model_dict = {}
        self.model = None
        
        self.nb_epochs=conf['fitting']['nb_epochs']
        return

    def setup_model(self, model_file):
        self.model = dc.models.TensorflowMultiTaskIRVClassifier(
                                                   len(self.label_names),
                                                   K=self.K,
                                                   learning_rate=0.001,
                                                   penalty=0.05,
                                                   batch_size=32,
                                                   fit_transformers=[],
                                                   logdir=model_file)        
        return
        
        
    def train_and_predict(self,
                          train_data,
                          val_data,
                          test_data,
                          model_file):
                              
        self.setup_model(model_file)
        self.model.fit(train_data, nb_epoch=self.nb_epochs)
        self.model.save()     
        
        self.predict_with_existing(train_data, 
                                   val_data, 
                                   test_data)
        return

    def predict_with_existing(self,
                              train_data,
                              val_data,
                              test_data):  
        
        y_pred_on_train = self.model.predict_proba(train_data)
        y_pred_on_val = self.model.predict_proba(val_data)
        y_pred_on_test = self.model.predict_proba(test_data)    
        
        y_train = train_data.y()
        y_val = val_data.y()
        y_test = test_data.y()
        w_train = train_data.w()
        w_val = val_data.w()
        w_test = test_data.w()
        
        y_train[:,2] = y_train[:,0]
        y_train[:,4] = y_train[:,3]
        y_val[:,2] = y_val[:,0]
        y_val[:,4] = y_val[:,3]
        y_test[:,2] = y_test[:,0]
        y_test[:,4] = y_test[:,3]
        
        w_train[:,2] = w_train[:,0]
        w_train[:,4] = w_train[:,3]
        w_val[:,2] = w_val[:,0]
        w_val[:,4] = w_val[:,3]
        w_test[:,2] = w_test[:,0]
        w_test[:,4] = w_test[:,3]
        
        for i, label in zip(range(len(self.label_names)), self.label_names):     
            y_train[np.where(w_train[:,i] == 0)[0],i] = -1
            y_val[np.where(w_val[:,i] == 0)[0],i] = -1
            y_test[np.where(w_test[:,i] == 0)[0],i] = -1
         
        y_train = np.insert(y_train, 3, y_train[:,1], axis=1)
        y_val = np.insert(y_val, 3, y_val[:,1], axis=1)
        y_test = np.insert(y_test, 3, y_test[:,1], axis=1)
        
        y_pred_on_train = np.insert(y_pred_on_train, 3, y_pred_on_train[:,2], axis=1)
        y_pred_on_val = np.insert(y_pred_on_val, 3, y_pred_on_val[:,2], axis=1)
        y_pred_on_test = np.insert(y_pred_on_test, 3, y_pred_on_test[:,2], axis=1)
        
        print
        print('train precision: {}'.format(precision_auc_multi(y_train, y_pred_on_train, range(y_train.shape[1]), np.mean)))
        print('train roc: {}'.format(roc_auc_multi(y_train, y_pred_on_train, range(y_train.shape[1]), np.mean)))
        print('train bedroc: {}'.format(bedroc_auc_multi(y_train, y_pred_on_train, range(y_train.shape[1]), np.mean)))
        print
        print('val precision: {}'.format(precision_auc_multi(y_val, y_pred_on_val, range(y_val.shape[1]), np.mean)))
        print('val roc: {}'.format(roc_auc_multi(y_val, y_pred_on_val, range(y_val.shape[1]), np.mean)))
        print('val bedroc: {}'.format(bedroc_auc_multi(y_val, y_pred_on_val, range(y_val.shape[1]), np.mean)))
        print
        print('test precision: {}'.format(precision_auc_multi(y_test, y_pred_on_test, range(y_test.shape[1]), np.mean)))
        print('test roc: {}'.format(roc_auc_multi(y_test, y_pred_on_test, range(y_test.shape[1]), np.mean)))
        print('test bedroc: {}'.format(bedroc_auc_multi(y_test, y_pred_on_test, range(y_test.shape[1]), np.mean)))
        print
        
        label_list = ['Keck_Pria_AS_Retest', 'Keck_Pria_FP_data', 
                      'Keck_Pria_Continuous_AS_Retest', 'Keck_Pria_Continuous_FP_data',
                      'Keck_RMI_cdd', 'FP counts % inhibition']
        nef_auc_mean = np.mean(np.array(nef_auc(y_train, y_pred_on_train, self.EF_ratio_list, label_list))) 
        print('train nef auc: {}'.format(nef_auc_mean))
        nef_auc_mean = np.mean(np.array(nef_auc(y_val, y_pred_on_val, self.EF_ratio_list, label_list))) 
        print('val nef auc: {}'.format(nef_auc_mean))
        nef_auc_mean = np.mean(np.array(nef_auc(y_test, y_pred_on_test, self.EF_ratio_list, label_list))) 
        print('test nef auc: {}'.format(nef_auc_mean))
        return

    def save_model_evaluation_metrics(self,
                              data,
                              model_file,
                              metric_dir,
                              label_names=None):
        
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)   
        
        y_pred = self.model.predict_proba(data)
        
        y_true = data.y()
        w_true = data.w()
        
        y_true[:,2] = y_true[:,0]
        y_true[:,4] = y_true[:,3]
        w_true[:,2] = w_true[:,0]
        w_true[:,4] = w_true[:,3]
        
        for i, label in zip(range(len(self.label_names)), self.label_names): 
            y_true[np.where(w_true[:,i] == 0)[0],i] = -1
        
        y_true = np.insert(y_true, 3, y_true[:,1], axis=1)
        y_pred = np.insert(y_pred, 3, y_pred[:,2], axis=1)
        
        label_list = ['Keck_Pria_AS_Retest', 'Keck_Pria_FP_data', 
                      'Keck_Pria_Continuous_AS_Retest', 'Keck_Pria_Continuous_FP_data',
                      'Keck_RMI_cdd', 'FP counts % inhibition']
        evaluate_model(y_true, y_pred, metric_dir, label_list)        
        return
        
    def save_model_params(self, config_csv_file):      
        data = str(self.param)
        with open(config_csv_file, 'w') as csvfile:
            csvfile.write(data)
        return

    def getK():
        return self.K;

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
    model_file = model_dir+'tf_checkpoints/'
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
    
    for i in range(k):    
        csv_file_list = output_file_list[:]
        test_pd = read_merged_data([csv_file_list[i]])
        csv_file_list.pop(i)
        val_pd = read_merged_data([csv_file_list[i%len(csv_file_list)]])
        csv_file_list.pop(i%len(csv_file_list))
        train_pd = read_merged_data(csv_file_list)
        
        labels = ['Keck_Pria_AS_Retest', 'Keck_Pria_FP_data', 'Keck_Pria_Continuous',
                  'Keck_RMI_cdd', 'FP counts % inhibition']
    
        featurizer='ECFP'
        if featurizer == 'ECFP':
            featurizer_func = dc.feat.CircularFingerprint(size=1024)
        elif featurizer == 'GraphConv':
            featurizer_func = dc.feat.ConvMolFeaturizer()
            
        loader = dc.data.CSVLoader(tasks=labels, 
                                   smiles_field="SMILES", 
                                   featurizer=featurizer_func)
        
        # extract data, and split training data into training and val
        train_data = loader.featurize(train_files, shard_size=2**15)
        val_data = loader.featurize(val_files, shard_size=2**15)
        test_data = loader.featurize(test_files, shard_size=2**15)
        
        train_data = transformer.transform(dc.trans.BalancingTransformer(transform_w=True, dataset=train_data))
        val_data = transformer.transform(dc.trans.BalancingTransformer(transform_w=True, dataset=val_data))
        test_data = transformer.transform(dc.trans.BalancingTransformer(transform_w=True, dataset=test_data))
           
        
        with open(config_json_file, 'r') as f:
            conf = json.load(f)
        task = Deepchem_IRV(conf=conf)
        K_neighbors = task.getK()
        transformers = [dc.trans.IRVTransformer(K_neighbors, len(labels), train_data)]
        for transformer in transformers:
            train_data = transformer.transform(train_data)
            val_data = transformer.transform(val_data)
            test_data = transformer.transform(test_data)        
        
        #train model
        task.train_and_predict(train_data, val_data, test_data, model_file)
        
        #Undo transfromations and get metrics
        train_data = undo_transforms(train_data, [dc.trans.BalancingTransformer(transform_w=True, dataset=train_data),
                                                  dc.trans.IRVTransformer(K_neighbors, len(labels), train_data)])
        val_data = undo_transforms(val_data, [dc.trans.BalancingTransformer(transform_w=True, dataset=val_data),
                                                  dc.trans.IRVTransformer(K_neighbors, len(labels), train_data)])
        test_data = undo_transforms(test_data, [dc.trans.BalancingTransformer(transform_w=True, dataset=test_data),
                                                  dc.trans.IRVTransformer(K_neighbors, len(labels), train_data)])
                                                       
        task.save_model_evaluation_metrics(train_data, model_file,
                                          model_dir+'fold_'+str(i)+'/train_metrics/',
                                          label_names=labels)
        task.save_model_evaluation_metrics(val_data, model_file,
                                          model_dir+'fold_'+str(i)+'/val_metrics/',
                                          label_names=labels)
        task.save_model_evaluation_metrics(test_data, model_file,
                                          model_dir+'fold_'+str(i)+'/test_metrics/',
                                          label_names=labels)
        
        task.save_model_params(config_csv_file)