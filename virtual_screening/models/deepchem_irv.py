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
from shutil import move
import copy

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
        
        self.nb_epochs = conf['fitting']['nb_epochs']
        self.batch_size = conf['fitting']['batch_size']
        self.learning_rate = conf['fitting']['learning_rate']
        self.penalty = conf['fitting']['penalty']
        
        self.early_stopping_patience = conf['fitting']['early_stopping']['patience']
        self.early_stopping_option = conf['fitting']['early_stopping']['option']
        return
                
    @property    
    def K(self):
        return self.K
        
    def get_prediction_info(self, data):
        y_pred = self.model.predict_proba(data)[:,:,1]
        
        y_true = copy.deepcopy(data.y)
        w_true = copy.deepcopy(data.w)
        
        y_true[:,2] = y_true[:,0]
        y_true[:,4] = y_true[:,3]
        w_true[:,2] = w_true[:,0]
        w_true[:,4] = w_true[:,3]
        
        for i, label in zip(range(len(self.label_names)), self.label_names): 
            y_true[np.where(w_true[:,i] == 0)[0],i] = -1
        
        y_true = np.insert(y_true, 3, y_true[:,1], axis=1)
        y_pred = np.insert(y_pred, 3, y_pred[:,2], axis=1)
        
        return y_true, y_pred
        
    def setup_model(self, logdir):
        self.model = dc.models.TensorflowMultiTaskIRVClassifier(
                                                   len(self.label_names),
                                                   K=self.K,
                                                   learning_rate=self.learning_rate,
                                                   penalty=self.penalty,
                                                   batch_size=self.batch_size,
                                                   fit_transformers=[],
                                                   logdir=logdir,
                                                   verbose=False)       
        return
        
        
    def train_and_predict(self,
                          train_data,
                          val_data,
                          test_data,
                          logdir):
        
        # DC saves tf models using tf.train.Saver, saving multiple checkpoints.
        # It always saves a checkpoint after training. 
        # We are training for 1 epoch, so model location is logdir+'model.ckpt-1'.        
        curr_ckpt_file = logdir+'model.ckpt-1'
        best_ckpt_file = logdir+'best_model.ckpt'
        
        self.setup_model(logdir)
        
        curr_pr = 0
        best_pr = 0
        counter = 0
        best_epoch = 0        
        for i in range(self.nb_epochs):            
            self.model.fit(train_data, nb_epoch=1)
            
            y_val, y_pred_on_val = self.get_prediction_info(val_data)
            curr_pr = precision_auc_multi(y_val, y_pred_on_val, 
                                      range(y_val.shape[1]), np.mean)
            if curr_pr < best_pr:
                if counter >= patience:
                    break
                counter += 1
            else:
                counter = 0
                best_pr = curr_pr
                #copy model file with different name to keep track of best model
                move(curr_ckpt_file, best_ckpt_file)

            
            if i%5 == 0:
                y_train, y_pred_on_train = self.get_prediction_info(train_data)
                curr_roc = roc_auc_multi(y_val, y_pred_on_val, 
                                      range(y_val.shape[1]), np.mean)
                train_roc = roc_auc_multi(y_train, y_pred_on_train, 
                                      range(y_train.shape[1]), np.mean)
                train_pr = precision_auc_multi(y_train, y_pred_on_train, 
                                      range(y_train.shape[1]), np.mean)

                print('Epoch {}/{}'.format(i + 1, self.nb_epoch))
                print 'Train\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
                        (train_roc, train_pr)
                print 'Val\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
                        (curr_roc, curr_pr)  
            
        print('\n\n Training Done: \n\n')
        #copy back best_model_ckpt so deepchem gets best model
        move(best_ckpt_file, curr_ckpt_file)
        self.predict_with_existing(train_data, 
                                   val_data, 
                                   test_data)
        return

    def predict_with_existing(self,
                              train_data,
                              val_data,
                              test_data):  
        
        y_train, y_pred_on_train = self.get_prediction_info(train_data)
        y_val, y_pred_on_val = self.get_prediction_info(val_data)
        y_test, y_pred_on_test = self.get_prediction_info(test_data)
        
        print
        print('train precision: {}'.format(precision_auc_multi(y_train, y_pred_on_train, range(y_train.shape[1]), np.mean)))
        print('train roc: {}'.format(roc_auc_multi(y_train, y_pred_on_train, range(y_train.shape[1]), np.mean)))
        print
        print('val precision: {}'.format(precision_auc_multi(y_val, y_pred_on_val, range(y_val.shape[1]), np.mean)))
        print('val roc: {}'.format(roc_auc_multi(y_val, y_pred_on_val, range(y_val.shape[1]), np.mean)))
        print
        print('test precision: {}'.format(precision_auc_multi(y_test, y_pred_on_test, range(y_test.shape[1]), np.mean)))
        print('test roc: {}'.format(roc_auc_multi(y_test, y_pred_on_test, range(y_test.shape[1]), np.mean)))
        print
        
        nef_auc_mean = np.mean(np.array(nef_auc(y_train, y_pred_on_train, self.EF_ratio_list, self.label_names))) 
        print('train nef auc: {}'.format(nef_auc_mean))
        nef_auc_mean = np.mean(np.array(nef_auc(y_val, y_pred_on_val, self.EF_ratio_list, self.label_names))) 
        print('val nef auc: {}'.format(nef_auc_mean))
        nef_auc_mean = np.mean(np.array(nef_auc(y_test, y_pred_on_test, self.EF_ratio_list, self.label_names))) 
        print('test nef auc: {}'.format(nef_auc_mean))
        return

    def save_model_evaluation_metrics(self,
                              data,
                              logdir,
                              metric_dir,
                              label_names=None):
        
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)   
        
        y_true, y_pred = self.get_prediction_info(data)
        
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
    if not os.path.exists(model_file):
        os.makedirs(model_file) 
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
        test_files = [csv_file_list[i]]
        csv_file_list.pop(i)
        val_files = [csv_file_list[i%len(csv_file_list)]]
        csv_file_list.pop(i%len(csv_file_list))
        train_files = csv_file_list
        
        labels = ['Keck_Pria_AS_Retest', 'Keck_Pria_FP_data', 'Keck_Pria_Continuous',
                  'Keck_RMI_cdd', 'FP counts % inhibition']
        #        labels = ['Keck_Pria_AS_Retest', 'Keck_Pria_FP_data', 'Keck_Pria_Continuous',
        #                  'Keck_RMI_cdd', 'FP counts % inhibition']
        
        featurizer='ECFP'
        if featurizer == 'ECFP':
            featurizer_func = dc.feat.CircularFingerprint(size=1024)
        elif featurizer == 'GraphConv':
            featurizer_func = dc.feat.ConvMolFeaturizer()
            
        loader = dc.data.CSVLoader(tasks=labels, 
                                   smiles_field="SMILES", 
                                   featurizer=featurizer_func,
                                   verbose=False)

        # extract data, and split training data into training and val
        orig_train_data = loader.featurize(train_files, shard_size=2**15)
        orig_val_data = loader.featurize(val_files, shard_size=2**15)
        orig_test_data = loader.featurize(test_files, shard_size=2**15)
        
        train_data = loader.featurize(train_files, shard_size=2**15)
        val_data = loader.featurize(val_files, shard_size=2**15)
        test_data = loader.featurize(test_files, shard_size=2**15)
        
        orig_train_data = dc.data.NumpyDataset(orig_train_data.X, orig_train_data.y, orig_train_data.w, orig_train_data.ids)
        orig_val_data = dc.data.NumpyDataset(orig_val_data.X, orig_val_data.y, orig_val_data.w, orig_val_data.ids)
        orig_test_data = dc.data.NumpyDataset(orig_test_data.X, orig_test_data.y, orig_test_data.w, orig_test_data.ids)

        train_data = dc.data.NumpyDataset(train_data.X, train_data.y, train_data.w, train_data.ids)
        val_data = dc.data.NumpyDataset(val_data.X, val_data.y, val_data.w, val_data.ids)
        test_data = dc.data.NumpyDataset(test_data.X, test_data.y, test_data.w, test_data.ids)
        
        train_data = dc.trans.BalancingTransformer(transform_w=True, dataset=train_data).transform(train_data)
        val_data = dc.trans.BalancingTransformer(transform_w=True, dataset=val_data).transform(val_data)
        test_data = dc.trans.BalancingTransformer(transform_w=True, dataset=test_data).transform(test_data)
           
        
        with open(config_json_file, 'r') as f:
            conf = json.load(f)
        task = Deepchem_IRV(conf=conf)
        K_neighbors = task.K
        transformers = [dc.trans.IRVTransformer(K_neighbors, len(labels), train_data)]
        for transformer in transformers:
            train_data = transformer.transform(train_data)
            val_data = transformer.transform(val_data)
            test_data = transformer.transform(test_data)        
        
        #train model
        task.train_and_predict(train_data, val_data, test_data, model_file)
        
        #Undo transfromations and get metrics
        transformers = [dc.trans.IRVTransformer(K_neighbors, len(labels), train_data)]
        for transformer in transformers:
            train_data = transformer.transform(orig_train_data)
            val_data = transformer.transform(orig_val_data)
            test_data = transformer.transform(orig_test_data)       
        
        task.predict_with_existing(train_data, val_data, test_data)                                         
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