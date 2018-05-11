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
from shutil import move, copy2, rmtree
import copy

rnd_state=1337
np.random.seed(seed=rnd_state)

class Deepchem_IRV:
    def __init__(self, conf, process_num, stage):
        self.conf = conf
        self.input_layer_dimension = 1024
        self.EF_ratio_list = conf['enrichment_factor']['ratio_list']
        
        self.process_num = process_num        
        
        self.stage = stage
        param_name = 'cross_validation'
        if self.stage == 2:
            param_name = 'prospective_screening'
        
        cnt = 0
        for param in ParameterGrid(conf[param_name]):
            if cnt != self.process_num:
                cnt += 1
                continue
            
            self.param = param
            self.K = param['K']            
            self.fold_num = param['fold_num']
            self.label_names = [str(param['label_names'])]
            print('param set:', param)            
            break
        
        self.model_dict = {}
        
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
        
    @property    
    def labels(self):
        return self.label_names
        
    @property    
    def fold_num(self):
        return self.fold_num
        
    def get_prediction_info(self, data):
        y_true = np.zeros(shape=(data[0].y.shape[0], len(self.label_names)))
        y_pred = np.zeros(shape=(data[0].y.shape[0], len(self.label_names)))
        w_true = np.zeros(shape=(data[0].y.shape[0], len(self.label_names)))
        
        for i in range(len(self.label_names)):
            try:
                y_pred[:,i] = self.model_dict[self.label_names[i]].predict_proba(data[i])[:,:,1].ravel()
            except:
                print('Error on predicting label ' + self.label_names[i])
                    
            
            y_true[:,i] = copy.deepcopy(data[i].y[:,0])
            w_true[:,i] = copy.deepcopy(data[i].w[:,0])    
            
        for i, label in zip(range(len(self.label_names)), self.label_names): 
            y_true[np.where(w_true[:,i] == 0)[0],i] = -1
                
        return y_true, y_pred
        
    def setup_model(self, logdir):
        for label in self.label_names:
            self.model_dict[label] = dc.models.TensorflowMultiTaskIRVClassifier(
                                                   1,
                                                   K=self.K,
                                                   learning_rate=self.learning_rate,
                                                   penalty=self.penalty,
                                                   batch_size=self.batch_size,
                                                   fit_transformers=[],
                                                   logdir=logdir+'/'+label+'/',
                                                   verbose=False)       
        return
        
        
    def train_and_predict(self,
                          train_data,
                          val_data,
                          test_data,
                          logdir):
  
        self.setup_model(logdir)
        for label_index in range(len(self.label_names)):
            # DC saves tf models using tf.train.Saver, saving multiple checkpoints.
            # It always saves a checkpoint after training. 
            # We are training for 1 epoch, so model location is logdir+'model.ckpt-1'.        
            curr_ckpt_file = logdir+'/'+self.label_names[label_index]+'/model.ckpt-'
            best_ckpt_file = logdir+'/'+self.label_names[label_index]+'/best.ckpt'
            ckpt_count = '2'
            
            curr_pr = 0
            best_pr = 0
            best_epoch = 0        
            iters_per_epoch = 10
            threshold_epoch = 20
            i=0
            while i < self.nb_epochs:      
                if i < threshold_epoch:
                    self.model_dict[self.label_names[label_index]].fit(train_data[label_index], nb_epoch=1)
                    i = i+1
                else:
                    self.model_dict[self.label_names[label_index]].fit(train_data[label_index], nb_epoch=iters_per_epoch)
                    i = i+iters_per_epoch
                
                y_val, y_pred_on_val = self.get_prediction_info(val_data)
                y_val = y_val[:,label_index:label_index+1]
                y_pred_on_val = y_pred_on_val[:,label_index:label_index+1]
                
                curr_pr = precision_auc_multi(y_val, y_pred_on_val, 
                                              [0], np.mean)
                if curr_pr > best_pr:
                    best_pr = curr_pr
                    #copy model file with different name to keep track of best model                        
                    copy2(self.model_dict[self.label_names[label_index]]._find_last_checkpoint()+'.data-00000-of-00001', best_ckpt_file+'.data-00000-of-00001')
                    copy2(self.model_dict[self.label_names[label_index]]._find_last_checkpoint()+'.index', best_ckpt_file+'.index')
                    copy2(self.model_dict[self.label_names[label_index]]._find_last_checkpoint()+'.meta', best_ckpt_file+'.meta')
    
                
                if i%30 == 0:
                    y_train, y_pred_on_train = self.get_prediction_info(train_data)
                    y_train = y_train[:,label_index:label_index+1]
                    y_pred_on_train = y_pred_on_train[:,label_index:label_index+1]
                
                    curr_roc = roc_auc_multi(y_val, y_pred_on_val, 
                                          range(y_val.shape[1]), np.mean)
                    train_roc = roc_auc_multi(y_train, y_pred_on_train, 
                                          range(y_train.shape[1]), np.mean)
                    train_pr = precision_auc_multi(y_train, y_pred_on_train, 
                                          range(y_train.shape[1]), np.mean)
    
                    print('Epoch {}/{}'.format(i + 1, self.nb_epochs))
                    print 'Train\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
                            (train_roc, train_pr)
                    print 'Val\tAUC[ROC]: %.6f\tAUC[PR]: %.6f' % \
                            (curr_roc, curr_pr)  
                
            print('\n\n Training Done: \n\n')
            #copy back best_model_ckpt so deepchem gets best model
            copy2(best_ckpt_file+'.data-00000-of-00001', self.model_dict[self.label_names[label_index]]._find_last_checkpoint()+'.data-00000-of-00001')
            copy2(best_ckpt_file+'.index', self.model_dict[self.label_names[label_index]]._find_last_checkpoint()+'.index')
            copy2(best_ckpt_file+'.meta', self.model_dict[self.label_names[label_index]]._find_last_checkpoint()+'.meta')
            
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
        if self.stage == 1:
            y_test, y_pred_on_test = self.get_prediction_info(test_data)
        
        print
        print('train precision: {}'.format(precision_auc_multi(y_train, y_pred_on_train, range(y_train.shape[1]), np.mean)))
        print('train roc: {}'.format(roc_auc_multi(y_train, y_pred_on_train, range(y_train.shape[1]), np.mean)))
        print
        print('val precision: {}'.format(precision_auc_multi(y_val, y_pred_on_val, range(y_val.shape[1]), np.mean)))
        print('val roc: {}'.format(roc_auc_multi(y_val, y_pred_on_val, range(y_val.shape[1]), np.mean)))
        print
        if self.stage == 1:
            print('test precision: {}'.format(precision_auc_multi(y_test, y_pred_on_test, range(y_test.shape[1]), np.mean)))
            print('test roc: {}'.format(roc_auc_multi(y_test, y_pred_on_test, range(y_test.shape[1]), np.mean)))
            print

        label_list = self.label_names     
        nef_auc_mean = np.mean(np.array(nef_auc(y_train, y_pred_on_train, self.EF_ratio_list, label_list))) 
        print('train nef auc: {}'.format(nef_auc_mean))
        nef_auc_mean = np.mean(np.array(nef_auc(y_val, y_pred_on_val, self.EF_ratio_list, label_list))) 
        print('val nef auc: {}'.format(nef_auc_mean))
        if self.stage == 1:
            nef_auc_mean = np.mean(np.array(nef_auc(y_test, y_pred_on_test, self.EF_ratio_list, label_list))) 
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
    
    with open(config_json_file, 'r') as f:
        conf = json.load(f)
    
    if stage == 1:                               
        task = Deepchem_IRV(conf=conf, process_num=process_num, stage=stage)
        K_neighbors = task.K
        labels = task.labels
        
        ti = task.fold_num
        i = ti
        vi = (ti+1)%k
        
        csv_file_list = output_file_list[:]
        test_files = [csv_file_list[ti]]
        val_files = [csv_file_list[vi]]
        
        train_files = [q for j, q in enumerate(csv_file_list) if j not in [ti, vi]]

        train_data = []
        val_data = []
        test_data = []    
        orig_train_data = []
        orig_val_data = []
        orig_test_data = []  
        
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
            orig_train_data.append(loader.featurize(train_files, shard_size=2**15))
            orig_val_data.append(loader.featurize(val_files, shard_size=2**15))
            orig_test_data.append(loader.featurize(test_files, shard_size=2**15))
            
            train_data.append(loader.featurize(train_files, shard_size=2**15))
            val_data.append(loader.featurize(val_files, shard_size=2**15))
            test_data.append(loader.featurize(test_files, shard_size=2**15))
            
            orig_train_data[label_index] = dc.data.NumpyDataset(orig_train_data[label_index].X, orig_train_data[label_index].y, 
                                                                orig_train_data[label_index].w, orig_train_data[label_index].ids)
            orig_val_data[label_index] = dc.data.NumpyDataset(orig_val_data[label_index].X, orig_val_data[label_index].y, 
                                                              orig_val_data[label_index].w, orig_val_data[label_index].ids)
            orig_test_data[label_index] = dc.data.NumpyDataset(orig_test_data[label_index].X, orig_test_data[label_index].y, 
                                                               orig_test_data[label_index].w, orig_test_data[label_index].ids)
    
            train_data[label_index] = dc.data.NumpyDataset(train_data[label_index].X, train_data[label_index].y, 
                                                           train_data[label_index].w, train_data[label_index].ids)
            val_data[label_index] = dc.data.NumpyDataset(val_data[label_index].X, val_data[label_index].y, 
                                                         val_data[label_index].w, val_data[label_index].ids)
            test_data[label_index] = dc.data.NumpyDataset(test_data[label_index].X, test_data[label_index].y, 
                                                          test_data[label_index].w, test_data[label_index].ids)
            
            train_data[label_index] = dc.trans.BalancingTransformer(transform_w=True, dataset=train_data[label_index]).transform(train_data[label_index])               
        
            transformers = [dc.trans.IRVTransformer(K_neighbors, 1, train_data[label_index])]
            for transformer in transformers:
                train_data[label_index] = transformer.transform(train_data[label_index])
                val_data[label_index] = transformer.transform(val_data[label_index])
                test_data[label_index] = transformer.transform(test_data[label_index])        
        
        #train model
        task.train_and_predict(train_data, val_data, test_data, model_file)
        
        #Undo transfromations and get metrics
        for label_index in range(len(labels)):            
            train_data[label_index] = orig_train_data[label_index]
            train_data[label_index] = dc.trans.BalancingTransformer(transform_w=True, dataset=train_data[label_index]).transform(train_data[label_index])
            transformers = [dc.trans.IRVTransformer(K_neighbors, 1, train_data[label_index])]
            for transformer in transformers:
                train_data[label_index] = transformer.transform(orig_train_data[label_index])
                val_data[label_index] = transformer.transform(orig_val_data[label_index])
                test_data[label_index] = transformer.transform(orig_test_data[label_index])       
        
        task.predict_with_existing(train_data, val_data, test_data)  
        for label_index in range(len(labels)): 
            if not os.path.exists(model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/'):
                os.makedirs(model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/')                                        
            task.save_model_evaluation_metrics(train_data, model_file,
                                              model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/train_metrics/',
                                              label_names=labels)
            task.save_model_evaluation_metrics(val_data, model_file,
                                              model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/val_metrics/',
                                              label_names=labels)
            task.save_model_evaluation_metrics(test_data, model_file,
                                              model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/test_metrics/',
                                              label_names=labels)
        
        task.save_model_params(config_csv_file)
        
        #save best_model for each fold 
        for label_index in range(len(labels)):      
            best_ckpt_file = model_file+'/'+labels[label_index]+'/best.ckpt'
            dest_ckpt_file = model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/best.ckpt'
            copy2(best_ckpt_file+'.data-00000-of-00001', dest_ckpt_file+'.data-00000-of-00001')
            copy2(best_ckpt_file+'.index', dest_ckpt_file+'.index')
            copy2(best_ckpt_file+'.meta', dest_ckpt_file+'.meta')
        
            rmtree(model_file+'/'+labels[label_index]+'/')
    elif stage == 2:
        vi = 0
        i = 1
        
        csv_file_list = output_file_list[:]
        val_files = [csv_file_list[vi]]
        
        train_files = [q for j, q in enumerate(csv_file_list) if j not in [vi]]

        train_data = []
        val_data = []  
        orig_train_data = []
        orig_val_data = [] 
        
        task = Deepchem_IRV(conf=conf, process_num=process_num, stage=stage)
        K_neighbors = task.K
        labels = task.labels
        
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
            orig_train_data.append(loader.featurize(train_files, shard_size=2**15))
            orig_val_data.append(loader.featurize(val_files, shard_size=2**15))
            
            train_data.append(loader.featurize(train_files, shard_size=2**15))
            val_data.append(loader.featurize(val_files, shard_size=2**15))
            
            orig_train_data[label_index] = dc.data.NumpyDataset(orig_train_data[label_index].X, orig_train_data[label_index].y, 
                                                                orig_train_data[label_index].w, orig_train_data[label_index].ids)
            orig_val_data[label_index] = dc.data.NumpyDataset(orig_val_data[label_index].X, orig_val_data[label_index].y, 
                                                              orig_val_data[label_index].w, orig_val_data[label_index].ids)
    
            train_data[label_index] = dc.data.NumpyDataset(train_data[label_index].X, train_data[label_index].y, 
                                                           train_data[label_index].w, train_data[label_index].ids)
            val_data[label_index] = dc.data.NumpyDataset(val_data[label_index].X, val_data[label_index].y, 
                                                         val_data[label_index].w, val_data[label_index].ids)
            
            train_data[label_index] = dc.trans.BalancingTransformer(transform_w=True, dataset=train_data[label_index]).transform(train_data[label_index])               
        
            transformers = [dc.trans.IRVTransformer(K_neighbors, 1, train_data[label_index])]
            for transformer in transformers:
                train_data[label_index] = transformer.transform(train_data[label_index])
                val_data[label_index] = transformer.transform(val_data[label_index])    
        
        #train model
        task.train_and_predict(train_data, val_data, None, model_file)
        
        #Undo transfromations and get metrics
        for label_index in range(len(labels)):            
            train_data[label_index] = orig_train_data[label_index]
            train_data[label_index] = dc.trans.BalancingTransformer(transform_w=True, dataset=train_data[label_index]).transform(train_data[label_index])
            transformers = [dc.trans.IRVTransformer(K_neighbors, 1, train_data[label_index])]
            for transformer in transformers:
                train_data[label_index] = transformer.transform(orig_train_data[label_index])
                val_data[label_index] = transformer.transform(orig_val_data[label_index])     
        
        task.predict_with_existing(train_data, val_data, None)  
        for label_index in range(len(labels)): 
            if not os.path.exists(model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/'):
                os.makedirs(model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/')                                       
            task.save_model_evaluation_metrics(train_data, model_file,
                                              model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/train_metrics/',
                                              label_names=labels)
            task.save_model_evaluation_metrics(val_data, model_file,
                                              model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/val_metrics/',
                                              label_names=labels)
        
        task.save_model_params(config_csv_file)
        
        #save best_model for each fold 
        for label_index in range(len(labels)):      
            best_ckpt_file = model_file+'/'+labels[label_index]+'/best.ckpt'
            dest_ckpt_file = model_dir+'fold_'+str(i)+'/'+labels[label_index]+'/best.ckpt'
            copy2(best_ckpt_file+'.data-00000-of-00001', dest_ckpt_file+'.data-00000-of-00001')
            copy2(best_ckpt_file+'.index', dest_ckpt_file+'.index')
            copy2(best_ckpt_file+'.meta', dest_ckpt_file+'.meta')
        
            rmtree(model_file+'/'+labels[label_index]+'/')
    
