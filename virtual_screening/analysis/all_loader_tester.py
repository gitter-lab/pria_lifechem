import sys
sys.path.insert(0, '..')  # Add path from parent folder
sys.path.insert(0, '.')  # Add path from current folder

from evaluation import *
from all_models_loader import *

model_directory = './job_results/'
data_directory = '../../dataset/keck/fold_5/'
s1 = stage_1_results(model_directory, data_directory)


#random forest 12
labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test = s1['random_forest']['sklearn_rf_390014_12']['fold_3']

y_true = np.copy(y_test) # deep copy so we don't change y_test in next statement
for i, label in zip(range(len(labels)), labels): 
    y_true[np.where(np.isnan(y_true[:,i]))[0],i] = -1 #set nan to -1 to ignore in eval

roc1 = roc_auc_multi(y_true, y_pred_on_test, [0], np.mean)
pr1 = precision_auc_multi(y_true, y_pred_on_test, [0], np.mean, mode='auc.integral')

print('rf_12: roc:', roc1, 'pr:', pr1)

#irv 1
labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test = s1['irv']['deepchem_irv_390010_1']['fold_3']

y_true = np.copy(y_test) # deep copy so we don't change y_test in next statement
for i, label in zip(range(len(labels)), labels): 
    y_true[np.where(np.isnan(y_true[:,i]))[0],i] = -1 #set nan to -1 to ignore in eval

roc2 = roc_auc_multi(y_true, y_pred_on_test, [0], np.mean)
pr2 = precision_auc_multi(y_true, y_pred_on_test, [0], np.mean, mode='auc.integral')

print('irv_1: roc:', roc2, 'pr:', pr2)


#loop on all models
for model_class in s1:
    for model_name in s1[model_class]:
        for fold_num in s1[model_class][model_name]:
            labels, y_tr, y_v, y_te, y_pred_on_train, y_pred_on_val, y_pred_on_test = s1['irv']['deepchem_irv_390010_1']['fold_3']

            y_train = np.copy(y_tr)
            y_val = np.copy(y_v)
            y_test = np.copy(y_te)
            for i, label in zip(range(len(labels)), labels): 
                y_train[np.where(np.isnan(y_train[:,i]))[0],i] = -1 
                y_val[np.where(np.isnan(y_val[:,i]))[0],i] = -1
                y_test[np.where(np.isnan(y_test[:,i]))[0],i] = -1
            
            #here you can add your code for evaluating 
    
    