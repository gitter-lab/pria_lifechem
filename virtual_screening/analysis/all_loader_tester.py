import sys
sys.path.insert(0, '..')  # Add path from parent folder
sys.path.insert(0, '.')  # Add path from current folder

from evaluation import *
from all_models_loader import *

model_directory = './rf_irv_results/'
data_directory = '../dataset/keck/fold_5/'
s1 = stage_1_results(model_directory, data_directory)


#random forest 12
labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test = s1['sklearn_rf_390014_12']

y_true = np.copy(y_test) # deep copy so we don't change y_test in next statement
y_true[np.where(np.isnan(y_true[:,i]))[0],i] = -1 #set nan to -1 to ignore in eval
roc1 = roc_auc_multi(y_true, y_pred_on_test, [0], np.mean)
pr1 = precision_auc_multi(y_true, y_pred_on_test, [0], np.mean, mode='auc.integral')

print('rf_12: roc:', roc1, 'pr:', pr1)

#irv 1
labels, y_train, y_val, y_test, y_pred_on_train, y_pred_on_val, y_pred_on_test = s1['deepchem_irv_390010_1']

y_true = np.copy(y_test) # deep copy so we don't change y_test in next statement
y_true[np.where(np.isnan(y_true[:,i]))[0],i] = -1 #set nan to -1 to ignore in eval
roc2 = roc_auc_multi(y_true, y_pred_on_test, [0], np.mean)
pr2 = precision_auc_multi(y_true, y_pred_on_test, [0], np.mean, mode='auc.integral')

print('irv_1: roc:', roc2, 'pr:', pr2)