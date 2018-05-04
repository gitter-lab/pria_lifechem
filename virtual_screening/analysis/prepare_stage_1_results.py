import sys
sys.path.insert(0, '..')  # Add path from parent folder
sys.path.insert(0, '.')  # Add path from current folder

from evaluation import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', action="store", dest="model_dir", required=True)
parser.add_argument('--output_dir_1', action="store", dest="output_dir_1", required=True)

#####
given_args = parser.parse_args()
model_directory = given_args.model_dir
output_dir_1 = given_args.output_dir_1

if not os.path.exists(output_dir_1):
    os.makedirs(output_dir_1)

EF_ratio_list = [0.001, 0.0015, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
perc_vec = EF_ratio_list

n_tests_list = [100, 500, 1000, 2500, 5000, 10000]

class_dirs = [model_directory+'/random_forest/stage_1/',
              model_directory+'/irv/stage_1/',
              model_directory+'/neural_networks/stage_1/',
              model_directory+'/docking/stage_1/']
model_classes=['random_forest',
              'irv',
              'neural_networks',
              'docking']

#loop on all models
for (model_class, c_dir) in zip(model_classes, class_dirs):
    for model_name in os.listdir(c_dir):
        print(model_name)
        for output_dir in [output_dir_1]:
            if not os.path.exists(output_dir+'/'+model_class+'/'):
                os.makedirs(output_dir+'/'+model_class+'/')
            if not os.path.exists(output_dir+'/'+model_class+'/stage_1/'):
                os.makedirs(output_dir+'/'+model_class+'/stage_1/')
            if not os.path.exists(output_dir+'/'+model_class+'/stage_1/'+model_name+'/'):
                os.makedirs(output_dir+'/'+model_class+'/stage_1/'+model_name+'/')
            
        num_of_folds = len(os.listdir(c_dir+'/'+model_name+'/'))
        for fold_num in range(num_of_folds):              
            fold_np = np.load(c_dir+'/'+model_name+'/fold_'+str(fold_num)+'.npz')
        
            labels, y_tr, y_v, y_te, y_pred_on_train, y_pred_on_val, y_pred_on_test = fold_np['labels'], fold_np['y_train'], fold_np['y_val'], fold_np['y_test'], fold_np['y_pred_on_train'], fold_np['y_pred_on_val'], fold_np['y_pred_on_test']
            
            labels = [e.decode('UTF-8') for e in labels]
            y_train = np.nan
            y_val = np.nan
            y_test = np.nan
            if not (y_tr.shape == ()):            
                y_train = np.copy(y_tr)
            if not (y_v.shape == ()):
                y_val = np.copy(y_v)
            if not (y_te.shape == ()):
                y_test = np.copy(y_te)
            for i, label in zip(range(len(labels)), labels): 
                if not (np.isscalar(y_train) and np.isnan(y_train)):
                    y_train[np.where(np.isnan(y_train[:,i]))[0],i] = -1 
                    y_train[np.where(np.isnan(y_pred_on_train[:,i]))[0],i] = -1 
                if not (np.isscalar(y_val) and np.isnan(y_val)):
                    y_val[np.where(np.isnan(y_val[:,i]))[0],i] = -1
                    y_val[np.where(np.isnan(y_pred_on_val[:,i]))[0],i] = -1
                if not (np.isscalar(y_test) and np.isnan(y_test)):
                    y_test[np.where(np.isnan(y_test[:,i]))[0],i] = -1
                    y_test[np.where(np.isnan(y_pred_on_test[:,i]))[0],i] = -1
                
            #record .out file format            
            if not os.path.exists(output_dir_1+'/'+model_class+'/stage_1/'+model_name+'/alt_format/'):
                os.makedirs(output_dir_1+'/'+model_class+'/stage_1/'+model_name+'/alt_format/')
            curr_model_dir = output_dir_1+'/'+model_class+'/stage_1/'+model_name+'/alt_format/'
            
            for l_index, label in zip(range(len(labels)), labels):
                if not os.path.exists(curr_model_dir+label+'/'):
                    os.makedirs(curr_model_dir+label+'/')
                
                num_of_iters = 4
                if num_of_folds > 5:
                    num_of_iters = 1
                    
                for j in range(num_of_iters):
                    q = fold_num*4 + j  
                    if num_of_folds > 5:
                        q = fold_num
                    
                    target = open(curr_model_dir+label+'/'+str(q)+'.out', 'w')
                    target.write('\n')
                    if np.isscalar(y_train) and np.isnan(y_train):
                        target.write('train precision: {}'.format(np.nan))
                        target.write('\n')
                        target.write('train roc: {}'.format(np.nan))
                        target.write('\n')
                        target.write('train bedroc: {}'.format(np.nan))
                        target.write('\n')
                        target.write('train nef_auc: {}'.format(np.nan))
                    else:
                        target.write('train precision: {}'.format(precision_auc_multi(y_train, y_pred_on_train, [l_index], np.mean)))
                        target.write('\n')
                        target.write('train roc: {}'.format(roc_auc_multi(y_train, y_pred_on_train, [l_index], np.mean)))
                        target.write('\n')
                        target.write('train bedroc: {}'.format(bedroc_auc_multi(y_train, y_pred_on_train, [l_index], np.mean)))
                        target.write('\n')
                        target.write('train nef_auc: {}'.format(np.array(nef_auc(y_train[:,l_index], y_pred_on_train[:,l_index], perc_vec))[0][0]))
                    target.write('\n')
                    target.write('\n')
                    if np.isscalar(y_val) and np.isnan(y_val):
                        target.write('validation precision: {}'.format(np.nan))                        
                        target.write('\n')
                        target.write('validation roc: {}'.format(np.nan))
                        target.write('\n')                       
                        target.write('validation bedroc: {}'.format(np.nan))
                        target.write('\n')
                        target.write('validation nef_auc: {}'.format(np.nan))
                    else:
                        target.write('validation precision: {}'.format(precision_auc_multi(y_val, y_pred_on_val, [l_index], np.mean)))
                        target.write('\n')
                        target.write('validation roc: {}'.format(roc_auc_multi(y_val, y_pred_on_val, [l_index], np.mean)))
                        target.write('\n')
                        target.write('validation bedroc: {}'.format(bedroc_auc_multi(y_val, y_pred_on_val, [l_index], np.mean)))
                        target.write('\n')
                        target.write('validation nef_auc: {}'.format(np.array(nef_auc(y_val[:,l_index], y_pred_on_val[:,l_index], perc_vec))[0][0]))
                    target.write('\n')
                    target.write('\n')
                    if np.isscalar(y_test) and np.isnan(y_test):
                        target.write('test precision: {}'.format(np.nan))
                        target.write('\n')
                        target.write('test roc: {}'.format(np.nan))
                        target.write('\n')
                        target.write('test bedroc: {}'.format(np.nan))
                        target.write('\n')
                        target.write('test nef_auc: {}'.format(np.nan))
                    else:
                        target.write('test precision: {}'.format(precision_auc_multi(y_test, y_pred_on_test, [l_index], np.mean)))
                        target.write('\n')
                        target.write('test roc: {}'.format(roc_auc_multi(y_test, y_pred_on_test, [l_index], np.mean)))
                        target.write('\n')
                        target.write('test bedroc: {}'.format(bedroc_auc_multi(y_test, y_pred_on_test, [l_index], np.mean)))
                        target.write('\n')
                        target.write('test nef_auc: {}'.format(np.array(nef_auc(y_test[:,l_index], y_pred_on_test[:,l_index], perc_vec))[0][0]))
                    target.write('\n')
                    target.write('\n')
                    
                    for EF_ratio in EF_ratio_list:
                        n_actives, ef, ef_max = enrichment_factor_single(y_test[:,l_index], y_pred_on_test[:,l_index], EF_ratio)
                        target.write('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))
                        target.write('\n')
                    target.write('\n')
                    for n_tests in n_tests_list:
                        n_hits = n_hits_calc_at_n_tests(y_test[:,l_index], y_pred_on_test[:,l_index], n_tests)
                        target.write('n_tests: {}, n_hits: {}'.format(n_tests, n_hits[0]))
                        target.write('\n')
                    
                    target.close()
            
            #record gather_df format
            curr_model_dir = output_dir_1+'/'+model_class+'/stage_1/'+model_name+'/fold_'+str(fold_num)+'/'
            if not os.path.exists(curr_model_dir):
                os.makedirs(curr_model_dir)
            if not os.path.exists(curr_model_dir+'/train_metrics/'):
                os.makedirs(curr_model_dir+'/train_metrics/')
            if not os.path.exists(curr_model_dir+'/val_metrics/'):
                os.makedirs(curr_model_dir+'/val_metrics/')
            if not os.path.exists(curr_model_dir+'/test_metrics/'):
                os.makedirs(curr_model_dir+'/test_metrics/')
            
            if not (np.isscalar(y_train) and np.isnan(y_train)):
                evaluate_model(y_train, y_pred_on_train, curr_model_dir+'/train_metrics/', labels, False)  
            if not (np.isscalar(y_val) and np.isnan(y_val)):
                evaluate_model(y_val, y_pred_on_val, curr_model_dir+'/val_metrics/', labels, False) 
            if not (np.isscalar(y_test) and np.isnan(y_test)):
                evaluate_model(y_test, y_pred_on_test, curr_model_dir+'/test_metrics/', labels, False)  
     