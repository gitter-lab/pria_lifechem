import pandas as pd
import csv
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from sklearn.metrics import auc
from croc import BEDROC, ScoredData
import os


'''
this if for multi-task evaluation
y_true and y_pred is two-dimension matrix
can evaluate on mean or median of array
called by
roc_auc_multi(y_true, y_pred, [-1], np.mean)
roc_auc_multi(y_true, y_pred, [0], np.median)
'''
def roc_auc_multi(y_true, y_pred, eval_indices, eval_mean_or_median):
    y_true = y_true[:, eval_indices]
    y_pred = y_pred[:, eval_indices]
    nb_classes = y_true.shape[1]
    auc = np.zeros(nb_classes)
    for i in range(len(auc)):
        # -1 represents missing value
        # and remove them when in evaluation
        non_missing_indices = np.argwhere(y_true[:, i] != -1)[:, 0]
        actual = y_true[non_missing_indices, i]
        predicted = y_pred[non_missing_indices, i]
        auc[i] = roc_auc_single(actual, predicted)
    return eval_mean_or_median(auc)


def roc_auc_single(actual, predicted):
    try:
        auc_ret = roc_auc_score(actual, predicted)
    except ValueError:
        auc_ret = np.nan
    
    return auc_ret


'''
this if for multi-task evaluation
y_true and y_pred is two-dimension matrix
can evaluate on mean or median of array
called by
bedroc_auc_multi(y_true, y_pred, [-1], np.mean)
bedroc_auc_multi(y_true, y_pred, [0], np.median)
'''
def bedroc_auc_multi(y_true, y_pred, eval_indices, eval_mean_or_median):
    y_true = y_true[:, eval_indices]
    y_pred = y_pred[:, eval_indices]
    nb_classes = y_true.shape[1]
    auc = np.zeros(nb_classes)
    for i in range(len(auc)):
        # -1 represents missing value
        # and remove them when in evaluation
        non_missing_indices = np.argwhere(y_true[:, i] != -1)[:, 0]
        actual = y_true[non_missing_indices, i:(i+1)]
        predicted = y_pred[non_missing_indices, i:(i+1)]
        auc[i] = bedroc_auc_single(actual, predicted) 
    return eval_mean_or_median(auc)


def bedroc_auc_single(actual, predicted, alpha=10):
    try:
        data = np.hstack((predicted, actual))
        data = ScoredData(data)
        results = BEDROC(data, alpha)
        return results['area']
    except:
        return np.nan


'''
this if for multi-task evaluation
y_true and y_pred is two-dimension matrix
can evaluate on mean or median of array
called by
precision_auc_multi(y_true, y_pred, [-1], np.mean)
precision_auc_multi(y_true, y_pred, [0], np.median)

we calculate each single AUC[PR] through a R package called PRROC or sklearn
the mode can be either 'auc.integral', 'auc.davis.goadrich', or 'auc.sklearn'
'''
def precision_auc_multi(y_true, y_pred, eval_indices, eval_mean_or_median,
                        mode='auc.integral'):
    y_true = y_true[:, eval_indices]
    y_pred = y_pred[:, eval_indices]
    nb_classes = y_true.shape[1]
    auc = np.zeros(nb_classes)
    for i in range(len(auc)):
        # -1 represents missing value
        # and remove them when in evaluation
        non_missing_indices = np.argwhere(y_true[:, i] != -1)[:, 0]
        actual = y_true[non_missing_indices, i]
        predicted = y_pred[non_missing_indices, i]
        auc[i] = precision_auc_single(actual, predicted, mode)
    return eval_mean_or_median(auc)


'''
the average_precision_score() function in sklearn has interpolation issue
we call this through a R package called PRROC or sklearn
the mode can be either 'auc.integral', 'auc.davis.goadrich', or 'auc.sklearn'
'''
def precision_auc_single(actual, predicted, mode='auc.integral'):
    if mode == 'auc.sklearn':
        try:
            prec_auc = average_precision_score(actual, predicted)
        except ValueError:
            prec_auc = np.nan
    else:
        try:
            prroc = rpackages.importr('PRROC')
            x = robjects.FloatVector(actual)
            y = robjects.FloatVector(predicted)
            pr = prroc.pr_curve(weights_class0=x, scores_class0=y, curve=False)
            prec_auc = pr.rx2(mode)[0]
        except ValueError:
            prec_auc = np.nan
    return prec_auc


def number_of_hit_single(actual, predicted, N):
    assert N <= actual.shape[0], \
        'Top Number N=[{}] must be no greater than total compound number [{}]'.format(N, actual.shape[0])

    if predicted.ndim == 2:
        predicted = predicted[:, 0]
    if actual.ndim == 2:
        actual = actual[:, 0]

    top_N_index = predicted.argsort()[::-1][:N]
    top_N = actual[top_N_index]
    n_hit = sum(top_N)
    return n_hit


def ratio_of_hit_single(actual, predicted, R):
    assert 0.0 <= R <= 1.0, 'Top Ratio R=[{}] must be within [0.0, 1.0]'.format(R)
    N = int(R * actual.shape[0])
    return number_of_hit_single(actual, predicted, N)


'''
Creates a plot for each label for
modes: pr, roc, efp_efm, nef
'''
def plot_curve_multi(actual, predicted, file_dir, mode='pr', label_names=None, perc_vec=None, ):
    plot_curve_function = { 'pr' : plot_pr_curve,
                            'roc' : plot_roc_curve,
                            'efp_efm' : plot_efp_efm,
                            'nef' : plot_nef
                            }    
    
    file_dir = file_dir+mode
    if (mode == 'efp_efm' or mode == 'nef'):
        if perc_vec == None:
            perc_vec = np.linspace(0.005, .2, 100)
        plot_curve_function[mode](actual, predicted, perc_vec, file_dir, label_names)
    else:
        plot_curve_function[mode](actual, predicted, file_dir, label_names)


def enrichment_factor_multi(actual, predicted, percentile, eval_indices):
    actual = actual[:, eval_indices]
    predicted = predicted[:, eval_indices]
    nb_classes = actual.shape[1]
    EF_list = []
    for i in range(nb_classes):
        n_actives, ef, ef_max = enrichment_factor_single(actual[:, i], predicted[:, i], percentile)
        temp = [n_actives, ef]
        EF_list.append(temp)
    return EF_list


def enrichment_factor_single(labels_arr, scores_arr, percentile):
    '''
    calculate the enrichment factor based on some upper fraction
    of library ordered by docking scores. upper fraction is determined
    by percentile (actually a fraction of value 0.0-1.0)

    -1 represents missing value
    and remove them when in evaluation
    '''
    non_missing_indices = np.argwhere(labels_arr != -1)[:, 0]
    labels_arr = labels_arr[non_missing_indices]
    scores_arr = scores_arr[non_missing_indices]

    sample_size = int(labels_arr.shape[0] * percentile)  # determine number mols in subset
    pred = np.sort(scores_arr, axis=0)[::-1][:sample_size]  # sort the scores list, take top subset from library
    indices = np.argsort(scores_arr, axis=0)[::-1][:sample_size]  # get the index positions for these in library
    n_actives = np.nansum(labels_arr)  # count number of positive labels in library
    total_actives = np.nansum(labels_arr)
    total_count = len(labels_arr)
    n_experimental = np.nansum(labels_arr[indices])  # count number of positive labels in subset
    temp = scores_arr[indices]

    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile  # calc EF at percentile
        ef_max = min(n_actives, sample_size) / (n_actives * percentile)
    else:
        ef = 'ND'
        ef_max = 'ND'
    return n_actives, ef, ef_max


def normalized_enrichment_factor_single(labels_arr, scores_arr, percentile):
    n_actives, ef, ef_max = enrichment_factor_single(labels_arr, scores_arr, percentile)
    return ef/ef_max


def enrichment_factor_single_perc(y_true, y_pred, percentile):
    """
    Calculates enrichment factor vector at the given percentile for multiple labels.
    This returns a 1D vector with the EF scores of the labels.
    """
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
    else:
        y_true = y_true.reshape((y_true.shape[0], 1))
        y_pred = y_pred.reshape((y_pred.shape[0], 1)) 
    
    ef = np.zeros(nb_classes)
    
    for i in range(len(ef)):
        non_missing_indices = np.argwhere(y_true[:, i] != -1)[:, 0]
        true_labels = y_true[non_missing_indices, i]
        pred = y_pred[non_missing_indices, i]
    
        sample_size = int(true_labels.shape[0] * percentile)
        indices = np.argsort(pred, axis=0)[::-1][:sample_size]
        
        n_actives = np.nansum(true_labels) 
        n_experimental = np.nansum( true_labels[indices] )
        
        try:
            ef[i] = ( float(n_experimental) /  n_actives ) / percentile 
        except ValueError:
            ef[i] = np.nan
            
    return ef


def max_enrichment_factor_single_perc(y_true, y_pred, percentile):
    """
    Calculates max enrichment factor vector at the given percentile for multiple labels.
    This returns a 1D vector with the EF scores of the labels.
    """
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
    else:
        y_true = y_true.reshape((y_true.shape[0], 1))
        y_pred = y_pred.reshape((y_pred.shape[0], 1)) 
    
    max_ef = np.zeros(nb_classes)
    
    for i in range(len(max_ef)):
        non_missing_indices = np.argwhere(y_true[:, i] != -1)[:, 0]
        true_labels = y_true[non_missing_indices, i]    
        n_actives = np.nansum(true_labels) 
        
        sample_size = int(true_labels.shape[0] * percentile)
        try:
            max_ef[i] = ( min(n_actives, sample_size) /  n_actives ) / percentile 
        except ValueError:
            max_ef[i] = np.nan
            
    return max_ef


def enrichment_factor(y_true, y_pred, perc_vec, label_names=None):     
    """
    Calculates enrichment factor vector at the percentile vectors. This returns
    2D panda matrix where the rows are the percentile.
    """
    p_count = len(perc_vec)    
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
    ef_mat = np.zeros((p_count, nb_classes))
    
    for curr_perc in range(p_count):
        ef_mat[curr_perc,:] = enrichment_factor_single_perc(y_true, 
                                        y_pred, perc_vec[curr_perc])                
        
    """
    Convert to pandas matrix row-col names
    """
    index_names = ['{:g}'.format(perc * 100) + ' %' for perc in perc_vec] 
    ef_pd = pd.DataFrame(data=np.concatenate((ef_mat,
                                              np.mean(ef_mat,axis=1).reshape(len(perc_vec),1),
                                              np.median(ef_mat,axis=1).reshape(len(perc_vec),1)),axis=1),
                         index=index_names,
                         columns=label_names+['Mean','Median'])
    ef_pd.index.name = 'EF'
    return ef_pd


def confident_hit_ratio(y_true, y_pred, cut_off=0.1):
    """
    This function return the hit ratio of the true-positive for confident molecules.
    Confident molecules are defined as confidence values that are higher than the cutoff.
    :param y_true:
    :param y_pred:
    :param cut_off: confident value that defines if a prediction are considered confident
    :return:
    """
    actual_indexes = np.where(y_true==1)[0]
    confident_indexes = np.where(y_pred>cut_off)[0]
    confident_hit = np.intersect1d(actual_indexes, confident_indexes)
    ratio = 1.0 * len(confident_hit) / len(actual_indexes)
    return ratio


def max_enrichment_factor(y_true, y_pred, perc_vec, label_names=None): 
    """
    Calculates max enrichment factor vector at the percentile vectors. This returns
    2D panda matrix where the rows are the percentile.
    """   
    p_count = len(perc_vec)        
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
        
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
    max_ef_mat = np.zeros((p_count, nb_classes))
    
    for curr_perc in range(p_count):
        max_ef_mat[curr_perc,:] = max_enrichment_factor_single_perc(y_true, 
                                        y_pred, perc_vec[curr_perc])                
        
    """
    Convert to pandas matrix row-col names
    """
    
    index_names = ['{:g}'.format(perc * 100) + ' %' for perc in perc_vec]    
    max_ef_pd = pd.DataFrame(data=np.concatenate((max_ef_mat,
                                              np.mean(max_ef_mat,axis=1).reshape(len(perc_vec),1),
                                              np.median(max_ef_mat,axis=1).reshape(len(perc_vec),1)),axis=1),
                         index=index_names,
                         columns=label_names+['Mean','Median'])
    max_ef_pd.index.name = 'Max_EF'                     
    return max_ef_pd
    

def norm_enrichment_factor(y_true, y_pred, perc_vec, label_names=None): 
    """
    Calculates normalized enrichment factor vector at the percentile vectors. 
    This returns three 2D panda matrices (norm_ef, ef, max_ef) where the rows 
    are the percentile.
    """       
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
        
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
    ef_pd = enrichment_factor(y_true, y_pred, 
                               perc_vec, label_names)
    max_ef_pd = max_enrichment_factor(y_true, y_pred, 
                                       perc_vec, label_names)
    
    nef_mat = ef_pd.as_matrix() / max_ef_pd.as_matrix() 
    index_names = ['{:g}'.format(perc * 100) + ' %' for perc in perc_vec]    
    nef_pd = pd.DataFrame(data=nef_mat,
                         index=index_names,
                         columns=label_names+['Mean','Median'])
    nef_pd.index.name = 'NEF' 
    return nef_pd, ef_pd, max_ef_pd


def nef_auc(y_true, y_pred, perc_vec, label_names=None):
    """
    Returns a pandas df of nef auc values, one for each label.
    """
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
        
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
    nef_mat, ef_mat, ef_max_mat  = norm_enrichment_factor(y_true, y_pred, 
                                                         perc_vec, label_names)
    nef_mat = nef_mat.as_matrix() 
    ef_mat = ef_mat.as_matrix()                                                         
    ef_max_mat = ef_max_mat.as_matrix()     
        
    nef_auc_arr = np.zeros(nb_classes) 
    for i in range(nb_classes):
        nef_auc_arr[i] = auc(perc_vec, nef_mat[:,i])
             
    mean_nef = nef_mat[:,-2]         
    
    random_mean_nef = 1 / ef_max_mat[:,-2]
    
    nef_auc_arr = np.concatenate((nef_auc_arr, 
                              auc(perc_vec, mean_nef).reshape(1,), 
                              auc(perc_vec, random_mean_nef).reshape(1,)))
    
    nef_auc_pd = pd.DataFrame(data=nef_auc_arr.reshape(1,len(nef_auc_arr)) / max(perc_vec),
                             index=['NEF_AUC'],
                             columns=label_names+['Mean', 'Random Mean'])
    return nef_auc_pd


def n_hits_calc(y_true, y_pred, n_tests_list, label_names=None): 
    """
    Calculates number of actives found in the top n_tests in n_tests_list.
    """   
    t_count = len(n_tests_list)    
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
        
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
    
    n_hits_mat = np.zeros((t_count, nb_classes))
    
    for curr_n_tests in range(t_count):
        n_hits_mat[curr_n_tests,:] = n_hits_calc_at_n_tests(y_true, 
                                            y_pred, n_tests_list[curr_n_tests])                
        
    """
    Convert to pandas matrix row-col names
    """
    index_names = ['{:g}'.format(n_tests) for n_tests in n_tests_list] 
    n_hits_pd = pd.DataFrame(data=np.concatenate((n_hits_mat,
                                              np.mean(n_hits_mat,axis=1).reshape(len(n_tests_list),1),
                                              np.median(n_hits_mat,axis=1).reshape(len(n_tests_list),1)),axis=1),
                         index=index_names,
                         columns=label_names+['Mean','Median'])
    n_hits_pd.index.name = 'n_hits'
    
    return n_hits_pd

def n_hits_calc_at_n_tests(y_true, y_pred, n_tests):
    """
    Calculates number of actives found in the top n_tests.
    """
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
    else:
        y_true = y_true.reshape((y_true.shape[0], 1))
        y_pred = y_pred.reshape((y_pred.shape[0], 1)) 
    
    n_hits = np.zeros(nb_classes)
    
    for i in range(len(n_hits)):
        non_missing_indices = np.argwhere(y_true[:, i] != -1)[:, 0]
        true_labels = y_true[non_missing_indices, i]
        pred = y_pred[non_missing_indices, i]
    
        indices = np.argsort(pred, axis=0)[::-1][:n_tests]
        
        n_actives = np.nansum(true_labels) 
        n_hits[i] = np.nansum( true_labels[indices] )
            
    return n_hits

    
def evaluate_model(y_true, y_pred, model_dir, label_names=None):
    """
    Call this function to evaluate a model. This will call all evaluations we 
    would like to store. It will save it in model_dir.
    """
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
        
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
    perc_vec = [0.001, 0.0015, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    perc_vec_plots = np.linspace(0.001, .2, 100) 
    n_tests_list = [100, 250, 500, 1000, 2500, 5000, 10000]

    metrics_dir = model_dir+'metrics.csv'
    
    
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
    else:
        y_true = y_true.reshape((y_true.shape[0], 1))
        y_pred = y_pred.reshape((y_pred.shape[0], 1)) 
        
    # dataframe metrics    
    roc_auc_df = roc_auc_multi(y_true, y_pred, range(nb_classes), np.copy)  
    roc_auc_df = np.concatenate((roc_auc_df,
                                 np.mean(roc_auc_df).reshape(1,),
                                 np.median(roc_auc_df).reshape(1,)))
    roc_auc_df = pd.DataFrame(data=roc_auc_df.reshape(1,len(roc_auc_df)),
                            index=['ROC AUC'],
                            columns=label_names+['Mean','Median'])
    roc_auc_df.index.name='metric'
      
    bedroc_auc_df = bedroc_auc_multi(y_true, y_pred, range(nb_classes), np.copy)  
    bedroc_auc_df = np.concatenate((bedroc_auc_df,
                                   np.mean(bedroc_auc_df).reshape(1,),
                                   np.median(bedroc_auc_df).reshape(1,)))
    bedroc_auc_df = pd.DataFrame(data=bedroc_auc_df.reshape(1,len(bedroc_auc_df)),
                                index=['BEDROC AUC'],
                                columns=label_names+['Mean','Median'])
    bedroc_auc_df.index.name='metric'
    
    sklearn_pr_auc_df = precision_auc_multi(y_true, y_pred, range(nb_classes), np.copy, 'auc.sklearn')
    sklearn_pr_auc_df = np.concatenate((sklearn_pr_auc_df,
                                       np.mean(sklearn_pr_auc_df).reshape(1,),
                                       np.median(sklearn_pr_auc_df).reshape(1,)))
    sklearn_pr_auc_df = pd.DataFrame(data=sklearn_pr_auc_df.reshape(1,len(sklearn_pr_auc_df)),
                                    index=['PR auc.sklearn'],
                                    columns=label_names+['Mean','Median'])
    sklearn_pr_auc_df.index.name='metric'
    
    integral_pr_auc_df = precision_auc_multi(y_true, y_pred, range(nb_classes), np.copy, 'auc.integral')
    integral_pr_auc_df = np.concatenate((integral_pr_auc_df,
                                       np.mean(integral_pr_auc_df).reshape(1,),
                                       np.median(integral_pr_auc_df).reshape(1,)))
    integral_pr_auc_df = pd.DataFrame(data=integral_pr_auc_df.reshape(1,len(integral_pr_auc_df)),
                                      index=['PR auc.integral'],
                                      columns=label_names+['Mean','Median'])
    integral_pr_auc_df.index.name='metric'
    
    dg_pr_auc_df = precision_auc_multi(y_true, y_pred, range(nb_classes), np.copy, 'auc.davis.goadrich')
    dg_pr_auc_df = np.concatenate((dg_pr_auc_df,
                                   np.mean(dg_pr_auc_df).reshape(1,),
                                   np.median(dg_pr_auc_df).reshape(1,)))
    dg_pr_auc_df = pd.DataFrame(data=dg_pr_auc_df.reshape(1,len(dg_pr_auc_df)),
                                index=['PR auc.davis.goadrich'],
                                columns=label_names+['Mean','Median'])
    dg_pr_auc_df.index.name='metric'
    
    nef_pd, ef_pd, max_ef_pd = norm_enrichment_factor(y_true, y_pred, perc_vec, label_names)
    nef_auc_df = nef_auc(y_true, y_pred, perc_vec, label_names) 
    n_hits_df = n_hits_calc(y_true, y_pred, n_tests_list, label_names)
    
    pr_roc_frames = [roc_auc_df, bedroc_auc_df, sklearn_pr_auc_df, 
                     integral_pr_auc_df, dg_pr_auc_df]
    pr_roc_frames = pd.concat(pr_roc_frames)
    
    pr_roc_frames.to_csv(metrics_dir)
    with open(metrics_dir,'a') as f:  
        f.write('\n')    
    for pd_df in [nef_pd, ef_pd, max_ef_pd, nef_auc_df, n_hits_df]:
        pd_df.to_csv(metrics_dir, mode='a')
        with open(metrics_dir,'a') as f:  
            f.write('\n')


def results_describe(true_label, pred_label):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    true_pos_count = 0
    pred_pos_count = 0
    for (x, y) in zip(true_label, pred_label):
        if x == y:
            TP += (x == 1)
            TN += (x == 0)
        else:
            FN += (x == 1)
            FP += (x == 0)
        true_pos_count += (x == 1)
        pred_pos_count += (y == 1)
    return TP, TN, FP, FN, true_pos_count, pred_pos_count
