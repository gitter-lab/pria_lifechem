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
import matplotlib
import matplotlib.pyplot as plt

"""
width and height of plots
"""
w, h = 16, 10

'''
this if for multi-task evaluation
y_true and y_pred is two-dimension matrix
can evaluate on mean or median of array
called by
roc_auc_multi(y_true, y_pred, [-1], np.mean)
roc_auc_multi(y_true, y_pred, [0], np.median)
'''
def roc_auc_multi(y_true, y_pred, eval_indices, eval_mean_or_median,
                  return_df=False, label_names=None):
    
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
    
    if return_df == True:
        if label_names == None:
            label_names = ['label ' + str(i) for i in range(nb_classes)]
        
        auc_data = np.concatenate((auc,
                                   np.mean(auc).reshape(1,),
                                   np.median(auc).reshape(1,)))
        auc_df = pd.DataFrame(data=auc_data.reshape(1,len(auc_data)),
                              index=['ROC AUC'],
                              columns=label_names+['Mean','Median'])
        auc_df.index.name='metric'
        return auc_df
    else:
        return eval_mean_or_median(auc)
    
def roc_auc_single(actual, predicted):
    return roc_auc_score(actual, predicted)


'''
this if for multi-task evaluation
y_true and y_pred is two-dimension matrix
can evaluate on mean or median of array
called by
bedroc_auc_multi(y_true, y_pred, [-1], np.mean)
bedroc_auc_multi(y_true, y_pred, [0], np.median)
'''
def bedroc_auc_multi(y_true, y_pred, eval_indices, eval_mean_or_median,
                     return_df=False, label_names=None):
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
    
    if return_df == True:
        if label_names == None:
            label_names = ['label ' + str(i) for i in range(nb_classes)]
        
        auc_data = np.concatenate((auc,
                                   np.mean(auc).reshape(1,),
                                   np.median(auc).reshape(1,)))        
        auc_df = pd.DataFrame(data=auc_data.reshape(1,len(auc_data)),
                              index=['BEDROC AUC'],
                              columns=label_names+['Mean','Median'])
        auc_df.index.name='metric'
        return auc_df
    else:    
        return eval_mean_or_median(auc)


def bedroc_auc_single(actual, predicted, alpha=10):
    data = np.hstack((predicted, actual))
    data = ScoredData(data)
    results = BEDROC(data, alpha)
    return results['area']


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
                        mode='auc.integral',
                        return_df=False, label_names=None):
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
    
    if return_df == True:
        if label_names == None:
            label_names = ['label ' + str(i) for i in range(nb_classes)]
        
        auc_data = np.concatenate((auc,
                                   np.mean(auc).reshape(1,),
                                   np.median(auc).reshape(1,))) 
        auc_df = pd.DataFrame(data=auc_data.reshape(1,len(auc_data)),
                                  index=['PR ' + mode],
                                  columns=label_names+['Mean','Median'])
        auc_df.index.name='metric'
        return auc_df
    else:
        return eval_mean_or_median(auc)

'''
the average_precision_score() function in sklearn has interpolation issue
we call this through a R package called PRROC or sklearn
the mode can be either 'auc.integral', 'auc.davis.goadrich', or 'auc.sklearn'
'''
def precision_auc_single(actual, predicted, mode='auc.integral'):
    if mode == 'auc.sklearn':
        prec_auc = average_precision_score(actual, predicted)
    else:
        prroc = rpackages.importr('PRROC')
        x = robjects.FloatVector(actual)
        y = robjects.FloatVector(predicted)
        pr = prroc.pr_curve(weights_class0=x, scores_class0=y, curve=False)
        prec_auc = pr.rx2(mode)[0]
    return prec_auc


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

def plot_pr_curve(actual, predicted, file_dir, label_names=None):
    prroc = rpackages.importr('PRROC')
    
    nb_classes = 1    
    if len(actual.shape) == 2:
        nb_classes = actual.shape[1]
        
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
    lw = 2
    mean_y, mean_x = np.array([]), np.array([])
    for i in range(nb_classes):  
        non_missing_indices = np.argwhere(actual[:, i] != -1)[:, 0]
        mean_x = np.concatenate((mean_x, actual[non_missing_indices,i]))
        mean_y = np.concatenate((mean_y, predicted[non_missing_indices,i]))
    
    mean_y, mean_x, _ = precision_recall_curve(mean_x.ravel(), 
                                               mean_y.ravel())
    
    mean_auc = precision_auc_multi(actual, predicted, range(nb_classes), np.mean)
    median_auc = precision_auc_multi(actual, predicted, range(nb_classes), np.median)
    for i in range(nb_classes):  
        non_missing_indices = np.argwhere(actual[:, i] != -1)[:, 0]
        plt.figure(figsize=(w,h))
        y, x, _ = precision_recall_curve(actual[non_missing_indices,i], predicted[non_missing_indices,i])
        auc = average_precision_score(actual[non_missing_indices,i], predicted[non_missing_indices,i])
        plt.plot(x, y, lw=lw, label='label '+label_names[i] + 
        '(sklearn = %0.5f)' % (auc))
        
        x = robjects.FloatVector(actual[non_missing_indices,i])
        y = robjects.FloatVector(predicted[non_missing_indices,i])
        pr = prroc.pr_curve(x, y, curve=True)
        pr_curve = np.array(pr[3])
        plt.plot(pr_curve[:, 0], pr_curve[:, 1], lw=lw,
                 label='label '+label_names[i] + 
                 '(integral area = %0.5f) \n (d+g area = %0.5f)' %
                       (pr[1][0], pr[2][0])) 
    
        plt.plot(mean_x, mean_y, color='g', linestyle='--',
             label='Mean (area = %0.2f)' % 
             (mean_auc), lw=lw)
        
            
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for label ' + label_names[i])
        plt.legend(loc="lower right")
        plt.savefig(file_dir+'_curve_{}.png'.format(label_names[i]),bbox_inches='tight')
        plt.close()

def plot_roc_curve(actual, predicted, file_dir, label_names=None):
    nb_classes = 1    
    if len(actual.shape) == 2:
        nb_classes = actual.shape[1]
        
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
    lw = 2
    mean_x, mean_y = np.array([]), np.array([])
    for i in range(nb_classes):
        non_missing_indices = np.argwhere(actual[:, i] != -1)[:, 0]
        mean_x = np.concatenate((mean_x, actual[non_missing_indices,i]))
        mean_y = np.concatenate((mean_y, predicted[non_missing_indices,i]))
    
    mean_x, mean_y, _ = roc_curve(mean_x.ravel(), 
                                  mean_y.ravel())
    
    mean_auc = roc_auc_multi(actual, predicted, range(nb_classes), np.mean)
    median_auc = roc_auc_multi(actual, predicted, range(nb_classes), np.median)
    for i in range(nb_classes):
        non_missing_indices = np.argwhere(actual[:, i] != -1)[:, 0]
        plt.figure(figsize=(w,h))
        x, y, _ = roc_curve(actual[non_missing_indices,i], predicted[non_missing_indices,i])
        auc = roc_auc_score(actual[non_missing_indices,i], predicted[non_missing_indices,i])
        plt.plot(x, y, lw=lw, label='label '+label_names[i] + 
        '(sklearn = %0.5f)' % (auc))
    
        plt.plot(mean_x, mean_y, color='g', linestyle='--',
             label='Mean (area = %0.2f)' % 
             (mean_auc), lw=lw)
        
            
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve for label ' + label_names[i])
        plt.legend(loc="lower right")
        plt.savefig(file_dir+'_curve_{}.png'.format(label_names[i]),bbox_inches='tight')
        plt.close()


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
    non_missing_indices = np.argwhere(labels_arr!=-1)[:, 0]
    labels_arr = labels_arr[non_missing_indices]
    scores_arr = scores_arr[non_missing_indices]

    sample_size = int(labels_arr.shape[0] * percentile)         # determine number mols in subset
    pred = np.sort(scores_arr, axis=0)[::-1][:sample_size]              # sort the scores list, take top subset from library
    indices = np.argsort(scores_arr, axis=0)[::-1][:sample_size]        # get the index positions for these in library
    n_actives = np.nansum(labels_arr)                           # count number of positive labels in library
    total_actives = np.nansum(labels_arr)
    total_count = len(labels_arr)
    n_experimental = np.nansum(labels_arr[indices])            # count number of positive labels in subset
    temp = scores_arr[indices]
    
    if n_actives > 0.0:
        ef = float(n_experimental) / n_actives / percentile     # calc EF at percentile
        ef_max = min(n_actives, sample_size) / ( n_actives * percentile )
    else:
        ef = 'ND'
        ef_max = 'ND'
    return n_actives, ef, ef_max


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
    
    non_missing_indices = np.argwhere(y_true!=-1)[:, 0]
    y_true = y_true[non_missing_indices,:]
    y_pred = y_pred[non_missing_indices,:]
    
    ef = np.zeros(nb_classes)
    sample_size = int(y_true.shape[0] * percentile)
    
    for i in range(len(ef)):
        true_labels = y_true[:, i]
        pred = np.sort(y_pred[:, i], axis=0)[::-1][:sample_size]
        indices = np.argsort(y_pred[:, i], axis=0)[::-1][:sample_size]
        
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
    
    non_missing_indices = np.argwhere(y_true!=-1)[:, 0]
    y_true = y_true[non_missing_indices,:]
    y_pred = y_pred[non_missing_indices,:]  
    
    max_ef = np.zeros(nb_classes)
    sample_size = int(y_true.shape[0] * percentile)
    
    for i in range(len(max_ef)):
        true_labels = y_true[:, i]        
        n_actives = np.nansum(true_labels) 
        
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
    nef_mat, ef_mat, ef_max_mat  = norm_enrichment_factor(y_true, y_pred, 
                                                         perc_vec, label_names)
    nef_mat = nef_mat.as_matrix() 
    ef_mat = ef_mat.as_matrix()                                                         
    ef_max_mat = ef_max_mat.as_matrix() 
    
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
        
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
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
    
def plot_nef(y_true, y_pred, perc_vec, file_dir, label_names=None):
    """
    Plots the EF_Normalized curve with notable info. Also returns the EF_Norm AUC; 
    upper bound is 1.
    If more than one label is given, draws curves for each label and 
    the mean curve. Returns a vector of auc values, one for each label.
    """
    nef_mat, ef_mat, ef_max_mat  = norm_enrichment_factor(y_true, y_pred, 
                                                         perc_vec, label_names)
    nef_mat = nef_mat.as_matrix() 
    ef_mat = ef_mat.as_matrix()                                                         
    ef_max_mat = ef_max_mat.as_matrix() 
    
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
        
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
    lw = 2
    nef_auc = np.zeros(nb_classes) 
    random_mean_nef = 1 / ef_max_mat[:,-2] 
    mean_nef = nef_mat[:,-2]     
    for i in range(nb_classes):
        plt.figure(figsize=(w,h))
        nef_auc[i] = auc(perc_vec, nef_mat[:,i])
        plt.plot(perc_vec, nef_mat[:,i], lw=lw,
             label=label_names[i] + ' (area = %0.2f)' % 
             (nef_auc[i] / max(perc_vec)))
             
        plt.plot(perc_vec, mean_nef, color='g', linestyle='--',
             label='Mean NEF (area = %0.2f)' % 
             (auc(perc_vec, mean_nef) / max(perc_vec)), lw=lw)
        
        plt.plot(perc_vec, random_mean_nef, linestyle='--',
             label='Random Mean NEF (area = %0.2f)' % 
             (auc(perc_vec, random_mean_nef) / max(perc_vec)), lw=lw)
            
        plt.xlim([-0.01, max(perc_vec) + 0.02])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Percentile')
        plt.ylabel('NEF')
        plt.title('Normalized EF Curve for label ' + label_names[i])
        plt.legend(loc="lower right")
        plt.savefig(file_dir+'_curve_{}.png'.format(label_names[i]),bbox_inches='tight')
        plt.close()
    
    return np.mean(nef_auc) / max(perc_vec)
    
def plot_efp_efm(y_true, y_pred, perc_vec, file_dir, label_names=None):
    """
    Plots the EF_perc+EF_max curves with notable info.
    If more than one label is given, draws curves for each label.
    """
    ef_mat = enrichment_factor(y_true, y_pred, 
                               perc_vec, label_names).as_matrix()
    max_ef_mat = max_enrichment_factor(y_true, y_pred, 
                                       perc_vec, label_names).as_matrix()    
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
        
    if label_names == None:
        label_names = ['label ' + str(i) for i in range(nb_classes)]
        
    lw = 2
    for i in range(nb_classes):
        plt.figure(figsize=(w,h))
        plt.plot(perc_vec, ef_mat[:,i], lw=lw,
             label=label_names[i])
        plt.plot(perc_vec, max_ef_mat[:,i], lw=lw,
             label=label_names[i] + ' max')
             
        plt.xlim([0.0, max(perc_vec) + 0.01])
        plt.ylim([-0.05, np.max(max_ef_mat)+10])
        plt.xlabel('Percentile')
        plt.ylabel('EF')
        plt.title('EF_perc and EF_max Curve for label ' + label_names[i])
        plt.legend(loc="lower right")
        plt.savefig(file_dir+'_curve_{}.png'.format(label_names[i]),bbox_inches='tight')
        plt.close()


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
        
    perc_vec = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
    perc_vec_plots = np.linspace(0.005, .2, 100) 
    
    metrics_dir = model_dir+'metrics.csv'
    roc_dir = model_dir+'roc_curves/'
    pr_dir = model_dir+'pr_curves/'
    efp_efm_dir = model_dir+'ef_curves/efp_efm/' 
    nef_dir = model_dir+'ef_curves/nef/' 
    
    #create directory if it doesn't exist
    dir_list = [roc_dir, pr_dir, efp_efm_dir, nef_dir]
    for file_dir in dir_list:
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
    
    nb_classes = 1    
    if len(y_true.shape) == 2:
        nb_classes = y_true.shape[1]
    else:
        y_true = y_true.reshape((y_true.shape[0], 1))
        y_pred = y_pred.reshape((y_pred.shape[0], 1)) 
        
    # dataframe metrics    
    roc_auc_df = roc_auc_multi(y_true, y_pred, range(nb_classes), np.mean, 
                               True, label_names)    
    bedroc_auc_df = bedroc_auc_multi(y_true, y_pred, range(nb_classes), np.mean, 
                               True, label_names)    
    sklearn_pr_auc_df = precision_auc_multi(y_true, y_pred, range(nb_classes), np.mean, 
                               'auc.sklearn', True, label_names)
    integral_pr_auc_df = precision_auc_multi(y_true, y_pred, range(nb_classes), np.mean, 
                               'auc.integral', True, label_names)
    dg_pr_auc_df = precision_auc_multi(y_true, y_pred, range(nb_classes), np.mean, 
                               'auc.davis.goadrich', True, label_names)
                               
    nef_pd, ef_pd, max_ef_pd = norm_enrichment_factor(y_true, y_pred, perc_vec, label_names)
    nef_auc_df = nef_auc(y_true, y_pred, perc_vec, label_names) 
    
    pr_roc_frames = [roc_auc_df, bedroc_auc_df, sklearn_pr_auc_df, 
                     integral_pr_auc_df, dg_pr_auc_df]
    pr_roc_frames = pd.concat(pr_roc_frames)
    
    pr_roc_frames.to_csv(metrics_dir)
    with open(metrics_dir,'a') as f:  
        f.write('\n')    
    for pd_df in [nef_pd, ef_pd, max_ef_pd, nef_auc_df]:
        pd_df.to_csv(metrics_dir, mode='a')
        with open(metrics_dir,'a') as f:  
            f.write('\n')  
    
    #plots
    plot_names = ['pr', 'roc', 'efp_efm', 'nef']    
    plot_curve_multi(y_true, y_pred, pr_dir, 'pr', label_names)
    plot_curve_multi(y_true, y_pred, roc_dir, 'roc', label_names)
    plot_curve_multi(y_true, y_pred, efp_efm_dir, 'efp_efm', label_names, perc_vec_plots)
    plot_curve_multi(y_true, y_pred, nef_dir, 'nef', label_names, perc_vec_plots)


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