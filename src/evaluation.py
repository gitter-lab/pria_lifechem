import pandas as pd
import csv
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from sklearn.metrics import auc


def roc_auc_multi(y_true, y_pred, eval_indices, eval_mean_or_median):
    '''
    this if for multi-task evaluation
    y_true and y_pred is two-dimension matrix
    can evaluate on mean or median of array
    call by
    roc_auc_multi(y_true, y_pred, [-1], np.mean)
    roc_auc_multi(y_true, y_pred, [0], np.median)
    '''
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
    return roc_auc_score(actual, predicted)


def precision_auc_multi(y_true, y_pred, eval_indices, eval_mean_or_median):
    '''
    this if for multi-task evaluation
    y_true and y_pred is two-dimension matrix
    can evaluate on mean or median of array
    call by
    precision_auc_multi(y_true, y_pred, [-1], np.mean)
    precision_auc_multi(y_true, y_pred, [0], np.median)
    '''
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
        auc[i] = precision_auc_single(actual, predicted)
    return eval_mean_or_median(auc)


def precision_auc_single(actual, predicted):
    return average_precision_score(actual, predicted)


def enrichment_factor_multi(actual, predicted, percentile):
    EF_list = []
    for i in range(actual.shape[1]):
        n_actives, ef = enrichment_factor_single(actual[:, i], predicted[:, i], percentile)
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
            ef[i] = 1
            
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
        
    max_ef = np.zeros(nb_classes)
    sample_size = int(y_true.shape[0] * percentile)
    
    for i in range(len(max_ef)):
        true_labels = y_true[:, i]        
        n_actives = np.nansum(true_labels) 
        
        try:
            max_ef[i] = ( min(n_actives, sample_size) /  n_actives ) / percentile 
        except ValueError:
            max_ef[i] = 1
            
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
    ef_pd = pd.DataFrame(data=ef_mat,
                         index=perc_vec,
                         columns=label_names)
    return ef_pd


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
    max_ef_pd = pd.DataFrame(data=max_ef_mat,
                         index=perc_vec,
                         columns=label_names)
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
    nef_pd = pd.DataFrame(data=nef_mat,
                         index=perc_vec,
                         columns=label_names)
    return nef_pd, ef_pd, max_ef_pd


def nef_plotter_auc(y_true, y_pred, perc_vec, file_name, label_names=None):
    """
    Plots the EF_Normalized curve with notable info. Also returns the EF_Norm AUC; 
    upper bound is 1.
    If more than one label is given, draws curves for each label and 
    the mean curve. Returns a vector of auc values, one for each label.
    """
    plt.gca().set_autoscale_on(False) 
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
    for i in range(nb_classes):
        nef_auc[i] = auc(perc_vec, nef_mat[:,i])
        plt.plot(perc_vec, nef_mat[:,i], lw=lw,
             label=label_names[i] + ' (area = %0.2f)' % 
             (nef_auc[i] / max(perc_vec)))
             
    mean_nef = np.mean(nef_mat, axis=1)         
    plt.plot(perc_vec, mean_nef, color='g', linestyle='--',
         label='Mean NEF (area = %0.2f)' % 
         (auc(perc_vec, mean_nef) / max(perc_vec)), lw=lw)
    
    random_mean_nef = 1 / np.mean(ef_max_mat, axis=1)  
    plt.plot(perc_vec, random_mean_nef, linestyle='--',
         label='Random Mean NEF (area = %0.2f)' % 
         (auc(perc_vec, random_mean_nef) / max(perc_vec)), lw=lw)
        
    plt.xlim([-0.01, max(perc_vec) + 0.02])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Percentile')
    plt.ylabel('NEF')
    plt.title('Normalized EF Curve')
    plt.legend(loc="lower right")
    plt.savefig(file_name)
    
    return np.mean(nef_auc) / max(perc_vec)
    

def efp_efm_plotter(y_true, y_pred, perc_vec, file_name, label_names=None):
    """
    Plots the EF_perc+EF_max curves with notable info.
    If more than one label is given, draws curves for each label.
    """
    plt.gca().set_autoscale_on(False)
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
        plt.plot(perc_vec, ef_mat[:,i], lw=lw,
             label=label_names[i])
        plt.plot(perc_vec, max_ef_mat[:,i], lw=lw,
             label=label_names[i] + ' max')
             
    plt.xlim([0.0, max(perc_vec) + 0.01])
    plt.ylim([-0.05, np.max(max_ef_mat)+10])
    plt.xlabel('Percentile')
    plt.ylabel('EF')
    plt.title('EF_perc and EF_max Curve')
    plt.legend(loc="lower right")
    plt.savefig(file_name)
