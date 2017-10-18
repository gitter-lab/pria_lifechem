import numpy as np
import pandas as pd
import os
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
from model_names import model_name_dict 

"""
    Gathers metrics from a directory with models using k-fold in pd.dataframe. 
    Assumes the WID Storage setup of 
    folder->fold_i->(train|val|test)_metrics->metrics.csv
"""
def gather_dir_metrics(directory, k, perc_vec=[0.001, 0.0015, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                       n_tests_list=[100, 500, 1000, 2500, 5000, 10000]):
    perc_vec = ['{:g}'.format(perc * 100) + ' %' for perc in perc_vec]
    n_tests_list = ['{:g}'.format(n_tests) for n_tests in n_tests_list]

    metric_names = ['ROC AUC', 'BEDROC AUC', 'PR auc.sklearn', 'PR auc.integral',
                    'PR auc.davis.goadrich'] + \
                    ['NEF_'+ str(s) for s in perc_vec] + \
                    ['EF_'+ str(s) for s in perc_vec] + \
                    ['Max_EF_'+ str(s) for s in perc_vec] + ['NEF AUC'] + \
                    ['n_hits_'+ str(s) for s in n_tests_list]
                    
    folds = ['fold ' + str(i) for i in range(k)] + ['Folds Mean', 'Folds Median']
    fold_folders = ['train_metrics', 'val_metrics', 'test_metrics']
    model_names = os.listdir(directory)
    #gather all results in one gather_matrix                
    n_models = len(model_names)
    col_count = len(pd.read_csv(directory+'/'+model_names[0]+'/fold_0/test_metrics/metrics.csv').columns)
    col_count = (col_count-1)*len(metric_names)
    gather_matrix = np.zeros(shape=(n_models, 3, len(folds), col_count))+np.nan
    for m, m_name in zip(range(n_models), model_names):
        for f in range(k):
            fold_dir = directory+'/'+m_name+'/fold_'+str(f)+'/'
            
            for r, metrics_folder in zip(range(3), fold_folders):
                csv_file = fold_dir + metrics_folder+'/metrics.csv'
                if not os.path.isfile(csv_file):
                    continue     
                
                df = pd.read_csv(csv_file)  
                
                cols = [metric+' '+label for metric in metric_names for label in list(df.columns[1:])]
                
                df = df[(df.metric != 'NEF') & (df.metric != 'EF') & 
                        (df.metric != 'n_hits') &
                        (df.metric != 'Max_EF') & (~pd.isnull(df.metric))]
                df = df[df.columns[1:]]            
                results_arr = np.array(df, dtype=np.float)
                label_count = results_arr.shape[1]
                for results_r in range(results_arr.shape[0]):
                    gather_matrix[m, r, f, 
                        results_r*label_count:(results_r+1)*label_count] = results_arr[results_r,:] 

    #calculate means and medians
    for m, m_name in zip(range(n_models), model_names):
        for r, metrics_folder in zip(range(3), fold_folders):
            gather_matrix[m, r, k,:] = np.mean(gather_matrix[m, r, 0:k,:], axis=0)
            gather_matrix[m, r, k+1,:] = np.median(gather_matrix[m, r, 0:k,:], axis=0)
    
    #convert to pd.df with suitable namings
    cols[cols.index('NEF AUC Median')] = 'NEF AUC Random Mean' 
    m_name_dict = model_name_dict()
    for i in range(len(model_names)):
        model_names[i] = m_name_dict[model_names[i]]
        
    midx = pd.MultiIndex.from_product([model_names, fold_folders, folds],
                                      names=['model', 'set', 'fold'])
    gather_df = pd.DataFrame(data=gather_matrix.reshape(len(midx), col_count), 
                             index=midx,
                             columns=cols)
                             
    #gather_df = gather_df.dropna()
    
    return gather_df

"""
    Uses gather_df to find the top N models for each metric defined by
    col_indices. the top N models are returned for the Fold Means and Medians
    only.
    Assumes gather_df format. 
    
    Returns top_model_dict with metric names as keys and inner_dicts
    as values. inner_dicts will have two keys: 'Folds Mean' and 'Folds Median' 
    each having sorted pd.dfs of top N models.
"""
def get_top_mm_models(gather_df, 
                      col_indices=list(range(10)) + list(range(15, 20)) + list(range(25, 65)) + list(range(145, 149))+ list(range(150, 180)),
                      N=5):
    metric_names = list(gather_df.columns.values[col_indices])
    top_model_dict = {}
    k = len(gather_df.index.levels[2]) - 2
    
    #drop folds
    gather_df = gather_df[metric_names]
    gather_df = gather_df.xs('test_metrics', level='set')
    for i in range(k):
        gather_df = gather_df.drop('fold ' + str(i), level='fold')     
    
    for i, metric in zip(range(len(metric_names)), metric_names):
        m_df = gather_df[metric]
        top_model_dict[metric] = {}
        for row_name in ['Folds Mean', 'Folds Median']:
            temp_df = m_df.xs(row_name, level='fold')
            temp_df = temp_df.sort_values(ascending=False)
            temp_df = temp_df.iloc[:N]
        
            top_model_dict[metric][row_name] = temp_df
    
    return top_model_dict
    
"""
    Uses gather_df to return comp_dicts which contains 2 comparison matrices
    for mean and median for each metric.
    Assumes gather_df format. 
    
    Returns comp_dicts with metric names as keys and inner_dicts
    as values. inner_dicts will have two keys: 'Folds Mean' and 'Folds Median' 
    each having comparison matrices of scores between models.
    
    Comparison matrix values: 1 row model is better than column model.
                              0 row model is not better than column model.
    Comparison is based on raw scores.
"""
def get_mean_median_comps(gather_df, 
                          col_indices=list(range(10)) + list(range(15, 20)) + list(range(25, 65)) + list(range(145, 149)) + list(range(150, 180)),
                          tol=1e-4):
    metric_names = list(gather_df.columns.values[col_indices])
    model_names = list(gather_df.index.levels[0])
    comp_dicts = {}    
    k = len(gather_df.index.levels[2]) - 2
    
    #drop folds
    gather_df = gather_df[metric_names]
    gather_df = gather_df.xs('test_metrics', level='set')
    for i in range(k):
        gather_df = gather_df.drop('fold ' + str(i), level='fold')     
    
    for i, metric in zip(range(len(metric_names)), metric_names):
        m_df = gather_df[metric]
        comp_dicts[metric] = {}
        
        for row_name in ['Folds Mean', 'Folds Median']:
            comp_df = pd.DataFrame(0,index=model_names, columns=model_names)
            
            temp_df = m_df.xs(row_name, level='fold')
            temp_df = temp_df.sort_values(ascending=False)
            
            for j, score in temp_df.iteritems():
                for q, other_score in temp_df.loc[j:].iteritems():
                    if np.isclose(score, other_score, tol):
                        comp_df.loc[j][q] = 0
                        comp_df.loc[q][j] = 0
                    elif score > other_score:
                        comp_df.loc[j][q] = 1
                        comp_df.loc[q][j] = 0
                    else:
                        comp_df.loc[j][q] = 0
                        comp_df.loc[q][j] = 1
            
            comp_df.columns.name = metric +" " + row_name + ' comp mat'
            comp_dicts[metric][row_name] = comp_df
            
    
    return comp_dicts
    
"""
    Runs Tukey's HSD for pairwise comparison among models for each metric in 
    df[col_indices] at the alpha significance level (fwer).
    Assumes gather_df format. 
    
    Returns tukey_dict with metric names as keys and TukeyHSDResults objects 
    as values.
"""
def tukey_multi_metrics(gather_df, 
                        col_indices=list(range(10)) + list(range(15, 20)) + list(range(25, 65)) + list(range(145, 149)) + list(range(150, 180)),
                        alpha=0.05):
    metric_names = list(gather_df.columns.values[col_indices])
    model_names = list(gather_df.index.levels[0])
    tukey_dict = {}
    
    #drop fold means and medians
    gather_df = gather_df[metric_names]
    gather_df = gather_df.xs('test_metrics', level='set')
    gather_df = gather_df.drop('Folds Mean', level='fold')
    gather_df = gather_df.drop('Folds Median', level='fold')        
    
    #get fold count
    model_names_rep = []
    for m in model_names:        
        k = gather_df.xs(m, level='model').shape[0]
        model_names_rep.extend([m for _ in range(k)])
        
    for i, metric in zip(range(len(metric_names)), metric_names):
        m_df = gather_df[metric]
        
        m_df.sort_index(inplace=True)
        m_df = m_df.loc[model_names]
        
        m_df_mat = np.around(m_df.as_matrix(), decimals=4)
        mc_obj = MultiComparison(m_df_mat, model_names_rep)                   
        tukey_res = mc_obj.tukeyhsd(alpha=alpha) 
        
        tukey_dict[metric] = tukey_res
    
    return tukey_dict
    
"""
    Given a tukey_dict with metric names as key and TukeyHSDResults objects as
    values, it creates pd dataframes from the results, reject matrix and 
    comparison matrix.
    
    Rejection matrix values: 0 failure to reject null hypothesis; models are same mean
                             1 reject null hypothesis; models are not same mean
                             
    Comparison matrix values: 0 row model is not better than column model
                              1 row model is better than column model
    
    Returns tukey_analysis_dict containing triple (tukey_df, reject_df, comp_df) 
    for each metric.
"""
def analyze_tukey_dict(tukey_dict):
    metric_names = list(tukey_dict.keys())
    model_names = list(tukey_dict[metric_names[0]].groupsunique)
    tukey_analysis_dict = {}
    
    for i, metric in zip(range(len(metric_names)), metric_names):               
        tukey_res = tukey_dict[metric]
        tukey_df = pd.DataFrame(data=tukey_res._results_table.data[1:], 
                                columns=tukey_res._results_table.data[0])
        
        fwer = tukey_res._results_table.title[tukey_res._results_table.title.find('FWER'):]
        tukey_df['p value' + fwer] = psturng(np.abs(tukey_res.meandiffs / tukey_res.std_pairs), 
                                      len(tukey_res.groupsunique), tukey_res.df_total)
        
        reject_df = pd.DataFrame(0,index=model_names, columns=model_names)
        comp_df = pd.DataFrame(0,index=model_names, columns=model_names)
        for _, row in tukey_df.iterrows():
            if row['reject'] == True:
                if row['meandiff'] < 0:
                    comp_df.loc[row['group1']][row['group2']] = 1
                else:
                    comp_df.loc[row['group2']][row['group1']] = 1
            
            reject_df.loc[row['group1']][row['group2']] = row['reject']
                
        tukey_df.columns.name = metric
        reject_df.columns.name = metric
        comp_df.columns.name = metric
        tukey_analysis_dict[metric] = (tukey_df, reject_df, comp_df)
    
    return tukey_analysis_dict
    
"""
    Uses mean, median, and tukey comparison matrices to produce an aggregated
    comparison matrix. Can be used as loose-proxy for comparing models.
    
    mean_w, median_w, and tukey_w are the weights associated with each 
    comparison matrix.
    
    Returns agg_comp_dict with metrics as keys and dicts as values:
    - agg_comp is the aggregated comparison matrix for the metric
    - top_models is sorted top models based on their row sums from agg_comp for metric.
"""
def get_agg_comp(mm_comp_dict, tukey_analysis_dict,
                 mean_w=1, median_w=1, tukey_w=1):
    
    agg_comp_dict = {}
    metric_names = list(mm_comp_dict.keys())
    for i, metric in zip(range(len(metric_names)), metric_names):  
        _, _, tukey_comp_df = tukey_analysis_dict[metric]
        mean_comp = mm_comp_dict[metric]['Folds Mean']
        median_comp = mm_comp_dict[metric]['Folds Median']
        
        agg_comp = (mean_w*mean_comp) + (median_w*median_comp) + (tukey_w*tukey_comp_df)
        top_models = agg_comp.sum(axis=1) 
        top_models = top_models.sort_values(ascending=False)
        top_models.name = 'Model Score'
        
        agg_comp.columns.name = metric + ' agg mat'
        agg_comp_dict[metric] = {}
        agg_comp_dict[metric]['agg'] = agg_comp
        agg_comp_dict[metric]['top'] = top_models
    
    return agg_comp_dict


"""
    Given agg_comp_dict returns a df with a column for each metric, and values
    of rows for each column are the ordering of the models
"""
def get_model_ordering(agg_comp_dict, metric_names):
    ordered_df = pd.DataFrame(data=np.zeros((len(agg_comp_dict[metric_names[0]]['top']), len(metric_names))),
                              columns=metric_names,
                              dtype=str)    
    for i, metric in zip(range(len(metric_names)), metric_names):  
        ordered_df[metric] = agg_comp_dict[metric]['top'].index.tolist()
        
    return ordered_df

"""
    Given agg_comp_dict returns overlap_df with counts of overlap for each model.
"""
def get_overlap(agg_comp_dict, N=5):
    overlap_dict = {}
    metric_names = list(agg_comp_dict.keys())
    for i, metric in zip(range(len(metric_names)), metric_names):  
        top_models = agg_comp_dict[metric]['top'].iloc[:N]
        for model, score in top_models.iteritems():
            if model in overlap_dict.keys():
                overlap_dict[model] = overlap_dict[model]+1
            else:
                overlap_dict[model] = 1.0
        
        overlap_dict
        
    overlap_df = pd.DataFrame.from_dict(data=overlap_dict, 
                                        orient ='index',
                                        dtype=float)
    overlap_df.columns = ['overlap_perc']
    overlap_df = overlap_df/len(metric_names)
    overlap_df = overlap_df.sort_values('overlap_perc', ascending=False)
    return overlap_df
    
"""
    Given agg_comp_dict returns a df with a column for n_hits, and the rows
    are the most similar metrics from top-to-bottom.
"""
def get_similar_to_nhits(agg_comp_dict, metric_names, n_hits_metrics, labels=['Keck_Pria_AS_Retest','Keck_Pria_FP_data','Keck_RMI_cdd']):
    nh_dict = {}                        
    for j, n_hit_metric in zip(range(len(n_hits_metrics)), n_hits_metrics):              
        ranked_nh_pd = agg_comp_dict[n_hit_metric]['top'].rank(method='max')
        ranked_nh_list = agg_comp_dict[n_hit_metric]['top'].index.tolist()
        
        label = labels[0]
        for i in range(len(labels)):
            if labels[i] in n_hit_metric:
                label = labels[i]
                
        curr_metrics = [m for m in metric_names if label in m]
                    
        nh_metric_pd = pd.Series(0,index=curr_metrics, name=n_hit_metric)
        
        for k, metric in zip(range(len(curr_metrics)), curr_metrics):
            total_dist = 0
            ranked_m_pd = agg_comp_dict[metric]['top'].rank(method='max')
            
            for model in ranked_nh_list:
                candidate_pos_1 = ranked_m_pd.loc[ranked_m_pd==ranked_m_pd.loc[model]].index.tolist()
                candidate_pos_2 = ranked_nh_pd.loc[ranked_nh_pd==ranked_nh_pd.loc[model]].index.tolist()
                best_dis = 10000
                for c_pos_1 in candidate_pos_1:
                    for c_pos_2 in candidate_pos_2:
                        curr_dist = abs(ranked_m_pd.index.tolist().index(c_pos_1) - ranked_nh_list.index(c_pos_2))
                        if curr_dist < best_dis:
                            best_dis = curr_dist
                
                total_dist += best_dis
            nh_metric_pd.loc[metric] = total_dist
        
        nh_dict[n_hit_metric] = nh_metric_pd.sort_values(ascending=True)
        
    ordered_df = pd.DataFrame(data=np.zeros((len(nh_dict[n_hits_metrics[0]]), len(n_hits_metrics))),
                              columns=n_hits_metrics,
                              dtype=str)    
    ordered_df[:] = ''
    for i, n_hit_metric in zip(range(len(n_hits_metrics)), n_hits_metrics): 
        ranked_nh_pd = nh_dict[n_hit_metric].rank(method='max')
        ranked_nh_list = nh_dict[n_hit_metric].index.tolist()
        
        j=0
        for model in ranked_nh_list:
            model_list = ranked_nh_pd.loc[ranked_nh_pd==ranked_nh_pd.loc[model]].index.tolist()
            for label in labels:
                model_list = [m.replace(" " + label, "") for m in model_list]
            model_list = ", ".join(model_list)
            if not any(ordered_df[n_hit_metric].str.match(model_list)):
                ordered_df[n_hit_metric][j] = model_list
                j=j+1
        
    return ordered_df
    
"""
    Adapted from: http://nipy.bic.berkeley.edu/nightly/statsmodels/doc/html/_modules/statsmodels/sandbox/stats/multicomp.html#TukeyHSDResults.plot_simultaneous
    
    Plot a universal confidence interval of each group mean
    
    Notes
    -----
    Hochberg et al. [1] first proposed this idea and used Tukey's Q critical 
    value to compute the interval widths. Unlike plotting the differences in 
    the means and their respective confidence intervals, any two pairs can be 
    compared for significance by looking for overlap.

    References
    ----------
    .. [1] Hochberg, Y., and A. C. Tamhane. Multiple Comparison Procedures.
           Hoboken, NJ: John Wiley & Sons, 1987.
"""
from statsmodels.graphics import utils
def plot_simultaneous(tukey_hsd, ax=None, figsize=(10,6),
                          xlabel=None, ylabel=None):
        fig, ax1 = utils.create_mpl_ax(ax)
        if figsize is not None:
            fig.set_size_inches(figsize)
        if getattr(tukey_hsd, 'halfwidths', None) is None:
            tukey_hsd._simultaneous_ci()
        means = tukey_hsd._multicomp.groupstats.groupmean
        sorted_index = np.argsort(means)[::-1]
        
        minrange = [means[i] - tukey_hsd.halfwidths[i] for i in sorted_index]
        maxrange = [means[i] + tukey_hsd.halfwidths[i] for i in sorted_index]
        
        
        ax1.errorbar(means[sorted_index], np.arange(len(means))[::-1], xerr=tukey_hsd.halfwidths[sorted_index],
                         marker='o', linestyle='None', color='k', ecolor='k')
        
        ax1.set_title('Multiple Comparisons Between All Pairs (Tukey)')
        r = np.max(maxrange) - np.min(minrange)
        ax1.set_ylim([-1, tukey_hsd._multicomp.ngroups])
        ax1.set_xlim([np.min(minrange) - r / 10., np.max(maxrange) + r / 10.])
        ax1.set_yticks(np.arange(-1,tukey_hsd._multicomp.ngroups))
        ax1.set_yticklabels(np.insert(tukey_hsd.groupsunique[sorted_index[::-1]].astype(str), 0, ''))
        ax1.set_xlabel(xlabel if xlabel is not None else '')
        ax1.set_ylabel(ylabel if ylabel is not None else '')
        plt.grid(axis='y')
        return fig
        
"""
    Given a tukey_dict with metric names as key and TukeyHSDResults objects as
    values, plots universal confidence intervals for each model under each metric.
"""
def plot_uconf_grid(tukey_dict, metric_names):
    fig = plt.figure()
    for i, metric in zip(range(len(metric_names)), metric_names):               
        tukey_res = tukey_dict[metric]
        
        tukey_ax = fig.add_subplot(1,len(metric_names),i+1)
        plot_simultaneous(tukey_res, xlabel=metric, ax=tukey_ax)    
    return
    
"""
    Given a tukey_dict with metric names as key and TukeyHSDResults objects as
    values, plots universal confidence intervals for each model under each metric.
"""
def plot_uconf_simple(tukey_dict, metric_names, figsize=(10,6)):
    for i, metric in zip(range(len(metric_names)), metric_names):    
        fig, tukey_ax = plt.subplots()         
        tukey_res = tukey_dict[metric]
        
        plot_simultaneous(tukey_res, xlabel=metric, ax=tukey_ax, figsize=figsize)    
    return
    
"""
    Given a tukey_dict with metric names as key and TukeyHSDResults objects as
    values, plots universal confidence intervals for each model under each metric.
    
    This one has a slider for easily seeing multiple plots at the same time
    without clutter.
"""
def plot_uconf_slider(tukey_dict, metric_names):
    gs = gridspec.GridSpec(2, 1,
                           width_ratios=[1],
                           height_ratios=[20, 1]
                           )
                           
    fig = plt.figure()
    metric = metric_names[0]        
    tukey_res = tukey_dict[metric]
    tukey_ax = plt.subplot(gs[0])
    tukey_ax.relim()
    tukey_ax.autoscale_view()
    tukey_ax.set_aspect('auto')
    plot_simultaneous(tukey_res, xlabel=metric, ax=tukey_ax, figsize=fig.get_size_inches())  
    
    axframe = plt.subplot(gs[1])
    sframe = Slider(axframe, 'Frame', 0, len(metric_names)-1, valinit=0,valfmt='%d')
    gs.update(left=0.25, hspace=0.25)     
    
    def update(val):
        tukey_ax.clear()
        frame = int(np.floor(sframe.val))
        metric = metric_names[frame]        
        tukey_res = tukey_dict[metric]
        plot_simultaneous(tukey_res, xlabel=metric, ax=tukey_ax, figsize=fig.get_size_inches())   
          
        
    sframe.on_changed(update)

    return
    
    
"""
    Finds the top 5 Sklearn RandomForests based only on the val_metrics.
"""
def get_best_skrf(directory, k=5, N=5):
    gather_df = gather_dir_metrics(directory, k)
    
    metric_names = ['ROC AUC Keck_Pria_AS_Retest',
                 'BEDROC AUC Keck_Pria_AS_Retest',
                 'PR auc.sklearn Keck_Pria_AS_Retest',
                 'PR auc.integral Keck_Pria_AS_Retest',
                 'PR auc.davis.goadrich Keck_Pria_AS_Retest',
                 'NEF AUC Keck_Pria_AS_Retest']
    gather_df = gather_df.xs('val_metrics', level='set')    
    gather_df = gather_df.xs('fold 0', level='fold')    
    gather_df = gather_df[metric_names]
    
    skrf_df = pd.DataFrame(data='', columns=metric_names, index=range(N))
    overlap_dict = {}
    for i, metric in zip(range(len(metric_names)), metric_names):
        temp_df = gather_df[metric].sort_values(ascending=False)
        temp_df = temp_df.iloc[:N]
        skrf_df[metric] = list(temp_df.index)
        
        for model in list(temp_df.index):
           if model in overlap_dict.keys():
                overlap_dict[model] = overlap_dict[model]+1
           else:
                overlap_dict[model] = 1.0
    
    overlap_df = pd.DataFrame.from_dict(data=overlap_dict, 
                                        orient ='index',
                                        dtype=float)
    overlap_df.columns = ['overlap_perc']
    overlap_df = overlap_df/len(metric_names)
    overlap_df = overlap_df.sort_values('overlap_perc', ascending=False)
    
    return skrf_df, overlap_df