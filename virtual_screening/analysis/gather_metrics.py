import numpy as np
import pandas as pd
import os
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.libqsturng import psturng
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import matplotlib.gridspec as gridspec
from model_names import model_name_dict 
from scipy.stats import spearmanr
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages

def model_name_dict():
    return {
    #lightchemPRAUC_1Layer1Model
    "PRAUC_5Layer1Model": "CBF_b",
    "PRAUC_10Layer1Model": "CBF_c",
    "ROCAUC_5Layer1Model": "CBF_e",
    "ROCAUC_10Layer1Model": "CBF_f",
    "ROCAUC_1Layer1Model": "CBF_d",
    "PRAUC_1Layer1Model": "CBF_a",
    #random forest
    "sklearn_rf_390014_24": "RandomForest_d",
    "sklearn_rf_390014_25": "RandomForest_e",
    "sklearn_rf_390014_97": "RandomForest_h",
    "sklearn_rf_390014_96": "RandomForest_g",
    "sklearn_rf_390014_14": "RandomForest_c",
    "sklearn_rf_390014_12": "RandomForest_a",
    "sklearn_rf_390014_13": "RandomForest_b",
    "sklearn_rf_390014_72": "RandomForest_f",
        
    "sklearn_rf_392335_24": "RandomForest_d",
    "sklearn_rf_392335_25": "RandomForest_e",
    "sklearn_rf_392335_97": "RandomForest_h",
    "sklearn_rf_392335_96": "RandomForest_g",
    "sklearn_rf_392335_14": "RandomForest_c",
    "sklearn_rf_392335_12": "RandomForest_a",
    "sklearn_rf_392335_13": "RandomForest_b",
    "sklearn_rf_392335_72": "RandomForest_f",
    #dnn
    "single_regression_2": "SingleRegression_a",
    "single_regression_11": "SingleRegression_b",
    "single_classification_22": "SingleClassification_a",
    "single_classification_42": "SingleClassification_b",
    "multi_classification_15": "MultiClassification_a",
    "multi_classification_18": "MultiClassification_b",
    "vanilla_lstm_8": "LSTM_a",
    "vanilla_lstm_19": "LSTM_b",
    #irv
    "deepchem_irv_5": "IRV_a",
    "deepchem_irv_10": "IRV_b",
    "deepchem_irv_20": "IRV_c",
    "deepchem_irv_40": "IRV_d",
    "deepchem_irv_80": "IRV_e",
    #docking
    "dockscore_hybrid": "Docking_hybrid",
    "dockscore_fred": "Docking_fred",
    "dockscore_dock6": "Docking_dock6",
    "dockscore_rdockint": "Docking_rdockint",
    "dockscore_rdocktot": "Docking_rdocktot",
    "dockscore_surflex": "Docking_surflex",
    "dockscore_ad4": "Docking_ad4",
    "dockscore_plants": "Docking_plants",
    "dockscore_smina": "Docking_smina",
    "consensus_dockscore_max": "ConsensusDocking_max",
    "consensus_bcs_efr1_opt": "ConsensusDocking_efr1_opt",
    "consensus_bcs_rocauc_opt": "ConsensusDocking_rocauc_opt",
    "consensus_dockscore_median": "ConsensusDocking_median",
    "consensus_dockscore_mean": "ConsensusDocking_mean",
    #baseline
    "baseline": "baseline"
    }
    
"""
    Gathers metrics from a directory with models using k-fold in pd.dataframe. 
    Assumes the WID Storage setup of 
    folder->fold_i->(train|val|test)_metrics->metrics.csv
"""
def gather_dir_metrics(directory, k, labels=['PriA-SSB AS','PriA-SSB FP','RMI-FANCM1'],
                       perc_vec=[0.001, 0.0015, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2],
                       n_tests_list=[100, 250, 500, 1000, 2500, 5000, 10000]):
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
                
                cols = [metric+' '+label for metric in metric_names for label in list(labels+['Mean', 'Median'])]
                
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
    for i in range(len(model_names)):
        model_names[i] = model_name_dict()[model_names[i]]
        
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
                      col_indices=list(range(10)) + list(range(15, 20)) + list(range(25, 65)) + list(range(145, 149))+ list(range(150, 183)),
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
                          col_indices=list(range(10)) + list(range(15, 20)) + list(range(25, 65)) + list(range(145, 149)) + list(range(150, 183)),
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
                        col_indices=list(range(10)) + list(range(15, 20)) + list(range(25, 65)) + list(range(145, 149)) + list(range(150, 183)),
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

def dtk_multi_metrics(gather_df, 
                      col_indices=list(range(10)) + list(range(15, 20)) + list(range(25, 65)) + list(range(145, 149)) + list(range(150, 183)),
                      alpha=0.05):
    metric_names = list(gather_df.columns.values[col_indices])
    model_names = list(gather_df.index.levels[0])
    dtk_dict = {}
    dtk_lib = rpackages.importr('DTK')
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
    
    
    index_names_1 = []
    index_names_2 = []
    for i in range(len(model_names)):
        for j in range(i+1, len(model_names)):
            index_names_1.append(model_names[j])
            index_names_2.append(model_names[i])
       
    for i, metric in zip(range(len(metric_names)), metric_names):
        m_df = gather_df[metric]
        m_df.sort_index(inplace=True)
        m_df = m_df.loc[model_names]
        m_df_mat = np.around(m_df.as_matrix(), decimals=4)
        
        dtk_results = dtk_lib.DTK_test(robjects.FloatVector(m_df_mat), robjects.FactorVector(model_names_rep), alpha)
        dtk_results = np.array(dtk_results[1])        
        dtk_pd = pd.DataFrame(data=[index_names_1, index_names_2, list(dtk_results[:,0]),list(dtk_results[:,1]),list(dtk_results[:,2]), [False for _ in range(len(index_names_1))]]).T
        dtk_pd.columns = ['group1', 'group2', 'meandiff', 'Lower CI', 'Upper CI', 'reject'] 
        
        for j in range(dtk_pd.shape[0]):      
            if dtk_pd.iloc[j,3] > 0 or dtk_pd.iloc[j,4] < 0:
                dtk_pd.iloc[j,5] = True
                
        dtk_dict[metric] = dtk_pd
    
    return dtk_dict
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
                    comp_df.loc[row['group1'],row['group2']] = 1
                else:
                    comp_df.loc[row['group2'],row['group1']] = 1
            
            reject_df.loc[row['group1'],row['group2']] = row['reject']
                
        tukey_df.columns.name = metric
        reject_df.columns.name = metric
        comp_df.columns.name = metric
        tukey_analysis_dict[metric] = (tukey_df, reject_df, comp_df)
    
    return tukey_analysis_dict

def analyze_dtk_dict(dtk_dict):
    metric_names = list(dtk_dict.keys())
    model_names = list(np.unique(list(dtk_dict[metric_names[0]]['group2']) + list(dtk_dict[metric_names[0]]['group1'])))
    dtk_analysis_dict = {}
    
    for i, metric in zip(range(len(metric_names)), metric_names):               
        dtk_df = dtk_dict[metric]
        
        reject_df = pd.DataFrame(0,index=model_names, columns=model_names)
        comp_df = pd.DataFrame(0,index=model_names, columns=model_names)
        for _, row in dtk_df.iterrows():
            if row['reject'] == True:
                if row['meandiff'] < 0:
                    comp_df.loc[row['group2'],row['group1']] = 1
                else:
                    comp_df.loc[row['group1'],row['group2']] = 1
            
            reject_df.loc[row['group1'],row['group2']] = row['reject']
                
        dtk_df.columns.name = metric
        reject_df.columns.name = metric
        comp_df.columns.name = metric
        dtk_analysis_dict[metric] = (dtk_df, reject_df, comp_df)
    
    return dtk_analysis_dict
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
        mean_comp = mm_comp_dict[metric]['Folds Mean']
        median_comp = mm_comp_dict[metric]['Folds Median']
        
        tukey_comp_df = np.zeros(mean_comp.shape)
        if tukey_analysis_dict != None:
            _, _, tukey_comp_df = tukey_analysis_dict[metric]
            
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
        ranking_list = agg_comp_dict[metric]['top'].rank(method='min', ascending=False).tolist()
        m_ordering_list = agg_comp_dict[metric]['top'].index.tolist()
        m_order_rank_list = [str(m) + ", " + str(r) for m, r in zip(m_ordering_list, ranking_list)]
        ordered_df[metric] = m_order_rank_list
        
    return ordered_df
    
"""
    Given gather_df returns a df with a column for each metric, and values of
    of rows for each column are (models, metric_score) pairs.
"""
def get_model_ordering_mscores(gather_df, metric_names, precision=4):
    ordered_df = pd.DataFrame(data=np.zeros((gather_df[metric_names[0]].xs('test_metrics', level='set').drop('fold 0', level='fold').xs('Folds Mean', level='fold').shape[0], len(metric_names))),
                              columns=metric_names,
                              dtype=str)      
    
    for i, metric in zip(range(len(metric_names)), metric_names):
        m_df = gather_df[metric].xs('test_metrics', level='set').drop('fold 0', level='fold').xs('Folds Mean', level='fold')
        m_df = m_df.sort_values(ascending=False)
        
        mscore_list = m_df.tolist()
        m_ordering_list = m_df.index.tolist()
        m_order_mscore_list = [str(m) + ", " + str(round(r,precision)) for m, r in zip(m_ordering_list, mscore_list)]
        ordered_df[metric] = m_order_mscore_list
        
    return ordered_df
    
"""
    Get spearman's rank-order correlation coefficient for between each metric's
    model ranking and that of n_hits ranking.
"""
def get_spearman_r(agg_comp_dict, metric_names, n_hits_metrics, labels=['PriA-SSB AS','PriA-SSB FP','RMI-FANCM1']):
    nh_dict = {}                        
    for j, n_hit_metric in zip(range(len(n_hits_metrics)), n_hits_metrics):              
        ranked_nh_pd = agg_comp_dict[n_hit_metric]['top'].rank(method='min', ascending=False)
        ranked_nh_list = agg_comp_dict[n_hit_metric]['top'].index.tolist()
        
        label = labels[0]
        for i in range(len(labels)):
            if labels[i] in n_hit_metric:
                label = labels[i]
                
        curr_metrics = [m for m in metric_names if label in m]
                    
        nh_metric_pd = pd.Series(0,index=curr_metrics, name=n_hit_metric)
        
        for k, metric in zip(range(len(curr_metrics)), curr_metrics):
            ranked_m_pd = agg_comp_dict[metric]['top'].rank(method='min', ascending=False)
            ranked_m_pd = ranked_m_pd.loc[ranked_nh_list]
            
            rho, pval = spearmanr(ranked_nh_pd.tolist(), ranked_m_pd.tolist())
        
            nh_metric_pd.loc[metric] = rho
            
        nh_dict[n_hit_metric] = nh_metric_pd
    
    curr_metrics = [m.replace(" " + label, "") for m in curr_metrics]    
    spearman_df = pd.DataFrame(data=np.zeros((len(nh_dict[n_hits_metrics[0]]), len(n_hits_metrics))),
                              columns=n_hits_metrics,
                              index=curr_metrics,
                              dtype=str) 
                              
    ordered_spearman_df = pd.DataFrame(data=np.zeros((len(nh_dict[n_hits_metrics[0]]), len(n_hits_metrics))),
                              columns=n_hits_metrics,
                              dtype=str)
    spearman_df[:] = ''
    ordered_spearman_df[:] = ''
    for i, n_hit_metric in zip(range(len(n_hits_metrics)), n_hits_metrics): 
        spearman_df[n_hit_metric] = nh_dict[n_hit_metric].tolist()
        metric_list = nh_dict[n_hit_metric].sort_values(ascending=False).index.tolist()
        for label in labels:
            metric_list = [m.replace(" " + label, "") for m in metric_list]    
        ordered_spearman_df[n_hit_metric] = metric_list
        
    return spearman_df, ordered_spearman_df

"""
    Compare two spearman_r dataframes and return difference beteween spearman
    correlation results, ranking difference via spearman, and sorted rank.
"""
def compare_spearman_r(spearman_df_1, spearman_df_2):
    diff_df = np.abs(spearman_df_1 - spearman_df_2)    
    ordered_diff_df = pd.DataFrame(data=np.zeros(spearman_df_2.shape),
                                  columns=list(spearman_df_2.columns),
                                  dtype=str)
    spearman_df = pd.DataFrame(data=np.zeros((1, spearman_df_2.shape[1])),
                              columns=list(spearman_df_2.columns),
                              index=['CV vs. PS'])
                              
    n_hits_metrics = list(spearman_df_2.columns)
    for j, n_hit_metric in zip(range(len(n_hits_metrics)), n_hits_metrics):
        ordered_diff_df[n_hit_metric] = diff_df[n_hit_metric].sort_values().index.tolist()
        
        rank_0 = spearman_df_1[n_hit_metric].rank(method='min', ascending=False).tolist()
        rank_1 = spearman_df_2[n_hit_metric].rank(method='min', ascending=False).tolist()
        
        rho, pval = spearmanr(rank_0, rank_1)
        spearman_df[n_hit_metric] = [rho]
    
    return diff_df, ordered_diff_df, spearman_df

def compare_cv_ps_model_ranking(df_1, df_2):
    index_names = df_1.index.tolist()
    df_2 = df_2.loc[index_names]
    spearman_df = pd.DataFrame(data=np.zeros((1, df_1.shape[1])),
                              columns=list(df_1.columns),
                              index=['CV vs. PS'])
    metrics = list(df_1.columns)
    for j, metric in zip(range(len(metrics)), metrics):
        rank_0 = df_1[metric].rank(method='min', ascending=False).tolist()
        rank_1 = df_2[metric].rank(method='min', ascending=False).tolist()
        rho, pval = spearmanr(rank_0, rank_1)
        spearman_df[metric] = [rho]
        
    return spearman_df
    
def plot_comparison_cv_ps(df_1, df_2, save_dir, figsize=(6.0, 6.0)):
    index_names = df_1.index.tolist()
    df_2 = df_2.loc[index_names]
    metrics = list(df_1.columns)
    for j, metric in zip(range(len(metrics)), metrics):
        rank_0 = df_1[metric].rank(method='min', ascending=False).tolist()
        rank_1 = df_2[metric].rank(method='min', ascending=False).tolist()
        metric_0 = metric + '_CV'
        metric_1 = metric + '_PS'
        plot_scatter(rank_0, rank_1, metric_0, metric_1, save_dir, figsize)

def plot_comparison_cv_ps_alt(df_1, df_2, save_dir, figsize=(6.0, 6.0)):
    N = 3
    fig, axs = plt.subplots(N, N, figsize=(N * 2.5, N * 2.5), sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.1)
    
    index_names = df_1.index.tolist()
    df_2 = df_2.loc[index_names]
    metrics = list(df_1.columns)
    for i, metric in zip(range(len(metrics)), metrics):
        rank_0 = df_1[metric].rank(method='min', ascending=False).tolist()
        rank_1 = df_2[metric].rank(method='min', ascending=False).tolist()
        metric_0 = metric + '_CV'
        metric_1 = metric + '_PS'
        
        k = i // N
        j = i % N
        axs[k, j].scatter(rank_1, rank_0)
        axs[k, j].set_xlim([0, len(rank_1)])
        axs[k, j].set_ylim([0, len(rank_1)])
        
        axs[k, j].axes.set_ylabel(metric_0)
        axs[k, j].axes.set_xlabel(metric_1)
        
        if j == 0:
                axs[k, j].axes.set_ylabel(metric_0)
        if k == N - 1:
            axs[k, j].axes.set_xlabel(metric_1)
                
        axs[k, j].xaxis.set_ticks_position('none')
        axs[k, j].yaxis.set_ticks_position('none')

    plt.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight')
    plt.show()
        
        
def get_model_winscores(agg_comp_dict, metric_names):
    model_names = agg_comp_dict[metric_names[0]]['top'].index.tolist()
    winscore_df = pd.DataFrame(data=np.zeros((len(model_names), len(metric_names))),
                                index=model_names,
                                columns=metric_names)        
    
    for i, metric in zip(range(len(metric_names)), metric_names):
        metric_df = agg_comp_dict[metric]['top'].loc[model_names]
        winscore_df[metric] = metric_df
        
    return winscore_df
          
"""
    Scatter plot for metrics vs n_hits.
"""
def plot_scatter_nhits(agg_comp_dict, metric_names, n_hits_metrics, save_dir, figsize=(6.0, 6.0), 
                       labels=['PriA-SSB AS','PriA-SSB FP','RMI-FANCM1']):
    for j, n_hit_metric in zip(range(len(n_hits_metrics)), n_hits_metrics):              
        ranked_nh_pd = agg_comp_dict[n_hit_metric]['top'].rank(method='min', ascending=False)
        ranked_nh_list = agg_comp_dict[n_hit_metric]['top'].index.tolist()
        
        label = labels[0]
        for i in range(len(labels)):
            if labels[i] in n_hit_metric:
                label = labels[i]
                
        curr_metrics = [m for m in metric_names if label in m]
        
        for k, metric in zip(range(len(curr_metrics)), curr_metrics):
            ranked_m_pd = agg_comp_dict[metric]['top'].rank(method='min', ascending=False)
            ranked_m_pd = ranked_m_pd.loc[ranked_nh_list]
            
            rank_0 = ranked_m_pd.tolist()
            rank_1 = ranked_nh_pd.tolist()
            metric_0 = metric
            metric_1 = n_hit_metric
            plot_scatter(rank_0, rank_1, metric_0, metric_1, save_dir, figsize)

def plot_scatter_nhits_alt(agg_comp_dict, metric_names, n_hits_metrics, save_dir, figsize=(6.0, 6.0), 
                           labels=['PriA-SSB AS','PriA-SSB FP','RMI-FANCM1']):
    N = len(metric_names)
    M = len(n_hits_metrics)
    fig, axs = plt.subplots(N, M, figsize=figsize, sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.1)
    
    for j, n_hit_metric in zip(range(len(n_hits_metrics)), n_hits_metrics):              
        ranked_nh_pd = agg_comp_dict[n_hit_metric]['top'].rank(method='min', ascending=False)
        ranked_nh_list = agg_comp_dict[n_hit_metric]['top'].index.tolist()
        
        label = labels[0]
        for i in range(len(labels)):
            if labels[i] in n_hit_metric:
                label = labels[i]
                
        curr_metrics = [m for m in metric_names if label in m]
        
        for k, metric in zip(range(len(curr_metrics)), curr_metrics):
            ranked_m_pd = agg_comp_dict[metric]['top'].rank(method='min', ascending=False)
            ranked_m_pd = ranked_m_pd.loc[ranked_nh_list]
            
            rank_0 = ranked_m_pd.tolist()
            rank_1 = ranked_nh_pd.tolist()
            metric_0 = metric
            metric_1 = n_hit_metric
            
            axs[k, j].scatter(rank_1, rank_0)
            axs[k, j].set_xlim([0, len(rank_1)])
            axs[k, j].set_ylim([0, len(rank_1)])
            
            if j == 0:
                axs[k, j].axes.set_ylabel(metric_0)
            if k == N - 1:
                axs[k, j].axes.set_xlabel(metric_1)
                
            axs[k, j].xaxis.set_ticks_position('none')
            axs[k, j].yaxis.set_ticks_position('none')

    plt.tight_layout()
    plt.savefig(save_dir, bbox_inches='tight')
    plt.show()
            
def plot_scatter(rank_0, rank_1, metric_0, metric_1, save_dir, figsize=(6.0, 6.0)):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.figure(figsize=figsize)
    plt.scatter(rank_0, rank_1)
    
    labels=['PriA-SSB AS','PriA-SSB FP','RMI-FANCM1']
    curr_label = ''
    for l in labels:
        if l in metric_0:
            curr_label = l
            
    filename = metric_1 + '_' + metric_0
    filename = filename.replace('%', '')
    filename = filename.replace(' ', '_')
    filename = filename.replace('.', '_')
    filename = save_dir+'/'+ filename +'.png'
    plt.xlim(0, len(rank_1)+1)
    plt.ylim(0, len(rank_1)+1)
    
    metric_0 = metric_0.replace(curr_label, '')
    metric_1 = metric_1.replace(curr_label, '')
            
    plt.xlabel('Metric {}'.format(metric_0))
    plt.ylabel('Metric {}'.format(metric_1))
    
    plt.title(curr_label)
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    
"""
    Given agg_comp_dict returns overlap_df with counts of overlap for each model.
"""
def get_overlap(agg_comp_dict, N=5):
    overlap_dict = {}
    metric_names = list(agg_comp_dict.keys())
    for i, metric in zip(range(len(metric_names)), metric_names):  
        top_models = agg_comp_dict[metric]['top'].loc[agg_comp_dict[metric]['top'].rank(method='min', ascending=False) <= N]
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
def get_similar_to_nhits(agg_comp_dict, metric_names, n_hits_metrics, 
                         labels=['PriA-SSB AS','PriA-SSB FP','RMI-FANCM1']):
    nh_dict = {}                        
    for j, n_hit_metric in zip(range(len(n_hits_metrics)), n_hits_metrics):              
        ranked_nh_pd = agg_comp_dict[n_hit_metric]['top'].rank(method='min', ascending=False)
        ranked_nh_list = agg_comp_dict[n_hit_metric]['top'].index.tolist()
        
        label = labels[0]
        for i in range(len(labels)):
            if labels[i] in n_hit_metric:
                label = labels[i]
                
        curr_metrics = [m for m in metric_names if label in m]
                    
        nh_metric_pd = pd.Series(0,index=curr_metrics, name=n_hit_metric)
        
        for k, metric in zip(range(len(curr_metrics)), curr_metrics):
            total_dist = 0
            ranked_m_pd = agg_comp_dict[metric]['top'].rank(method='min', ascending=False)
            
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
        ranked_nh_pd = nh_dict[n_hit_metric].rank(method='min', ascending=False)
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
        
        #ax1.set_title('Multiple Comparisons Between All Pairs (Tukey)')
        r = np.max(maxrange) - np.min(minrange)
        ax1.set_ylim([-1, tukey_hsd._multicomp.ngroups])
        ax1.set_xlim([np.min(minrange) - r / 10., np.max(maxrange) + r / 10.])
        ax1.set_yticks(np.arange(-1,tukey_hsd._multicomp.ngroups))
        ax1.set_yticklabels(np.insert(tukey_hsd.groupsunique[sorted_index[::-1]].astype(str), 0, ''))
        ax1.set_xlabel(xlabel if xlabel is not None else '')
        ax1.set_ylabel(ylabel if ylabel is not None else '')
        ax1.grid(axis='y')
        return fig

def plot_simultaneous_alt(tukey_hsd, ax=None, figsize=(10,6),
                          xlabel=None, ylabel=None):
        fig, ax1 = utils.create_mpl_ax(ax)
        if figsize is not None:
            fig.set_size_inches(figsize)
        if getattr(tukey_hsd, 'halfwidths', None) is None:
            tukey_hsd._simultaneous_ci()
        means = tukey_hsd._multicomp.groupstats.groupmean
        g_names = tukey_hsd.groupsunique.astype(str)
        
        classes = ['RandomForest', 'Docking', 'CBF', 'LSTM', 'SingleRegression', 'SingleClassification', 'MultiClassification']
        indices = []
        for c in classes:
            rep_indices = [i for i in range(len(means)) if c in g_names[i]]
            rep_means = [means[i] for i in range(len(means)) if c in g_names[i]]
            sorted_index = np.argsort(rep_means)[::-1]
            indices.append(rep_indices[sorted_index[0]])
        
        means = tukey_hsd._multicomp.groupstats.groupmean[indices]
        g_names = tukey_hsd.groupsunique[indices]
        sorted_index = np.argsort(means)[::-1]
        minrange = [means[i] - tukey_hsd.halfwidths[i] for i in sorted_index]
        maxrange = [means[i] + tukey_hsd.halfwidths[i] for i in sorted_index]
        
        ax1.errorbar(means[sorted_index], np.arange(len(means))[::-1], xerr=tukey_hsd.halfwidths[sorted_index],
                         marker='o', linestyle='None', color='k', ecolor='k')
        
        #ax1.set_title('Multiple Comparisons Between All Pairs (Tukey)')
        r = np.max(maxrange) - np.min(minrange)
        ax1.set_ylim([-1, len(sorted_index)])
        ax1.set_xlim([np.min(minrange) - r / 10., np.max(maxrange) + r / 10.])
        ax1.set_yticks(np.arange(-1,len(sorted_index)))
        ax1.set_yticklabels(np.insert(g_names[sorted_index[::-1]].astype(str), 0, ''))
        ax1.set_xlabel(xlabel if xlabel is not None else '')
        ax1.set_ylabel(ylabel if ylabel is not None else '')
        ax1.grid(axis='y')
        return fig

def plot_metrics_alt(axs, x, ml, m):
    x = np.array(x)
    ml = np.array(ml)
    means = x
    g_names = ml
    
    classes = ['RandomForest', 'Docking', 'CBF', 'LSTM', 'SingleRegression', 'SingleClassification', 'MultiClassification']
    indices = []
    for c in classes:
        rep_indices = [i for i in range(len(means)) if c in g_names[i]]
        rep_means = [means[i] for i in range(len(means)) if c in g_names[i]]
        sorted_index = np.argsort(rep_means)[::-1]
        indices.append(rep_indices[sorted_index[0]])
    
    x = x[indices]
    ml = ml[indices]
    sorted_index = np.argsort(x)[::-1]
    x = x[sorted_index]
    
    axs.scatter(x, range(x.shape[0]))
                    
    r = np.max(x) - np.min(x)
    axs.set_ylim([-1, len(x)])
    axs.set_xlim([np.min(x) - r / 10., np.max(x) + r / 10.])
    axs.set_yticks(np.arange(-1,len(x)))
    axs.set_yticklabels(np.insert(ml[sorted_index[::-1]], 0, ''))
    axs.set_xlabel(m.replace(' PriA-SSB AS',''))
    axs.grid(axis='y')
    
    return
    
"""
    Given a tukey_dict with metric names as key and TukeyHSDResults objects as
    values, plots universal confidence intervals for each model under each metric.
"""
def plot_uconf_grid(tukey_dict, metric_names, labels, save_dir, figsize=(10,6), alt=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for l in labels:
        if not os.path.exists(save_dir+l.replace(' ', '_')):
            os.makedirs(save_dir+l.replace(' ', '_'))
        
        label_metrics = [m for m in metric_names if l in m]
        
        m_list = []
        file_count = 0
        for i, metric in zip(range(len(label_metrics)), label_metrics):
            if len(m_list) == 4 or i == (len(label_metrics)-1):
                if i == (len(label_metrics)-1):
                    m_list.append((tukey_dict[metric], metric))
                fig, axs = plt.subplots(2, 2, figsize=figsize)          
                plt.subplots_adjust(wspace=0.1)
                for (j, (tukey_res, m)) in zip(range(len(m_list)), m_list):
                    ri = j // 2
                    ci = j % 2
                    if alt:
                        plot_simultaneous_alt(tukey_res, xlabel=m.replace(' '+l, ''), ax=axs[ri,ci], figsize=figsize)
                    else:
                        plot_simultaneous(tukey_res, xlabel=m.replace(' '+l, ''), ax=axs[ri,ci], figsize=figsize)
                    axs[ri,ci].set_title(l)
                plt.tight_layout()
                plt.savefig(save_dir+l.replace(' ', '_')+'/'+str(file_count)+'.png', 
                            bbox_inches='tight')
                plt.show()
                file_count += 1
                m_list = []
            m_list.append((tukey_dict[metric], metric))
    return

def plot_metrics_bp(gather_df, metric_names, labels, save_dir):
    for l in labels:
        curr_metrics = [m for m in metric_names if l in m]
        for metric in curr_metrics:
            metric_df = gather_df[metric]
            metric_df = metric_df.xs('test_metrics', level='set')
            metric_df = metric_df.drop('Folds Mean', level='fold')
            k = len(metric_df.index.levels[1]) - 2
            for i in range(k):
                metric_df = metric_df.drop('fold ' + str(i), level='fold')
            metric_df = metric_df.sort_values(ascending=False)
            boxplot_names = [m for (m,f) in metric_df.index.tolist()]

            metric_df = gather_df[metric]
            metric_df = metric_df.xs('test_metrics', level='set')
            metric_df = metric_df.drop('Folds Mean', level='fold')
            metric_df = metric_df.drop('Folds Median', level='fold')

            boxplot_data = []
            for model in boxplot_names:
                boxplot_data.append(metric_df.loc[model])

            file_name = metric.replace(' '+l, '')
            file_name = file_name.replace('%', '_')
            file_name = file_name.replace('.', '_')
            file_name = file_name.replace(' ', '_')
            if not os.path.exists(save_dir+'/'+l.replace(' ','_')+'/'):
                os.makedirs(save_dir+'/'+l.replace(' ','_')+'/')
            file_name = save_dir+'/'+l.replace(' ','_')+'/'+file_name+'.png'

            plt.figure(figsize=(30, 10))
            plt.boxplot(x=boxplot_data, labels=boxplot_names)
            plt.xticks(rotation=90)
            plt.ylabel(metric.replace(' '+l, ''))
            plt.title(metric)
            plt.tight_layout()
            plt.savefig(file_name, bbox_inches='tight')
            plt.show()

def plot_metric_grid(gather_df, save_dir, figsize=(10,6), alt=False):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gather_df = gather_df.xs('test_metrics', level='set').xs('fold 0', level='fold')
    
    m_list = []
    file_count = 0
    for i, metric in zip(range(len(gather_df.columns)), gather_df.columns):
        sorted_index = np.argsort(gather_df[metric])
        x = np.array(gather_df[metric][sorted_index])
        model_list = gather_df[metric][sorted_index].index.tolist()
        if len(m_list) == 4 or i == (len(gather_df.columns)-1):
            if i == (len(gather_df.columns)-1):
                m_list.append((x, model_list, metric))
            fig, axs = plt.subplots(2, 2, figsize=figsize)
            plt.subplots_adjust(wspace=0.1)
            for (j, (x, ml, m)) in zip(range(len(m_list)), m_list):
                ri = j // 2
                ci = j % 2
                if alt:
                    plot_metrics_alt(axs[ri,ci], x, ml, m)
                else:
                    axs[ri,ci].scatter(x, range(x.shape[0]))
                    
                    r = np.max(x) - np.min(x)
                    axs[ri,ci].set_ylim([-1, len(x)])
                    axs[ri,ci].set_xlim([np.min(x) - r / 10., np.max(x) + r / 10.])
                    axs[ri,ci].set_yticks(np.arange(-1,len(x)))
                    axs[ri,ci].set_yticklabels(np.insert(ml, 0, ''))
                    axs[ri,ci].set_xlabel(m.replace(' PriA-SSB AS',''))
                    axs[ri,ci].grid(axis='y')
                axs[ri,ci].set_title('PriA-SSB AS')
                
            plt.tight_layout()
            plt.savefig(save_dir+'/'+str(file_count)+'.png', bbox_inches='tight')
            plt.show()
            file_count += 1
            m_list = []
        m_list.append((x, model_list, metric))
    return  
"""
    Given a tukey_dict with metric names as key and TukeyHSDResults objects as
    values, plots universal confidence intervals for each model under each metric.
"""
def plot_uconf_simple(tukey_dict, metric_names, figsize=(10,6), alt=False):
    for i, metric in zip(range(len(metric_names)), metric_names):    
        fig, tukey_ax = plt.subplots()         
        tukey_res = tukey_dict[metric]
        
        if alt:
            plot_simultaneous_alt(tukey_res, xlabel=metric, ax=tukey_ax, figsize=figsize)   
        else:
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
    
    metric_names = ['ROC AUC PriA-SSB AS',
                 'BEDROC AUC PriA-SSB AS',
                 'PR auc.sklearn PriA-SSB AS',
                 'PR auc.integral PriA-SSB AS',
                 'PR auc.davis.goadrich PriA-SSB AS',
                 'NEF AUC PriA-SSB AS']
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