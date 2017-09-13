import pandas as pd
import numpy as np
from virtual_screening.evaluation import precision_auc_single, roc_auc_single, bedroc_auc_single, \
    enrichment_factor_single
from virtual_screening.function import reshape_data_into_2_dim
from sklearn import metrics


function_mapping = {'precision_auc_single': precision_auc_single,
                    'roc_auc_single': roc_auc_single,
                    'bedroc_auc_single': bedroc_auc_single}

docking_methods = ['dockscore_ad4', 'dockscore_dock6', 'dockscore_fred', 'dockscore_hybrid',
                   'dockscore_plants', 'dockscore_rdockint', 'dockscore_rdocktot', 'dockscore_smina', 'dockscore_surflex',
                   'consensus_dockscore_mean', 'consensus_dockscore_STD', 'consensus_dockscore_median',
                   'consensus_dockscore_max', 'consensus_dockscore_min']

docking_methods = ['consensus_bcs_efr1_opt', 'consensus_bcs_rocauc_opt',
                   'consensus_dockscore_max', 'consensus_dockscore_mean', 'consensus_dockscore_median',
                   'dockscore_ad4', 'dockscore_dock6', 'dockscore_fred', 'dockscore_hybrid',
                   'dockscore_plants', 'dockscore_rdockint', 'dockscore_rdocktot',
                   'dockscore_smina', 'dockscore_surflex']


def get_auc_table(file_path, target_name, auc_list, auc_header, title):
    pria_pd = pd.read_csv(file_path)
    title = '## {}'.format(title)

    header = '| docking method |'
    for name in auc_header:
        header = '{} {} |'.format(header, name)

    splitter = '| --- |'
    for _ in auc_header:
        splitter = '{} {} |'.format(splitter, '---')

    content = ''

    if target_name == 'Keck_Pria_AS_Retest':
        ground = '../../output/docking/stage_1/lc123-pria-dockdata-qnorm.csv.gz'
    elif target_name == 'Keck_Pria_FP_data':
        ground = '../../output/docking/stage_1/lc123-pria-dockdata-qnorm.csv.gz'
    elif target_name == 'Keck_RMI_cdd':
        ground = '../../output/docking/stage_1/lc123-rmi-dockdata-qnorm.csv.gz'
    else:
        raise ValueError('Target name {} not found.'.format(target_name))

    ground_pd = pd.read_csv(ground)
    ground_pd = ground_pd[['Unnamed: 0', target_name]]
    ground_pd.columns = ['molid', target_name]
    pria_pd = pd.merge(pria_pd, ground_pd, on='molid', how='outer')


    for docking_method in docking_methods:
        # temp_pd = pria_pd[['Unnamed: 0', target_name, docking_method]]
        temp_pd = pria_pd[['molid', target_name, docking_method]]
        filtered_pd = temp_pd.dropna()
        true_label_list = filtered_pd[target_name].tolist()
        docking_ranked_list = filtered_pd[docking_method].tolist()
        true_label_array = reshape_data_into_2_dim(np.array(true_label_list))
        docking_ranked_array = reshape_data_into_2_dim(np.array(docking_ranked_list))
        row = '| {} |'.format(docking_method)

        for auc_method_name in auc_list:
            auc_method = function_mapping[auc_method_name]
            auc = auc_method(true_label_array, docking_ranked_array)
            row = '{} {:.6f} |'.format(row, auc)
        content = '{}{}\n'.format(content, row)
    content = '{}\n{}\n{}\n{}'.format(title, header, splitter, content)

    return content


def get_ef_table(file_path, target_name, efr_list, ef_header, title):
    """
    :param file_path: Docking results
    :param efr_list: EF ratio list
    :param ef_header: Table header
    :param title: Markdown Table caption
    :return: the markdown content

    example run: get_ef_table(file_path='../../output/docking_result/lc123-pria-dockdata-qnorm.csv.gz',
                              target_name='Keck_Pria_AS_Retest',
                              efr_list=[0.02, 0.01, 0.0015, 0.001],
                              ef_header=['EF_2', 'EF_1', 'EF_015', 'EF_01'],
                              title='Enrichment Factor for Docking Methods')
    """
    pria_pd = pd.read_csv(file_path)
    title = '## {}'.format(title)

    header = '| docking method |'
    for name in ef_header:
        header = '{} {} |'.format(header, name)

    splitter = '| --- |'
    for _ in efr_list:
        splitter = '{} {} |'.format(splitter, '---')

    if target_name == 'Keck_Pria_AS_Retest':
        ground = '../../output/docking/stage_1/lc123-pria-dockdata-qnorm.csv.gz'
    elif target_name == 'Keck_Pria_FP_data':
        ground = '../../output/docking/stage_1/lc123-pria-dockdata-qnorm.csv.gz'
    elif target_name == 'Keck_RMI_cdd':
        ground = '../../output/docking/stage_1/lc123-rmi-dockdata-qnorm.csv.gz'
    else:
        raise ValueError('Target name {} not found.'.format(target_name))
    
    ground_pd = pd.read_csv(ground)
    ground_pd = ground_pd[['Unnamed: 0', target_name]]
    ground_pd.columns = ['molid', target_name]
    pria_pd = pd.merge(pria_pd, ground_pd, on='molid', how='outer')

    content = ''
    for docking_method in docking_methods:
        # temp_pd = pria_pd[['Unnamed: 0', target_name, docking_method]]
        temp_pd = pria_pd[['molid', target_name, docking_method]]
        filtered_pd = temp_pd.dropna()
        # TODO: may find the difference with panda.series for EF calculation
        # true_label_list = filtered_pd[target_name]
        # docking_ranked_list = filtered_pd[docking_method]
        true_label_list = np.array(filtered_pd[target_name].tolist())
        docking_ranked_list = np.array(filtered_pd[docking_method].tolist())
        row = '| {} |'.format(docking_method)
        for ratio in efr_list:
            n_actives, ef, ef_max = enrichment_factor_single(true_label_list, docking_ranked_list, ratio)
            row = '{} {} |'.format(row, ef)
        content = '{}{}\n'.format(content, row)
    content = '{}\n{}\n{}\n{}'.format(title, header, splitter, content)
    return content