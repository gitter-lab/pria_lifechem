import pandas as pd
from virtual_screening.evaluation import enrichment_factor_single


docking_methods = ['dockscore_ad4', 'dockscore_dock6', 'dockscore_fred', 'dockscore_hybrid', 'dockscore_plants', 'dockscore_rdockint', 'dockscore_smina', 'dockscore_surflex', 'consensus_dockscore_mean','consensus_dockscore_STD','consensus_dockscore_median','consensus_dockscore_max','consensus_dockscore_min']

def get_ef_table(file_path='../../output/docking_result/lc123-pria-dockdata-qnorm.csv.gz',
                 efr_list = [0.02, 0.01, 0.0015, 0.001],
                 title=''):
    pria_pd = pd.read_csv(file_path)
    title = '## {}'.format(title)
    header = '| docking method | EF_2 | EF_1 | EF_015 | EF_01 |'
    splitter = '| --- |'
    for _ in efr_list:
        splitter = '{} {} |'.format(splitter, '---')

    content = ''
    for docking_method in docking_methods:
        temp_pd = pria_pd[['Unnamed: 0', 'Keck_Pria_AS_Retest', docking_method]]
        filtered_pd = temp_pd.dropna()
        true_label_list = filtered_pd['Keck_Pria_AS_Retest']
        docking_ranked_list = filtered_pd[docking_method]
        row = '| {} |'.format(docking_method)
        for ratio in efr_list:
            n_actives, ef, ef_max = enrichment_factor_single(true_label_list, docking_ranked_list, ratio)
            row = '{} {} |'.format(row, ef)
        content = '{}{}\n'.format(content, row)
    content = '{}\n{}\n{}\n{}'.format(title, header, splitter, content)
    return content