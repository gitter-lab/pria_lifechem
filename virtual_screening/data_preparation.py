import pandas as pd
import numpy as np
from rdkit.Chem import AllChem
from rdkit import Chem
import operator
import json
import os


'''
Transform the data from raw xlsx into csv format
'''
def transform_data(output_file_name):
    discrete_file = pd.ExcelFile('../dataset/data_preprocessing/screening_smsf_actives_2017_03_10.xlsx')
    Keck_Pria_Retest = discrete_file.parse('Keck_Pria_Retest')
    Keck_Pria_FP = discrete_file.parse('Keck_Pria_FP')
    Keck_RMI = discrete_file.parse('Keck_RMI')
    Xing_MTDH_Retest = discrete_file.parse('Xing_MTDH_Retest')
    Xing_MTDH_DR = discrete_file.parse('Xing_MTDH_DR')
    
    continuous_file = pd.ExcelFile('../dataset/data_preprocessing/screening_smsf_continuous_2017_03_10.xlsx')
    Keck_Pria_Primary = continuous_file.parse('Keck_Pria_Primary')
    Keck_RMI_cdd = continuous_file.parse('Keck_RMI_cdd')
    Xing_MTDH_cdd = continuous_file.parse('Xing_MTDH_cdd')
    
    f = open('../dataset/data_preprocessing/lifechem123_cleaned_2017_03_10.smi', 'r')
    mol_smile_dict = {}
    mol_fps_dict = {}

    for line in f:
        line = line.strip()
        row = line.split(' ')
        smiles = row[0]
        molecule_id = row[1]
        
        # Get SMILE
        mol_smile_dict[molecule_id] = smiles
        
        # Get molecule descriptor from SMILES
        # Then generate Fingerprints from molecule descriptor
        mol = Chem.MolFromSmiles(smiles)
        fingerprints = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
        mol_fps_dict[molecule_id] = fingerprints.ToBitString()
        
    mol_smile_map = sorted(mol_smile_dict.items())
    molecules = [item[0] for item in mol_smile_map]
    smiles = [item[1] for item in mol_smile_map]
    smiles_df = pd.DataFrame({'Molecule': molecules, 'SMILES': smiles})
    
    molecules = [key for key,_ in mol_fps_dict.iteritems()]
    fingerprints = [value for _,value in mol_fps_dict.iteritems()]
    fingerprints_df = pd.DataFrame({'Molecule': molecules, 'Fingerprints': fingerprints})
    
    result = pd.merge(smiles_df, fingerprints_df, on='Molecule', how='outer')
    result = pd.merge(result, Keck_Pria_Retest, on='Molecule', how='outer')
    result = pd.merge(result, Keck_Pria_FP, on='Molecule', how='outer')
    result = pd.merge(result, Keck_Pria_Primary, on='Molecule', how='outer')
    result = pd.merge(result, Keck_RMI, on='Molecule', how='outer')
    result = pd.merge(result, Keck_RMI_cdd, on='Molecule', how='outer')

    result.to_csv(output_file_name, index=None)

    return


'''
Create sample weights, for weighted model
'''
def generate_sample_weights(target_dir, k, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for i in range(k):
        data_pd = pd.read_csv(target_dir + 'file_{}.csv'.format(i))

        # First three labels are Molecule, SMILES, and Finterprints
        # The remaining ones are labels
        labels = data_pd.columns.tolist()[3:]
        data_pd = data_pd.replace([0], 1)
        data_pd = data_pd.replace([np.NaN], 0)
        data_pd.to_csv(dest_dir + 'sample_weight_{}.csv'.format(i), index=None)

    return


'''
Get the mappings for SMILES
Pre-train this for only once
'''
fixed_SMILES_mapping_json_file = '../json/SMILES_mapping.json'
def mapping_SMILES(data_pd, json_file=fixed_SMILES_mapping_json_file):
    dictionary_set = set()
    for smile in data_pd['SMILES']:
        dictionary_set = dictionary_set | set(list(smile))
    index = 1
    dictionary = {}
    for element in dictionary_set:
        dictionary[element] = index
        index += 1
    print dictionary
    print 'alphabet set size {}'.format(len(dictionary))

    with open(json_file, 'w') as f:
        json.dump(dictionary, f)

    return


"""
    splits pcba dataset into k folds using a greedy, smallest actives first
    approach that achieves good splitting across labels.
"""


def split_pcba_into_folds(data_dir, k, dest_dir):
    nb_classes = 128

    pcba_df = pd.read_csv(data_dir)
    pcba_df = pcba_df.sample(frac=1).reset_index(drop=True)  # shuffle rows
    pcba_df = pcba_df.sample(frac=1).reset_index(drop=True)  # shuffle rows
    pcba_df = pcba_df.sample(frac=1).reset_index(drop=True)  # shuffle rows
    pcba_df = pcba_df.sample(frac=1).reset_index(drop=True)  # shuffle rows

    label_names = pcba_df.columns[3:]
    total_compounds = pcba_df.shape[0]

    num_actives_list = list()
    for i in range(nb_classes):
        curr_label = np.array(pcba_df[label_names[i]])
        actives_count = curr_label[curr_label == 1].shape[0]
        num_actives_list.append(actives_count)
    sorted_label_indexes = np.argsort(num_actives_list).tolist()[::-1]

    fold_size = total_compounds // k
    fold_indexes = [np.array([], dtype=np.int64) for _ in range(k)]
    for i in sorted_label_indexes:
        curr_label = np.array(pcba_df[label_names[i]])

        # actives
        active_indexes = np.where(curr_label == 1)[0]
        kf = KFold(len(active_indexes), n_folds=k, shuffle=False)
        for fi, (_, test_index) in zip(range(k), kf):
            fold_indexes[fi] = np.append(fold_indexes[fi], active_indexes[test_index])

        # inactives
        inactive_indexes = np.where(curr_label == 0)[0]
        kf = KFold(len(inactive_indexes), n_folds=k, shuffle=False)
        for fi, (_, test_index) in zip(range(k), kf):
            fold_indexes[fi] = np.append(fold_indexes[fi], inactive_indexes[test_index])

        # missing
        # TODO: For PCBA_missing, it's fine. But what if no missing data?
        missing_indexes = np.where(curr_label == -1)[0]
        kf = KFold(len(missing_indexes), n_folds=k, shuffle=False)
        for fi, (_, test_index) in zip(range(k), kf):
            fold_indexes[fi] = np.append(fold_indexes[fi], missing_indexes[test_index])

        # now uniquify the indexes in each fold
        for ki in range(k):
            fold_indexes[ki] = np.unique(fold_indexes[ki])

        for ki in np.random.permutation(k):
            for kj in np.random.permutation(k):
                if ki == kj:
                    continue
                remove_indexes = np.where(np.in1d(fold_indexes[kj], fold_indexes[ki]))
                print ki, ' ', kj, ' ', 'remove ', remove_indexes[0].shape
                fold_indexes[kj] = np.delete(fold_indexes[kj], remove_indexes)
        print

    # now uniquify the indexes in each fold
    for i in range(k):
        fold_indexes[i] = np.unique(fold_indexes[i])

    # check if any folds have overlapping rows
    for i in range(k):
        for j in range(k):
            if i != j and np.any(np.in1d(fold_indexes[i], fold_indexes[j])):
                print('Found overlapping indexes!!!')

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

        # save label statistics for each fold
    label_stat_columns = list()
    for i in range(len(label_names)):
        label_stat_columns.extend([label_names[i] + '_actives',
                                   label_names[i] + '_inactives',
                                   label_names[i] + '_missing'])

    index_names = ['fold_' + str(i) for i in range(k)]
    index_names.extend(['mean', 'stdev'])
    label_stats_df = pd.DataFrame(data=np.zeros((k + 2, len(label_stat_columns))),
                                  columns=label_stat_columns, dtype=np.int64,
                                  index=index_names)
    for i in range(k):
        curr_fold_df = pcba_df.iloc[fold_indexes[i], :]
        for j, label in zip(range(len(label_names)), label_names):
            curr_label = np.array(curr_fold_df[label])

            ind = 3 * j
            label_stats_df.iloc[i][ind] = curr_label[curr_label == 1].shape[0]
            label_stats_df.iloc[i][ind + 1] = curr_label[curr_label == 0].shape[0]
            label_stats_df.iloc[i][ind + 2] = curr_label[curr_label == -1].shape[0]

    label_stats_df.iloc[-2] = np.mean(label_stats_df.iloc[0:k][:])
    label_stats_df.iloc[-1] = np.std(label_stats_df.iloc[0:k][:])

    label_stats_df.to_csv(dest_dir + 'label_fold_stats.csv')
    with open(dest_dir + 'label_fold_stats.csv', 'a') as f:
        f.write('\n')
        s = 0
        for i in range(k):
            f.write('fold_' + str(i) + ' size:,' + str(len(fold_indexes[i])) + '\n')
            s = s + len(fold_indexes[i])
        f.write('sum of all folds:,' + str(s))

    # create the fold csv files
    pcba_df.replace(to_replace=-1, value='', inplace=True)
    cols = pcba_df.columns.values
    cols[0] = 'Molecule'
    cols[1] = 'SMILES'
    cols[2] = 'Fingerprints'
    pcba_df.columns = cols
    pcba_df['Molecule'] = 'PCBA-' + pcba_df['Molecule'].astype(str)

    file_list = []
    for i in range(k):
        file_list.append(dest_dir + 'file_{}.csv'.format(i))

    for i in range(k):
        pcba_df.iloc[fold_indexes[i], :].to_csv(file_list[i], index=None)


"""
    merges keck folds with pcba folds in a straightforward manner.
"""
def merge_keck_pcba(keck_dir, pcba_dir, k, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    # now perform merging of folds
    overlap_counts = list()
    for i in range(k):
        keck_fold_df = pd.read_csv(keck_dir + 'file_{}.csv'.format(i))
        pcba_fold_df = pd.read_csv(pcba_dir + 'file_{}.csv'.format(i))
        merged_fold_df = pd.merge(keck_fold_df, pcba_fold_df, how='outer',
                                  on='SMILES', indicator=True)
        overlap_counts.append(merged_fold_df[merged_fold_df._merge == 'both'].shape[0])
        merged_fold_df.Fingerprints_x[pd.isnull(merged_fold_df.Fingerprints_x)] = merged_fold_df.Fingerprints_y[
            pd.isnull(merged_fold_df.Fingerprints_x)]
        merged_fold_df['Fingerprints'] = merged_fold_df.Fingerprints_x

        merged_fold_df.replace(to_replace=np.nan, value='', inplace=True)
        merged_fold_df['Molecule'] = merged_fold_df.Molecule_x + '_' + merged_fold_df.Molecule_y

        label_names = keck_fold_df.columns[3:].tolist() + pcba_fold_df.columns[3:].tolist()
        cols = ['Molecule', 'SMILES', 'Fingerprints'] + label_names
        merged_fold_df = merged_fold_df[cols]
        merged_fold_df.to_csv(dest_dir + 'file_{}.csv'.format(i), index=None)

    keck_fold_df = pd.read_csv(keck_dir + 'file_{}.csv'.format(0))
    pcba_fold_df = pd.read_csv(pcba_dir + 'file_{}.csv'.format(0))
    label_names = keck_fold_df.columns[3:].tolist() + pcba_fold_df.columns[3:].tolist()

    # save label statistics for each fold
    label_stat_columns = list()
    label_names.remove('Keck_Pria_Continuous')
    label_names.remove('FP counts % inhibition')
    for i in range(len(label_names)):
        label_stat_columns.extend([label_names[i] + '_actives',
                                   label_names[i] + '_inactives',
                                   label_names[i] + '_missing'])

    index_names = ['fold_' + str(i) for i in range(k)]
    index_names.extend(['mean', 'stdev'])
    label_stats_df = pd.DataFrame(data=np.zeros((k + 2, len(label_stat_columns))),
                                  columns=label_stat_columns, dtype=np.int64,
                                  index=index_names)
    fold_sizes = [0 for _ in range(k)]
    for i in range(k):
        curr_fold_df = pd.read_csv(dest_dir + 'file_{}.csv'.format(i))
        for j, label in zip(range(len(label_names)), label_names):
            curr_label = np.array(curr_fold_df[label])
            fold_sizes[i] = curr_label.shape[0]

            ind = 3 * j
            label_stats_df.iloc[i][ind] = curr_label[curr_label == 1].shape[0]
            label_stats_df.iloc[i][ind + 1] = curr_label[curr_label == 0].shape[0]
            label_stats_df.iloc[i][ind + 2] = np.where(np.isnan(curr_label))[0].shape[0]

    label_stats_df.iloc[-2] = np.mean(label_stats_df.iloc[0:k][:])
    label_stats_df.iloc[-1] = np.std(label_stats_df.iloc[0:k][:])

    label_stats_df.to_csv(dest_dir + 'label_fold_stats.csv')
    with open(dest_dir + 'label_fold_stats.csv', 'a') as f:
        f.write('\n')
        s_size = 0
        s_overlap = 0
        f.write(',size,overlap_count\n')
        for i in range(k):
            f.write('fold_' + str(i) + ',' + str(fold_sizes[i]) + ',' + str(overlap_counts[i]) + '\n')
            s_size = s_size + fold_sizes[i]
            s_overlap = s_overlap + overlap_counts[i]
        f.write('sum :,' + str(s_size) + ',' + str(s_overlap))


if __name__ == '__main__':
    transform_data('../dataset/keck_complete.csv')