import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold


'''
input_file is whole 
output_file_list is the files that we will split data into
Make sure the output_file_list length is equal to k
'''
def split_data(input_file, output_file_list, k):
    data_pd = pd.read_csv(input_file)
    y_data = data_pd['true_label']
    y_data = y_data.astype(np.float64)
    if y_data.ndim == 1:
        n = y_data.shape[0]
        y_data = y_data.reshape(n, 1)

    cnt = 0
    split = StratifiedKFold(y_data[:, -1], n_folds=k, shuffle=True, random_state=0)
    for train_index, test_index in split:
        # For testing
        # Can list all existing active ones
        # data_batch[data_batch['true_label']>0]['molecule ID(RegID)']
        data_batch = data_pd.iloc[test_index]
        data_batch.to_csv(output_file_list[cnt], index_label=None, compression='gzip')
        cnt += 1
    return


'''
Read the data from all files in input_file_list
And merged into one dataset
'''
def read_merged_data(input_file_list):
    whole_pd = pd.DataFrame()
    for input_file in input_file_list:
        data_pd = pd.read_csv(input_file, dtype=str)
        whole_pd = whole_pd.append(data_pd)
    return whole_pd


'''
Get the fingerprints, with feature_name specified, and label_name specified
'''
def extract_feature_and_label(data_pd,
                              feature_name='1024_fingerprint',
                              label_name='true_label'):
    X_data = np.zeros(shape=(data_pd.shape[0], 1024))
    y_data = np.zeros(shape=(data_pd.shape[0], 1))
    for index, row in data_pd.iterrows():
        feature = list(row[feature_name])
        label = row[label_name]
        X_data[index] = np.array(feature)
        y_data[index] = label
    X_data = X_data.astype(float)
    y_data = y_data.astype(float)

    # In case we just train on one target
    # y would be (n,) vector
    # then we should change it to (n,1) 1D matrix
    # to keep consistency
    if y_data.ndim == 1:
        n = y_data.shape[0]
        y_data = y_data.reshape(n, 1)

    return X_data, y_data
