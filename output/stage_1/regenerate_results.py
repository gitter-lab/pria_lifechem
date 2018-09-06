import numpy as np
import pria_lifechem
import os
from pria_lifechem.function import *
from pria_lifechem.evaluation import *
from pria_lifechem.models.CallBacks import *
from pria_lifechem.models.deep_classification import *
from pria_lifechem.models.deep_regression import *
from pria_lifechem.models.vanilla_lstm import *
from pria_lifechem.models.tree_net import *


model_name_list = (
    'single_classification_22',
    'single_classification_42',
    'single_regression_2',
    'single_regression_11',
    'multi_classification_15',
    'multi_classification_18'
)


task_list = ('cross_validation_Keck_Pria_AS_Retest', 'cross_validation_Keck_FP', 'cross_validation_RMI')


record = {
    ('single_classification_22', 'cross_validation_Keck_Pria_AS_Retest'): 45540945,
    ('single_classification_42', 'cross_validation_Keck_Pria_AS_Retest'): 45539118,
    ('single_regression_2', 'cross_validation_Keck_Pria_AS_Retest'): 45540957,
    ('single_regression_11', 'cross_validation_Keck_Pria_AS_Retest'): 45540958,
    ('multi_classification_15', 'cross_validation_Keck_Pria_AS_Retest'):45983730,
    ('multi_classification_18', 'cross_validation_Keck_Pria_AS_Retest'): 45983774,

    ('single_classification_22', 'cross_validation_Keck_FP'): 45710870,
    ('single_classification_42', 'cross_validation_Keck_FP'): 45710871,
    ('single_regression_2', 'cross_validation_Keck_FP'): 45710874,
    ('single_regression_11', 'cross_validation_Keck_FP'): 45710875,
    ('multi_classification_15', 'cross_validation_Keck_FP'): 46007208,
    ('multi_classification_18', 'cross_validation_Keck_FP'): 46029086,

    ('single_classification_22', 'cross_validation_RMI'): 46244865,
    ('single_classification_42', 'cross_validation_RMI'): 46101892,
    ('single_regression_2', 'cross_validation_RMI'): 46245031,
    ('single_regression_11', 'cross_validation_RMI'): 46245032,
    ('multi_classification_15', 'cross_validation_RMI'): 46245054,
    ('multi_classification_18', 'cross_validation_RMI'): 46245048,
}

fold_num = 20


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_list', dest="model_name_list", action="store", required=True)
    parser.add_argument('--task', dest="task", action="store", required=True)
    parser.add_argument('--fold_idx', dest="fold_idx", action="store", type=int, required=True)
    given_args = parser.parse_args()

    model_name = given_args.model_name_list
    task_name = given_args.task
    fold_idx = given_args.fold_idx

    weight_file = '{}/{}/{}/{}.weight'.format(task_name, model_name, record[model_name, task_name], fold_idx)
    print os.path.isfile(weight_file)

    k = 5
    directory = '../../dataset/fixed_dataset/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('{}file_{}.csv'.format(directory, i))
    file_list = np.array(file_list)

    test_index = fold_idx / 4
    val_index = fold_idx % 4 + (fold_idx % 4 >= test_index)
    complete_index = np.arange(k)
    train_index = np.where((complete_index != test_index) & (complete_index != val_index))[0]
    print train_index

    train_file_list = file_list[train_index]
    val_file_list = file_list[val_index:val_index + 1]
    test_file_list = file_list[test_index:test_index + 1]

    if 'single_classification' in model_name:
        with open('../../json/single_classification_keck_pria_retest.json', 'r') as f:
            conf = json.load(f)
        label_name_list = conf['label_name_list']

        train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
        val_pd = filter_out_missing_values(read_merged_data(val_file_list), label_list=label_name_list)
        test_pd = filter_out_missing_values(read_merged_data(test_file_list), label_list=label_name_list)

        # extract data, and split training data into training and val
        X_train, y_train = extract_feature_and_label(train_pd,
                                                     feature_name='Fingerprints',
                                                     label_name_list=label_name_list)
        X_val, y_val = extract_feature_and_label(val_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=label_name_list)
        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=label_name_list)
        task = SingleClassification(conf=conf)
        task.predict_with_existing(X_train, y_train,
                                   X_val, y_val,
                                   X_test, y_test,
                                   weight_file)

    elif 'single_regression' in model_name:
        with open('../../json/single_regression_keck_pria_retest.json', 'r') as f:
            conf = json.load(f)
        label_name_list = conf['label_name_list']

        train_pd = filter_out_missing_values(read_merged_data(train_file_list), label_list=label_name_list)
        val_pd = filter_out_missing_values(read_merged_data(val_file_list), label_list=label_name_list)
        test_pd = filter_out_missing_values(read_merged_data(test_file_list), label_list=label_name_list)

        # extract data, and split training data into training and val
        X_train, y_train = extract_feature_and_label(train_pd,
                                                     feature_name='Fingerprints',
                                                     label_name_list=label_name_list)
        X_val, y_val = extract_feature_and_label(val_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=label_name_list)
        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=label_name_list)

        y_train_classification = reshape_data_into_2_dim(y_train[:, 0])
        y_train_regression = reshape_data_into_2_dim(y_train[:, 1])
        y_val_classification = reshape_data_into_2_dim(y_val[:, 0])
        y_val_regression = reshape_data_into_2_dim(y_val[:, 1])
        y_test_classification = reshape_data_into_2_dim(y_test[:, 0])
        y_test_regression = reshape_data_into_2_dim(y_test[:, 1])

        task = SingleRegression(conf=conf)
        task.predict_with_existing(X_t, y_t_regression, y_t_classification,
                                   X_val, y_val_regression, y_val_classification,
                                   X_test, y_test_regression, y_test_classification,
                                   weight_file)

    elif 'multi_classification' in model_name:
        with open('../../json/multi_classification_keck_pria_retest.json', 'r') as f:
            conf = json.load(f)
        label_name_list = conf['label_name_list']

        train_pd = read_merged_data(train_file_list)
        train_pd.fillna(0, inplace=True)
        val_pd = read_merged_data(val_file_list)
        val_pd.fillna(0, inplace=True)
        # TODO: may only consider Keck label
        test_pd = read_merged_data(test_file_list)
        test_pd.fillna(0, inplace=True)

        multi_name_list = train_pd.columns[-128:].tolist()
        multi_name_list.extend(label_name_list)
        print 'multi_name_list ', multi_name_list

        X_train, y_train = extract_feature_and_label(train_pd,
                                                     feature_name='Fingerprints',
                                                     label_name_list=multi_name_list)
        X_val, y_val = extract_feature_and_label(val_pd,
                                                 feature_name='Fingerprints',
                                                 label_name_list=multi_name_list)
        X_test, y_test = extract_feature_and_label(test_pd,
                                                   feature_name='Fingerprints',
                                                   label_name_list=multi_name_list)

        sample_weight_dir = '../../dataset/sample_weights/keck_pcba/fold_5/'
        file_list = []
        for i in range(k):
            file_list.append('sample_weight_{}.csv'.format(i))
        sample_weight_file = [sample_weight_dir + f_ for f_ in file_list]
        sample_weight_file = np.array(sample_weight_file)
        sample_weight_pd = read_merged_data(sample_weight_file[train_index])
        _, sample_weight = extract_feature_and_label(sample_weight_pd,
                                                     feature_name='Fingerprints',
                                                     label_name_list=multi_name_list)

        task = MultiClassification(conf=conf)
        task.predict_with_existing(X_train, y_train,
                                   X_val, y_val,
                                   X_test, y_test,
                                   weight_file,
                                   '{}/{}/{}/{}/score'.format(task_name, model_name, record[model_name, task_name], fold_idx))