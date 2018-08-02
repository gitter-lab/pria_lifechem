from __future__ import print_function

import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from pria_lifechem.function import *
from prospective_screening_model_names import *
from prospective_screening_metric_names import *



dataframe = pd.read_csv('../../dataset/fixed_dataset/pria_prospective.csv.gz')
molecule_ids = dataframe['Molecule'].tolist()
actual_labels = dataframe['Keck_Pria_AS_Retest'].tolist()
inhibits = dataframe['Keck_Pria_Continuous'].tolist()

complete_df = pd.DataFrame({'molecule': molecule_ids, 'label': actual_labels, 'inhibition': inhibits})

column_names = ['molecule', 'label', 'inhibition']
complete_df = complete_df[column_names]

dir_ = '../../output/stage_2_predictions/Keck_Pria_AS_Retest'
model_names = []

for model_name in model_name_mapping.keys():
    file_path = '{}/{}.npz'.format(dir_, model_name)
    if not os.path.exists(file_path):
        print('model: {} doesn\'t exist'.format(model_name))
        continue
    data = np.load(file_path)
    # print(file_path, '\t', data.keys(), '\t', data['y_pred_on_test'].shape)

    y_pred = data['y_pred_on_test'][:, 2]
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]
    print(y_pred.shape, y_pred[:5])

    model_name = model_name_mapping[model_name]
    model_names.append(model_name)
    complete_df[model_name] = y_pred

model_names = sorted(model_names)
column_names.extend(model_names)

complete_df = complete_df[column_names]
print()

### Generate Metric DF
true_label = complete_df['label'].as_matrix()
true_label = reshape_data_into_2_dim(true_label)

model_names.remove('Baseline')

roc_auc_list = []
metric_df = pd.DataFrame({'Model': model_names})

for (metric_name, metric_) in metric_name_mapping.iteritems():
    print(metric_name)
    metric_values = []
    for model_name in model_names:
        pred = complete_df[model_name].as_matrix()
        pred = reshape_data_into_2_dim(pred)

        actual, pred = collectively_drop_nan(true_label, pred)
        print(actual.shape, '\t', pred.shape)
        value = metric_['function'](actual, pred, **metric_['argument'])
        metric_values.append(value)
    metric_df[metric_name] = metric_values

metric_df.to_csv('../../output/stage_2_predictions/Keck_Pria_AS_Retest/VS_RMI_metric.csv', index=None)