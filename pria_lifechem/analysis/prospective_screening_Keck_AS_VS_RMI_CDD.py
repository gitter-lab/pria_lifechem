from __future__ import print_function

import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from pria_lifechem.function import *
from prospective_screening_model_names import *
from prospective_screening_metric_names import *


complete_df = pd.read_csv('LC4_complete.csv')
column_names = ['molecule name', 'molecule id', 'label', 'inhibition']

dir_ = '../../output/stage_2_predictions/RMI'

file_path = '{}/{}.npz'.format(dir_, 'vanilla_lstm_19')
data = np.load(file_path)
molecule_id = data['molecule_id']

model_names = []
special_models = ['irv', 'random_forest', 'dockscore', 'consensus', 'baseline']

for model_name in model_name_mapping.keys():
    file_path = '{}/{}.npz'.format(dir_, model_name)
    if not os.path.exists(file_path):
        continue
    data = np.load(file_path)

    if any(x in model_name for x in special_models):
        y_pred = data['y_pred_on_test']
    else:
        y_pred = data['y_pred']
    if y_pred.ndim == 2:
        y_pred = y_pred[:, 0]

    temp_df = pd.DataFrame({'molecule id': molecule_id,
                            model_name_mapping[model_name]: y_pred})

    model_names.append(model_name_mapping[model_name])
    complete_df = complete_df.join(temp_df.set_index('molecule id'), on='molecule id')

model_names = sorted(model_names)
column_names.extend(model_names)

complete_df = complete_df[column_names]

### Generate Metric DF
true_label = complete_df['label'].as_matrix()
true_label = reshape_data_into_2_dim(true_label)

roc_auc_list = []
metric_df = pd.DataFrame({'Model': model_names})

for (metric_name, metric_) in metric_name_mapping.iteritems():
    print(metric_name)
    metric_values = []
    for model_name in model_names:
        pred = complete_df[model_name].as_matrix()
        pred = reshape_data_into_2_dim(pred)

        actual, pred = collectively_drop_nan(true_label, pred)
        value = metric_['function'](actual, pred, **metric_['argument'])
        metric_values.append(value)
    metric_df[metric_name] = metric_values

metric_df.to_csv('../../output/stage_2_predictions/Keck_Pria_AS_Retest/VS_RMI_metric.csv', index=None)