import pandas as pd
import numpy as np
import os
from collections import OrderedDict
from virtual_screening.function import *
from prospective_screening_model_names import *
from prospective_screening_metric_names import *


def merge_prediction():
    dataframe = pd.read_excel('../../output/stage_2_predictions/Keck_LC4_export.xlsx')

    supplier_id = dataframe['Supplier ID'].tolist()
    failed_id = ['F0401-0050', 'F2964-1411', 'F2964-1523']
    inhibits = dataframe[
        'PriA-SSB AS, normalized for plate and edge effects, correct plate map: % inhibition Alpha, normalized (%)'].tolist()

    positive_enumerate = filter(lambda x: x[1] >= 35 and supplier_id[x[0]] not in failed_id, enumerate(inhibits))
    positive_idx = map(lambda x: x[0], positive_enumerate)
    actual_label = map(lambda x: 1 if x in positive_idx else 0, range(len(supplier_id)))

    complete_df = pd.DataFrame({'molecule id': supplier_id, 'label': actual_label, 'inhibition': inhibits})
    column_names = ['molecule id', 'label', 'inhibition']
    complete_df = complete_df[column_names]
    # complete_df[complete_df['actual label'] > 0]

    dir_ = '../../output/stage_2_predictions/Keck_Pria_AS_Retest'

    file_path = '{}/{}.npz'.format(dir_, 'vanilla_lstm_19')
    data = np.load(file_path)
    molecule_id = data['molecule_id']

    model_names = []
    special_models = ['irv', 'random_forest', 'dockscore', 'consensus', 'baseline']

    for model_name in model_name_mapping.keys():
        file_path = '{}/{}.npz'.format(dir_, model_name)
        if not os.path.exists(file_path):
            continue
        print 'model: {} exists'.format(model_name)
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

        print

    model_names = sorted(model_names)
    column_names.extend(model_names)

    complete_df = complete_df[column_names]
    complete_df.to_csv('{}/complete_prediction.csv'.format(dir_), index=None)


def merge_rank_with_ensemble():
    dir_ = '../../output/stage_2_predictions/Keck_Pria_AS_Retest'
    complete_df = pd.read_csv('{}/complete_prediction.csv'.format(dir_))
    model_names = complete_df.columns[3:]
    rank_df = complete_df[['molecule id', 'label', 'inhibition']]
    for (idx, model_name) in enumerate(model_names):
        order = complete_df[model_name].rank(ascending=False).tolist()
        order = np.array(order)
        order = order.astype(np.int)
        rank_df[model_name] = order

    ensemble_model_names_pairs = OrderedDict()
    ensemble_model_names_pairs['Ensemble_a'] = ['SingleRegression_a', 'SingleClassification_a']
    ensemble_model_names_pairs['Ensemble_b'] = ['SingleRegression_a', 'SingleClassification_b']
    ensemble_model_names_pairs['Ensemble_c'] = ['SingleRegression_b', 'SingleClassification_a']
    ensemble_model_names_pairs['Ensemble_d'] = ['SingleRegression_b', 'SingleClassification_b']

    for ensemble_name, ensemble_model_names in ensemble_model_names_pairs.items():
        ensemble_orders = []
        for (idx, model_name) in enumerate(model_names):
            order = complete_df[model_name].rank(ascending=False).tolist()
            order = np.array(order)
            order = order.astype(np.int)
            if model_name in ensemble_model_names:
                ensemble_orders.append(order)
        ensemble_orders = np.vstack(ensemble_orders)
        ensemble_order = np.zeros((ensemble_orders.shape[1]))
        for i in range(ensemble_orders.shape[1]):
            ensemble_order[i] = min(ensemble_orders[0, i], ensemble_orders[1, i])
        rank_df[ensemble_name] = ensemble_order

    rank_df.to_csv('{}/complete_rank.csv'.format(dir_), index=None)


def merge_evaluation():
    dir_ = '../../output/stage_2_predictions/Keck_Pria_AS_Retest'
    complete_df = pd.read_csv('{}/complete_prediction.csv'.format(dir_))
    model_names = complete_df.columns[3:]

    metric_df = pd.DataFrame({'Model': model_names})

    actual_oracle = complete_df['label'].as_matrix()
    actual_oracle = reshape_data_into_2_dim(actual_oracle)

    for (metric_name, metric_) in metric_name_mapping.iteritems():
        metric_values = []
        for model_name in model_names:
            pred = complete_df[model_name].as_matrix()
            pred = reshape_data_into_2_dim(pred)

            actual, pred = collectively_drop_nan(actual_oracle, pred)

            value = metric_['function'](actual, pred, **metric_['argument'])
            metric_values.append(value)
            print metric_name, '\t', model_name, '\t', value
        metric_df[metric_name] = metric_values
        print

    print 'saving to {}/complete_evaluation.csv'.format(dir_)
    metric_df.to_csv('{}/complete_evaluation.csv'.format(dir_), index=None)


def filter_model_name(model_name):
    model_name = model_name.replace('SingleClassification', 'STC')
    model_name = model_name.replace('SingleRegression', 'STR')
    model_name = model_name.replace('MultiClassification', 'MTC')
    model_name = model_name.replace('RandomForest', 'RF')
    model_name = model_name.replace('LightChem', 'LC')
    model_name = model_name.replace('ConsensusDocking', 'ConDock')
    model_name = model_name.replace('Docking', 'Dock')
    return model_name


if __name__ == '__main__':
    merge_prediction()
    merge_rank_with_ensemble()
    merge_evaluation()