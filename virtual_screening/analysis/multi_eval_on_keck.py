from virtual_screening.function import *
from virtual_screening.evaluation import *
from virtual_screening.models.CallBacks import *
from virtual_screening.models.deep_classification import *


def predict_with_existing(task,
                          X_train, y_train,
                          X_val, y_val,
                          X_test, y_test,
                          PMTNN_weight_file,
                          output_file,
                          eval_indices=[-1],
                          eval_mean_or_median=np.mean):
    def get_model_roc_auc(true_label,
                          predicted_label,
                          eval_indices=eval_indices,
                          eval_mean_or_median=eval_mean_or_median):
        return roc_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

    def get_model_bedroc_auc(true_label,
                             predicted_label,
                             eval_indices=eval_indices,
                             eval_mean_or_median=eval_mean_or_median):
        return bedroc_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

    def get_model_precision_auc(true_label,
                                predicted_label,
                                eval_indices=eval_indices,
                                eval_mean_or_median=eval_mean_or_median):
        return precision_auc_multi(true_label, predicted_label, eval_indices, eval_mean_or_median)

    model = task.setup_model()
    model.load_weights(PMTNN_weight_file)

    y_pred_on_train = model.predict(X_train)
    y_pred_on_val = model.predict(X_val)
    y_pred_on_test = model.predict(X_test)

    print 'wirting to ', output_file
    output_file = open(output_file, 'w')

    print >> output_file, 'train precision: {}'.format(get_model_precision_auc(y_train, y_pred_on_train))
    print >> output_file, 'train roc: {}'.format(get_model_roc_auc(y_train, y_pred_on_train))
    print >> output_file, 'train bedroc: {}'.format(get_model_bedroc_auc(y_train, y_pred_on_train))
    print >> output_file, ''
    print >> output_file, 'validation precision: {}'.format(get_model_precision_auc(y_val, y_pred_on_val))
    print >> output_file, 'validation roc: {}'.format(get_model_roc_auc(y_val, y_pred_on_val))
    print >> output_file, 'validation bedroc: {}'.format(get_model_bedroc_auc(y_val, y_pred_on_val))
    print >> output_file, ''
    print >> output_file, 'test precision: {}'.format(get_model_precision_auc(y_test, y_pred_on_test))
    print >> output_file, 'test roc: {}'.format(get_model_roc_auc(y_test, y_pred_on_test))
    print >> output_file, 'test bedroc: {}'.format(get_model_bedroc_auc(y_test, y_pred_on_test))
    print >> output_file, ''

    # Just print last target EF into output file.
    for EF_ratio in task.EF_ratio_list:
        n_actives, ef, ef_max = enrichment_factor_single(y_test[:, -1], y_pred_on_test[:, -1], EF_ratio)
        print >> output_file, 'ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives)

    output_file.flush()
    output_file.close()
    return


# TODO: may merge this with the virtual_screening.function.extract_feature_and_label
def extract_feature_and_label(data_pd,
                              feature_name,
                              label_name_list):
    X_data = data_pd[feature_name].tolist()
    X_data = map(lambda x: list(x), X_data)
    X_data = np.array(X_data)
    X_data = X_data.astype(float)

    label = label_name_list[0]  # TODO: This is for hacking
    y_data = data_pd[label].tolist()
    y_data = np.array(y_data)
    y_data = y_data.astype(float)

    # In case we just train on one target
    # y would be (n,) vector
    # then we should change it to (n,1) 1D matrix
    # to keep consistency
    print y_data.shape
    if y_data.ndim == 1:
        n = y_data.shape[0]
        y_data = y_data.reshape(n, 1)

    return X_data, y_data



def transform(old_dir, neo_dir, json_file):
    number = 20
    k = 5
    directory = '../../dataset/keck_pcba/fold_{}/'.format(k)
    file_list = []
    for i in range(k):
        file_list.append('{}file_{}.csv'.format(directory, i))
    file_list = np.array(file_list)

    for running_index in range(number):
        # reload model
        PMTNN_weight_file = old_dir + '{}.weight'.format(running_index)
        with open(json_file, 'r') as f:
            conf = json.load(f)
        multi_name_list = conf['label_name_list']
        print 'Testing name_list: ', multi_name_list
        extractor = ['Fingerprints']
        extractor.extend(multi_name_list)
        multi_task = MultiClassification(conf=conf)

        # prepare file paths
        test_index = running_index / 4
        val_index = running_index % 4 + (running_index % 4 >= test_index)
        complete_index = np.arange(k)
        train_index = np.where((complete_index != test_index) & (complete_index != val_index))[0]

        train_file_list = file_list[train_index]
        val_file_list = file_list[val_index:val_index + 1]
        test_file_list = file_list[test_index:test_index + 1]

        neo_file = neo_dir + 'old_{}.out'.format(running_index)
        if not os.path.exists(neo_file):
            # load data, zero-out missing values
            train_pd = read_merged_data(train_file_list, usecols=extractor)
            train_pd.fillna(0, inplace=True)
            val_pd = read_merged_data(val_file_list, usecols=extractor)
            val_pd.fillna(0, inplace=True)
            test_pd = read_merged_data(test_file_list, usecols=extractor)
            test_pd.fillna(0, inplace=True)
            X_train, y_train = extract_feature_and_label(train_pd,
                                                         feature_name='Fingerprints',
                                                         label_name_list=multi_name_list)
            X_val, y_val = extract_feature_and_label(val_pd,
                                                     feature_name='Fingerprints',
                                                     label_name_list=multi_name_list)
            X_test, y_test = extract_feature_and_label(test_pd,
                                                       feature_name='Fingerprints',
                                                       label_name_list=multi_name_list)
            predict_with_existing(multi_task,
                                  X_train, y_train,
                                  X_val, y_val,
                                  X_test, y_test,
                                  PMTNN_weight_file,
                                  neo_file)

        neo_file = neo_dir + '{}.out'.format(running_index)
        if not os.path.exists(neo_file):
            # load data, remove missing values
            train_pd = read_merged_data(train_file_list, usecols=extractor)
            train_pd.dropna(axis=0, subset=multi_name_list, how='any', inplace=True)
            val_pd = read_merged_data(val_file_list, usecols=extractor)
            val_pd.dropna(axis=0, subset=multi_name_list, how='any', inplace=True)
            test_pd = read_merged_data(test_file_list, usecols=extractor)
            test_pd.dropna(axis=0, subset=multi_name_list, how='any', inplace=True)
            X_train, y_train = extract_feature_and_label(train_pd,
                                                         feature_name='Fingerprints',
                                                         label_name_list=multi_name_list)
            X_val, y_val = extract_feature_and_label(val_pd,
                                                     feature_name='Fingerprints',
                                                     label_name_list=multi_name_list)
            X_test, y_test = extract_feature_and_label(test_pd,
                                                       feature_name='Fingerprints',
                                                       label_name_list=multi_name_list)
            predict_with_existing(multi_task,
                                  X_train, y_train,
                                  X_val, y_val,
                                  X_test, y_test,
                                  PMTNN_weight_file,
                                  neo_file)