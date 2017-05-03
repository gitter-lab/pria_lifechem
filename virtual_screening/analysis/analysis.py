import re
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


evaluations = {0: 'train prec', 1: 'train roc', 2: 'train bedroc',
               3: 'val prec', 4: 'val roc', 5: 'val bedroc',
               6: 'test prec', 7: 'test roc', 8: 'test bedroc',
               9: 'EF_2', 10: 'EF_1', 11: 'EF_015', 12: 'EF_01'}

facecolors = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'b']


def check_result_completeness(dir_, file_path, number):
    cnt = 0
    for i in range(number):
        whole_path = dir_ + file_path + '{}.out'.format(i)
        if not os.path.isfile(whole_path):
            cnt += 1
            print 'Missing hyperparameter set result: {}'.format(i)
    if cnt == 0:
        print 'All output result complete.'
    return


def get_number(string):
    m = re.search('\d+\.\d+', string)
    return float(m.group())


def get_EF_number(string):
    m = re.findall('\d+\.\d+', string)
    ret = np.array(m).astype(float)
    return ret


def action(dir_name, k):
    train_prec_list = []
    train_roc_list = []
    train_bedroc_list = []

    val_prec_list = []
    val_roc_list = []
    val_bedroc_list = []

    test_prec_list = []
    test_roc_list = []
    test_bedroc_list = []

    EF_2_list = []
    EF_1_list = []
    EF_015_list = []
    EF_01_list = []

    for x in range(k):
        file_path = dir_name + '{}.out'.format(x)
        
        if not os.path.isfile(file_path):
            train_prec_list.append(0)
            train_roc_list.append(0)
            train_bedroc_list.append(0)
            val_prec_list.append(0)
            val_roc_list.append(0)
            val_bedroc_list.append(0)
            test_prec_list.append(0)
            test_roc_list.append(0)
            test_bedroc_list.append(0)
            EF_2_list.append([0, 0, 0])
            EF_1_list.append([0, 0, 0])
            EF_015_list.append([0, 0, 0])
            EF_01_list.append([0, 0, 0])
            continue
            
        with open(file_path, 'r') as f:
            read_data = f.readlines()
            for line in read_data:
                if 'train precision' in line:
                    train_prec_list.append(get_number(line))
                if 'train roc' in line:
                    train_roc_list.append(get_number(line))
                if 'train bedroc' in line:
                    train_bedroc_list.append(get_number(line))
                if 'validation precision' in line:
                    val_prec_list.append(get_number(line))
                if 'validation roc' in line:
                    val_roc_list.append(get_number(line))
                if 'validation bedroc' in line:
                    val_bedroc_list.append(get_number(line))
                if 'test precision' in line:
                    test_prec_list.append(get_number(line))
                if 'test roc' in line:
                    test_roc_list.append(get_number(line))
                if 'test bedroc' in line:
                    test_bedroc_list.append(get_number(line))
                if 'ratio: 0.02,' in line:
                    EF_2_list.append(get_EF_number(line))
                if 'ratio: 0.01,' in line:
                    EF_1_list.append(get_EF_number(line))
                if 'ratio: 0.0015,' in line:
                    EF_015_list.append(get_EF_number(line))
                if 'ratio: 0.001,' in line:
                    EF_01_list.append(get_EF_number(line))

    train_prec_list = np.array(train_prec_list)
    train_roc_list = np.array(train_roc_list)
    train_bedroc_list = np.array(train_bedroc_list)

    val_prec_list = np.array(val_prec_list)
    val_roc_list = np.array(val_roc_list)
    val_bedroc_list = np.array(val_bedroc_list)

    test_prec_list = np.array(test_prec_list)
    test_roc_list = np.array(test_roc_list)
    test_bedroc_list = np.array(test_bedroc_list)

    EF_2_list = np.array(EF_2_list)
    EF_1_list = np.array(EF_1_list)
    EF_015_list = np.array(EF_015_list)
    EF_01_list = np.array(EF_01_list)

    return train_prec_list, train_roc_list, train_bedroc_list, \
           val_prec_list, val_roc_list, val_bedroc_list, \
           test_prec_list, test_roc_list, test_bedroc_list, \
           EF_2_list, EF_1_list, EF_015_list, EF_01_list


'''
Simply ignore the non-existing files
'''
def action_ignore(dir_name, k):
    train_prec_list = []
    train_roc_list = []
    train_bedroc_list = []

    val_prec_list = []
    val_roc_list = []
    val_bedroc_list = []

    test_prec_list = []
    test_roc_list = []
    test_bedroc_list = []

    EF_2_list = []
    EF_1_list = []
    EF_015_list = []
    EF_01_list = []

    for x in range(k):
        file_path = dir_name + '{}.out'.format(x)

        if not os.path.isfile(file_path):
            continue

        with open(file_path, 'r') as f:
            read_data = f.readlines()
            for line in read_data:
                if 'train precision' in line:
                    train_prec_list.append(get_number(line))
                if 'train roc' in line:
                    train_roc_list.append(get_number(line))
                if 'train bedroc' in line:
                    train_bedroc_list.append(get_number(line))
                if 'validation precision' in line:
                    val_prec_list.append(get_number(line))
                if 'validation roc' in line:
                    val_roc_list.append(get_number(line))
                if 'validation bedroc' in line:
                    val_bedroc_list.append(get_number(line))
                if 'test precision' in line:
                    test_prec_list.append(get_number(line))
                if 'test roc' in line:
                    test_roc_list.append(get_number(line))
                if 'test bedroc' in line:
                    test_bedroc_list.append(get_number(line))
                if 'ratio: 0.02,' in line:
                    EF_2_list.append(get_EF_number(line))
                if 'ratio: 0.01,' in line:
                    EF_1_list.append(get_EF_number(line))
                if 'ratio: 0.0015,' in line:
                    EF_015_list.append(get_EF_number(line))
                if 'ratio: 0.001,' in line:
                    EF_01_list.append(get_EF_number(line))

    train_prec_list = np.array(train_prec_list)
    train_roc_list = np.array(train_roc_list)
    train_bedroc_list = np.array(train_bedroc_list)

    val_prec_list = np.array(val_prec_list)
    val_roc_list = np.array(val_roc_list)
    val_bedroc_list = np.array(val_bedroc_list)

    test_prec_list = np.array(test_prec_list)
    test_roc_list = np.array(test_roc_list)
    test_bedroc_list = np.array(test_bedroc_list)

    EF_2_list = np.array(EF_2_list)
    EF_1_list = np.array(EF_1_list)
    EF_015_list = np.array(EF_015_list)
    EF_01_list = np.array(EF_01_list)

    return train_prec_list, train_roc_list, train_bedroc_list, \
           val_prec_list, val_roc_list, val_bedroc_list, \
           test_prec_list, test_roc_list, test_bedroc_list, \
           EF_2_list, EF_1_list, EF_015_list, EF_01_list


def plot_single_model_single_evaluation(dir_path, k, evaluation, title):
    plt.figure(dpi=80)
    X = np.arange(k)

    train_prec_list, train_roc_list, train_bedroc_list, \
    val_prec_list, val_roc_list, val_bedroc_list, \
    test_prec_list, test_roc_list, test_bedroc_list, \
    EF_2_list, EF_1_list, EF_015_list, EF_01_list = action(dir_path, k)
    
    print 'k ', k
    print 'train size : ', train_prec_list.shape

    if evaluation == 'train prec':
        Y = train_prec_list
    elif evaluation == 'train roc':
        Y = train_roc_list
    elif evaluation == 'train bedroc':
        Y = train_bedroc_list
    elif evaluation == 'val prec':
        Y = val_prec_list
    elif evaluation == 'val roc':
        Y = val_roc_list
    elif evaluation == 'val bedroc':
        Y = val_bedroc_list
    elif evaluation == 'test prec':
        Y = test_prec_list
    elif evaluation == 'test roc':
        Y = test_roc_list
    elif evaluation == 'test bedroc':
        Y = test_bedroc_list
    elif evaluation == 'EF_2':
        Y = EF_2_list[:, 1]
    elif evaluation == 'EF_1':
        Y = EF_1_list[:, 1]
    elif evaluation == 'EF_015':
        Y = EF_015_list[:, 1]
    elif evaluation == 'EF_01':
        Y = EF_01_list[:, 1]
    else:
        raise Exception('No such evaluation method.')

    plt.bar(X, Y, width=0.4, edgecolor='white', alpha=0.8, label=k)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3)
    plt.xlim(0, k)
    plt.xticks(X)
    plt.title(title)
    plt.show()
    
    return


def plot_single_model_multi_evaluations(dir_path, k, evaluation_list, title):
    print evaluation_list
    plt.figure(dpi=80)
    X = np.arange(k)

    train_prec_list, train_roc_list, train_bedroc_list, \
    val_prec_list, val_roc_list, val_bedroc_list, \
    test_prec_list, test_roc_list, test_bedroc_list, \
    EF_2_list, EF_1_list, EF_015_list, EF_01_list = action(dir_path, k)

    for i in range(len(evaluation_list)):
        evaluation = evaluation_list[i]
        if evaluation == 'train prec':
            Y = train_prec_list
        elif evaluation == 'train roc':
            Y = train_roc_list
        elif evaluation == 'train bedroc':
            Y = train_bedroc_list
        elif evaluation == 'val prec':
            Y = val_prec_list
        elif evaluation == 'val roc':
            Y = val_roc_list
        elif evaluation == 'val bedroc':
            Y = val_bedroc_list
        elif evaluation == 'test prec':
            Y = test_prec_list
        elif evaluation == 'test roc':
            Y = test_roc_list
        elif evaluation == 'test bedroc':
            Y = test_bedroc_list
        elif evaluation == 'EF_2':
            Y = EF_2_list[:, 1]
        elif evaluation == 'EF_1':
            Y = EF_1_list[:, 1]
        elif evaluation == 'EF_015':
            Y = EF_015_list[:, 1]
        elif evaluation == 'EF_01':
            Y = EF_01_list[:, 1]
        else:
            raise Exception('No such evaluation method.')

        plt.bar(X+0.2*i, Y, width=0.2, edgecolor='white', facecolor=facecolors[i], alpha=0.8, label=evaluation)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3)
    plt.xlim(0, k)
    plt.xticks(X)
    plt.title(title)
    plt.show()
    
    return


def get_ranked_analysis(dir_path, k, evaluation_list, fetch_top_num):

    train_prec_list, train_roc_list, train_bedroc_list, \
    val_prec_list, val_roc_list, val_bedroc_list, \
    test_prec_list, test_roc_list, test_bedroc_list, \
    EF_2_list, EF_1_list, EF_015_list, EF_01_list = action(dir_path, k)

    for i in range(len(evaluation_list)):
        evaluation = evaluation_list[i]
        print 'Analyze {}:'.format(evaluation)
        if evaluation == 'train prec':
            Y = train_prec_list
        elif evaluation == 'train roc':
            Y = train_roc_list
        elif evaluation == 'train bedroc':
            Y = train_bedroc_list
        elif evaluation == 'val prec':
            Y = val_prec_list
        elif evaluation == 'val roc':
            Y = val_roc_list
        elif evaluation == 'val bedroc':
            Y = val_bedroc_list
        elif evaluation == 'test prec':
            Y = test_prec_list
        elif evaluation == 'test roc':
            Y = test_roc_list
        elif evaluation == 'test bedroc':
            Y = test_bedroc_list
        elif evaluation == 'EF_2':
            Y = EF_2_list[:, 1]
        elif evaluation == 'EF_1':
            Y = EF_1_list[:, 1]
        elif evaluation == 'EF_015':
            Y = EF_015_list[:, 1]
        elif evaluation == 'EF_01':
            Y = EF_01_list[:, 1]
        else:
            raise Exception('No such evaluation method.')

        sorted_y = [i[0] for i in sorted(enumerate(Y), key=lambda x:x[1], reverse=True)]
        print 'Top {} rankings: '.format(fetch_top_num),
        print sorted_y[:fetch_top_num]

        print


    return


def fetch_one_model(dir_path, number, evaluation_list, model):
    train_prec_list, train_roc_list, train_bedroc_list, \
    val_prec_list, val_roc_list, val_bedroc_list, \
    test_prec_list, test_roc_list, test_bedroc_list, \
    EF_2_list, EF_1_list, EF_015_list, EF_01_list = action_ignore(dir_path, number)

    evaluation_column = []
    value_column = []
    model_column = []

    if 'train prec' in evaluation_list:
        evaluation_column.extend(['train prec' for _ in train_prec_list])
        value_column.extend(list(train_prec_list))
        model_column.extend([model for _ in train_prec_list])
    if 'train roc' in evaluation_list:
        evaluation_column.extend(['train roc' for _ in train_roc_list])
        value_column.extend(list(train_roc_list))
        model_column.extend([model for _ in train_roc_list])
    if 'train bedroc' in evaluation_list:
        evaluation_column.extend(['train bedroc' for _ in train_bedroc_list])
        value_column.extend(list(train_bedroc_list))
        model_column.extend([model for _ in train_bedroc_list])

    if 'val prec' in evaluation_list:
        evaluation_column.extend(['val prec' for _ in val_prec_list])
        value_column.extend(list(val_prec_list))
        model_column.extend([model for _ in val_prec_list])
    if 'val roc' in evaluation_list:
        evaluation_column.extend(['val roc' for _ in val_roc_list])
        value_column.extend(list(val_roc_list))
        model_column.extend([model for _ in val_roc_list])
    if 'val bedroc' in evaluation_list:
        evaluation_column.extend(['val bedroc' for _ in val_bedroc_list])
        value_column.extend(list(val_bedroc_list))
        model_column.extend([model for _ in val_bedroc_list])

    if 'test prec' in evaluation_list:
        evaluation_column.extend(['test prec' for _ in test_prec_list])
        value_column.extend(list(test_prec_list))
        model_column.extend([model for _ in test_prec_list])
    if 'test roc' in evaluation_list:
        evaluation_column.extend(['test roc' for _ in test_prec_list])
        value_column.extend(list(test_prec_list))
        model_column.extend([model for _ in test_prec_list])
    if 'test bedroc' in evaluation_list:
        evaluation_column.extend(['test bedroc' for _ in test_prec_list])
        value_column.extend(list(test_prec_list))
        model_column.extend([model for _ in test_prec_list])

    if 'EF_2' in evaluation_list:
        evaluation_column.extend(['EF_2' for _ in EF_2_list])
        value_column.extend(list(EF_2_list[:,1]))
        model_column.extend([model for _ in EF_2_list])
    if 'EF_1' in evaluation_list:
        evaluation_column.extend(['EF_1' for _ in EF_1_list])
        value_column.extend(list(EF_1_list[:,1]))
        model_column.extend([model for _ in EF_1_list])
    if 'EF_015' in evaluation_list:
        evaluation_column.extend(['EF_015' for _ in EF_015_list])
        value_column.extend(list(EF_015_list[:,1]))
        model_column.extend([model for _ in EF_015_list])
    if 'EF_01' in evaluation_list:
        evaluation_column.extend(['EF_01' for _ in EF_01_list])
        value_column.extend(list(EF_01_list[:,1]))
        model_column.extend([model for _ in EF_01_list])

    return evaluation_column, value_column, model_column


def plot_cross_validation(dir_path_list, evaluation_list, model_list, title):
    evaluation_column = []
    value_column = []
    model_column = []

    for i in range(len(dir_path_list)):
        dir_ = dir_path_list[i]
        model = model_list[i]

        c1, c2, c3 = fetch_one_model(dir_, 20, evaluation_list=evaluation_list, model=model)
        evaluation_column.extend(c1)
        value_column.extend(c2)
        model_column.extend(c3)

    data_pd = pd.DataFrame({'evaluation method': evaluation_column,
                            'value': value_column,
                            'model': model_column})

    sns.boxplot(x="evaluation method", y="value", hue="model", data=data_pd, palette="PRGn")
    # sns.violinplot(x="evaluation method", y="value", hue="model", data=data_pd, palette="PRGn", inner=None)
    sns.swarmplot(x="evaluation method", y="value", hue="model", data=data_pd, palette="YlOrBr", split=True, size=3)
    sns.despine(offset=20, trim=True)
    sns.plt.rcParams['figure.figsize'] = (10.0, 5.0)
    sns.plt.title(title)

    return