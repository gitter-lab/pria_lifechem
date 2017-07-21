from virtual_screening.function import *
from virtual_screening.evaluation import *
from virtual_screening.models.deep_classification import *
from math import sqrt


def compare_rank(y_true, y_pred_on_single_NN, y_pred_on_xgboost):
    args = np.argsort(y_pred_on_single_NN[:, 0])
    ranked_NN = np.arange(len(args))[args.argsort()]

    args = np.argsort(y_pred_on_xgboost[:, 0])
    # ranked_xgboost = map(lambda x: args[x], np.arange(len(args)))
    ranked_xgboost = np.arange(len(args))[args.argsort()]

    n = y_true.shape[0]

    for i in range(n):
        if y_true[i] == 1:
            print('{:6}|\t'.format(i)),
            print('{:.6f}\t{}\tRank: {:.4f}\t|\t'.format(y_pred_on_single_NN[i, 0],
                                                             n - 1 - ranked_NN[i],
                                                             1.0 - 1.0 * ranked_NN[i] / n)),
            print('{:.6f}\t{}\tRank: {:.4f}'.format(y_pred_on_xgboost[i, 0],
                                                        n - 1 - ranked_xgboost[i],
                                                        1.0 - 1.0 * ranked_xgboost[i] / n))
    return


def similarity(mol_a, mol_b):
    return np.dot(mol_a, mol_b)/(sqrt(sum(mol_a)) * sqrt(sum(mol_b)))


def print_list(test_list, X_fps, X_train, y_train,
               top=20, threshold=0.6, importance_index=None, important_fetcher=50):
    for test_id in test_list:
        test_molecule = X_fps[test_id]
        print 'testing Molecule: ', test_id
        id_list = []
        sim_list = []
        label_list = []
        important_sim_list = []
        importance_index = importance_index[:important_fetcher]
        for i in range(len(X_train)):
            x_val = X_train[i]
            sim = similarity(test_molecule, x_val)
            sim_important = similarity(test_molecule[importance_index],
                                       x_val[importance_index])
            if sim >= threshold or sim_important >= threshold:
                id_list.append(i)
                sim_list.append(sim)
                label_list.append(y_train[i, 0])
                important_sim_list.append(sim_important)

        args = np.argsort(sim_list)[::-1]
        for i in range(len(id_list)):
            if i >= top:
                break
            index = args[i]
            if label_list[index] == 1:
                print '\033[31m{:7} |  sim: {:.6f}\tsim(only important): {:.6}\t\ttrue label: {}  *\x1b[0m'.\
                    format(id_list[index],
                           sim_list[index],
                           important_sim_list[index],
                           int(label_list[index]))
            else:
                print '{:7} |  sim: {:.6f}\tsim(only important): {:.6}\t\ttrue label: {}'.\
                    format(id_list[index],
                           sim_list[index],
                           important_sim_list[index],
                           int(label_list[index]))
        print
    return

def get_rank(y_true, y_pred):
    args = np.argsort(y_pred[:,0])
    ranked = np.arange(len(args))[args.argsort()]
    n = y_true.shape[0]
    for i in range(n):
        if y_true[i] == 1:
            print('{:.6f}\t{:6}/{:6}\t\tRank: {:.6f}'.format(y_pred[i, 0],
                                                     n-ranked[i],
                                                     n,
                                                     1-1.0*ranked[i]/n))
    return

def predict_with_existing(model,  X_train, y_train, X_val, y_val, X_test, y_test):
    y_pred_on_train = reshape_data_into_2_dim(model.predict_proba(X_train)[:, 1])
    y_pred_on_val = reshape_data_into_2_dim(model.predict_proba(X_val)[:, 1])
    y_pred_on_test = reshape_data_into_2_dim(model.predict_proba(X_test)[:, 1])

    print('train precision: {}'.format(precision_auc_single(y_train, y_pred_on_train)))
    print('train roc: {}'.format(roc_auc_single(y_train, y_pred_on_train)))
    print('train bedroc: {}'.format(bedroc_auc_single(y_train, y_pred_on_train)))
    print
    print('validation precision: {}'.format(precision_auc_single(y_val, y_pred_on_val)))
    print('validation roc: {}'.format(roc_auc_single(y_val, y_pred_on_val)))
    print('validation bedroc: {}'.format(bedroc_auc_single(y_val, y_pred_on_val)))
    print
    print('test precision: {}'.format(precision_auc_single(y_test, y_pred_on_test)))
    print('test roc: {}'.format(roc_auc_single(y_test, y_pred_on_test)))
    print('test bedroc: {}'.format(bedroc_auc_single(y_test, y_pred_on_test)))
    print

    for EF_ratio in [0.02, 0.01, 0.0015, 0.001]:
        n_actives, ef, ef_max = enrichment_factor_single(y_test, y_pred_on_test, EF_ratio)
        print('ratio: {}, EF: {},\tactive: {}'.format(EF_ratio, ef, n_actives))

    return