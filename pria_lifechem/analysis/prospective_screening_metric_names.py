from pria_lifechem.evaluation import *
from collections import OrderedDict
import matplotlib.pyplot as plt


metric_name_mapping = OrderedDict()
metric_name_mapping['roc_auc'] = {'function': roc_auc_single, 'argument': {}}
metric_name_mapping['bed_roc_auc'] = {'function': bedroc_auc_single, 'argument': {}}
metric_name_mapping['precision_auc_single'] = {'function': precision_auc_single, 'argument': {}}

metric_name_mapping['nef_001'] = {'function': normalized_enrichment_factor_single,  'argument': {'percentile': 0.01}}
metric_name_mapping['nef_01'] = {'function': normalized_enrichment_factor_single,  'argument': {'percentile': 0.1}}
metric_name_mapping['nef_02'] = {'function': normalized_enrichment_factor_single,  'argument': {'percentile': 0.2}}

metric_name_mapping['number_of_hit_250'] =  {'function': number_of_hit_single, 'argument': {'N': 250}}
metric_name_mapping['number_of_hit_500'] = {'function': number_of_hit_single, 'argument': {'N': 500}}
metric_name_mapping['number_of_hit_1000'] = {'function': number_of_hit_single, 'argument': {'N': 1000}}

metric_name_mapping['ratio_of_hit_001'] = {'function': ratio_of_hit_single, 'argument': {'R': 0.01}}
metric_name_mapping['ratio_of_hit_01'] = {'function': ratio_of_hit_single, 'argument': {'R': 0.1}}
metric_name_mapping['ratio_of_hit_02'] = {'function': ratio_of_hit_single, 'argument': {'R': 0.2}}


def collectively_drop_nan(actual, predicted):
    existing_index = map(lambda x: True if np.isfinite(x) else False, predicted)
    actual = actual[existing_index]
    predicted = predicted[existing_index]
    return actual, predicted


def from_ndarray_to_rank(ndarray):
    rank = ndarray.argsort().argsort()
    return rank


def plot_metric_comparison(metric_df):
    metric_names = metric_name_mapping.keys()
    model_names = metric_df['Model'].tolist()
    print metric_names

    N = len(metric_names)
    fig, axs = plt.subplots(N, N, figsize=(N * 2.5, N * 2.5), sharex='col', sharey='row')
    plt.subplots_adjust(wspace=0.1)

    for idx_a, metric_a in enumerate(metric_names):
        eval_a = metric_df[metric_a].as_matrix()
        rank_a = from_ndarray_to_rank(-eval_a)

        for idx_b, metric_b in enumerate(metric_names):
            if idx_a == idx_b:
                axs[idx_a, idx_b].text(0.5, 0.5, '', fontsize=18, ha='center')
            else:
                eval_b = metric_df[metric_b].as_matrix()
                rank_b = from_ndarray_to_rank(-eval_b)
                axs[idx_a, idx_b].scatter(rank_a, rank_b)
                axs[idx_a, idx_b].set_xlim([0, len(model_names)])
                axs[idx_a, idx_b].set_ylim([0, len(model_names)])

            if idx_b == 0:
                axs[idx_a, idx_b].axes.set_ylabel(metric_a)
            if idx_a == N - 1:
                axs[idx_a, idx_b].axes.set_xlabel(metric_b)

            axs[idx_a, idx_b].xaxis.set_ticks_position('none')
            axs[idx_a, idx_b].yaxis.set_ticks_position('none')

            #             if idx_a == 0:
            #                 axs[idx_a, idx_b].xaxis.set_ticks_position('top')
            #             else:
            #                 axs[idx_a, idx_b].set_xticks([])
            #             if idx_b == N - 1:
            #                 axs[idx_a, idx_b].yaxis.set_ticks_position('right')
            #             else:
            #                 axs[idx_a, idx_b].set_yticks([])

            if idx_b >= N - 1:
                break
        if idx_a >= N - 1:
            break

    plt.tight_layout()
    plt.savefig('./plottings/prospective_screening_metric_comparison/metric_comparison', bbox_inches='tight')
    plt.show()