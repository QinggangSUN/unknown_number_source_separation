# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 11:45:13 2023

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import museval
import pandas as pd
import seaborn as sns

from error import ParameterError
from file_operation import list_dirs
from metric import samerate_acc_np, mse_np
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data, save_datas
from see_metric_separation_extract_table import create_source_names, create_table_1_names, query_targets_of_input, \
    table_names_to_result_names


def query_metric_from_file(path_result, name_input, name_channel, name_target, metric_key, name_set, table_num,
                           name_targets, name_sets=('train', 'val', 'test'), name_para=None, way=0, algo_type=1):
    """Query metric from file.
    Args:
        path_result (str): path where to save files.
        name_input (str): name of the input source.
        name_channel (str): name of the channel.
        name_target (str): name of the target signal.
        metric_key (str): keyword name of the metric.
        name_set (str): name of the dataset. e.g. 'train', 'val', 'test'.
        table_num (int): number of the table.
        name_targets (list[str]): names of the target signals.
        name_sets (list, optional): names of the datasets. Defaults to ['train', 'val', 'test'].
        name_para (str, optional): name of the parameters for train model. Defaults to None.
        way (int, optional): the way data saved. Defaults to 0.
        algo_type (int, optional): type of the algorithm. Defaults to 1.
    Returns:
        metric (float32) : the metric value.
    """
    metric = None

    if table_num == 1:  # table 1
        if metric_key in ('mse', 'sr', 'si_snr'):
            if algo_type == 0:
                path_metric = os.path.join(path_result, name_input[:-3], 'metric', name_para, name_channel)
            elif algo_type >= 1:
                path_metric = os.path.join(path_result, name_input, name_channel)
            if way == 0:
                file_name = metric_key
            elif way == 1:
                file_name = f'{metric_key}_{name_set}'
            metric = read_data(path_metric, file_name, dict_key=metric_key)[name_set]
        elif metric_key in ('sdr', 'sar', 'sir', 'isr',
                            'si_sdr', 'si_sar', 'si_sir', 'si_sir',
                            'sd_sdr', 'sd_sar', 'sd_sir', 'sd_sir'):
            if algo_type == 0:
                path_metric = os.path.join(path_result, name_input[:-3], 'metric', name_para)
            elif algo_type >= 1:
                path_metric = os.path.join(path_result, name_input)
            metric = read_data(path_metric, f'{metric_key}_{name_set}',
                               dict_key=f'{metric_key}_{name_set}')[name_target]
    elif table_num == 2:  # table 2
        pass
    return metric


def compute_sum_target_numbers_same(n_src, bool_include_one=True):
    """Compute sum of the numbers of the targets with same number of targets.
        e.g. 4 one targets, 3 two targets, 1 three targets.
    Args:
        n_src (int): number of the sources, same as number of channels.
        bool_include_one (bool, optional): whether including one. Defaults to True.
    Return:
        num_list (list(int)): list of the numbers.
    Example:
        # >>> print(compute_sum_target_numbers_same(n_src=4))
        [4, 3, 1]
    """
    from itertools import combinations
    if bool_include_one:
        num_list = [n_src, ]
    else:
        num_list = []
    for i in range(2, n_src):
        index_channels_i = list(combinations(range(1, n_src), i))
        num_channels_i = 0
        for index_channels_j in index_channels_i:
            num_channels_i += len(index_channels_j)
        num_list.append(num_channels_i)
    return num_list


def metric_hdf5_to_df(n_src, path_result, names_inputs, names_channels, names_targets,
                      metric_key, name_set, name_sets_all, table_num, path_save=None, name_para=None, way=0,
                      algorithm_type=None, model_type=None):
    """Extract metric from .hdf5 to pandas DataFrame.
    Args:
        n_src (int): number of the sources, same as number of channels.
        path_result (str): path where to save files.
        names_inputs (list[str]): names of the input sources.
        names_channels (list[str]): names of the channels.
        names_targets (list[str]): names of the target signals.
        metric_key (str): keyword name of the metric.
        name_set (str): name of the dataset. e.g. 'train', 'val', 'test'.
        name_sets_all (tuple): names of the datasets. e.g. ('train', 'val', 'test').
        table_num (int): number of the table.
        path_save (str, optional): path where to save the table. Defaults to None.
        name_para (str, optional): name of the parameters for train model. Defaults to None.
        way (int, optional): the way data saved. Defaults to 0.
        algorithm_type (int, optional): type of the algorithm. Defaults to None.
        model_type (str, optional): the way data saved. Defaults to None.
    Returns:
        metric_df (pd.DataFrame): DataFrame of the metric.
    """
    metric_mode = None
    if metric_key in ('mse', 'sr', 'si_snr'):
        metric_mode = 0
    elif metric_key in ('sdr', 'sar', 'sir', 'isr',
                        'si_sdr', 'si_sar', 'si_sir', 'si_sir',
                        'sd_sdr', 'sd_sar', 'sd_sir', 'sd_sir'):
        metric_mode = 1

    str_end = 'ns' if algorithm_type == 0 else 'zero'
    if algorithm_type == 0:
        names_inputs = names_inputs[n_src:]
        names_channels = names_channels[n_src:]
        names_targets = names_targets[n_src:]

    metrics = []
    for name_input_i, name_channel_i, name_target_i in zip(names_inputs, names_channels, names_targets):
        name_input_i = table_names_to_result_names(name_input_i, n_src, str_end=str_end)
        name_channel_i = table_names_to_result_names(name_channel_i, n_src, str_end=str_end)
        name_target_i = table_names_to_result_names(name_target_i, n_src, str_end=str_end)
        name_targets_i = query_targets_of_input(n_src, name_input_i, str_end=str_end)
        if (metric_mode == 0) or (metric_mode == 1 and name_input_i != name_target_i):
            metric_i = query_metric_from_file(path_result, name_input_i, name_channel_i, name_target_i,
                                              metric_key, name_set, table_num,
                                              name_sets=name_sets_all, name_targets=name_targets_i, name_para=name_para,
                                              way=way, algo_type=algorithm_type)
            metrics.append(metric_i)

    if algorithm_type == 0:
        metrics_targets = [None, ]
    else:
        metrics_targets = []
    bool_include_one, num_min, num_max = None, None, None
    if metric_mode == 1 or algorithm_type == 0:
        bool_include_one = False
    elif metric_mode == 0:
        bool_include_one = True
    if metric_mode == 0 or algorithm_type == 0:
        num_min = 1
    elif metric_mode == 1:
        num_min = 2

    num_sum_target_list = compute_sum_target_numbers_same(n_src, bool_include_one)
    for i, num_sum_target_i in enumerate(num_sum_target_list):
        if i == 0:
            metrics_targets.append(np.hstack(metrics[0:num_sum_target_i]).tolist())
        else:
            metrics_targets.append(
                np.hstack(metrics[num_sum_target_list[i - 1]:num_sum_target_list[i - 1] + num_sum_target_i]).tolist())

    if metric_mode == 1 or algorithm_type == 0:
        num_max = len(num_sum_target_list) + 2
    elif metric_mode == 0:
        num_max = len(num_sum_target_list) + 1

    metric_dict = {'target_number': list(range(num_min, num_max)), 'metric': metrics_targets,
                   'metric_key': [metric_key, ] * len(metrics_targets)}
    if algorithm_type is not None:
        metric_dict.update({'algorithm_type': algorithm_type})
    if model_type:
        metric_dict.update({'model_type': model_type})

    metric_df = pd.DataFrame(metric_dict)
    if path_save:
        metric_df.to_json(os.path.join(path_save, f'figure_table_{table_num}_{metric_key}_{name_set}.json'))
        # metric_df_read = pd.read_json(os.path.join(path_save, f'figure_table_{table_num}_{metric_key}_{name_set}.json'))
        # metric_df_read_values = metric_df_read['metric'].values

    return metric_df


class DisplayMetric(object):
    """Display predict metric."""

    def __init__(self, info_list):
        self.info_list = info_list

    def import_data_to_df(self, n_src, names_inputs_1, names_channels_1, names_targets_1, metric_key, name_set):
        """Import data from .hdf5 files to pandas DataFrame.
        Args:
            n_src (int): number of the sources, same as number of channels.
            names_inputs_1 (list[str]): names of the input mixture signals from table 1.
            names_channels_1 (list[str]): names of the channels for the specific targets from table 1.
            names_targets_1 (list[str]): names of the targets from table 1.
            metric_key (str): keyword name of the metric.
            name_set (str): name of the dataset. e.g. 'train', 'val', 'test'.
        Returns:
            metrics_df (pd.DataFrame): DataFrame of the metrics.
        """
        info_list = self.info_list
        metric_df_list = []
        for info_dict_i in info_list:
            name_para = info_dict_i['name_para'] if info_dict_i['algorithm_type'] == 0 else None
            metric_df = metric_hdf5_to_df(n_src, info_dict_i['path'],
                                          names_inputs_1, names_channels_1, names_targets_1, metric_key,
                                          name_set, ('train', 'val', 'test'), 1, path_save=info_dict_i['path'],
                                          name_para=name_para,
                                          way=info_dict_i['way'],
                                          algorithm_type=info_dict_i['algorithm_type'],
                                          model_type=info_dict_i['model_type'])
            metric_df_list.append(metric_df)
        metrics_df = pd.concat(metric_df_list)
        return metrics_df

    def display_cat_violin(self, metrics_df, metric_key, path_save=None, save_name=None, num_algo=3, save_mode=1):
        """Display predict metric violin by channel.
        Args:
            metrics_df (pd.DataFrame): DataFrame of the metrics.
            metric_key (str): keyword name of the metric.
            path_save (str): path where to save the figures.
            save_name (str, optional): front keywords of the saved files. Defaults to None.
            num_algo (int, optional): number of the algorithms. Defaults to 3.
            save_mode (int, optional): mode for save files. Defaults to 1.
        """
        config_dict = {}
        if metric_key in ('mse',):
            config_dict.update({'cut': 0})

        if save_mode == 1:
            fig, axes = plt.subplots(1, num_algo+1, sharey='all', figsize=(40, 15), dpi=1000.0)
        for i_algo in range(0, num_algo + 1):
            metric_i_algo = metrics_df[metrics_df.algorithm_type == i_algo]
            metric_i_algo = metric_i_algo.loc[:, ['model_type', 'target_number', 'metric']]
            metric_i_algo = metric_i_algo.explode('metric')
            metric_i_algo['metric'] = metric_i_algo['metric'].astype('float32')
            if save_mode == 0 and path_save:
                cat = sns.catplot(data=metric_i_algo,
                                  x='model_type', y='metric', hue='target_number',
                                  kind='violin', scale='count', **config_dict
                                  )
                cat.set_xlabels('Model')
                cat.set_ylabels(f'{metric_key.upper()}')
                plt.savefig(os.path.join(path_save, f'{save_name}_algo_{i_algo}_cat_violin.svg'))
                plt.close()
            if save_mode == 1:
                axes_i = sns.violinplot(ax=axes[i_algo - 1], data=metric_i_algo,
                                        x='model_type', y='metric', hue='target_number',
                                        scale='count', **config_dict
                                        )
                axes_i.set_xlabel('Model')
                axes_i.set_ylabel(f'{metric_key.upper()}')
                axes_i.set_title(f'Algorithm {i_algo}')

        if save_mode == 1 and path_save:
            plt.savefig(os.path.join(path_save, f'{save_name}_cat_violin.svg'))


if __name__ == '__main__':
    n_src = 4
    names_sources = create_source_names()
    names_result_sources = [f'Z_{i}_zero' for i in range(len(names_sources))]
    names_inputs_1, names_channels_1, names_targets_1 = create_table_1_names(names_sources, n_src)

    path_result_root = '../results'
    info_list = [{'path': f'{path_result_root}/Known Number/model_13_2_1_lr1e-3_ep100/metric', 'name_para': '1_n3_100',
                  'algorithm_type': 0, 'model_type': 'BLSTM-MD', 'way': 0},
                 {'path': f'{path_result_root}/Known Number/model_15_2_6_lr1e-3_ep200/metric', 'name_para': '1_n3_200',
                  'algorithm_type': 0, 'model_type': 'Conv-TasNet-MD', 'way': 0},
                 {'path': f'{path_result_root}/Known Number/model_21_6_10_lr1e-3_ep800/metric', 'name_para': '1_n3_800',
                  'algorithm_type': 0, 'model_type': 'Wave-U-Net-MD', 'way': 0},
                 {'path': f'{path_result_root}/Algorithm 1/model_13_2_1_lr1e-3_ep100/metric/1_n3_100',
                  'algorithm_type': 1, 'model_type': 'BLSTM-MD', 'way': 0},
                 {'path': f'{path_result_root}/Algorithm 1/model_15_2_6_lr1e-3_ep200/metric/1_n3_200',
                   'algorithm_type': 1, 'model_type': 'Conv-TasNet-MD', 'way': 0},
                 {'path': f'{path_result_root}/Algorithm 1/model_21_6_10_lr1e-4_ep800/metric/1_n4_800',
                  'algorithm_type': 1, 'model_type': 'Wave-U-Net-MD', 'way': 0},
                 {'path': f'{path_result_root}/Algorithm 2/model_13_2_1_lr1e-3_ep100/metric/best_mse',
                  'algorithm_type': 2, 'model_type': 'BLSTM-MD', 'way': 1},
                 {'path': f'{path_result_root}/Algorithm 2/model_15_2_6_lr1e-3_ep200_tanh_noNL/metric/best_mse',
                   'algorithm_type': 2, 'model_type': 'Conv-TasNet-MD', 'way': 1},
                 {'path': f'{path_result_root}/Algorithm 2/model_21_6_10_lr1e-4_ep1600/metric/best_mse',
                   'algorithm_type': 2, 'model_type': 'Wave-U-Net-MD', 'way': 1},
                 {'path': f'{path_result_root}/Algorithm 3/model_13_2_1_lr1e-3_ep100/model_lr1e-3_ep100/metric/1_n3_100',
                   'algorithm_type': 3, 'model_type': 'BLSTM-MD', 'way': 0},
                 {'path': f'{path_result_root}/Algorithm 3/model_15_2_6_lr1e-3_ep200/model_lr1e-3_ep200/metric/1_n3_200',
                   'algorithm_type': 3, 'model_type': 'Conv-TasNet-MD', 'way': 0},
                 {'path': f'{path_result_root}/Algorithm 3/model_21_6_10_lr1e-4_ep1600/model_lr1e-4_ep800/metric/1_n4_800',
                   'algorithm_type': 3, 'model_type': 'Wave-U-Net-MD', 'way': 0}
                ]

    display_metric_mse = DisplayMetric(info_list)
    metrics_df = display_metric_mse.import_data_to_df(n_src, names_inputs_1, names_channels_1, names_targets_1, 'mse',
                                                      'test')
    display_metric_mse.display_cat_violin(metrics_df, 'mse', path_save=path_result_root, save_name='fig_2_mse', save_mode=0)
    display_metric_mse.display_cat_violin(metrics_df, 'mse', path_save=path_result_root, save_name='fig_2_mse')

    display_metric_sdr = DisplayMetric(info_list)
    metrics_df = display_metric_sdr.import_data_to_df(n_src, names_inputs_1, names_channels_1, names_targets_1, 'sdr',
                                                      'test')
    display_metric_sdr.display_cat_violin(metrics_df, 'sdr', path_save=path_result_root, save_name='fig_2_sdr', save_mode=0)
    display_metric_sdr.display_cat_violin(metrics_df, 'sdr', path_save=path_result_root, save_name='fig_2_sdr')

    display_metric_si_snr = DisplayMetric(info_list)
    metrics_df = display_metric_si_snr.import_data_to_df(n_src, names_inputs_1, names_channels_1, names_targets_1,
                                                         'si_snr', 'test')
    display_metric_si_snr.display_cat_violin(metrics_df, 'si_snr', path_save=path_result_root, save_name='fig_2_si_snr', save_mode=0)
    display_metric_si_snr.display_cat_violin(metrics_df, 'si_snr', path_save=path_result_root, save_name='fig_2_si_snr')

    print('finished')
