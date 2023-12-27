# -*- coding: utf-8 -*-
"""
Created on Sat May 28 10:41:21 2022

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, too-many-arguments, too-many-branches, too-many-locals, useless-object-inheritance
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
from separation_mix_shipsear_s0tos3_preprocess import z_labels_create
from train_separation_multiple_autoencoder import index_mix_src_ns


def query_targets_of_input(n_src, name_input, str_end='zero'):
    """Query targets of the input.
    Args:
        n_src (int): number of the target, including one background noise.
        name_input (str): name of the input source.
        str_end (str, optional): for file name. Defaults to 'zero'.
    Returns:
        name_targets (list[str]): names of the target signals.
    """
    index_targets = [(index_i,) for index_i in range(n_src)]+index_mix_src_ns(n_src)
    names_targets = [[f'Z_{i}_{str_end}' for i in mix_i] for mix_i in index_targets]
    names_inputs = [f'Z_{i}_{str_end}' for i in range(len(index_targets))]
    index_target = names_inputs.index(name_input)
    name_targets = names_targets[index_target]
    return name_targets


def query_metric_from_file(path_result, name_input, name_channel, name_target, metric_key, name_set, table_num,
                           name_targets, name_sets=['train', 'val', 'test'], name_para=None, way=0):
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
    Returns:
        metric (float32) : the metric value.
    """
    metric = None
    if table_num == 0:  # table 0
        # if name_input == name_target:  # single target
        #     if metric_key in ('mse', 'sr', 'si_snr'):
        #         path_metric = os.path.join(path_result, name_input, name_channel)
        #         metric = read_data(path_metric, metric_key, dict_key=f'{metric_key}_mean')[name_sets.index(name_set)]
        # else:  # multiple targets
        path_metric = os.path.join(path_result, name_input[:-len('_ns')], 'metric')
        if metric_key in ('sdr', 'sar', 'sir', 'isr',
                          'si_sdr', 'si_sar', 'si_sir', 'si_sir'
                          'sd_sdr', 'sd_sar', 'sd_sir', 'sd_sir'):
            metric = read_data(path_metric, f'{metric_key}_mean', dict_key=name_para
                               )[name_set][name_targets.index(name_target)]
        elif metric_key in ('mse', 'sr', 'si_snr'):
            metric = read_data(path_metric, f'{metric_key}_mean', dict_key=name_para
                               )[name_target][name_sets.index(name_set)]
    elif table_num == 1:  # table 1
        if name_input == name_target:  # single target
            if metric_key in ('mse', 'sr', 'si_snr'):
                path_metric = os.path.join(path_result, name_input, name_channel)
                if way == 0:
                    metric = read_data(path_metric, metric_key,
                                       dict_key=f'{metric_key}_mean')[name_sets.index(name_set)]
                elif way == 1:
                    metric = np.float32(read_data(path_metric, f'{metric_key}_{name_set}',
                                                  dict_key=f'{metric_key}_mean'))
        else:  # multiple targets
            if metric_key in ('sdr', 'sar', 'sir', 'isr',
                              'si_sdr', 'si_sar', 'si_sir', 'si_sir'
                              'sd_sdr', 'sd_sar', 'sd_sir', 'sd_sir'):
                path_metric = os.path.join(path_result, name_input)
                metric = read_data(path_metric, f'{metric_key}_{name_set}',
                                   dict_key=f'{metric_key}_{name_set}_mean')[name_targets.index(name_target)]
            elif metric_key in ('mse', 'sr', 'si_snr'):
                path_metric = os.path.join(path_result, name_input, name_channel)
                if way == 0:
                    metric = read_data(path_metric, metric_key,
                                       dict_key=f'{metric_key}_mean')[name_sets.index(name_set)]
                elif way == 1:
                    metric = np.float32(read_data(path_metric, f'{metric_key}_{name_set}',
                                                  dict_key=f'{metric_key}_mean'))
    elif table_num == 2:  # table 2
        if name_input == name_target:
            if metric_key in ('sr', 'si_snr'):
                path_metric = os.path.join(path_result, name_input, name_channel, name_target)
                if way == 0:
                    metric = read_data(path_metric, metric_key,
                                       dict_key=f'{metric_key}_mean')[name_sets.index(name_set)]
                elif way == 1:
                    metric = np.float32(read_data(path_metric, f'{metric_key}_{name_set}',
                                                  dict_key=f'{metric_key}_mean'))
        elif name_channel == name_target:
            if metric_key in ('mse',):
                path_metric = os.path.join(path_result, name_input, name_channel)
                if way == 0:
                    metric = read_data(path_metric, metric_key,
                                       dict_key=f'{metric_key}_mean')[name_sets.index(name_set)]
                elif way == 1:
                    metric = np.float32(read_data(path_metric, f'{metric_key}_{name_set}',
                                                  dict_key=f'{metric_key}_mean'))
        else:
            if metric_key in ('sr', 'si_snr'):
                path_metric = os.path.join(path_result, name_input, name_channel, name_target)
                if way == 0:
                    metric = read_data(path_metric, metric_key, dict_key=f'{metric_key}_mean')[name_sets.index(name_set)]
                elif way == 1:
                    metric = np.float32(read_data(path_metric, f'{metric_key}_{name_set}',
                                                  dict_key=f'{metric_key}_mean'))
    return metric


def metric_to_csv(n_src, path_result, names_inputs, names_channels, names_targets,
                  metric_key, name_set, name_sets_all, table_num, path_save, name_para=None, way=0):
    """Extract metric to .csv file.
    Args:
        n_src (int): number of the sources, same as number of channels.
        path_result (str): path where to save files.
        names_inputs (list[str]): names of the input sources.
        names_channels (list[str]): names of the channels.
        names_targets (list[str]): names of the target signals.
        metric_key (str): keyword name of the metric.
        name_set (str): name of the dataset. e.g. 'train', 'val', 'test'.
        name_sets_all (list, optional): names of the datasets. Defaults to ['train', 'val', 'test'].
        table_num (int): number of the table.
        path_save (str): path where to save the table.
        name_para (str, optional): name of the parameters for train model. Defaults to None.
        way (int, optional): the way data saved. Defaults to 0.
    Returns:
        metric_df (pd.DataFrame): DataFrame of the metric.
    """
    metrics = []
    for name_input_i, name_channel_i, name_target_i in zip(names_inputs, names_channels, names_targets):
        str_end = 'ns' if table_num == 0 else 'zero'
        name_input_i = table_names_to_result_names(name_input_i, n_src, str_end=str_end)
        name_channel_i = table_names_to_result_names(name_channel_i, n_src, str_end=str_end)
        name_target_i = table_names_to_result_names(name_target_i, n_src, str_end=str_end)
        name_targets_i = query_targets_of_input(n_src, name_input_i, str_end=str_end)
        metric_i = query_metric_from_file(path_result, name_input_i, name_channel_i, name_target_i,
                                          metric_key, name_set, table_num,
                                          name_sets=name_sets_all, name_targets=name_targets_i, name_para=name_para,
                                          way=way)
        metrics.append(metric_i)
    metric_dict = {'input': names_inputs, 'channel': names_channels, 'target': names_targets, metric_key: metrics}
    metric_df = pd.DataFrame(metric_dict)
    metric_df.to_csv(os.path.join(path_save, f'table_{table_num}_{metric_key}_{name_set}.csv'))
    return metric_df


def create_source_names(name_srcs=['A', 'B', 'C', 'D']):
    """Create names of sources for table.
    Args:
        name_srcs (list[str], optional): names of the single sources. Defaults to ['A', 'B', 'C', 'D'].
    Returns:
        names_sources (list[str]): names of the mixtures, including single target and multiple targets.
    Examples:
        >>> print(create_source_names())
        ['A', 'B', 'C', 'D', 'BC', 'BD', 'CD', 'BCD']
    """
    names_sources = []
    source_labels = z_labels_create(len(name_srcs))
    for label_i in source_labels:
        name_i = str()
        for j, label_j in enumerate(label_i):
            if label_j == 1:
                name_i += name_srcs[j]
        names_sources.append(name_i)
    return names_sources


def create_table_1_names(names_sources, n_src):
    """Create the text lines of the tables for same channel and target.
        Evaluate the similarity between the separation predict and real targets.
    Args:
        names_sources (list[str]): names of the mixtures, including single target and multiple targets.
        n_src (int): number of the target, including one background noise.
    Returns:
        names_inputs (list[str]): names of the input mixture signals.
        names_channels (list[str]): names of the channels for the specific targets.
        names_targets (list[str]): names of the targets.
    Examples:
        names_inputs, names_channels, names_targets = create_table_1_names(create_source_names(), 4)
        >>> print(names_inputs)
        ['A', 'B', 'C', 'D', 'BC', 'BC', 'BD', 'BD', 'CD', 'CD', 'BCD', 'BCD', 'BCD']
        >>> print(names_channels)
        ['A', 'B', 'C', 'D', 'B', 'C', 'B', 'D', 'C', 'D', 'B', 'C', 'D']
        >>> print(names_targets)
        ['A', 'B', 'C', 'D', 'B', 'C', 'B', 'D', 'C', 'D', 'B', 'C', 'D']
    """
    names_inputs, names_channels, names_targets = [], [], []
    for i in range(n_src):  # single target
        names_inputs.append(names_sources[i])
        names_channels.append(names_sources[i])
        names_targets.append(names_sources[i])
    mix_srcs = index_mix_src_ns(n_src)
    for i in range(n_src, len(names_sources)):  # multitarget
        name_input_i = names_sources[i]
        for mix_src_j in mix_srcs[i-n_src]:
            name_channel_j = names_sources[mix_src_j]
            names_inputs.append(name_input_i)
            names_channels.append(name_channel_j)
            names_targets.append(name_channel_j)
    return names_inputs, names_channels, names_targets


def create_table_2_names(names_sources, n_src, metric_key):
    """Create the text lines of the tables for different channel and target.
        Evaluate how bad the channel predict between the silence signal.
    Args:
        names_sources (list[str]): names of the mixtures, including single target and multiple targets.
        n_src (int): number of the target, including one background noise.
        metric_key (str): name of the metric.
    Returns:
        names_inputs (list[str]): names of the input mixture signals.
        names_channels (list[str]): names of the channels for the specific targets.
        names_targets (list[str]): names of the targets.
    Examples:
        names_inputs, names_channels, names_targets = create_table_2_names(create_source_names(), 4, 'sr')
        >>> print(names_inputs)
        ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D',
         'BC', 'BC', 'BC', 'BC', 'BD', 'BD', 'BD', 'BD', 'CD', 'CD', 'CD', 'CD', 'BCD', 'BCD', 'BCD']
        >>> print(names_channels)
        ['B', 'C', 'D', 'A', 'C', 'D', 'A', 'B', 'D', 'A', 'B', 'C',
         'A', 'A', 'D', 'D', 'A', 'A', 'C', 'C', 'A', 'A', 'B', 'B', 'A', 'A', 'A']
        >>> print(names_targets)
        ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D',
         'B', 'C', 'B', 'C', 'B', 'D', 'B', 'D', 'C', 'D', 'C', 'D', 'B', 'C', 'D']
        names_inputs, names_channels, names_targets = create_table_2_names(create_source_names(), 4, 'mse')
        >>> print(names_inputs)
        ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D', 'BC', 'BC', 'BD', 'BD', 'CD', 'CD', 'BCD']
        >>> print(names_channels)
        ['B', 'C', 'D', 'A', 'C', 'D', 'A', 'B', 'D', 'A', 'B', 'C', 'A', 'D', 'A', 'C', 'A', 'B', 'A']
        >>> print(names_targets)
        ['B', 'C', 'D', 'A', 'C', 'D', 'A', 'B', 'D', 'A', 'B', 'C', 'A', 'D', 'A', 'C', 'A', 'B', 'A']
    """
    if metric_key in ('sr', 'si_snr'):
        names_inputs, names_channels, names_targets = [], [], []
        for i in range(n_src):  # single target
            name_channels_i = [name_source_j for name_source_j in names_sources[:n_src]]
            name_channels_i.pop(i)
            for name_source_j in name_channels_i:
                names_inputs.append(names_sources[i])
                names_channels.append(name_source_j)
                names_targets.append(names_sources[i])
        mix_targets = index_mix_src_ns(n_src)
        for i in range(n_src, len(names_sources)):
            mix_targets_i = mix_targets[i-n_src]
            mute_channels_i = list(range(n_src))
            for mix_target_j in mix_targets_i:
                mute_channels_i.remove(mix_target_j)
            for channel_j in mute_channels_i:
                for target_k in mix_targets_i:
                    names_inputs.append(names_sources[i])
                    names_channels.append(names_sources[channel_j])
                    names_targets.append(names_sources[target_k])
    elif metric_key in ('mse',):
        names_inputs, names_channels, names_targets = [], [], []
        for i in range(n_src):  # single target
            name_channels_i = [name_source_j for name_source_j in names_sources[:n_src]]
            name_channels_i.pop(i)
            for name_source_j in name_channels_i:
                names_inputs.append(names_sources[i])
                names_channels.append(name_source_j)
                names_targets.append(name_source_j)
        mix_targets = index_mix_src_ns(n_src)
        for i in range(n_src, len(names_sources)):
            mix_targets_i = mix_targets[i-n_src]
            mute_channels_i = list(range(n_src))
            for mix_target_j in mix_targets_i:
                mute_channels_i.remove(mix_target_j)
            for channel_j in mute_channels_i:
                names_inputs.append(names_sources[i])
                names_channels.append(names_sources[channel_j])
                names_targets.append(names_sources[channel_j])
    return names_inputs, names_channels, names_targets


def table_names_to_result_names(name_source, n_src, name_srcs=['A', 'B', 'C', 'D'], str_end='zero'):
    """Convert name of the source in tables to name in saved results.
    Args:
        name_source (str): name of the source to convert.
        n_src (int): number of the target, including one background noise.
        name_srcs (list[str], optional): names of the single sources. Defaults to ['A', 'B', 'C', 'D'].
        str_end (str, optional): end sting of names of the files. Defaults to 'zero'.
    Returns:
        name_result_source (str): name of the source in saved results.
    Examples:
        >>> print(table_names_to_result_names('A', 4))
        Z_0
        >>> print(table_names_to_result_names('BC', 4))
        Z_4
    """
    names_sources = create_source_names(name_srcs)
    n_sources = n_src+len(index_mix_src_ns(n_src))
    names_result_sources = [f'Z_{i}_{str_end}' for i in range(n_sources)]
    index_source = names_sources.index(name_source)
    name_result_source = names_result_sources[index_source]
    return name_result_source


if __name__ == '__main__':
    import logging
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO)

    n_src = 4
    names_sources = create_source_names()
    names_result_sources = [f'Z_{i}_zero' for i in range(len(names_sources))]

    def create_table_algorithm_1_3(path_result):
        names_inputs_1, names_channels_1, names_targets_1 = create_table_1_names(names_sources, n_src)
        for metric_key_i in ['mse', 'sr', 'si_snr', 'sdr']:
            for name_set_j in ['train', 'val', 'test']:
                metric_to_csv(n_src, path_result, names_inputs_1, names_channels_1, names_targets_1, metric_key_i,
                            name_set_j, ['train', 'val', 'test'], 1, path_result)

        for metric_key_i in ['mse', 'sr', 'si_snr']:
            names_inputs_2_i, names_channels_2_i, names_targets_2_i = create_table_2_names(
                names_sources, n_src, metric_key_i)
            for name_set_j in ['train', 'val', 'test']:
                metric_to_csv(n_src, path_result, names_inputs_2_i, names_channels_2_i, names_targets_2_i, metric_key_i,
                            name_set_j, ['train', 'val', 'test'], 2, path_result)

    # Algorithm 1
    path_result_root = '../result_separation_one_autoencoder/'

    path_result = path_result_root+'model_8_2_1/metric/1_n3_100'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_10_2_1/metric/1_n3_100'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_10_3_1/metric/1_n3_100'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_11_2_1/metric/1_n3_100'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_12_2_1/metric/1_n3_100'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_12_3_1/metric/1_n3_100'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_13_2_1/metric/1_n3_100'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_13_3_1/metric/1_n3_100'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_14_1_6/metric/1_n3_200'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_14_2_6/metric/1_n3_200'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_15_2_6/metric/1_n3_200'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_20_5_10/metric/1_n4_800'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_20_6_10/metric/1_n4_800'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_21_5_10/metric/1_n4_800'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_21_6_10/metric/1_n4_800'
    create_table_algorithm_1_3(path_result)

    # Algorithm 3
    path_result_root = '../result_separation_ae_ns_single/'

    path_result = path_result_root+'model_8_2_1/load_decoder/'
    path_result += 'model_1_n3_100/metric/1_n3_100'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_13_2_1/load_decoder/'
    path_result += 'model_1_n3_100/metric/1_n3_100'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_13_3_1/load_decoder/'
    path_result += 'model_1_n3_100/metric/1_n3_50'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_15_2_6/load_decoder/'
    path_result += 'model_1_n3_200/metric/1_n3_200'
    create_table_algorithm_1_3(path_result)

    path_result = path_result_root+'model_21_6_10/load_decoder/'
    path_result += '1_n4_1600/metric/1_n4_800'
    create_table_algorithm_1_3(path_result)

    def create_table_algorithm_2(path_result):
        names_inputs_1, names_channels_1, names_targets_1 = create_table_1_names(names_sources, n_src)
        for metric_key_i in ['mse', 'sr', 'si_snr', 'sdr']:
            for name_set_j in ['test']:
                metric_to_csv(n_src, path_result, names_inputs_1, names_channels_1, names_targets_1, metric_key_i,
                            name_set_j, ['train', 'val', 'test'], 1, path_result)

        for metric_key_i in ['mse', 'sr', 'si_snr']:
            names_inputs_2_i, names_channels_2_i, names_targets_2_i = create_table_2_names(
                names_sources, n_src, metric_key_i)
            for name_set_j in ['test']:
                metric_to_csv(n_src, path_result, names_inputs_2_i, names_channels_2_i, names_targets_2_i, metric_key_i,
                            name_set_j, ['train', 'val', 'test'], 2, path_result)

    # Algorithm 2
    path_result_root = '../result_separation_ae_ns_single/'

    path_result = path_result_root+'model_13_2_1/search_decoder/'
    path_result += 'model_1_n3_100/metric/best_mse'
    create_table_algorithm_2(path_result)

    path_result = path_result_root+'model_15_2_6/search_decoder/'
    path_result += 'model_1_n3_200/metric/best_mse'
    create_table_algorithm_2(path_result)

    path_result = path_result_root+'model_15_2_6_tanh/search_decoder/'
    path_result += 'model_1_n3_200/metric/best_mse'
    create_table_algorithm_2(path_result)

    path_result = path_result_root+'model_21_6_10/search_decoder/'
    path_result += '1_n4_1600/metric/best_mse'
    create_table_algorithm_2(path_result)

    def create_table_known_number(path_result, name_para):
        names_inputs_0, names_channels_0, names_targets_0 = create_table_1_names(names_sources, n_src)
        names_inputs_0_mix = names_inputs_0[n_src:]
        names_channels_0_mix = names_channels_0[n_src:]
        names_targets_0_mix = names_targets_0[n_src:]
        for metric_key_i in ['mse', 'sr', 'si_sdr', 'sdr']:
            for name_set_j in ['train', 'val', 'test']:
                metric_to_csv(n_src, path_result, names_inputs_0_mix, names_channels_0_mix, names_targets_0_mix,
                            metric_key_i, name_set_j, ['train', 'val', 'test'], 0, path_result,
                            name_para=name_para)

    names_sources_mix = names_sources[n_src:]
    names_result_sources_mix = names_result_sources[n_src:]

    # Known number
    # path_result_root = '../result_separation_multiple_autoencoder/'
    path_result_root = '../result_separation_known_number/'

    path_result = path_result_root+'model_8_2_1/'
    name_para = '1_n3_50'
    create_table_known_number(path_result, name_para)

    path_result = path_result_root+'model_8_3_1/'
    name_para = '1_n3_100'
    create_table_known_number(path_result, name_para)

    path_result = path_result_root+'model_10_2_1/'
    name_para = '1_n3_100'
    create_table_known_number(path_result, name_para)

    path_result = path_result_root+'model_10_3_1/'
    name_para = '1_n3_100'
    create_table_known_number(path_result, name_para)

    path_result = path_result_root+'model_13_2_1/'
    name_para = '1_n3_100'
    create_table_known_number(path_result, name_para)

    path_result = path_result_root+'model_13_3_1/'
    name_para = '1_n3_100'
    create_table_known_number(path_result, name_para)

    path_result = path_result_root+'model_14_1_6/'
    name_para = '1_n3_200'
    create_table_known_number(path_result, name_para)

    path_result = path_result_root+'model_15_2_6/'
    name_para = '1_n3_200'
    create_table_known_number(path_result, name_para)

    path_result = path_result_root+'model_20_5_10/'
    name_para = '1_n4_400'
    create_table_known_number(path_result, name_para)

    path_result = path_result_root+'model_21_6_10/'
    name_para = '1_n3_800'
    create_table_known_number(path_result, name_para)

    logging.info('finished')
