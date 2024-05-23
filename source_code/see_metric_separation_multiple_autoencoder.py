# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:34:19 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, too-few-public-methods, too-many-arguments, too-many-branches, too-many-function-args
# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-statements, useless-object-inheritance


import logging
import os

import numpy as np
import h5py

from file_operation import list_dirs, list_dirs_start_str, list_files_end_str, mkdir
from metric import mse_np, samerate_acc_np, si_snr_np
from prepare_data import list_transpose
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data
from see_metric_autoencoder import compute_metric, compute_sdr, compute_si_sdr, display_metric
from see_metric_autoencoder import metric_mean_src_to_file, metric_mean_to_csv, save_metric
from see_metric_autoencoder import recover_pad_num_samples, recover_pad_num_samples_list
from see_metric_single_source_ae_ns import data_reshape_same


class ListDecodedFiles(object):
    """class for list predict decoded files."""

    def __init__(self, path_save_root):
        """__init__
        Args:
            path_save_root (str): Root path where save result.
        """
        path_mix_srcs = list_dirs(path_save_root)
        name_source_models = [list_dirs_start_str(path_source, 'model_', False) for path_source in path_mix_srcs]
        # [source][model]
        path_source_models = [list_dirs_start_str(path_source, 'model_') for path_source in path_mix_srcs]
        # [source][model]

        self.name_source_models = name_source_models
        self.path_source_models = path_source_models

        path_source_model_srcs = []  # [source][model][src]
        path_source_model_src_paras = []  # [source][model][src][para_name]
        name_source_model_src_paras = []  # [source][model][src][para_name]
        path_weight_files = []  # [source][model][src][para_name][weight_file]
        name_weight_files = []  # [source][model][src][para_name][weight_file]
        for path_source_model_i in path_source_models:  # level i: [source]
            path_source_model_src_i = []  # [model][src]
            path_source_model_src_para_i = []  # [model][src][para_name]
            name_source_model_src_para_i = []  # [model][src][para_name]
            path_weight_files_i = []
            name_weight_files_i = []
            for path_source_model_j in path_source_model_i:  # level j: [model]
                path_source_model_src_j = list_dirs(os.path.join(path_source_model_j, 'auto_model'))
                path_source_model_src_para_j = []
                name_source_model_src_para_j = []
                path_weight_files_j = []
                name_weight_files_j = []
                for path_source_model_src_k in path_source_model_src_j:  # level k: [src]
                    path_source_model_src_para_k = list_dirs(path_source_model_src_k)
                    name_source_model_src_para_k = list_dirs(path_source_model_src_k, False)
                    path_weight_files_k = []
                    name_weight_files_k = []
                    for path_source_model_src_para_l in path_source_model_src_para_k:
                        path_weight_files_k.append(list_files_end_str(path_source_model_src_para_l, 'hdf5'))
                        name_weight_files_k.append(list_files_end_str(path_source_model_src_para_l, 'hdf5', False))
                    path_source_model_src_para_j.append(path_source_model_src_para_k)
                    name_source_model_src_para_j.append(name_source_model_src_para_k)
                    path_weight_files_j.append(path_weight_files_k)
                    name_weight_files_j.append(name_weight_files_k)
                path_source_model_src_i.append(path_source_model_src_j)
                path_source_model_src_para_i.append(path_source_model_src_para_j)
                name_source_model_src_para_i.append(name_source_model_src_para_j)
                path_weight_files_i.append(path_weight_files_j)
                name_weight_files_i.append(name_weight_files_j)
            path_source_model_srcs.append(path_source_model_src_i)
            path_source_model_src_paras.append(path_source_model_src_para_i)
            name_source_model_src_paras.append(name_source_model_src_para_i)
            path_weight_files.append(path_weight_files_i)
            name_weight_files.append(name_weight_files_i)

        self.path_source_model_srcs = path_source_model_srcs
        self.path_source_model_src_paras = path_source_model_src_paras
        self.name_source_model_src_paras = name_source_model_src_paras
        self.path_weight_files = path_weight_files
        self.name_weight_files = name_weight_files

        path_out_models = []  # [source][model][src][para_name]
        for path_source_model_i, name_source_model_src_para_i in zip(  # level i: [source]
                path_source_models, name_source_model_src_paras):
            path_out_model_i = []  # [model][para]
            for path_source_model_j, name_source_model_src_para_j in zip(  # level j: [model]
                    path_source_model_i, name_source_model_src_para_i):
                path_out_model_j = []
                for name_source_model_src_para_k in name_source_model_src_para_j:  # level k: [src]
                    path_out_model_k = []
                    for name_source_model_src_para_l in name_source_model_src_para_k:
                        path_out_model_k.append(os.path.join(path_source_model_j,
                                                             'auto_out',
                                                             name_source_model_src_para_l))
                    path_out_model_j.append(path_out_model_k)
                path_out_model_i.append(path_out_model_j)
            path_out_models.append(path_out_model_i)

        self.path_out_models = path_out_models

        path_metrics = []  # [source][model]
        for path_source_model_i, name_source_model_src_para_i in zip(  # level i: [source]
                path_source_models, name_source_model_src_paras):
            path_metric_i = []
            for path_source_model_j, name_source_model_src_para_j in zip(  # level j: [model]
                    path_source_model_i, name_source_model_src_para_i):
                path_metric_j = os.path.join(path_source_model_j, 'metric')
                mkdir(path_metric_j)
                path_metric_i.append(path_metric_j)
            path_metrics.append(path_metric_i)

        self.path_metrics = path_metrics

        path_metrics_paras = []  # [source][model][para_name]
        for path_metric_i, name_source_model_src_para_i in zip(  # level i: [source]
                path_metrics, name_source_model_src_paras):
            path_metric_para_i = []  # [model][para_name]
            for path_metric_j, name_source_model_src_para_j in zip(  # level j: [model]
                    path_metric_i, name_source_model_src_para_i):
                path_metric_para_j = []  # [para_name]
                for name_model_para_k in name_source_model_src_para_j[0]:
                    path_metric_para_k = os.path.join(path_metric_j, name_model_para_k)
                    mkdir(path_metric_para_k)
                    path_metric_para_j.append(path_metric_para_k)
                path_metric_para_i.append(path_metric_para_j)
            path_metrics_paras.append(path_metric_para_i)

        self.path_metrics_paras = path_metrics_paras


def compute_metrics(object_decoded_files, s_names, src_names, s_data, sm_index, set_names):
    """compute metrics.
    Args:
        object_decoded_files (class ListDecodedFiles): object for list predict decoded files
        s_names (list[list[str]]): list of datasets.
        src_names (list[str]): list of name of sources including mixed sources, e.g. ['Z_0_ns',...,'Z_7_ns'].
        s_data (list[list[h5pyFile]]): file object of datas.
        sm_index (list[tuple]): clean target source of mixed sources, e.g. [(1, 2),...,(1,2,3)].
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
    """
    name_weight_files = object_decoded_files.name_weight_files  # [source][model][src][para_name][weight_file]
    path_out_models = object_decoded_files.path_out_models  # [source][model][src][para_name]
    path_metrics_paras = object_decoded_files.path_metrics_paras  # [source][model][para_name]

    for name_weight_files_i, path_metrics_para_i, path_out_model_i, sm_index_i in zip(  # level i: [source]
            name_weight_files, path_metrics_paras, path_out_models, sm_index):
        src_name_set_i = [s_names[k] for k in sm_index_i]  # [src][set]
        src_name_i = [src_names[k] for k in sm_index_i]  # [src]
        s_data_i = [s_data[k] for k in sm_index_i]  # [src][set]
        data_s_i = list_transpose(s_data_i, warn=True)  # [set][src]

        for name_weight_files_j, path_metrics_para_j, path_out_model_j in zip(  # level j: [model]
                name_weight_files_i, path_metrics_para_i, path_out_model_i):
            name_weight_files_j = list_transpose(name_weight_files_j)  # [para_name][src][weight_file]
            path_out_model_j = list_transpose(path_out_model_j)  # [para_name][src]

            for name_weight_files_k, path_metric_para_k, path_out_model_k in zip(  # level k: [para_name]
                    name_weight_files_j, path_metrics_para_j, path_out_model_j):
                data_sp_k = []  # [src][set]
                for name_weight_files_l, path_out_model_l, src_name_l, src_name_set_l, data_s_l in zip(  # level l:[src]
                        name_weight_files_k, path_out_model_k, src_name_i, src_name_set_i, s_data_i):
                    mkdir(path_out_model_l)
                    path_metric_para_l = os.path.join(path_metric_para_k, src_name_l)
                    mkdir(path_metric_para_l)

                    name_weight_file_m = name_weight_files_l[-1]  # level m: [weight_file]
                    data_sp_l = []
                    for src_name_set_n in src_name_set_l:  # level n: [set]
                        path_sp_l = os.path.join(path_out_model_l,
                                                 f'{src_name_set_n}_autodecoded_{name_weight_file_m}.hdf5')
                        data_sp_l.append(read_data(os.path.dirname(path_sp_l), os.path.basename(path_sp_l)))
                    data_sp_k.append(data_sp_l)

                    data_sp_l = recover_pad_num_samples_list(data_s_l, data_sp_l)

                    mse_list, mse_mean = compute_metric(data_s_l, data_sp_l, mse_np)
                    display_metric(mse_list, path_metric_para_l, set_names,
                                   # hist_bins=(0, 1e-1, 10),
                                   save_name='mse')
                    save_metric(path_metric_para_l, 'mse', ['mse_mean'], [np.asarray(mse_mean)],
                                {'mse': dict(zip(set_names, mse_list))})

                    sr_list, sr_mean = compute_metric(data_s_l, data_sp_l, samerate_acc_np)
                    display_metric(sr_list, path_metric_para_l, set_names,
                                   # hist_bins=(0.6, 1.0, 40),
                                   save_name='sr')
                    save_metric(path_metric_para_l, 'sr', ['sr_mean'], [np.asarray(sr_mean)],
                                {'sr': dict(zip(set_names, sr_list))})

                    si_snr_list, si_snr_mean = compute_metric(data_s_l, data_sp_l, si_snr_np)
                    display_metric(si_snr_list, path_metric_para_l, set_names,
                                   # hist_bins=(0.6, 1.0, 40),
                                   save_name='si_snr')
                    save_metric(path_metric_para_l, 'si_snr', ['si_snr_mean'], [np.asarray(si_snr_mean)],
                                {'si_snr': dict(zip(set_names, si_snr_list))})

                data_sp_k = list_transpose(data_sp_k)  # [set][src]

                for data_s_n, data_sp_n, set_name_n in zip(
                        data_s_i, data_sp_k, set_names):
                    data_s_n = np.asarray(data_s_n).transpose((1, 0, 3, 2))
                    # (n_src, n_sams, 1, fl)->(n_sams, n_src, fl, 1)
                    data_sp_n = np.asarray(data_sp_n)
                    if data_sp_n.ndim == 3:
                        data_sp_n = np.expand_dims(data_sp_n, -2)  # (n_src, n_sams, 1, fl)
                    data_sp_n = data_sp_n.transpose((1, 0, 3, 2))  # ->(n_sams, n_src, fl, 1)

                    data_sp_n = recover_pad_num_samples(data_s_n, data_sp_n)

                    if tuple(data_s_n.shape) != tuple(data_sp_n.shape):
                        data_s_n, data_sp_n = data_reshape_same(data_s_n, data_sp_n)

                    compute_sdr(data_s_n, data_sp_n, path_metric_para_k,
                                [f'sdr_{set_name_n}', f'isr_{set_name_n}', f'sir_{set_name_n}', f'sar_{set_name_n}'],
                                src_name_i, hist_bins=[(0, 30, 30), (0, 30, 30), (0, 30, 30), (0, 30, 30)],
                                save_name=[f'sdr_{set_name_n}', f'isr_{set_name_n}',
                                           f'sir_{set_name_n}', f'sar_{set_name_n}'])

                    data_s_n = data_s_n.transpose((0, 2, 3, 1))
                    # (n_sams, n_src, fl, 1) -> (n_sams, fl, n_channel, n_src)
                    data_sp_n = data_sp_n.transpose((0, 2, 3, 1))
                    # (n_sams, n_src, fl, 1) -> (n_sams, fl, n_channel, n_src)

                    compute_si_sdr(data_s_n, data_sp_n, True, path_metric_para_k,
                                   [f'si_sdr_{set_name_n}', f'si_sir_{set_name_n}', f'si_sar_{set_name_n}'],
                                   src_name_i,
                                   # hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                                   save_name=[f'si_sdr_{set_name_n}', f'si_sir_{set_name_n}', f'si_sar_{set_name_n}'])

                    compute_si_sdr(data_s_n, data_sp_n, False, path_metric_para_k,
                                   [f'sd_sdr_{set_name_n}', f'sd_sir_{set_name_n}', f'sd_sar_{set_name_n}'],
                                   src_name_i,
                                   # hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                                   save_name=[f'sd_sdr_{set_name_n}', f'sd_sir_{set_name_n}', f'sd_sar_{set_name_n}'])


def metric_mean_mix_to_file(path_metric, set_names, metric_keys, path_save):
    """Merge metric_mean files to .hdf5 file.
    Args:
        path_metric (str): path where to save files.
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
        metric_keys (list[str]): name of the metrics.
        path_save (str): path to save merged file.
    """
    for metric_key in metric_keys:
        f_h5 = h5py.File(os.path.join(path_save, f'{metric_key}_mean.hdf5'), 'a')
        dir_names = list_dirs(path_metric, False)
        for dir_name in dir_names:
            grp_dir = f_h5.create_group(dir_name)
            path_metric_dir = os.path.join(path_metric, dir_name)
            for set_name in set_names:
                grp_dir[set_name] = np.asarray(h5py.File(os.path.join(path_metric_dir,
                                                                      f'{metric_key}_{set_name}.hdf5'),
                                                         'r')[f'{metric_key}_{set_name}_mean'], dtype=np.float32)
        f_h5.close()


def metrics_to_files(path_metrics, src_names, sm_index, set_names):  # [source][model][para_name]
    """Merged metrics files to .hdf5 file.
    Args:
        path_metrics (list[list[str]]): [source][model], path of metric dirs.
        src_names (list[str]): list of name of sources including mixed sources, e.g. ['Z_0_ns',...,'Z_7_ns'].
        sm_index (list[tuple]): clean target source of mixed sources, e.g. [(1, 2),...,(1,2,3)].
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
    """
    sm_names = [[src_names[k] for k in sm_index_i] for sm_index_i in sm_index]  # [source][src]

    for path_metric_i, sm_name_i in zip(path_metrics, sm_names):  # [model][para_name]
        for path_metric_j in path_metric_i:  # [para_name]
            metric_mean_src_to_file(path_metric_j, sm_name_i, ['mse', 'sr', 'si_snr'], path_metric_j)
            metric_mean_mix_to_file(path_metric_j, set_names,
                                    ['sdr', 'isr', 'sir', 'sar'], path_metric_j)
            metric_mean_mix_to_file(path_metric_j, set_names,
                                    ['si_sdr', 'si_sir', 'si_sar'], path_metric_j)
            metric_mean_mix_to_file(path_metric_j, set_names,
                                    ['sd_sdr', 'sd_sir', 'sd_sar'], path_metric_j)

            for file_name in ['mse', 'sr', 'si_snr']:
                metric_mean_to_csv(os.path.join(path_metric_j, f'{file_name}_mean.hdf5'), level=2)
            for file_name in ['sdr', 'isr', 'sir', 'sar']:
                metric_mean_to_csv(os.path.join(path_metric_j, f'{file_name}_mean.hdf5'), level=2)
            for file_name in ['si_sdr', 'si_sir', 'si_sar']:
                metric_mean_to_csv(os.path.join(path_metric_j, f'{file_name}_mean.hdf5'), level=2)
            for file_name in ['sd_sdr', 'sd_sir', 'sd_sar']:
                metric_mean_to_csv(os.path.join(path_metric_j, f'{file_name}_mean.hdf5'), level=2)


if __name__ == '__main__':
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    # ========================================================================================
    # for shipsear data
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    SCALER_DATA = 'max_one'
    # SCALER_DATA = 'or'
    SUB_SET_WAY = 'rand'
    # SUB_SET_WAY = 'order'
    # SPLIT_WAY = None
    SPLIT_WAY = 'split'
    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav',
                                   scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY, split_way=SPLIT_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source
    # -----------------------------------------------------------------------------------------
    SET_NAMES = ['train', 'val', 'test']
    SRC_NAMES = [f'Z_{i}_ns' for i in range(4)]
    S_NAMES = [[f'{src_name}_{set_name_j}' for set_name_j in SET_NAMES] for src_name in SRC_NAMES]
    S_DATA = [[read_data(PATH_DATA, name_j) for name_j in name_i] for name_i in S_NAMES]
    SM_NAMES = [f'Z_{i}_ns' for i in range(4, 8)]
    # =========================================================================================
    PATH_SAVE_ROOT = '../result_separation_multiple_autoencoder'

    object_decoded_files_ = ListDecodedFiles(PATH_SAVE_ROOT)
    SM_INDEX = [(1, 2), (1, 3), (2, 3), (1, 2, 3)]
    compute_metrics(object_decoded_files_, S_NAMES, SRC_NAMES, S_DATA, SM_INDEX, SET_NAMES)

    metrics_to_files(object_decoded_files_.path_metrics, SRC_NAMES, SM_INDEX, SET_NAMES)

    logging.info('finished')
