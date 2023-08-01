# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 20:33:00 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, too-few-public-methods, too-many-arguments
# pylint: disable=too-many-instance-attributes, too-many-locals, too-many-statements, useless-object-inheritance

import logging
import os
import numpy as np

from file_operation import list_dirs, list_dirs_start_str, list_files_end_str, mkdir
from metric import mse_np, samerate_acc_np, si_snr_np
from prepare_data import list_transpose
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data
from see_metric_autoencoder import save_metric, compute_metric, display_metric, compute_sdr
from see_metric_autoencoder import compute_si_sdr, metric_mean_src_to_file, metric_mean_mix_to_file, metric_mean_to_csv
from see_metric_autoencoder import recover_pad_num_samples, recover_pad_num_samples_list


class ListDecodedFiles(object):
    """class for list predict decoded files."""

    def __init__(self, path_save_root, src_names=None):
        """__init__
        Args:
            path_save_root (str): Root path where save result.
            src_names (list[str], optional): all name of sources including mixed. Defaults to None.
        """
        name_models = list_dirs_start_str(path_save_root, 'model_', False)
        self.name_models = name_models
        path_models = [os.path.join(path_save_root, name_model) for name_model in name_models]
        self.path_models = path_models

        path_model_srcs = []  # [model][src]
        for path_model in path_models:
            path_model_srcs.append([os.path.join(path_model, 'auto_model', src_name) for src_name in src_names])

        path_model_src_paras = []  # [model][src][para_name], every src must has same num of para_name
        name_model_src_paras = []  # [model][src][para_name]
        path_weight_files = []  # [model][src][para_name][weight_file]
        name_weight_files = []  # [model][src][para_name][weight_file]
        for path_model_src_i in path_model_srcs:
            path_model_src_paras_i = []
            name_model_src_paras_i = []
            path_weight_files_i = []
            name_weight_files_i = []

            for path_model_src_j in path_model_src_i:
                path_model_src_paras_j = list_dirs(path_model_src_j)
                path_model_src_paras_i.append(path_model_src_paras_j)
                name_model_src_paras_i.append(list_dirs(path_model_src_j, False))

                path_weight_files_j = []
                name_weight_files_j = []
                for path_model_src_paras_k in path_model_src_paras_j:
                    path_weight_files_j.append(list_files_end_str(path_model_src_paras_k, 'hdf5'))
                    name_weight_files_j.append(list_files_end_str(path_model_src_paras_k, 'hdf5', False))
                path_weight_files_i.append(path_weight_files_j)
                name_weight_files_i.append(name_weight_files_j)

            path_model_src_paras.append(path_model_src_paras_i)
            name_model_src_paras.append(name_model_src_paras_i)
            path_weight_files.append(path_weight_files_i)
            name_weight_files.append(name_weight_files_i)

        self.path_model_src_paras = path_model_src_paras
        self.name_model_src_paras = name_model_src_paras
        self.path_weight_files = path_weight_files
        self.name_weight_files = name_weight_files

        path_out_models = []  # [model][src][para_name]
        path_metrics = []  # [model]
        for path_model_i, name_model_src_paras_i in zip(path_models, name_model_src_paras):
            path_metric_i = os.path.join(path_model_i, 'metric')
            mkdir(path_metric_i)
            path_metrics.append(path_metric_i)
            path_out_model_i = []
            for name_model_src_paras_j in name_model_src_paras_i:
                path_out_model_j = []
                for name_model_src_paras_k in name_model_src_paras_j:
                    path_out_model_j.append(os.path.join(path_model_i, 'auto_out', name_model_src_paras_k))
                path_out_model_i.append(path_out_model_j)
            path_out_models.append(path_out_model_i)

        self.path_out_models = path_out_models
        self.path_metrics = path_metrics

        path_metrics_paras = []  # [model][para_name]
        for path_metric_i, name_model_src_paras_i in zip(path_metrics, name_model_src_paras):
            path_metrics_para_i = []
            for name_model_para_j in name_model_src_paras_i[0]:
                path_metric_para_j = os.path.join(path_metric_i, name_model_para_j)
                mkdir(path_metric_para_j)
                path_metrics_para_i.append(path_metric_para_j)
            path_metrics_paras.append(path_metrics_para_i)

        self.path_metrics_paras = path_metrics_paras

    def filter_model(self, model_name, item):
        """Filter item belong to specific model.
        Args:
            model_name (str): name of the model.
            item (str): keword name of the item.
        Returns:
            items_filter: item belong to specific model.
        """
        num_index_model = self.name_models.index(model_name)
        if item == 'name_weight_files':
            items_filter = self.name_weight_files[num_index_model]
        elif item == 'path_weight_files':
            items_filter = self.path_weight_files[num_index_model]
        elif item == 'name_model_src_paras':
            items_filter = self.name_model_src_paras[num_index_model]
        return items_filter


def data_reshape_same(data_1, data_2, mode='transpose'):
    """Reshape data_2 to shape of data_1.
        data_1.shape == (nsams, channel=1, fl)
        Be carefull, most time should use transpose rather than reshape.
    Args:
        data_1 (list[np.ndarray] or np.ndarray): data_1 with target data shape.
        data_2 (list[np.ndarray] or np.ndarray): data_2 to reshape.
        mode (str, optional): type of reshape method. Defaults to 'transpose'.
    Returns:
        data_1 (list[np.ndarray] or np.ndarray): data_1 with target data shape.
        data_2 (list[np.ndarray] or np.ndarray): data_2 after reshape.
    """
    def func_reshape(data_1, data_2, mode):
        if mode == 'transpose':
            if data_2.ndim == 3:
                data_2 = data_2.transpose(0, 2, 1)
            elif data_2.ndim == 4:
                data_2 = data_2.transpose(0, 1, 3, 2)
        elif mode == 'reshape':
            data_2 = data_2.reshape(data_1.shape)
        return data_2

    if isinstance(data_1, list):
        for i, (data_1_i, data_2_i) in enumerate(zip(data_1, data_2)):
            data_1_i, data_2_i = np.asarray(data_1_i), np.asarray(data_2_i)
            if data_1_i.shape != data_2_i.shape:
                data_2[i] = func_reshape(data_1_i, data_2_i, mode)
    else:
        data_1, data_2 = np.asarray(data_1), np.asarray(data_2)
        if data_1.shape != data_2.shape:
            data_2 = func_reshape(data_1, data_2, mode)
    return data_1, data_2


def compute_metrics(object_decoded_files, s_names, src_names, s_data, set_names):
    """compute metrics.
    Args:
        object_decoded_files (class ListDecodedFiles): object for list predict decoded files
        s_names (list[list[str]]): list of datasets.
        src_names (list[str]): list of name of sources including mixed sources, e.g. ['Z_0_ns',...,'Z_7_ns'].
        s_data (list[list[h5pyFile]]): file object of datas.
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
    """
    name_weight_files = object_decoded_files.name_weight_files  # [model][src][para_name][weight_file]
    path_metrics_paras = object_decoded_files.path_metrics_paras  # [model][para_name]
    path_out_models = object_decoded_files.path_out_models  # [model][src][para_name]

    for name_weight_files_i, path_metrics_para_i, path_out_model_i in zip(  # level i: [model]
            name_weight_files, path_metrics_paras, path_out_models):
        for name_weight_files_j, src_name_set_j, src_name_j, data_s_j, path_out_model_j in zip(  # level j: [src]
                name_weight_files_i, s_names, src_names, s_data, path_out_model_i):
            for name_weight_files_k, path_metric_para_k, path_out_model_k in zip(  # level k: [para_name]
                    name_weight_files_j, path_metrics_para_i, path_out_model_j):
                path_metric_para_j = os.path.join(path_metric_para_k, src_name_j)
                mkdir(path_metric_para_j)

                name_weight_file_l = name_weight_files_k[-1]  # level l: [weight_file]
                data_sp_j = []
                for src_name_set_k in src_name_set_j:
                    path_sp_k = os.path.join(path_out_model_k, f'{src_name_set_k}_{name_weight_file_l}_decoded')
                    data_sp_j.append(read_data(os.path.dirname(path_sp_k), os.path.basename(path_sp_k)))

                data_sp_j = recover_pad_num_samples_list(data_s_j, data_sp_j)

                mse_list, mse_mean = compute_metric(data_s_j, data_sp_j, mse_np)
                display_metric(mse_list, path_metric_para_j, set_names,
                               # hist_bins=(0, 1e-1, 10),
                               save_name='mse')
                save_metric(path_metric_para_j, 'mse', ['mse_mean'], [np.asarray(mse_mean)],
                            {'mse': dict(zip(set_names, mse_list))})

                sr_list, sr_mean = compute_metric(data_s_j, data_sp_j, samerate_acc_np)
                display_metric(sr_list, path_metric_para_j, set_names,
                               # hist_bins=(0.6, 1.0, 40),
                               save_name='sr')
                save_metric(path_metric_para_j, 'sr', ['sr_mean'], [np.asarray(sr_mean)],
                            {'sr': dict(zip(set_names, sr_list))})

                si_snr_list, si_snr_mean = compute_metric(data_s_j, data_sp_j, si_snr_np)
                display_metric(si_snr_list, path_metric_para_j, set_names,
                               # hist_bins=(0.6, 1.0, 40),
                               save_name='si_snr')
                save_metric(path_metric_para_j, 'si_snr', ['si_snr_mean'], [np.asarray(si_snr_mean)],
                            {'si_snr': dict(zip(set_names, si_snr_list))})

        sm_index = [(1, 2), (1, 3), (2, 3), (1, 2, 3)]
        name_weight_files_i = list_transpose(name_weight_files_i)  # [para_name][src][weight_file]
        path_out_model_i = list_transpose(path_out_model_i)  # [para_name][src]
        for name_weight_files_k, path_metric_para_k, path_out_model_k in zip(  # level k: [para_name]
                name_weight_files_i, path_metrics_para_i, path_out_model_i):
            for sm_index_j, sm_name_j in zip(sm_index, SM_NAMES):  # level j: [src_mix]
                src_name_j = [src_names[m] for m in sm_index_j]  # level m: [src]
                data_s_j = list_transpose([s_data[m] for m in sm_index_j], warn=True)  # [set][src], level n: [set]
                src_name_set_j = [s_names[m] for m in sm_index_j]  # [src][set]
                set_src_name_j = list_transpose(src_name_set_j, warn=True)  # [set][src]
                name_weight_files_j = [name_weight_files_k[m] for m in sm_index_j]  # [src][weight_file]

                path_out_model_j = [path_out_model_k[m] for m in sm_index_j]
                data_sp_j = []  # [set][src]
                for set_src_name_n in set_src_name_j:
                    name_sp_n = []
                    for set_src_name_m, name_weight_file_m in zip(set_src_name_n, name_weight_files_j):
                        name_weight_file_l = name_weight_file_m[-1]  # level l: [weight_file]
                        name_sp_n.append(f'{set_src_name_m}_{name_weight_file_l}_decoded')
                    data_sp_n = []
                    for path_out_model_m, name_sp_m in zip(path_out_model_j, name_sp_n):
                        data_sp_n.append(read_data(path_out_model_m, name_sp_m))
                    data_sp_j.append(data_sp_n)

                path_metric_para_j = os.path.join(path_metric_para_k, sm_name_j)
                mkdir(path_metric_para_j)

                for data_s_n, data_sp_n, set_name_n in zip(data_s_j, data_sp_j, set_names):
                    data_s_n = np.asarray(data_s_n).transpose((1, 0, 3, 2))
                    # (n_src, n_sams, 1, fl) -> (n_sams, n_src, fl, 1)
                    data_sp_n = np.asarray(data_sp_n)
                    if data_sp_n.ndim == 3:
                        data_sp_n = np.expand_dims(data_sp_n, -2)  # (n_src, n_sams, 1, fl)
                    data_sp_n = data_sp_n.transpose((1, 0, 3, 2))  # ->(n_sams, n_src, fl, channel=1)

                    data_sp_n = recover_pad_num_samples(data_s_n, data_sp_n)

                    if tuple(data_s_n.shape) != tuple(data_sp_n.shape):
                        data_s_n, data_sp_n = data_reshape_same(data_s_n, data_sp_n)
                    compute_sdr(data_s_n, data_sp_n, path_metric_para_j,
                                [f'sdr_{set_name_n}', f'isr_{set_name_n}', f'sir_{set_name_n}', f'sar_{set_name_n}'],
                                src_name_j, hist_bins=[(0, 30, 30), (0, 30, 30), (0, 30, 30), (0, 30, 30)],
                                save_name=[f'sdr_{set_name_n}', f'isr_{set_name_n}',
                                           f'sir_{set_name_n}', f'sar_{set_name_n}'])

                    data_s_n = data_s_n.transpose((0, 2, 3, 1))
                    # (n_sams, n_src, fl, 1) -> (n_sams, fl, n_channel, n_src)
                    data_sp_n = data_sp_n.transpose((0, 2, 3, 1))
                    # (n_sams, n_src, fl, 1) -> (n_sams, fl, n_channel, n_src)

                    compute_si_sdr(data_s_n, data_sp_n, True, path_metric_para_j,
                                   [f'si_sdr_{set_name_n}', f'si_sir_{set_name_n}', f'si_sar_{set_name_n}'],
                                   src_name_j,
                                   # hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                                   save_name=[f'si_sdr_{set_name_n}', f'si_sir_{set_name_n}', f'si_sar_{set_name_n}'])

                    compute_si_sdr(data_s_n, data_sp_n, False, path_metric_para_j,
                                   [f'sd_sdr_{set_name_n}', f'sd_sir_{set_name_n}', f'sd_sar_{set_name_n}'],
                                   src_name_j,
                                   # hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                                   save_name=[f'sd_sdr_{set_name_n}', f'sd_sir_{set_name_n}', f'sd_sar_{set_name_n}'])


def metrics_to_files(path_models, src_names, sm_names, set_names):
    """Merged metrics files to .hdf5 file.
    Args:
        path_models (list[list[list[str]]]): [model][src][para_name], path of models.
        src_names (list[str]): list of name of sources including mixed sources, e.g. ['Z_0_ns',...,'Z_7_ns'].
        sm_names (list[str]): list of name of mixed sources, e.g. ['Z_4_ns',...,'Z_7_ns'].
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
    """
    for path_models_i in path_models:
        path_metric_i = os.path.join(path_models_i, 'metric')

        metric_mean_src_to_file(path_metric_i, src_names, ['mse', 'sr'], path_metric_i)
        metric_mean_mix_to_file(path_metric_i, sm_names, set_names,
                                ['sdr', 'isr', 'sir', 'sar'], path_metric_i)
        metric_mean_mix_to_file(path_metric_i, sm_names, set_names,
                                ['si_sdr', 'si_sir', 'si_sar'], path_metric_i)
        metric_mean_mix_to_file(path_metric_i, sm_names, set_names,
                                ['sd_sdr', 'sd_sir', 'sd_sar'], path_metric_i)

        for file_name in ['mse', 'sr']:
            metric_mean_to_csv(os.path.join(path_metric_i, f'{file_name}_mean.hdf5'), level=2)
        for file_name in ['sdr', 'isr', 'sir', 'sar']:
            metric_mean_to_csv(os.path.join(path_metric_i, f'{file_name}_mean.hdf5'), level=3)
        for file_name in ['si_sdr', 'si_sir', 'si_sar']:
            metric_mean_to_csv(os.path.join(path_metric_i, f'{file_name}_mean.hdf5'), level=3)
        for file_name in ['sd_sdr', 'sd_sir', 'sd_sar']:
            metric_mean_to_csv(os.path.join(path_metric_i, f'{file_name}_mean.hdf5'), level=3)


if __name__ == '__main__':
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO)
    # ========================================================================================
    # for shipsear data
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    SCALER_DATA = 'max_one'
    SUB_SET_WAY = 'rand'

    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav', scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source
    # -----------------------------------------------------------------------------------------
    SET_NAMES = ['train', 'val', 'test']
    SRC_NAMES = [f'Z_{i}_ns' for i in range(4)]
    S_NAMES = [[f'{src_name}_{set_name_j}' for set_name_j in SET_NAMES] for src_name in SRC_NAMES]
    S_DATA = [[read_data(PATH_DATA, name_j) for name_j in name_i] for name_i in S_NAMES]
    SM_NAMES = [f'Z_{i}_ns' for i in range(4, 8)]
    # =========================================================================================
    PATH_SAVE_ROOT = '../result_separation_ae_ns_single'

    object_decoded_files_ = ListDecodedFiles(PATH_SAVE_ROOT, SRC_NAMES)

    compute_metrics(object_decoded_files_, S_NAMES, SRC_NAMES, S_DATA, SET_NAMES)

    metrics_to_files(object_decoded_files_.path_models, SRC_NAMES, SM_NAMES, SET_NAMES)

    logging.info('finished')
