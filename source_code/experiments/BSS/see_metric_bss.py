# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 18:14:00 2025

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""

import logging
import os

import numpy as np
import h5py

from file_operation import list_dirs, list_dirs_start_str, mkdir
from metric import mse_np, samerate_acc_np, si_snr_np
from prepare_data import list_transpose
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data
from see_metric_autoencoder import compute_metric, compute_sdr, compute_si_sdr, display_and_save_metric
from see_metric_single_source_ae_ns import data_reshape_same


def compute_metrics(path_results, s_names, src_names, s_data, sm_names, sm_index, set_names):
    """compute metrics.
    Args:
        path_results (list[str]): list paths where result saved.
        s_names (list[list[str]]): list of datasets.
        src_names (list[str]): list of name of sources including mixed sources, e.g. ['Z_0_ns',...,'Z_7_ns'].
        s_data (list[list[h5pyFile]]): file object of datas.
        sm_names (list[str]): names of mixed sources, e.g. ['Z_4_ns', 'Z_5_ns', 'Z_6_ns', 'Z_7_ns'].
        sm_index (list[tuple(int)]): clean target source of mixed sources, e.g. [(1, 2),...,(1, 2, 3)].
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
    """

    for path_result_j in path_results:
        path_metric_j = os.path.join(path_result_j, 'metric')
        for sm_name_i, sm_index_i in zip(sm_names, sm_index):
            src_name_set_i = [s_names[k] for k in sm_index_i]  # [src][set]
            # set_src_name_i = list_transpose(src_name_set_i, warn=True)  # [set][src]
            src_name_i = [src_names[k] for k in sm_index_i]  # [src]
            s_data_i = [s_data[k] for k in sm_index_i]  # [src][set]
            data_s_i = list_transpose(s_data_i, warn=True)  # [set][src]

            path_metric_j_i = os.path.join(path_metric_j, sm_name_i)
            mkdir(path_metric_j_i)

            data_sp_k = []  # [set]
            for set_name_n in set_names:  # level n: [set]
                data_sp_k.append(read_data(path_result_j, f'{sm_name_i}_{set_name_n}_predict.hdf5'))

            for l_src, (src_name_l, src_name_set_l, data_s_l) in enumerate(zip(  # level l:[src]
                    src_name_i, src_name_set_i, s_data_i)):
                data_sp_l = [data_sp_k_i[:, :, l_src] for data_sp_k_i in data_sp_k]

                path_metric_para_l = os.path.join(path_metric_j_i, src_name_l)
                mkdir(path_metric_para_l)

                mse_list, mse_mean = compute_metric(data_s_l, data_sp_l, mse_np)
                display_and_save_metric(mse_list, mse_mean, path_metric_para_l, set_names,
                                        # hist_bins=(0, 1e-1, 10),
                                        save_name='mse', file_name='mse')

                sr_list, sr_mean = compute_metric(data_s_l, data_sp_l, samerate_acc_np)
                display_and_save_metric(sr_list, sr_mean, path_metric_para_l, set_names,
                                        # hist_bins=(0.6, 1.0, 40),
                                        save_name='sr', file_name='sr')

                si_snr_list, si_snr_mean = compute_metric(data_s_l, data_sp_l, si_snr_np)
                display_and_save_metric(si_snr_list, si_snr_mean, path_metric_para_l, set_names,
                                        # hist_bins=(-40., 40., 40),
                                        save_name='si_snr', file_name='si_snr')

            # data_sp_k = list_transpose(data_sp_k)  # [set][src]

            for data_s_n, data_sp_n, set_name_n in zip(
                    data_s_i, data_sp_k, set_names):
                data_s_n = np.asarray(data_s_n).transpose((1, 0, 3, 2))
                # (n_src, n_sams, 1, fl)->(n_sams, n_src, fl, 1)
                data_sp_n = np.asarray(data_sp_n)
                # if data_sp_n.ndim == 3:
                #     data_sp_n = np.expand_dims(data_sp_n, -2)  # (n_src, n_sams, 1, fl)
                # data_sp_n = data_sp_n.transpose((1, 0, 3, 2))  # ->(n_sams, n_src, fl, 1)
                if data_sp_n.ndim == 3:  # (n_sams, fl, n_src)
                    data_sp_n = np.expand_dims(data_sp_n, -1)  # (n_sams, fl, n_src, 1)
                data_sp_n = data_sp_n.transpose((0, 2, 1, 3))  # ->(n_sams, n_src, fl, 1)

                if tuple(data_s_n.shape) != tuple(data_sp_n.shape):
                    data_s_n, data_sp_n = data_reshape_same(data_s_n, data_sp_n)

                compute_sdr(data_s_n, data_sp_n, path_metric_j_i,
                            [f'sdr_{set_name_n}', f'isr_{set_name_n}', f'sir_{set_name_n}', f'sar_{set_name_n}'],
                            src_name_i, hist_bins=[(0, 30, 30), (0, 30, 30), (0, 30, 30), (0, 30, 30)],
                            save_name=[f'sdr_{set_name_n}', f'isr_{set_name_n}',
                                       f'sir_{set_name_n}', f'sar_{set_name_n}'])

                data_s_n = data_s_n.transpose((0, 2, 3, 1))
                # (n_sams, n_src, fl, 1) -> (n_sams, fl, n_channel, n_src)
                data_sp_n = data_sp_n.transpose((0, 2, 3, 1))
                # (n_sams, n_src, fl, 1) -> (n_sams, fl, n_channel, n_src)

                compute_si_sdr(data_s_n, data_sp_n, True, path_metric_j_i,
                               [f'si_sdr_{set_name_n}', f'si_sir_{set_name_n}', f'si_sar_{set_name_n}'],
                               src_name_i,
                               # hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                               save_name=[f'si_sdr_{set_name_n}', f'si_sir_{set_name_n}', f'si_sar_{set_name_n}'],
                               compute_permutation=True)

                compute_si_sdr(data_s_n, data_sp_n, False, path_metric_j_i,
                               [f'sd_sdr_{set_name_n}', f'sd_sir_{set_name_n}', f'sd_sar_{set_name_n}'],
                               src_name_i,
                               # hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                               save_name=[f'sd_sdr_{set_name_n}', f'sd_sir_{set_name_n}', f'sd_sar_{set_name_n}'],
                               compute_permutation=True)


def metric_mean_src_to_file(path_metric, metric_keys):
    """Merge metric_mean files to .hdf5 file.
    Args:
        path_metric (str): path where to save files.
        metric_keys (list[str]): name of the metrics.
    """
    for metric_key in metric_keys:
        f_h5 = h5py.File(os.path.join(path_metric, f'{metric_key}_mean.hdf5'), 'a')
        dir_names = list_dirs(path_metric, False)
        # dir_names = [name for name in dir_names if os.path.isdir(os.path.join(path_metric, name))]
        for dir_name in dir_names:
            path_metric_dir = os.path.join(path_metric, dir_name)
            grp_dir = f_h5.create_group(dir_name)
            src_names = list_dirs(path_metric_dir, False)
            for src_name in src_names:
                path_metric_src = os.path.join(path_metric_dir, src_name)
                grp_dir[src_name] = np.asarray(h5py.File(os.path.join(
                    path_metric_src, f'{metric_key}.hdf5'), 'r')[f'{metric_key}_mean'], dtype=np.float32)
        f_h5.close()


def metric_mean_mix_to_file(path_metric, set_names, metric_keys):
    """Merge metric_mean files to .hdf5 file.
    Args:
        path_metric (str): path where to save files.
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
        metric_keys (list[str]): name of the metrics.
    """
    for metric_key in metric_keys:
        f_h5 = h5py.File(os.path.join(path_metric, f'{metric_key}_mean.hdf5'), 'a')
        dir_names = list_dirs(path_metric, False)
        # dir_names = [name for name in dir_names if os.path.isdir(os.path.join(path_metric, name))]
        for dir_name in dir_names:
            grp_dir = f_h5.create_group(dir_name)
            path_metric_dir = os.path.join(path_metric, dir_name)
            for set_name in set_names:
                grp_dir[set_name] = np.asarray(h5py.File(os.path.join(path_metric_dir,
                                                                      f'{metric_key}_{set_name}.hdf5'),
                                                         'r')[f'{metric_key}_{set_name}_mean'], dtype=np.float32)
        f_h5.close()


def metrics_to_files(path_metrics, set_names):  # [source][model][para_name]
    """Merged metrics files to .hdf5 file.
    Args:
        path_metrics (list[str]): [model], path of metric dirs.
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
    """

    for path_metric_j in path_metrics:  # [model][para_name]
        path_metric_j = os.path.join(path_metric_j, 'metric')
        metric_mean_src_to_file(path_metric_j, ['mse', 'sr', 'si_snr'])
        metric_mean_mix_to_file(path_metric_j, set_names,
                                ['sdr', 'isr', 'sir', 'sar'])
        metric_mean_mix_to_file(path_metric_j, set_names,
                                ['si_sdr', 'si_sir', 'si_sar'])
        metric_mean_mix_to_file(path_metric_j, set_names,
                                ['sd_sdr', 'sd_sir', 'sd_sar'])


if __name__ == '__main__':
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    # for shipsear data
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'
    SCALER_DATA = 'max_one'
    # SCALER_DATA = 'or'
    SUB_SET_WAY = 'rand'
    # SUB_SET_WAY = 'order'
    # SPLIT_WAY = None
    SPLIT_WAY = 'split'
    # SPLIT_WAY = 'select'
    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav',
                                   scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY, split_way=SPLIT_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source

    SET_NAMES = ['train', 'val', 'test']
    SRC_NAMES = [f'Z_{i}_ns' for i in range(4)]
    S_NAMES = [[f'{src_name}_{set_name_j}' for set_name_j in SET_NAMES] for src_name in SRC_NAMES]
    S_DATA = [[read_data(PATH_DATA, name_j) for name_j in name_i] for name_i in S_NAMES]
    SM_NAMES = [f'Z_{i}_ns' for i in range(4, 8)]

    PATH_RESULT_ROOT = '../../../result_separation_bss'

    path_results_ = list_dirs_start_str(PATH_RESULT_ROOT, 'model_')
    SM_INDEX = [(1, 2), (1, 3), (2, 3), (1, 2, 3)]
    compute_metrics(path_results_, S_NAMES, SRC_NAMES, S_DATA, SM_NAMES, SM_INDEX, SET_NAMES)
    metrics_to_files(path_results_, SET_NAMES)

    logging.info('finished')
