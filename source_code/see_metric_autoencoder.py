 # -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:43:44 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation, too-many-arguments, too-many-locals
import logging
import os

import h5py
import museval
import numpy as np

from file_operation import list_files, list_dirs
from metric import mse_np, samerate_acc_np, si_snr_np
from see_metric import DisplayMetric


def save_metric(path_save, file_name, name_list=None, data_list=None, data_dict=None, dtype='float32'):
    """Save mean and per set metric to .hdf5 file.
    Args:
        path_save (str): path where to save file.
        file_name (str): name of the file to save.
        name_list (list[str], optional): list of names of data to save. Defaults to None.
        data_list (list[np.ndarray], optional): list of data to save. Defaults to None.
        data_dict (dict{str:dict{str:np.ndarray}}, optional): dict of data to save, 2 level dict,
            first level create h5py grp, second level create dataset. Defaults to None.
        dtype (np.dtype, optional): data type to save. Defaults to 'float32'.
    """
    full_file_name = os.path.join(path_save, f'{file_name}.hdf5')
    with h5py.File(full_file_name, 'a') as f_a:
        if name_list is not None:
            for name_i, data_i in zip(name_list, data_list):
                f_a.create_dataset(name_i, data=data_i, dtype=dtype)
        if data_dict is not None:
            for name_i, dict_i in data_dict.items():
                grp_i = f_a.create_group(name_i)
                for name_j, data_j in dict_i.items():
                    grp_i.create_dataset(name_j, data=data_j, dtype=dtype)


def compute_metric(s_list, sp_list, func_metric):
    """Compute metric and metric mean.
    Args:
        s_list (list[np.ndarray]): list of target data.
        sp_list (list[np.ndarray]): list of predict data.
        func_metric (function): function of compute metric.
    Returns:
        metric_list (list[np.ndarray]): list of metric.
        metric_mean (list[np.ndarray]): list of mean metric.
    """
    dm_object = DisplayMetric(s_list, sp_list, None)
    metric_list = dm_object.compute_metric(func_metric, mode=2)
    metric_mean = dm_object.compute_mean_metrics(metric_list)
    return metric_list, metric_mean


def compute_metric_mean(metric_list):
    """Compute mean of metric.
    Args:
        metric_list (list[np.ndarray]): list of metric.
    Returns:
        metric_mean (list[np.ndarray]): list of mean metric.
    """
    dm_object = DisplayMetric(None, None, None)
    metric_mean = dm_object.compute_mean_metrics(metric_list)
    return metric_mean


def display_metric(metric_list, path_save, set_name, hist_bins=None, save_name=None):
    """Save display metric pictures and return metrics mean.
    Args:
        metric_list (list[np.ndarray]): list of metric.
        path_save (str): path where to save files.
        set_name (list[str]): name of the sources, e.g. ['s_1', 's_2', 's_3'].
        hist_bins (tuple(float, float, int), optional): paras for hist. Defaults to None.
        save_name (str, optional): for name to save. Defaults to None.
    """
    if hist_bins is None:
        value_min = min([min(metric_i) for metric_i in metric_list])
        value_max = max([max(metric_i) for metric_i in metric_list])
        hist_bins = (value_min, value_max, 10)
    dm_object = DisplayMetric(None, None, path_save)
    dm_object.display_static_metrics(metric_list, set_name, ['violin', 'hist'],
                              hist_bins=hist_bins, save_name=save_name)


def compute_sdr(s_list, sp_list, path_metric, file_name, set_name, hist_bins=None, save_name=None):
    """Compute and save sdr related metrics.
    Args:
        s_list (list[np.ndarray,shape==(n_sams,n_src,fl,n_channel=1)]): list of target data.
        sp_list (list[np.ndarray,shape==(n_sams,n_src,fl,n_channel=1)]): list of predict data.
        path_metric (str): path where to save files.
        file_name (list[str]): list of metric name, e.g. ['sdr', 'isr', 'sir', 'sar'].
        set_name (list[str]): name of the sources, e.g. ['s_1', 's_2', 's_3'].
        hist_bins (tuple(float, float, int), optional): paras for hist. Defaults to None.
        save_name (str, optional): for name to save. Defaults to None.
    Returns:
        metric_list (list[np.ndarray]): [metric](n_src, n_sams), list of metric.
        metric_mean (list[np.ndarray]): [metric], list of mean metric.
    """
    if hist_bins is None:
        hist_bins = [None for i in range(len(file_name))]

    s_arr = np.asarray(s_list)  # (n_sams, n_src, fl, 1) (nsrc, nsampl, nchan)
    sp_arr = np.asarray(sp_list)

    n_sams = s_arr.shape[0]
    n_src = s_arr.shape[1]

    sdr_arr = np.empty((n_sams, n_src), dtype=np.float32)
    isr_arr = np.empty((n_sams, n_src), dtype=np.float32)
    sir_arr = np.empty((n_sams, n_src), dtype=np.float32)
    sar_arr = np.empty((n_sams, n_src), dtype=np.float32)
    for j, (s_j, sp_j) in enumerate(zip(s_arr, sp_arr)):
        sdr_arr[j], isr_arr[j], sir_arr[j], sar_arr[j] = np.squeeze(museval.evaluate(
            s_j, sp_j, win=np.inf, hop=np.inf, padding=False))
    sdr_arr = sdr_arr.transpose()  # (n_src, n_sams)
    isr_arr = isr_arr.transpose()
    sir_arr = sir_arr.transpose()
    sar_arr = sar_arr.transpose()

    metric_list = [sdr_arr, isr_arr, sir_arr, sar_arr]
    metric_mean_list = []
    for file_name_i, hist_bins_i, save_name_i, metric_arr_i in zip(file_name, hist_bins, save_name, metric_list):
        display_metric(metric_arr_i, path_metric, set_name, hist_bins_i, save_name_i)
        metric_mean_i = compute_metric_mean(metric_arr_i)
        save_metric(path_metric, file_name_i, [f'{file_name_i}_mean'], [np.asarray(metric_mean_i)],
                    {file_name_i: dict(zip(set_name, metric_arr_i))})
        metric_mean_list.append(metric_mean_i)
    return metric_list, metric_mean_list


def compute_si_sdr(s_list, sp_list, scaling, path_metric, file_name, set_name, hist_bins=None, save_name=None):
    """Compute and save si_sdr sd_sdr related metrics.
    Args:
        s_list (list[np.ndarray,shape==(n_src,n_sams,n_channel=1,fl)]): list of target data.
        sp_list (list[np.ndarray,shape==(n_src,n_sams,n_channel=1,fl)]): list of predict data.
        scaling (bool): si_sdr True or sd_sdr False.
        path_metric (str): path where to save files.
        file_name (list[str]): list of metric name, e.g. ['sdr', 'isr', 'sir', 'sar'].
        set_name (list[str]): name of the sources, e.g. ['s_1', 's_2', 's_3'].
        hist_bins (tuple(float, float, int), optional): paras for hist. Defaults to None.
        save_name (str, optional): for name to save. Defaults to None.
    Returns:
        metric_list (list[np.ndarray]): [metric](n_src, n_sams), list of metric.
        metric_mean (list[np.ndarray]): [metric], list of mean metric.
    """
    from metrics.si_sdr import ScaleInvariantSDR  # pylint: disable=import-outside-toplevel

    if hist_bins is None:
        hist_bins = [None for i in range(len(file_name))]

    s_arr = np.asarray(s_list)  # (n_sams, fl, n_channel=1, n_src)
    sp_arr = np.asarray(sp_list)

    n_sams = s_arr.shape[0]
    n_src = s_arr.shape[-1]

    sdr_arr = np.empty((n_sams, n_src), dtype=np.float32)
    sir_arr = np.empty((n_sams, n_src), dtype=np.float32)
    sar_arr = np.empty((n_sams, n_src), dtype=np.float32)

    for j, (s_j, sp_j) in enumerate(zip(s_arr, sp_arr)):
        si_sdr_class = ScaleInvariantSDR(s_j, sp_j, scaling=scaling)
        score_dict = si_sdr_class.evaluate()
        scores = [value for key, value in score_dict.items()][1:]
        for k, score in enumerate(scores):
            sdr_arr[j][k] = score['SDR'][0]
            sir_arr[j][k] = score['SIR'][0]
            sar_arr[j][k] = score['SAR'][0]

    sdr_arr = sdr_arr.transpose()  # (n_src, n_sams)
    sir_arr = sir_arr.transpose()
    sar_arr = sar_arr.transpose()

    metric_list = [sdr_arr, sir_arr, sar_arr]
    metric_mean_list = []
    for file_name_i, hist_bins_i, save_name_i, metric_arr_i in zip(file_name, hist_bins, save_name, metric_list):
        display_metric(metric_arr_i, path_metric, set_name, hist_bins_i, save_name_i)
        metric_mean_i = compute_metric_mean(metric_arr_i)
        save_metric(path_metric, file_name_i, [f'{file_name_i}_mean'], [np.asarray(metric_mean_i)],
                    {file_name_i: dict(zip(set_name, metric_arr_i))})
        metric_mean_list.append(metric_mean_i)
    return metric_list, metric_mean_list


def metric_mean_src_to_file(path_metric, src_names, metric_keys, path_save):
    """Merge metric_mean files to .hdf5 file.
    Args:
        path_metric (str): path where to save files.
        src_names (list[str]): name of the sources.
        metric_keys (list[str]): name of the metrics.
        path_save (str): path to save merged file.
    """
    for metric_key in metric_keys:
        f_h5 = h5py.File(os.path.join(path_save, f'{metric_key}_mean.hdf5'), 'a')
        dir_names = list_dirs(path_metric, False)
        # dir_names = [name for name in dir_names if os.path.isdir(os.path.join(path_metric, name))]
        for dir_name in dir_names:
            path_metric_dir = os.path.join(path_metric, dir_name)
            grp_dir = f_h5.create_group(dir_name)
            for src_name in src_names:
                path_metric_src = os.path.join(path_metric_dir, src_name)
                grp_dir[src_name] = np.asarray(h5py.File(os.path.join(
                    path_metric_src, f'{metric_key}.hdf5'), 'r')[f'{metric_key}_mean'], dtype=np.float32)
        f_h5.close()


def metric_mean_mix_to_file(path_metric, src_names, sub_sets, metric_keys, path_save):
    """Merge metric_mean files to .hdf5 file.
    Args:
        path_metric (str): path where to save files.
        src_names (list[str]): name of the sources.
        sub_sets (list[str]): name of the sets of datas.
        metric_keys (list[str]): name of the metrics.
        path_save (str): path to save merged file.
    """
    for metric_key in metric_keys:
        f_h5 = h5py.File(os.path.join(path_save, f'{metric_key}_mean.hdf5'), 'a')
        dir_names = list_dirs(path_metric, False)
        # dir_names = [name for name in dir_names if os.path.isdir(os.path.join(path_metric, name))]
        for dir_name in dir_names:
            grp_dir = f_h5.create_group(dir_name)
            path_metric_dir = os.path.join(path_metric, dir_name)
            for src_name in src_names:
                grp_src = grp_dir.create_group(src_name)
                path_metric_src = os.path.join(path_metric_dir, src_name)
                for sub_set in sub_sets:
                    grp_src[sub_set] = np.asarray(
                        h5py.File(os.path.join(path_metric_src, f'{metric_key}_{sub_set}.hdf5'),
                                  'r')[f'{metric_key}_{sub_set}_mean'], dtype=np.float32)
        f_h5.close()


def metric_mean_to_csv(file_name, level=1):
    """Merge metric_mean files to .csv file.
    Args:
        file_name (str): file name of the .hdf5 metric.
        level (int, optional): level of the data saved. Defaults to 1.
    """
    with h5py.File(file_name, 'r') as h5_f:
        if level == 1:
            data_arr = np.vstack([np.asarray(h5_f[key]) for key in h5_f.keys()])
        elif level == 2:
            data_list = []
            for key in h5_f.keys():
                data_list_2 = []
                for key_2 in h5_f[key].keys():
                    data_list_2.append(np.asarray(h5_f[key][key_2]))
                data_list.append(np.concatenate(data_list_2))
            data_arr = np.vstack(data_list)
        elif level == 3:
            data_list = []
            for key in h5_f.keys():
                data_list_2 = []
                for key_2 in h5_f[key].keys():
                    data_list_3 = []
                    for key_3 in h5_f[key][key_2].keys():
                        data_list_3.append(np.asarray(h5_f[key][key_2][key_3]))
                    data_list_2.append(np.concatenate(data_list_3))
                data_list.append(np.concatenate(data_list_2))
            data_arr = np.vstack(data_list)
        file_name, _ = os.path.splitext(file_name)
        np.savetxt(f'{file_name}.csv', data_arr, delimiter=",")


if __name__ == '__main__':
    import json
    from file_operation import mkdir  # pylint: disable=ungrouped-imports
    from prepare_data_shipsear_recognition_mix_s0tos3 import PathSourceRoot, read_data, read_datas

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO)
    # ========================================================================================
    # for shipsear data
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    PATH_CLASS = PathSourceRoot(PATH_DATA_ROOT, form_src='wav')
    PATH_DATA_SM = PATH_CLASS.path_source_root
    SM_NAMES = json.load(open(os.path.join(PATH_DATA_SM, 'dirname.json'), 'r'))['dirname']
    # ========================================================================================
    # # for no-silence sources
    PATH_DATA = os.path.join(PATH_DATA_SM, 'max_one_aens')
    # ========================================================================================
    X_NAMES = ['X_train', 'X_val', 'X_test']
    SET_NAMES = ['train', 'val', 'test']
    S_NAMES = [X_NAMES]  # [src][set]
    for j_src in range(4):
        S_NAMES.append([f'Z_{j_src}_{set_name_i}' for set_name_i in SET_NAMES])
    DATA_LIST = []
    for s_names_j in S_NAMES:
        DATA_LIST.append(read_datas(PATH_DATA, s_names_j))
    # ========================================================================================
    PATH_SAVE_ROOT = '../result_autoencoder'

    model_names = [name for name in list_files(PATH_SAVE_ROOT, False) if '.' not in name]

    # for no-silence sources
    model_names = [name for name in model_names if name.endswith('_2')]
    # ========================================================================================
    SP_NAMES = ['X']
    for j_src in range(4):
        SP_NAMES.append(f'Z_{j_src}')
    SET_NAMES = ['train', 'val', 'test']

    for model_name in model_names:
        path_metric_root = os.path.join(PATH_SAVE_ROOT, model_name, 'metric')
        mkdir(path_metric_root)

        path_out_root = os.path.join(PATH_SAVE_ROOT, model_name, 'auto_out')
        model_para_names = [name for name in list_files(path_out_root, False) if '.' not in name]
        for model_para_name in model_para_names:
            path_out = os.path.join(path_out_root, model_para_name)
            path_metric_para = os.path.join(path_metric_root, model_para_name)

            for i, s_names_i in enumerate(SP_NAMES):  # source_i
                path_save_i = os.path.join(path_metric_para, s_names_i)
                mkdir(path_save_i)
                logging.debug(f'source {i} {s_names_i}')

                s_list_i = DATA_LIST[i]
                sp_list_i = []
                for set_name_j in SET_NAMES:
                    sp_list_i.append(read_data(path_out, f'{s_names_i}_{set_name_j}_autodecoded'))
                    logging.debug(f'{path_out} {s_names_i}_{set_name_j}_autodecoded')

                mse_list, mse_mean = compute_metric(s_list_i, sp_list_i, mse_np)
                display_metric(mse_list, path_save_i, SET_NAMES,
                               hist_bins=(0, 1e-3, 10), save_name='mse')
                save_metric(path_save_i, 'mse', ['mse_mean'], [np.asarray(mse_mean)],
                            {'mse': dict(zip(SET_NAMES, mse_list))})

                sr_list, sr_mean = compute_metric(s_list_i, sp_list_i, samerate_acc_np)
                display_metric(sr_list, path_save_i, SET_NAMES,
                               hist_bins=(0.6, 1.0, 40), save_name='sr')
                save_metric(path_save_i, 'sr', ['sr_mean'], [np.asarray(sr_mean)],
                            {'sr': dict(zip(SET_NAMES, sr_list))})

                si_snr_list, si_snr_mean = compute_metric(s_list_i, sp_list_i, si_snr_np)
                display_metric(si_snr_list, path_save_i, SET_NAMES,
                               hist_bins=(0.6, 1.0, 40), save_name='si_snr')
                save_metric(path_save_i, 'si_snr', ['si_snr_mean'], [np.asarray(si_snr_mean)],
                            {'si_snr': dict(zip(SET_NAMES, si_snr_list))})

            s_mix = [(1, 2), (1, 3), (2, 3), (1, 2, 3)]
            for i, (sm_i, sm_name_i) in enumerate(zip(s_mix, SM_NAMES[4:])):
                path_save_i = os.path.join(path_metric_para, sm_name_i)
                mkdir(path_save_i)
                s_names_i = [SM_NAMES[sm_ik] for sm_ik in sm_i]

                z_list_i = []  # [set][src]
                for j_set in range(len(SET_NAMES)):
                    z_list_ij = [DATA_LIST[k+1][j_set] for k in sm_i]
                    z_list_i.append(z_list_ij)

                zp_list_i = []
                for set_name_j in SET_NAMES:
                    zp_list_ij = [read_data(path_out, f'Z_{k}_{set_name_j}_autodecoded') for k in sm_i]
                    zp_list_i.append(zp_list_ij)

                for z_list_ij, zp_list_ij, set_name_j in zip(z_list_i, zp_list_i, SET_NAMES):
                    z_arr_ij = np.asarray(z_list_ij).transpose(1, 0, 3, 2)  # (n_src,n_sams,1,fl)->(n_sams,n_src,fl,1)
                    zp_arr_ij = np.asarray(zp_list_ij)
                    if zp_arr_ij.ndim == 3:
                        zp_arr_ij = np.expand_dims(zp_arr_ij, -2)  # (n_src, n_sams, 1, fl)
                    zp_arr_ij = zp_arr_ij.transpose(1, 0, 3, 2)  # ->(n_sams, n_src, fl, 1)

                    compute_sdr(z_arr_ij, zp_arr_ij, path_save_i,
                                [f'sdr_{set_name_j}', f'isr_{set_name_j}', f'sir_{set_name_j}', f'sar_{set_name_j}'],
                                s_names_i, hist_bins=[(0, 30, 30), (0, 30, 30), (0, 30, 30), (0, 30, 30)],
                                save_name=[f'sdr_{set_name_j}', f'isr_{set_name_j}',
                                           f'sir_{set_name_j}', f'sar_{set_name_j}'])

                    z_arr_ij = z_arr_ij.transpose(0, 2, 3, 1)  # (n_sams, n_src, fl, 1)->(n_sams, fl, n_channel, n_src)
                    zp_arr_ij = zp_arr_ij.transpose(0, 2, 3, 1)  # (n_sams, n_src, fl, 1)->(n_sams, fl, n_channel,n_src)

                    compute_si_sdr(z_arr_ij, zp_arr_ij, True, path_save_i,
                                   [f'si_sdr_{set_name_j}', f'si_sir_{set_name_j}', f'si_sar_{set_name_j}'],
                                   s_names_i, hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                                   save_name=[f'si_sdr_{set_name_j}', f'si_sir_{set_name_j}', f'si_sar_{set_name_j}'])

                    compute_si_sdr(z_arr_ij, zp_arr_ij, False, path_save_i,
                                   [f'sd_sdr_{set_name_j}', f'sd_sir_{set_name_j}', f'sd_sar_{set_name_j}'],
                                   s_names_i, hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                                   save_name=[f'sd_sdr_{set_name_j}', f'sd_sir_{set_name_j}', f'sd_sar_{set_name_j}'])

        metric_mean_src_to_file(path_metric_root, SP_NAMES, ['mse', 'sr', 'si_snr'], path_metric_root)
        metric_mean_mix_to_file(path_metric_root, SM_NAMES[4:], SET_NAMES,
                                ['sdr', 'isr', 'sir', 'sar'], path_metric_root)
        metric_mean_mix_to_file(path_metric_root, SM_NAMES[4:], SET_NAMES,
                                ['si_sdr', 'si_sir', 'si_sar'], path_metric_root)
        metric_mean_mix_to_file(path_metric_root, SM_NAMES[4:], SET_NAMES,
                                ['sd_sdr', 'sd_sir', 'sd_sar'], path_metric_root)

        for metric_name in ['mse', 'sr', 'si_snr']:
            metric_mean_to_csv(os.path.join(path_metric_root, f'{metric_name}_mean.hdf5'), level=2)
        for metric_name in ['sdr', 'isr', 'sir', 'sar']:
            metric_mean_to_csv(os.path.join(path_metric_root, f'{metric_name}_mean.hdf5'), level=3)
        for metric_name in ['si_sdr', 'si_sir', 'si_sar']:
            metric_mean_to_csv(os.path.join(path_metric_root, f'{metric_name}_mean'), level=3)
        for metric_name in ['sd_sdr', 'sd_sir', 'sd_sar']:
            metric_mean_to_csv(os.path.join(path_metric_root, f'{metric_name}_mean'), level=3)
