# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:38:03 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation, too-few-public-methods, too-many-arguments
# pylint: disable=too-many-branches, too-many-locals, too-many-nested-blocks, too-many-statements

import logging
import os
import numpy as np

from file_operation import mkdir
from metric import mse_np, samerate_acc_np, si_snr_np
from prepare_data import list_transpose
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data, read_datas

from see_metric_autoencoder import save_metric, compute_metric, display_metric, compute_sdr
from see_metric_autoencoder import compute_si_sdr
from see_metric_autoencoder import recover_pad_num_samples, recover_pad_num_samples_list
from see_metric_single_source_ae_ns import data_reshape_same


def z_pred_filter(z_pred, y_pred, label):
    """Filter data with specific label.
    Args:
        z_pred (np.ndarray): datas to filter.
        y_pred (np.ndarray): labels of the datas.
        label (tuple(int)): the specific label to filter datas.
    Returns:
        np.ndarray: datas after filter.
    """
    z_pred_label = []
    for z_i, y_i in zip(np.asarray(z_pred), np.asarray(y_pred)):
        if tuple(y_i[0]) == label:
            z_pred_label.append(z_i)
    return np.asarray(z_pred_label)


def compute_metrics(object_decoded_files, s_names, src_names, src_names_st,
                    s_data, sm_index, set_names, zp_name_end, y_data, labels):
    """compute metrics.
    Args:
        object_decoded_files (class ListDecodedFiles): object for list predict decoded files
        s_names (list[str]): [src][set], list of name of datasets,
                                e.g. ['Z_zero_train', 'Z_zero_val', 'Z_zero_test'].
        src_names (list[str]): [src], list of name of sources including mixed sources,
                                e.g. ['Z_0_zero',...,'Z_7_zero'].
        src_names_st (list[str]): [src], list of name of single target sources, e.g. ['Z_0_zero',...,'Z_3_zero'].
        s_data (list[h5pyFile]): [set](nsams, nsrc, channel=1, fl), file object of datas.
        sm_index (list[tuple]): clean target source of mixed sources, e.g. [(1,2),...,(1,2,3)].
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
        zp_name_end (str): name of the predict file key word, e.g. 'autodecoded'.
        y_data (list[np.ndarray]): n_hot_labels of the samples.
        labels (list[tuple(int)]): standard n_hot_labels of the samples.
    """
    name_weight_files = object_decoded_files.name_weight_files  # [model][src][para_name][weight_file]
    path_out_models = object_decoded_files.path_out_models  # [model][src][para_name]
    path_metrics_paras = object_decoded_files.path_metrics_paras  # [model][para_name]

    for sm_index_i, sm_src_name_i, label_i in zip(sm_index, src_names, labels):  # level i: [source]
        logging.debug(f'sm_index_i {sm_index_i}')   # indexes of channels output target signals
        logging.debug(f'sm_src_name_i {sm_src_name_i}')
        data_s_filter_i = [z_pred_filter(s_data_i, y_data_i, label_i)
                           for y_data_i, s_data_i in zip(y_data, s_data)]  # [set](nsams, src, channel=1, fl)
        data_s_i = [[np.asarray(data_s_f_i)[:, k, :, :] for k in range(len(src_names_st))]
                    for data_s_f_i in data_s_filter_i]  # [set][src]
        s_data_i = list_transpose(data_s_i, warn=True)  # [src][set](nsams, channel=1, fl)

        src_name_sm_i = [src_names[k] for k in sm_index_i]  # [src] multitarget sources
        data_sm_i = [[np.asarray(data_s_i)[:, k, :, :] for k in sm_index_i]
                     for data_s_i in data_s_filter_i]  # [set][src]
        sm_data_i = list_transpose(data_sm_i, warn=True)  # [src][set](nsams, channel=1, fl)

        sz_index_i = list(range(len(src_names_st)))  # indexes of channels output zero
        for k in sm_index_i:
            sz_index_i.remove(k)
        src_name_sz_i = [src_names[k] for k in sz_index_i]  # [src] zero output channels

        for name_weight_files_j, path_metrics_para_j, path_out_model_j in zip(  # level j: [model]
                name_weight_files, path_metrics_paras, path_out_models):
            name_weight_files_j = list_transpose(name_weight_files_j)  # [para_name][src][weight_file]
            path_out_model_j = list_transpose(path_out_model_j)  # [para_name][src]

            for name_weight_files_k, path_metric_para_k, path_out_model_k in zip(  # level k: [para_name]
                    name_weight_files_j, path_metrics_para_j, path_out_model_j):
                path_metric_para_i = os.path.join(path_metric_para_k, sm_src_name_i)
                mkdir(path_metric_para_i)

                for name_weight_files_l, path_out_model_l in zip(  # level l:[src]
                        name_weight_files_k, path_out_model_k):
                    mkdir(path_out_model_l)

                    name_weight_file_m = name_weight_files_l[-1]  # level m: [weight_file]
                    path_sp_l = []  # [set]
                    data_sp_z_l = []  # [set]
                    for s_name_seti in s_names:
                        name_data_sp_z_l_i = f'{s_name_seti}_{zp_name_end}_{name_weight_file_m}.hdf5'
                        path_sp_l.append(os.path.join(path_out_model_l, name_data_sp_z_l_i))
                        data_sp_z_l.append(np.asarray(read_data(path_out_model_l, name_data_sp_z_l_i)))
                    data_sp_z_l = [z_pred_filter(data_sp_z_l_i, y_data_i, label_i)
                                   for y_data_i, data_sp_z_l_i in zip(y_data, data_sp_z_l)]  # [set](nsams, fl, nsrc=4)

                    sp_data_s_i = []  # [src][set](n_sams, fl, nsrc=1)
                    for k in range(len(src_names_st)):
                        sp_data_s_i.append([np.asarray(data_sp_z_l_n)[:, :, k:k+1] for data_sp_z_l_n in data_sp_z_l])

                    for data_s_l, data_sp_l, src_name_l in zip(s_data_i, sp_data_s_i, src_names_st):
                        path_metric_para_l = os.path.join(path_metric_para_i, src_name_l)
                        mkdir(path_metric_para_l)

                        data_sp_l_t = []  # [set](nsams, channel=1, fl)
                        for k, (data_s_l_k, data_sp_l_k) in enumerate(zip(data_s_l, data_sp_l)):
                            data_sp_l_t.append(data_reshape_same(data_s_l_k, data_sp_l_k)[1])  # (nsams, channel=1, fl)

                        data_sp_l_t = recover_pad_num_samples_list(data_s_l, data_sp_l_t)

                        mse_list, mse_mean = compute_metric(data_s_l, data_sp_l_t, mse_np)
                        display_metric(mse_list, path_metric_para_l, set_names,
                                       # hist_bins=(0, 1e-1, 10),
                                       save_name='mse')
                        save_metric(path_metric_para_l, 'mse', ['mse_mean'], [np.asarray(mse_mean)],
                                    {'mse': dict(zip(set_names, mse_list))})

                    sp_data_sm_i = []  # [src][set](n_sams, fl, nsrc=1)
                    for k in sm_index_i:
                        sp_data_sm_i.append([np.asarray(data_sp_z_l_n)[:, :, k:k+1] for data_sp_z_l_n in data_sp_z_l])

                    for data_s_l, data_sp_l, src_name_l in zip(sm_data_i, sp_data_sm_i, src_name_sm_i):
                        path_metric_para_l = os.path.join(path_metric_para_i, src_name_l)
                        mkdir(path_metric_para_l)

                        data_sp_l_t = []  # [set](nsams, channel=1, fl)
                        for k, (data_s_l_k, data_sp_l_k) in enumerate(zip(data_s_l, data_sp_l)):
                            data_sp_l_t.append(data_reshape_same(data_s_l_k, data_sp_l_k)[1])  # (nsams, channel=1, fl)

                        data_sp_l_t = recover_pad_num_samples_list(data_s_l, data_sp_l_t)

                        sr_list, sr_mean = compute_metric(data_s_l, data_sp_l_t, samerate_acc_np)
                        display_metric(sr_list, path_metric_para_l, set_names,
                                       # hist_bins=(0.6, 1.0, 40),
                                       save_name='sr')
                        save_metric(path_metric_para_l, 'sr', ['sr_mean'], [np.asarray(sr_mean)],
                                    {'sr': dict(zip(set_names, sr_list))})

                        si_snr_list, si_snr_mean = compute_metric(data_s_l, data_sp_l_t, si_snr_np)
                        display_metric(si_snr_list, path_metric_para_l, set_names,
                                       # hist_bins=(0.6, 1.0, 40),
                                       save_name='si_snr')
                        save_metric(path_metric_para_l, 'si_snr', ['si_snr_mean'], [np.asarray(si_snr_mean)],
                                    {'si_snr': dict(zip(set_names, si_snr_list))})

                    sp_data_sz_i = []  # [src][set](n_sams, fl, nsrc=1)
                    for k in sz_index_i:
                        sp_data_sz_i.append([np.asarray(data_sp_z_l_n)[:, :, k:k+1] for data_sp_z_l_n in data_sp_z_l])

                    for data_sp_l, src_name_sz_l in zip(sp_data_sz_i, src_name_sz_i):   # [set] in [src][set]
                        path_metric_para_l = os.path.join(path_metric_para_i, src_name_sz_l)

                        for data_s_l_lj, src_name_l_lj in zip(sm_data_i, src_name_sm_i):   # [set] in [src][set]
                            path_metric_para_l_lj = os.path.join(path_metric_para_l, src_name_l_lj)
                            mkdir(path_metric_para_l_lj)

                            data_sp_l_t = []  # [set](nsams, channel=1, fl)
                            for k, (data_s_l_lj_k, data_sp_l_k) in enumerate(zip(data_s_l_lj, data_sp_l)):
                                data_sp_l_t.append(data_reshape_same(data_s_l_lj_k, data_sp_l_k)[1])

                            data_sp_l_t = recover_pad_num_samples_list(data_s_l_lj, data_sp_l_t)

                            sr_list, sr_mean = compute_metric(data_s_l_lj, data_sp_l_t, samerate_acc_np)
                            display_metric(sr_list, path_metric_para_l_lj, set_names,
                                           # hist_bins=(0.6, 1.0, 40),
                                           save_name='sr')
                            save_metric(path_metric_para_l_lj, 'sr', ['sr_mean'], [np.asarray(sr_mean)],
                                        {'sr': dict(zip(set_names, sr_list))})

                            si_snr_list, si_snr_mean = compute_metric(data_s_l_lj, data_sp_l_t, si_snr_np)
                            display_metric(si_snr_list, path_metric_para_l_lj, set_names,
                                           # hist_bins=(0.6, 1.0, 40),
                                           save_name='si_snr')
                            save_metric(path_metric_para_l_lj, 'si_snr', ['si_snr_mean'], [np.asarray(si_snr_mean)],
                                        {'si_snr': dict(zip(set_names, si_snr_list))})

                    if len(sm_index_i) > 1:
                        data_sp_sm_i_c = list_transpose(sp_data_sm_i)  # [set][src](n_sams, fl, nsrc=1)

                        for data_s_n, data_sp_n, set_name_n in zip(
                                data_sm_i, data_sp_sm_i_c, set_names):

                            data_s_n = np.concatenate(data_s_n, axis=1)  # (n_sams, nsrc, fl)
                            data_sp_n = np.concatenate(data_sp_n, axis=2)  # (n_sams, fl, nsrc)

                            if data_s_n.ndim == 3:
                                data_s_n = np.expand_dims(data_s_n, -1)  # (n_sams, nsrc, fl, channel=1)
                            if data_sp_n.ndim == 3:
                                data_sp_n = np.expand_dims(data_sp_n, -1)  # (n_sams, fl, nsrc, channel=1)
                            data_sp_n = data_sp_n.transpose((0, 2, 1, 3))  # (n_sams, nsrc=4, fl, channel=1)

                            data_sp_n = recover_pad_num_samples(data_s_n, data_sp_n)

                            compute_sdr(data_s_n, data_sp_n, path_metric_para_i,
                                        [f'sdr_{set_name_n}', f'isr_{set_name_n}',
                                            f'sir_{set_name_n}', f'sar_{set_name_n}'],
                                        src_name_sm_i, hist_bins=[(0, 30, 30), (0, 30, 30), (0, 30, 30), (0, 30, 30)],
                                        save_name=[f'sdr_{set_name_n}', f'isr_{set_name_n}',
                                                   f'sir_{set_name_n}', f'sar_{set_name_n}'])

                            data_s_n = data_s_n.transpose((0, 2, 3, 1))
                            # (n_sams, n_src, fl, 1) -> (n_sams, fl, n_channel, n_src)
                            data_sp_n = data_sp_n.transpose((0, 2, 3, 1))
                            # (n_sams, n_src, fl, 1) -> (n_sams, fl, n_channel, n_src)

                            compute_si_sdr(data_s_n, data_sp_n, True, path_metric_para_i,
                                           [f'si_sdr_{set_name_n}', f'si_sir_{set_name_n}', f'si_sar_{set_name_n}'],
                                           src_name_sm_i,
                                           # hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                                           save_name=[f'si_sdr_{set_name_n}',
                                                      f'si_sir_{set_name_n}',
                                                      f'si_sar_{set_name_n}'])

                            compute_si_sdr(data_s_n, data_sp_n, False, path_metric_para_i,
                                           [f'sd_sdr_{set_name_n}', f'sd_sir_{set_name_n}', f'sd_sar_{set_name_n}'],
                                           src_name_sm_i,
                                           # hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                                           save_name=[f'sd_sdr_{set_name_n}',
                                                      f'sd_sir_{set_name_n}',
                                                      f'sd_sar_{set_name_n}'])


if __name__ == '__main__':
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep
    from recognition_mix_shipsear_s0tos3_preprocess import n_hot_labels
    from see_metric_single_source_ae_ns import ListDecodedFiles  # pylint: disable=ungrouped-imports

    # for shipEear data
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    SCALER_DATA = 'max_one'
    SUB_SET_WAY = 'rand'

    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav', scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source

    SET_NAMES = ['train', 'val', 'test']
    X_NAMES = ['X']
    X_SET_NAMES = [[f'{x_names_i}_{name_set_j}' for name_set_j in SET_NAMES] for x_names_i in X_NAMES]
    X_DATA = [read_datas(PATH_DATA, name_i) for name_i in X_SET_NAMES]
    X_DICT = [dict(zip(x_set_names_j, read_datas(PATH_DATA, x_set_names_j))) for x_set_names_j in X_SET_NAMES]
    Y_NAMES = ['Y']
    Y_SET_NAMES = [f'{y_names_i}_{name_set_j}' for name_set_j in SET_NAMES for y_names_i in Y_NAMES]
    Y_DATA = [read_data(PATH_DATA, name_i) for name_i in Y_SET_NAMES]
    Z_NAMES_R = ['Z']
    Z_SET_NAMES_R = [f'{z_name_i}_{name_set_j}_zero' for name_set_j in SET_NAMES for z_name_i in Z_NAMES_R]
    Z_DATA = [read_data(PATH_DATA, name_i) for name_i in Z_SET_NAMES_R]

    Z_SRC_NAME = ['Z_zero']
    Z_SRC_NAMES = [f'Z_{i}_zero' for i in range(8)]
    Z_NAMES = [f'{src_name}_{set_name_j}' for set_name_j in SET_NAMES for src_name in Z_SRC_NAME]
    Z_SET_NAMES = [[f'{z_name_i}_zero_{name_set_j}' for name_set_j in SET_NAMES] for z_name_i in Z_NAMES]

    SM_INDEX = [(0,), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    LABELS_N_HOT = tuple(tuple(label_i) for label_i in n_hot_labels(4))
    SM_NAMES = [f'Z_{i}_zero' for i in range(4, 8)]

    PATH_SAVE_ROOT = '../result_separation_one_autoencoder'

    object_decoded_files_ = ListDecodedFiles(PATH_SAVE_ROOT, Z_SRC_NAME)

    compute_metrics(object_decoded_files_, Z_NAMES, Z_SRC_NAMES, Z_SRC_NAMES[:4],
                    Z_DATA, SM_INDEX, SET_NAMES, 'autodecoded',
                    Y_DATA, LABELS_N_HOT)

    logging.info('finished')
