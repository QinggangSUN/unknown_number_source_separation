# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:38:03 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation, too-few-public-methods, too-many-arguments
# pylint: disable=too-many-branches, too-many-locals, too-many-nested-blocks, too-many-statements
# pylint: disable=useless-object-inheritance
from pathlib import Path
import logging
import os
import shutil
import numpy as np

from error import ParameterError
from file_operation import list_dirs, list_dirs_start_str, list_files_end_str, mkdir
from metric import mse_np, samerate_acc_np, si_snr_np
from prepare_data import list_transpose
from prepare_data_shipsear_recognition_mix_s0tos3 import concat_h5py, read_data, read_datas, save_datas
from see_metric_autoencoder import save_metric, compute_metric, display_metric, compute_sdr, compute_si_sdr
from see_metric_single_source_ae_ns import data_reshape_same
from see_metric_separation_one_autoencoder import z_pred_filter


class ListDecodedFiles(object):
    """class for list predict decoded files."""

    def __init__(self, path_save_root, search_dir_name='search_decoder', predict_dir_name='predict_decoder'):
        """__init__
        Args:
            path_save_root (str): Root path where save result.
            search_dir_name (str): dir name where save search_decoder result root.
            predict_dir_name (str): dir name where save search_decoder predict result.
        """
        name_models = list_dirs_start_str(path_save_root, 'model_', False)
        self.name_models = name_models
        path_models = [os.path.join(path_save_root, name_model) for name_model in name_models]
        self.path_models = path_models

        path_paras_one = []  # [model][para_1]
        for path_model in path_models:
            path_paras_one_model = list_dirs(os.path.join(path_model, search_dir_name))
            path_paras_one.append(path_paras_one_model)
        self.path_paras_one = path_paras_one

        name_paras_two = []  # [model][para_1][para_2]
        path_paras_two = []  # [model][para_1][para_2]
        path_metrics = []  # [model][para_1][para_2]
        for path_paras_one_model in path_paras_one:
            name_paras_two_model = []  # [para_1][para_2]
            path_paras_two_model = []  # [para_1][para_2]
            path_metrics_model = []  # [para_1][para_2]
            for path_para_one in path_paras_one_model:
                path_metric_para_one = os.path.join(path_para_one, 'metric')
                mkdir(path_metric_para_one)
                name_paras_two_para_one = list_dirs(os.path.join(path_para_one, predict_dir_name), False)
                path_paras_two_para_one = [os.path.join(path_para_one, predict_dir_name, name)
                                           for name in name_paras_two_para_one]
                path_metrics_para_one = [os.path.join(path_metric_para_one, name) for name in name_paras_two_para_one]
                name_paras_two_model.append(name_paras_two_para_one)
                path_paras_two_model.append(path_paras_two_para_one)
                path_metrics_model.append(path_metrics_para_one)
            name_paras_two.append(name_paras_two_model)
            path_paras_two.append(path_paras_two_model)
            path_metrics.append(path_metrics_model)
        self.name_paras_two = name_paras_two
        self.path_paras_two = path_paras_two
        self.path_metrics = path_metrics

    def filter_model(self, model_name, item):
        """Filter item belong to specific model.
        Args:
            model_name (str): name of the model.
            item (str): keword name of the item.
        Returns:
            items_filter: item belong to specific model.
        """
        num_index_model = self.name_models.index(model_name)
        if item == 'path_paras_one':
            items_filter = self.path_paras_one[num_index_model]
        elif item == 'name_paras_two':
            items_filter = self.name_paras_two[num_index_model]
        elif item == 'path_paras_two':
            items_filter = self.path_paras_two[num_index_model]
        elif item == 'path_metrics':
            items_filter = self.path_metrics[num_index_model]
        elif item == 'path_model':
            items_filter = self.path_models[num_index_model]
        return items_filter


def compute_metrics(object_decoded_files, s_names, src_names, src_names_st,
                    s_data, sm_index, set_names, zp_name_end, y_data, labels):
    """compute metrics. Only use this method with one para model.
    Args:
        object_decoded_files (class ListDecodedFiles): object for list predict decoded files
        s_names (list[str]): [src][set], list of name of datasets, e.g. ['Z_zero_train', 'Z_zero_val', 'Z_zero_test'].
        src_names (list[str]): [src], list of name of all sources including mixed sources,
                                e.g. ['Z_0_zero',...,'Z_7_zero'].
        src_names_st (list[str]): [src], list of name of single target sources, e.g. ['Z_0_zero',...,'Z_3_zero'].
        s_data (list[h5pyFile]): [set](nsams, nsrc, channel=1, fl), file object of datas.
        sm_index (list[tuple]): clean target source of mixed sources, e.g. [(1,2),...,(1,2,3)].
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
        zp_name_end (str): name of the predict file key word, e.g. 'decoded'.
        y_data (list[np.ndarray]): n_hot_labels of the samples.
        labels (list[tuple(int)]): standard n_hot_labels of the samples.
    Examples:
        ./search_decoder/1_n3_50
        ├─loss_search
        | └─1_n1_50
        |     X_test_0_loss_1_n1_50.svg
        |     ...
        |     X_test_8160_loss_1_n1_50.svg
        ├─predict_decoder
        | └─1_n1_50
        |     X_test_conv_weight.hdf5 shape==(4,8160,1,1,84224)
        |     Z_zero_test_decoded.hdf5 shape==(8160,10547,4)
        |     Z_0_zero_test_decoded.hdf5 shape==(8160,10547,1)
        |     Z_0_zero_test_encoded.hdf5
        |     ...
        |     Z_3_zero_test_decoded.hdf5 shape==(8160,10547,1)
        |     Z_3_zero_test_encoded.hdf5
        └─metric
          ├─1_n1_50
          | ├─Z_0_zero
          | | ├─Z_0_zero
          | | |   mse.hdf5
          | | |   sr.hdf5
          | | ├─Z_1_zero
          | | | | mse.hdf5
          | | | └─Z_0_zero
          | | |     sr.hdf5
          | | ├─Z_2_zero
          | | | | mse.hdf5
          | | | └─Z_0_zero
          | | |     sr.hdf5
          | | └─Z_3_zero
          | |   | mse.hdf5
          | |   └─Z_0_zero
          | |       sr.hdf5
          | ├─...
          | └─Z_7_zero
          |   | sdr_test.hdf5
          |   | si_sdr_test.hdf5
          |   ├─Z_0_zero
          |   | | mse.hdf5
          |   | ├─Z_1_zero
          |   | |   sr.hdf5
          |   | ├─Z_2_zero
          |   | |   sr.hdf5
          |   | └─Z_3_zero
          |   |     sr.hdf5
          |   ├─Z_1_zero
          |   |   mse.hdf5
          |   |   sr.hdf5
          |   ├─Z_2_zero
          |   |   mse.hdf5
          |   |   sr.hdf5
          |   └─Z_3_zero
          |       mse.hdf5
          |       sr.hdf5
          ├─2_n2_100
          ├─...
          └─best_merge
            ├─Z_0_zero
            | ├─best_mse
            | ├─...
            | └─best_sr
            ├─...
            |
            └─Z_7_zero
              ├─best_mse
              ├─...
              └─best_sr
    """
    path_paras_two = object_decoded_files.path_paras_two  # [model][para_1][para_2]
    path_metrics = object_decoded_files.path_metrics  # [model][para_1][para_2]

    for sm_index_i, sm_src_name_i, label_i in zip(sm_index, src_names, labels):  # level i: [source]
        logging.debug(f'sm_index_i {sm_index_i}')
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
        # data_sz_i = [[np.asarray(data_s_i)[:, k, :, :] for k in sz_index_i]
        #              for data_s_i in data_s_filter_i]  # [set][src]
        # sz_data_i = list_transpose(data_sz_i, warn=True)  # [src][set](nsams, channel=1, fl)

        for path_paras_two_model, path_metrics_model in zip(  # level: [model]
                path_paras_two, path_metrics):

            for path_paras_two_para_one, path_metrics_para_one in zip(  # level: [para_1]
                    path_paras_two_model, path_metrics_model):

                for path_paras_two_para_two, path_metrics_para_two in zip(  # level: [para_2]
                        path_paras_two_para_one, path_metrics_para_one):
                    mkdir(path_metrics_para_two)

                    path_metric_i = os.path.join(path_metrics_para_two, sm_src_name_i)
                    mkdir(path_metric_i)

                    path_sp = []  # [set], level l: [set]
                    data_sp_z = []  # [set]
                    for s_name_l in s_names:
                        name_data_sp_z_l = f'{s_name_l}_{zp_name_end}.hdf5'  # (nsams, fl, nsrc=4)
                        path_sp.append(os.path.join(path_paras_two_para_two, name_data_sp_z_l))
                        data_sp_z.append(np.asarray(read_data(path_paras_two_para_two, name_data_sp_z_l)))
                    data_sp_z = [z_pred_filter(data_sp_z_l, y_data_l, label_i)
                                 for y_data_l, data_sp_z_l in zip(y_data, data_sp_z)]  # [set](nsams, fl, nsrc=4)

                    sp_data_s_i = []  # [src][set](n_sams, fl, nsrc=1)
                    for k in range(len(src_names_st)):
                        sp_data_s_i.append([np.asarray(data_sp_z_l)[:, :, k:k+1] for data_sp_z_l in data_sp_z])

                    for data_s_l, data_sp_l, src_name_l in zip(s_data_i, sp_data_s_i, src_names_st):
                        path_metric_l = os.path.join(path_metric_i, src_name_l)
                        mkdir(path_metric_l)

                        data_sp_l_t = []  # [set](nsams, channel=1, fl)
                        for k, (data_s_l_k, data_sp_l_k) in enumerate(zip(data_s_l, data_sp_l)):
                            data_sp_l_t.append(data_reshape_same(data_s_l_k, data_sp_l_k)[1])  # (nsams, channel=1, fl)

                        mse_list, mse_mean = compute_metric(data_s_l, data_sp_l_t, mse_np)
                        display_metric(mse_list, path_metric_l, set_names,
                                       # hist_bins=(0, 1e-1, 10),
                                       save_name='mse')
                        save_metric(path_metric_l, 'mse', ['mse_mean'], [np.asarray(mse_mean)],
                                    {'mse': dict(zip(set_names, mse_list))})

                    sp_data_sm_i = []  # [src][set](n_sams, fl, nsrc=1)
                    for k in sm_index_i:
                        sp_data_sm_i.append([np.asarray(data_sp_z_l)[:, :, k:k+1] for data_sp_z_l in data_sp_z])

                    for data_s_l, data_sp_l, src_name_l in zip(sm_data_i, sp_data_sm_i, src_name_sm_i):
                        path_metric_l = os.path.join(path_metric_i, src_name_l)
                        mkdir(path_metric_l)

                        data_sp_l_t = []
                        for k, (data_s_l_k, data_sp_l_k) in enumerate(zip(data_s_l, data_sp_l)):
                            data_sp_l_t.append(data_reshape_same(data_s_l_k, data_sp_l_k)[1])  # (nsams, channel=1, fl)

                        sr_list, sr_mean = compute_metric(data_s_l, data_sp_l_t, samerate_acc_np)
                        display_metric(sr_list, path_metric_l, set_names,
                                       # hist_bins=(0.6, 1.0, 40),
                                       save_name='sr')
                        save_metric(path_metric_l, 'sr', ['sr_mean'], [np.asarray(sr_mean)],
                                    {'sr': dict(zip(set_names, sr_list))})

                        si_snr_list, si_snr_mean = compute_metric(data_s_l, data_sp_l_t, si_snr_np)
                        display_metric(si_snr_list, path_metric_l, set_names,
                                       # hist_bins=(0.6, 1.0, 40),
                                       save_name='si_snr')
                        save_metric(path_metric_l, 'si_snr', ['si_snr_mean'], [np.asarray(si_snr_mean)],
                                    {'si_snr': dict(zip(set_names, si_snr_list))})

                    sp_data_sz_i = []  # [src][set](n_sams, fl, nsrc=1)
                    for k in sz_index_i:
                        sp_data_sz_i.append([np.asarray(data_sp_z_l)[:, :, k:k+1] for data_sp_z_l in data_sp_z])

                    for data_sp_l, src_name_sz_l in zip(sp_data_sz_i, src_name_sz_i):   # [set] in [src][set]
                        path_metric_l = os.path.join(path_metric_i, src_name_sz_l)
                        # mkdir(path_metric_l)
                        for data_s_l_lj, src_name_l_lj in zip(sm_data_i, src_name_sm_i):   # [set] in [src][set]
                            path_metric_l_lj = os.path.join(path_metric_l, src_name_l_lj)
                            mkdir(path_metric_l_lj)

                            data_sp_l_t = []  # [set](nsams, channel=1, fl)
                            for k, (data_s_l_lj_k, data_sp_l_k) in enumerate(zip(data_s_l_lj, data_sp_l)):
                                data_sp_l_t.append(data_reshape_same(data_s_l_lj_k, data_sp_l_k)[1])

                            sr_list, sr_mean = compute_metric(data_s_l_lj, data_sp_l_t, samerate_acc_np)
                            display_metric(sr_list, path_metric_l_lj, set_names,
                                           # hist_bins=(0.6, 1.0, 40),
                                           save_name='sr')
                            save_metric(path_metric_l_lj, 'sr', ['sr_mean'], [np.asarray(sr_mean)],
                                        {'sr': dict(zip(set_names, sr_list))})

                            si_snr_list, si_snr_mean = compute_metric(data_s_l_lj, data_sp_l_t, si_snr_np)
                            display_metric(si_snr_list, path_metric_l_lj, set_names,
                                           # hist_bins=(0.6, 1.0, 40),
                                           save_name='si_snr')
                            save_metric(path_metric_l_lj, 'si_snr', ['si_snr_mean'], [np.asarray(si_snr_mean)],
                                        {'si_snr': dict(zip(set_names, si_snr_list))})

                    if len(sm_index_i) > 1:
                        # data_s_i_c = list_transpose(s_data_sm_i)  # [set][src](n_sams, nsrc=1, fl)
                        data_sp_sm_i_c = list_transpose(sp_data_sm_i)  # [set][src](n_sams, fl, nsrc=1)

                        for data_s_n, data_sp_n, set_name_n in zip(
                                data_sm_i, data_sp_sm_i_c, set_names):

                            data_s_n = np.concatenate(data_s_n, axis=1)  # (n_sams, nsrc, fl)
                            data_sp_n = np.concatenate(data_sp_n, axis=2)  # (n_sams, fl, nsrc)

                            if data_s_n.ndim == 3:
                                data_s_n = np.expand_dims(data_s_n, -1)  # (n_sams, nsrc, fl, channel=1)
                            if data_sp_n.ndim == 3:
                                data_sp_n = np.expand_dims(data_sp_n, -1)  # (n_sams, fl, nsrc, channel=1)
                            data_sp_n = data_sp_n.transpose(0, 2, 1, 3)  # (n_sams, nsrc=4, fl, channel=1)

                            compute_sdr(data_s_n, data_sp_n, path_metric_i,
                                        [f'sdr_{set_name_n}', f'isr_{set_name_n}',
                                         f'sir_{set_name_n}', f'sar_{set_name_n}'],
                                        src_name_sm_i, hist_bins=[(0, 30, 30), (0, 30, 30), (0, 30, 30), (0, 30, 30)],
                                        save_name=[f'sdr_{set_name_n}', f'isr_{set_name_n}',
                                                   f'sir_{set_name_n}', f'sar_{set_name_n}'])

                            data_s_n = data_s_n.transpose(0, 2, 3, 1)
                            # (n_sams, n_src, fl, 1) -> (n_sams, fl, n_channel, n_src)
                            data_sp_n = data_sp_n.transpose(0, 2, 3, 1)
                            # (n_sams, n_src, fl, 1) -> (n_sams, fl, n_channel, n_src)

                            compute_si_sdr(data_s_n, data_sp_n, True, path_metric_i,
                                           [f'si_sdr_{set_name_n}', f'si_sir_{set_name_n}', f'si_sar_{set_name_n}'],
                                           src_name_sm_i,
                                           # hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                                           save_name=[f'si_sdr_{set_name_n}',
                                                      f'si_sir_{set_name_n}',
                                                      f'si_sar_{set_name_n}'])

                            compute_si_sdr(data_s_n, data_sp_n, False, path_metric_i,
                                           [f'sd_sdr_{set_name_n}', f'sd_sir_{set_name_n}', f'sd_sar_{set_name_n}'],
                                           src_name_sm_i,
                                           # hist_bins=[(0, 100, 20), (0, 100, 20), (0, 100, 20)],
                                           save_name=[f'sd_sdr_{set_name_n}',
                                                      f'sd_sir_{set_name_n}',
                                                      f'sd_sar_{set_name_n}'])


def concat_files(path_dir_1, path_dir_2, path_dir_new, str_end, create_h5py, axis):
    """Concatenate files in two directories to a new directory. The files in two directories
        should have save names.
    Args:
        path_dir_1 (str): absolute path of a directory.
        path_dir_2 (str): absolute path of a directory.
        path_dir_new (str): absolute path of a directory.
        str_end (str): keyword to search and filter files.
        create_h5py (bool): whether create a new file in the new directory.
        axis (int): the axis to concatenate.
    """
    file_names = list_files_end_str(path_dir_1, str_end, full=False)
    for file_name in file_names:
        fname_1 = os.path.join(path_dir_1, file_name)
        fname_2 = os.path.join(path_dir_2, file_name)
        fname_new = os.path.join(path_dir_new, file_name)
        concat_h5py(fname_1, fname_2, fname_new, create_h5py=create_h5py, axis=axis,
                    save_key='data', mode_batch='batch_h5py', batch_num=200, dtype=None)


def concat_results(num_list, path_root, dir_para_2, dir_para_1='predict_decoder', name_set='test'):
    """Concatenate result files, only use this method when train samples by batches.
    Args:
        num_list (list[int]): last index numbers of the samples of each batch.
            e.g. [1000, 2000, 5000, 8160].
        path_root (str): root path where saved search conv paras results.
        dir_para_2 (str): name of the directory, also is the name of the para during searching.
        dir_para_1 (str, optional): name of the directory where saved model output files. Defaults to 'predict_decoder'.
        name_set (str): name of the dataset.
    Example:
        before cancatenate files:
        ./search_decoder/1_n3_50
        └─test_sets_1_n1_50
          ├─test_0_4000
          | ├─loss_search
          | |   X_test_0_loss_1_n1_50.svg
          | |   ...
          | |   X_test_4000_loss_1_n1_50.svg
          | └─predict_decoder
          |   └─1_n1_50
          |       X_test_conv_weight.hdf5 shape==(4,4000,1,1,84224)
          |       Z_zero_test_decoded.hdf5 shape==(4000,10547,4)
          |       Z_0_zero_test_decoded.hdf5 shape==(4000,10547,1)
          |       Z_0_zero_test_encoded.hdf5
          |       ...
          |       Z_3_zero_test_decoded.hdf5 shape==(4000,10547,1)
          |       Z_3_zero_test_encoded.hdf5
          └─test_4000_8160
            ├─loss_search
            |   X_test_0_loss_1_n1_50.svg
            |   ...
            |   X_test_4160_loss_1_n1_50.svg
            └─predict_decoder
              └─1_n1_50
                  X_test_conv_weight.hdf5 shape==(4,4160,1,1,84224)
                  Z_zero_test_decoded.hdf5 shape==(4160,10547,4)
                  Z_0_zero_test_decoded.hdf5 shape==(4160,10547,1)
                  Z_0_zero_test_encoded.hdf5
                  ...
                  Z_3_zero_test_decoded.hdf5 shape==(4160,10547,1)
                  Z_3_zero_test_encoded.hdf5

            after cancatenate files:
            ./search_decoder/1_n3_50
            └─1_n1_50_test_sets
              └─test_0_8160
                ├─loss_search
                |   X_test_0_loss_1_n1_50.svg
                |   ...
                |   X_test_8160_loss_1_n1_50.svg
                └─predict_decoder
                  └─1_n1_50
                      X_test_conv_weight.hdf5 shape==(4,4160,1,1,84224)
                      Z_zero_test_decoded.hdf5 shape==(4160,10547,4)
                      Z_0_zero_test_decoded.hdf5 shape==(4160,10547,1)
                      Z_0_zero_test_encoded.hdf5
                      ...
                      Z_3_zero_test_decoded.hdf5 shape==(4160,10547,1)
                      Z_3_zero_test_encoded.hdf5

            manually organize files:
            ./search_decoder/1_n3_50
            ├─loss_search
            | └─1_n1_50
            |     X_test_0_loss_1_n1_50.svg
            |     ...
            |     X_test_8160_loss_1_n1_50.svg
            └─predict_decoder
              └─1_n1_50
                  X_test_conv_weight.hdf5 shape==(4,8160,1,1,84224)
                  Z_zero_test_decoded.hdf5 shape==(8160,10547,4)
                  Z_0_zero_test_decoded.hdf5 shape==(8160,10547,1)
                  Z_0_zero_test_encoded.hdf5
                  ...
                  Z_3_zero_test_decoded.hdf5 shape==(8160,10547,1)
                  Z_3_zero_test_encoded.hdf5
    """
    len_num = len(num_list)
    for j, num_j in enumerate(num_list):
        if j+1 < len_num:
            num_k = num_list[j+1]
            path_dir_1 = os.path.join(path_root, f'{name_set}_{0}_{num_j}', dir_para_1, dir_para_2)
            path_dir_2 = os.path.join(path_root, f'{name_set}_{num_j}_{num_k}', dir_para_1, dir_para_2)
            path_dir_new = os.path.join(path_root, f'{name_set}_{0}_{num_k}', dir_para_1, dir_para_2)
            mkdir(path_dir_new)
            create_h5py = bool(j == 0)
            concat_files(path_dir_1, path_dir_2, path_dir_new, '_weight.hdf5', create_h5py, axis=1)
            concat_files(path_dir_1, path_dir_2, path_dir_new, '_encoded.hdf5', create_h5py, axis=0)
            concat_files(path_dir_1, path_dir_2, path_dir_new, '_decoded.hdf5', create_h5py, axis=0)
            if j > 0:
                shutil.rmtree(path_dir_1)


def compute_best_metric_st(path_metrics, path_best_metric, name_metric, name_set, better='big'):
    """Compute the best metric with which para for sources containing single target.
    Args:
        path_metrics (list[str]): list of the paths where saved results of metrics.
            e.g. ['./metric/1_n1_50/Z_0_zero/Z_0_zero', './metric/2_n2_50/Z_0_zero/Z_0_zero'].
        path_best_metric (str): path where to save index and para of merged best metrics.
            e.g. '/metric/best_merge/Z_0_zero/best_mse'.
        name_metric (str): name of the metric to compute. e.g. 'mse'.
        name_set (str): name of the dataset. e.g. 'train', 'val', 'test'.
        better (str, optional): the standard to decide better metrics. Defaults to 'big'.
    """
    best_metric = []  # length==number of the samples
    best_metric_para = []
    best_metric_index = []
    for i, path_metric_i in enumerate(path_metrics):
        data_metric_i = read_data(path_metric_i, name_metric, dict_key=name_metric)[name_set]
        for k, data_metric_ki in enumerate(data_metric_i):
            if i == 0:
                best_metric.append(data_metric_ki)
                best_metric_para.append(os.path.basename(Path(path_metric_i).parents[1]))
                best_metric_index.append(i)
            else:
                if ((better == 'big' and data_metric_ki > best_metric[k])
                        or (better == 'small' and data_metric_ki < best_metric[k])):
                    best_metric[k] = data_metric_ki
                    best_metric_para[k] = os.path.basename(Path(path_metric_i).parents[1])
                    best_metric_index[k] = i

    save_datas({f'{name_metric}_{name_set}_para': np.asarray(best_metric_para)}, path_best_metric, **{'dtype': 'str'})
    save_datas({f'{name_metric}_{name_set}_para': [best_metric_para]}, path_best_metric, **{'form_save': 'csv'})
    save_datas({f'{name_metric}_{name_set}_index': np.asarray(best_metric_index)}, path_best_metric, **{'dtype': int})


def compute_best_metric_mt(path_metrics, path_metric_subs, path_best_metric,
                           name_metric, name_set, better='big'):
    """Compute the best metric with which para for sources containing multiple targets.
    Args:
        path_metrics (list[str]): list of the paths where saved results of metrics.
            e.g. ['./metric/1_n1_50/Z_4_zero', './metric/2_n2_50/Z_4_zero'].
        path_metric_subs (list[list[str]]): 2D list of the paths where saved results of metrics of the sources.
            e.g. [['./metric/1_n1_50/Z_4_zero/Z_1_zero', './metric/1_n1_50/Z_4_zero/Z_2_zero'],
                  ['./metric/2_n2_50/Z_4_zero/Z_1_zero', './metric/2_n2_50/Z_4_zero/Z_2_zero']].
        path_best_metric (str): path where to save index and para of merged best metrics.
            e.g. '/metric/best_merge/Z_4_zero/best_mse'.
        name_metric (str): name of the metric to compute. e.g. 'mse'.
        name_set (str): name of the dataset.  e.g. 'train', 'val', 'test'.
        better (str, optional): the standard to decide better metrics. Defaults to 'big'.
    """
    best_metric = []  # length==number of the samples
    best_metric_para = []
    best_metric_index = []

    if name_metric in ['mse', 'sr', 'si_snr']:
        num_samples = read_data(path_metric_subs[0][0],
                                name_metric,
                                dict_key=name_metric)[name_set].shape[0]
    elif name_metric in ['sdr', 'si_sdr', 'sd_sdr']:
        grp_srcs = read_data(path_metrics[0],
                             f'{name_metric}_{name_set}',
                             dict_key=f'{name_metric}_{name_set}')
        num_samples = grp_srcs[next(iter(grp_srcs))].shape[0]

    for k in range(num_samples):
        for i, (path_metric_i, path_subs_i) in enumerate(zip(path_metrics, path_metric_subs)):  # [para][src]
            data_metric_ki = []  # [src]
            if name_metric in ['mse', 'sr', 'si_snr']:
                for path_sub_j in path_subs_i:
                    data_metric_ki.append(read_data(path_sub_j,
                                                    name_metric,
                                                    dict_key=name_metric)[name_set][k])
            elif name_metric in ['sdr', 'si_sdr', 'sd_sdr']:
                grp_srcs = read_data(path_metric_i,
                                     f'{name_metric}_{name_set}',
                                     dict_key=f'{name_metric}_{name_set}')
                for name_src in grp_srcs:
                    data_metric_ki.append(grp_srcs[name_src][k])

            if i == 0:
                best_metric.append(data_metric_ki)
                best_metric_para.append(os.path.basename(Path(path_metric_i).parents[0]))
                best_metric_index.append(i)
            else:
                if ((better == 'big' and np.average(data_metric_ki) > np.average(best_metric[k]))
                        or (better == 'small' and np.average(data_metric_ki) < np.average(best_metric[k]))):
                    best_metric[k] = data_metric_ki
                    best_metric_para[k] = os.path.basename(Path(path_metric_i).parents[0])
                    best_metric_index[k] = i

    save_datas({f'{name_metric}_{name_set}_para': np.asarray(best_metric_para)}, path_best_metric, **{'dtype': 'str'})
    save_datas({f'{name_metric}_{name_set}_para': [best_metric_para]}, path_best_metric, **{'form_save': 'csv'})
    save_datas({f'{name_metric}_{name_set}_index': np.asarray(best_metric_index)}, path_best_metric, **{'dtype': int})


def get_path_src_metric(name_srcs, path_src_metrics, name_get_src):
    """Get the name the path with the keyword.
    Args:
        name_srcs (list[str]): names of the keywords for query.
            e.g. ['1_n1_50', '2_n2_50']
        path_src_metrics (list[str]): list of the paths where saved results of metrics.
            e.g. ['./metric/1_n1_50/Z_0_zero', './metric/2_n2_50/Z_0_zero'].
        name_get_src (str): name of the keyword to get.
            e.g. '1_n1_50'.
    Returns:
        path_src_metric_get (str): path with the keyword.
    """
    for name_src_i, path_src_metric_i in zip(name_srcs, path_src_metrics):
        if name_get_src == name_src_i:
            path_src_metric_get = path_src_metric_i
            break
    return path_src_metric_get


def compute_all_metrics_zp_nzt(path_src_metrics, name_query_metrics, name_set, name_channels, name_sms,
                               path_best_src_metric, name_best_metric):
    """After finding out best paras of the samples, compute the other metrics
        between zero-channel output and non-zero channel true target.
    Args:
        path_src_metrics (list[str]): list of the paths where saved results of metrics.
            e.g. ['./metric/1_n1_50/Z_0_zero', './metric/2_n2_50/Z_0_zero'].
        name_query_metrics (list[str]): name of the metric to compute. e.g. ['sr'].
        name_set (str): name of the dataset.  e.g. 'train', 'val', 'test'.
        name_channels (list[str]): names of the channels. e.g. ['Z_0_zero', 'Z_1_zero', 'Z_2_zero', 'Z_3_zero'].
        name_sms (list[str]): names of the channels containing multiple targets.
            e.g. ['Z_1_zero', 'Z_2_zero'].
        path_best_src_metric (str): path of the directory where to save metrics with the best parameters.
            e.g. '/metric/best_merge/Z_0_zero/best_mse'.
        name_best_metric (str): name of the best metric.  e.g. 'mse'.
    """
    name_szs = [_ for _ in name_channels]  # pylint:disable=unnecessary-comprehension
    for name_sm_i in name_sms:
        name_szs.remove(name_sm_i)
    name_paras = [os.path.basename(Path(path_metric_i).parents[0]) for path_metric_i in path_src_metrics]
    best_metric_para = read_data(path_best_src_metric, f'{name_best_metric}_{name_set}_para',
                                 dict_key=f'{name_best_metric}_{name_set}_para')
    for name_query_metric in name_query_metrics:
        for name_sz_k in name_szs:  # channel k
            for name_sm_l in name_sms:  # target l
                query_metrics_l = []
                for j, best_metric_para_j in enumerate(best_metric_para):  # sample j
                    path_query_metric_j = get_path_src_metric(name_paras, path_src_metrics, best_metric_para_j)
                    path_query_metric_j_l = os.path.join(path_query_metric_j, name_sz_k, name_sm_l)
                    query_metric_j_l = read_data(path_query_metric_j_l,
                                                 name_query_metric,
                                                 dict_key=name_query_metric)[name_set][j]
                    query_metrics_l.append(query_metric_j_l)
                path_best_src_metric_l = os.path.join(path_best_src_metric, name_sz_k, name_sm_l)
                mkdir(path_best_src_metric_l)
                save_metric(path_best_src_metric_l, f'{name_query_metric}_{name_set}',
                            [f'{name_query_metric}_mean'],
                            [np.mean(query_metrics_l)],
                            {name_query_metric: {name_set: np.asarray(query_metrics_l)}})
                display_metric([query_metrics_l], path_best_src_metric_l, [name_set],
                               # hist_bins=(0, 1e-1, 10),
                               save_name=f'{name_query_metric}_{name_set}')


def compute_all_metrics_st(path_src_metrics, name_query_metrics, name_set,
                           path_best_src_metric, path_best_src_metric_sub, name_best_metric):
    """After finding out best paras of the samples containing single target, compute the other metrics.
    Args:
        path_src_metrics (list[str]): list of the paths where saved results of metrics.
            e.g. ['./metric/1_n1_50/Z_0_zero/Z_0_zero', './metric/2_n2_50/Z_0_zero/Z_0_zero'].
        name_query_metrics (list[str]): name of the metric to compute. e.g. ['sr'].
        name_set (str): name of the dataset.  e.g. 'train', 'val', 'test'.
        path_best_src_metric (str): path of the directory where to save metrics with the best parameters.
            e.g. '/metric/best_merge/Z_0_zero/best_mse'.
        path_best_src_metric_sub (str): path of the directory where to save metrics of the none-zero target
            with the best parameters. e.g. '/metric/best_merge/Z_0_zero/best_mse/Z_0_zero'.
        name_best_metric (str): name of the best metric.  e.g. 'mse'.
    """
    name_paras = [os.path.basename(Path(path_metric_i).parents[1]) for path_metric_i in path_src_metrics]
    best_metric_para = read_data(path_best_src_metric, f'{name_best_metric}_{name_set}_para',
                                 dict_key=f'{name_best_metric}_{name_set}_para')
    for name_query_metric in name_query_metrics:
        query_metrics = []
        for j, best_metric_para_j in enumerate(best_metric_para):
            path_query_metric_j = get_path_src_metric(name_paras, path_src_metrics, best_metric_para_j)
            query_metric_j = read_data(path_query_metric_j, name_query_metric, dict_key=name_query_metric)[name_set][j]
            query_metrics.append(query_metric_j)
        save_metric(path_best_src_metric_sub, f'{name_query_metric}_{name_set}',
                    [f'{name_query_metric}_mean'],
                    [np.mean(query_metrics)],
                    {name_query_metric: {name_set: np.asarray(query_metrics)}})
        display_metric([query_metrics], path_best_src_metric_sub, [name_set],
                       # hist_bins=(0, 1e-1, 10),
                       save_name=f'{name_query_metric}_{name_set}')


def compute_all_metrics_mt(name_subs, name_subs_srcs_st, path_src_metrics, name_query_metrics, name_set,
                           path_best_src_metric, name_best_metric):
    """After finding out best paras of the samples containing multitargets, compute the other metrics.
    Args:
        name_subs (list[str]): name of the targets in multitarget signal. e.g. ['Z_1_zero', 'Z_2_zero'].
        name_subs_srcs_st (list[str]): name of the all targets. e.g. ['Z_0_zero', 'Z_1_zero', 'Z_2_zero', 'Z_3_zero'].
        path_src_metrics (list[str]):  list of the paths where saved results of metrics.
            e.g. ['./metric/1_n1_50/Z_4_zero', './metric/2_n2_50/Z_4_zero'].
        name_query_metrics (list[str]): name of the metric to compute. e.g. ['mse', 'sr', 'sdr', 'si_sdr'].
        name_set (str): name of the dataset.  e.g. 'train', 'val', 'test'.
        path_best_src_metric (str): path of the directory where to save metrics with the best parameters.
            e.g. '/metric/best_merge/Z_4_zero/best_mse'.
        name_best_metric (str): name of the best metric.  e.g. 'mse'.
    """
    name_paras = [os.path.basename(Path(path_metric_i).parents[0]) for path_metric_i in path_src_metrics]
    best_metric_para = read_data(path_best_src_metric, f'{name_best_metric}_{name_set}_para',
                                 dict_key=f'{name_best_metric}_{name_set}_para')
    for name_query_metric in name_query_metrics:
        if name_query_metric in ['sdr', 'si_sdr', 'sd_sdr']:
            query_metrics_srcs = [[] for i in range(len(name_subs))]  # [n_sub][n_samples]
            for j, best_metric_para_j in enumerate(best_metric_para):  # sample j
                path_query_metric_j = get_path_src_metric(name_paras, path_src_metrics, best_metric_para_j)
                for i, name_sub_i in enumerate(name_subs):
                    query_metric_j_i = read_data(path_query_metric_j, f'{name_query_metric}_{name_set}',
                                                 dict_key=f'{name_query_metric}_{name_set}')[name_sub_i][j]
                    query_metrics_srcs[i].append(query_metric_j_i)
            save_metric(path_best_src_metric, f'{name_query_metric}_{name_set}',
                        [f'{name_query_metric}_{name_set}_mean'],
                        [np.mean(query_metrics_srcs, axis=-1)],
                        {f'{name_query_metric}_{name_set}': dict(zip(name_subs, query_metrics_srcs))})
            display_metric(np.asarray(query_metrics_srcs), path_best_src_metric,
                           name_subs,
                           # hist_bins=(0, 1e-1, 10),
                           save_name=f'{name_query_metric}_{name_set}')

        elif name_query_metric in ['mse']:  # including zero channel
            query_metrics_srcs = [[] for i in range(len(name_subs_srcs_st))]  # [n_sub][n_samples]
            for j, best_metric_para_j in enumerate(best_metric_para):  # sample j
                path_query_metric_j = get_path_src_metric(name_paras, path_src_metrics, best_metric_para_j)
                for i, name_sub_i in enumerate(name_subs_srcs_st):
                    query_metric_j_i = read_data(os.path.join(path_query_metric_j, name_sub_i),
                                                 name_query_metric,
                                                 dict_key=name_query_metric)[name_set][j]
                    query_metrics_srcs[i].append(query_metric_j_i)
            for name_sub_i, query_metrics_src_i in zip(name_subs_srcs_st, query_metrics_srcs):
                path_best_src_metric_i = os.path.join(path_best_src_metric, name_sub_i)
                mkdir(path_best_src_metric_i)
                save_metric(path_best_src_metric_i, f'{name_query_metric}_{name_set}',
                            [f'{name_query_metric}_mean'],
                            [np.mean(query_metrics_src_i)],
                            {name_query_metric: {name_set: query_metrics_src_i}})
                display_metric([query_metrics_src_i], path_best_src_metric_i, [name_set],
                               # hist_bins=(0, 1e-1, 10),
                               save_name=f'{name_query_metric}_{name_set}')

        elif name_query_metric in ['sr', 'si_snr']:  # excluding zero channel
            query_metrics_srcs = [[] for i in range(len(name_subs))]  # [n_sub][n_samples]
            for j, best_metric_para_j in enumerate(best_metric_para):  # sample j
                path_query_metric_j = get_path_src_metric(name_paras, path_src_metrics, best_metric_para_j)
                for i, name_sub_i in enumerate(name_subs):
                    query_metric_j_i = read_data(os.path.join(path_query_metric_j, name_sub_i),
                                                 name_query_metric,
                                                 dict_key=name_query_metric)[name_set][j]
                    query_metrics_srcs[i].append(query_metric_j_i)
            for name_sub_i, query_metrics_src_i in zip(name_subs, query_metrics_srcs):
                path_best_src_metric_i = os.path.join(path_best_src_metric, name_sub_i)
                mkdir(path_best_src_metric_i)
                save_metric(path_best_src_metric_i, f'{name_query_metric}_{name_set}',
                            [f'{name_query_metric}_mean'],
                            [np.mean(query_metrics_src_i)],
                            {name_query_metric: {name_set: query_metrics_src_i}})
                display_metric([query_metrics_src_i], path_best_src_metric_i, [name_set],
                               # hist_bins=(0, 1e-1, 10),
                               save_name=f'{name_query_metric}_{name_set}')


def compute_best_metrics(path_root, name_set, names_metric_st, names_metric_mt, names_st, names_mt, names_mt_sub,
                         name_query_metrics_mt, name_query_metrics_zp_nzt,
                         kw_dir='metric', kw_para=None, kw_best='best_merge',
                         ):
    """Compute the best metric with which paras.
    Args:
        path_root (str): root path where saved metrics.
        name_set (str): name of the dataset.
        names_metric_st (list[str]): names of the metrics of sources containing single target to compute.
            e.g. ['mse', 'sr'].
        names_metric_mt (list[str]): names of the metrics of sources containing multiple targets to compute.
            e.g. ['mse', 'sr', 'sdr', 'si_sdr'].
        names_st (list[str]): names of the sources containing single target.
            e.g. ['Z_0_zero', 'Z_1_zero', 'Z_2_zero', 'Z_3_zero'].
        names_mt (list[str]): names of the sources containing multiple targets.
            e.g. ['Z_4_zero', 'Z_5_zero', 'Z_6_zero', 'Z_7_zero'].
        names_mt_sub (str): name of the targets of sources containing multiple targets.
            e.g. [['Z_1_zero', 'Z_2_zero'], ['Z_1_zero', 'Z_3_zero'], ['Z_2_zero', 'Z_3_zero'],
                  ['Z_1_zero', 'Z_2_zero', 'Z_3_zero']].
        name_query_metrics_mt (list[str]): names of the metrics of sources containing multiple targets to query.
            e.g. ['sr', 'mse', 'si_snr', 'sdr'].
        name_query_metrics_zp_nzt (list[str]): names of the metrics between zero-channel output and non-zero channel
             true target.
            e.g. ['sr', 'mse', 'si_snr', 'sdr'].
        kw_dir (str, optional): name of the metric directory. Defaults to 'metric'.
        kw_para (str, optional): keyword of the paras directory of metric . Defaults to None.
        kw_best (str, optional): name of the best metric directory. Defaults to 'best_merge'.
    Raises:
        ParameterError: check name_metric_st, raise: f'Unsupport metric {name_metric_st}'.
        ParameterError: check name_metric_mt, raise: f'Unsupport metric {name_metric_mt}'.
    """
    # compute metrics save index and metrics
    path_metric = os.path.join(path_root, kw_dir)
    path_best = os.path.join(path_metric, kw_best)
    if kw_para is None:
        path_dir_paras = list_dirs(path_metric, full=True)
    else:
        path_dir_paras = list_dirs_start_str(path_metric, kw_para, full=True)
    if os.path.isdir(path_best):
        for path_dir_para_i in path_dir_paras:
            if os.path.basename(path_dir_para_i) == kw_best:
                path_dir_paras.remove(path_dir_para_i)
                break
    else:
        mkdir(path_best)

    def get_path_dir_metrics_subs(path_dir_metrics_src, names_src_subs):
        """Get sub directories of the source.
        Args:
            path_dir_metrics_src (list[str]): path of the sources with different parameters.
                e.g. ['./metric/1_n1_50/Z_4_zero', './metric/2_n2_50/Z_4_zero'].
            names_src_subs (list[str]): name of the channels.
                e.g. ['Z_0_zero', 'Z_1_zero', 'Z_2_zero', 'Z_3_zero'].
        Returns:
        path_dir_metrics_subs (list[list[str]]): 2D list of the paths where saved results of metrics of the sources.
            e.g. [['./metric/1_n1_50/Z_4_zero/Z_1_zero', './metric/1_n1_50/Z_4_zero/Z_2_zero'],
                  ['./metric/2_n2_50/Z_4_zero/Z_1_zero', './metric/2_n2_50/Z_4_zero/Z_2_zero']].
        """
        path_dir_metrics_subs = []  # [para][subs]
        for path_dir_metric_src in path_dir_metrics_src:  # str in [para]
            path_dir_metrics_subs_i = []  # [subs]
            for name_src_sub in names_src_subs:
                path_dir_metrics_subs_i.append(os.path.join(path_dir_metric_src, name_src_sub))
            path_dir_metrics_subs.append(path_dir_metrics_subs_i)
        return path_dir_metrics_subs

    def get_path_best_src_metric_subs(path_best_src_metric, names_src_subs):
        """Get sub directories of the merged best result directory.
        Args:
            path_best_src_metric (str): path of the directory where to save metrics with the best parameters.
                e.g. '/metric/best_merge/Z_4_zero/best_mse'.
            names_src_subs (list[str]): name of the channels.
                e.g. ['Z_0_zero', 'Z_1_zero', 'Z_2_zero', 'Z_3_zero'].
        Returns:
        path_dir_metrics_subs (list[list[str]]): 2D list of the paths where saved results of metrics of the sources.
            e.g. [['./metric/1_n1_50/Z_4_zero/Z_1_zero', './metric/1_n1_50/Z_4_zero/Z_2_zero'],
                  ['./metric/2_n2_50/Z_4_zero/Z_1_zero', './metric/2_n2_50/Z_4_zero/Z_2_zero']].
        """
        path_best_src_metric_subs = []
        for name_src_sub in names_src_subs:
            path_best_metric_sub_i = os.path.join(path_best_src_metric, name_src_sub)
            mkdir(path_best_metric_sub_i)
            path_best_src_metric_subs.append(path_best_metric_sub_i)
        return path_best_src_metric_subs

    # single target
    for name_src in names_st:
        path_dir_metrics_src = [os.path.join(path_dir_para, name_src) for path_dir_para in path_dir_paras]  # [para]
        path_dir_metrics_subs = get_path_dir_metrics_subs(path_dir_metrics_src, names_st)  # [para][subs]
        path_dir_metrics_subs_t = list_transpose(path_dir_metrics_subs)  # [subs][para]

        path_dir_metrics_src_st = [os.path.join(path_dir_para, name_src, name_src)
                                   for path_dir_para in path_dir_paras]  # [para]

        path_best_src = os.path.join(path_best, name_src)
        mkdir(path_best_src)

        for name_metric_st in names_metric_st:
            path_best_src_metric = os.path.join(path_best_src, f'best_{name_metric_st}')
            mkdir(path_best_src_metric)
            path_best_src_metric_subs = get_path_best_src_metric_subs(path_best_src_metric, names_st)  # [subs]
            path_best_src_metric_subs_st = os.path.join(path_best_src_metric, name_src)

            if name_metric_st == 'mse':  # smaller better
                # # only compute channels with targets
                # compute_best_metric_st(path_dir_metrics_src_st, path_best_src_metric, name_metric_st, name_set,
                #                        better='small')
                # compute all channels
                compute_best_metric_mt(path_dir_metrics_src, path_dir_metrics_subs,
                                       path_best_src_metric, name_metric_st, name_set,
                                       better='small')
            elif name_metric_st in ['sr', 'si_snr']:  # bigger better
                # # only compute channels with targets
                compute_best_metric_st(path_dir_metrics_src_st, path_best_src_metric, name_metric_st, name_set,
                                       better='big')
            else:
                raise ParameterError(f'Unsupport metric {name_metric_st}')

            compute_all_metrics_st(path_dir_metrics_src_st, ['sr', 'si_snr'], name_set,
                                   path_best_src_metric, path_best_src_metric_subs_st, name_metric_st)
            for path_dir_metrics_subs_t_i, path_best_src_metric_sub_i in zip(  # level channel
                    path_dir_metrics_subs_t, path_best_src_metric_subs):  # level mix input type
                compute_all_metrics_st(path_dir_metrics_subs_t_i, ['mse'], name_set,
                                       path_best_src_metric, path_best_src_metric_sub_i, name_metric_st)
            compute_all_metrics_zp_nzt(path_dir_metrics_src, name_query_metrics_zp_nzt, name_set,
                                       names_st, [name_src], path_best_src_metric, name_metric_st)

            del path_best_src_metric
            del path_best_src_metric_subs
            del path_best_src_metric_subs_st

        del path_dir_metrics_src
        del path_dir_metrics_subs
        del path_dir_metrics_subs_t
        del path_dir_metrics_src_st
        del path_best_src

    # multiple target
    for name_src, name_src_subs in zip(names_mt, names_mt_sub):  # str; [subs] in [src]; [src][subs]
        path_dir_metrics_src = [os.path.join(path_dir_para, name_src) for path_dir_para in path_dir_paras]  # [para]
        path_dir_metrics_subs = get_path_dir_metrics_subs(path_dir_metrics_src, names_st)  # [para][subs]
        path_dir_metrics_subs_mt = get_path_dir_metrics_subs(path_dir_metrics_src, name_src_subs)  # [para][subs]

        path_best_src = os.path.join(path_best, name_src)
        mkdir(path_best_src)

        for name_metric_mt in names_metric_mt:
            path_best_src_metric = os.path.join(path_best_src, f'best_{name_metric_mt}')
            mkdir(path_best_src_metric)

            # path_best_src_metric_subs = get_path_best_src_metric_subs(path_best_src_metric, names_st)  # [subs]
            # path_best_src_metric_subs_mt = get_path_best_src_metric_subs(path_best_src_metric, name_src_subs) # [subs]

            if name_metric_mt in ['mse']:  # smaller better
                # # only compute channels with targets
                # compute_best_metric_mt(path_dir_metrics_src, path_dir_metrics_subs_mt,
                #                        path_best_src_metric, name_metric_mt, name_set,
                #                        better='small')
                # compute all channels
                compute_best_metric_mt(path_dir_metrics_src, path_dir_metrics_subs,
                                       path_best_src_metric, name_metric_mt, name_set,
                                       better='small')
            elif name_metric_mt in ['sr', 'sdr', 'si_sdr', 'sd_sdr']:  # bigger better
                compute_best_metric_mt(path_dir_metrics_src, path_dir_metrics_subs_mt,
                                       path_best_src_metric, name_metric_mt, name_set,
                                       better='big')
            else:
                raise ParameterError(f'Unsupport metric {name_metric_mt}')
            compute_all_metrics_mt(name_src_subs, names_st,
                                   path_dir_metrics_src, name_query_metrics_mt, name_set,
                                   path_best_src_metric, name_metric_mt)
            compute_all_metrics_zp_nzt(path_dir_metrics_src, name_query_metrics_zp_nzt, name_set,
                                       names_st, name_src_subs, path_best_src_metric, name_metric_mt)

            del path_best_src_metric

        del path_dir_metrics_src
        del path_dir_metrics_subs
        del path_dir_metrics_subs_mt
        del path_best_src


if __name__ == '__main__':
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep
    from recognition_mix_shipsear_s0tos3_preprocess import n_hot_labels

    # for shipsear data
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    SCALER_DATA = 'max_one'
    # SCALER_DATA = 'or'
    SUB_SET_WAY = 'rand'
    # SUB_SET_WAY = 'order'

    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav', scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source

    SET_NAMES = ['train', 'val', 'test'][2:3]
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
    SM_SRC_NAMES = [[f'Z_{sm_j}_zero' for sm_j in sm_i] for sm_i in SM_INDEX]

    PATH_SAVE_ROOT = '../result_separation_ae_ns_single'

    # # Concatenate result files, only use this method when train samples by batches.
    # concat_results(num_list=[1000, 2000, 3000, 4000, 5000, 6000, 7000, 8160],
    #                 path_root=PATH_SAVE_ROOT+'/model_8_2_1/search_decoder/1_n3_50/test_sets_1_n1_50',
    #                 dir_para_2='1_n1_50', dir_para_1='predict_decoder', name_set='test')

    object_decoded_files_ = ListDecodedFiles(PATH_SAVE_ROOT)

    compute_metrics(object_decoded_files_, Z_NAMES, Z_SRC_NAMES, Z_SRC_NAMES[:4], Z_DATA, SM_INDEX, SET_NAMES,
                    'decoded', Y_DATA, LABELS_N_HOT)

    for name_set_j in SET_NAMES:
        # compute_best_metrics(os.path.join(PATH_SAVE_ROOT, 'model_8_2_1', 'search_decoder', '1_n3_50'), name_set_j,
        #                      ['mse', 'sr'], [], Z_SRC_NAMES[:4], Z_SRC_NAMES[4:], SM_SRC_NAMES[4:],
        #                      [], ['sr'], kw_dir='metric', kw_para=None, kw_best='best_merge')

        # compute_best_metrics(os.path.join(PATH_SAVE_ROOT, 'model_8_2_1', 'search_decoder', '1_n3_50'), name_set_j,
        #                      [], ['mse', 'sdr'], Z_SRC_NAMES[:4], Z_SRC_NAMES[4:], SM_SRC_NAMES[4:],
        #                      ['mse', 'sdr', 'si_sdr'], ['sr'], kw_dir='metric', kw_para=None, kw_best='best_merge')

        compute_best_metrics(os.path.join(PATH_SAVE_ROOT, 'model_8_2_1', 'search_decoder', '1_n3_50'), name_set_j,
                             ['mse', 'sr'], ['mse', 'sr', 'sdr', 'si_sdr'],
                             Z_SRC_NAMES[:4], Z_SRC_NAMES[4:], SM_SRC_NAMES[4:],
                             ['mse', 'sr', 'si_snr', 'sdr', 'si_sdr'], ['sr', 'si_snr'],
                             kw_dir='metric', kw_para=None, kw_best='best_merge')

    logging.info('finished')
