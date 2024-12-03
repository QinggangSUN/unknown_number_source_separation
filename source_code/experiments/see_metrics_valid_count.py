# -*- coding: utf-8 -*-
"""
@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation, too-few-public-methods, too-many-arguments
# pylint: disable=too-many-branches, too-many-locals, too-many-nested-blocks, too-many-statements

import os
import numpy as np

from prepare_data_shipsear_recognition_mix_s0tos3 import read_data, save_datas


def compute_num_active_matrix(path_result, f_name_num_active_refs, f_name_num_active_seps, set_name, num_channels):
    """extract outputs from keras predict file.
    Args:
        path_result (str): path where result file saved.
        f_name_num_active_refs (str): file name of the num_active_refs file key word.
        f_name_num_active_seps (str): file name of the num_active_seps file key word.
        set_name (str): name of set, e.g. 'train', 'val', 'test'.
        num_channels (int): number of channels of networks.
    Returns:
        matrix (np.ndarray): (num_channels + 1, num_channels + 1) confusion matrix.
    """
    num_active_refs = read_data(path_result, f_name_num_active_refs, dict_key=f_name_num_active_refs)[set_name]
    num_active_seps = read_data(path_result, f_name_num_active_seps, dict_key=f_name_num_active_seps)[set_name]

    matrix = np.zeros((num_channels + 1, num_channels + 1), dtype=np.int32)
    for num_active_ref, num_active_sep in zip(num_active_refs, num_active_seps):
        num_active_ref, num_active_sep = int(num_active_ref), int(num_active_sep)
        matrix[num_active_ref][num_active_sep] = matrix[num_active_ref][num_active_sep] + 1

    return matrix


def compute_num_active_matrices(object_decoded_files,
                                f_name_num_active_refs='num_active_refs', f_name_num_active_seps='num_active_seps',
                                set_names=('train', 'val', 'test'), num_channels=4, **kwargs):
    """compute p-si-snr.
    Args:
        object_decoded_files (class ListDecodedFiles): object for list predict decoded files.
        f_name_num_active_refs (str, optional): file name of the num_active_refs file key word. Default to
            'num_active_refs'.
        f_name_num_active_seps (str, optional): file name of the num_active_seps file key word. Default to
            'num_active_seps'.
        set_names (tuple(str), optional): tuple of names of sets. Default to ('train', 'val', 'test').
        num_channels (int, optional): number of channels of networks. Defaults to 4.
    """
    penalty = kwargs['penalty'] if 'penalty' in kwargs.keys() else None
    reward = kwargs['reward'] if 'reward' in kwargs.keys() else 0.

    path_metrics_paras = object_decoded_files.path_metrics_paras  # [model][para_name]
    for path_metrics_para_j in path_metrics_paras:  # level j: [model]
        for path_metric_para_k in path_metrics_para_j:  # level k: [para_name]
            path_metric_result_k = os.path.join(path_metric_para_k, f'penalty_{penalty}_reward_{reward}')
            for set_name_i in set_names:
                matrix = compute_num_active_matrix(path_metric_result_k, f_name_num_active_refs, f_name_num_active_seps,
                                                   set_name_i, num_channels)
                save_datas({f'num_active_matrix_confusion_{set_name_i}': matrix}, path_metric_result_k,
                           **{'form_save': 'csv'})


if __name__ == '__main__':
    from see_metric_single_source_ae_ns import ListDecodedFiles as ListDecodedFilesOne

    Z_SRC_NAME = ['Z_zero']

    PATH_SAVE_ROOT = '../../result_separation_fuss'
    object_decoded_files_one = ListDecodedFilesOne(PATH_SAVE_ROOT, Z_SRC_NAME)
    compute_num_active_matrices(object_decoded_files_one)

    PATH_SAVE_ROOT = '../../result_separation_one_autoencoder'
    object_decoded_files_one = ListDecodedFilesOne(PATH_SAVE_ROOT, Z_SRC_NAME)
    compute_num_active_matrices(object_decoded_files_one)

    PATH_SAVE_ROOT = '../../result_separation_search_decoder'
    object_decoded_files_one = ListDecodedFilesOne(PATH_SAVE_ROOT, Z_SRC_NAME)
    compute_num_active_matrices(object_decoded_files_one, set_names=('test',))

    PATH_SAVE_ROOT = '../../result_separation_ae_ns_single'
    model_names = ('model_13_2_1', 'model_13_3_1', 'model_15_2_6', 'model_21_6_10')
    for model_name in model_names:
        path_save_model = os.path.join(PATH_SAVE_ROOT, model_name, 'load_decoder')
        object_decoded_files_one = ListDecodedFilesOne(path_save_model, Z_SRC_NAME)
        compute_num_active_matrices(object_decoded_files_one)

    PATH_SAVE_ROOT = '../../result_separation_kim'
    object_decoded_files_one = ListDecodedFilesOne(PATH_SAVE_ROOT, Z_SRC_NAME)
    compute_num_active_matrices(object_decoded_files_one)
