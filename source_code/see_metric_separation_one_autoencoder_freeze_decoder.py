# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 19:38:03 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long


if __name__ == '__main__':
    import logging
    from prepare_data_shipsear_recognition_mix_s0tos3 import read_data, read_datas
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep
    from recognition_mix_shipsear_s0tos3_preprocess import n_hot_labels
    from see_metric_single_source_ae_ns import ListDecodedFiles
    from see_metric_separation_one_autoencoder import compute_metrics

    # for shipsear data
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    SCALER_DATA = 'max_one'
    # SCALER_DATA = 'or'
    SUB_SET_WAY = 'rand'
    # SUB_SET_WAY = 'order'

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

    PATH_SAVE_ROOT = '../result_separation_ae_ns_single/model_8_2_1/load_decoder'

    object_decoded_files = ListDecodedFiles(PATH_SAVE_ROOT, Z_SRC_NAME)

    compute_metrics(object_decoded_files, Z_NAMES, Z_SRC_NAMES, Z_SRC_NAMES[:4],
                    Z_DATA, SM_INDEX, SET_NAMES, 'autodecoded', Y_DATA, LABELS_N_HOT)

    logging.info('finished')
