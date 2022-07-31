# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:34:29 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation, no-member
# pylint: disable=too-many-arguments, too-many-branches, too-many-lines, too-many-locals, too-many-statements

import numpy as np


def index_mix_src_ns(n_src):
    """Index mix sources without silences.
    Input: number of sources include silence,
    Return: index list of combinations.
    Example:
        s0 to s3 [0, 1, 2, 3]
        >>> print(index_mix_src_ns(n_src=4))
        return: [(1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    from itertools import combinations
    index_list = []
    for i in range(2, n_src):
        index_list = index_list + list(combinations(range(1, n_src), i))
    return index_list


if __name__ == '__main__':
    from keras import backend as K
    import logging
    import os
    import tensorflow as tf

    from file_operation import mkdir
    from prepare_data_shipsear_recognition_mix_s0tos3 import read_datas
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep
    from train_single_source_autoencoder_ns import search_model

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
    np.random.seed(1337)  # for reproducibility
    # The below tf.set_random_seed() will make random number generation in the
    # TensorFlow backend have a well-defined initial state. For further details,
    # see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(1234)
    # # Force TensorFlow to use single thread. Multiple threads are a potential
    # # source of non-reproducible results. For further details,
    # # see: https://stackoverflow.com/questions/42022950/
    # session_conf = tf.ConfigProto(
    #     intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)  # pylint: disable=invalid-name
    session_conf = tf.ConfigProto()

    # Limiting GPU memory growth (forbidden GPU OOM)
    session_conf.gpu_options.allow_growth = True

    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)  # pylint: disable=invalid-name
    K.set_session(sess)

    # for shipsear data
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    SET_NAMES = ['train', 'val', 'test']
    Z_INDEX = index_mix_src_ns(4)
    X_NAMES = [[f'Z_{i}_ns_{name_set_j}' for name_set_j in SET_NAMES] for i in range(4, 8)]
    Z_NAMES = [[[f'Z_{i}_ns_{name_set_j}' for name_set_j in SET_NAMES] for i in mix_i] for mix_i in Z_INDEX]

    SCALER_DATA = 'max_one'
    # SCALER_DATA = 'or'
    SUB_SET_WAY = 'rand'
    # SUB_SET_WAY = 'order'

    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav', scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source

    X_DICT = [dict(zip(name_i, read_datas(PATH_DATA, name_i))) for name_i in X_NAMES]
    Z_DICT = [[dict(zip(name_i_j, read_datas(PATH_DATA, name_i_j))) for name_i_j in name_i] for name_i in Z_NAMES]

    PATH_RESULT = '../result_separation_multiple_autoencoder'
    mkdir(PATH_RESULT)

    # lr_i_j = [(i, j) for i in range(1, 2) for j in range(-2, -5, -1)]
    # lr_i_j = [(i, j) for i in range(2, 3) for j in range(-4, -5, -1)]
    lr_i_j = [(i, j) for i in range(5, 6) for j in range(-4, -5, -1)]
    for (lr_i, lr_j) in lr_i_j:
        for x_name_i, x_dict_i, z_name_i, z_dict_i in zip(X_NAMES, X_DICT, Z_NAMES, Z_DICT):
            logging.info(x_dict_i.keys())

            path_result_i = os.path.join(PATH_RESULT, x_name_i[0][:-9])
            mkdir(path_result_i)

            input_dim = x_dict_i[x_name_i[-1]].shape[-1]  # (nsamples, 1, frame_length)
            logging.debug(f'input_dim {input_dim}')

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.squeeze(value)  # (nsamples, frame_length)

            for z_name_i_j, z_dict_i_j in zip(z_name_i, z_dict_i):
                for key, value in z_dict_i_j.items():
                    z_dict_i_j[key] = np.squeeze(value)  # (nsamples, frame_length)

                # search_model(path_result_i, 'model_1_3_3', input_dim, x_dict_i, z_dict_i_j, z_name_i_j[0][:-6],
                #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'encoding_dim': 256, 'n_nodes': [16, 8, 8, 8],
                #                 'batch_norm': True,
                #                 'bool_train': True, 'bool_test_ae': False,
                #                 'bool_save_ed': False, 'bool_test_ed': False,
                #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
                #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

                # search_model(path_result_i, 'model_1_4_3', input_dim, x_dict_i, z_dict_i_j, z_name_i_j[0][:-6],
                #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'encoding_dim': 256, 'n_nodes': [16, 8, 8, 8],
                #                 'batch_norm': True, 'use_bias': False,
                #                 'bool_train': True, 'bool_test_ae': False,
                #                 'bool_save_ed': False, 'bool_test_ed': False,
                #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
                #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

                # search_model(path_result_i, 'model_1_5_3', input_dim, x_dict_i, z_dict_i_j, z_name_i_j[0][:-6],
                #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'encoding_dim': 256, 'n_nodes': [16, 8, 8, 8],
                #                 'layer_norm': True,
                #                 'bool_train': True, 'bool_test_ae': False,
                #                 'bool_save_ed': False, 'bool_test_ed': False,
                #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
                #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

                # search_model(path_result_i, 'model_1_6_3', input_dim, x_dict_i, z_dict_i_j, z_name_i_j[0][:-6],
                #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'encoding_dim': 256, 'n_nodes': [16, 8, 8, 8],
                #                 'layer_norm': True, 'use_bias': False,
                #                 'bool_train': True, 'bool_test_ae': False,
                #                 'bool_save_ed': False, 'bool_test_ed': False,
                #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
                #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (nsamples, frame_length, 1)
            for z_name_i_j, z_dict_i_j in zip(z_name_i, z_dict_i):
                for key, value in z_dict_i_j.items():
                    z_dict_i_j[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (nsamples, frame_length)

                # search_model(path_result_i, 'model_8_1_1', input_dim, x_dict_i, z_dict_i_j, z_name_i_j[0][:-6],
                #              **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                #                  'n_conv_encoder': 1, 'n_filters': 64,
                #                  'rnn_type': 'LSTM', 'latent_dim': 256,
                #                  'bool_train': True, 'bool_test_ae': False,
                #                  'bool_save_ed': False, 'bool_test_ed': False,
                #                  'bool_clean_weight_file': True, 'bool_test_weight': True,
                #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

                search_model(path_result_i, 'model_8_2_1', input_dim, x_dict_i, z_dict_i_j, z_name_i_j[0][:-6],
                             **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                                 'n_conv_encoder': 1, 'n_filters': 64,
                                 'rnn_type': 'BLSTM', 'latent_dim': 256,
                                 'bool_train': True, 'bool_test_ae': False,
                                 'bool_save_ed': False, 'bool_test_ed': False,
                                 'bool_clean_weight_file': True, 'bool_test_weight': True,
                                'bool_test_ae_w': True, 'bool_test_ed_w': False})

                # search_model(path_result_i, 'model_8_2_2', input_dim, x_dict_i, z_dict_i_j, z_name_i_j[0][:-6],
                #               **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                #                   'n_conv_encoder': 1, 'n_filters': 64,
                #                   'rnn_type': 'BLSTM', 'latent_dim': 256,  'n_rnn_decoder': 2,
                #                   'bool_train': True, 'bool_test_ae': False,
                #                   'bool_save_ed': False, 'bool_test_ed': False,
                #                   'bool_clean_weight_file': True, 'bool_test_weight': True,
                #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

                # search_model(path_result_i, 'model_8_3_1', input_dim, x_dict_i, z_dict_i_j, z_name_i_j[0][:-6],
                #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'batch_size': 8, 'bs_pred': 8,
                #                  'n_conv_encoder': 1, 'n_filters': 64,
                #                  'rnn_type': 'dprnn', 'latent_dim': 256,
                #                  'bool_train': True, 'bool_test_ae': False,
                #                  'bool_save_ed': False, 'bool_test_ed': False,
                #                  'bool_clean_weight_file': True, 'bool_test_weight': True,
                #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

                # search_model(path_result_i, 'model_8_4_1', input_dim, x_dict_i, z_dict_i_j, z_name_i_j[0][:-6],
                #              **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                #                  'n_conv_encoder': 1, 'n_filters': 64, 'use_bias': False,
                #                  'rnn_type': 'BLSTM', 'latent_dim': 256, 'n_rnn_decoder': 1,
                #                  'bool_train': True, 'bool_test_ae': False,
                #                  'bool_save_ed': False, 'bool_test_ed': False,
                #                  'bool_clean_weight_file': True, 'bool_test_weight': True,
                #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

    logging.info('finished')
