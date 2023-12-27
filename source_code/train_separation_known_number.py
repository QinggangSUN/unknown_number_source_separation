# -*- coding: utf-8 -*-
"""

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation
# pylint: disable=too-many-arguments, too-many-branches, too-many-lines, too-many-locals, too-many-statements

import numpy as np

from train_separation_multiple_autoencoder import index_mix_src_ns


if __name__ == '__main__':
    import logging
    import os

    import tensorflow
    if tensorflow.__version__ >= '2.0':
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        from tensorflow import keras
        # import keras
        import tensorflow.compat.v1.keras.backend as K
    else:
        import tensorflow as tf
        import keras
        from keras import backend as K

    from file_operation import mkdir
    from prepare_data_shipsear_recognition_mix_s0tos3 import read_datas
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep
    from train_single_source_autoencoder_ns import search_model

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
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
    SUB_SET_WAY = 'rand'

    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav', scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source

    X_DICT = [dict(zip(name_i, read_datas(PATH_DATA, name_i))) for name_i in X_NAMES]
    Z_DICT = [[dict(zip(name_i_j, read_datas(PATH_DATA, name_i_j))) for name_i_j in name_i] for name_i in Z_NAMES]

    PATH_RESULT = '../result_separation_known_number'
    mkdir(PATH_RESULT)

    lr_i_j = [(i, j) for i in range(1, 2) for j in range(-3, -4, -1)]
    for (lr_i, lr_j) in lr_i_j:
        for x_name_i, x_dict_i, z_name_i, z_dict_i in zip(X_NAMES, X_DICT, Z_NAMES, Z_DICT):
            logging.info(x_dict_i.keys())

            path_result_i = os.path.join(PATH_RESULT, x_name_i[0][:-9])
            mkdir(path_result_i)

            input_dim = x_dict_i[x_name_i[-1]].shape[-1]  # (nsamples, 1, frame_length)
            logging.debug(f'input_dim {input_dim}')

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.squeeze(value)  # (nsamples, frame_length)

            n_outputs_i = len(z_name_i)

            z_list_set_i = []
            for j_set, name_set_j in enumerate(x_name_i):
                z_data_set_ij = []
                for k_src, z_dict_i_k in enumerate(z_dict_i):
                    # (nsamples, frame_length, 1)
                    z_name_set_j = list(z_dict_i_k.keys())[j_set]
                    z_data_set_ij.append(np.expand_dims(np.squeeze(z_dict_i_k[z_name_set_j]), axis=-1))
                z_data_set_arr_ij = np.concatenate(z_data_set_ij, axis=-1)  # (nsamples, frame_length, n_outputs_i)
                z_list_set_i.append(z_data_set_arr_ij)
            z_dict_set_i = dict(zip(x_name_i, z_list_set_i))

            # search_model(path_result_i, 'model_1_6_3', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'encoding_dim': 256, 'n_nodes': [16, 8, 8, 8],
            #                 'layer_norm': True, 'use_bias': False,
            #                 'n_outputs': n_outputs_i,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (nsamples, frame_length, 1)
                logging.debug(f'x_i.name {key}')
                logging.debug(f'x_i.shape {x_dict_i[key].shape}')

            # search_model(path_result_i, 'model_8_2_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'batch_size': 8, 'bs_pred': 8,
            #                 'n_conv_encoder': 1, 'n_filters': 64,
            #                 'rnn_type': 'BLSTM', 'latent_dim': 200,
            #                 'n_outputs': n_outputs_i,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})
            #
            # search_model(path_result_i, 'model_8_3_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                 'n_conv_encoder': 1, 'n_filters': 64,
            #                 'rnn_type': 'dprnn', 'latent_dim': 200,
            #                 'n_outputs': n_outputs_i,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False,
            #                 'bool_num_padd': True})

            # model_10 RNN TasNet
            search_model(path_result_i, 'model_10_1_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 50, 'batch_size': 8, 'bs_pred': 8,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'LSTM', 'latent_dim': 200,
                            'n_block_encoder': 1,  'n_block_decoder': 1,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            search_model(path_result_i, 'model_10_2_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'BLSTM', 'latent_dim': 200,
                            'n_block_encoder': 1,  'n_block_decoder': 1,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            search_model(path_result_i, 'model_10_3_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                             'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                             'n_conv_encoder': 1, 'n_filters_conv': 64,
                             'block_type': 'dprnn', 'latent_dim': 200,
                             'n_block_encoder': 1,  'n_block_decoder': 1,
                             'bool_train': True, 'bool_test_ae': False,
                             'bool_save_ed': False, 'bool_test_ed': False,
                             'bool_clean_weight_file': True, 'bool_test_weight': True,
                             'bool_test_ae_w': True, 'bool_test_ed_w': False,
                             'bool_num_padd': True})

            # # model_11 multiple decoder RNN TasNet
            # search_model(path_result_i, 'model_11_1_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
            #                 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                 'n_conv_encoder': 1, 'n_filters_conv': 64,
            #                 'block_type': 'LSTM', 'latent_dim': 200,
            #                 'n_block_encoder': 1,  'n_block_decoder': 1,
            #                 'is_multiple_decoder': True,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # search_model(path_result_i, 'model_11_2_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
            #                 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                 'n_conv_encoder': 1, 'n_filters_conv': 64,
            #                 'block_type': 'BLSTM', 'latent_dim': 200,
            #                 'n_block_encoder': 1,  'n_block_decoder': 1,
            #                 'is_multiple_decoder': True,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # search_model(path_result_i, 'model_11_3_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
            #                 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                 'n_conv_encoder': 1, 'n_filters_conv': 64,
            #                 'block_type': 'dprnn', 'latent_dim': 200,
            #                 'n_block_encoder': 1,  'n_block_decoder': 1,
            #                 'is_multiple_decoder': True,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False,
            #                 'bool_num_padd': True})

            # # model_12 RNN TasNet without mask
            # search_model(path_result_i, 'model_12_1_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
            #                 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                 'n_conv_encoder': 1, 'n_filters_conv': 64,
            #                 'block_type': 'LSTM', 'latent_dim': 200,
            #                 'n_block_encoder': 1, 'n_block_decoder': 1,
            #                 'use_mask': False,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # search_model(path_result_i, 'model_12_2_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
            #                 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                 'n_conv_encoder': 1, 'n_filters_conv': 64,
            #                 'block_type': 'BLSTM', 'latent_dim': 200,
            #                 'n_block_encoder': 1, 'n_block_decoder': 1,
            #                 'use_mask': False,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # search_model(path_result_i, 'model_12_3_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
            #                 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                 'n_conv_encoder': 1, 'n_filters_conv': 64,
            #                 'block_type': 'dprnn', 'latent_dim': 200,
            #                 'use_mask': False,
            #                 'n_block_encoder': 1, 'n_block_decoder': 1,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False,
            #                 'bool_num_padd': True})

            # model_13 multiple decoder RNN TasNet without mask
            search_model(path_result_i, 'model_13_1_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'LSTM', 'latent_dim': 200,
                            'n_block_encoder': 1, 'n_block_decoder': 1,
                            'is_multiple_decoder': True, 'use_mask': False,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            search_model(path_result_i, 'model_13_2_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'BLSTM', 'latent_dim': 200,
                            'n_block_encoder': 1, 'n_block_decoder': 1,
                            'is_multiple_decoder': True, 'use_mask': False,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            search_model(path_result_i, 'model_13_3_1', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'dprnn', 'latent_dim': 200,
                            'n_block_encoder': 1, 'n_block_decoder': 1,
                            'is_multiple_decoder': True, 'use_mask': False,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False,
                            'bool_num_padd': True})

            # Conv-Tasnet
            search_model(path_result_i, 'model_14_1_6', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 200, 'batch_size': 6, 'bs_pred': 6,
                            'n_conv_encoder': 1, 'n_filters_encoder': 64,
                            'n_channels_conv': 128, 'n_channels_bottleneck': 64, 'n_channels_skip': 64,
                            'n_layer_each_block': 5, 'n_block_encoder': 3, 'n_block_decoder': 0,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # Conv-Tasnet without mask
            # search_model(path_result_i, 'model_14_2_6', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
            #                 'epochs': 200, 'batch_size': 6, 'bs_pred': 6,
            #                 'use_mask': False,
            #                 'n_conv_encoder': 1, 'n_filters_encoder': 64,
            #                 'n_channels_conv': 128, 'n_channels_bottleneck': 64, 'n_channels_skip': 64,
            #                 'n_layer_each_block': 5, 'n_block_encoder': 3, 'n_block_decoder': 0,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # Multiple-Decoder Conv-Tasnet
            # search_model(path_result_i, 'model_15_1_6', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
            #                 'epochs': 200, 'batch_size': 6, 'bs_pred': 6,
            #                 'is_multiple_decoder': True,
            #                 'n_conv_encoder': 1, 'n_filters_encoder': 64,
            #                 'n_channels_conv': 128, 'n_channels_bottleneck': 64, 'n_channels_skip': 64,
            #                 'n_layer_each_block': 5, 'n_block_encoder': 1, 'n_block_decoder': 2,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # Multiple-Decoder Conv-Tasnet without mask
            search_model(path_result_i, 'model_15_2_6', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 200, 'batch_size': 6, 'bs_pred': 6,
                            'is_multiple_decoder': True, 'use_mask': False,
                            'n_conv_encoder': 1, 'n_filters_encoder': 64,
                            'n_channels_conv': 128, 'n_channels_bottleneck': 64, 'n_channels_skip': 64,
                            'n_layer_each_block': 5, 'n_block_encoder': 1, 'n_block_decoder': 2,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # Wave-U-Net
            search_model(path_result_i, 'model_20_5_10', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
                         **{'i': lr_i, 'j': -4, 'n_outputs': n_outputs_i,
                            'epochs': 400, 'batch_size': 16, 'bs_pred': 16,
                            'n_pad_input': 13, 'num_layers': 4, 'batch_norm': True,
                            'output_type': 'direct', 'output_activation': 'tanh',
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # # Wave-U-Net without skip connections
            # search_model(path_result_i, 'model_20_6_10', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
            #                 'epochs': 600, 'batch_size': 16, 'bs_pred': 16,
            #                 'n_pad_input': 13, 'num_layers': 4, 'batch_norm': True,
            #                 'use_skip': False, 'output_type': 'direct', 'output_activation': 'tanh',
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # # Multiple-Decoder Wave-U-Net
            # search_model(path_result_i, 'model_21_5_10', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
            #              **{'i': lr_i, 'j': -4, 'n_outputs': n_outputs_i,
            #                 'epochs': 800, 'batch_size': 16, 'bs_pred': 16,
            #                 'n_pad_input': 13, 'num_layers': 4, 'batch_norm': True,
            #                 'output_type': 'direct', 'output_activation': 'tanh',
            #                 'is_multiple_decoder': True,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': False, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                 'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # Multiple-Decoder Wave-U-Net without skip connections
            search_model(path_result_i, 'model_21_6_10', input_dim, x_dict_i, z_dict_set_i, x_name_i[0][:-6],
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 800, 'batch_size': 16, 'bs_pred': 16,
                            'n_pad_input': 13, 'num_layers': 4,
                            'use_skip': False, 'output_type': 'direct', 'output_activation': 'tanh',
                            'is_multiple_decoder': True,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

    logging.info('finished')
