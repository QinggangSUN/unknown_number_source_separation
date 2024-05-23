# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:05:51 2023

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""

if __name__ == '__main__':
    import logging
    import os

    import numpy as np
    import tensorflow

    if tensorflow.__version__ >= '2.0':
        import tensorflow.compat.v1 as tf

        tf.disable_v2_behavior()
        from tensorflow import keras
        import tensorflow.compat.v1.keras.backend as K
    else:
        import tensorflow as tf
        import keras
        from keras import backend as K

    from file_operation import mkdir
    from prepare_data_shipsear_recognition_mix_s0tos3 import read_datas
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep
    from train_separation_multiple_autoencoder import index_mix_src_ns
    from train_separation_one_autoencoder_freeze_decoder import search_model

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

    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    SET_NAMES = ['train', 'val', 'test']
    Z_INDEX = index_mix_src_ns(4)
    X_NAMES = [[f'Z_{i}_ns_{name_set_j}' for name_set_j in SET_NAMES] for i in range(4, 8)]
    Z_NAMES = [[[f'Z_{i}_ns_{name_set_j}' for name_set_j in SET_NAMES] for i in mix_i] for mix_i in Z_INDEX]
    Z_SRC_NAMES = [[f'Z_{i}_ns' for i in mix_i] for mix_i in Z_INDEX]

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

    X_DICT = [dict(zip(name_i, read_datas(PATH_DATA, name_i))) for name_i in X_NAMES]
    Z_DICT = [[dict(zip(name_i_j, read_datas(PATH_DATA, name_i_j))) for name_i_j in name_i] for name_i in Z_NAMES]

    PATH_MODEL = '../result_separation_ae_ns_single'

    mkdir(PATH_MODEL)

    PATH_RESULT = '../result_separation_known_number_load_decoder'
    mkdir(PATH_RESULT)

    lr_i_j = [(i, j) for i in range(1, 2) for j in range(-3, -4, -1)]
    for (lr_i, lr_j) in lr_i_j:
        for x_name_i, x_dict_i, z_src_name_i, z_dict_i in zip(X_NAMES, X_DICT, Z_SRC_NAMES, Z_DICT):
            logging.info(x_dict_i.keys())

            path_result_i = os.path.join(PATH_RESULT, x_name_i[0][:-9])
            mkdir(path_result_i)

            input_dim = x_dict_i[x_name_i[-1]].shape[-1]  # (nsamples, 1, frame_length)
            logging.debug(f'input_dim {input_dim}')

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.squeeze(value)  # (nsamples, frame_length)

            n_outputs_i = len(z_src_name_i)

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

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (nsamples, frame_length, 1)
                logging.debug(f'x_i.name {key}')
                logging.debug(f'x_i.shape {x_dict_i[key].shape}')

            # Multiple-Decoder TasNet without mask
            search_model(PATH_MODEL, 'model_13_2_1', z_src_name_i, x_dict_i, z_dict_set_i, None, input_dim,
                         bool_train=True, bool_clean_weight_file=True, bool_predict=True,
                         path_save_load_decoder=os.path.join(path_result_i, 'model_13_2_1'),
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'BLSTM', 'latent_dim': 200,
                            'n_block_encoder': 1, 'n_block_decoder': 1,
                            'use_mask': False,
                            'is_multiple_decoder': True, 'model_type': 'ae', 'encoder_multiple_out': True})

            search_model(PATH_MODEL, 'model_13_3_1', z_src_name_i, x_dict_i, z_dict_set_i, None, input_dim,
                         bool_train=True, bool_clean_weight_file=True, bool_predict=True,
                         path_save_load_decoder=os.path.join(path_result_i, 'model_13_3_1'),
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 200, 'batch_size': 8, 'bs_pred': 8,
                            'bool_num_padd': True,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'dprnn', 'latent_dim': 200,
                            'n_block_encoder': 1, 'n_block_decoder': 1,
                            'use_mask': False,
                            'is_multiple_decoder': True, 'model_type': 'ae', 'encoder_multiple_out': True})

            # Multiple-Decoder Conv-TasNet without mask
            search_model(PATH_MODEL, 'model_15_2_6', z_src_name_i, x_dict_i, z_dict_set_i, None, input_dim,
                         bool_train=True, bool_clean_weight_file=True, bool_predict=True,
                         path_save_load_decoder=os.path.join(path_result_i, 'model_15_2_6'),
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 200, 'batch_size': 6, 'bs_pred': 6,
                            'n_conv_encoder': 1, 'n_filters_encoder': 64,
                            'n_channels_conv': 128, 'n_channels_bottleneck': 64, 'n_channels_skip': 64,
                            'n_layer_each_block': 5, 'n_block_encoder': 1, 'n_block_decoder': 2,
                            'output_activation': 'tanh',  # 'norm_type': None,
                            'use_mask': False,
                            'is_multiple_decoder': True, 'model_type': 'ae', 'encoder_multiple_out': True})

            # Multiple-Decoder Wave-U-Net without skip connections
            search_model(PATH_MODEL, 'model_21_6_10', z_src_name_i, x_dict_i, z_dict_set_i, None, input_dim,
                         bool_train=True, bool_clean_weight_file=True, bool_predict=True,
                         path_save_load_decoder=os.path.join(path_result_i, 'model_21_6_10'),
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': n_outputs_i,
                            'epochs': 800, 'batch_size': 16, 'bs_pred': 16,
                            #  'batch_norm': True,
                            'n_pad_input': 13, 'num_layers': 4,
                            'use_skip': False, 'output_type': 'direct', 'output_activation': 'tanh',
                            'is_multiple_decoder': True, 'model_type': 'ae', 'encoder_multiple_out': True})

    logging.info('finished')
