# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:34:29 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation
# pylint: disable=too-many-arguments, too-many-branches, too-many-lines, too-many-locals, too-many-statements

if __name__ == '__main__':
    import logging

    import numpy as np
    import tensorflow
    if tensorflow.__version__ >= '2.0':
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        # from tensorflow import keras
        import tensorflow.compat.v1.keras.backend as K
    else:
        import tensorflow as tf
        import keras
        from keras import backend as K

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

    SCALER_DATA = 'max_one'
    # SCALER_DATA = 'or'
    SUB_SET_WAY = 'rand'
    # SUB_SET_WAY = 'order'

    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav', scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source

    SET_NAMES = ['train', 'val', 'test']
    X_NAMES = ['X']
    X_SET_NAMES = [[f'{x_names_i}_{set_name_j}' for set_name_j in SET_NAMES] for x_names_i in X_NAMES]
    X_DICT = [dict(zip(x_set_names_j, read_datas(PATH_DATA, x_set_names_j))) for x_set_names_j in X_SET_NAMES]
    # Z_NAMES = [f'Z_{i}_ns' for i in range(4)]
    Z_NAMES = ['Z']
    Z_SET_NAMES_R = [[f'{z_name_i}_{name_set_j}_zero' for name_set_j in SET_NAMES] for z_name_i in Z_NAMES]
    Z_SET_NAMES = [[f'{z_name_i}_zero_{name_set_j}' for name_set_j in SET_NAMES] for z_name_i in Z_NAMES]
    Z_DICT = [dict(zip(z_set_names_s_j, read_datas(PATH_DATA, z_set_names_r_j)))
              for z_set_names_s_j, z_set_names_r_j in zip(Z_SET_NAMES, Z_SET_NAMES_R)]

    PATH_RESULT = '../result_separation_one_autoencoder'
    mkdir(PATH_RESULT)

    # lr_i_j = [(i, j) for i in range(5, 6) for j in range(-4, -5, -1)]
    lr_i_j = [(i, j) for i in range(1, 2) for j in range(-3, -4, -1)]
    for (lr_i, lr_j) in lr_i_j:
        logging.info('model test')
        for x_name_i, x_dict_i, z_name_i, z_dict_i in zip(X_SET_NAMES, X_DICT, Z_SET_NAMES, Z_DICT):
            logging.info(x_dict_i.keys())

            input_dim = x_dict_i[x_name_i[-1]].shape[-1]  # (nsamples, 1, frame_length)
            if input_dim == 1:
                input_dim = x_dict_i[x_name_i[-1]].shape[-2]  # (nsamples, frame_length, 1)
            logging.debug(f'input_dim {input_dim}')

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.squeeze(value)  # (nsamples, frame_length)
                logging.debug(f'x_i.shape {x_dict_i[key].shape}')

            for key, value in z_dict_i.items():
                z_dict_i[key] = np.squeeze(value).transpose((0, 2, 1))  # -> (nsamples, frame_length, 4)
                logging.debug(f'z_i.name {key}')
                logging.debug(f'z_i.shape {z_dict_i[key].shape}')

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (nsamples, frame_length, 1)
                logging.debug(f'x_i.name {key}')
                logging.debug(f'x_i.shape {x_dict_i[key].shape}')

            # search_model(PATH_RESULT, 'model_8_2_1', input_dim, x_dict_i, z_dict_i, None,
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                  'n_conv_encoder': 1, 'n_filters': 64,
            #                  'rnn_type': 'BLSTM', 'latent_dim': 200, 'n_rnn_decoder': 1,
            #                  'bool_train': True, 'bool_test_ae': False,
            #                  'bool_save_ed': False, 'bool_test_ed': False,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                  'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # search_model(PATH_RESULT, 'model_8_4_1', input_dim, x_dict_i, z_dict_i, None,
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                  'n_conv_encoder': 1, 'n_filters': 64, 'use_bias': False,
            #                  'rnn_type': 'BLSTM', 'latent_dim': 256, 'n_rnn_decoder': 1,
            #                  'bool_train': True, 'bool_test_ae': False,
            #                  'bool_save_ed': False, 'bool_test_ed': False,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True,
            #                  'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # model_10 RNN TasNet
            search_model(PATH_RESULT, 'model_10_2_1', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                             'n_conv_encoder': 1, 'n_filters_conv': 64,
                             'block_type': 'BLSTM', 'latent_dim': 200,
                             'n_block_encoder': 1,  'n_block_decoder': 1,
                             'bool_train': True, 'bool_test_ae': False,
                             'bool_save_ed': False, 'bool_test_ed': False,
                             'bool_clean_weight_file': True, 'bool_test_weight': True,
                             'bool_test_ae_w': True, 'bool_test_ed_w': False})

            search_model(PATH_RESULT, 'model_10_3_1', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                             'n_conv_encoder': 1, 'n_filters_conv': 64,
                             'block_type': 'dprnn', 'latent_dim': 200,
                             'n_block_encoder': 1,  'n_block_decoder': 1,
                             'bool_train': True, 'bool_test_ae': False,
                             'bool_save_ed': False, 'bool_test_ed': False,
                             'bool_clean_weight_file': True, 'bool_test_weight': True,
                             'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # # model_12 RNN TasNet without mask
            search_model(PATH_RESULT, 'model_13_2_1', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                             'n_conv_encoder': 1, 'n_filters_conv': 64,
                             'block_type': 'BLSTM', 'latent_dim': 200,
                             'n_block_encoder': 1,  'n_block_decoder': 1,
                             'is_multiple_decoder': True, 'use_mask': False,
                             'bool_train': True, 'bool_test_ae': False,
                             'bool_save_ed': False, 'bool_test_ed': False,
                             'bool_clean_weight_file': True, 'bool_test_weight': True,
                             'bool_test_ae_w': True, 'bool_test_ed_w': False})

            search_model(PATH_RESULT, 'model_13_3_1', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                             'n_conv_encoder': 1, 'n_filters_conv': 64,
                             'block_type': 'dprnn', 'latent_dim': 200,
                             'n_block_encoder': 1,  'n_block_decoder': 1,
                             'is_multiple_decoder': True, 'use_mask': False,
                             'bool_train': True, 'bool_test_ae': False,
                             'bool_save_ed': False, 'bool_test_ed': False,
                             'bool_clean_weight_file': True, 'bool_test_weight': True,
                             'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # Multiple-Decoder Conv-Tasnet without mask
            search_model(PATH_RESULT, 'model_15_2_6', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4,
                            'epochs': 200, 'batch_size': 6, 'bs_pred': 6,
                            'is_multiple_decoder': True, 'use_mask': False,
                            'n_conv_encoder': 1, 'n_filters_encoder': 64,
                            'n_channels_conv': 128, 'n_channels_bottleneck': 64, 'n_channels_skip': 64,
                            'n_layer_each_block': 5, 'n_block_encoder': 1, 'n_block_decoder': 2,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # Multiple-Decoder Wave-U-Net without skip connections
            search_model(PATH_RESULT, 'model_21_6_10', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': -4, 'n_outputs': 4,
                            'epochs': 800, 'batch_size': 16, 'bs_pred': 16,
                            'n_pad_input': 13, 'num_layers': 4,
                            'use_skip': False, 'output_type': 'direct', 'output_activation': 'tanh',
                            'is_multiple_decoder': True,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

    logging.info('finished')
