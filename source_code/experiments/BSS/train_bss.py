# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 18:13:42 2025

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

FastICA using scikit-learn

"""
from error import ParameterError
from prepare_data_shipsear_recognition_mix_s0tos3 import save_datas

from train_single_source_autoencoder_ns import compute_num_model
from experiments.BSS.ica import train_ica_model


def train_bss_model(path_result_root, model_name, x_dict, train_params, **kwargs):
    """For search best model.
    Args:
        path_result_root (str): where to save result.
        model_name (str): name of the model.
        x_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model input, mixed data.
        train_params (dict): dict of parameters for training.
    """
    num_model = compute_num_model(model_name)
    config_dict = dict()
    config_dict.update(train_params)
    config_dict.update(kwargs)

    path_result = os.path.join(path_result_root, model_name)
    mkdir(path_result)

    for (set_name_i, x_dict_set_i) in x_dict.items():
        if num_model in {1000, 2000}:
            s_predict, config_dict_train = train_ica_model(num_model, x_dict_set_i, train_params, **kwargs)
        else:
            raise ParameterError(f"Error, unsupported num_model {num_model}")

        save_name = f'{set_name_i}_predict'
        save_datas(dict({save_name: s_predict}), path_result, dtype=np.float32)
        config_dict.update(config_dict_train)
    save_datas({'config_dict': config_dict}, path_result, form_save='json', file_name='config_dict')


if __name__ == '__main__':
    import numpy as np

    import logging
    import os

    from file_operation import mkdir
    from prepare_data_shipsear_recognition_mix_s0tos3 import read_datas, get_sample_rate
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep
    from train_separation_known_number import index_mix_src_ns

    np.random.seed(1337)  # for reproducibility
    _SR = get_sample_rate()

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
    Z_INDEX = index_mix_src_ns(4)
    X_NAMES = [[f'Z_{i}_ns_{name_set_j}' for name_set_j in SET_NAMES] for i in range(4, 8)]
    Z_NAMES = [[[f'Z_{i}_ns_{name_set_j}' for name_set_j in SET_NAMES] for i in mix_i] for mix_i in Z_INDEX]

    X_DICT = [dict(zip(name_i, read_datas(PATH_DATA, name_i))) for name_i in X_NAMES]
    Z_DICT = [[dict(zip(name_i_j, read_datas(PATH_DATA, name_i_j))) for name_i_j in name_i] for name_i in Z_NAMES]

    PATH_RESULT = '../../../result_separation_bss'
    mkdir(PATH_RESULT)


    def format_data(x_dict, x_name, z_dict):
        for key, value in x_dict.items():
            x_dict[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (nsamples, frame_length, 1)
            logging.debug(f'x_i.name {key}')
            logging.debug(f'x_i.shape {x_dict[key].shape}')

        z_list_set = []
        for j_set, name_set_j in enumerate(x_name):
            z_data_set_ij = []
            for k_src, z_dict_i_k in enumerate(z_dict):
                # (nsamples, frame_length, 1)
                z_name_set_j = list(z_dict_i_k.keys())[j_set]
                z_data_set_ij.append(np.expand_dims(np.squeeze(z_dict_i_k[z_name_set_j]), axis=-1))
            z_data_set_arr_ij = np.concatenate(z_data_set_ij, axis=-1)  # (nsamples, frame_length, n_outputs_i)
            z_list_set.append(z_data_set_arr_ij)
        z_dict_set = dict(zip(x_name, z_list_set))

        return x_dict, z_dict_set


    for x_name_i, x_dict_i, z_name_i, z_dict_i in zip(X_NAMES, X_DICT, Z_NAMES, Z_DICT):
        logging.info(x_dict_i.keys())

        input_dim = x_dict_i[x_name_i[-1]].shape[-1]  # (nsamples, 1, frame_length)
        logging.debug(f'input_dim {input_dim}')
        n_outputs_i = len(z_name_i)

        x_dict_i, z_dict_set_i = format_data(x_dict_i, x_name_i, z_dict_i)

        train_bss_model(PATH_RESULT, 'model_2000_3_8', x_dict_i,
                        {'n_components': n_outputs_i},
                        **{'expand_channel_mode': 'interval_resample', 'num_virtual_channel': n_outputs_i * 8,
                           'orig_sr': _SR})

        train_bss_model(PATH_RESULT, 'model_2000_3_10', x_dict_i,
                        {'n_components': n_outputs_i},
                        **{'expand_channel_mode': 'interval_resample', 'num_virtual_channel': n_outputs_i * 10,
                           'orig_sr': _SR})

        train_bss_model(PATH_RESULT, 'model_2000_6_2_2', x_dict_i,
                        {'n_components': n_outputs_i},
                        **{'expand_channel_mode': 'time_delay_iir', 'num_virtual_channel': n_outputs_i,
                           'n_time_delay': 528, 'filter_param': (3, 0.8)})

