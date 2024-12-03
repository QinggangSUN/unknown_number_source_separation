# -*- coding: utf-8 -*-
"""
@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation, too-few-public-methods, too-many-arguments
# pylint: disable=too-many-branches, too-many-locals, too-many-nested-blocks, too-many-statements

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
from keras.layers import Input, Lambda
from keras.models import Model

from prepare_data_shipsear_recognition_mix_s0tos3 import read_data, read_datas, save_datas
from file_operation import mkdir

from see_metric_autoencoder import save_metric, display_metric
from see_metric_autoencoder import recover_pad_num_samples_list

from experiments.metrics import PenalizedSISNR


def display_and_save_metrics(path_save, set_names, metric_list, hist_bins=None, save_name=None):
    """display and save metrics.
    Args:
        path_save (str): path where to save file.
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
        metric_list (list): metric values.
        hist_bins (tuple(float, float, int), optional): paras for hist. Defaults to None.
        save_name (str, optional): for name to save. Defaults to None.
    """
    metric_mean_list = [np.average(metric_set_i) for metric_set_i in metric_list]
    display_metric(metric_list, path_save, set_names, hist_bins=hist_bins, save_name=save_name)
    save_metric(path_save, save_name, [f'{save_name}_mean'], [np.asarray(metric_mean_list)],
                {save_name: dict(zip(set_names, metric_list))})


def compute_metrics(object_decoded_files, data_s_sets, set_names, zp_name_end, **kwargs):
    """compute p-si-snr.
    Args:
        object_decoded_files (class ListDecodedFiles): object for list predict decoded files
        data_s_sets (list[np.ndarray]): pure sources.
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
        zp_name_end (str): name of the predict file key word, e.g. 'autodecoded'.
    """
    pit = kwargs['pit'] if 'pit' in kwargs.keys() else True
    penalty = kwargs['penalty'] if 'penalty' in kwargs.keys() else None
    name_weight_files = object_decoded_files.name_weight_files  # [model][src][para_name][weight_file]
    path_out_models = object_decoded_files.path_out_models  # [model][src][para_name]
    path_metrics_paras = object_decoded_files.path_metrics_paras  # [model][para_name]

    for name_weight_files_j, path_metrics_para_j, path_out_model_j in zip(  # level j: [model]
            name_weight_files, path_metrics_paras, path_out_models):
        name_weight_files_j = name_weight_files_j[0]  # [para_name][weight_file]
        path_out_model_j = path_out_model_j[0]  # [para_name]

        for name_weight_files_k, path_metric_para_k, path_out_model_k in zip(  # level k: [para_name]
                name_weight_files_j, path_metrics_para_j, path_out_model_j):

            name_weight_file_l = name_weight_files_k[-1]  # level l: [weight_file]
            path_sp_l = []  # [set]
            data_sp_z_l = []  # [set](sample, fl, channel)
            for set_name_i in set_names:
                name_data_sp_z_l_i = f'Z_zero_{set_name_i}_{zp_name_end}_{name_weight_file_l}.hdf5'
                path_sp_l.append(os.path.join(path_out_model_k, name_data_sp_z_l_i))
                data_sp_z_l.append(np.asarray(read_data(path_out_model_k, name_data_sp_z_l_i)))

            result_list_dict = {'num_active_refs': [], 'num_active_seps': [], 'num_penalty_pairs': [],
                                'p_si_snr': [], 'avg_oracle_si_snr': [],
                                'num_tp': [], 'num_tn': [], 'num_fp': [], 'num_fn': []}

            for data_s_l, data_sp_l in zip(data_s_sets, data_sp_z_l):
                data_s_l = np.squeeze(data_s_l)
                data_sp_l_t = [data_sp_l_k.transpose() for data_sp_l_k in data_sp_l]  # (sample, channel, fl)

                data_sp_l_t = recover_pad_num_samples_list(data_s_l, data_sp_l_t)
                data_sp_l = np.asarray(data_sp_l_t)

                result_tensor_list_dict = dict()
                for name_result in result_list_dict.keys():
                    result_tensor_list_dict.update({name_result: []})

                data_s_l_tensor = tf.convert_to_tensor(data_s_l)
                data_sp_l_tensor = tf.convert_to_tensor(data_sp_l)
                n_samples = data_s_l.shape[0]
                for k in range(n_samples):
                    obj_p_si_snr = PenalizedSISNR(penalty=penalty)
                    data_s_k_tensor, data_sp_k_tensor = data_s_l_tensor[k], data_sp_l_tensor[k]
                    obj_p_si_snr.compute_fuss_p_si_snr(data_s_k_tensor, data_sp_k_tensor, pit=pit)
                    result_dict_k = obj_p_si_snr.result_dict

                    for name_result in result_list_dict.keys():
                        result_tensor_list_dict[name_result].append(result_dict_k[name_result])

                for name_result in result_list_dict.keys():
                    result_list_dict[name_result].append(
                        tf.stack(result_tensor_list_dict[name_result]).eval(session=tf.Session()))

            path_metric_result_k = os.path.join(path_metric_para_k, f'penalty_{penalty}')
            mkdir(path_metric_result_k)
            for name_result in result_list_dict.keys():
                display_and_save_metrics(path_metric_result_k, set_names, result_list_dict[name_result],
                                         save_name=name_result)


def extract_metric_from_keras_result(path_result, file_name, set_names, result_names):
    """extract metric from keras predict file.
    Args:
        path_result (str): path where to save file.
        file_name (str): file name.
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
        result_names (tuple(str)): tuple of names of metrics.
    """

    def extract_metric(path_result_, file_name_, set_names_, index):
        """extract metric from keras predict file.
        Args:
            path_result_ (str): path where to save file.
            file_name_ (str): file name.
            set_names_ (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
            index (int): metric index in result file.
        Returns:
            metric_list (list): [set] sets of metrics.
        """
        metric_list = []
        for set_name_i in set_names_:
            data = read_data(path_result_, f'{file_name_}_{set_name_i}')
            metric_list.append(data[index, :])
        return metric_list

    for i, name_i in enumerate(result_names):
        metric_list_i = extract_metric(path_result, file_name, set_names, i)
        display_and_save_metrics(path_result, set_names, metric_list_i, save_name=name_i)


def compute_metrics_with_keras(object_decoded_files, data_s_sets, set_names, zp_name_end, **kwargs):
    """compute p-si-snr.
    Args:
        object_decoded_files (class ListDecodedFiles): object for list predict decoded files
        data_s_sets (list[np.ndarray]): pure sources.
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
        zp_name_end (str): name of the predict file key word, e.g. 'autodecoded'.
    """
    pit = kwargs['pit'] if 'pit' in kwargs.keys() else True
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 1
    penalty = kwargs['penalty'] if 'penalty' in kwargs.keys() else None
    reward = kwargs['reward'] if 'reward' in kwargs.keys() else 0.
    model_type = kwargs['model_type'] if 'model_type' in kwargs.keys() else 1
    result_names = kwargs['result_names'] if 'result_names' in kwargs.keys() else (
        'num_active_refs', 'num_active_seps', 'num_penalty_pairs',
        'p_si_snr', 'avg_oracle_si_snr',
        'num_tp', 'num_tn', 'num_fp', 'num_fn')

    def layer_p_si_snr(inputs, n_channels_, frame_length_, pit_, batch_size_, result_names_):
        """function for keras layer.
        Args:
            inputs (tf.Tensor): layer inputs.
            n_channels_ (int): number of channels.
            frame_length_ (int): length of a frame of data.
            pit_ (bool): whether to use permutation invariant training.
            batch_size_ (int): batch size.
            result_names_ (tuple(str)): names of the results.
        Returns:
            (list): outputs of the layer.
        """
        data_s, data_sp = inputs
        result_tensor_list_dict = dict()
        for name_result in result_names_:
            result_tensor_list_dict.update({name_result: []})
        data_s_tensor = tf.reshape(data_s, (batch_size_, n_channels_, frame_length_))
        data_sp_tensor = tf.reshape(data_sp, (batch_size_, n_channels_, frame_length_))
        for k in range(batch_size_):
            obj_p_si_snr_k = PenalizedSISNR(penalty=penalty, reward=reward)
            data_s_k_tensor, data_sp_k_tensor = data_s_tensor[k], data_sp_tensor[k]
            obj_p_si_snr_k.compute_fuss_p_si_snr(data_s_k_tensor, data_sp_k_tensor, pit=pit_)
            result_dict_k = obj_p_si_snr_k.result_dict
            for name_result in result_tensor_list_dict.keys():
                result_tensor_list_dict[name_result].append(result_dict_k[name_result])

        result_tensor_list = []
        for name_result in result_names_:
            result_tensor_list.append(tf.stack(result_tensor_list_dict[name_result]))

        return result_tensor_list

    def create_keras_model(n_channels_, frame_length_, pit_, batch_size_, result_names_):
        """create a keras model.
        Args:
            n_channels_ (int): number of channels.
            frame_length_ (int): length of a frame of data.
            pit_ (bool): whether to use permutation invariant training.
            batch_size_ (int): batch size.
            result_names_ (tuple(str)): names of the results.
       Returns:
            model (keras.Model): a keras model.
        """
        input_source_frames = Input(shape=(n_channels_, frame_length_))
        input_separated_frames = Input(shape=(n_channels_, frame_length_))
        outputs = Lambda(layer_p_si_snr, arguments={'n_channels_': n_channels_,
                                                    'frame_length_': frame_length_,
                                                    'pit_': pit_,
                                                    'batch_size_': batch_size_,
                                                    'result_names_': result_names_}
                         )([input_source_frames, input_separated_frames])
        model_ = Model(inputs=[input_source_frames, input_separated_frames], outputs=outputs)
        return model_

    if model_type == 1:
        name_weight_files = object_decoded_files.name_weight_files  # [model][src][para_name][weight_file]
        path_out_models = object_decoded_files.path_out_models  # [model][src][para_name]
        path_metrics_paras = object_decoded_files.path_metrics_paras  # [model][para_name]

        for name_weight_files_j, path_metrics_para_j, path_out_model_j in zip(  # level j: [model]
                name_weight_files, path_metrics_paras, path_out_models):
            name_weight_files_j = name_weight_files_j[0]  # [para_name][weight_file]
            path_out_model_j = path_out_model_j[0]  # [para_name]

            for name_weight_files_k, path_metric_para_k, path_out_model_k in zip(  # level k: [para_name]
                    name_weight_files_j, path_metrics_para_j, path_out_model_j):
                path_metric_result_k = os.path.join(path_metric_para_k, f'penalty_{penalty}_reward_{reward}')
                mkdir(path_metric_result_k)

                name_weight_file_l = name_weight_files_k[-1]  # level l: [weight_file]
                path_sp_l = []  # [set]
                data_sp_z_l = []  # [set](sample, fl, channel)
                for set_name_i in set_names:
                    name_data_sp_z_l_i = f'Z_zero_{set_name_i}_{zp_name_end}_{name_weight_file_l}.hdf5'
                    path_sp_l.append(os.path.join(path_out_model_k, name_data_sp_z_l_i))
                    data_sp_z_l.append(np.asarray(read_data(path_out_model_k, name_data_sp_z_l_i)))

                for data_s_l, data_sp_l, set_name_i in zip(data_s_sets, data_sp_z_l, set_names):
                    data_s_l = np.squeeze(data_s_l)
                    data_sp_l_t = [data_sp_l_k.transpose() for data_sp_l_k in data_sp_l]  # (sample, channel, fl)

                    data_sp_l_t = recover_pad_num_samples_list(data_s_l, data_sp_l_t)
                    data_sp_l = np.asarray(data_sp_l_t)

                    n_channels = data_sp_l.shape[1]
                    frame_length = data_s_l.shape[2]
                    model = create_keras_model(n_channels, frame_length, pit, batch_size, result_names)
                    result_dict = model.predict([data_s_l, data_sp_l], batch_size=batch_size)
                    save_datas({f'result_{set_name_i}': result_dict}, path_metric_result_k)
                extract_metric_from_keras_result(path_metric_result_k, 'result', set_names, result_names)
    if model_type == 2:
        path_paras_two = object_decoded_files.path_paras_two  # [model][para_1][para_2]
        path_metrics = object_decoded_files.path_metrics  # [model][para_1][para_2]
        for path_paras_two_model, path_metrics_model in zip(  # level: [model]
                path_paras_two, path_metrics):

            for path_paras_two_para_one, path_metrics_para_one in zip(  # level: [para_1]
                    path_paras_two_model, path_metrics_model):

                for path_paras_two_para_two, path_metrics_para_two in zip(  # level: [para_2]
                        path_paras_two_para_one, path_metrics_para_one):
                    mkdir(path_metrics_para_two)

                    path_metric_result_k = os.path.join(path_metrics_para_two, f'penalty_{penalty}_reward_{reward}')
                    mkdir(path_metric_result_k)

                    path_sp_l = []  # [set]
                    data_sp_z_l = []  # [set](sample, fl, channel)
                    for set_name_i in set_names:
                        name_data_sp_z_l_i = f'Z_zero_{set_name_i}_{zp_name_end}.hdf5'
                        path_sp_l.append(os.path.join(path_paras_two_para_two, name_data_sp_z_l_i))
                        data_sp_z_l.append(np.asarray(read_data(path_paras_two_para_two, name_data_sp_z_l_i)))

                    for data_s_l, data_sp_l, set_name_i in zip(data_s_sets, data_sp_z_l, set_names):
                        data_s_l = np.squeeze(data_s_l)
                        data_sp_l_t = [data_sp_l_k.transpose() for data_sp_l_k in data_sp_l]  # (sample, channel, fl)

                        data_sp_l_t = recover_pad_num_samples_list(data_s_l, data_sp_l_t)
                        data_sp_l = np.asarray(data_sp_l_t)

                        n_channels = data_sp_l.shape[1]
                        frame_length = data_s_l.shape[2]
                        model = create_keras_model(n_channels, frame_length, pit, batch_size, result_names)
                        result_dict = model.predict([data_s_l, data_sp_l], batch_size=batch_size)
                        save_datas({f'result_{set_name_i}': result_dict}, path_metric_result_k)
                    extract_metric_from_keras_result(path_metric_result_k, 'result', set_names, result_names)


def compute_metrics_with_keras_kim(object_decoded_files, data_s_sets, set_names, zp_name_end,
                                   data_x_sets, data_y_sets, hparams, **kwargs):
    """compute p-si-snr.
    Args:
        object_decoded_files (class ListDecodedFiles): object for list predict decoded files.
        data_s_sets (list[np.ndarray]): pure sources.
        set_names (list[str]): list of names of sets, e.g. ['train', 'val', 'test'].
        zp_name_end (str): name of the predict file key word, e.g. 'autodecoded'.
        data_x_sets (list[np.ndarray]): mixture waveforms.
        data_y_sets (list[np.ndarray]): labels of mixture.
        hparams (HParams): Model hyperparameters.
    """
    batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 1
    penalty = kwargs['penalty'] if 'penalty' in kwargs.keys() else None
    reward = kwargs['reward'] if 'reward' in kwargs.keys() else 0.
    result_names = kwargs['result_names'] if 'result_names' in kwargs.keys() else (
        'num_active_refs', 'num_active_seps', 'num_penalty_pairs',
        'p_si_snr', 'avg_oracle_si_snr',
        'num_tp', 'num_tn', 'num_fp', 'num_fn')

    def layer_p_si_snr(inputs, n_channels_, frame_length_, batch_size_, result_names_, hparams_):
        """function for keras layer.
        Args:
            inputs (tf.Tensor): layer inputs.
            n_channels_ (int): number of channels.
            frame_length_ (int): length of a frame of data.
            batch_size_ (int): batch size.
            result_names_ (tuple(str)): names of the results.
            hparams_ (HParams): Model hyperparameters.
        Returns:
            (list): outputs of the layer.
        """
        data_s, data_sp, data_x, data_y = inputs
        result_tensor_list_dict = dict()
        for name_result in result_names_:
            result_tensor_list_dict.update({name_result: []})
        data_s_tensor = tf.reshape(data_s, (batch_size_, n_channels_, frame_length_))
        data_sp_tensor = tf.reshape(data_sp, (batch_size_, n_channels_, frame_length_))
        data_x_tensor = tf.reshape(data_x, (batch_size_, frame_length_))
        data_y_tensor = tf.reshape(data_y, (batch_size_, n_channels_))
        for k in range(batch_size_):
            obj_p_si_snr_k = PenalizedSISNR(penalty=penalty, reward=reward)
            data_s_k_tensor, data_sp_k_tensor = data_s_tensor[k], data_sp_tensor[k]
            data_x_k_tensor, data_y_k_tensor = data_x_tensor[k], data_y_tensor[k]
            obj_p_si_snr_k.compute_kim_p_si_snr(data_s_k_tensor, data_sp_k_tensor, data_x_k_tensor, data_y_k_tensor,
                                                hparams_)
            result_dict_k = obj_p_si_snr_k.result_dict
            for name_result in result_tensor_list_dict.keys():
                result_tensor_list_dict[name_result].append(result_dict_k[name_result])

        result_tensor_list = []
        for name_result in result_names_:
            result_tensor_list.append(tf.stack(result_tensor_list_dict[name_result]))

        return result_tensor_list

    def create_keras_model(n_channels_, frame_length_, batch_size_, result_names_, hparams_):
        """create a keras model.
        Args:
            n_channels_ (int): number of channels.
            frame_length_ (int): length of a frame of data.
            batch_size_ (int): batch size.
            result_names_ (tuple(str)): names of the results.
            hparams_ (HParams): Model hyperparameters.
       Returns:
            model (keras.Model): a keras model.
        """
        input_source_frames = Input(shape=(n_channels_, frame_length_))
        input_separated_frames = Input(shape=(n_channels_, frame_length_))
        input_mixture_frames = Input(shape=(frame_length_,))
        input_y_labels = Input(shape=(n_channels_,))
        outputs = Lambda(layer_p_si_snr, arguments={'n_channels_': n_channels_,
                                                    'frame_length_': frame_length_,
                                                    'batch_size_': batch_size_,
                                                    'result_names_': result_names_,
                                                    'hparams_': hparams_, }
                         )([input_source_frames, input_separated_frames, input_mixture_frames, input_y_labels])
        model_ = Model(inputs=[input_source_frames, input_separated_frames, input_mixture_frames, input_y_labels],
                       outputs=outputs)
        return model_

    name_weight_files = object_decoded_files.name_weight_files  # [model][src][para_name][weight_file]
    path_out_models = object_decoded_files.path_out_models  # [model][src][para_name]
    path_metrics_paras = object_decoded_files.path_metrics_paras  # [model][para_name]

    for name_weight_files_j, path_metrics_para_j, path_out_model_j in zip(  # level j: [model]
            name_weight_files, path_metrics_paras, path_out_models):
        name_weight_files_j = name_weight_files_j[0]  # [para_name][weight_file]
        path_out_model_j = path_out_model_j[0]  # [para_name]

        for name_weight_files_k, path_metric_para_k, path_out_model_k in zip(  # level k: [para_name]
                name_weight_files_j, path_metrics_para_j, path_out_model_j):
            path_metric_result_k = os.path.join(path_metric_para_k, f'penalty_{penalty}_reward_{reward}')
            mkdir(path_metric_result_k)

            name_weight_file_l = name_weight_files_k[-1]  # level l: [weight_file]
            path_sp_l = []  # [set]
            data_sp_z_l = []  # [set](sample, fl, channel)
            for set_name_i in set_names:
                name_data_sp_z_l_i = f'Z_zero_{set_name_i}_{zp_name_end}_{name_weight_file_l}.hdf5'
                path_sp_l.append(os.path.join(path_out_model_k, name_data_sp_z_l_i))
                data_sp_z_l.append(np.asarray(read_data(path_out_model_k, name_data_sp_z_l_i)))

            for data_s_l, data_sp_l, set_name_i, data_x_l, data_y_l in zip(
                    data_s_sets, data_sp_z_l, set_names, data_x_sets, data_y_sets):
                data_s_l = np.squeeze(data_s_l)
                data_sp_l_t = [data_sp_l_k.transpose() for data_sp_l_k in data_sp_l]  # (sample, channel, fl)
                data_sp_l_t = recover_pad_num_samples_list(data_s_l, data_sp_l_t)
                data_sp_l = np.asarray(data_sp_l_t)
                data_x_l = np.squeeze(data_x_l)
                data_y_l = np.squeeze(data_y_l)

                n_channels = data_sp_l.shape[1]
                frame_length = data_s_l.shape[2]
                model = create_keras_model(n_channels, frame_length, batch_size, result_names, hparams)
                result_dict = model.predict([data_s_l, data_sp_l, data_x_l, data_y_l], batch_size=batch_size)
                save_datas({f'result_{set_name_i}': result_dict}, path_metric_result_k)
            extract_metric_from_keras_result(path_metric_result_k, 'result', set_names, result_names)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO)

    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep
    from recognition_mix_shipsear_s0tos3_preprocess import n_hot_labels
    from see_metric_single_source_ae_ns import ListDecodedFiles as ListDecodedFilesOne
    from see_metric_separation_search_encoded import ListDecodedFiles as ListDecodedFilesTwo
    from experiments.KIM2023.train_separation_kim import get_kim_hparams

    # for ShipsEar data
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    SCALER_DATA = 'max_one'
    SUB_SET_WAY = 'rand'
    SPLIT_WAY = 'split'
    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav',
                                   scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY, split_way=SPLIT_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source

    SET_NAMES = ['train', 'val', 'test']
    X_NAMES = ['X']
    X_SET_NAMES = [[f'{x_names_i}_{name_set_j}' for name_set_j in SET_NAMES] for x_names_i in X_NAMES]
    X_DATA = [read_datas(PATH_DATA, name_i) for name_i in X_SET_NAMES][0]
    X_DICT = [dict(zip(x_set_names_j, read_datas(PATH_DATA, x_set_names_j))) for x_set_names_j in X_SET_NAMES][0]
    Y_NAMES = ['Y_f']
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

    PATH_SAVE_ROOT = '../../result_separation_fuss'
    object_decoded_files_one = ListDecodedFilesOne(PATH_SAVE_ROOT, Z_SRC_NAME)
    compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                               **{'batch_size': 8})
    compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                               **{'penalty': 20., 'batch_size': 8})
    compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                               **{'penalty': 20., 'reward': 20., 'batch_size': 8})
    compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                               **{'penalty': 30., 'batch_size': 8})
    compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                               **{'penalty': 30., 'reward': 30., 'batch_size': 8})

    PATH_SAVE_ROOT = '../../result_separation_one_autoencoder'
    object_decoded_files_one = ListDecodedFilesOne(PATH_SAVE_ROOT, Z_SRC_NAME)
    compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                               **{'batch_size': 8, 'pit': False})
    compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                               **{'batch_size': 8, 'pit': False, 'penalty': 20., })
    compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                               **{'batch_size': 8, 'pit': False, 'penalty': 20., 'reward': 20., })
    compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                               **{'batch_size': 8, 'pit': False, 'penalty': 30., })
    compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                               **{'batch_size': 8, 'pit': False, 'penalty': 30., 'reward': 30., })

    PATH_SAVE_ROOT = '../../result_separation_search_decoder'
    object_decoded_files_two = ListDecodedFilesTwo(PATH_SAVE_ROOT)
    compute_metrics_with_keras(object_decoded_files_two, Z_DATA[2:], SET_NAMES[2:], 'decoded',
                               **{'batch_size': 8, 'pit': False, 'model_type': 2})
    compute_metrics_with_keras(object_decoded_files_two, Z_DATA[2:], SET_NAMES[2:], 'decoded',
                               **{'batch_size': 8, 'pit': False, 'penalty': 20., 'model_type': 2})
    compute_metrics_with_keras(object_decoded_files_two, Z_DATA[2:], SET_NAMES[2:], 'decoded',
                               **{'batch_size': 8, 'pit': False, 'penalty': 20., 'reward': 20., 'model_type': 2})
    compute_metrics_with_keras(object_decoded_files_two, Z_DATA[2:], SET_NAMES[2:], 'decoded',
                               **{'batch_size': 8, 'pit': False, 'penalty': 30., 'model_type': 2})
    compute_metrics_with_keras(object_decoded_files_two, Z_DATA[2:], SET_NAMES[2:], 'decoded',
                               **{'batch_size': 8, 'pit': False, 'penalty': 30., 'reward': 30., 'model_type': 2})

    PATH_SAVE_ROOT = '../../result_separation_ae_ns_single'
    model_names = ('model_13_2_1', 'model_13_3_1', 'model_15_2_6', 'model_21_6_10')
    for model_name in model_names:
        path_save_model = os.path.join(PATH_SAVE_ROOT, model_name, 'load_decoder')
        object_decoded_files_one = ListDecodedFilesOne(path_save_model, Z_SRC_NAME)
        compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                                   **{'batch_size': 8, 'pit': False})
        compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                                   **{'batch_size': 8, 'pit': False, 'penalty': 20., })
        compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                                   **{'batch_size': 8, 'pit': False, 'penalty': 20., 'reward': 20., })
        compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                                   **{'batch_size': 8, 'pit': False, 'penalty': 30.})
        compute_metrics_with_keras(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded',
                                   **{'batch_size': 8, 'pit': False, 'penalty': 30., 'reward': 30., })

    PATH_SAVE_ROOT = '../../result_separation_kim'
    object_decoded_files_one = ListDecodedFilesOne(PATH_SAVE_ROOT, Z_SRC_NAME)
    hparams_kim = get_kim_hparams()
    compute_metrics_with_keras_kim(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded', X_DATA, Y_DATA,
                                   hparams_kim, **{'batch_size': 8})
    compute_metrics_with_keras_kim(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded', X_DATA, Y_DATA,
                                   hparams_kim, **{'penalty': 20., 'batch_size': 8})
    compute_metrics_with_keras_kim(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded', X_DATA, Y_DATA,
                                   hparams_kim, **{'penalty': 20., 'reward': 20., 'batch_size': 8})
    compute_metrics_with_keras_kim(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded', X_DATA, Y_DATA,
                                   hparams_kim, **{'penalty': 30., 'batch_size': 8})
    compute_metrics_with_keras_kim(object_decoded_files_one, Z_DATA, SET_NAMES, 'autodecoded', X_DATA, Y_DATA,
                                   hparams_kim, **{'penalty': 30., 'reward': 30., 'batch_size': 8})

    logging.info('finished')
