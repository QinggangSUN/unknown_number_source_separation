# -*- coding: utf-8 -*-
"""Compute parameters of eta_k."""

import itertools
import os
import numpy as np
import tensorflow

if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
else:
    import tensorflow as tf
from itertools import combinations
from keras.layers import Input, Lambda
from keras.models import Model

from prepare_data_shipsear_recognition_mix_s0tos3 import read_data, save_datas, update_json
from metric import samerate_acc_np
from experiments.metrics import cosine_similarity_tf
from see_metric_autoencoder import display_metric


def compute_valid_frequency_channel(y_true):
    """compute valid frequency of each channel.
    Args:
        y_true (np.ndarray): (n_samples, 1, n_channels) n-channel label of mix sources.
    Returns:
        frequency_channels (list[float]): valid frequency of each channel.
    """
    y_true = np.squeeze(y_true)  # (n_samples, n_channels)
    n_channel = y_true.shape[1]
    num_channels = [np.sum(y_true[:, i:i + 1]) for i in range(n_channel)]
    frequency_channels = []
    for num_channel_i in num_channels:
        frequency_channels.append(num_channel_i / np.sum(num_channels))
    return frequency_channels


def compute_and_write_valid_frequency_channel(path_y, path_save,
                                              y_set_names=('Y_f_train', 'Y_f_val', 'Y_f_test'),
                                              set_names=('train', 'val', 'test'),
                                              save_key_name='valid_frequency_channel',
                                              save_file_name='params'):
    """Read labels and compute and write valid frequency channel.
    Args:
        path_y (os.Path): path where saved labels.
        path_save (os.Path): path where saved result.
        y_set_names (tuple(str), optional): names of Y sets. Default to ('Y_f_train', 'Y_f_val', 'Y_f_test').
        set_names (tuple(str), optional): names of data sets. Default to ('train', 'val', 'test').
        save_key_name (str): key name of valid frequency of each channel.
        save_file_name (str): file name of valid frequency of each channel.
    """
    valid_frequency_channel_sets = {}
    for y_file_name, set_name in zip(y_set_names, set_names):
        y_true = read_data(path_y, y_file_name)
        valid_frequency_channel = compute_valid_frequency_channel(y_true)
        valid_frequency_channel_sets.update({set_name: valid_frequency_channel})
    save_datas({save_key_name: valid_frequency_channel_sets}, path_save,
               **{'file_name': save_file_name, 'form_save': 'json'})


def compute_eta_1(x_true, y_true, z_true):
    """compute cosine similarity between input and outputs.
    Args:
        x_true (np.ndarray): (n_samples, 1, frame_length) mix sources.
        y_true (np.ndarray): (n_samples, 1, n_channels) n-channel label of mix sources.
        z_true (np.ndarray): (n_samples, n_channels, 1, frame_length) n-channel label of mix sources.
    Returns:
        cosine_similarity_mt (list[float]): cosine similarity between input and outputs with multiple targets.
        eta_1 (float): maximum value of cosine_similarity_mt.
    """
    x_true = np.squeeze(x_true)  # (n_samples, frame_length)
    y_true = np.squeeze(y_true)  # (n_samples, n_channels)
    z_true = np.squeeze(z_true)  # (n_samples, n_channels, frame_length)
    cosine_similarity_mt = []
    for x_i, y_i, z_i in (x_true, y_true, z_true):
        if np.sum(y_i) > 1:
            for y_j, z_j in (y_i, z_i):
                if y_j == 1:
                    cosine_similarity_mt.append(samerate_acc_np(x_i, z_j))
    eta_1 = np.max(cosine_similarity_mt)
    return cosine_similarity_mt, eta_1


def compute_eta_1_tf(x_true, y_true, z_true, n_channels_):
    """compute cosine similarity between input and outputs.
    Args:
        x_true (tf.Tensor): (frame_length,) mix sources.
        y_true (tf.Tensor): (n_channels,) n-channel label of mix sources.
        z_true (tf.Tensor): (n_channels, frame_length) n-channel label of mix sources.
        n_channels_ (int): number of channels.
    Returns:
        cosine_similarity (tf.Tensor): (n_channels,) cosine similarity between input and outputs with multiple targets.
    """
    mask_mt_sample = tf.cond(tf.greater(tf.reduce_sum(y_true), 1.), lambda: True, lambda: False)
    cosine_similarity = []
    for j in range(n_channels_):
        y_j, z_j = y_true[j], z_true[j]
        cosine_similarity_j = tf.cond(tf.logical_and(mask_mt_sample, tf.cast(y_j, tf.bool)),
                                      lambda: cosine_similarity_tf(x_true, z_j),
                                      lambda: -1.)
        cosine_similarity.append(cosine_similarity_j)
    cosine_similarity = tf.stack(cosine_similarity)
    return cosine_similarity


def compute_eta_1_with_keras(path_source_data,
                             x_set_names=('X_train', 'X_val', 'X_test'),
                             y_set_names=('Y_f_train', 'Y_f_val', 'Y_f_test'),
                             z_set_names=('Z_train_zero', 'Z_val_zero', 'Z_test_zero'),
                             set_names=('train', 'val', 'test'),
                             result_model_output_name='result_eta_1',
                             path_save=None,
                             save_key_name='eta_1',
                             save_file_name='params',
                             **kwargs):
    """Compute eta_1 with keras.
    Args:
        path_source_data (str): path where data is.
        x_set_names (tuple, optional): set names of X. Defaults to ('X_train', 'X_val', 'X_test').
        y_set_names (tuple, optional): set names of Y. Defaults to ('Y_f_train', 'Y_f_val', 'Y_f_test').
        z_set_names (tuple, optional): set names of Z. Defaults to ('Z_train_zero', 'Z_val_zero', 'Z_test_zero').
        set_names (tuple(str), optional): tuple of names of sets. Defaults to ('train', 'val', 'test').
        result_model_output_name (str, optional): file name of model output. Defaults to 'result_eta_1'.
        path_save (str, optional): path where to save file. Defaults to None.
        save_key_name (str, optional): key name of eta_1. Defaults to 'eta_1'.
        save_file_name (str, optional): result file name. Defaults to 'params'.
    """
    path_save = path_source_data if path_save is None else path_save

    def layer_eta_1(inputs, batch_size_, frame_length_, n_channels_):
        """keras layer for compute eta_1.
        Args:
            inputs (tf.Tensor): inputs of the layer.
            batch_size_ (int): batch size.
            frame_length_ (int): length of a frame of data.
            n_channels_ (int): number of channels.
        Returns:
            cosine_similarity_ (tf.Tensor): (batch, n_channels) cosine_similarity between input and outputs.
        """
        x, y, z = inputs
        x = tf.reshape(tf.squeeze(x), (batch_size_, frame_length_,))
        y = tf.reshape(tf.squeeze(y), (batch_size_, n_channels_,))
        z = tf.reshape(tf.squeeze(z), (batch_size_, n_channels_, frame_length_))
        cosine_similarity_ = []
        for i in range(batch_size_):
            cosine_similarity_i = compute_eta_1_tf(x[i], y[i], z[i], n_channels_)
            cosine_similarity_.append(cosine_similarity_i)
        cosine_similarity_ = tf.stack(cosine_similarity_)
        return cosine_similarity_

    def create_keras_model(batch_size_, frame_length_, n_channels_):
        """create a keras model.
        Args:
            batch_size_ (int): batch size.
            frame_length_ (int): length of a frame of data.
            n_channels_ (int): number of channels.
       Returns:
            model (keras.Model): a keras model.
        """
        input_x = Input(shape=(1, frame_length_))
        input_y = Input(shape=(1, n_channels_))
        input_z = Input(shape=(n_channels_, 1, frame_length_))
        outputs = Lambda(layer_eta_1, arguments={'batch_size_': batch_size_,
                                                 'frame_length_': frame_length_,
                                                 'n_channels_': n_channels_})([input_x, input_y, input_z])
        model_ = Model(inputs=[input_x, input_y, input_z], outputs=outputs)
        return model_

    if not os.path.exists(os.path.join(path_save, f'{result_model_output_name}.hdf5')):
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 1
        result_dict = {}
        for x_file_name, y_file_name, z_file_name, set_name in zip(
                x_set_names, y_set_names, z_set_names, set_names):
            x_data = read_data(path_source_data, x_file_name)
            y_data = read_data(path_source_data, y_file_name)
            z_data = read_data(path_source_data, z_file_name)
            frame_length = z_data.shape[3]
            n_channels = z_data.shape[1]
            model = create_keras_model(batch_size, frame_length, n_channels)
            result_data = model.predict([x_data, y_data, z_data], batch_size=batch_size)
            result_dict.update({set_name: np.asarray(result_data)})
        save_datas(result_dict, path_save, file_name=result_model_output_name, **{'mode_batch': 'one_file_no_chunk'})

    def filter_n_target(cosine_similarity_, y_data_, n_target):
        """filter cosine_similarity with number of n_target.
        Args:
            cosine_similarity_ (np.ndarray): cosine_similarity between input and outputs.
            y_data_ (np.ndarray): (n_channels,) n-channel label of mix sources.
            n_target (int): number of targets.
        Returns:
            cosine_similarity_n_target (np.ndarray): cosine_similarity between input and outputs with n_target.
        """
        cosine_similarity_n_target = []
        for cs_i, y_i in zip(cosine_similarity_, y_data_):
            if np.sum(y_i) == n_target:
                cosine_similarity_n_target.append(cs_i)
        cosine_similarity_n_target = np.asarray(cosine_similarity_n_target).flatten()
        cosine_similarity_n_target = cosine_similarity_n_target[cosine_similarity_n_target > 0.]
        return cosine_similarity_n_target

    cosine_similarity_mt_list = []
    cosine_similarity_two_target_list = []
    cosine_similarity_three_target_list = []
    eta_1_sets = {}
    for set_name_i, y_file_name in zip(set_names, y_set_names):
        cosine_similarity = np.asarray(read_data(path_save, result_model_output_name, dict_key=set_name_i)).flatten()
        cosine_similarity_mt = cosine_similarity[cosine_similarity > 0.]
        cosine_similarity_mt_list.append(cosine_similarity_mt)
        eta_1 = np.max(cosine_similarity_mt)
        eta_1_sets.update({set_name_i: float(eta_1)})

        cosine_similarity = np.asarray(read_data(path_save, result_model_output_name, dict_key=set_name_i))
        y_data = np.squeeze(read_data(path_source_data, y_file_name))
        cosine_similarity_two_target_list.append(filter_n_target(cosine_similarity, y_data, 2))
        cosine_similarity_three_target_list.append(filter_n_target(cosine_similarity, y_data, 3))

    display_metric(cosine_similarity_mt_list, path_save, set_names, save_name=save_key_name)
    display_metric(cosine_similarity_two_target_list, path_save, set_names, save_name=f'{save_key_name}_2t')
    display_metric(cosine_similarity_three_target_list, path_save, set_names, save_name=f'{save_key_name}_3t')
    update_json({save_key_name: eta_1_sets}, os.path.join(path_save, f'{save_file_name}.json'))


def filter_n_target_z_data(z_data, y_data, n_target):
    """filter z_data with target number of n_target.
    Args:
        z_data (np.ndarray): data of sources.
        y_data (np.ndarray): (n_channels,) n-channel label of mix sources.
        n_target (int): number of targets.
    Returns:
        z_data_filter (np.ndarray): z_data with target number of n_target.
    """
    z_data_filter = []
    for z_i, y_i in zip(z_data, y_data):
        if np.sum(y_i) == n_target:
            z_data_filter_i = z_i[y_i > 0.]
            z_data_filter.append(z_data_filter_i)
    z_data_filter = np.asarray(z_data_filter)
    return z_data_filter


# first remove a channel to make sure the target number is less than n_channels, set eta_3 as -1.

# then compute eta_2 to judge whether is two target or three target
def compute_cosine_similarity_pairs_outputs_tf(n_channels, predict_outputs):
    """compute cosine similarity between output pairs with TensorFlow.
    Args:
        n_channels (int): number of channels.
        predict_outputs (tf.Tensor): predict outputs of the model.
    Returns:
        similarity_output_pairs (tf.Tensor): cosine similarity between output pairs.
        valid_pairs_mask (tf.Tensor): mask for output pairs.
        combs (tf.Tensor): combination of index of channels.
    """
    combs = [_ for _ in combinations(range(n_channels), 2)]  # combination of index of channels
    similarity_output_pairs = []  # cosine similarity between output pairs
    for (c1, c2) in combs:
        similarity_output_pairs.append(cosine_similarity_tf(predict_outputs[c1], predict_outputs[c2]))
    similarity_output_pairs = tf.convert_to_tensor(similarity_output_pairs)
    valid_pairs_mask = tf.convert_to_tensor([True, ] * len(combs))
    combs = tf.convert_to_tensor(combs)
    return similarity_output_pairs, valid_pairs_mask, combs


def compute_eta_k_with_keras(path_source_data,
                             n_target,
                             y_set_names=('Y_f_train', 'Y_f_val', 'Y_f_test'),
                             z_set_names=('Z_train_zero', 'Z_val_zero', 'Z_test_zero'),
                             set_names=('train', 'val', 'test'),
                             result_model_output_name='result_eta_k',
                             path_save=None,
                             save_key_name='eta_k',
                             save_file_name='params',
                             **kwargs):
    """Compute eta_k with keras.
    Args:
        path_source_data (str): path where data is.
        n_target (int): number of targets.
        y_set_names (tuple, optional): set names of Y. Defaults to ('Y_f_train', 'Y_f_val', 'Y_f_test').
        z_set_names (tuple, optional): set names of Z. Defaults to ('Z_train_zero', 'Z_val_zero', 'Z_test_zero').
        set_names (tuple(str), optional): tuple of names of sets. Defaults to ('train', 'val', 'test').
        result_model_output_name (str, optional): file name of model output. Defaults to 'result_eta_k'.
        path_save (str, optional): path where to save file. Defaults to None.
        save_key_name (str, optional): key name of eta_k. Defaults to 'eta_k'.
        save_file_name (str, optional): result file name. Defaults to 'params'.
    """
    path_save = path_source_data if path_save is None else path_save

    def layer_eta_k(inputs, batch_size_, frame_length_, n_channels_):
        """keras layer for compute eta_1.
        Args:
            inputs (tf.Tensor): inputs of the layer.
            batch_size_ (int): batch size.
            frame_length_ (int): length of a frame of data.
            n_channels_ (int): number of channels.
        Returns:
            cosine_similarity_ (tf.Tensor): cosine_similarity between output pairs.
        """
        z = inputs
        z = tf.reshape(tf.squeeze(z), (batch_size_, n_channels_, frame_length_))
        cosine_similarity_ = []
        for i in range(batch_size_):
            cosine_similarity_i, _, _ = compute_cosine_similarity_pairs_outputs_tf(n_channels_, z[i])
            cosine_similarity_.append(cosine_similarity_i)
        cosine_similarity_ = tf.stack(cosine_similarity_)
        return cosine_similarity_

    def create_keras_model(batch_size_, frame_length_, n_channels_):
        """create a keras model.
        Args:
            batch_size_ (int): batch size.
            frame_length_ (int): length of a frame of data.
            n_channels_ (int): number of channels.
       Returns:
            model (keras.Model): a keras model.
        """
        input_z = Input(shape=(n_channels_, frame_length_))
        outputs = Lambda(layer_eta_k, arguments={'batch_size_': batch_size_,
                                                 'frame_length_': frame_length_,
                                                 'n_channels_': n_channels_})(input_z)
        model_ = Model(inputs=[input_z], outputs=outputs)
        return model_

    if not os.path.exists(os.path.join(path_save, f'{result_model_output_name}_{n_target}.hdf5')):
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 1
        result_dict = {}
        cosine_similarity_mt_list = []
        eta_k_sets = {}
        for set_name_i, y_file_name, z_file_name in zip(set_names, y_set_names, z_set_names):
            y_data = np.squeeze(read_data(path_source_data, y_file_name))
            z_data = np.squeeze(read_data(path_source_data, z_file_name))
            z_data = filter_n_target_z_data(z_data, y_data, n_target)
            frame_length = z_data.shape[2]
            n_channels = z_data.shape[1]
            model = create_keras_model(batch_size, frame_length, n_channels)
            cosine_similarity_mt = model.predict([z_data], batch_size=batch_size)
            result_dict.update({set_name_i: np.asarray(cosine_similarity_mt)})
            cosine_similarity_mt = np.asarray(cosine_similarity_mt).flatten()
            cosine_similarity_mt_list.append(cosine_similarity_mt)
            eta_k = np.max(cosine_similarity_mt)
            eta_k_sets.update({set_name_i: float(eta_k)})
        save_datas(result_dict, path_save, file_name=f'{result_model_output_name}_{n_target}t',
                   **{'mode_batch': 'one_file_no_chunk'})
        display_metric(cosine_similarity_mt_list, path_save, set_names, save_name=f'{save_key_name}_{n_target}t')
        update_json({f'{save_key_name}_{n_target}t': eta_k_sets}, os.path.join(path_save, f'{save_file_name}.json'))


# one way is use the minimum value of the cosine similarity between the three targets as eta_2
# another way is also compute eta_3 to decide eta_2

if __name__ == '__main__':
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep
    from recognition_mix_shipsear_s0tos3_preprocess import create_y_labels_nchannel

    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'
    OBJ_PATH = PathSourceRootSep
    SCALER_DATA = 'max_one'
    SUB_SET_WAY = 'rand'
    SPLIT_WAY = 'split'

    path_source = OBJ_PATH(PATH_DATA_ROOT, form_src='wav',
                           scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY, split_way=SPLIT_WAY).path_source
    create_y_labels_nchannel(path_source=path_source)

    compute_and_write_valid_frequency_channel(path_source, path_source)
    compute_eta_1_with_keras(path_source, **{'batch_size': 8})

    compute_eta_k_with_keras(path_source, 3)
