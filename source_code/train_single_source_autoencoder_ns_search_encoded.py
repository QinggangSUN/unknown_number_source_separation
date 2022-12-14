# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 15:47:50 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation
# pylint: disable=no-member, unused-import, ungrouped-imports, wrong-import-position
# pylint: disable=too-many-arguments, too-many-branches, too-many-lines, too-many-locals
# pylint: disable=too-many-instance-attributes, too-many-statements
# pylint: disable=invalid-name, redefined-outer-name
import gc
import logging
import os
import shutil

import h5py
import keras
from keras import backend as K
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
if keras.__version__ >= '2.3.1':
    from keras.layers import LayerNormalization
else:
    from keras_layer_normalization import LayerNormalization
from keras.layers import BatchNormalization, Bidirectional, Conv1D, Dense, Input, Lambda, LSTM, Reshape
import keras.losses
from keras.models import load_model, Model
import numpy as np
import tensorflow as tf

from error import Error
from file_operation import list_dirs_start_str, list_files_end_str, mkdir
from models.models_autoencoder import segment_encoded_signal, overlap_and_add_in_decoder, DprnnBlock
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data, read_datas, save_datas
from see_metric_single_source_ae_ns import ListDecodedFiles
from train_functions import output_history, save_model_struct
from train_single_source_autoencoder_ns import build_dict_model_load, compute_num_model, clear_model_weight_file
from train_single_source_autoencoder_ns import predict_autoencoder


def create_decoder_weight_model(decoder_model=None, path_save_model=None, auto_model_name=None,
                                weight_file_name=None, decoder_model_name=None, decoder_weight_model_name=None,
                                path_result_model=None, data_set_name=None, modelname=None,
                                paras=None, dict_model_load=None):
    """Split and save encoder and decoder model from autoencoder model with weights, through graph of model.
    Args:
        decoder_model (keras.Model, optional): decoder model. Defaults to None.
        path_save_model (str, optional): where to save model. Defaults to None.
        auto_model_name (str, optional): name of the autoencoder model. Defaults to None.
        weight_file_name (str, optional): name of the autoencoder model weight file. Defaults to None.
        decoder_model_name (str, optional): name of the decoder model. Defaults to None.
        decoder_weight_model_name (str, optional): full path name of the decoder model after load weight file.
                                                Defaults to None.
        path_result_model (str, optional): where to save model. Defaults to None.
        data_set_name (str, optional): name of the dataset, sub-dir of model. Defaults to None.
        modelname (str, optional): model name for models, e.g. '1_n3_2'. Defaults to None.
        paras (dict, optional): parameters for train model. Defaults to None.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
    """
    if modelname is None:
        i = paras['i']
        j = paras['j']
        epochs = paras['epochs']
        strj = f'n{str(j)[1:]}' if j < 0 else str(j)
        modelname = f'{i}_{strj}_{epochs}'
    if path_save_model is None:
        path_save_model = os.path.join(path_result_model, 'auto_model', data_set_name, modelname)
    if auto_model_name is None:
        auto_model_name = os.path.join(path_save_model, f'{modelname}_auto.h5')
    if weight_file_name is None:
        weight_file_name = list_files_end_str(path_save_model, 'hdf5', False)
        if not weight_file_name:
            raise Error('weight_file_name not found')
    if isinstance(weight_file_name, list):
        weight_file_name = weight_file_name[-1]
    if decoder_model_name is None:
        decoder_model_name = os.path.join(path_save_model, f'{modelname}_decoder.h5')
    if decoder_weight_model_name is None:
        decoder_weight_model_name = os.path.join(path_save_model, f'{modelname}_{weight_file_name}_decoder.h5')
    if decoder_model is None:
        decoder_model = load_model(decoder_model_name, custom_objects=dict_model_load)
        logging.debug(f'decoder model \n {decoder_model.summary()}')

    auto_model = load_model(auto_model_name, custom_objects=dict_model_load)
    auto_model.load_weights(os.path.join(path_save_model, weight_file_name))
    # n_decoder_layer (int): number of layers of encoder model.
    n_decoder_layer = len(decoder_model.layers[1:])
    for i, layer in enumerate(auto_model.layers[-n_decoder_layer:]):
        layer_weights = layer.get_weights()
        decoder_model.layers[i+1].set_weights(layer_weights)

    decoder_model.save(decoder_weight_model_name)


def test_predict_decoder(data_set_encoder, decoder_model, path_save, name_data_set_decoded,
                         mode_pred='batch_save', bs_pred=8):
    """Load and test encoder and decoder.
    Args:
        data_set_encoder (list[np.ndarray]): data of the encoded vector.
        decoder_model (keras.Model): model of decoder.
        path_save (str): where to save model.
        name_data_set_decoded (list[str]): name of the decoded datasets to save.
        mode_pred (str, optional): mode to predict decoder. Defaults to 'batch_save'.
        bs_pred (int, optional): batch size when predict model. Defaults to 8.
    """
    if not isinstance(data_set_encoder, list):
        data_set_encoder = [data_set_encoder]
    if not isinstance(name_data_set_decoded, list):
        name_data_set_decoded = [name_data_set_decoded]

    for data_set, name_data_decoded in zip(data_set_encoder, name_data_set_decoded):
        if os.path.isfile(os.path.join(path_save, f'{name_data_decoded}.hdf5')):
            name_data_decoded += '_test'
        predict_autoencoder(decoder_model, data_set, path_save, name_data_decoded, compile_model=True,
                            mode=mode_pred, bs_pred=bs_pred)


def test_decoder_weight_model_out(decoder_model=None, path_save_model=None, auto_model_name=None,
                                  weight_file_name=None, decoder_model_name=None, decoder_weight_model_name=None,
                                  path_result_model=None, data_set_name=None, modelname=None,
                                  paras=None, dict_model_load=None,
                                  data_set_encoder=None, path_out=None, name_data_set_decoded=None):
    """Test function, test create decoder model with checkpoint weight, and compute decoded output.
    Args:
        decoder_model (keras.Model, optional): model of decoder. Defaults to None.
        path_save_model (str, optional): path of the autoencoder model. Defaults to None.
        auto_model_name (str, optional): autoencoder model name. Defaults to None.
        weight_file_name (str, optional): name of the weight file name. Defaults to None.
        decoder_model_name (str, optional): name of the decoder model. Defaults to None.
        decoder_weight_model_name (str, optional): full path name of the decoder model after load checkpoint weight.
                                                Defaults to None.
        path_result_model (str, optional): where to save model. Defaults to None.
        data_set_name (str, optional): name of the dataset, sub-dir of model. Defaults to None.
        modelname (str, optional): model name for models. Defaults to None.
        paras (dict, optional): parameters for train model. Defaults to None.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
        data_set_encoder (list[np.ndarray], optional): data of the encoded vector. Defaults to None.
        path_out (str, optional): path to save predict output. Defaults to None.
        name_data_set_decoded (list[str], optional): name of the decoded datasets to save. Defaults to None.
    Examples:
        SET_NAMES = ['train', 'val', 'test']
        S_NAMES = [[f'Z_{i}_ns_{name_set_j}' for name_set_j in SET_NAMES] for i in range(4)] # [n_source][n_set]
        SRC_NAMES = [f'Z_{i}_ns' for i in range(4)]  # [n_src]
        for i in range(1, 2, 1):
            for j in range(-3, -4, -1):
                weight_file_names = ['weights_1_n3_2_01_0.02.hdf5', 'weights_1_n3_2_02_0.00.hdf5',
                                    'weights_1_n3_2_02_0.00.hdf5', 'weights_1_n3_2_01_0.00.hdf5']
                weight_file_names = ['weights_1_n3_2_02_2.26.hdf5', 'weights_1_n3_2_02_1.61.hdf5',
                                     'weights_1_n3_2_02_1.68.hdf5', 'weights_1_n3_2_02_1.99.hdf5']
                for s_names_j, src_name_j, weight_file_name in zip(
                        S_NAMES, SRC_NAMES, weight_file_names):
                    path_out = os.path.join(PATH_RESULT, 'model_1_3_2', 'auto_out', '1_n3_2')
                    path_save_model = os.path.join(PATH_RESULT, 'model_1_3_2', 'auto_model', src_name_j, '1_n3_2')
                    decoder_weight_model_name = os.path.join(path_save_model, f'1_n3_2_{weight_file_name}_decoder.h5')
                    test_decoder_weight_model_out(decoder_model=None,
                                                path_save_model=path_save_model,
                                                auto_model_name=None,
                                                weight_file_name=weight_file_name,
                                                decoder_model_name=None,
                                                # decoder_weight_model_name=decoder_weight_model_name,
                                                decoder_weight_model_name=None,
                                                path_result_model=os.path.join(PATH_RESULT, 'model_1_3_2'),
                                                data_set_name=src_name_j, modelname='1_n3_2',
                                                paras={'i': i, 'j': j, 'epochs': 2},
                                                dict_model_load=dict(zip(['samerate_acc_d2'], [samerate_acc_d2])),
                                                # data_set_encoder=[read_data(path_out,
                                                #                             f'{s_name_k}_{weight_file_name}_encoded')
                                                #                   for s_name_k in s_names_j],
                                                data_set_encoder=[np.zeros((1, 2)) for k in range(len(s_names_j))],
                                                path_out=path_out,
                                                # name_data_set_decoded=[f'{s_name_k}_{weight_file_name}_decoded'
                                                # name_data_set_decoded=[f'{s_name_k}_{weight_file_name}_decoded_later'
                                                name_data_set_decoded=[f'{s_name_k}_{weight_file_name}_decoded_zero'
                                                                        for s_name_k in s_names_j])

                    path_out = os.path.join(PATH_RESULT, 'model_8_4_1', 'auto_out', '1_n3_2')
                    path_save_model = os.path.join(PATH_RESULT, 'model_8_4_1', 'auto_model', src_name_j, '1_n3_2')
                    decoder_weight_model_name = os.path.join(path_save_model, f'1_n3_2_{weight_file_name}_decoder.h5')
                    dict_model_load = {'samerate_acc_d2': samerate_acc_d2,
                                       'LayerNormalization': LayerNormalization,
                                       'segment_encoded_signal': segment_encoded_signal,
                                       'overlap_and_add_in_decoder': overlap_and_add_in_decoder}
                    test_decoder_weight_model_out(decoder_model=None,
                                                  path_save_model=path_save_model,
                                                  auto_model_name=None,
                                                  weight_file_name=weight_file_name,
                                                  decoder_model_name=None,
                                                  # decoder_weight_model_name=decoder_weight_model_name,
                                                  decoder_weight_model_name=None,
                                                  path_result_model=os.path.join(PATH_RESULT, 'model_8_4_1'),
                                                  data_set_name=src_name_j, modelname='1_n3_2',
                                                  paras={'i': i, 'j': j, 'epochs': 2},
                                                  dict_model_load=dict_model_load,
                                                  # data_set_encoder=[read_data(path_out,
                                                  #                             f'{s_name_k}_{weight_file_name}_encoded')
                                                  #                   for s_name_k in s_names_j],
                                                  data_set_encoder=[np.zeros((1, 329, 2))
                                                                             for k in range(len(s_names_j))],
                                                  path_out=path_out,
                                                  # name_data_set_decoded=[f'{s_name_k}_{weight_file_name}_decoded'
                                                  # name_data_set_decoded=[f'{s_name_k}_{weight_file_name}_decoded_later'
                                                  name_data_set_decoded=[f'{s_name_k}_{weight_file_name}_decoded_zero'
                                                                         for s_name_k in s_names_j])
    """
    if decoder_weight_model_name is None:
        decoder_weight_model_name = os.path.join(path_save_model, f'{modelname}_{weight_file_name}_decoder.h5')
        if os.path.isfile(decoder_weight_model_name):
            decoder_weight_model_name = os.path.join(path_save_model,
                                                     f'{modelname}_{weight_file_name}_decoder_later.h5')
    create_decoder_weight_model(decoder_model=decoder_model, path_save_model=path_save_model,
                                auto_model_name=auto_model_name, weight_file_name=weight_file_name,
                                decoder_model_name=decoder_model_name,
                                decoder_weight_model_name=decoder_weight_model_name,
                                path_result_model=path_result_model, data_set_name=data_set_name, modelname=modelname,
                                paras=paras, dict_model_load=dict_model_load)
    decoder_weight_model = load_model(decoder_weight_model_name, custom_objects=dict_model_load)
    test_predict_decoder(data_set_encoder, decoder_weight_model, path_out, name_data_set_decoded,
                         mode_pred='batch_save', bs_pred=32)


def build_model_search(n_outputs, encoded_shape, model_sub_decoders):
    """Build model search encoded vector.
    Args:
        n_outputs (int): number of the outputs.
        encoded_shape (tuple(int)): shape of the encoded vector.
        model_sub_decoders (list[keras.Model]): sub model of the decoders.
    """
    def layer_para_input(inputs, encoded_shape):
        """Trick for trainable input layer through a 1*1 conv layer.
        Args:
            inputs (tensor): keras.Input, the dummy ones input.
            encoded_shape (tuple(int)): shape of the encoded data (encoder vector).
        Returns:
            x (tensor): keras tensor, the real input vecotor of decoder.
        """
        x = inputs
        logging.debug(f'inputs {x}')
        x = Reshape((1, 1))(x)
        logging.debug(f'para input reshape {x}')
        x = Dense(np.prod(encoded_shape), use_bias=False)(x)
        x = Reshape(encoded_shape)(x)
        logging.debug(f'para input reshape after {x}')
        return x

    input_frames = Input(shape=(n_outputs, 1))
    logging.debug(f'input_frames {input_frames}')
    decoded_srcs = []
    rank_input = K.ndim(input_frames)
    for i in range(n_outputs):
        if rank_input == 3:
            x = Lambda(lambda x: x[:, i, :])(input_frames)
        elif K.ndim(input_frames) == 4:
            x = Lambda(lambda x: x[:, i, :, :])(input_frames)
        elif K.ndim(input_frames) == 5:
            x = Lambda(lambda x: x[:, i, :, :, :])(input_frames)
        logging.debug(f'input_frames[i] {x}')
        x = layer_para_input(x, encoded_shape)
        logging.debug(f'para input {x}')
        decoded_src_i = model_sub_decoders[i](x)
        logging.debug(f'decoded_src_i {decoded_src_i}')
        if K.ndim(decoded_src_i) == 2:
            decoded_src_i = Lambda(lambda x: K.expand_dims(x, axis=-2))(decoded_src_i)
            logging.debug(f'decoded_src_i after expand dim {decoded_src_i}')
        decoded_srcs.append(decoded_src_i)
    if n_outputs > 1:
        if K.ndim(decoded_srcs[0]) == 2:
            for i, decoded_src_i in enumerate(decoded_srcs):
                decoded_srcs[i] = Reshape(tuple(K.int_shape(decoded_src_i)[1:])+(1,))(decoded_src_i)
        decoded = keras.layers.Add()(decoded_srcs)
        logging.debug(f'decoded {decoded}')
    else:
        decoded = decoded_srcs[0]
    model_search = Model(input_frames, decoded)
    return model_search


def build_model_1_search(encoded_shape, output_dim, path_weight_decoder_files, dict_model_load=None,
                         n_outputs=1, encoding_dim=2, act_c='relu', n_nodes=[8, 4, 2],
                         batch_norm=False, layer_norm=False, use_bias=True):
    """Build search model with struct model_1.
    Args:
        encoded_shape (tuple(int)): shape of the encoded vector.
        output_dim (int): dim of the output vector.
        path_weight_decoder_files (list[str]): full path name of the decoder models.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
        n_outputs (int, optional): number of output srcs. Defaults to 1.
        encoding_dim (int, optional): para for compute numbers of the neurals. Defaults to 2.
        act_c (str, optional): activation function of the conv layers. Defaults to 'relu'.
        n_nodes (list, optional): para for compute numbers of the neurals. Defaults to [8, 4, 2].
        batch_norm (bool, optional): wether use batch normalizaion layer. Defaults to False.
        layer_norm (bool, optional): wether use layer normalizaion layer. Defaults to False.
        use_bias (bool, optional): wether use bias in all neurals. Defaults to True.
    """
    def sub_decoder(weight_model_name):
        """Create sub decoder model from autodecoder model with trianed weights.
        Args:
            weight_model_name (str): full file name with path of the trained autodecoder model.
        Returns:
            model_decoder (keras.Model): sub decoder model of the autodecoder model.
        """
        inputs = Input(encoded_shape)
        x = inputs
        for i in range(len(n_nodes)-1, -1, -1):
            x = Dense(int(encoding_dim * n_nodes[i]), activation=act_c, use_bias=use_bias)(x)
            if batch_norm:
                x = BatchNormalization()(x)
            if layer_norm:
                x = LayerNormalization(center=False, scale=False)(x)
        decoded = Dense(output_dim, activation='tanh', use_bias=use_bias)(x)
        model_decoder = Model(inputs, decoded)
        logging.debug(f'model_decoder create \n {model_decoder.summary()}')

        model_weight = load_model(weight_model_name, dict_model_load)
        logging.debug(f'model_decoder loaded \n {model_weight.summary()}')
        for i, layer in enumerate(model_weight.layers[1:]):
            model_decoder.layers[i+1].trainable = False
            layer_weights = layer.get_weights()
            model_decoder.layers[i+1].set_weights(layer_weights)
        return model_decoder

    model_sub_decoders = [sub_decoder(name_i) for name_i in path_weight_decoder_files]
    model_search = build_model_search(n_outputs, encoded_shape, model_sub_decoders)
    return model_search


def build_model_8_search(encoded_shape, path_weight_decoder_files, dict_model_load,
                         input_dim, n_pad_input, chunk_size, chunk_advance,
                         n_filters_conv, n_rnn_decoder, rnn_type, units_r, act_r, use_bias,
                         n_outputs):
    """Build search model with struct model_1.
    Args:
        encoded_shape (tuple(int)): shape of the encoded vector.
        path_weight_decoder_files (list[str]): full path name of the decoder models.
        dict_model_load (dict): custom objects for load model. Defaults to None.
        input_dim (int): dim of the input vector.
        n_pad_input (int): network input = input_dim + n_pad_input.
        chunk_size (int): chunk size in the RNN layer.
        chunk_advance (int): strides size in the RNN layer.
        n_filters_conv (int): number of the filters in encoder layer.
        n_rnn_decoder (int): number of the RNN layers.
        rnn_type (str): type of the RNN layer.
        units_r (int): number of the units in a RNN layer.
        act_r (str): activation function in RNN layer.
        use_bias (bool): wether use bias in all neurals.
        n_outputs (int): number of output srcs.
    """
    def rnn_layer(rnn_type, x):
        """RNN layer of the model.
        Args:
            rnn_type (str): type of the RNN layer.
            x (tensor): keras tensor, input of the rnn layer.
        Returns:
            x (tensor): keras tensor, output of the rnn layer.
        """
        if rnn_type == 'LSTM':
            x = LSTM(units_r, use_bias=use_bias, activation=act_r, return_sequences=True)(x)
        if rnn_type == 'BLSTM':
            x = Bidirectional(LSTM(units_r//2, use_bias=use_bias, activation=act_r, return_sequences=True))(x)
        elif rnn_type == 'dprnn':
            frame_length = input_dim + n_pad_input
            n_full_chunks = frame_length // chunk_size
            n_overlapping_chunks = n_full_chunks*2-1
            x = DprnnBlock(is_last_dprnn=False, num_overlapping_chunks=n_overlapping_chunks, chunk_size=chunk_size,
                           num_filters_in_encoder=n_filters_conv, units_per_lstm=units_r, act_r=act_r)(x)
        # logging.debug(f'RNN out {x}')  # (bs, time_step=329, dim_fc)
        return x

    def sub_decoder(weight_model_name):
        """Create sub decoder model from autodecoder model with trianed weights.
        Args:
            weight_model_name (str): full file name with path of the trained autodecoder model.
        Returns:
            model_decoder (keras.Model): sub decoder model of the autodecoder model.
        """
        inputs = Input(encoded_shape)
        x = inputs
        for i in range(n_rnn_decoder):
            x = rnn_layer(rnn_type, x)
        logging.debug(f'rnn out {x}')  # (32, 329, units) or (32, 329, units, n_filters)
        if rnn_type == 'dprnn':
            x = Reshape(tuple(K.int_shape(x)[1:-2]+(-1,)))(x)

        x = Dense(units=chunk_size, use_bias=use_bias)(x)
        x = LayerNormalization(center=False, scale=False)(x)
        x = Lambda(lambda x: overlap_and_add_in_decoder(x, chunk_advance))(x)
        x = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
        decoded = Lambda(lambda x: x[:, 0:input_dim, :])(x)

        model_decoder = Model(inputs, decoded)
        logging.debug(f'model_decoder create \n {model_decoder.summary()}')

        model_weight = load_model(weight_model_name, dict_model_load)
        logging.debug(f'model_decoder loaded \n {model_weight.summary()}')
        for i, layer in enumerate(model_weight.layers[1:]):
            model_decoder.layers[i+1].trainable = False
            layer_weights = layer.get_weights()
            model_decoder.layers[i+1].set_weights(layer_weights)
        return model_decoder

    model_sub_decoders = [sub_decoder(name_i) for name_i in path_weight_decoder_files]
    model_search = build_model_search(n_outputs, encoded_shape, model_sub_decoders)
    return model_search


def train_model_search(z_dict, paras, path_save, modelname=None,
                       path_model_name=None, dict_model_load=None, **kwargs):
    """Train model to search best decoder input (max likehood separation).
    Args:
        z_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model output, data to predict,
            sum of the outputs, which is the given input vector in separation problem.
        paras (dict): parameters for train model.
        path_save (str): where to save model.
        modelname (str, optional): model name for models. Defaults to None.
        path_model_name (str, optional): full file name with path of the model to train. Defaults to None.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
    """
    batch_size = paras['batch_size']
    epochs = paras['epochs']
    if modelname is None:
        lr_i = paras['lr_i']
        lr_j = paras['lr_j']
        strj = f'n{str(lr_j)[1:]}' if lr_j < 0 else str(lr_j)
        modelname = f'{lr_i}_{strj}_{epochs}'
    if 'loss_func' in paras.keys():
        loss_func = paras['loss_func']
    else:
        loss_func = keras.losses.mean_squared_error
        # loss_func = 'MSE'
        # loss_func = 'binary_crossentropy'
    metrics_func = paras['metrics_func'] if 'metrics_func' in paras.keys() else [samerate_acc_d2]
    metrics_name = paras['metrics_name'] if 'metrics_name' in paras.keys() else ['samerate_acc_d2']
    optimizer_type = paras['optimizer'] if 'optimizer' in paras.keys() else 'adm'
    learn_rate = paras['learn_rate'] if 'learn_rate' in paras.keys() else paras['lr_i']*(10**paras['lr_j'])
    bool_clean_weight_file = kwargs['bool_clean_weight_file'] if 'bool_clean_weight_file' in kwargs.keys() else True

    z_name = next(iter(z_dict.keys()))
    z_train = next(iter(z_dict.values()))

    logging.info('start train autoencoder model')
    path_board = os.path.join(path_save, 'tensorbord_search', modelname)
    mkdir(path_board)
    path_check = os.path.join(path_save, 'search_model', z_name, modelname)
    mkdir(path_check)
    for num_j, z_train_j in enumerate(z_train):
        str_num_j = '0'*(5-len(str(num_j)))+str(num_j)
        path_check_j = os.path.join(path_check, f'num_{str_num_j}')
        mkdir(path_check_j)

        model = load_model(path_model_name, custom_objects=dict_model_load, compile=False)
        if optimizer_type == 'adm':
            optimizer = optimizers.Adam(lr=learn_rate)
        elif optimizer_type == 'sgd':
            optimizer = optimizers.SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)

        z_train_j = np.expand_dims(np.asarray(z_train_j), axis=0)
        dummy_input = np.ones(tuple(K.int_shape(model.inputs[0])[1:2])+(1,))
        x_train_j = np.expand_dims(np.asarray(dummy_input), axis=0)  # (1, n_outputs, 1)

        model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics_func)
        check_filename = os.path.join(path_check_j, f'weights_{modelname}'+'_{epoch:02d}_{loss:.2f}.hdf5')
        checkpoint = ModelCheckpoint(filepath=check_filename, monitor='loss', mode='auto',
                                     verbose=1, period=1, save_best_only=True)
        history = model.fit(x=x_train_j,
                            y=z_train_j,
                            epochs=epochs,
                            batch_size=batch_size,
                            # callbacks=[TensorBoard(log_dir=path_board), checkpoint])  # huge space
                            callbacks=[checkpoint])

        path_save_history = os.path.join(path_save, 'loss_search')
        mkdir(path_save_history)
        logging.debug(history.history.keys())
        output_history(
            colors=['r'],
            y_out=[history.history['loss']],
            savename=os.path.join(path_save_history, f'{z_name}_{num_j}_loss_{modelname}.svg'),
            show=False, title='loss',
            label_x='epoch', label_y='loss', loc='upper left')
        output_history(
            colors=['b'],
            y_out=[history.history[metrics_name[0]]],
            savename=os.path.join(path_save_history, f'{z_name}_{num_j}_{metrics_name[0]}_{modelname}.svg'),
            show=False, title=f'{metrics_name[0]}',
            label_x='epoch', label_y='accuracy', loc='upper left')
        if bool_clean_weight_file:
            clear_model_weight_file(path_check_j)

        del z_train_j
        del dummy_input
        del x_train_j
        del model
        del optimizer
        gc.collect()
        K.clear_session()


def load_weights_from_checkfile(fp, layer_names):
    """Implements topological (order-based) weight loading.
    Args:
        fp (h5py.File): A pointer to a HDF5 group.
        layer_names (list[str]): List of the names of the layers.d
    Returns:
        weight_values (list[tensor]): weight values of the layers.
    """
    weight_group = fp['model_weights']
    weight_values = []
    for layer_name in layer_names:
        layer_key = next(iter(weight_group[layer_name].keys()))
        weight_values.append(np.asarray(weight_group[layer_name][layer_key]['kernel:0']))
    return weight_values


def predict_model_search_decoder(path_search_model_weight, path_predict, dataset_name, n_samples, search_model,
                                 decoder_weight_model_names, dict_model_load,
                                 weight_layer_index=None, bs_pred=8):
    """Give search encoded vector model input which is the encoder vector, predict model output.
    Args:
        path_search_model_weight (str): full file name with path of the search encoded vector model.
        path_predict (str): path of the dir where to save predict output.
        dataset_name (str): name of the dataset to predict.
        n_samples (int): number of the samples to predict.
        search_model (keras.Model): the search encoded vector model.
        decoder_weight_model_names (list[str]): full path name of the decoder models.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
        weight_layer_index (list[int], optional): index of the layers to load weights. Defaults to None.
        bs_pred (int, optional): batch size of the samples when predict. Defaults to 8.
    """
    fname_dense_weight = os.path.join(path_predict, f'{dataset_name}_dense_weight.hdf5')
    if not os.path.isfile(fname_dense_weight):
        path_samples = list_dirs_start_str(path_search_model_weight, 'num_')
        weight_values_nums = []  # (src, nsams)+(weight_shape)
        for path_num_i in path_samples:
            path_weight_file = list_files_end_str(path_num_i, '.hdf5')[0]
            name_weight_layers = [layer.name for layer in search_model.layers]
            if weight_layer_index:
                name_weight_layers = [name_weight_layers[layer_j] for layer_j in weight_layer_index]
            weight_values_layers_i = load_weights_from_checkfile(h5py.File(path_weight_file), name_weight_layers)
            weight_values_layers_i = [np.expand_dims(weight_values_layers_i_j, axis=0
                                                     ) for weight_values_layers_i_j in weight_values_layers_i]
            weight_values_layers_i = np.concatenate(weight_values_layers_i, axis=0)  # (n_src, weight_shape)
            weight_values_nums.append(np.expand_dims(weight_values_layers_i, axis=1))
            shutil.rmtree(path_num_i)
        weight_values_nums = np.concatenate(weight_values_nums, axis=1)  # (n_src, n_samples, weight_shape)
        save_datas({f'{dataset_name}_dense_weight': weight_values_nums}, path_predict)

    weight_values_nums = read_data(path_predict, f'{dataset_name}_dense_weight')
    name_predict_srcs = []
    for src_i, decoder_weight_model_name_i in enumerate(decoder_weight_model_names):
        weight_values_i = weight_values_nums[src_i]
        decoder_model_i = load_model(decoder_weight_model_name_i, dict_model_load)
        name_encoded_src_i = f'Z_{src_i}_zero_{dataset_name[2:]}_encoded'
        if not os.path.isfile(os.path.join(path_predict, name_encoded_src_i)):
            encoded_samples_shape = (n_samples,) + tuple(K.int_shape(decoder_model_i.inputs[0])[1:])
            if not tuple(weight_values_i.shape) == encoded_samples_shape:
                encoded_src_i = np.reshape(weight_values_i, encoded_samples_shape)
            save_datas(dict({name_encoded_src_i: encoded_src_i}), path_predict)
        encoded_src_i = read_data(path_predict, name_encoded_src_i)

        predict_autoencoder(decoder_model_i, encoded_src_i, path_predict, f'Z_{src_i}_zero_{dataset_name[2:]}_decoded',
                            compile_model=True, bs_pred=bs_pred)
        name_predict_srcs.append(f'Z_{src_i}_zero_{dataset_name[2:]}_decoded')
        del weight_values_i
        del decoder_model_i
        del encoded_src_i
        gc.collect()
    z_predict = np.concatenate(np.asarray(read_datas(path_predict, name_predict_srcs)), axis=-1)
    save_datas({f'Z_zero_{dataset_name[2:]}_decoded': z_predict}, path_predict)


def transpose_names_to_para_src(path_weight_files):
    """Transpose list names to [para][src].
    Args:
        path_weight_files (list(list(list(str)))): [src][para][weights], list of strs,
            all src should have same para names, different weight file names.
    Returns:
        path_weight_files_paras (list(list(str))): [para][src], list of names.
    """
    num_src = len(path_weight_files)
    num_name_paras = len(path_weight_files[0])
    path_weight_files_paras = []  # [para][src]
    for i_para in range(num_name_paras):
        path_weight_files_src_j = []
        for j_src in range(num_src):
            path_weight_files_src_j.append(path_weight_files[j_src][i_para][-1])
        path_weight_files_paras.append(path_weight_files_src_j)
    path_weight_files_paras.append(path_weight_files_src_j)
    return path_weight_files_paras


def search_best_model(path_result_root, src_names,
                      model_name, encoded_shape, output_dim, z_dict, model_para_name=None, **kwargs):
    """For search best model.
    Args:
        path_result_root (str): where to save result.
        src_names (list[str]): list of src names, e.g. ['Z_0_train',...,'Z_7_ns_test'].
        model_name (str): name of the model.
        encoded_shape (tuple(int)): shape of the encoded vector.
        output_dim (int): shape of output tensor, length of the sample.
        z_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model output, data to predict,
            sum of the outputs, which is the given input vector in separation problem.
        model_para_name (str, optional): for model name. Defaults to None.
    """
    path_result_model = os.path.join(path_result_root, model_name)
    path_save_search_decoder = os.path.join(path_result_model, 'search_decoder')
    mkdir(path_save_search_decoder)

    num_model = compute_num_model(model_name)
    dict_model_load = build_dict_model_load(num_model, **kwargs)

    z_name = next(iter(z_dict.keys()))
    z_data = next(iter(z_dict.values()))

    object_decoded_files = ListDecodedFiles(path_result_root, src_names)
    path_weight_files = object_decoded_files.filter_model(model_name, 'path_weight_files')  # [src][para][weights]
    name_model_src_paras = object_decoded_files.filter_model(model_name, 'name_model_src_paras')  # [src][para]

    name_paras = name_model_src_paras[0]  # [para]
    path_weight_files_paras = transpose_names_to_para_src(path_weight_files)  # [para][src]
    for name_para_i, path_weight_files_para_i in zip(name_paras, path_weight_files_paras):
        decoder_weight_model_names_i = []
        for path_weight_files_para_i_j in path_weight_files_para_i:  # [src]
            path_decoder_model_j, weight_file_name_j = os.path.split(path_weight_files_para_i_j)
            decoder_weight_model_name_j = os.path.join(path_decoder_model_j,
                                                       f'{name_para_i}_{weight_file_name_j}_decoder.h5')
            if not os.path.isfile(decoder_weight_model_name_j):
                create_decoder_weight_model(path_save_model=path_decoder_model_j, modelname=name_para_i,
                                            dict_model_load=dict_model_load)
            decoder_weight_model_names_i.append(decoder_weight_model_name_j)

        path_save_search_decoder_para_i = os.path.join(path_save_search_decoder, name_para_i)
        mkdir(path_save_search_decoder_para_i)
        path_search_model_i = os.path.join(path_save_search_decoder_para_i, 'search_model')
        mkdir(path_search_model_i)
        search_model_name_i = os.path.join(path_search_model_i, f'{name_para_i}_search.h5')

        if num_model in (8,):
            input_dim = kwargs['input_dim'] if 'input_dim' in kwargs.keys() else 10547
            n_pad_input = kwargs['n_pad_input'] if 'n_pad_input' in kwargs.keys() else 13
            chunk_size = kwargs['chunk_size'] if 'chunk_size' in kwargs.keys() else 64
            chunk_advance = kwargs['chunk_advance'] if 'chunk_advance' in kwargs.keys() else chunk_size // 2
            n_rnn_decoder = kwargs['n_rnn_decoder'] if 'n_rnn_decoder' in kwargs.keys() else 1
            n_filters_conv = kwargs['n_filters_conv'] if 'n_filters_conv' in kwargs.keys() else 1
            rnn_type = kwargs['rnn_type'] if 'rnn_type' in kwargs.keys() else 'LSTM'
            units_r = kwargs['units_r'] if 'units_r' in kwargs.keys() else 2
            act_r = kwargs['act_r'] if 'act_r' in kwargs.keys() else 'tanh'
            use_bias = kwargs['use_bias'] if 'use_bias' in kwargs.keys() else True
            n_outputs = kwargs['n_outputs'] if 'n_outputs' in kwargs.keys() else 4

        if os.path.isfile(search_model_name_i):
            search_model = load_model(search_model_name_i, custom_objects=dict_model_load)
        elif num_model == 1:
            n_outputs = kwargs['n_outputs'] if 'n_outputs' in kwargs.keys() else 4
            encoding_dim = kwargs['encoding_dim'] if 'encoding_dim' in kwargs.keys() else 2
            act_c = kwargs['act_c'] if 'act_c' in kwargs.keys() else 'relu'
            n_nodes = kwargs['n_nodes'] if 'n_nodes' in kwargs.keys() else [8, 4, 2]
            batch_norm = kwargs['batch_norm'] if 'batch_norm' in kwargs.keys() else False
            use_bias = kwargs['use_bias'] if 'use_bias' in kwargs.keys() else True
            search_model = build_model_1_search(encoded_shape, output_dim, decoder_weight_model_names_i,
                                                dict_model_load=dict_model_load,
                                                n_outputs=n_outputs, encoding_dim=encoding_dim, act_c=act_c,
                                                n_nodes=n_nodes,
                                                batch_norm=batch_norm, use_bias=use_bias)
            search_model.save(search_model_name_i)
            save_model_struct(search_model, path_search_model_i, 'search_model_struct')
        elif num_model in (8,):
            search_model = build_model_8_search(encoded_shape, decoder_weight_model_names_i, dict_model_load,
                                                input_dim, n_pad_input, chunk_size, chunk_advance,
                                                n_filters_conv, n_rnn_decoder, rnn_type, units_r, act_r, use_bias,
                                                n_outputs)
            search_model.save(search_model_name_i)
            save_model_struct(search_model, path_search_model_i, 'search_model_struct')

        lr_i = kwargs['lr_i'] if 'lr_i' in kwargs.keys() else None
        lr_j = kwargs['lr_j'] if 'lr_j' in kwargs.keys() else None
        epochs = kwargs['epochs'] if 'epochs' in kwargs.keys() else 2
        model_para_name = kwargs['model_para_name'] if 'model_para_name' in kwargs.keys() else None
        if model_para_name is None:
            strj = f'n{str(lr_j)[1:]}' if lr_j < 0 else str(lr_j)
            model_para_name = f'{lr_i}_{strj}_{epochs}'
        bool_train = kwargs['bool_train'] if 'bool_train' in kwargs.keys() else True
        if bool_train:
            batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 32
            paras = {'lr_i': lr_i, 'lr_j': lr_j, 'epochs': epochs, 'batch_size': batch_size}
            if num_model in (8,) and rnn_type == 'dprnn':
                if 'n_overlapping_chunks' in kwargs.keys():
                    n_overlapping_chunks = kwargs['n_overlapping_chunks']
                else:
                    n_full_chunks = (input_dim + n_pad_input) // chunk_size
                    n_overlapping_chunks = n_full_chunks*2-1
                dict_dprnn = {'DprnnBlock': DprnnBlock(is_last_dprnn=False,
                                                       num_overlapping_chunks=n_overlapping_chunks,
                                                       chunk_size=chunk_size,
                                                       num_filters_in_encoder=n_filters_conv,
                                                       units_per_lstm=units_r,
                                                       act_r=act_r,
                                                       use_bias=use_bias)}
                paras.update(**dict_dprnn)

            bool_clean_weight_file = kwargs['bool_clean_weight_file'] if ('bool_clean_weight_file' in kwargs.keys()
                                                                          ) else True
            train_model_search(z_dict, paras, path_save_search_decoder_para_i, modelname=model_para_name,
                               path_model_name=search_model_name_i, dict_model_load=dict_model_load,
                               **{'bool_clean_weight_file': bool_clean_weight_file})

        bool_predict_weight_file = kwargs['bool_predict_weight_file'] if ('bool_predict_weight_file' in kwargs.keys()
                                                                          ) else True
        bs_pred = kwargs['bs_pred'] if 'bs_pred' in kwargs.keys() else 8
        if bool_predict_weight_file:
            if num_model in {1, 8}:
                weight_layer_index = list(range(9, 13))

            path_save_search_decoder_para_j = os.path.join(path_save_search_decoder_para_i, 'search_model',
                                                           z_name, model_para_name)
            path_predict_j = os.path.join(path_save_search_decoder_para_i, 'predict_decoder', model_para_name)
            mkdir(path_predict_j)
            predict_model_search_decoder(path_save_search_decoder_para_j, path_predict_j, z_name, z_data.shape[0],
                                         search_model, decoder_weight_model_names_i, dict_model_load,
                                         weight_layer_index=weight_layer_index, bs_pred=bs_pred)


if __name__ == '__main__':
    from loss_acc_separation import samerate_acc_d2
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO)

    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    np.random.seed(1337)  # for reproducibility
    # The below tf.set_random_seed() will make random number generation in the
    # TensorFlow backend have a well-defined initial state. For further details,
    # see: https://www.tensorflow.org/api_docs/python/tf/set_random_seed
    tf.set_random_seed(1234)
    # # Force TensorFlow to use single thread. Multiple threads are a potential
    # # source of non-reproducible results. For further details,
    # # see: https://stackoverflow.com/questions/42022950/
    # SESSION_CONF = tf.ConfigProto(
    #     intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    SESSION_CONF = tf.ConfigProto()

    # Limiting GPU memory growth (forbidden GPU OOM)
    SESSION_CONF.gpu_options.allow_growth = True
    # SESSION_CONF.gpu_options.per_process_gpu_memory_fraction = 0.3

    SESS = tf.Session(graph=tf.get_default_graph(), config=SESSION_CONF)
    K.set_session(SESS)

    # for shipsear data
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    SCALER_DATA = 'max_one'
    # SCALER_DATA = 'or'
    SUB_SET_WAY = 'rand'
    # SUB_SET_WAY = 'order'

    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav', scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source

    SET_NAMES = ['train', 'val', 'test'][2:3]   # you may only run on test set
    S_NAMES = []  # [n_source][n_set]
    for i in range(4):
        S_NAMES.append([f'Z_{i}_ns_{name_set_j}' for name_set_j in SET_NAMES])

    X_NAMES = [f'X_{name_set_j}' for name_set_j in SET_NAMES]
    X_DICT = dict(zip(X_NAMES, read_datas(PATH_DATA, X_NAMES)))

    SRC_NAMES = [f'Z_{i}_ns' for i in range(4)]  # [n_src]

    PATH_RESULT = '../result_separation_ae_ns_single'
    mkdir(PATH_RESULT)

    lr_i_j = [(i, j) for i in range(1, 2) for j in range(-2, -4, -1)]
    for (lr_i, lr_j) in lr_i_j:
        for x_name_j, x_data_j in X_DICT.items():
            # ------------------------------------------------------------------------------------------------------- #
            # x_data_j = x_data_j[0:200]
            # # If the parameter is too large and out of memory, you may train samples by batches.
            # # Then manually create and orgnize the files such as follow:
            # # ./search_decoder/1_n3_100
            # # ??????1_n1_100_test_sets
            # #   ??????test_0_4000
            # #   | ??????loss_search
            # #   | |   X_test_0_loss_1_n1_100.svg
            # #   | |   ...
            # #   | |   X_test_4000_loss_1_n1_100.svg
            # #   | ??????predict_decoder
            # #   |   ??????1_n1_100
            # #   |       X_test_dense_weight.hdf5 shape==(4,4000,1,1,84224)
            # #   |       Z_zero_test_decoded.hdf5 shape==(4000,10547,4)
            # #   |       Z_0_zero_test_decoded.hdf5 shape==(4000,10547,1)
            # #   |       Z_0_zero_test_encoded.hdf5
            # #   |       ...
            # #   |       Z_3_zero_test_decoded.hdf5 shape==(4000,10547,1)
            # #   |       Z_3_zero_test_encoded.hdf5
            # #   ??????test_4000_8160
            # #     ??????loss_search
            # #     |   X_test_0_loss_1_n1_100.svg
            # #     |   ...
            # #     |   X_test_4160_loss_1_n1_100.svg
            # #     ??????predict_decoder
            # #       ??????1_n1_100
            # #           X_test_dense_weight.hdf5 shape==(4,4160,1,1,84224)
            # #           Z_zero_test_decoded.hdf5 shape==(4160,10547,4)
            # #           Z_0_zero_test_decoded.hdf5 shape==(4160,10547,1)
            # #           Z_0_zero_test_encoded.hdf5
            # #           ...
            # #           Z_3_zero_test_decoded.hdf5 shape==(4160,10547,1)
            # #           Z_3_zero_test_encoded.hdf5
            # #
            # # Then concatenate the files with module see_metric_separation_search_encoded.py
            # ------------------------------------------------------------------------------------------------------- #
            x_dict_j = dict({x_name_j: x_data_j})
            input_dim = x_data_j.shape[-1]  # (nsamples, 1, frame_length)

            # search_best_model(PATH_RESULT, SRC_NAMES,
            #              'model_1_2_1', (256,), input_dim, x_dict_j,
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 100,
            #                  'encoding_dim': 256, 'batch_norm': True,
            #                  'bool_train': True, 'bool_predict_weight_file': True})

            # search_best_model(PATH_RESULT, SRC_NAMES,
            #              'model_1_2_1_5', (256,), input_dim, x_dict_j,
            #               **{'lr_i': lr_i, 'lr_j': lr_j, 'epochs': 100,
            #                  'encoding_dim': 256, 'layer_norm': True,
            #                  'bool_train': True, 'bool_predict_weight_file': True})

            for key, value in x_dict_j.items():
                x_dict_j[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (nsamples, frame_length, 1)

            # search_best_model(PATH_RESULT, SRC_NAMES,
            #              'model_8_1_1', (329, 256), input_dim, x_dict_j,
            #              **{'lr_i': lr_i, 'lr_j': lr_j, 'epochs': 100,
            #                 'rnn_type': 'LSTM', 'units_r': 256, 'n_rnn_decoder': 1, 'n_filters_conv': 64,
            #                 'bool_train': True, 'bool_predict_weight_file': True})

            search_best_model(PATH_RESULT, SRC_NAMES,
                         'model_8_2_1', (329, 256), input_dim, x_dict_j,
                         **{'lr_i': lr_i, 'lr_j': lr_j, 'epochs': 100,
                            'rnn_type': 'BLSTM', 'units_r': 256, 'n_rnn_decoder': 1, 'n_filters_conv': 64,
                            'bool_train': True, 'bool_predict_weight_file': True})

            # search_best_model(PATH_RESULT, SRC_NAMES,
            #              'model_8_4_1', (329, 256), input_dim, x_dict_j,
            #              **{'lr_i': lr_i, 'lr_j': lr_j, 'epochs': 100,
            #                 'rnn_type': 'BLSTM', 'units_r': 256, 'n_rnn_decoder': 1, 'n_filters_conv': 64,
            #                 'use_bias': False,
            #                 'bool_train': True, 'bool_predict_weight_file': True})

    logging.info('finished')
