# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:34:29 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation
# pylint: disable=no-member, unused-import, ungrouped-imports, wrong-import-position
# pylint: disable=too-many-arguments, too-many-branches, too-many-lines, too-many-locals
# pylint: disable=too-many-instance-attributes, too-many-statements

import gc
import logging
import os

import keras.losses
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model, Model
from keras import backend as K
from keras import optimizers
import numpy as np
if keras.__version__ >= '2.3.1':
    from keras.layers import LayerNormalization
else:
    from keras_layer_normalization import LayerNormalization
import tensorflow as tf

from error import Error, ParameterError
from file_operation import list_dirs_start_str, list_files_end_str, mkdir, walk_files_end_str
from loss_acc_separation import samerate_acc_d2, vae_loss
from model_functions import part_front_model, part_back_model
from prepare_data_shipsear_recognition_mix_s0tos3 import data_save_reshape
from prepare_data_shipsear_recognition_mix_s0tos3 import save_process_batch
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data, read_datas, save_datas
from separation_mix_shipsear_s0tos3_preprocess import sourceframes_mo_create
from train_functions import output_history, save_model_struct
from models.models_autoencoder import segment_encoded_signal, overlap_and_add_in_decoder, DprnnBlock
from models.models_autoencoder import build_model_1, build_model_2, build_model_3
from models.models_autoencoder import build_model_4, build_model_5, build_model_6
from models.models_autoencoder import build_model_7, build_model_8


def data_save_reshape_ae(data):
    """Reshape data to last dim is not 1, when shape ndim > 2.
    Args:
        data (np.ndarray): data to reshape.
    Returns:
        data (np.ndarray): data after reshape.
    """
    if data.ndim == 2:
        return data
    return data_save_reshape(data)


def z_sets_ns_create(s_list, nums, scaler_data, path_save):
    """Create train, val, test data set through nums.
    Inputs:
        s_list (list[np.ndarray, shape==(n_sams, 1, frame_length)]): [n_src] samples of sources.
        scaler_data (str): way of scaler data.
        nums (list[int]): [n_sams], index of the samples per set of sources.
        path_save (str): path where to save z data sets.
    Returns:
        data_sources(list[list[np.ndarray, shape==(n_sams, 1, frame_length)]]): [n_src][set], data sets of sources.
    """
    def z_set_ns_create(source_arr, nums, set_names=None, save_name=None):
        """Create train, val, test data set through index nums.
        Inputs:
            source_arr (np.ndarray, shape==(n_sams, 1, frame_length)): samples of a source.
            nums (list[list[int]]): [n_sets][n_sams], index of the samples per set.
            set_names (list[str], optional): names of the sets of data. Default None.
            save_name (str): path+file_name to save data to files.
        Returns:
            source_sets(list[np.ndarray, shape==(n_sams, 1, frame_length)]): [set], data sets.
        """
        if set_names is None:
            set_names = ['train', 'val', 'test']

        source_sets = []
        for nums_set_i in nums:
            source_set_i = [source_arr[nums_i_j] for nums_i_j in nums_set_i]
            source_set_i = np.asarray(source_set_i)
            source_sets.append(source_set_i)

        if save_name is not None:
            path_save, file_name = os.path.split(save_name)
            for name_i, data_i in zip(set_names, source_sets):
                save_datas({f'{file_name}_{name_i}': data_i}, path_save)
        return source_sets  # [set](n_sams, 1, frame_length)

    if scaler_data == 'max_one':
        s_list = sourceframes_mo_create(s_list)

    data_sources = []
    for i, s_ci in enumerate(s_list):
        data_sources.append(
            z_set_ns_create(s_ci, nums, save_name=os.path.join(path_save, f'Z_{i}_ns')))

    return data_sources  # [n_src][set](n_sams, 1, frame_length)


def subset_seq(seq, n_sams):
    """Split index seqs to different sets.
    Args:
        seq (list[int]): index of the source.
        n_sams (list[int]): numbers of the sets.
    Returns:
        nums list[list[int]]: index of different sets.
    """
    nums = []
    i = 0
    for n_sam in n_sams:
        nums.append(seq[i:i+n_sam])
        i += n_sam

    return nums


def predict_autoencoder(model, z_test, path_save, save_name, mode='batch_process', bs_pred=32, compile_model=False,
                        reshape_save=False):
    """Predict encoder and decoder.
    Args:
        model (keras.Model): a keras model, encoder or decoder.
        z_test (np.ndarray(float),shape=(n_sams,1,fl)): model input, data to predict.
        path_save (str): where to save predict outputs.
        save_name (str]): path and name of the output file.
        mode (str, optional): {'batch_process', 'batch_save'}, way to predict and save data.
                             Defaults to 'batch_process'.
        bs_pred (int, optional): batch size when predict model. Defaults to 32.
        compile_model (bool, optional): wether compile model before predict,
                                        use when load a unconpiled model. Defaults to False.
        reshape_save (bool, optional): wether reshape data when save data. Defaults to False.
    """
    if compile_model:
        optimizer = optimizers.Adam(lr=1e-3)
        loss = 'MSE'
        metrics = [samerate_acc_d2]
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def func_predict(data):
        """Predict function.
        Args:
            data (np.ndarray(float)): data to be predict.
        Returns:
            data_pred (np.ndarray(float)): model predict output of data.
        """
        shape_input = tuple(K.int_shape(model.inputs[0])[1:])
        if data.shape[1:] != shape_input:
            data = data.reshape(data.shape[0:1]+shape_input)
        data_pred = model.predict(data, batch_size=bs_pred)
        if reshape_save:
            data_pred = data_save_reshape_ae(data_pred)
        return data_pred

    if mode == 'batch_process':
        save_process_batch(z_test, func_predict, path_save, save_name, bs_pred, mode_batch='batch_h5py')
    elif mode == 'batch_save':
        if z_test.shape[1:] != tuple(K.int_shape(model.inputs[0])[1:]):
            z_test = np.asarray(z_test).reshape(z_test.shape[0:1]+tuple(K.int_shape(model.inputs[0])[1:]))
        z_test_pred = model.predict(z_test, batch_size=bs_pred)
        if reshape_save:
            z_test_pred = data_save_reshape_ae(z_test_pred)
        save_datas({save_name: z_test_pred}, path_save, **{'mode_batch': 'batch_h5py'})


def test_autoencoder(x_dict, z_names=None, modelname=None, paras=None,
                     path_save_model=None, model_para=None, path_save=None,
                     auto_model_name=None, path_out=None, dict_model_load=None,
                     weight_file_name=None):
    """Load and test autoencoder.
    Args:
        x_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model input, data to predict.
        z_names (list[str]): names of the dataset. Defaults to None.
        modelname (str, optional): for autoencoder model name. Defaults to None.
        paras (dict, optional): parameters for train model. Defaults to None.
        path_save_model (str, optional): path of the autoencoder model. Defaults to None.
        model_para (str, optional): para of the model. Defaults to None.
        path_save (str, optional): where to save autoencoder output predict. Defaults to None.
        auto_model_name (str, optional): autoencoder model name. Defaults to None.
        path_out (str, optional): path to save predict output. Defaults to None.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
        weight_file_name (str, optional): full name of the weight file name. Defaults to None.
    """
    if z_names is None:
        z_names = list(x_dict.keys())
    if modelname is None:
        i = paras['i']
        j = paras['j']
        epochs = paras['epochs']
        strj = f'n{str(j)[1:]}' if j < 0 else str(j)
        modelname = f'{i}_{strj}_{epochs}'
    if path_save_model is None:
        if model_para is None:
            model_para = z_names[0][:-6]
        path_save_model = os.path.join(path_save, 'auto_model', model_para, modelname)
    if auto_model_name is None:
        auto_model_name = os.path.join(path_save_model, f'{modelname}_auto.h5')

    model = load_model(auto_model_name, custom_objects=dict_model_load)
    if weight_file_name is not None:
        model.load_weights(weight_file_name)

    if path_out is None:
        path_out = os.path.join(path_save, 'auto_out', modelname)
        mkdir(path_out)

    mode_pred = paras['mode_pred'] if 'mode_pred' in paras.keys() else 'batch_process'
    bs_pred = paras['bs_pred'] if 'bs_pred' in paras.keys() else 32
    for x_dataset, z_set_name in zip(x_dict.values(), z_names):
        save_name = f'{z_set_name}_autodecoded'
        if weight_file_name is not None:
            save_name += f'_{os.path.split(weight_file_name)[-1]}'
        predict_autoencoder(model, x_dataset, path_out, save_name, mode=mode_pred, bs_pred=bs_pred)


def train_autoencoder(autoencoder, paras, x_dict, z_dict, path_save, modelname=None,
                      dict_model_load=None, fname_model_load=None):
    """Train autoencoder model.
    Args:
        autoencoder (keras.Model): a keras autoencoder model, no use when dict_model_load is not None.
        paras (dict): parameters for train model.
        x_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model input, data to predict.
        z_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model output, data predict target.
        path_save (str): where to save autoencoder output predict.
        modelname (str, optional): for autoencoder model name. Defaults to None.
    """
    if fname_model_load is None:
        # give a new moel for the first time training
        model = autoencoder
        if 'optimizer' in paras.keys():
            optimizer = paras['paras']
        else:
            learn_rate = paras['learn_rate'] if 'learn_rate' in paras.keys() else paras['i']*(10**paras['j'])
            optimizer_type = paras['optimizer_type'] if 'optimizer_type' in paras.keys() else 'adm'
            if optimizer_type == 'adm':
                optimizer = optimizers.Adam(lr=learn_rate)
            elif optimizer_type == 'sgd':
                optimizer = optimizers.SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        # laod a pre-trained model and continue train
        if dict_model_load is None:
            model = load_model(fname_model_load)
        else:
            model = load_model(fname_model_load, custom_objects=dict_model_load)
        optimizer = model.optimizer

    epochs = paras['epochs']
    if modelname is None:
        lr_i = paras['i']
        lr_j = paras['j']
        strj = f'n{str(lr_j)[1:]}' if lr_j < 0 else str(lr_j)
        modelname = f'{lr_i}_{strj}_{epochs}'

    if 'loss_func' in paras.keys():
        loss_func = paras['loss_func']
    else:
        loss_func = keras.losses.mean_squared_error
        # loss_func = 'MSE'
        # loss_func = 'mean_squared_logarithmic_error'
        # loss_func = 'binary_crossentropy'
        # loss_func = samerate_loss

    metrics_func = paras['metrics_func'] if 'metrics_func' in paras.keys() else [samerate_acc_d2]
    metrics_name = paras['metrics_name'] if 'metrics_name' in paras.keys() else ['samerate_acc_d2']

    model.compile(optimizer=optimizer, loss=loss_func, metrics=metrics_func)

    x_train, x_val, _ = x_dict.values()
    _, z_val_name, _ = z_dict.keys()
    z_train, z_val, _ = z_dict.values()

    path_check = os.path.join(path_save, 'auto_model', z_val_name[:-4], modelname)
    mkdir(path_check)
    check_filename = os.path.join(path_check, f'weights_{modelname}'+'_{epoch:02d}_{val_loss:.2f}.hdf5')
    checkpoint = ModelCheckpoint(filepath=check_filename, monitor='val_loss', mode='auto',
                                 verbose=1, period=1, save_best_only=True)
    path_board = os.path.join(path_save, 'tensorbord', modelname)
    mkdir(path_board)

    batch_size = paras['batch_size']
    shuffle = paras['shuffle'] if 'shuffle' in paras.keys() else True
    logging.info('start train autoencoder model')
    history = model.fit(x=x_train,
                        y=z_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        validation_data=(x_val, z_val),
                        callbacks=[TensorBoard(log_dir=path_board), checkpoint])

    path_save_model = os.path.join(path_save, 'auto_model', z_val_name[:-4], modelname)
    mkdir(path_save_model)

    save_model_struct(model, path_save_model, 'auto_model_struct')

    model.save(os.path.join(path_save_model, f'{modelname}_auto.h5'))

    path_save_history = os.path.join(path_save, 'loss')
    mkdir(path_save_history)

    logging.debug(history.history.keys())
    output_history(
        colors=['r', 'g'],
        y_out=[history.history['loss'], history.history['val_loss']],
        savename=os.path.join(path_save_history, f'{z_val_name[:-3]}loss_{modelname}.svg'),
        show=False, title='loss val_loss',
        label_x='epoch', label_y='loss', loc='upper left')

    output_history(
        colors=['b', 'y'],
        y_out=[history.history[metrics_name[0]], history.history[f'val_{metrics_name[0]}']],
        savename=os.path.join(path_save_history, f'{z_val_name[:-3]}{metrics_name[0]}_{modelname}.svg'),
        show=False, title=f'{metrics_name[0]} val_{metrics_name[0]}',
        label_x='epoch', label_y='accuracy', loc='upper left')


def save_encoder_decoder_layers(n_encoder_layer, encoder_input, n_decoder_layer, decoder_input, path_save_model=None,
                                auto_model_name=None, encoder_model_name=None, decoder_model_name=None,
                                path_save=None, data_set_name=None, modelname=None, paras=None, dict_model_load=None):
    """Split and save encoder and decoder model from autoencoder model with weights, through number of layers.
    Args:
        n_encoder_layer (int): number of layers of encoder model.
        encoder_input (keras.layers.Input): input tensor of encoder.
        n_decoder_layer (int): number of layers of decoder model.
        decoder_input (keras.layers.Input): input tensor of decoder.
        path_save_model (str, optional): where to save model. Defaults to None.
        auto_model_name (str, optional): name of the autoencoder model. Defaults to None.
        encoder_model_name (str, optional): name of the encoder model. Defaults to None.
        decoder_model_name (str, optional): name of the decoder model. Defaults to None.
        path_save (str, optional): where to save model. Defaults to None.
        data_set_name (str, optional): name of the dataset, sub-dir of model. Defaults to None.
        modelname (str, optional): model name for models. Defaults to None.
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
        path_save_model = os.path.join(path_save, 'auto_model', data_set_name, modelname)
    if auto_model_name is None:
        auto_model_name = os.path.join(path_save_model, f'{modelname}_auto.h5')
    if encoder_model_name is None:
        encoder_model_name = os.path.join(path_save_model, f'{modelname}_encoder.h5')
    if decoder_model_name is None:
        decoder_model_name = os.path.join(path_save_model, f'{modelname}_decoder.h5')

    auto_model = load_model(auto_model_name, custom_objects=dict_model_load)

    encoder_layers = [auto_model.layers[i] for i in range(1, n_encoder_layer)]
    x = encoder_input
    for encoder_layer in encoder_layers:
        x = encoder_layer(x)
    encoded = x
    encoder = Model(encoder_input, encoded)
    logging.debug(encoder.summary())
    encoder.save(encoder_model_name)
    save_model_struct(encoder, path_save_model, 'encoder_model_struct')

    decoder_layers = [auto_model.layers[i] for i in range(-n_decoder_layer, 0)]
    x = decoder_input
    for decoder_layer in decoder_layers:
        x = decoder_layer(x)
    decoder_output = x
    decoder = Model(decoder_input, decoder_output)
    logging.debug(decoder.summary())
    decoder.save(decoder_model_name)
    save_model_struct(decoder, path_save_model, 'decoder_model_struct')


def save_encoder_decoder(n_encoder_layer, decoder=None, path_save_model=None,
                         auto_model_name=None, encoder_model_name=None, decoder_model_name=None,
                         path_save=None, data_set_name=None, modelname=None, paras=None, dict_model_load=None):
    """Split and save encoder and decoder model from autoencoder model with weights, through graph of model.
    Args:
        n_encoder_layer (int): number of layers of encoder model.
        decoder (keras.Model, optional): decoder model. Defaults to None.
        path_save_model (str, optional): where to save model. Defaults to None.
        auto_model_name (str, optional): name of the autoencoder model. Defaults to None.
        encoder_model_name (str, optional): name of the encoder model. Defaults to None.
        decoder_model_name (str, optional): name of the decoder model. Defaults to None.
        path_save (str, optional): where to save model. Defaults to None.
        data_set_name (str, optional): name of the dataset, sub-dir of model. Defaults to None.
        modelname (str, optional): model name for models. Defaults to None.
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
        path_save_model = os.path.join(path_save, 'auto_model', data_set_name, modelname)
    if auto_model_name is None:
        auto_model_name = os.path.join(path_save_model, f'{modelname}_auto.h5')
        encoder_model_name = os.path.join(path_save_model, f'{modelname}_encoder.h5')
        decoder_model_name = os.path.join(path_save_model, f'{modelname}_decoder.h5')

    auto_model = load_model(auto_model_name, dict_model_load)
    logging.debug(f'auto model \n {auto_model.summary()}')

    encoder_out_layer_name = auto_model.layers[n_encoder_layer].name
    encoder = part_front_model(auto_model, encoder_out_layer_name)
    logging.debug(encoder.summary())
    encoder.save(encoder_model_name)
    save_model_struct(encoder, path_save_model, 'encoder_model_struct')

    if decoder is None:
        # DO NOT use this, not working !
        decoder = part_back_model(auto_model, auto_model.get_layer(encoder_out_layer_name).output)
    else:
        logging.debug(f'num auto layer {len(auto_model.layers[n_encoder_layer+1:])}')
        logging.debug(f'num decoder layer {len(decoder.layers[1:])}')
        for i, layer in enumerate(auto_model.layers[n_encoder_layer+1:]):
            layer_weights = layer.get_weights()
            # logging.debug(f'auto_weights {layer_weights}')
            decoder.layers[i+1].set_weights(layer_weights)
            # logging.debug(f'decoder_weights {decoder.layers[i+1].get_weights()}')
    logging.debug(decoder.summary())
    decoder.save(decoder_model_name)
    save_model_struct(decoder, path_save_model, 'decoder_model_struct')


def test_encoder_decoder(z_dict, path_save_model=None, encoder_model_name=None, decoder_model_name=None,
                         path_save=None, modelname=None, paras=None, dict_model_load=None, n_encoder_layer=None,
                         weight_file_name=None, auto_model_name=None, decoder_weight_model_name=None):
    """Load and test encoder and decoder.
    Args:
        z_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model input, data to predict.
        path_save_model (str, optional): where to save model. Defaults to None.
        encoder_model_name (str, optional): name of the encoder model. Defaults to None.
        decoder_model_name (str, optional): name of the decoder model. Defaults to None.
        path_save (str, optional): where to save model. Defaults to None.
        modelname (str, optional): model name for models. Defaults to None.
        paras (dict, optional): parameters for train model. Defaults to None.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
        n_encoder_layer (int, optional): number of layers of encoder model. Defaults to None.
        weight_file_name (str, optional): for name of the weight file name. Defaults to None.
        auto_model_name (str, optional): name of the autoencoder model. Defaults to None.
        decoder_weight_model_name (str, optional): name of the decoder model after load weight file. Defaults to None.
    """
    if modelname is None:
        i = paras['i']
        j = paras['j']
        epochs = paras['epochs']
        strj = f'n{str(j)[1:]}' if j < 0 else str(j)
        modelname = f'{i}_{strj}_{epochs}'
    if path_save_model is None:
        path_save_model = os.path.join(path_save, 'auto_model', list(z_dict.keys())[0][:-6], modelname)
    if encoder_model_name is None:
        encoder_model_name = os.path.join(path_save_model, f'{modelname}_encoder.h5')
        decoder_model_name = os.path.join(path_save_model, f'{modelname}_decoder.h5')

    encoder_model = load_model(encoder_model_name, custom_objects=dict_model_load)
    decoder_model = load_model(decoder_model_name, custom_objects=dict_model_load)

    path_auto = os.path.join(path_save, 'auto_out', modelname)

    mode_pred = paras['mode_pred'] if 'mode_pred' in paras.keys() else 'batch_process'
    bs_pred = paras['bs_pred'] if 'bs_pred' in paras.keys() else 32

    if weight_file_name is not None:
        if auto_model_name is None:
            auto_model_name = os.path.join(path_save_model, f'{modelname}_auto.h5')
        auto_model = load_model(auto_model_name, custom_objects=dict_model_load)
        auto_model.load_weights(os.path.join(path_save_model, weight_file_name))

        # not necessary, only for verify autoencoder output is same as decoder output.
        for name_set, data_set in z_dict.items():
            predict_autoencoder(auto_model, data_set, path_auto,
                                f'{name_set}_{weight_file_name}_autodecoded', mode=mode_pred, bs_pred=bs_pred)

        for i, layer in enumerate(auto_model.layers[1:n_encoder_layer+1]):
            layer_weights = layer.get_weights()
            encoder_model.layers[i+1].set_weights(layer_weights)

        for i, layer in enumerate(auto_model.layers[n_encoder_layer+1:]):
            layer_weights = layer.get_weights()
            decoder_model.layers[i+1].set_weights(layer_weights)

        if decoder_weight_model_name is None:
            decoder_weight_model_name = os.path.join(path_save_model, f'{modelname}_{weight_file_name}_decoder.h5')
        decoder_model.save(decoder_weight_model_name)

    for name_set, data_set in z_dict.items():
        if weight_file_name is not None:
            name_set += f'_{weight_file_name}'
        predict_autoencoder(encoder_model, data_set, path_auto, f'{name_set}_encoded', compile_model=True,
                            mode=mode_pred, bs_pred=bs_pred)
        data_set_encoded = read_data(path_auto, f'{name_set}_encoded')
        predict_autoencoder(decoder_model, data_set_encoded, path_auto, f'{name_set}_decoded', compile_model=True,
                            mode='batch_save', bs_pred=bs_pred)


def train_test_ae(autoencoder, decoder, n_encoder_layer, paras, x_dict, z_dict, z_set_name, path_save,
                  dict_model_load=None, fname_model_load=None,
                  bool_train=True, bool_test_ae=True, bool_save_ed=True, bool_test_ed=True):
    """Train and test autoencoder models.
    Args:
        autoencoder (keras.Model): model of autoencoder.
        decoder (keras.Model): model of decoder.
        n_encoder_layer (int): number of layers of encoder model.
        paras (dict): parameters for train model.
        x_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model input, data to predict.
        z_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model output, data to predict.
        z_set_name (str): name of the output dataset, sub-dir of model.
        path_save (str): where to save model.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
        fname_model_load (os.Path, optional): file name of the model to load. Defaults to None.
        bool_train (bool, optional): whether train ae model. Defaults to True.
        bool_test_ae (bool, optional): whether test ae model. Defaults to True.
        bool_save_ed (bool, optional): whether save encoder and decoder model. Defaults to True.
        bool_test_ed (bool, optional): whether test encoder and decoder model. Defaults to True.
    """
    if bool_train:
        if fname_model_load is None:
            train_autoencoder(autoencoder, paras, x_dict, z_dict, path_save)
        else:
            train_autoencoder(None, paras, x_dict, z_dict, path_save,
                              dict_model_load=dict_model_load, fname_model_load=fname_model_load)

    if bool_test_ae:
        test_autoencoder(x_dict, z_names=list(z_dict.keys()), path_save=path_save, paras=paras,
                         dict_model_load=dict_model_load)
    if bool_save_ed:
        save_encoder_decoder(n_encoder_layer, decoder=decoder, path_save=path_save,
                             data_set_name=z_set_name, paras=paras, dict_model_load=dict_model_load)
    if bool_test_ed:
        test_encoder_decoder(dict(zip(z_dict.keys(), x_dict.values())), path_save=path_save, paras=paras,
                             dict_model_load=dict_model_load)


def test_weight_ae(data_dict, n_encoder_layer, path_save=None, paras=None, dict_model_load=None,
                   weight_file_names=None, path_save_model=None, modelname=None,
                   bool_test_ae=False, bool_test_ed=True):
    """Test encoder and decoder through saved weight files.
    Args:
        data_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model input, data to predict.
        n_encoder_layer (int): number of layers of encoder model.
        path_save (str, optional): where to save model. Defaults to None.
        paras (dict, optional): parameters for train model. Defaults to None.
        weight_file_name (str, optional): for name of the weight file name. Defaults to None.
        path_save_model (str, optional): path of the autoencoder model. Defaults to None.
        modelname (str, optional): for autoencoder model name. Defaults to None.
        bool_test_ae (bool, optional): whether test ae model. Defaults to False.
        bool_test_ed (bool, optional): whether test encoder and decoder model. Defaults to True.
    Raises:
        Error: 'weight_file_names not found'.
    """
    if path_save_model is None:
        if modelname is None:
            i = paras['i']
            j = paras['j']
            epochs = paras['epochs']
            strj = f'n{str(j)[1:]}' if j < 0 else str(j)
            modelname = f'{i}_{strj}_{epochs}'
        path_save_model = os.path.join(path_save, 'auto_model', list(data_dict.keys())[0][:-6], modelname)
    if weight_file_names is None:
        weight_file_names = list_files_end_str(path_save_model, 'hdf5', False)
        if not weight_file_names:
            raise Error('weight_file_names not found')
    if not isinstance(weight_file_names, list):
        weight_file_names = [weight_file_names]

    for weight_file_name in weight_file_names:
        if bool_test_ed:
            test_encoder_decoder(data_dict, n_encoder_layer=n_encoder_layer,
                                 path_save=path_save, paras=paras, dict_model_load=dict_model_load,
                                 weight_file_name=weight_file_name)
        if bool_test_ae:
            test_autoencoder(data_dict, z_names=list(data_dict.keys()), path_save=path_save, paras=paras,
                             dict_model_load=dict_model_load,
                             weight_file_name=os.path.join(path_save_model, weight_file_name),)


def clear_model_weight_file(path_model_dir):
    """Clear model files with weights, saved the last model.
    Args:
        path_model (str): Root path, where save weight files of a model.
    """
    path_model_weights = walk_files_end_str(path_model_dir, '.hdf5')
    for i in range(len(path_model_weights)-1):
        if os.path.dirname(path_model_weights[i+1]) == os.path.dirname(path_model_weights[i]):
            os.remove(path_model_weights[i])

    path_model_weights = walk_files_end_str(path_model_dir, '.hdf5')


def clear_model_weight_files(path, model_dirname='auto_model'):
    """Clear model files with weights, saved the last model.
    Args:
        path (str): Root path, where save models.
    """
    path_models = list_dirs_start_str(path, 'model_')
    for path_model in path_models:
        path_model_dir = os.path.join(path_model, model_dirname)
        clear_model_weight_file(path_model_dir)


def compute_num_model(model_name):
    """Compute num of the model from model name str.
    Args:
        model_name (str): name of the model.
    Returns:
        int: number of the name, e.g. 3.
    Examples:
        >>> print(compute_num_model('model_3_2_1'))
        3
    """
    return int(model_name.split('_')[1])


def build_dict_model_load(num_model, **kwargs):
    """User defined items in model, for model loading.
    Args:
        num_model (int): index number of the model.
    Returns:
        dict_model_load (dict): items of the model.
    """
    if num_model in (1, 2, 3, 4, 7, 8, 9):
        if 'user_metrics_func' in kwargs.keys():
            user_metrics_func = kwargs['user_metrics_func']
        else:
            user_metrics_func = [samerate_acc_d2]
        if 'user_metrics_name' in kwargs.keys():
            user_metrics_name = kwargs['user_metrics_name']
        else:
            user_metrics_name = ['samerate_acc_d2']
    elif num_model in (5, 6):
        if 'user_metrics_func' in kwargs.keys():
            user_metrics_func = kwargs['user_metrics_func']
        else:
            user_metrics_func = [vae_loss(z_mean=0., z_log_var=1.)]
        if 'user_metrics_name' in kwargs.keys():
            user_metrics_name = kwargs['user_metrics_name']
        else:
            user_metrics_name = ['loss']

    if 'dict_model_load' in kwargs.keys():
        dict_model_load = kwargs['dict_model_load']
    else:
        dict_model_load = dict(zip(user_metrics_name, user_metrics_func))

    if num_model in (8, 9):
        rnn_type = kwargs['rnn_type'] if 'rnn_type' in kwargs.keys() else 'LSTM'

        dict_model_load.update(**{'LayerNormalization': LayerNormalization,
                                  'segment_encoded_signal': segment_encoded_signal,
                                  'overlap_and_add_in_decoder': overlap_and_add_in_decoder,
                                  'tf': tf,
                                  })
        if rnn_type == 'dprnn':
            input_dim = kwargs['input_dim'] if 'input_dim' in kwargs.keys() else 10547
            n_pad_input = kwargs['n_pad_input'] if 'n_pad_input' in kwargs.keys() else 13
            chunk_size = kwargs['chunk_size'] if 'chunk_size' in kwargs.keys() else 64
            units_r = kwargs['units_r'] if 'units_r' in kwargs.keys() else 2
            act_r = kwargs['act_r'] if 'act_r' in kwargs.keys() else 'tanh'
            use_bias = kwargs['use_bias'] if 'use_bias' in kwargs.keys() else True
            n_filters_conv = kwargs['n_filters_conv'] if 'n_filters_conv' in kwargs.keys() else 1
            n_full_chunks = (input_dim + n_pad_input) // chunk_size
            n_overlapping_chunks = kwargs['n_overlapping_chunks'] if 'n_overlapping_chunks' in kwargs.keys(
            ) else n_full_chunks*2-1
            dict_dprnn = {'DprnnBlock': DprnnBlock(is_last_dprnn=False,
                                                   num_overlapping_chunks=n_overlapping_chunks,
                                                   chunk_size=chunk_size,
                                                   num_filters_in_encoder=n_filters_conv,
                                                   units_per_lstm=units_r,
                                                   act_r=act_r,
                                                   use_bias=use_bias)}
            dict_model_load.update(**dict_dprnn)
    return dict_model_load


class BuildModel(object):
    """class BuildModel for build model.
    Args:
        object (class): base class.
    """

    def __init__(self, num_model, input_dim, **kwargs):
        """ __init__
        Args:
            num_model (int): index number of the model.
            input_dim (tuple(int))): shape of the input vector.
        """
        self.num_model = num_model
        self.input_dim = input_dim
        self.encoding_dim = kwargs['encoding_dim'] if 'encoding_dim' in kwargs.keys() else 2
        self.latent_dim = kwargs['latent_dim'] if 'latent_dim' in kwargs.keys() else 2
        self.intermediate_dim = kwargs['intermediate_dim'] if 'intermediate_dim' in kwargs.keys() else 2
        self.act_c = kwargs['act_c'] if 'act_c' in kwargs.keys() else 'relu'
        self.act_r = kwargs['act_r'] if 'act_r' in kwargs.keys() else 'tanh'
        self.batch_norm = kwargs['batch_norm'] if 'batch_norm' in kwargs.keys() else False
        self.n_pad_input = kwargs['n_pad_input'] if 'n_pad_input' in kwargs.keys() else 13
        self.n_filters = kwargs['n_filters'] if 'n_filters' in kwargs.keys() else 1
        self.use_bias = kwargs['use_bias'] if 'use_bias' in kwargs.keys() else True
        self.n_outputs = kwargs['n_outputs'] if 'n_outputs' in kwargs.keys() else 1
        self.epsilon_std = kwargs['epsilon_std'] if 'epsilon_std' in kwargs.keys() else 1.0

    def build_model(self, **kwargs):
        """Build a autoencoder model.
        Returns:
            autoencoder (keras.Model): the autoencoder model.
            decoder (keras.Model): the decoder model.
            n_encoder_layer (int): number of the encoder layers.
            n_decoder_layer (int): number of the decoder layers.
        """
        num_model = self.num_model
        if num_model == 1:
            input_dim = self.input_dim
            encoding_dim = self.encoding_dim
            act_c = self.act_c
            n_outputs = self.n_outputs
            use_bias = self.use_bias
            n_nodes = kwargs['n_nodes'] if 'n_nodes' in kwargs.keys() else [8, 4, 2]
            batch_norm = kwargs['batch_norm'] if 'batch_norm' in kwargs.keys() else False
            autoencoder, decoder, n_encoder_layer, n_decoder_layer = build_model_1(
                input_dim=input_dim, encoding_dim=encoding_dim, output_dim=input_dim, act_c=act_c,
                n_nodes=n_nodes, batch_norm=batch_norm, n_outputs=n_outputs, use_bias=use_bias)
        elif num_model == 2:
            input_dim = self.input_dim
            act_c = self.act_c
            use_bias = self.use_bias
            n_pad_input = self.n_pad_input
            n_filters = self.n_filters
            batch_norm = kwargs['batch_norm'] if 'batch_norm' in kwargs.keys() else True
            dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else None
            autoencoder, decoder, n_encoder_layer, n_decoder_layer = build_model_2(
                input_dim=input_dim, n_pad_input=n_pad_input, n_filters=n_filters,
                dropout=dropout, batch_norm=batch_norm, use_bias=use_bias, act_c=act_c)
        elif num_model == 3:
            input_dim = self.input_dim
            act_c = self.act_c
            use_bias = self.use_bias
            n_pad_input = self.n_pad_input
            n_filters = self.n_filters
            batch_norm = kwargs['batch_norm'] if 'batch_norm' in kwargs.keys() else True
            dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else None
            autoencoder, decoder, n_encoder_layer, n_decoder_layer = build_model_3(
                input_dim=input_dim, n_pad_input=n_pad_input, n_filters=n_filters,
                dropout=dropout, batch_norm=batch_norm, use_bias=use_bias, act_c=act_c)
        elif num_model == 4:
            input_dim = self.input_dim
            act_c = self.act_c
            use_bias = self.use_bias
            n_pad_input = self.n_pad_input
            n_filters = self.n_filters
            latent_dim = self.latent_dim
            act_r = self.act_r
            batch_norm = kwargs['batch_norm'] if 'batch_norm' in kwargs.keys() else True
            autoencoder, decoder, n_encoder_layer, n_decoder_layer = build_model_4(
                input_dim=input_dim, latent_dim=latent_dim, n_pad_input=n_pad_input, n_filters=n_filters,
                act_c=act_c, batch_norm=batch_norm, use_bias=use_bias, act_r=act_r)
        elif num_model == 5:
            input_dim = self.input_dim
            latent_dim = self.latent_dim
            intermediate_dim = self.intermediate_dim
            act_c = self.act_c
            use_bias = self.use_bias
            n_pad_input = self.n_pad_input
            n_filters = self.n_filters
            epsilon_std = self.epsilon_std
            batch_norm = kwargs['batch_norm'] if 'batch_norm' in kwargs.keys() else True
            dropout = kwargs['dropout'] if 'dropout' in kwargs.keys() else None
            autoencoder, decoder, n_encoder_layer, n_decoder_layer = build_model_5(
                input_dim=input_dim, latent_dim=latent_dim, intermediate_dim=intermediate_dim, n_pad_input=n_pad_input,
                n_filters=n_filters, use_bias=use_bias, act_c=act_c, dropout=dropout, batch_norm=batch_norm,
                epsilon_std=epsilon_std)
        elif num_model == 6:
            input_dim = self.input_dim
            encoding_dim = self.encoding_dim
            latent_dim = self.latent_dim
            intermediate_dim = self.intermediate_dim
            act_c = self.act_c
            use_bias = self.use_bias
            epsilon_std = self.epsilon_std
            n_nodes = kwargs['n_nodes'] if 'n_nodes' in kwargs.keys() else [0.5]
            autoencoder, decoder, n_encoder_layer, n_decoder_layer = build_model_6(
                input_dim=input_dim, encoding_dim=encoding_dim, latent_dim=latent_dim,
                intermediate_dim=intermediate_dim, output_dim=input_dim, use_bias=use_bias, act_c=act_c,
                epsilon_std=epsilon_std, n_nodes=n_nodes)
        elif num_model == 7:
            input_dim = self.input_dim
            latent_dim = self.latent_dim
            use_bias = self.use_bias
            act_r = self.act_r
            autoencoder, decoder, n_encoder_layer, n_decoder_layer = build_model_7(
                input_dim=input_dim, latent_dim=latent_dim, use_bias=use_bias, act_r=act_r)
        elif num_model in (8, 9):
            input_dim = self.input_dim
            n_pad_input = self.n_pad_input
            n_filters = self.n_filters
            act_c = self.act_c
            latent_dim = self.latent_dim
            act_r = self.act_r
            use_bias = self.use_bias
            n_outputs = self.n_outputs
            n_conv_encoder = kwargs['n_conv_encoder'] if 'n_conv_encoder' in kwargs.keys() else 0
            chunk_size = kwargs['chunk_size'] if 'chunk_size' in kwargs.keys() else 64
            chunk_advance = kwargs['chunk_advance'] if 'chunk_advance' in kwargs.keys() else chunk_size // 2
            kernel_size = kwargs['kernel_size'] if 'kernel_size' in kwargs.keys() else 2
            strides = kwargs['strides'] if 'strides' in kwargs.keys() else 1
            batch_norm = kwargs['batch_norm'] if 'batch_norm' in kwargs.keys() else False
            rnn_type = kwargs['rnn_type'] if 'rnn_type' in kwargs.keys() else 'LSTM'
            n_full_chunks = (input_dim + n_pad_input) // chunk_size
            n_overlapping_chunks = n_full_chunks*2-1
            self.chunk_size = chunk_size
            self.n_overlapping_chunks = n_overlapping_chunks
            encoder_multiple_out = kwargs['encoder_multiple_out'] if 'encoder_multiple_out' in kwargs.keys() else False

            if num_model == 8:
                n_rnn_encoder = kwargs['n_rnn_encoder'] if 'n_rnn_encoder' in kwargs.keys() else 1
                n_rnn_decoder = kwargs['n_rnn_decoder'] if 'n_rnn_decoder' in kwargs.keys() else 1
            if num_model == 9:
                n_rnn_encoder = kwargs['n_rnn_encoder'] if 'n_rnn_encoder' in kwargs.keys() else 2
                n_rnn_decoder = kwargs['n_rnn_decoder'] if 'n_rnn_decoder' in kwargs.keys() else 0

            autoencoder, decoder, n_encoder_layer, n_decoder_layer = build_model_8(
                input_dim, n_pad_input, chunk_size, chunk_advance,
                n_conv_encoder, n_filters, kernel_size, strides, batch_norm, act_c,
                n_rnn_encoder, n_rnn_decoder, rnn_type, latent_dim, act_r, use_bias,
                n_outputs, encoder_multiple_out)
        return autoencoder, decoder, n_encoder_layer, n_decoder_layer

    def build_model_paras(self, **kwargs):
        """For build parameters for train model.
        Returns:
            paras (dict): parameters for train model.
        """
        num_model = self.num_model
        lr_i = kwargs['i'] if 'i' in kwargs.keys() else None
        lr_j = kwargs['j'] if 'j' in kwargs.keys() else None
        epochs = kwargs['epochs'] if 'epochs' in kwargs.keys() else 2
        batch_size = kwargs['batch_size'] if 'batch_size' in kwargs.keys() else 32
        if num_model in (1, 2, 3, 4, 7, 8, 9):
            paras = {'i': lr_i, 'j': lr_j, 'epochs': epochs, 'batch_size': batch_size}
        elif num_model in (5, 6):
            loss_func = kwargs['loss_func'] if 'loss_func' in kwargs.keys() else vae_loss(z_mean=0., z_log_var=1.)
            # metrics_func = kwargs['metrics_func'] if ('metrics_func' in kwargs.keys()
            #                                           ) else [keras.losses.mean_squared_error]
            # metrics_name = kwargs['metrics_name'] if 'metrics_name' in kwargs.keys() else ['mean_squared_error']
            user_metrics_func = kwargs['user_metrics_func'] if ('user_metrics_func' in kwargs.keys()
                                                                ) else [vae_loss(z_mean=0., z_log_var=1.)]
            user_metrics_name = kwargs['user_metrics_name'] if 'user_metrics_name' in kwargs.keys() else ['loss']
            # paras = {'i': lr_i, 'j': lr_j, 'epochs': epochs, 'batch_size': batch_size,
            #          'loss_func': loss_func, 'metrics_func': metrics_func, 'metrics_name': metrics_name}
            paras = {'i': lr_i, 'j': lr_j, 'epochs': epochs, 'batch_size': batch_size,
                     'loss_func': loss_func, 'metrics_func': user_metrics_func, 'metrics_name': user_metrics_name}
        if num_model in (8, 9):
            if hasattr(self, 'chunk_size'):
                chunk_size = self.chunk_size
            elif 'chunk_size' in kwargs.keys():
                chunk_size = kwargs['chunk_size']
            else:
                chunk_size = 64
            self.chunk_size = chunk_size
            if hasattr(self, 'n_overlapping_chunks'):
                n_overlapping_chunks = self.n_overlapping_chunks
            elif 'n_overlapping_chunks' in kwargs.keys():
                n_overlapping_chunks = kwargs['n_overlapping_chunks']
            else:
                input_dim = self.input_dim
                n_pad_input = self.n_pad_input
                n_full_chunks = (input_dim + n_pad_input) // chunk_size
                n_overlapping_chunks = n_full_chunks*2-1
            self.n_overlapping_chunks = n_overlapping_chunks
            n_filters = self.n_filters
            latent_dim = self.latent_dim
            act_r = self.act_r
            use_bias = self.use_bias
            if 'rnn_type' in kwargs.keys() and kwargs['rnn_type'] == 'dprnn':
                dict_dprnn = {'DprnnBlock': DprnnBlock(is_last_dprnn=False,
                                                       num_overlapping_chunks=n_overlapping_chunks,
                                                       chunk_size=chunk_size,
                                                       num_filters_in_encoder=n_filters,
                                                       units_per_lstm=latent_dim,
                                                       act_r=act_r,
                                                       use_bias=use_bias)}
                paras.update(**dict_dprnn)
        return paras


def search_model(path_result, model_name, input_dim, x_dict, z_dict, z_set_name, **kwargs):
    """For search best model.
    Args:
        path_result (str): where to save result.
        model_name (str): name of the model.
        input_dim (tuple, np.ndarray.shape): shape of the input tensor.
        x_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model input, mixed data.
        z_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model output, data target.
        z_set_name (str): name of the predict dataset, sub-dir of model.
    """
    num_model = compute_num_model(model_name)
    object_build_model = BuildModel(num_model, input_dim, **kwargs)
    autoencoder, decoder, n_encoder_layer, _ = object_build_model.build_model(**kwargs)
    dict_model_load = build_dict_model_load(num_model, **kwargs)
    paras = object_build_model.build_model_paras(**kwargs)

    path_save = os.path.join(path_result, model_name)
    bool_train = kwargs['bool_train'] if 'bool_train' in kwargs.keys() else True
    bool_test_ae = kwargs['bool_test_ae'] if 'bool_test_ae' in kwargs.keys() else True
    bool_save_ed = kwargs['bool_save_ed'] if 'bool_save_ed' in kwargs.keys() else True
    bool_test_ed = kwargs['bool_test_ed'] if 'bool_test_ed' in kwargs.keys() else True
    train_test_ae(autoencoder, decoder, n_encoder_layer, paras,
                  x_dict, z_dict, z_set_name, path_save=path_save,
                  dict_model_load=dict_model_load,
                  bool_train=bool_train, bool_test_ae=bool_test_ae,
                  bool_save_ed=bool_save_ed, bool_test_ed=bool_test_ed)
    bool_clean_weight_file = kwargs['bool_clean_weight_file'] if 'bool_clean_weight_file' in kwargs.keys() else True
    if bool_clean_weight_file:
        model_dirname = kwargs['model_dirname'] if 'model_dirname' in kwargs.keys() else 'auto_model'
        clear_model_weight_file(os.path.join(path_save, model_dirname))
    bool_test_weight = kwargs['bool_test_weight'] if 'bool_test_weight' in kwargs.keys() else True
    if bool_test_weight:
        bool_test_ae_w = kwargs['bool_test_ae_w'] if 'bool_test_ae_w' in kwargs.keys() else False
        bool_test_ed_w = kwargs['bool_test_ed_w'] if 'bool_test_ed_w' in kwargs.keys() else True
        test_weight_ae(dict(zip(z_dict.keys(), x_dict.values())), n_encoder_layer=n_encoder_layer,
                       path_save=path_save, paras=paras, dict_model_load=dict_model_load,
                       bool_test_ae=bool_test_ae_w, bool_test_ed=bool_test_ed_w)
    del autoencoder
    del decoder
    K.clear_session()
    gc.collect()


def search_model_continue(fname_model_load, path_result, model_name, input_dim, x_dict, z_dict, z_set_name, **kwargs):
    """For search best model with a loaded model and continue to train.
    Args:
        fname_model_load (os.Path): file name of the model to load.
        path_result (str): where to save result.
        model_name (str): name of the model.
        input_dim (tuple, np.ndarray.shape): shape of the input tensor.
        x_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model input, mixed data.
        z_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model output, data target.
        z_set_name (str): name of the predict dataset, sub-dir of model.
    """
    num_model = compute_num_model(model_name)
    object_build_model = BuildModel(num_model, input_dim, **kwargs)
    _, decoder, n_encoder_layer, _ = object_build_model.build_model(**kwargs)
    dict_model_load = build_dict_model_load(num_model, **kwargs)
    paras = object_build_model.build_model_paras(**kwargs)

    path_save = os.path.join(path_result, model_name)

    bool_train = kwargs['bool_train'] if 'bool_train' in kwargs.keys() else True
    bool_test_ae = kwargs['bool_test_ae'] if 'bool_test_ae' in kwargs.keys() else True
    bool_save_ed = kwargs['bool_save_ed'] if 'bool_save_ed' in kwargs.keys() else True
    bool_test_ed = kwargs['bool_test_ed'] if 'bool_test_ed' in kwargs.keys() else True

    train_test_ae(None, decoder, n_encoder_layer, paras, x_dict, z_dict, z_set_name, path_save,
                  dict_model_load=dict_model_load, fname_model_load=fname_model_load,
                  bool_train=bool_train, bool_test_ae=bool_test_ae,
                  bool_save_ed=bool_save_ed, bool_test_ed=bool_test_ed)

    bool_clean_weight_file = kwargs['bool_clean_weight_file'] if 'bool_clean_weight_file' in kwargs.keys() else True
    if bool_clean_weight_file:
        model_dirname = kwargs['model_dirname'] if 'model_dirname' in kwargs.keys() else 'auto_model'
        clear_model_weight_file(os.path.join(path_save, model_dirname))
    bool_test_weight = kwargs['bool_test_weight'] if 'bool_test_weight' in kwargs.keys() else True
    if bool_test_weight:
        bool_test_ae_w = kwargs['bool_test_ae_w'] if 'bool_test_ae_w' in kwargs.keys() else False
        bool_test_ed_w = kwargs['bool_test_ed_w'] if 'bool_test_ed_w' in kwargs.keys() else True
        test_weight_ae(dict(zip(z_dict.keys(), x_dict.values())), n_encoder_layer=n_encoder_layer,
                       path_save=path_save, paras=paras, dict_model_load=dict_model_load,
                       bool_test_ae=bool_test_ae_w, bool_test_ed=bool_test_ed_w)
    del decoder
    K.clear_session()
    gc.collect()


if __name__ == '__main__':
    import json
    import pickle

    import tensorflow as tf

    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO)
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
    #  ----------------------------------------------------------------------------------------
    # only for first time generate data set
    S_NAMES = json.load(open(os.path.join(PATH_DATA_S, 'dirname.json'), 'r'))['dirname']
    S_LIST = read_datas(os.path.join(PATH_DATA_S, 's_hdf5'), S_NAMES)

    N_SAMS = S_LIST[0].shape[0]
    if SUB_SET_WAY == 'rand':
        with open(os.path.join(PATH_DATA_S, 'randseq.pickle'), 'rb') as f_rb:
            nums_seq = pickle.load(f_rb)
    elif SUB_SET_WAY == 'order':
        nums_seq = list(range(N_SAMS))

    nums = subset_seq(nums_seq, [3055, 1018, 1020])

    z_sets_ns_create(S_LIST, nums, SCALER_DATA, PATH_DATA)
    #  ----------------------------------------------------------------------------------------
    SET_NAMES = ['train', 'val', 'test']
    S_NAMES = []  # [n_source][n_set]
    for i in range(0, 4):
        S_NAMES.append([f'Z_{i}_ns_{name_set_j}' for name_set_j in SET_NAMES])

    DATA_DICT = []
    for s_names_j in S_NAMES:
        s_list_j = read_datas(PATH_DATA, s_names_j)
        DATA_DICT.append(dict(zip(s_names_j, s_list_j)))

    PATH_RESULT = '../result_separation_ae_ns_single'
    mkdir(PATH_RESULT)

    PATH_RESULT_CONTINUE = '../result_separation_ae_ns_single_continue'
    mkdir(PATH_RESULT)

    # lr_i_j = [(i, j) for i in range(1, 2) for j in range(-4, -5, -1)]
    lr_i_j = [(i, j) for i in range(1, 2) for j in range(-3, -4, -1)]
    for (lr_i, lr_j) in lr_i_j:
        for s_names_j, data_dict_j in zip(S_NAMES, DATA_DICT):
            logging.info(data_dict_j.keys())

            input_dim_j = data_dict_j[s_names_j[-1]].shape[-1]  # (nsamples, 1, frame_length)
            logging.debug(f'input_dim {input_dim_j}')

            for key, value in data_dict_j.items():
                data_dict_j[key] = np.squeeze(value)  # (nsamples, frame_length)

            # search_model(PATH_RESULT, 'model_1_2_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'encoding_dim': 256, 'batch_norm': True,
            #                 'bool_train': True, 'bool_test_ae': False,
            #                 'bool_save_ed': True, 'bool_test_ed': False,
            #                 'bool_clean_weight_file': True, 'bool_test_weight': True})

            # search_model(PATH_RESULT, 'model_6_1_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'encoding_dim': 256, 'intermediate_dim': 256,
            #                  'bool_train': True, 'bool_test_ae': False,
            #                  'bool_save_ed': True, 'bool_test_ed': False,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

            for key, value in data_dict_j.items():
                data_dict_j[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (nsamples, frame_length, 1)

            # search_model(PATH_RESULT, 'model_2_2_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'n_filters': 16, 'batch_norm': False,
            #                  'bool_train': True, 'bool_test_ae': True,
            #                  'bool_save_ed': True, 'bool_test_ed': True,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

            # search_model(PATH_RESULT, 'model_3_1_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'n_filters': 16,
            #                  'bool_train': True, 'bool_test_ae': True,
            #                  'bool_save_ed': True, 'bool_test_ed': True,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

            # search_model(PATH_RESULT, 'model_4_1_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'latent_dim': 256, 'n_filters': 16,
            #                  'bool_train': True, 'bool_test_ae': True,
            #                  'bool_save_ed': True, 'bool_test_ed': True,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

            # search_model(PATH_RESULT, 'model_5_1_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'n_filters': 16, 'intermediate_dim': 256,
            #                  'bool_train': True, 'bool_test_ae': True,
            #                  'bool_save_ed': True, 'bool_test_ed': True,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

            # search_model(PATH_RESULT, 'model_8_1_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                  'n_conv_encoder': 1, 'n_filters': 64,
            #                  'rnn_type': 'LSTM', 'latent_dim': 256, 'n_rnn_decoder': 1,
            #                  'bool_train': True, 'bool_test_ae': True,
            #                  'bool_save_ed': True, 'bool_test_ed': True,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

            search_model(PATH_RESULT, 'model_8_2_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
                         **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                             'n_conv_encoder': 1, 'n_filters': 64,
                             'rnn_type': 'BLSTM', 'latent_dim': 256, 'n_rnn_decoder': 1,
                             'bool_train': True, 'bool_test_ae': True,
                             'bool_save_ed': True, 'bool_test_ed': True,
                             'bool_clean_weight_file': True, 'bool_test_weight': True})

            # search_model(PATH_RESULT, 'model_8_2_2', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                  'n_conv_encoder': 1, 'n_filters': 64,
            #                  'rnn_type': 'BLSTM', 'latent_dim': 256, 'n_rnn_decoder': 2,
            #                  'bool_train': True, 'bool_test_ae': True,
            #                  'bool_save_ed': True, 'bool_test_ed': True,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

            # search_model(PATH_RESULT, 'model_8_3_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                  'n_conv_encoder': 1, 'n_filters': 64,
            #                  'rnn_type': 'dprnn', 'latent_dim': 256, 'n_rnn_decoder': 1,
            #                  'bool_train': True, 'bool_test_ae': True,
            #                  'bool_save_ed': True, 'bool_test_ed': True,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

            # search_model(PATH_RESULT, 'model_8_4_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                  'n_conv_encoder': 1, 'n_filters': 64, 'use_bias': False,
            #                  'rnn_type': 'BLSTM', 'latent_dim': 256, 'n_rnn_decoder': 1,
            #                  'bool_train': True, 'bool_test_ae': True,
            #                  'bool_save_ed': True, 'bool_test_ed': True,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

            # search_model(PATH_RESULT, 'model_8_5_1', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                  'n_conv_encoder': 8, 'n_filters': 64, 'use_bias': False,
            #                  'rnn_type': 'BLSTM', 'latent_dim': 256, 'n_rnn_decoder': 1,
            #                  'bool_train': True, 'bool_test_ae': True,
            #                  'bool_save_ed': True, 'bool_test_ed': True,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

            # search_model(PATH_RESULT, 'model_8_5_4', input_dim_j, data_dict_j, data_dict_j, s_names_j[0][:-6],
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'batch_size': 8, 'bs_pred': 8,
            #                  'n_conv_encoder': 8, 'n_filters': 64, 'use_bias': False,
            #                  'rnn_type': 'BLSTM', 'latent_dim': 256, 'n_rnn_decoder': 4,
            #                  'bool_train': True, 'bool_test_ae': True,
            #                  'bool_save_ed': True, 'bool_test_ed': True,
            #                  'bool_clean_weight_file': True, 'bool_test_weight': True})

    logging.info('finished')
