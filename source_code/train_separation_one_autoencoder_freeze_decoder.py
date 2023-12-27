# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:34:29 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, logging-fstring-interpolation
# pylint: disable=too-many-arguments, too-many-branches, too-many-lines, too-many-locals, too-many-statements

import logging
import os

import keras.losses
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
from keras import optimizers
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.models import load_model
import h5py
import numpy as np
import tables

from error import Error, ParameterError
from file_operation import list_files_end_str, mkdir
from loss_acc_separation import samerate_acc_d2
from prepare_data_shipsear_recognition_mix_s0tos3 import compute_chunk_size, read_datas, save_datas
from see_metric_single_source_ae_ns import ListDecodedFiles
from train_functions import output_history, save_model_struct
from train_single_source_autoencoder_ns import BuildModel, build_dict_model_load, clear_model_weight_file
from train_single_source_autoencoder_ns import data_save_reshape_ae, test_autoencoder, train_autoencoder
from train_single_source_autoencoder_ns_search_encoded_dense import compute_num_model, create_decoder_weight_model
from train_single_source_autoencoder_ns_search_encoded_dense import transpose_names_to_para_src


def build_freeze_decoder_model(num_model, path_weight_decoder_files, dict_model_load=None, **kwargs):
    """Build a new autoencoder model, load and freeze decoder weights while make encoder trainable.
    Args:
        num_model (int): the index number of model.
        path_weight_decoder_files (list[str]): full path name of the decoder models.
        dict_model_load (dict, optional): custom objects of model. Defaults to None.
    Returns:
        autoencoder (keras.models.Model): the build autoencoder model.
    """
    n_outputs = kwargs['n_outputs'] if 'n_outputs' in kwargs.keys() else None  # 4
    assert n_outputs == len(path_weight_decoder_files)

    if num_model in (1, 15, 21):
        n_no_para_decoder_layer = -0
    elif num_model in (8, 9, 13):
        n_no_para_decoder_layer = -2

    object_build_model = BuildModel(num_model, input_dim, **kwargs)
    autoencoder, _, n_encoder_layer, _ = object_build_model.build_model(**kwargs)

    logging.debug(f'autoencoder create \n {autoencoder.summary()}')
    for i, path_weight_decoder_file_i in enumerate(path_weight_decoder_files):
        model_weight = load_model(path_weight_decoder_file_i, custom_objects=dict_model_load)
        logging.debug(f'model_decoder loaded \n {model_weight.summary()}')
        for j, layer_j in enumerate(model_weight.layers[1:n_no_para_decoder_layer]):
            num_ae_layer_j = n_encoder_layer+1+n_outputs*j+i
            autoencoder.layers[num_ae_layer_j].trainable = False
            logging.debug(f' model_autoencoder layer decoder {i} layer {j}')
            logging.debug(f'autoencoder.layers[{j}] {autoencoder.layers[num_ae_layer_j]}')
            decoder_layer_weights = layer_j.get_weights()
            autoencoder.layers[num_ae_layer_j].set_weights(decoder_layer_weights)
    return autoencoder


def train_freeze_autoencoder(autoencoder, paras, x_dict, z_dict, path_save, modelname=None,
                             dict_model_load=None, fname_model_load=None):
    """Train autoencoder model.
    Args:
        autoencoder (keras.Model): a keras autoencoder model, no use when dict_model_load is not None.
        paras (dict): parameters for train model.
        x_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model input, data to predict.
        z_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model output, data predict target.
        path_save (str): where to save autoencoder output predict.
        modelname (str, optional): for autoencoder model name. Defaults to None.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
        fname_model_load (str, optional): file name of model to load. Defaults to None.
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

    dummy_input_train = np.ones(shape=(x_train.shape[0], 1), dtype=np.float32)
    dummy_input_val = np.ones(shape=(x_val.shape[0], 1), dtype=np.float32)

    path_check = os.path.join(path_save, 'auto_model', z_val_name[:-4], modelname)
    mkdir(path_check)
    check_filename = os.path.join(path_check, f'weights_{modelname}'+'_{epoch:02d}_{val_loss:.2f}.hdf5')
    checkpoint = ModelCheckpoint(filepath=check_filename, monitor='val_loss', mode='auto',
                                 verbose=1, period=1, save_best_only=True)
    path_board = os.path.join(path_save, 'tensorbord', modelname)
    mkdir(path_board)

    batch_size = paras['batch_size']
    shuffle = paras['shuffle'] if 'shuffle' in paras.keys() else True
    logging.info('start train autoencoder model with freeze decoder')
    history = model.fit(x=[x_train, dummy_input_train],
                        y=z_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        validation_data=([x_val, dummy_input_val], z_val),
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


def predict_freeze_autoencoder(model, z_test, path_save, save_name, dummy_input,
                               mode='batch_process', bs_pred=32, compile_model=False,
                               reshape_save=False):
    """Predict encoder and decoder.
    Args:
        model (keras.Model): a keras model, encoder or decoder.
        z_test (np.ndarray(float),shape=(n_sams,1,fl)): model input, data to predict.
        path_save (str): where to save predict outputs.
        save_name (str): path and name of the output file.
        dummy_input (np.ndarray): dummy input of the model.
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

    def func_predict(data_list):
        """Predict function.
        Args:
            data_list (list(np.ndarray(float))): data to be predict.
        Returns:
            data_pred (np.ndarray(float)): model predict output of data.
        """
        data = data_list[0]
        dummy_input = data_list[1]
        shape_input = tuple(K.int_shape(model.inputs[0])[1:])
        if data.shape[1:] != shape_input:
            # logging.debug(f'model input shape, {tuple(K.int_shape(model.inputs[0])[1:])}')
            data = data.reshape(data.shape[0:1]+shape_input)
        data_pred = model.predict([data, dummy_input], batch_size=bs_pred)
        if reshape_save:
            data_pred = data_save_reshape_ae(data_pred)
        return data_pred

    def save_process_batch(data_list, func, path_save, file_name, batch_num=200, save_key='data', mode_batch='batch',
                           chunk_type='back_dim', *args, **kwargs):
        """Process data by batch through func, save to path_save.
        Args:
            data (np.ndarray,shape==(nsam, - - )): data to save
            func (function): function to process data
            path_save (str): where to save data
            file_name (str): name of the saved file
            batch_num (int, optional): each batch process batch_num data
            save_key (str, optional): data save keyword. Defaults to 'data'.
            mode_batch (str, optional): use pytables(default) or h5py to save data. Defaults to 'batch'.
            chunk_type (str, optional): for compute chunk size to save .h5 file. Defaults to 'back_dim'.
        """
        dtype = kwargs['dtype'] if 'dtype' in kwargs.keys() else np.dtype('float32')

        for j in range(0, data_list[0].shape[0], batch_num):
            if j+batch_num > data_list[0].shape[0]:
                data_list_j = [data_i[j:] for data_i in data_list]
            else:
                data_list_j = [data_i[j:j+batch_num] for data_i in data_list]

            data_result = func(data_list_j, *args, **kwargs)
            chunk_size = compute_chunk_size(data_result, chunk_type)

            if mode_batch == 'batch':  # pytables
                with tables.open_file(os.path.join(path_save, f'{file_name}.hdf5'), 'a') as f_w:
                    if save_key not in f_w.root:
                        data_earray = f_w.create_earray(f_w.root, save_key,
                                                        atom=tables.Atom.from_dtype(dtype),
                                                        shape=((0,)+data_result.shape[1:]),
                                                        chunkshape=chunk_size,
                                                        filters=tables.Filters(complevel=9, complib='blosc'))
                    else:
                        data_earray = getattr(f_w.root, save_key)
                    data_earray.append(data_result)
            elif mode_batch == 'batch_h5py':  # h5py
                if j == 0:
                    with h5py.File(os.path.join(path_save, f'{file_name}.hdf5'), 'w') as f:
                        f.create_dataset(
                            save_key, data=data_result,
                            dtype=dtype,
                            chunks=chunk_size,
                            maxshape=((None,)+data_result.shape[1:]),
                            compression="gzip", compression_opts=9)
                else:
                    with h5py.File(os.path.join(path_save, f'{file_name}.hdf5'), 'a') as f:
                        f[save_key].resize(
                            (f[save_key].shape[0] + data_result.shape[0]), axis=0)
                        f[save_key][-data_result.shape[0]:] = data_result
            else:
                raise ParameterError('Invalid mode_batch keyword.')

    if mode == 'batch_process':
        save_process_batch([z_test, dummy_input], func_predict, path_save, save_name, bs_pred, mode_batch='batch_h5py')
    elif mode == 'batch_save':
        if z_test.shape[1:] != tuple(K.int_shape(model.inputs[0])[1:]):
            z_test = np.asarray(z_test).reshape(z_test.shape[0:1]+tuple(K.int_shape(model.inputs[0])[1:]))
        z_test_pred = model.predict([z_test, dummy_input], batch_size=bs_pred)
        if reshape_save:
            z_test_pred = data_save_reshape_ae(z_test_pred)
        save_datas({save_name: z_test_pred}, path_save, **{'mode_batch': 'batch_h5py'})


def test_freeze_autoencoder(x_dict, z_names=None, modelname=None, paras=None,
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
        dummy_input_set = np.ones(shape=(x_dataset.shape[0], 1), dtype=np.float32)
        predict_freeze_autoencoder(model, x_dataset, path_out, save_name, dummy_input_set,
                                   mode=mode_pred, bs_pred=bs_pred)


def train_predict_ae(autoencoder, paras, x_dict, z_dict, path_save, dict_model_load=None,
                     bool_train=True, bool_clean_weight_file=True, bool_predict=True, **kwargs):
    """Train and predict autoencoder models.
    Args:
        autoencoder (keras.Model): model of autoencoder.
        paras (dict): parameters for train model.
        x_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model input, data to predict.
        z_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): dict of model output, data to predict.
        path_save (str): where to save model.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
        bool_train (bool, optional): whether train ae model. Defaults to True.
        bool_clean_weight_file (bool, optional): whether clear checkpoint models. Defaults to True.
        bool_predict (bool, optional): whether predict ae model. Defaults to True.
    """
    modelname = kwargs['modelname'] if 'modelname' in kwargs.keys() else None

    if bool_train:
        train_autoencoder(autoencoder, paras, x_dict, z_dict, path_save, modelname=modelname)

    if bool_clean_weight_file or bool_predict:
        if modelname is None:
            i = paras['i']
            j = paras['j']
            epochs = paras['epochs']
            strj = f'n{str(j)[1:]}' if j < 0 else str(j)
            modelname = f'{i}_{strj}_{epochs}'
        _, z_val_name, _ = z_dict.keys()
        model_dirname = kwargs['model_dirname'] if 'model_dirname' in kwargs.keys() else 'auto_model'
        path_save_model = os.path.join(path_save, model_dirname, z_val_name[:-4], modelname)
        if not os.path.isdir(path_save_model):
            mkdir(path_save_model)

    if bool_clean_weight_file:
        clear_model_weight_file(path_save_model)
    if bool_predict:
        weight_file_name = list_files_end_str(path_save_model, 'hdf5', True)[-1]
        if not weight_file_name:
            raise Error('weight_file_names not found')
        test_autoencoder(x_dict, z_names=list(z_dict.keys()), path_save=path_save, paras=paras,
                         dict_model_load=dict_model_load, weight_file_name=weight_file_name)


def search_model(path_result_root, model_name, src_names, x_dict, z_dict, paras,
                 dict_model_load=None, bool_train=True, bool_clean_weight_file=True, bool_predict=True,
                 **kwargs):
    """For search best model.
    Args:
        path_result_root (os.Path): root directory where to save result.
        model_name (str): name of the model.
        src_names (list[str]): list of src names, e.g. ['Z_0_ns',...,'Z_4_ns'].
        x_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): data of the clear sources,
            which is the output vector in separation problem.
        z_dict (dict{str:np.ndarray(float),shape=(n_sams,1,fl)}): sum of the outputs,
            which is the input vector in separation problem.
        paras (dict): parameters for train model.
        dict_model_load (dict, optional): custom objects for load model. Defaults to None.
        bool_train (bool, optional): whether train ae model. Defaults to True.
        bool_clean_weight_file (bool, optional): whether clear checkpoint models. Defaults to True.
        bool_predict (bool, optional): whether predict ae model. Defaults to True.
    """
    path_result_model = os.path.join(path_result_root, model_name)
    path_save_load_decoder = os.path.join(path_result_model, 'load_decoder')
    mkdir(path_save_load_decoder)

    num_model = compute_num_model(model_name)

    if dict_model_load is None:
        dict_model_load = build_dict_model_load(num_model, **kwargs)
    if paras is None:
        object_build_model = BuildModel(num_model, input_dim, **kwargs)
        paras = object_build_model.build_model_paras(**kwargs)

    object_decoded_files = ListDecodedFiles(path_result_root, src_names)
    path_weight_files = object_decoded_files.filter_model(model_name, 'path_weight_files')  # [src][para][weights]
    name_model_src_paras = object_decoded_files.filter_model(model_name, 'name_model_src_paras')  # [src][para]

    name_paras = name_model_src_paras[0]  # [para]
    path_weight_files_paras = transpose_names_to_para_src(path_weight_files)  # [para][src]
    for name_para_i, path_weight_files_para_i in zip(name_paras, path_weight_files_paras):
        path_save_load_decoder_para_i = os.path.join(path_save_load_decoder, f'model_{name_para_i}')
        mkdir(path_save_load_decoder_para_i)
        # path_ae_model_i = os.path.join(path_save_load_decoder_para_i, 'auto_model', 'Z_zero', 'name_para_i')
        # mkdir(path_ae_model_i)
        # ae_model_name_i = os.path.join(path_ae_model_i, f'{name_para_i}_ae.h5')
        ae_model_name_i = os.path.join(path_save_load_decoder_para_i, f'{name_para_i}_ae.h5')

        if os.path.isfile(ae_model_name_i):
            ae_model_i = load_model(ae_model_name_i, custom_objects=dict_model_load)
        else:
            decoder_weight_model_names_i = []
            for path_weight_files_para_i_j in path_weight_files_para_i:  # [src]
                path_decoder_model_j, weight_file_name_j = os.path.split(path_weight_files_para_i_j)
                decoder_weight_model_name_j = os.path.join(path_decoder_model_j,
                                                           f'{name_para_i}_{weight_file_name_j}_decoder.h5')
                if not os.path.isfile(decoder_weight_model_name_j):
                    create_decoder_weight_model(path_save_model=path_decoder_model_j, modelname=name_para_i,
                                                dict_model_load=dict_model_load)
                decoder_weight_model_names_i.append(decoder_weight_model_name_j)

            ae_model_i = build_freeze_decoder_model(num_model, decoder_weight_model_names_i,
                                                    dict_model_load=dict_model_load, **kwargs)
            ae_model_i.save(ae_model_name_i)
            save_model_struct(ae_model_i, path_save_load_decoder_para_i, 'ae_model_struct')
            # ae_model_weights_name_i = os.path.join(path_save_load_decoder_para_i, f'{name_para_i}_weights.hdf5')
            # ae_model_i.save_weights(ae_model_weights_name_i)

        train_predict_ae(ae_model_i, paras, x_dict, z_dict, path_save_load_decoder_para_i,
                         dict_model_load=dict_model_load, bool_train=bool_train,
                         bool_clean_weight_file=bool_clean_weight_file, bool_predict=bool_predict, **kwargs)


if __name__ == '__main__':
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep

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

    SRC_NAMES = [f'Z_{i}_ns' for i in range(4)]  # [n_src]

    PATH_RESULT = '../result_separation_ae_ns_single'
    mkdir(PATH_RESULT)

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
                z_dict_i[key] = np.squeeze(value).transpose(0, 2, 1)  # -> (nsamples, frame_length, 4)
                logging.debug(f'z_i.name {key}')
                logging.debug(f'z_i.shape {z_dict_i[key].shape}')

            # search_model(PATH_RESULT, 'model_1_2_1', SRC_NAMES, x_dict_i, z_dict_i, None,
            #              bool_train=True, bool_clean_weight_file=True, bool_predict=True,
            #              **{'i': lr_i, 'j': lr_j, 'epochs': 50, 'encoding_dim': 256, 'batch_norm': True,
            #                 'n_outputs': 4})

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (nsamples, frame_length, 1)
                logging.debug(f'x_i.name {key}')
                logging.debug(f'x_i.shape {x_dict_i[key].shape}')

            # search_model(PATH_RESULT, 'model_8_2_1', SRC_NAMES, x_dict_i, z_dict_i, None,
            #              bool_train=True, bool_clean_weight_file=True, bool_predict=True,
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                 'n_conv_encoder': 1, 'n_filters': 64, 'encoder_multiple_out': True,
            #                 'rnn_type': 'BLSTM', 'latent_dim': 256, 'n_rnn_decoder': 1})

            # search_model(PATH_RESULT, 'model_8_4_1', SRC_NAMES, x_dict_i, z_dict_i, None,
            #              bool_train=True, bool_clean_weight_file=True, bool_predict=True,
            #              **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
            #                 'n_conv_encoder': 1, 'n_filters': 64,
            #                 'use_bias': False,'encoder_multiple_out': True,
            #                 'rnn_type': 'BLSTM', 'latent_dim': 256, 'n_rnn_decoder': 1})

            # model_13 multiple decoder RNN TasNet without mask
            search_model(PATH_RESULT, 'model_13_2_1', SRC_NAMES, x_dict_i, z_dict_i, None,
                         bool_train=True, bool_clean_weight_file=True, bool_predict=True,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                             'n_conv_encoder': 1, 'n_filters_conv': 64,
                             'block_type': 'BLSTM', 'latent_dim': 200,
                             'n_block_encoder': 1, 'n_block_decoder': 1,
                             'model_type': 'ae', 'encoder_multiple_out': True,
                             'is_multiple_decoder': True, 'use_mask': False})

            search_model(PATH_RESULT, 'model_13_3_1', SRC_NAMES, x_dict_i, z_dict_i, None,
                         bool_train=True, bool_clean_weight_file=True, bool_predict=True,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 50, 'batch_size': 8, 'bs_pred': 8,
                            'bool_num_padd': True,
                             'n_conv_encoder': 1, 'n_filters_conv': 64,
                             'block_type': 'dprnn', 'latent_dim': 200,
                             'n_block_encoder': 1, 'n_block_decoder': 1,
                             'model_type': 'ae', 'encoder_multiple_out': True,
                             'is_multiple_decoder': True, 'use_mask': False})

            # Multiple-Decoder Conv-Tasnet without mask
            search_model(PATH_RESULT, 'model_15_2_6', SRC_NAMES, x_dict_i, z_dict_i, None,
                         bool_train=True, bool_clean_weight_file=True, bool_predict=True,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 200, 'batch_size': 6, 'bs_pred': 6,
                            'n_conv_encoder': 1, 'n_filters_encoder': 64,
                            'n_channels_conv': 128, 'n_channels_bottleneck': 64, 'n_channels_skip': 64,
                            'n_layer_each_block': 5, 'n_block_encoder': 1, 'n_block_decoder': 2,
                            'model_type': 'ae', 'encoder_multiple_out': True,
                            'is_multiple_decoder': True, 'use_mask': False})

            # Multiple-Decoder Wave-U-Net without skip connections
            search_model(PATH_RESULT, 'model_21_6_10', SRC_NAMES, x_dict_i, z_dict_i, None,
                         bool_train=True, bool_clean_weight_file=True, bool_predict=True,
                         **{'i': lr_i, 'j': -4, 'n_outputs': 4, 'epochs': 800, 'batch_size': 16, 'bs_pred': 16,
                            'n_pad_input': 13, 'num_layers': 4,
                            'use_skip': False, 'output_type': 'direct', 'output_activation': 'tanh',
                            'is_multiple_decoder': True,
                            'model_type': 'ae', 'encoder_multiple_out': True})

    logging.info('finished')
