# -*- coding: utf-8 -*-
"""
Created on Wed July 3 10:41:04 2024

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

Implement of CBIR training strategy in following article:
    H. Kim and J. W. Shin, “On Training Speech Separation Models With Various Numbers of Speakers,” IEEE Signal Process.
    Lett., vol. 30, pp. 1202–1206, 2023, doi: 10.1109/LSP.2023.3310881.

"""
# pylint: disable=line-too-long, logging-fstring-interpolation
# pylint: disable=too-many-arguments, too-many-branches, too-many-lines, too-many-locals, too-many-statements

import typing
import attr

import tensorflow

if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf
    from tensorflow import keras
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.keras.backend as K
else:
    import tensorflow as tf
    import keras
    from keras import backend as K

import keras.optimizers.schedules

from experiments.fuss.groupwise import apply as groupwise_apply
from experiments.fuss.model import weights_for_nonzero_refs
from experiments.metrics import neg_si_snr_stabilizer_loss_batch_tf
from experiments.KIM2023.identify_valid_output_channels import determine_active_channels


@attr.attrs
class HParams(object):
    """Model hyperparameters."""
    # A list of strings same length as signal_names specifying signal type, used for groupwise permutation-invariance.
    signal_types = attr.attrib(type=typing.List[typing.Text],
                               default=['background', 'foreground_1', 'foreground_2', 'foreground_3'])

    batch_size = attr.attrib(type=int, default=3)
    # max possible number of sources, equal to number of channels of the models
    num_sources = attr.attrib(type=int, default=4)
    # length of the input frame
    frame_length = attr.attrib(type=int, default=10547)
    eta = attr.attrib(type=typing.Tuple[float], default=(0.9502468705177307, 0.564647912979126, -1.))
    valid_frequency_channel = attr.attrib(type=typing.Tuple[float],
                                          default=(0.07692307692307693, 0.3076923076923077, 0.3076923076923077,
                                                   0.3076923076923077))


def get_kim_hparams():
    """Get model hyperparameters.
    Returns:
        HParams: model hyperparameters.
    """
    return HParams()


def loss_kim(model, hparams):
    """Loss function wrap for keras model.
    Args:
        model (keras.Model): keras model, for getting tensors inside the model.
        hparams (HParams): Model hyperparameters.
    """

    def loss_kim_keras(y_true, y_pred):
        """Loss function in kim for keras model.
        Args:
            y_true (keras.Tensor): true outputs.
            y_pred (keras.Tensor): predict outputs.
        Returns:
            loss_total: kim loss.
        """
        # convert [batch, frame_length, n_source] to [batch, n_source, frame_length]
        source_waveforms = tf.transpose(y_true, [0, 2, 1])
        separated_waveforms = tf.transpose(y_pred, [0, 2, 1])
        mixture_waveforms = tf.transpose(model.inputs[0], [0, 2, 1])

        batch_size = hparams.batch_size
        num_sources = hparams.num_sources
        frame_length = hparams.frame_length
        source_waveforms = tf.reshape(source_waveforms, (batch_size, num_sources, frame_length))
        separated_waveforms = tf.reshape(separated_waveforms, (batch_size, num_sources, frame_length))
        mixture_waveforms = tf.reshape(mixture_waveforms, (batch_size, 1, frame_length))

        signal_types = hparams.signal_types
        unique_signal_types = list(set(signal_types))
        eta = hparams.eta
        valid_frequency_channel = hparams.valid_frequency_channel

        # Initial loss functions for resolving permutations.
        loss_fns = {signal_type: neg_si_snr_stabilizer_loss_batch_tf for signal_type in unique_signal_types}

        separated_waveforms_nonzero = []
        separated_is_nonzero_masks = []
        for i in range(batch_size):
            mixture_waveform_i, separated_waveform_i = tf.squeeze(mixture_waveforms[i]), separated_waveforms[i]
            separated_is_nonzero_mask = determine_active_channels(mixture_waveform_i, separated_waveform_i,
                                                                  eta, valid_frequency_channel)
            separated_is_nonzero_mask_tensor = tf.expand_dims(tf.cast(separated_is_nonzero_mask, tf.float32), axis=-1)
            separated_waveforms_nonzero.append(tf.multiply(separated_waveform_i, separated_is_nonzero_mask_tensor))
            separated_is_nonzero_masks.append(separated_is_nonzero_mask)
        separated_waveforms_nonzero = tf.stack(separated_waveforms_nonzero, axis=0)
        separated_is_nonzero_masks = tf.stack(separated_is_nonzero_masks, axis=0)

        # Permute separated to match references through loss_fns.
        loss_total, _ = groupwise_apply(
            loss_fns, signal_types, source_waveforms,
            separated_waveforms_nonzero, unique_signal_types,
            batch_size, num_sources, frame_length)  # (batch, n_source)

        # When compute loss, compute true-positive non-zero outputs.
        source_is_nonzero_masks = weights_for_nonzero_refs(source_waveforms)
        loss_nonzero = 0.
        for i in range(batch_size):
            loss_total_i = loss_total[i]
            separated_is_nonzero_masks_i = separated_is_nonzero_masks[i]
            num_predict_active_i = tf.reduce_sum(tf.cast(separated_is_nonzero_masks_i, tf.int32))
            source_is_nonzero_masks_i = source_is_nonzero_masks[i]
            nums_real_active_i = tf.reduce_sum(tf.cast(source_is_nonzero_masks_i, tf.int32))
            loss_top_m_i = tf.sort(loss_total_i)[:tf.reduce_min([num_predict_active_i, nums_real_active_i])]  # min m
            loss_nonzero += tf.reduce_mean(loss_top_m_i)
        loss_nonzero /= tf.cast(batch_size, tf.float32)

        return loss_nonzero

    return loss_kim_keras


if __name__ == '__main__':
    import logging
    import numpy as np

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

    SCALER_DATA = 'max_one'
    SUB_SET_WAY = 'rand'
    SPLIT_WAY = 'split'
    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav',
                                   scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY, split_way=SPLIT_WAY)
    PATH_DATA_S = PATH_CLASS.path_source_root
    PATH_DATA = PATH_CLASS.path_source

    SET_NAMES = ['train', 'val', 'test']
    X_NAMES = ['X']
    X_SET_NAMES = [[f'{x_names_i}_{set_name_j}' for set_name_j in SET_NAMES] for x_names_i in X_NAMES]
    X_DICT = [dict(zip(x_set_names_j, read_datas(PATH_DATA, x_set_names_j))) for x_set_names_j in X_SET_NAMES]

    Z_NAMES = ['Z']
    Z_SET_NAMES_R = [[f'{z_name_i}_{name_set_j}_zero' for name_set_j in SET_NAMES] for z_name_i in Z_NAMES]
    Z_SET_NAMES = [[f'{z_name_i}_zero_{name_set_j}' for name_set_j in SET_NAMES] for z_name_i in Z_NAMES]
    Z_DICT = [dict(zip(z_set_names_s_j, read_datas(PATH_DATA, z_set_names_r_j)))
              for z_set_names_s_j, z_set_names_r_j in zip(Z_SET_NAMES, Z_SET_NAMES_R)]

    PATH_RESULT = '../../../result_separation_kim'
    mkdir(PATH_RESULT)

    dict_model_load = {'kim': loss_kim, 'loss_kim_keras': loss_kim}

    lr_i_j = [(i, j) for i in range(1, 2) for j in range(-3, -5, -1)]
    for (lr_i, lr_j) in lr_i_j:
        logging.info('model test')
        for x_name_i, x_dict_i, z_name_i, z_dict_i in zip(X_SET_NAMES, X_DICT, Z_SET_NAMES, Z_DICT):
            logging.info(x_dict_i.keys())

            input_dim = x_dict_i[x_name_i[-1]].shape[-1]  # (n_samples, 1, frame_length)
            if input_dim == 1:
                input_dim = x_dict_i[x_name_i[-1]].shape[-2]  # (n_samples, frame_length, 1)

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.squeeze(value)  # (n_samples, frame_length)

            for key, value in z_dict_i.items():
                z_dict_i[key] = np.squeeze(value).transpose((0, 2, 1))  # -> (n_samples, frame_length, 4)

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (n_samples, frame_length, 1)

            def adam_cosine_decay_restart_optimizer(lr_i_, lr_j_):
                initial_learning_rate = lr_i_ * (10 ** lr_j_)
                first_decay_steps = 4
                lr_decayed_fn = (
                    tf.keras.experimental.CosineDecayRestarts(
                        initial_learning_rate,
                        first_decay_steps))
                optimizer = keras.optimizers.Adam(learning_rate=lr_decayed_fn)
                return optimizer


            adam_cosine_decay_warm_restart_optimizer = adam_cosine_decay_restart_optimizer(lr_i, lr_j)

            # model_13 multiple decoder TasNet without mask
            search_model(PATH_RESULT, 'model_13_2_1', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                            'optimizer': adam_cosine_decay_warm_restart_optimizer,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'BLSTM', 'latent_dim': 200,
                            'n_block_encoder': 1, 'n_block_decoder': 1,
                            'is_multiple_decoder': True, 'use_mask': False,
                            'loss_name': 'kim',
                            'loss_func': loss_kim,
                            'user_metrics_name': ['loss_kim_keras'],
                            'user_metrics_func': [loss_kim],
                            'dict_model_load': dict_model_load,
                            'bool_num_padd': True,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            search_model(PATH_RESULT, 'model_13_3_1', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                            'optimizer': adam_cosine_decay_warm_restart_optimizer,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'dprnn', 'latent_dim': 200,
                            'n_block_encoder': 1, 'n_block_decoder': 1,
                            'loss_name': 'kim',
                            'loss_func': loss_kim,
                            'user_metrics_name': ['loss_kim_keras'],
                            'user_metrics_func': [loss_kim],
                            'dict_model_load': dict_model_load,
                            'bool_num_padd': True,
                            'is_multiple_decoder': True, 'use_mask': False,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # Multiple-Decoder Conv-TasNet without mask
            search_model(PATH_RESULT, 'model_15_2_6', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4,
                            'optimizer': adam_cosine_decay_warm_restart_optimizer,
                            'epochs': 100, 'batch_size': 6, 'bs_pred': 6,
                            'is_multiple_decoder': True, 'use_mask': False,
                            'n_conv_encoder': 1, 'n_filters_encoder': 64,
                            'n_channels_conv': 128, 'n_channels_bottleneck': 64, 'n_channels_skip': 64,
                            'n_layer_each_block': 5, 'n_block_encoder': 1, 'n_block_decoder': 2,
                            'loss_name': 'kim',
                            'loss_func': loss_kim,
                            'user_metrics_name': ['loss_kim_keras'],
                            'user_metrics_func': [loss_kim],
                            'dict_model_load': dict_model_load,
                            'bool_num_padd': True,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # # Multiple-Decoder Wave-U-Net without skip connections
            search_model(PATH_RESULT, 'model_21_6_10', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4,
                            'optimizer': adam_cosine_decay_warm_restart_optimizer,
                            'epochs': 1600, 'batch_size': 16, 'bs_pred': 16,
                            'n_pad_input': 13, 'num_layers': 4, 'batch_norm': True,
                            'use_skip': False, 'output_type': 'direct', 'output_activation': 'tanh',
                            'is_multiple_decoder': True,
                            'loss_name': 'kim',
                            'loss_func': loss_kim,
                            'user_metrics_name': ['loss_kim_keras'],
                            'user_metrics_func': [loss_kim],
                            'dict_model_load': dict_model_load,
                            'bool_num_padd': True,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

    print('finished')
