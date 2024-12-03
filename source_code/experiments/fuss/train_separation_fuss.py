# -*- coding: utf-8 -*-
"""
Created on Wed July 3 10:41:04 2024

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

Run FUSS training strategy in following article:
    S. Wisdom et al., “What’s all the Fuss about Free Universal Sound Separation Data?,” in ICASSP 2021 - 2021 IEEE
    International Conference on Acoustics, Speech and Signal Processing (ICASSP), Toronto, ON, Canada: IEEE, Jun. 2021,
    pp. 186–190. doi: 10.1109/ICASSP39728.2021.9414774.
"""
# pylint: disable=line-too-long, logging-fstring-interpolation
# pylint: disable=too-many-arguments, too-many-branches, too-many-lines, too-many-locals, too-many-statements

import attr
import typing

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


from experiments.fuss.model import log_mse_loss, weights_for_signal_type, weights_for_nonzero_refs
from experiments.fuss.groupwise import apply as groupwise_apply


@attr.attrs
class HParams(object):
    """Model hyperparameters."""
    # Weight on loss component for zero reference signals.
    loss_zero_ref_weight = attr.attrib(type=float, default=1.0)
    # mix_weights_type = Type of weights to use for mixture consistency. Options:
    #     ``: No mixture consistency.
    #     `uniform`: Uniform weight of 1 / num_sources (also called unweighted).
    #     `magsq`: Weight for source j is \sum_{mic, time} \hat{x}_j ^ 2
    #              over \sum_{source, mic, time} \hat{x}_j ^ 2.
    #     `pred_source`: Predict weights with shape (batch, source, 1).
    mix_weights_type = attr.attrib(type=typing.Text, default='pred_source')
    # List of signal names, e.g. ['signal_1', 'signal_2'].
    signal_names = attr.attrib(type=typing.List[typing.Text],
                               default=['background', 'foreground_1',
                                        'foreground_2', 'foreground_3'])
    # A list of strings same length as signal_names specifying signal type, used
    # for groupwise permutation-invariance.
    signal_types = attr.attrib(type=typing.List[typing.Text],
                               default=['background', 'foreground_1',
                                        'foreground_2', 'foreground_3'])
    # List of signal types to consider as having a reference of zero for the
    # purposes of loss calculations and summary reports.
    # It could be beneficial to add noise, or sensor_noise to this list.
    ref_zero_signal_types = attr.attrib(
        type=typing.Optional[typing.List[typing.Text]], default=None)

    max_snr = attr.attrib(type=float, default=30.0)
    max_snr_for_zero_sources = attr.attrib(type=float, default=20.0)
    loss_weights = attr.attrib(default=None)
    batch_size = attr.attrib(type=int, default=3)
    # max possible number of sources, equal to number of channels of the models
    num_sources = attr.attrib(type=int, default=4)
    # length of the input frame
    frame_length = attr.attrib(type=int, default=10547)


def get_fuss_hparams():
    """Get model hyperparameters.
    Returns:
        HParams: model hyperparameters.
    """
    return HParams()


def loss_fuss(model, hparams):
    """Loss function wrap for keras model.
    Args:
        model (keras.Model): keras model, for getting tensors inside the model.
        hparams (HParams): Model hyperparameters.
    """
    def loss_fuss_keras(y_true, y_pred):
        """Loss function in fuss for keras model.
        Args:
            y_true (keras.Tensor): true outputs.
            y_pred (keras.Tensor): predict outputs.
        Returns:
            loss_total: fuss loss.
        """
        # convert [batch, frame_length, n_source] to [batch, n_source, frame_length]
        source_waveforms = tf.transpose(y_true, [0, 2, 1])
        separated_waveforms = tf.transpose(y_pred, [0, 2, 1])

        loss_zero_ref_weight = hparams.loss_zero_ref_weight
        # mixture_waveforms = hparams.mixture_waveforms
        mixture_waveforms = model.inputs
        max_snr = hparams.max_snr
        max_snr_for_zero_sources = hparams.max_snr_for_zero_sources
        ref_zero_signal_types = hparams.ref_zero_signal_types
        signal_types = hparams.signal_types
        loss_weights = hparams.loss_weights
        batch_size = hparams.batch_size
        num_sources = hparams.num_sources
        frame_length = hparams.frame_length
        if not loss_weights:
            # loss_weights = [1.0] * num_sources
            loss_weights = tf.ones(shape=(1, num_sources), dtype=tf.float32)
        loss_weights = tf.broadcast_to(loss_weights, (batch_size, num_sources))
        unique_signal_types = list(set(signal_types))
        # Initial loss functions for resolving permutations.
        loss_fns = {signal_type: log_mse_loss for signal_type in unique_signal_types}

        # Permutation-invariant training requires realigning the sources to match
        # the references, both in loss computations and computing summary metrics.

        # Build loss split between all-zero and nonzero reference signals.
        # Some signal types may be considered as always being zero for loss and
        # summary purposes.
        source_is_considered_zero = weights_for_signal_type(
            batch_size, signal_types, ref_zero_signal_types)
        source_is_nonzero = weights_for_nonzero_refs(
            source_waveforms, consider_as_zero=source_is_considered_zero)
        count_nonzero = tf.reduce_sum(
            tf.cast(source_is_nonzero, tf.float32))
        source_is_zero = tf.logical_not(source_is_nonzero)
        total_count = tf.cast(batch_size * num_sources, tf.float32)
        count_zero = total_count - count_nonzero

        # Waveforms with nonzero references.
        source_waveforms_nonzero = tf.boolean_mask(
            source_waveforms, source_is_nonzero)[:, tf.newaxis]

        # Since the sources are re-arranged when nonzero sources in the whole batch
        # are picked using a boolean_mask, we need to also pick and re-arrange
        # the loss weights corresponding to them.
        loss_weights_nonzero = tf.boolean_mask(
            loss_weights, source_is_nonzero)[:, tf.newaxis]

        source_waveforms_zero = tf.boolean_mask(
            source_waveforms, source_is_zero)[:, tf.newaxis]

        loss_weights_zero = tf.boolean_mask(
            loss_weights, source_is_zero)[:, tf.newaxis]

        # For loss for zero references.
        if loss_zero_ref_weight:
            mixture_waveforms = tf.reshape(mixture_waveforms, (batch_size, frame_length))
            mixture_waveforms_zero = tf.boolean_mask(
                tf.reshape(tf.tile(mixture_waveforms, (1, num_sources)),
                           (batch_size, num_sources, frame_length)),
                source_is_zero)[:, tf.newaxis]

        # Permute separated to match references through loss_fns.
        _, separated_waveforms = groupwise_apply(
            loss_fns, signal_types, source_waveforms,
            separated_waveforms, unique_signal_types,
            batch_size, num_sources, frame_length)
        separated_waveforms_nonzero = tf.boolean_mask(
            separated_waveforms, source_is_nonzero)[:, tf.newaxis]
        separated_waveforms_zero = tf.boolean_mask(
            separated_waveforms, source_is_zero)[:, tf.newaxis]

        loss_total = 0
        # Use eventual loss function as log_mse_loss.
        # Loss for zero references only if loss_zero_ref_weight is nonzero.
        if loss_zero_ref_weight:
            def loss_zero_fn():
                return tf.reduce_sum(loss_weights_zero * log_mse_loss(
                    source_waveforms_zero,
                    separated_waveforms_zero,
                    max_snr=max_snr_for_zero_sources,
                    bias_ref_signal=mixture_waveforms_zero))

            calc_losses_zero_ref = tf.logical_and(count_zero > 0,
                                                  count_zero < total_count)
            loss_zero = tf.cond(calc_losses_zero_ref, loss_zero_fn, lambda: 0.0)
            loss_weight = loss_zero_ref_weight / total_count
            loss_zero = tf.identity(loss_weight * loss_zero, name='loss_ref_zero')
            loss_total += loss_zero

        # Loss for nonzero references.
        def loss_nonzero_fn():
            return tf.reduce_sum(loss_weights_nonzero * log_mse_loss(
                source_waveforms_nonzero,
                separated_waveforms_nonzero,
                max_snr=max_snr))

        loss_nonzero = tf.cond(count_nonzero > 0., loss_nonzero_fn,
                               lambda: 0.)
        loss_nonzero = tf.identity(loss_nonzero / total_count,
                                   name='loss_ref_nonzero')
        loss_total += loss_nonzero

        return loss_total

    return loss_fuss_keras


if __name__ == '__main__':
    import logging
    import numpy as np

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

    PATH_RESULT = '../../../result_separation_fuss'
    mkdir(PATH_RESULT)

    dict_model_load = {'fuss': loss_fuss,
                       'loss_fuss_keras': loss_fuss}

    lr_i_j = [(i, j) for i in range(1, 2) for j in range(-3, -4, -1)]
    for (lr_i, lr_j) in lr_i_j:
        logging.info('model test')
        for x_name_i, x_dict_i, z_name_i, z_dict_i in zip(X_SET_NAMES, X_DICT, Z_SET_NAMES, Z_DICT):
            logging.info(x_dict_i.keys())

            input_dim = x_dict_i[x_name_i[-1]].shape[-1]  # (n_samples, 1, frame_length)
            if input_dim == 1:
                input_dim = x_dict_i[x_name_i[-1]].shape[-2]  # (n_samples, frame_length, 1)
            logging.debug(f'input_dim {input_dim}')

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.squeeze(value)  # (n_samples, frame_length)
                logging.debug(f'x_i.shape {x_dict_i[key].shape}')

            for key, value in z_dict_i.items():
                z_dict_i[key] = np.squeeze(value).transpose((0, 2, 1))  # -> (n_samples, frame_length, 4)
                logging.debug(f'z_i.name {key}')
                logging.debug(f'z_i.shape {z_dict_i[key].shape}')

            for key, value in x_dict_i.items():
                x_dict_i[key] = np.expand_dims(np.squeeze(value), axis=-1)  # (n_samples, frame_length, 1)
                logging.debug(f'x_i.name {key}')
                logging.debug(f'x_i.shape {x_dict_i[key].shape}')

            # model_13 multiple decoder TasNet without mask
            search_model(PATH_RESULT, 'model_13_2_1', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'BLSTM', 'latent_dim': 200,
                            'n_block_encoder': 1, 'n_block_decoder': 1,
                            'is_multiple_decoder': True, 'use_mask': False,
                            'loss_name': 'fuss',
                            'loss_func': loss_fuss,
                            'user_metrics_name': ['loss_fuss_keras'],
                            'user_metrics_func': [loss_fuss],
                            'dict_model_load': dict_model_load,
                            'bool_num_padd': True,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            search_model(PATH_RESULT, 'model_13_3_1', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': lr_j, 'n_outputs': 4, 'epochs': 100, 'batch_size': 8, 'bs_pred': 8,
                            'n_conv_encoder': 1, 'n_filters_conv': 64,
                            'block_type': 'dprnn', 'latent_dim': 200,
                            'n_block_encoder': 1, 'n_block_decoder': 1,
                            'loss_name': 'fuss',
                            'loss_func': loss_fuss,
                            'user_metrics_name': ['loss_fuss_keras'],
                            'user_metrics_func': [loss_fuss],
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
                            'epochs': 200, 'batch_size': 6, 'bs_pred': 6,
                            'is_multiple_decoder': True, 'use_mask': False,
                            'n_conv_encoder': 1, 'n_filters_encoder': 64,
                            'n_channels_conv': 128, 'n_channels_bottleneck': 64, 'n_channels_skip': 64,
                            'n_layer_each_block': 5, 'n_block_encoder': 1, 'n_block_decoder': 2,
                            'loss_name': 'fuss',
                            'loss_func': loss_fuss,
                            'user_metrics_name': ['loss_fuss_keras'],
                            'user_metrics_func': [loss_fuss],
                            'dict_model_load': dict_model_load,
                            'bool_num_padd': True,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

            # Multiple-Decoder Wave-U-Net without skip connections
            search_model(PATH_RESULT, 'model_21_6_10', input_dim, x_dict_i, z_dict_i, None,
                         **{'i': lr_i, 'j': -4, 'n_outputs': 4,
                            'epochs': 800, 'batch_size': 16, 'bs_pred': 16,
                            'n_pad_input': 13, 'num_layers': 4, 'batch_norm': True,
                            'use_skip': False, 'output_type': 'direct', 'output_activation': 'tanh',
                            'is_multiple_decoder': True,
                            'loss_name': 'fuss',
                            'loss_func': loss_fuss,
                            'user_metrics_name': ['loss_fuss_keras'],
                            'user_metrics_func': [loss_fuss],
                            'dict_model_load': dict_model_load,
                            'bool_num_padd': True,
                            'bool_train': True, 'bool_test_ae': False,
                            'bool_save_ed': False, 'bool_test_ed': False,
                            'bool_clean_weight_file': True, 'bool_test_weight': True,
                            'bool_test_ae_w': True, 'bool_test_ed_w': False})

    print('finished')
