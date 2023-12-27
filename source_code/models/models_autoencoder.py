# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:38:49 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, too-many-arguments, too-many-branches, too-many-locals, too-many-statements
# pylint: disable=invalid-name, logging-fstring-interpolation, no-member, useless-object-inheritance
import logging

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
from keras.layers import Activation, BatchNormalization, Bidirectional, Concatenate, Conv1D, Conv2DTranspose
from keras.layers import Dense, Dropout, Input, Lambda, LeakyReLU, LSTM, MaxPool1D, Multiply, PReLU, Permute
from keras.layers import ReLU, RepeatVector, Reshape, TimeDistributed, UpSampling1D
from keras.layers.convolutional import ZeroPadding1D
from keras.models import Model
import numpy as np
if tf.__version__ < '1.15':
    from .reconstruction_ops import overlap_and_add
if keras.__version__ < '2.3.1':
    from keras_layer_normalization import LayerNormalization
else:
    from keras.layers import LayerNormalization

from .dprnn.block import DprnnBlock, DprnnBlockModel, dprnn_block
from .conv_tasnet.blocks import tcn_branch, normal_layer
from .wave_u_net.layers import AudioClipLayer, CropLayer, CropConcatLayer, InterpolationLayer
from error import ParameterError


def build_model_1(input_dim, encoding_dim, output_dim, act_c, n_nodes=[8, 4, 2], batch_norm=False, layer_norm=False,
                  n_outputs=1, use_bias=True):
    """Mlp autoencoder."""
    input_frames = Input(shape=(input_dim, ))

    x = input_frames
    for n_node in n_nodes:
        x = Dense(int(encoding_dim * n_node), activation=act_c, use_bias=use_bias)(x)
        if batch_norm:
            x = BatchNormalization()(x)
        if layer_norm:
            x = LayerNormalization(center=False, scale=False)(x)
        if batch_norm and layer_norm:
            logging.warning('batch normalization and layer normalizaion both used')
    encoded = Dense(encoding_dim, activation=act_c, use_bias=use_bias)(x)

    def decoder_func(inputs):
        decoded_srcs = []
        for i in range(n_outputs):
            x = inputs
            for j in range(len(n_nodes)-1, -1, -1):
                x = Dense(int(encoding_dim * n_nodes[j]), activation=act_c, use_bias=use_bias)(x)
                if batch_norm:
                    x = BatchNormalization()(x)
                if layer_norm:
                    x = LayerNormalization(center=False, scale=False)(x)
            decoded_src_i = Dense(output_dim, activation='tanh', use_bias=use_bias)(x)
            decoded_srcs.append(decoded_src_i)
        if n_outputs > 1:
            if K.ndim(decoded_srcs[0]) == 2:
                for i, decoded_src_i in enumerate(decoded_srcs):
                    decoded_srcs[i] = Reshape(tuple(K.int_shape(decoded_src_i)[1:])+(1,))(decoded_src_i)
            decoded = Concatenate(axis=-1)(decoded_srcs)
        else:
            decoded = decoded_srcs[0]
        return decoded

    autoencoder = Model(input_frames, decoder_func(encoded))
    logging.debug(autoencoder.summary())

    decoder_inputs = Input(shape=K.int_shape(encoded)[1:], dtype=K.dtype(encoded))
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs))
    logging.debug(decoder.summary())

    n_layer_dense = 1
    if batch_norm:
        n_layer_dense += 1
    if layer_norm:
        n_layer_dense += 1
    n_encoder_layer = len(n_nodes)*n_layer_dense+1
    n_decoder_layer = n_outputs * ((len(n_nodes)-1)*n_layer_dense+1)
    if n_outputs > 1:
        n_decoder_layer += n_outputs  # reshape
        n_decoder_layer += 1  # concat

    return autoencoder, decoder, n_encoder_layer, n_decoder_layer


# convolutional_autoencoder
def conv1d(x, n_filters, kernel_size=3, strides=1, padding='same',
           kernel_initializer='glorot_uniform', batch_norm=True, act_c='relu', use_bias=True):
    """Conv1D + BN + 'relu'."""
    x = Conv1D(n_filters, kernel_size=kernel_size, strides=strides, padding=padding,
               kernel_initializer=kernel_initializer, use_bias=use_bias)(x)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act_c)(x)
    return x


def conv1d_transpose(input_tensor, filters, kernel_size, strides=None, use_bias=True,
                     activation='relu', padding='same', kernel_initializer='glorot_uniform'):
    """Cov1d transpose (unti-conv)."""
    if strides is None:
        strides = kernel_size
    x = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    x = Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1),
                        activation=activation, use_bias=use_bias,
                        padding=padding, kernel_initializer=kernel_initializer)(x)
    x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
    return x


def deconv1d(layer_input, filters, skip_input=None, upsize=2, kernel_size=None, strides=None,
             act_c='relu', use_bias=True, dropout_rate=None, batch_norm=True, up_sam_type='conv1dtranspose'):
    """Layers used during upsampling"""
    if skip_input is not None:
        logging.debug(f'layer_input {layer_input}')
        logging.debug(f'skip_input {skip_input}')
        x = Concatenate()([layer_input, skip_input])
        logging.debug(f'x {x}')
    else:
        x = layer_input
    if up_sam_type == 'upsampling':
        x = UpSampling1D(size=upsize)(x)
        x = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding='same',
                   activation=act_c, use_bias=use_bias)(x)
    elif up_sam_type == 'conv1dtranspose':
        x = conv1d_transpose(layer_input, filters, kernel_size, strides, activation=act_c, use_bias=use_bias)
    else:
        raise ValueError('Wrong upsampling type.')
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    if batch_norm:
        x = BatchNormalization()(x)

    return x


def build_model_2(input_dim, n_pad_input=13, n_filters=16, act_c='relu', use_bias=True,
                  dropout=0.1, batch_norm=True, up_sam_type='conv1dtranspose'):
    """1D Unet CNN without concat."""

    input_frames = Input(shape=(input_dim, 1))
    logging.debug(input_frames)
    x = ZeroPadding1D((0, n_pad_input))(input_frames)
    logging.debug(x)

    # Encoder
    for nf_i in [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 16, 16]:
        x = conv1d(x, n_filters*nf_i, 3, 1, padding='same',
                   kernel_initializer='he_normal', batch_norm=batch_norm, act_c=act_c)
        x = MaxPool1D(2, padding='same')(x)

    for nf_i in [16]:
        x = conv1d(x, n_filters*nf_i, 3, 1, padding='valid', use_bias=use_bias,
                   kernel_initializer='he_normal', batch_norm=batch_norm, act_c=act_c)

    encoded = x
    logging.debug(encoded)

    def decoder_func(inputs):
        x = inputs
        for nf_i in [16]:
            x = deconv1d(x, n_filters*nf_i, kernel_size=1, strides=3, use_bias=use_bias,
                         dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 3

        for nf_i in [16, 16]:
            x = deconv1d(x, n_filters*nf_i, kernel_size=2, strides=2, use_bias=use_bias,
                         dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)

        for nf_i in [16, 16]:
            x = Conv1D(n_filters*nf_i, kernel_size=2, strides=1, padding='valid',
                       use_bias=use_bias, activation=act_c)(x)  # 12->11, 22->21
            x = deconv1d(x, n_filters*nf_i, kernel_size=2, strides=2, use_bias=use_bias,
                         dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)

        for nf_i in [8, 8]:
            x = deconv1d(x, n_filters*nf_i, kernel_size=2, strides=2, use_bias=use_bias,
                         dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)
            x = Conv1D(n_filters*nf_i, kernel_size=2, strides=1, use_bias=use_bias,
                       padding='valid', activation=act_c)(x)  # 84->83, 166->165

        for nf_i in [4, 4, 2, 2, 1, 1]:
            x = deconv1d(x, n_filters*nf_i, kernel_size=2, strides=2, use_bias=use_bias,
                         dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)

        x = Conv1D(1, 1, use_bias=use_bias, activation='tanh')(x)
        logging.debug(f'x {x}')
        decoded = Lambda(lambda x: x[:, 0:input_dim, :])(x)
        logging.debug(f'decoded {decoded}')
        return decoded

    autoencoder = Model(input_frames, decoder_func(encoded))
    logging.debug(autoencoder.summary())

    decoder_inputs = Input(shape=K.int_shape(encoded)[1:], dtype=K.dtype(encoded))
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs))
    logging.debug(decoder.summary())

    n_conv = 2
    if batch_norm:
        n_conv += 1
    if batch_norm:
        n_encoder_layer = 1 + (n_conv+1)*12 + n_conv  # padding, conv+pool, conv
    else:
        n_encoder_layer = 1 + (n_conv+1)*12 + n_conv  # padding, conv+pool, conv

    if up_sam_type == 'conv1dtranspose':
        n_deconv = 3  # num of layers of deconv1d
    elif up_sam_type == 'upsampling':
        n_deconv = 2
    if dropout:
        n_deconv += 1
    if batch_norm:
        n_deconv += 1
    n_decoder_layer = n_deconv*3 + (1+n_deconv)*2 + (n_deconv+1)*2 + n_deconv*6 + 2

    return autoencoder, decoder, n_encoder_layer, n_decoder_layer


def build_model_3(input_dim, n_pad_input=13, n_filters=16, act_c='relu', use_bias=True,
                  dropout=None, batch_norm=True, up_sam_type='conv1dtranspose'):
    """1D Unet with concat."""

    input_frames = Input(shape=(input_dim, 1))
    logging.debug(input_frames)
    x = ZeroPadding1D((0, n_pad_input))(input_frames)  # 10560
    logging.debug(x)

    # Encoder
    def layer_encoder(x, n_filters, nf_i, batch_norm, act_c, use_bias):
        x = conv1d(x, n_filters*nf_i, 3, 1, padding='same', use_bias=use_bias,
                   kernel_initializer='he_normal', batch_norm=batch_norm, act_c=act_c)
        x = MaxPool1D(2, padding='same')(x)
        return x

    x1 = layer_encoder(x, n_filters, 1, batch_norm, act_c, use_bias)  # 5280
    x2 = layer_encoder(x1, n_filters, 1, batch_norm, act_c, use_bias)  # 2640
    x3 = layer_encoder(x2, n_filters, 2, batch_norm, act_c, use_bias)  # 1320
    x4 = layer_encoder(x3, n_filters, 2, batch_norm, act_c, use_bias)  # 660
    x5 = layer_encoder(x4, n_filters, 4, batch_norm, act_c, use_bias)  # 330
    x6 = layer_encoder(x5, n_filters, 4, batch_norm, act_c, use_bias)  # 165
    x7 = layer_encoder(x6, n_filters, 8, batch_norm, act_c, use_bias)  # 83
    x8 = layer_encoder(x7, n_filters, 8, batch_norm, act_c, use_bias)  # 42
    x9 = layer_encoder(x8, n_filters, 16, batch_norm, act_c, use_bias)  # 21
    x10 = layer_encoder(x9, n_filters, 16, batch_norm, act_c, use_bias)  # 11
    x11 = layer_encoder(x10, n_filters, 16, batch_norm, act_c, use_bias)  # 6
    x12 = layer_encoder(x11, n_filters, 16, batch_norm, act_c, use_bias)  # 3
    logging.debug(f'x12 {str(x12)}')
    x13 = conv1d(x12, n_filters*16, 3, 1, padding='valid', use_bias=use_bias,
                 kernel_initializer='he_normal', batch_norm=batch_norm, act_c=act_c)  # 1

    encoded = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13]
    logging.debug(encoded)

    def decoder_func(inputs):
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13 = inputs
        d1 = deconv1d(x13, n_filters*16, kernel_size=1, strides=3, use_bias=use_bias,
                      dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 3
        logging.debug(f'd1 {d1}')
        d2 = deconv1d(d1, n_filters*16, skip_input=x12, kernel_size=2, strides=2, use_bias=use_bias,
                      dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 6
        d3 = deconv1d(d2, n_filters*16, skip_input=x11, kernel_size=2, strides=2, use_bias=use_bias,
                      dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 12
        d4 = Conv1D(n_filters*16, kernel_size=2, strides=1, padding='valid',
                    use_bias=use_bias, activation=act_c)(d3)  # 12->11
        d5 = deconv1d(d4, n_filters*16, skip_input=x10, kernel_size=2, strides=2, use_bias=use_bias,
                      dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 22
        d6 = Conv1D(n_filters*16, kernel_size=2, strides=1, padding='valid',
                    use_bias=use_bias, activation=act_c)(d5)  # 22->21
        d7 = deconv1d(d6, n_filters*16, skip_input=x9, kernel_size=2, strides=2, use_bias=use_bias,
                      dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 42
        d8 = deconv1d(d7, n_filters*8, skip_input=x8, kernel_size=2, strides=2, use_bias=use_bias,
                      dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 84
        d9 = Conv1D(n_filters*8, kernel_size=2, strides=1, padding='valid',
                    use_bias=use_bias, activation=act_c)(d8)  # 84->83
        d10 = deconv1d(d9, n_filters*8, skip_input=x7, kernel_size=2, strides=2, use_bias=use_bias,
                       dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 166
        d11 = Conv1D(n_filters*8, kernel_size=2, strides=1, padding='valid',
                     use_bias=use_bias, activation=act_c)(d10)  # 166->165
        d12 = deconv1d(d11, n_filters*4, skip_input=x6, kernel_size=2, strides=2, use_bias=use_bias,
                       dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 330
        d13 = deconv1d(d12, n_filters*4, skip_input=x5, kernel_size=2, strides=2, use_bias=use_bias,
                       dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 660
        d14 = deconv1d(d13, n_filters*2, skip_input=x4, kernel_size=2, strides=2, use_bias=use_bias,
                       dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 1320
        d15 = deconv1d(d14, n_filters*2, skip_input=x3, kernel_size=2, strides=2, use_bias=use_bias,
                       dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 2640
        d16 = deconv1d(d15, n_filters*1, skip_input=x2, kernel_size=2, strides=2, use_bias=use_bias,
                       dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 5280
        d17 = deconv1d(d16, n_filters*1, skip_input=x1, kernel_size=2, strides=2, use_bias=use_bias,
                       dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 10560

        x = Conv1D(1, 1, use_bias=use_bias, activation='tanh')(d17)
        logging.debug(f'x {x}')
        decoded = Lambda(lambda x: x[:, 0:input_dim, :])(x)
        logging.debug(f'decoded {decoded}')
        return decoded

    autoencoder = Model(input_frames, decoder_func(encoded))
    logging.debug(autoencoder.summary())

    decoder_inputs = [Input(shape=K.int_shape(input_i)[1:], dtype=K.dtype(input_i))
                      for input_i in encoded]
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs))
    logging.debug(decoder.summary())

    n_conv = 2
    if batch_norm:
        n_conv += 1
    n_encoder_layer = 1 + (n_conv+1)*12 + n_conv  # padding, conv+pool, conv

    if up_sam_type == 'conv1dtranspose':
        n_deconv = 3  # num of layers of deconv1d
    elif up_sam_type == 'upsampling':
        n_deconv = 2
    if dropout:
        n_deconv += 1
    if batch_norm:
        n_deconv += 1
    n_decoder_layer = n_deconv*13 + 5 + 1  # unti_conv + conv + lambda

    return autoencoder, decoder, n_encoder_layer, n_decoder_layer


def build_model_4(input_dim, latent_dim, n_pad_input=13, n_filters=16, act_c='relu', batch_norm=True,
                  use_bias=True, act_r='tanh'):
    """1D Unet CNN without concat + LSTM autoenconder."""

    input_frames = Input(shape=(input_dim, 1))
    logging.debug(input_frames)
    x = ZeroPadding1D((0, n_pad_input))(input_frames)
    logging.debug(x)

    # Encoder
    for nf_i in [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 16, 16]:
        x = conv1d(x, n_filters*nf_i, 3, 1, padding='same', use_bias=use_bias,
                   kernel_initializer='he_normal', batch_norm=batch_norm, act_c=act_c)
        x = MaxPool1D(2, padding='same')(x)

    for nf_i in [16]:
        x = conv1d(x, n_filters*nf_i, 3, 1, padding='valid', use_bias=use_bias,
                   kernel_initializer='he_normal', batch_norm=batch_norm, act_c=act_c)
    x = Reshape((n_filters*16, 1))(x)
    encoded = LSTM(latent_dim, use_bias=use_bias, activation=act_r)(x)

    def decoder_func(inputs):
        x = inputs
        x = RepeatVector(input_dim)(x)
        x = LSTM(latent_dim, use_bias=use_bias, activation=act_r, return_sequences=True)(x)
        decoded = TimeDistributed(Dense(1, use_bias=use_bias, activation=act_r))(x)
        return decoded

    autoencoder = Model(input_frames, decoder_func(encoded))
    logging.debug(autoencoder.summary())

    decoder_inputs = Input(shape=K.int_shape(encoded)[1:], dtype=K.dtype(encoded))
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs))
    logging.debug(decoder.summary())

    n_conv = 2
    if batch_norm:
        n_conv += 1
    n_encoder_layer = 1 + (n_conv+1)*12 + n_conv + 1 + 1  # padding, conv+pool, conv, reshape, LSTM
    n_decoder_layer = 3  # repeat, LSTM, TimeDistribute
    return autoencoder, decoder, n_encoder_layer, n_decoder_layer


def build_model_5(input_dim, latent_dim, intermediate_dim, n_pad_input=13, n_filters=16, use_bias=True,
                  act_c='relu', dropout=None, batch_norm=True, up_sam_type='conv1dtranspose', epsilon_std=1.0):
    """1D Unet CNN without concat Variatrional Auto-Enconder.
        Auto-Encoding Variational Bayes https://arxiv.org/abs/1312.6114
    """

    input_frames = Input(shape=(input_dim, 1))
    logging.debug(input_frames)
    x = ZeroPadding1D((0, n_pad_input))(input_frames)
    logging.debug(x)

    # Encoder
    for nf_i in [1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 16, 16]:
        x = conv1d(x, n_filters*nf_i, 3, 1, padding='same', use_bias=use_bias,
                   kernel_initializer='he_normal', batch_norm=batch_norm, act_c=act_c)
        x = MaxPool1D(2, padding='same')(x)

    for nf_i in [16]:
        x = conv1d(x, n_filters*nf_i, 3, 1, padding='valid', use_bias=use_bias,
                   kernel_initializer='he_normal', batch_norm=batch_norm, act_c=act_c)
    x = Reshape((-1, ))(x)
    logging.debug(f'x {x}')

    def sampling(para):
        z_mean, z_log_var = para
        # logging.debug(f'z_mean {z_mean}')
        # logging.debug(f'z_log_var {z_log_var}')
        epsilon = K.random_normal(shape=(K.shape(z_mean)[1], latent_dim), mean=0., stddev=epsilon_std)
        # logging.debug(f'epsilon {epsilon}')  # warning z_mean[0] or z_mean[1] ?
        # logging.debug(f'z_return {z_mean + K.exp(z_log_var) * epsilon}')
        return z_mean + K.exp(z_log_var) * epsilon

    z_mean = Dense(latent_dim, use_bias=use_bias)(x)
    z_log_var = Dense(latent_dim, use_bias=use_bias)(x)
    logging.debug(f'z_mean {z_mean}')
    logging.debug(f'z_log_var {z_log_var}')

    # note that "output_shape" isn't necessary with the TensorFlow backend,
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoded = z
    logging.debug(f'encoded {encoded}')

    def decoder_func(inputs):
        x = inputs

        x = Dense(intermediate_dim, use_bias=use_bias, activation='relu')(x)
        logging.debug(f'x {x}')
        x = Lambda(lambda x: K.expand_dims(x, axis=-2))(x)
        logging.debug(f'x {x}')

        for nf_i in [16]:
            x = deconv1d(x, n_filters*nf_i, kernel_size=1, strides=3, use_bias=use_bias,
                         dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)  # 3

        for nf_i in [16, 16]:
            x = deconv1d(x, n_filters*nf_i, kernel_size=2, strides=2, use_bias=use_bias,
                         dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)

        for nf_i in [16, 16]:
            x = Conv1D(n_filters*nf_i, kernel_size=2, strides=1, use_bias=use_bias,
                       padding='valid', activation=act_c)(x)  # 12->11, 22->21
            x = deconv1d(x, n_filters*nf_i, kernel_size=2, strides=2, use_bias=use_bias,
                         dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)

        for nf_i in [8, 8]:
            x = deconv1d(x, n_filters*nf_i, kernel_size=2, strides=2, use_bias=use_bias,
                         dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)
            x = Conv1D(n_filters*nf_i, kernel_size=2, strides=1, use_bias=use_bias,
                       padding='valid', activation=act_c)(x)  # 84->83, 166->165

        for nf_i in [4, 4, 2, 2, 1, 1]:
            x = deconv1d(x, n_filters*nf_i, kernel_size=2, strides=2, use_bias=use_bias,
                         dropout_rate=dropout, batch_norm=batch_norm, up_sam_type=up_sam_type)

        x = Conv1D(1, 1, use_bias=use_bias, activation='tanh')(x)
        logging.debug(f'x {x}')
        decoded = Lambda(lambda x: x[:, 0:input_dim, :])(x)
        logging.debug(f'decoded {decoded}')
        return decoded

    autoencoder = Model(input_frames, decoder_func(encoded))
    logging.debug(autoencoder.summary())

    decoder_inputs = Input(shape=K.int_shape(encoded)[1:], dtype=K.dtype(encoded))
    logging.debug(f'decoder_inputs {decoder_inputs}')
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs))
    logging.debug(decoder.summary())

    n_conv = 2
    if batch_norm:
        n_conv += 1
    n_encoder_layer = 1 + (n_conv+1)*12 + n_conv + 1 + 3  # padding, conv+pool, conv, reshape, sampling

    if up_sam_type == 'conv1dtranspose':
        n_deconv = 3  # num of layers of deconv1d
    elif up_sam_type == 'upsampling':
        n_deconv = 2
    if dropout:
        n_deconv += 1
    if batch_norm:
        n_deconv += 1
    n_decoder_layer = 2 + n_deconv*3 + (1+n_deconv)*2 + (n_deconv+1)*2 + n_deconv*6 + 2
    # dense+lambda, deconv, conv+deconv, deconv+conv, deconv, conv+lambda
    return autoencoder, decoder, n_encoder_layer, n_decoder_layer


def build_model_6(input_dim, encoding_dim, latent_dim, intermediate_dim, output_dim,
                  use_bias=True, act_c='relu', epsilon_std=1.0, n_nodes=[0.5]):
    """MLP Variatrional Auto-Enconder.
        Auto-Encoding Variational Bayes https://arxiv.org/abs/1312.6114
    """

    input_frames = Input(shape=(input_dim, ))
    logging.debug(input_frames)

    x = input_frames
    for n_node in n_nodes:
        x = Dense(int(encoding_dim * n_node), use_bias=use_bias, activation=act_c)(x)
    x = Dense(encoding_dim, use_bias=use_bias, activation=act_c)(x)
    logging.debug(f'x {x}')

    def sampling(para):
        z_mean, z_log_var = para
        epsilon = K.random_normal(shape=(K.shape(z_mean)[1], latent_dim), mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon

    z_mean = Dense(latent_dim, use_bias=use_bias)(x)
    z_log_var = Dense(latent_dim, use_bias=use_bias)(x)
    logging.debug(f'z_mean {z_mean}')
    logging.debug(f'z_log_var {z_log_var}')

    # note that "output_shape" isn't necessary with the TensorFlow backend,
    # so you could write `Lambda(sampling)([z_mean, z_log_var])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
    encoded = z
    logging.debug(f'encoded {encoded}')

    def decoder_func(inputs):
        x = inputs
        x = Dense(intermediate_dim, use_bias=use_bias, activation='relu')(x)
        logging.debug(f'x {x}')
        x = Lambda(lambda x: K.expand_dims(x, axis=-2))(x)
        logging.debug(f'x {x}')

        for i in range(len(n_nodes)-1, -1, -1):
            x = Dense(int(encoding_dim * n_nodes[i]), use_bias=use_bias, activation=act_c)(x)
        x = Dense(output_dim, use_bias=use_bias, activation='tanh')(x)
        logging.debug(f'x {x}')
        decoded = Lambda(lambda x: x[:, 0:input_dim, :])(x)
        logging.debug(f'decoded {decoded}')
        return decoded

    autoencoder = Model(input_frames, decoder_func(encoded))
    logging.debug(autoencoder.summary())

    decoder_inputs = Input(shape=K.int_shape(encoded)[1:], dtype=K.dtype(encoded))
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs))
    logging.debug(decoder.summary())

    n_encoder_layer = len(n_nodes) + 1 + 3  # dense, dense, sampling
    n_decoder_layer = 2+len(n_nodes)+2  # dense+lambda, dense, dense+lambda
    return autoencoder, decoder, n_encoder_layer, n_decoder_layer


def build_model_7(input_dim, latent_dim, use_bias=True, act_r='tanh'):
    """1D LSTM autoenconder, with repeat vector."""

    input_frames = Input(shape=(input_dim, 1))
    encoded = LSTM(latent_dim, use_bias=use_bias, activation=act_r)(input_frames)

    def decoder_func(inputs):
        x = inputs
        x = RepeatVector(input_dim)(x)
        decoded = LSTM(1, use_bias=use_bias, activation=act_r, return_sequences=True)(x)

        return decoded

    autoencoder = Model(input_frames, decoder_func(encoded))
    logging.debug(autoencoder.summary())

    decoder_inputs = Input(shape=K.int_shape(encoded)[1:], dtype=K.dtype(encoded))
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs))
    logging.debug(decoder.summary())

    n_encoder_layer = 1
    n_decoder_layer = 2  # repeat+LSTM

    return autoencoder, decoder, n_encoder_layer, n_decoder_layer


def segment_encoded_signal(x, signal_length_samples, chunk_size, num_filters_in_encoder,
                           chunk_advance, num_full_chunks):
    # logging.debug(f'segmentation in {x}')
    x1 = Reshape((signal_length_samples//chunk_size, chunk_size, num_filters_in_encoder))(x)
    x2 = tf.roll(x, shift=-chunk_advance, axis=1)
    x2 = Reshape((signal_length_samples//chunk_size, chunk_size, num_filters_in_encoder))(x2)
    x2 = x2[:, :-1, :, :]  # Discard last segment with invalid data

    x_concat = tf.concat([x1, x2], 1)
    x = x_concat[:, ::num_full_chunks, :, :]
    for i in range(1, num_full_chunks):
        x = tf.concat([x, x_concat[:, i::num_full_chunks, :, :]], 1)
    return x


def overlap_and_add_mask_segments(x, chunk_advance):
    x = tf.transpose(x, [0, 3, 1, 2])
    if tf.__version__ >= '1.15':
        x = tf.signal.overlap_and_add(x, chunk_advance)
    else:
        x = overlap_and_add(x, chunk_advance)
    return tf.transpose(x, [0, 2, 1])


def overlap_and_add_in_decoder(x, encoder_hop_size):
    if tf.__version__ >= '1.15':
        return tf.signal.overlap_and_add(x, encoder_hop_size)
    return overlap_and_add(x, encoder_hop_size)


def build_model_8(input_dim, n_pad_input, chunk_size, chunk_advance,
                  n_conv_encoder, n_filters_conv, kernel_size, strides, batch_norm, act_c,
                  n_rnn_encoder, n_rnn_decoder, rnn_type, units_r, act_r, use_bias,
                  n_outputs, encoder_multiple_out, batch_size=None):
    """Conv 1D + 1D LSTM autoenconder, with frame segment, overlap, and add. Decoder n_outputs RNN.
    Args:
        input_dim (int): dim of the input vector.
        n_pad_input (int): network input = input_dim + n_pad_input.
        chunk_size (int): chunk size in the RNN layer.
        chunk_advance (int): strides size in the RNN layer.
        n_conv_encoder (int): number of the convolutional layers in encoder layers.
        n_filters_conv (int): number of the filters in encoder layer.
        kernel_size (int): kernel_size of the convolutional layers in encoder layers.
        strides (int): strides of the convolutional layers in encoder layers.
        batch_norm (bool): whether using batch normolization layers in encoder layers.
        act_c (str): activation function in the convolutional layers in encoder layersr.
        n_rnn_encoder (int): number of the RNN layers in encoder layers.
        n_rnn_decoder (int): number of the RNN layers in decoder layers.
        rnn_type (str): type of the RNN layer.
        units_r (int): number of the units in an RNN layer.
        act_r (str): activation function in RNN layer.
        use_bias (bool): whether use bias in all neurals.
        n_outputs (int): number of output srcs.
        use_mask (bool): whether using masks layers on encoder vector.
        encoder_multiple_out (bool): trick, for module train_separation_one_autoencoder_freeze_decoder.
        batch_size (int, optional): batch size of the samples during training. Default None.
    """
    frame_length = input_dim + n_pad_input
    n_full_chunks = frame_length // chunk_size
    n_overlapping_chunks = n_full_chunks*2-1

    input_frames = Input(shape=(input_dim, 1))
    logging.debug(f'input {input_frames}')
    x = ZeroPadding1D((0, n_pad_input))(input_frames)  # (?, 10560, 1)

    # Encoder
    for _ in range(n_conv_encoder):
        x = conv1d(x, n_filters=n_filters_conv, kernel_size=kernel_size, strides=strides, padding='same',
                   kernel_initializer='he_normal', batch_norm=batch_norm, use_bias=use_bias, act_c=act_c)
    logging.debug(f'conv out {x}')
    # Segmentation
    x = Lambda(lambda x: segment_encoded_signal(x, frame_length, chunk_size, n_filters_conv,
                                                chunk_advance, n_full_chunks))(x)
    logging.debug(f'segmentation out {x}')  # (bs=32, n_chunkds=329, chunk_siz=64, n_filters_conv=64)
    if rnn_type in {'LSTM', 'BLSTM'}:
        x = Reshape(tuple(K.int_shape(x)[1:-2])+(chunk_size*n_filters_conv,))(x)
        logging.debug(f'reshape {x}')  # (bs=32, n_chunkds=329, chunk_size*n_filters_conv=64*64)

    def rnn_layer(rnn_type, x):
        if rnn_type == 'LSTM':
            x = LSTM(units_r, activation=act_r, use_bias=use_bias, return_sequences=True)(x)
        if rnn_type == 'BLSTM':
            x = Bidirectional(LSTM(units_r//2, activation=act_r, use_bias=use_bias, return_sequences=True))(x)
        elif rnn_type == 'dprnn':
            x = dprnn_block(x, is_last_dprnn=False, num_overlapping_chunks=n_overlapping_chunks, chunk_size=chunk_size,
                            num_filters_in_encoder=n_filters_conv, units_per_lstm=units_r,
                            use_bias=use_bias, act_r=act_r, batch_size=batch_size)
        return x

    # RNN encoder
    for _ in range(n_rnn_encoder):
        x = rnn_layer(rnn_type, x)

    encoded = x
    # (bs, time_step=329, out_dim=units_r) or (bs, time_step=329, out_dim=units_r, n_outputs)
    logging.debug(f'encoded {encoded}')
    # (bs, time_step=329, out_dim=units_r) with LSTM, BLSTM
    # (bs, time_step=329, chunk_size=64, n_filters_conv=64) with dprnn

    def decoder_out(inputs, dim_fc, n_outputs, n_rnn_decoder):
        logging.debug(f'decoder_out inputs {inputs}')  # (bs, time_step=329, units_r, n_outputs)
        decoded_srcs = []
        for i in range(n_outputs):
            if K.ndim(inputs) == 3:  # (bs, time_step=329, out_dim=units_r)
                x = inputs
            elif K.ndim(inputs) == 4:
                if rnn_type in {'dprnn'}:
                    x = inputs
                    logging.debug(f'x {x}')  # (bs, time_step=329, chunk_size=64, n_filters_conv=64)
                else:
                    raise ValueError('Wrong encoder output dimision.')
            else:
                raise ValueError('Wrong encoder output dimision.')

            if encoder_multiple_out:  # Trick, only use in separator modules
                x = Dense(units=units_r, use_bias=use_bias)(x)
                logging.debug(f'first dense out {x}')  # (bs, time_step=329, units_r)

            for _ in range(n_rnn_decoder):
                x = rnn_layer(rnn_type, x)
            logging.debug(f'rnn out {x}')  # (bs, time_step=329, units_r)
            if rnn_type == 'dprnn':
                x = Reshape(tuple(K.int_shape(x)[1:-2]+(-1,)))(x)
                logging.debug(f'rnn out reshape {x}')  # (bs, time_step=329, chunk_size*n_filters_conv=64*64)

            x = Dense(units=dim_fc, use_bias=use_bias)(x)
            logging.debug(f'dense out {x}')  # (bs, time_step=329, dim_f=chunk_size)
            x = LayerNormalization(center=False, scale=False)(x)
            logging.debug(f'ln out {x}')  # (bs, time_step=329, dim_fc=chunk_size)
            x = Lambda(lambda x: overlap_and_add_in_decoder(x, chunk_advance))(x)
            logging.debug(f'decoded_src_i {x}')  # (bs=32, fl)
            decoded_src_i = Lambda(lambda x: K.expand_dims(x, axis=-1))(x)
            logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs=32, fl, 1)
            decoded_srcs.append(decoded_src_i)
        if n_outputs > 1:
            x = Concatenate(axis=-1)(decoded_srcs)
        else:
            x = decoded_srcs[0]
        logging.debug(f'decoded_srcs {x}')  # (bs=32, fl, n_outputs)
        return x

    def decoder_func(inputs, n_rnn_decoder, n_outputs):
        x = decoder_out(inputs, chunk_size, n_outputs, n_rnn_decoder)
        logging.debug(f'x {x}')  # (bs=32, fl, n_outputs)
        decoded = Lambda(lambda x: x[:, 0:input_dim, :])(x)
        logging.debug(f'decoded {decoded}')  # (bs=32, input_dim, n_outputs)
        return decoded

    autoencoder = Model(input_frames, decoder_func(encoded, n_rnn_decoder, n_outputs))
    logging.debug(autoencoder.summary())

    decoder_inputs = Input(shape=K.int_shape(encoded)[1:], dtype=K.dtype(encoded))
    # (bs, time_step=329, out_dim=units_r) or (bs, time_step=329, out_dim=units_r, n_outputs)
    logging.debug(f'decoder_inputs {decoder_inputs}')
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs, n_rnn_decoder, n_outputs))
    logging.debug(decoder.summary())

    n_conv_layer = 2
    if batch_norm:
        n_conv_layer += 1
    if rnn_type == 'LSTM':
        n_rnn_layer = 1
    elif rnn_type == 'BLSTM':
        n_rnn_layer = 1
    elif rnn_type == 'dprnn':
        n_rnn_layer = 18  # when using dprnn_block (function)
    # padding, conv, seg
    n_encoder_layer = 1 + n_conv_encoder*n_conv_layer + 1
    # reshape
    if rnn_type in {'LSTM', 'BLSTM'}:
        n_encoder_layer += 1
    # rnn
    n_encoder_layer += n_rnn_encoder*n_rnn_layer

    # encoder_multiple_out, Trick, only use in separator modules
    if encoder_multiple_out:
        n_encoder_layer += n_outputs

    # RNN + (dense+LN+overlapadd+expand_dims)
    n_decoder_out_layer = n_rnn_decoder*n_rnn_layer+4
    # reshape
    if rnn_type == 'dprnn':
        n_decoder_out_layer += 1

    n_decoder_layer = n_outputs*n_decoder_out_layer
    # concat
    if n_outputs > 1:
        n_decoder_layer += 1
    # clip
    n_decoder_layer += 1

    return autoencoder, decoder, n_encoder_layer, n_decoder_layer


def build_model_10(input_dim, n_pad_input, chunk_size, chunk_advance,
                   n_conv_encoder, n_filters_encoder, kernel_size, strides, batch_norm, act_c,
                   n_block_encoder, n_block_decoder, block_type, use_bias,
                   n_outputs, is_multiple_decoder, use_mask=True,
                   units_r=None, act_r=None, use_ln_decoder=False, model_type='separator', encoder_multiple_out=None,
                   batch_size=None,
                   ):
    """Tasnet or Encoder-Decoder Networks with Dprnn, BLSTM, LSTM.
        LUO Y, CHEN Z, YOSHIOKA T. Dual-Path RNN: Efficient Long Sequence Modeling for Time-Domain Single-Channel Speech Separation[C/OL]
        //ICASSP 2020 - 2020 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). Barcelona, Spain: IEEE, 2020: 46-50.
        https://ieeexplore.ieee.org/document/9054266/. DOI:10.1109/ICASSP40776.2020.9054266.
    Args:
        input_dim (int): dim of the input vector.
        n_pad_input (int): network input = input_dim + n_pad_input.
        chunk_size (int): chunk size in the RNN layer.
        chunk_advance (int): strides size in the RNN layer.
        n_conv_encoder (int): number of the convolutional layers in encoder layers.
        n_filters_encoder (int): number of the filters in encoder layer.
        kernel_size (int): kernel_size of the convolutional layers in encoder layers.
        strides (int): strides of the convolutional layers in encoder layers.
        batch_norm (bool): whether using batch normolization layers in encoder layers.
        act_c (str): activation function in the convolutional layers in encoder layersr.
        n_block_encoder (int): number of the RNN layers in encoder layers.
        n_block_decoder (int): number of the RNN layers in decoder layers.
        block_type (str): type of the RNN layer.
        use_bias (bool): whether use bias in all neurals.
        n_outputs (int): number of output srcs.
        is_multiple_decoder (bool): whether the network is encoder - multiple decoder structure.
        use_mask (bool): whether using masks layers on encoder vector.
        units_r (int): number of the units in an RNN layer.
        act_r (str): activation function in RNN layer.
        use_ln_decoder (bool): whether using layer normalization layers in decoder layers.
        model_type (str, optional): trick, the model used for separator or autoencoder. Defaults to 'separator'.
        encoder_multiple_out (bool, optional): trick, for module train_separation_one_autoencoder_freeze_decoder. Defaults to False.
        batch_size (int, optional): batch size of the samples during training. Defaults to None.
    """
    frame_length = input_dim + n_pad_input
    n_full_chunks = frame_length // chunk_size
    n_overlapping_chunks = n_full_chunks * 2 - 1

    input_frames = Input(shape=(input_dim, 1))
    logging.debug(f'input {input_frames}')
    x = ZeroPadding1D((0, n_pad_input))(input_frames)  # (bs=32, fl=10560, 1)

    # Encoder
    for _ in range(n_conv_encoder):
        x = conv1d(x, n_filters=n_filters_encoder, kernel_size=kernel_size, strides=strides, padding='same',
                   kernel_initializer='he_normal', batch_norm=batch_norm, use_bias=use_bias, act_c=act_c)
    encoded_conv = x
    logging.debug(f'conv out {encoded_conv}')  # (bs=32, fl=10560, n_filters_encoder=64)
    # Segmentation
    x = Lambda(lambda x: segment_encoded_signal(x, frame_length, chunk_size, n_filters_encoder,
                                                chunk_advance, n_full_chunks))(encoded_conv)
    logging.debug(f'segmentation out {x}')  # (bs=32, n_chunks=329, chunk_siz=64, n_filters_encoder=64)
    if block_type in {'LSTM', 'BLSTM'}:
        x = Reshape(tuple(K.int_shape(x)[1:-2]) + (chunk_size * n_filters_encoder,))(x)
        logging.debug(f'reshape {x}')  # (bs=32, n_chunks=329, chunk_size*n_filters_encoder=64*64)

    def layer_block(block_type, inputs, **kwargs):
        """RNN layer of the model.
        Args:
            block_type (str): type of the RNN layer.
            inputs (keras.tensor): keras tensor, input of the rnn layer.
        Returns:
            outputs (keras.tensor): keras tensor, output of the rnn layer.
        """
        if block_type == 'LSTM':
            outputs = LSTM(units_r, activation=act_r, use_bias=use_bias, return_sequences=True)(inputs)
        if block_type == 'BLSTM':
            outputs = Bidirectional(LSTM(units_r // 2, activation=act_r,
                                    use_bias=use_bias, return_sequences=True))(inputs)
        elif block_type == 'dprnn':
            is_last_block = kwargs['is_last_block'] if 'is_last_block' in kwargs.keys() else False
            n_speakers = kwargs['n_speakers'] if 'n_speakers' in kwargs.keys() else 1
            outputs = dprnn_block(inputs, is_last_dprnn=is_last_block,
                                  num_overlapping_chunks=n_overlapping_chunks, chunk_size=chunk_size,
                                  num_filters_in_encoder=n_filters_encoder, units_per_lstm=units_r,
                                  act_r=act_r, batch_size=batch_size, use_bias=use_bias, num_speakers=n_speakers)
        return outputs

    # Encoder
    for _ in range(n_block_encoder):
        x = layer_block(block_type, x)

    encoded = x
    logging.debug(f'encoded {encoded}')

    # (bs, time_step=329, out_dim=units_r) with LSTM, BLSTM
    # (bs, time_step=329, chunk_size=64, units_r) with dprnn

    def tasnet_decoder(inputs, dim_fc, n_outputs, n_block_decoder, block_type, is_multiple_decoder):
        def slice_tensor(tensor, index_channel):
            return tensor[:, :, :, index_channel]

        logging.debug(f'decoder_out inputs {inputs}')
        decoded_srcs = []
        n_branches = n_outputs if is_multiple_decoder else 1
        for i_branch in range(n_branches):
            if K.ndim(inputs) == 3:  # (bs, time_step=329, out_dim=units_r)
                x = inputs
            elif K.ndim(inputs) == 4:  # (bs, time_step=329, chunk_size=64, units_r)
                if block_type in {'dprnn'}:
                    x = inputs
                    logging.debug(f'x {x}')  # (bs, time_step=329, chunk_size=64, units_r)
                else:
                    raise ValueError('Wrong encoder output dimision.')
            else:
                raise ValueError('Wrong encoder output dimision.')

            if encoder_multiple_out:  # Trick, only use in separator modules
                if block_type in {'BLSTM', 'LSTM'}:
                    x = Dense(units=units_r, use_bias=use_bias,
                              name=f'Dense_me_{i_branch}')(x)
                elif block_type in {'dprnn'}:
                    x = Dense(units=n_filters_encoder, use_bias=use_bias,
                              name=f'Dense_me_{i_branch}')(x)

                logging.debug(f'first dense out {x}')  # (bs, time_step=329, units_r)

            for i_block in range(1, n_block_decoder + 1):
                if block_type in {'BLSTM', 'LSTM'}:
                    x = layer_block(block_type, x)
                elif block_type in {'dprnn'}:
                    n_speakers = 1 if is_multiple_decoder else n_outputs
                    is_last_block = True if not is_multiple_decoder and i_block == n_block_decoder else False
                    x = layer_block(block_type, x, **{'is_last_block': is_last_block,
                                                      'n_speakers': n_speakers})
            logging.debug(f'decoder block out {x}')
            # (bs, time_step=329, units_r) with LSTM, BLSTM
            # (bs, time_step=329, chunk_size=64, n_filters_encoder*n_speakers) with dprnn when is_multiple_decoder == False
            # (bs, time_step=329, chunk_size=64, n_filters_encoder) with dprnn when is_multiple_decoder == True

            if block_type in {'LSTM', 'BLSTM', 'dprnn'} and not is_multiple_decoder and n_branches == 1:  # One Encoder - One Decoder
                if block_type in {'LSTM', 'BLSTM'}:
                    x = Dense(units=chunk_size * n_filters_encoder * n_outputs, use_bias=use_bias)(x)
                    logging.debug(f'dense for mask {x}')
                    # (bs=32, time_step=329, chunk_size*n_filters_encoder*n_outputs=64*64*4)
                    x = Reshape(tuple(K.int_shape(x)[1:-1]) + (chunk_size, n_filters_encoder * n_outputs))(x)
                    logging.debug(f'reshape {x}')  # (bs=32, time_step=329, chunk_size, n_filters_encoder*n_speakers)
                x = Lambda(lambda x: overlap_and_add_mask_segments(x, chunk_advance))(x)
                logging.debug(f'x {x}')  # (bs, fl=10560, n_filters_encoder * n_speakers)
                x = Reshape((frame_length, n_filters_encoder, n_outputs))(x)
                logging.debug(f'x reshape {x}')  # (bs, fl=10560, n_filters_encoder=64, n_speakers=4)
                # Apply speaker masks to encoded mixture signal
                for i_src in range(n_outputs):
                    # decoded_src_i = Lambda(lambda x: x[:, :, :, i_src])(x)
                    decoded_src_i = Lambda(slice_tensor, arguments={'index_channel': i_src})(x)
                    if use_mask:  # original TasNet
                        logging.debug(f'mask_src_i {decoded_src_i}')  # (bs, fl=10560, n_filters_encoder)
                        logging.debug(f'encoded_conv {encoded_conv}')  # (bs, fl=10560, n_filters_encoder)
                        decoded_src_i = Multiply()([encoded_conv, decoded_src_i])
                    logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs, fl=10560, n_filters_encoder)
                    decoded_src_i = Dense(units=dim_fc, use_bias=use_bias)(decoded_src_i)
                    logging.debug(f'dense out {decoded_src_i}')  # (bs, fl=10560, dim_fc)
                    if use_ln_decoder:
                        decoded_src_i = LayerNormalization(center=False, scale=False)(decoded_src_i)
                        logging.debug(f'ln out {x}')  # (bs, fl=10560, dim_fc)
                    decoded_src_i = Lambda(lambda x: overlap_and_add_in_decoder(x, strides))(decoded_src_i)
                    logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs=32, fl=10560,)
                    decoded_src_i = Lambda(lambda x: K.expand_dims(x, axis=-1))(decoded_src_i)
                    logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs=32, fl=10560, 1)
                    decoded_srcs.append(decoded_src_i)
            elif block_type in {'LSTM', 'BLSTM', 'dprnn'} and n_branches == n_outputs:  # One Encoder - Multiple Decoder
                decoded_src_i = x

                if block_type in {'LSTM', 'BLSTM'}:
                    decoded_src_i = Dense(units=chunk_size * n_filters_encoder, use_bias=use_bias)(decoded_src_i)
                    # (bs=32, time_step=329, chunk_size*n_filters_encoder=64*64)
                    logging.debug(f'reshape {decoded_src_i}')
                    decoded_src_i = Reshape(tuple(K.int_shape(decoded_src_i)[1:-1]) + (chunk_size, n_filters_encoder)
                                            )(decoded_src_i)
                    # (bs=32, time_step=329, chunk_size=64, n_filters_encoder=64)
                    logging.debug(f'reshape {decoded_src_i}')
                decoded_src_i = Lambda(lambda x: overlap_and_add_mask_segments(x, chunk_advance))(decoded_src_i)
                if use_mask:  # TasNet with Multiple Decoder
                    logging.debug(f'mask_src_i {decoded_src_i}')  # (bs, fl=10560, n_filters_encoder)
                    decoded_src_i = Multiply()([encoded_conv, decoded_src_i])
                logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs, fl=10560, n_filters_encoder)
                decoded_src_i = Dense(units=dim_fc, use_bias=use_bias)(decoded_src_i)
                logging.debug(f'dense out {decoded_src_i}')  # (bs, fl=10560, dim_fc)
                if use_ln_decoder:
                    decoded_src_i = LayerNormalization(center=False, scale=False)(decoded_src_i)
                    logging.debug(f'ln out {decoded_src_i}')  # (bs, fl=10560, dim_fc)
                decoded_src_i = Lambda(lambda x: overlap_and_add_in_decoder(x, strides))(decoded_src_i)
                logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs=32, fl=10560,)
                decoded_src_i = Lambda(lambda x: K.expand_dims(x, axis=-1))(decoded_src_i)
                logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs=32, fl=10560, 1)
                decoded_srcs.append(decoded_src_i)
        if n_outputs > 1:
            outputs = Concatenate(axis=-1)(decoded_srcs)
        else:
            outputs = decoded_srcs[0]
        logging.debug(f'decoded_srcs {outputs}')  # (bs=32, fl, n_outputs)
        return outputs

    def decoder_func(inputs, n_block_decoder, n_outputs):
        x = tasnet_decoder(inputs, kernel_size, n_outputs, n_block_decoder,
                           block_type, is_multiple_decoder)
        logging.debug(f'x {x}')  # (bs=32, fl, n_outputs)
        decoded = Lambda(lambda x: x[:, 0:input_dim, :])(x)
        logging.debug(f'decoded {decoded}')  # (bs=32, input_dim, n_outputs)
        return decoded

    autoencoder = Model(input_frames, decoder_func(encoded, n_block_decoder, n_outputs))
    logging.debug(autoencoder.summary())
    if model_type == 'separator':
        return autoencoder, None, None, None

    decoder_inputs = Input(shape=K.int_shape(encoded)[1:], dtype=K.dtype(encoded))
    # (bs, time_step=329, out_dim=units_r) with LSTM, BLSTM
    # (bs, time_step=329, chunk_size=64, units_r) with dprnn
    logging.debug(f'decoder_inputs {decoder_inputs}')
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs, n_block_decoder, n_outputs))
    logging.debug(decoder.summary())

    n_conv_layer = 2
    if batch_norm:
        n_conv_layer += 1
    if block_type in {'LSTM', 'BLSTM'}:
        n_rnn_layer = 1
    elif block_type in {'dprnn'}:
        n_rnn_layer = 18

    # padding, conv, seg
    n_encoder_layer = 1 + n_conv_encoder*n_conv_layer + 1
    # reshape
    if block_type in {'LSTM', 'BLSTM'}:
        n_encoder_layer += 1
    # rnn
    n_encoder_layer += n_block_encoder*n_rnn_layer

    # Trick, only use when model_type == 'ae_sep'
    if model_type == 'ae_sep' and is_multiple_decoder:
        # lambda layer x[:,:,:,i_src]
        n_encoder_layer += n_outputs

    # encoder_multiple_out, Trick, only use in separator modules
    if encoder_multiple_out:
        n_encoder_layer += n_outputs

    # RNN
    n_decoder_rnn_layer = n_block_decoder*n_rnn_layer

    # {dense+reshape} + overlap_add_mask + {reshape}
    n_decoder_mask_layer = 1
    if block_type in {'LSTM', 'BLSTM'}:
        n_decoder_mask_layer += 2
    if not is_multiple_decoder:
        n_decoder_mask_layer += 1

    # {mask} + dense + {LN} + overlap_add_decoder + expand_dims
    n_decoder_out_layer = 3
    if use_mask:
        n_decoder_out_layer += 1
    if use_ln_decoder:
        n_decoder_out_layer += 1
    if not is_multiple_decoder:
        n_decoder_out_layer += 1

    if is_multiple_decoder:
        n_decoder_layer = n_outputs*(n_decoder_rnn_layer+n_decoder_mask_layer+n_decoder_out_layer)
    else:
        n_decoder_layer = n_decoder_rnn_layer+n_decoder_mask_layer+n_outputs*n_decoder_out_layer

    # concat
    if n_outputs > 1:
        n_decoder_layer += 1
    # clip
    n_decoder_layer += 1

    return autoencoder, decoder, n_encoder_layer, n_decoder_layer


def build_model_14(input_dim, n_pad_input,
                   n_conv_encoder, n_filters_encoder, kernel_size_encoder, strides_encoder, act_c, use_bias,
                   n_channels_conv, n_channels_bottleneck, n_channels_skip,
                   n_block_encoder, n_block_decoder, n_layer_each_block, kernel_size, causal, norm_type,
                   n_outputs, is_multiple_decoder, use_residual=True, use_skip=True, use_sum_repeats=False,
                   use_mask=True, act_mask='sigmoid', output_activation=None, model_type='separator', encoder_multiple_out=False,
                   ):
    # n_filters_encoder     N  = 512    Number of filters in autoencoder
    # kernel_size_encoder   L  =  16    Length of the filters (in samples)
    # n_channels_bottleneck B  = 128    Number of channels in bottleneck and the residual path's 1x1-conv block
    # n_channels_skip       Sc = 128    Number of channels in skip-connection paths' 1*1-conv blocks
    # n_channels_conv       H  = 512    Number of channels in convolutional blocks
    # kernel_size           P  =   3    Kernel size in convolutional blocks
    # n_layer_each_block    X  =   8    Number of convolutional blocks in each repeat
    # n_repeats             R  =   3    Numner of repeats, n_repeats = n_block_encoder + n_block_decoder
    # n_outputs             C  =   1    Numebr of speakers
    # n_channels_bottleneck should be equal to n_channels_skip

    # tcn_branch
    # use_mask (bool): whether multiple mask with encoded vector.
    # is_multiple_decoder (bool): whether use multiple decoder in networks.
    # use_residual (bool): whether use a 1x1-conv of residual path in TCN block.
    # use_skip (bool): whether use a 1x1-conv of skip-connection path in TCN block.
    # use_sum_repeats (bool): whether use sums of skip-connection paths and residual path in each repeat
    #                           as next repeats' input
    #   use_mask, use_residual, use_skip, use_sum_repeats, is_multiple_decoder, -> works   network structure
    # True/False,         True,     True,      True/False,                True, ->  True   Conv-TasNet or Encoder-Decoder with Multiple Decoder
    # True/False,         True,     True,      True/False,               False, ->  True   Conv-TasNet or Encoder-Decoder
    # True/False,         True,    False,      True/False,                True, ->  True   Normal ResNet Multiple Decoder
    # True/False,         True,    False,      True/False,               False, ->  True   Normal ResNet
    # True/False,        False,     True,      True/False,                True, -> False   Parallel CNN with Multiple Decoder
    # True/False,        False,     True,      True/False,               False, ->  True   Parallel CNN
    # True/False,        False,    False,      True/False,                True, -> False   no network (add inputs)
    # True/False,        False,    False,      True/False,               False, -> False   no network (add inputs)
    """Tasnet or Encoder-Decoder Networks with TCN (conv-TasNet).
        LUO Y, MESGARANI N. Conv-tasnet: Surpassing ideal time-frequency magnitude masking for speech separation[J].
        IEEE/ACM transactions on audio, speech, and language processing, 2019, 27(8): 1256-1266.
    Args:
        input_dim (int): dim of the input vector.
        n_pad_input (int): network input = input_dim + n_pad_input.
        n_conv_encoder (int): number of the convolutional layers in encoder layers.
        n_filters_encoder (int): N.
        kernel_size_encoder (int): L.
        strides_encoder (int): strides of the convolutional layers in encoder layers.
        act_c (str): activation function in the convolutional layers in encoder layersr.
        use_bias (bool): whether use bias in all neurals.
        n_channels_conv (int): H.
        n_channels_bottleneck (int): B.
        n_channels_skip (int): Sc.
        n_block_encoder (int): number of the TCN blocks in encoder layers.
        n_block_decoder (int): number of the TCN blocks in decoder layers.
        n_layer_each_block (int): X.
        kernel_size (int): P.
        causal (bool): whether support causal.
        norm_type (str): type of normolization layers.
        n_outputs (int): number of output srcs.
        is_multiple_decoder (bool): whether the network is encoder - multiple decoder structure.
        use_residual (bool, optional): whether use a 1x1-conv of residual path in TCN block. Defaults to True.
        use_skip (bool, optional): whether use a 1x1-conv of skip-connection path in TCN block. Defaults to True.
        use_sum_repeats (bool, optional): whether use sums of skip-connection paths and residual path in each repeat. Defaults to False.
        use_mask (bool, optional): whether multiple mask with encoded vector. Defaults to True.
        act_mask (str, optional): type of activation of mask layer. Defaults to 'sigmoid'.
        output_activation (str, optional): activation function of output layers. Defaults to None which equal to "linear".
        model_type (str, optional): trick, the model used for separator or autoencoder. Defaults to 'separator'.
        encoder_multiple_out (bool): trick, for module train_separation_one_autoencoder_freeze_decoder. Defaults to False.
    """
    frame_length = input_dim

    input_frames = Input(shape=(input_dim, 1))
    logging.debug(f'input {input_frames}')
    x = input_frames

    # Encoder
    for _ in range(n_conv_encoder):
        x = Conv1D(n_filters_encoder, kernel_size=kernel_size_encoder, strides=strides_encoder,
                   padding='same', kernel_initializer='he_normal', use_bias=use_bias, activation=act_c)(x)
    encoded_conv = x
    logging.debug(f'conv out {encoded_conv}')  # (bs=32, fl=10547, n_filters_encoder=128)

    x = normal_layer(x, 'cln')
    x = Conv1D(n_channels_bottleneck, 1, name="bottleneck_layer")(x)

    x = tcn_branch(x, n_block_encoder, n_layer_each_block, n_channels_conv,
                   n_channels_bottleneck, n_channels_skip, kernel_size, causal, norm_type,
                   use_residual=use_residual, use_skip=use_skip, use_sum_repeats=use_sum_repeats)
    encoded = x
    logging.debug(f'encoded {encoded}')

    def layer_activate_mask(x, act_mask):
        if act_mask == 'sigmoid':
            x = Activation('sigmoid')(x)
        elif act_mask == 'relu':
            x = ReLU()(x)
        else:
            raise ParameterError('Unsupport act_mask type {act_mask}')
        return x

    def tasnet_decoder(inputs, kernel_size, n_outputs, n_block_decoder, is_multiple_decoder):
        def slice_tensor(tensor, index_channel):
            return tensor[:, :, :, index_channel]

        x = inputs
        logging.debug(f'decoder_out inputs {x}')
        decoded_srcs = []
        n_branches = n_outputs if is_multiple_decoder else 1
        for i_branch in range(n_branches):
            if not is_multiple_decoder and n_branches == 1:  # One Encoder - One Decoder
                if n_block_decoder > 0:
                    x = tcn_branch(x, n_block_decoder, n_layer_each_block, n_channels_conv,
                                   n_channels_bottleneck, n_channels_skip, kernel_size, causal, norm_type,
                                   use_residual=use_residual, use_skip=use_skip, use_sum_repeats=use_sum_repeats)
                logging.debug(f'block out {x}')  # (bs, fl, n_channels_conv)
                x = PReLU(shared_axes=[1])(x)
                # (bs, fl, n_filters_encoder*n_outputs)
                x = Conv1D(filters=n_filters_encoder*n_outputs, kernel_size=1)(x)
                x = Reshape((frame_length, n_outputs, n_filters_encoder))(x)  # (bs, fl, n_outputs, n_filters_encoder)
                if norm_type:
                    x = normal_layer(x, norm_type)
                logging.debug(f'after norm {x}')
                x = Permute((1, 3, 2))(x)  # (bs, fl, n_filters_encoder, n_outputs)
                logging.debug(f'x Permute {x}')  # (bs, fl=10560, n_filters_encoder=64, n_outputs=4)

                # Apply speaker masks to encoded mixture signal
                for i_src in range(n_outputs):
                    decoded_src_i = Lambda(slice_tensor, arguments={'index_channel': i_src})(x)

                    if use_mask:  # original conv-TasNet
                        decoded_src_i = layer_activate_mask(decoded_src_i, act_mask)
                        logging.debug(f'mask_src_i {decoded_src_i}')  # (bs, fl=10560, n_filters_encoder)
                        logging.debug(f'encoded_conv {encoded_conv}')  # (bs, fl=10560, n_filters_encoder)
                        decoded_src_i = Multiply()([encoded_conv, decoded_src_i])
                    logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs, fl=10560, n_filters_encoder)
                    decoded_src_i = conv1d_transpose(decoded_src_i, 1, kernel_size_encoder,
                                                     strides=strides_encoder, activation=output_activation, padding='same')
                    logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs=32, fl=10560, 1)
                    decoded_srcs.append(decoded_src_i)
            elif n_branches == n_outputs:  # One Encoder - Multiple Decoder
                decoded_src_i = x

                if encoder_multiple_out:  # Trick, only use in separator modules
                    decoded_src_i = Dense(units=n_channels_bottleneck, use_bias=use_bias,
                                          name=f'Dense_me_{i_branch}')(decoded_src_i)
                    logging.debug(f'first dense out {decoded_src_i}')  # (bs, fl=10547, n_channels_conv=64)

                if n_block_decoder > 0:
                    decoded_src_i = tcn_branch(decoded_src_i, n_block_decoder, n_layer_each_block, n_channels_conv,
                                               n_channels_bottleneck, n_channels_skip, kernel_size, causal, norm_type,
                                               use_residual=use_residual, use_skip=use_skip,
                                               use_sum_repeats=use_sum_repeats)
                decoded_src_i = PReLU(shared_axes=[1])(decoded_src_i)
                decoded_src_i = Conv1D(filters=n_filters_encoder, kernel_size=1)(decoded_src_i)
                decoded_src_i = Reshape((frame_length, 1, n_filters_encoder))(decoded_src_i)
                if norm_type:
                    decoded_src_i = normal_layer(decoded_src_i, norm_type)
                decoded_src_i = Reshape((frame_length, n_filters_encoder))(decoded_src_i)
                # (bs, fl=10547, n_filters_encoder=64)
                logging.debug(f'decoded_src_i reshape {decoded_src_i}')
                if use_mask:
                    decoded_src_i = layer_activate_mask(decoded_src_i, act_mask)
                    logging.debug(f'mask_src_i {decoded_src_i}')  # (bs, fl=10547, n_filters_encoder)
                    logging.debug(f'encoded_conv {encoded_conv}')  # (bs, fl=10547, n_filters_encoder)
                    decoded_src_i = Multiply()([encoded_conv, decoded_src_i])
                    logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs, fl=10547, n_filters_encoder)
                decoded_src_i = conv1d_transpose(decoded_src_i, 1, kernel_size_encoder,
                                                 strides=strides_encoder, activation=output_activation, padding='same')
                logging.debug(f'decoded_src_i {decoded_src_i}')  # (bs=32, fl=10547, 1)
                decoded_srcs.append(decoded_src_i)

        if n_outputs > 1:
            outputs = Concatenate(axis=-1)(decoded_srcs)
        else:
            outputs = decoded_srcs[0]
        logging.debug(f'decoded_srcs {outputs}')  # (bs=32, fl, n_outputs)
        return outputs

    def decoder_func(inputs, n_block_decoder, n_outputs):
        decoded = tasnet_decoder(inputs, kernel_size, n_outputs, n_block_decoder, is_multiple_decoder)
        logging.debug(f'decoded {decoded}')  # (bs=32, fl, n_outputs)
        return decoded

    autoencoder = Model(input_frames, decoder_func(encoded, n_block_decoder, n_outputs))
    logging.debug(autoencoder.summary())
    if model_type == 'separator':
        return autoencoder, None, None, None

    decoder_inputs = Input(shape=K.int_shape(encoded)[1:], dtype=K.dtype(encoded))
    logging.debug(f'decoder_inputs {decoder_inputs}')
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs, n_block_decoder, n_outputs))
    logging.debug(decoder.summary())

    num_conv_encoder_layer = 1
    # # padding, Conv(encoder), LN, Conv(bottleneck_layer)
    # num_encoder_layer = 1 + n_conv_encoder*num_conv_encoder_layer + 1 + 1
    # Conv(encoder), LN, Conv(bottleneck_layer)
    num_encoder_layer = n_conv_encoder*num_conv_encoder_layer + 1 + 1

    # conv, PReLU, GLN, ZeroPadding, DepthwiseConv, PReLU, GLN
    num_block_conv_layer = 7
    if use_skip:
        num_block_conv_layer += 1  # Conv
    if use_residual:
        num_block_conv_layer += 2  # Conv, Add

    num_encoder_layer += num_block_conv_layer * n_layer_each_block * n_block_encoder

    if not use_sum_repeats:
        if use_residual:
            num_encoder_layer -= 2  # last block residual channel: Conv, Add
        num_encoder_layer += n_block_encoder  # Add per block
        if n_block_encoder > 1:
            num_encoder_layer += 1  # Add all blocks
    else:
        num_encoder_layer += (n_block_encoder - 1)  # Add per block

    if encoder_multiple_out:
        num_encoder_layer += n_outputs

    num_decoder_block_layer = num_block_conv_layer * n_layer_each_block * n_block_decoder
    if not use_sum_repeats:
        if use_residual:
            num_decoder_block_layer -= 2  # last block residual channel: Conv, Add
        num_decoder_block_layer += n_block_decoder  # Add per block
        num_decoder_block_layer += 1  # Add all blocks
    else:
        num_decoder_block_layer += (n_block_decoder - 1)  # Add per block

    # {PReLU, Conv, Reshape, GLN, Permute/Reshape}
    num_decoder_sep_layer = 5

    # Lambda
    num_decoder_mask_layer = 0 if is_multiple_decoder else 1  # slice_tensor
    if use_mask:
        num_decoder_mask_layer += 2  # {Activation, Multiply}

    # conv1d_transpose={Lambda, Conv2DTranspose, Lambda}
    num_decoder_decoder_layer = 3

    if is_multiple_decoder:
        # Block + Sep + Mask+ Decoder + (Lambda Clip)
        num_decoder_layer = n_outputs * (num_decoder_block_layer
                                         + num_decoder_sep_layer
                                         + num_decoder_mask_layer
                                         + num_decoder_decoder_layer
                                         # + 1
                                         )
        if n_outputs > 1:
            num_decoder_layer += 1  # Concatenate
    else:
        # Block + Sep + {Mask+Decoder} + (Lambda Clip)
        num_decoder_layer = num_decoder_block_layer\
            + num_decoder_sep_layer\
            + n_outputs * (num_decoder_mask_layer + num_decoder_decoder_layer)\
            # + 1
        if n_outputs > 1:
            num_decoder_layer += 1  # Concatenate

    if encoder_multiple_out:
        num_decoder_layer += n_outputs

    return autoencoder, decoder, num_encoder_layer, num_decoder_layer


def build_model_20(input_dim, n_pad_input, n_outputs, is_multiple_decoder, num_channels=1,
                   num_layers=12, num_initial_filters=24, kernel_size=15, use_bias=True, merge_filter_size=5,
                   output_filter_size=1, padding="same", context=False, upsampling_type="learned", batch_norm=False,
                   output_activation="linear", output_type="difference", use_skip=True, model_type='separator',
                   encoder_multiple_out=False):
    """Autoencoder or Encoder - Multiple Decoder networks with Wave-U-Net structure.
    STOLLER D, EWERT S, DIXON S. Wave-U-Net: A multi-scale neural network for end-to-end audio source separation[C]
    //19th International Society for Music Information Retrieval Conference, ISMIR 2018, September 23, 2018 - September 27, 2018.
    Paris, France: International Society for Music Information Retrieval, 2018: 334-340.
    Args:
        input_dim (int): dim of the input vector.
        n_pad_input (int): network input = input_dim + n_pad_input.
        n_outputs (int): number of output srcs.
        is_multiple_decoder (bool): whether the network is encoder - multiple decoder structure.
        num_channels (int, optional): number of channels of audio. Defaults to 1.
        num_layers (int, optional): number of layers of U-Net encoder or decoder. Defaults to 12.
        num_initial_filters (int, optional): initial number of filters of convolutional layers. Defaults to 24.
        kernel_size (int, optional): Kernel size in convolutional layers. Defaults to 15.
        use_bias (bool): whether use bias in all neurals.
        merge_filter_size (int, optional): Kernel size in convolutional layers after concat layers. Defaults to 5.
        output_filter_size (int, optional): number of filters of convolutional layers before output layers. Defaults to 1.
        padding (str, optional): padding type of convolutional layers. Defaults to "same".
        context (bool, optional): shape in bilinear interpolation resize layers. Defaults to False.
        upsampling_type (str, optional): type of upsampling layers. Defaults to "learned".
        batch_norm (bool, optional): whether using batch normolization layers. Defaults to False.
        output_activation (str, optional): activation function of output layers. Defaults to "linear".
        output_type (str, optional): type of output layer. Defaults to "difference".
        use_skip (bool, optional): wether using skip connection layers. Defaults to True.
        model_type (str, optional): trick, the model used for separator or autoencoder. Defaults to 'separator'.
        encoder_multiple_out (bool, optional): trick, for module train_separation_one_autoencoder_freeze_decoder. Defaults to False.
    """

    # `enc_outputs` stores the downsampled outputs to re-use during upsampling.
    enc_outputs = []

    # `raw_input` is the input to the network
    raw_input = Input(shape=(input_dim, num_channels), name="raw_input")
    logging.debug(f'input {raw_input}')

    X = ZeroPadding1D((0, n_pad_input))(raw_input)  # (bs=32, fl=12288, 1)
    inp = X

    # Down sampling
    for i_layer in range(num_layers):
        X = Conv1D(filters=num_initial_filters + (num_initial_filters * i_layer),
                   kernel_size=kernel_size, strides=1, use_bias=use_bias,
                   padding=padding, name="Down_Conv_"+str(i_layer))(X)
        X = LeakyReLU(name="Down_Conv_Activ_"+str(i_layer))(X)
        if batch_norm:
            X = BatchNormalization(name="Down_BN_"+str(i_layer))(X)
        enc_outputs.append(X)

        X = Lambda(lambda x: x[:, ::2, :], name="Decimate_"+str(i_layer))(X)

    X = Conv1D(filters=num_initial_filters + (num_initial_filters * num_layers),
               kernel_size=kernel_size, strides=1, use_bias=use_bias,
               padding=padding, name="Down_Conv_"+str(num_layers))(X)
    X = LeakyReLU(name="Down_Conv_Activ_"+str(num_layers))(X)
    if batch_norm:
        X = BatchNormalization(name="Down_BN_"+str(num_layers))(X)
    encoded = X
    logging.debug(f'encoded {encoded}')

    def img_resize(x, shape, align_corners=False):
        return tf.image.resize_bilinear(x, [1, shape], align_corners=align_corners)

    def decoder_block(X, num_layers, use_skip, i_branch=None):
        # Up sampling
        for i_layer in range(num_layers):
            X = Lambda(lambda x: K.expand_dims(x, axis=1),
                       name=f'exp_dims_{i_layer}' if i_branch is None else f'exp_dims_{i_branch}_{i_layer}'
                       )(X)
            if upsampling_type == "learned":
                X = InterpolationLayer(padding=padding,
                                       name=f'IntPol_{i_layer}' if i_branch is None else f'IntPol_{i_branch}_{i_layer}'
                                       )(X)
            else:
                if i_branch is None:
                    name_bilinear = f'bilinear_interpol_{i_layer}'
                else:
                    name_bilinear = f'bilinear_interpol_{i_branch}_{i_layer}'
                if context:
                    shape_up = X.shape.as_list()[2] * 2 - 1

                    X = Lambda(img_resize, arguments={'shape': shape_up, 'align_corners': True},
                               name=name_bilinear)(X)
                else:
                    shape_up = X.shape.as_list()[2] * 2
                    X = Lambda(img_resize, arguments={'shape': shape_up},
                               name=name_bilinear)(X)

            X = Lambda(lambda x: K.squeeze(x, axis=1),
                       name=f'sq_dims_{i_layer}' if i_branch is None else f'sq_dims_{i_branch}_{i_layer}'
                       )(X)

            if use_skip:
                name_crop_concat = f'CropConcat_{i_layer}' if i_branch is None else f'CropConcat_{i_branch}_{i_layer}'
                X = CropConcatLayer(match_feature_dim=False,
                                    name=name_crop_concat,
                                    )([enc_outputs[-i_layer-1], X])

            X = Conv1D(filters=num_initial_filters + (num_initial_filters * (num_layers - i_layer - 1)),
                       kernel_size=merge_filter_size, strides=1, use_bias=use_bias,
                       padding=padding,
                       name=f'Up_Conv_{i_layer}' if i_branch is None else f'Up_Conv_{i_branch}_{i_layer}'
                       )(X)
            X = LeakyReLU(name=f'Up_Conv_Activ_{i_layer}' if i_branch is None else f'Up_Conv_Activ_{i_branch}_{i_layer}'
                          )(X)
            if batch_norm:
                X = BatchNormalization(
                    name=f'Up_BN_{i_layer}' if i_branch is None else f'Up_BN_{i_branch}_{i_layer}')(X)

        if use_skip:
            name_crop_concat_last = f'CropConcat_{num_layers}' if i_branch is None else f'CropConcat_{i_branch}_{num_layers}'
            X = CropConcatLayer(match_feature_dim=False, name=name_crop_concat_last)([inp, X])
        return X

    def decoder_out(inputs, n_outputs, is_multiple_decoder):
        X = inputs
        logging.debug(f'decoder_out inputs {X}')  # (bs, time_step=329, units_r, n_outputs)
        decoded_srcs = []
        n_branches = n_outputs if is_multiple_decoder else 1
        for i_branch in range(n_branches):
            if not is_multiple_decoder and n_branches == 1:  # One Encoder - One Decoder
                X = decoder_block(X, num_layers, use_skip)
                if output_type == "direct":
                    for i_outputs in range(n_outputs):
                        decoded_src_i = Conv1D(num_channels, output_filter_size, use_bias=use_bias,
                                               padding=padding, activation=output_activation,
                                               name="independent_out_"+str(i_outputs))(X)
                        if output_activation not in {'tanh'}:
                            decoded_src_i = AudioClipLayer()(decoded_src_i)
                        decoded_srcs.append(decoded_src_i)
                else:  # Difference Output
                    cropped_input = CropLayer(match_feature_dim=False, name="Crop_layer_"+str(num_layers+1))([inp, X])
                    sum_source = []
                    for i_outputs in range(n_outputs-1):
                        decoded_src_i = Conv1D(num_channels, output_filter_size, padding=padding, use_bias=use_bias,
                                               activation=output_activation)(X)
                        if output_activation not in {'tanh'}:
                            decoded_src_i = AudioClipLayer()(decoded_src_i)
                        decoded_srcs.append(decoded_src_i)
                        sum_source.append(decoded_src_i)
                    if len(sum_source) > 1:
                        sum_source = keras.layers.Add()(sum_source)
                    else:
                        sum_source = sum_source[0]
                    last_source = CropLayer(name="Crop_layer_out")([cropped_input, sum_source])

                    last_source = keras.layers.Subtract()([last_source, sum_source])
                    decoded_srcs.append(last_source)
            elif n_branches == n_outputs:  # One Encoder - Multiple Decoder
                assert output_type == "direct"
                decoded_src_i = X

                if encoder_multiple_out:  # Trick, only use in separator modules
                    decoded_src_i = Dense(units=num_initial_filters + (num_initial_filters * num_layers),
                                          use_bias=use_bias,
                                          name=f'Dense_me_{i_branch}')(decoded_src_i)
                    logging.debug(f'first dense out {decoded_src_i}')  # (bs, fl=10547, n_channels_conv=64)

                decoded_src_i = decoder_block(decoded_src_i, num_layers, use_skip, i_branch=i_branch)
                decoded_src_i = Conv1D(num_channels, output_filter_size, use_bias=use_bias,
                                       padding=padding, activation=output_activation,
                                       name="independent_out_"+str(i_branch))(decoded_src_i)
                if output_activation not in {'tanh'}:
                    decoded_src_i = AudioClipLayer()(decoded_src_i)
                decoded_srcs.append(decoded_src_i)

        if n_outputs > 1:
            decoded_outputs = Concatenate(axis=-1)(decoded_srcs)
        else:
            decoded_outputs = decoded_srcs[0]
        logging.debug(f'decoded_srcs {decoded_outputs}')  # (bs=32, fl, n_outputs)
        return decoded_outputs

    def decoder_func(inputs):
        x = decoder_out(inputs, n_outputs, is_multiple_decoder)
        logging.debug(f'x {x}')  # (bs=32, fl, n_outputs)
        decoded = Lambda(lambda x: x[:, 0:input_dim, :], name='Out_clip')(x)
        logging.debug(f'decoded {decoded}')  # (bs=32, input_dim, n_outputs)
        return decoded

    autoencoder = Model(raw_input, decoder_func(encoded))
    logging.debug(autoencoder.summary())
    if model_type == 'separator':
        return autoencoder, None, None, None

    decoder_inputs = Input(shape=K.int_shape(encoded)[1:], dtype=K.dtype(encoded))
    logging.debug(f'decoder_inputs {decoder_inputs}')
    decoder = Model(decoder_inputs, decoder_func(decoder_inputs))
    logging.debug(decoder.summary())

    # Input Padding
    num_encoder_layer = 1

    # Conv1D, LeakyReLU, {BN}, Dcimate
    num_layer_down = 4 if batch_norm else 3
    num_encoder_layer += num_layers * num_layer_down
    # Conv1D, LeakyReLU
    num_encoder_layer += 2
    # BN
    if batch_norm:
        num_encoder_layer += 1

    # encoder_multiple_out, Trick, only use in separator modules
    if encoder_multiple_out:
        num_encoder_layer += n_outputs

    # exp_dims, InterpolationLayer/bilinear_interpol, sq_dims, Conv1D, LeakyReLU, {BN}
    num_layer_up = 6 if batch_norm else 5
    if use_skip:
        # CropConcatLayer
        num_layer_up += 1

    num_decoder_block_layer = num_layers * num_layer_up

    # Conv1D
    num_layer_output = 1
    if output_activation not in {'tanh'}:
        # AudioClipLayer
        num_layer_output += 1

    if is_multiple_decoder:
        # DecoderBlock + Out + Out_clip
        num_decoder_layer = n_outputs * (num_decoder_block_layer + num_layer_output)\
            + 1  # Out_clip

        if n_outputs > 1:
            num_decoder_layer += 1  # Concatenate
    else:
        # DecoderBlock + Out + Out_clip
        if output_type == "direct":
            num_decoder_layer = num_decoder_block_layer\
                + n_outputs * num_layer_output\
                + 1  # Out_clip
        else:  # Difference Output
            # DecoderBlock + Crop_layer + Out + Crop_layer_out, Subtract + Out_clip
            num_decoder_layer = num_decoder_block_layer\
                + 1\
                + (n_outputs-1) * num_layer_output\
                + 2\
                + 1  # Out_clip
            if n_outputs > 3:
                num_decoder_layer += 1  # Add in Difference output

        if n_outputs > 1:
            num_decoder_layer += 1  # Concatenate

    if encoder_multiple_out:
        num_decoder_layer += n_outputs

    return autoencoder, decoder, num_encoder_layer, num_decoder_layer
