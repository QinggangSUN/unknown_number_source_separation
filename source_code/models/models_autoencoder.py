# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:38:49 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=line-too-long, too-many-arguments, too-many-branches, too-many-locals, too-many-statements
# pylint: disable=invalid-name, logging-fstring-interpolation, no-member, useless-object-inheritance
import logging

import keras
from keras import backend as K
from keras.layers import Activation, BatchNormalization, Concatenate, Conv1D, Conv2DTranspose, Dense, Dropout
from keras.layers import Bidirectional, Input, Lambda, LSTM, MaxPool1D, RepeatVector, Reshape
from keras.layers import TimeDistributed, UpSampling1D
from keras.layers.convolutional import ZeroPadding1D
from keras.models import Model
import tensorflow as tf

# from .dprnn.network import DprnnBlock
from .dprnn.block import DprnnBlock

if tf.__version__ < '1.15':
    from .reconstruction_ops import overlap_and_add
if keras.__version__ < '2.3.1':
    import keras_layer_normalization


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
            if keras.__version__ >= '2.3.1':
                x = keras.layers.LayerNormalization(center=False, scale=False)(x)
            else:
                x = keras_layer_normalization.LayerNormalization(center=False, scale=False)(x)
        if batch_norm and layer_norm:
            logging.warn('batch normalization and layer normalizaion both used')
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
                    if keras.__version__ >= '2.3.1':
                        x = keras.layers.LayerNormalization(center=False, scale=False)(x)
                    else:
                        x = keras_layer_normalization.LayerNormalization(center=False, scale=False)(x)
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


def Conv1DTranspose(input_tensor, filters, kernel_size, strides=None, use_bias=True,
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
        x = Conv1DTranspose(layer_input, filters, kernel_size, strides, activation=act_c, use_bias=use_bias)
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

    # note that "output_shape" isn't necessary with the TensorFlow backend
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

    # note that "output_shape" isn't necessary with the TensorFlow backend
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
                  n_outputs, encoder_multiple_out):
    """Conv 1D + 1D LSTM autoenconder, with frame segment, overlap, and add. Decoder n_outputs RNN."""
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
    logging.debug(f'segmentation out {x}')  # (bs=32, n_chunkds=329, chunk_siz=64, n_filters_conv=1)
    if rnn_type in {'LSTM', 'BLSTM'}:
        x = Reshape(tuple(K.int_shape(x)[1:-2])+(chunk_size*n_filters_conv,))(x)
        logging.debug(f'reshape {x}')  # (bs=32, n_chunkds=329, chunk_size*n_filters_conv=64*64)

    def rnn_layer(rnn_type, x):
        if rnn_type == 'LSTM':
            x = LSTM(units_r, activation=act_r, use_bias=use_bias, return_sequences=True)(x)
        if rnn_type == 'BLSTM':
            x = Bidirectional(LSTM(units_r//2, activation=act_r, use_bias=use_bias, return_sequences=True))(x)
        elif rnn_type == 'dprnn':
            x = DprnnBlock(is_last_dprnn=False, num_overlapping_chunks=n_overlapping_chunks, chunk_size=chunk_size,
                           num_filters_in_encoder=n_filters_conv, units_per_lstm=units_r,
                           use_bias=use_bias, act_r=act_r)(x)
        return x

    # RNN encoder
    for _ in range(n_rnn_encoder):
        x = rnn_layer(rnn_type, x)

    encoded = x
    # (bs, time_step=329, out_dim=units_r) or (bs, time_step=329, out_dim=units_r, n_outputs)
    logging.debug(f'encoded {encoded}')

    def decoder_out(inputs, dim_fc, n_outputs, n_rnn_decoder):
        logging.debug(f'decoder_out inputs {inputs}')  # (bs, time_step=329, units_r, n_outputs)
        decoded_srcs = []
        for i in range(n_outputs):
            if K.ndim(inputs) == 3:  # (bs, time_step=329, out_dim=units_r, n_outputs)
                x = inputs
            elif K.ndim(inputs) == 4:  # (bs, time_step=329, out_dim=units_r, n_outputs)
                x = Lambda(lambda x: x[:, :, :, i])(inputs)
                logging.debug(f'inputs {i} {x}')  # (bs, time_step=329, units_r)
            else:
                raise ValueError('Wrong encoder output dimision.')

            if encoder_multiple_out:  # Trick, only use in separator modules
                x = Dense(units=units_r, use_bias=use_bias)(x)
                logging.debug(f'first dense out {x}')  # (bs, time_step=329, units_r)

            for _ in range(n_rnn_decoder):
                x = rnn_layer(rnn_type, x)
            logging.debug(f'rnn out {x}')  # (32, time_step=329, units_r)
            if rnn_type == 'dprnn':
                x = Reshape(tuple(K.int_shape(x)[1:-2]+(-1,)))(x)
                logging.debug(f'rnn out reshape {x}')  # (32, time_step=329, units_r)

            x = Dense(units=dim_fc, use_bias=use_bias)(x)
            logging.debug(f'dense out {x}')  # (bs, time_step=329, dim_fc)
            if keras.__version__ >= '2.3.1':
                x = keras.layers.LayerNormalization(center=False, scale=False)(x)
            else:
                x = keras_layer_normalization.LayerNormalization(center=False, scale=False)(x)
            logging.debug(f'ln out {x}')  # (bs, time_step=329, dim_fc)
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
        n_rnn_layer = 1
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
    if encoder_multiple_out:
        n_decoder_layer += n_outputs
    # concat
    if n_outputs > 1:
        n_decoder_layer += 1
    # clip
    n_decoder_layer += 1

    return autoencoder, decoder, n_encoder_layer, n_decoder_layer
