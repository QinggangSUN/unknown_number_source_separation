# -*- coding: utf-8 -*-
# Reference https://github.com/sp-uhh/dual-path-rnn


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
from keras.layers import Add, Bidirectional, Dense, Lambda, LSTM, Permute
if keras.__version__ >= '2.3.1':
    from keras.layers import LayerNormalization
else:
    from keras_layer_normalization import LayerNormalization


class DprnnBlock(keras.layers.Layer):
    """This layer does not have weights when only save weights of model."""

    def __init__(self, is_last_dprnn, num_overlapping_chunks, chunk_size, num_filters_in_encoder,
                 units_per_lstm, act_r, use_bias=True, num_speakers=1,
                 **kwargs):
        super(DprnnBlock, self).__init__(**kwargs)
        self.is_last_dprnn = is_last_dprnn
        self.num_overlapping_chunks = num_overlapping_chunks
        self.chunk_size = chunk_size
        self.num_filters_in_encoder = num_filters_in_encoder
        self.units_per_lstm = units_per_lstm
        self.act_r = act_r
        self.use_bias = use_bias,
        self.num_speakers = num_speakers

        if is_last_dprnn:
            self.fc_units = self.num_filters_in_encoder*num_speakers
        else:
            self.fc_units = self.num_filters_in_encoder

        self.intra_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.units_per_lstm//2,
                                                                      return_sequences=True,
                                                                      use_bias=use_bias,
                                                                      activation=act_r))
        self.intra_fc = keras.layers.Dense(units=self.num_filters_in_encoder, use_bias=use_bias)
        self.intra_ln = LayerNormalization(center=False, scale=False)

        self.inter_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.units_per_lstm//2,
                                                                      return_sequences=True,
                                                                      use_bias=use_bias,
                                                                      activation=act_r))
        self.inter_fc = keras.layers.Dense(units=self.fc_units, use_bias=use_bias)
        self.inter_ln = LayerNormalization(center=False, scale=False)

    def call(self, T):
        self.batch_size = keras.backend.shape(T)[0]
        # Intra-Chunk Processing
        T_shaped = tf.reshape(T, (self.batch_size*self.num_overlapping_chunks,
                              self.chunk_size, self.num_filters_in_encoder))
        U = self.intra_rnn(T_shaped)
        U = tf.reshape(U, (self.batch_size*self.num_overlapping_chunks*self.chunk_size, self.units_per_lstm))
        U_Hat = self.intra_fc(U)
        U_Hat = tf.reshape(U_Hat, (self.batch_size, self.num_overlapping_chunks *
                           self.chunk_size*self.num_filters_in_encoder))
        LN_U_Hat = self.intra_ln(U_Hat)
        LN_U_Hat = tf.reshape(LN_U_Hat, (self.batch_size, self.num_overlapping_chunks,
                              self.chunk_size, self.num_filters_in_encoder))
        T_Hat = T + LN_U_Hat

        # Inter-Chunk Processing
        T_Hat = tf.transpose(T_Hat, [0, 2, 1, 3])
        T_Hat_shaped = tf.reshape(T_Hat, (self.batch_size*self.chunk_size,
                                  self.num_overlapping_chunks, self.num_filters_in_encoder))
        V = self.inter_rnn(T_Hat_shaped)
        V = tf.reshape(V, (self.batch_size*self.chunk_size*self.num_overlapping_chunks, self.units_per_lstm))
        V_Hat = self.inter_fc(V)
        V_Hat = tf.reshape(V_Hat, (self.batch_size, self.num_overlapping_chunks*self.fc_units*self.chunk_size))
        LN_V_Hat = self.inter_ln(V_Hat)
        T_Out = tf.reshape(LN_V_Hat, (self.batch_size, self.chunk_size, self.num_overlapping_chunks, self.fc_units))
        if not self.is_last_dprnn:
            T_Out = T_Hat + T_Out
        T_Out = tf.transpose(T_Out, [0, 2, 1, 3])

        return T_Out

    def get_config(self):
        config = {
            'is_last_dprnn': self.is_last_dprnn,
            'num_overlapping_chunks': self.num_overlapping_chunks,
            'chunk_size': self.chunk_size,
            'num_filters_in_encoder': self.num_filters_in_encoder,
            'units_per_lstm': self.units_per_lstm,
            'use_bias': self.use_bias,
            'act_r': self.act_r,
            'num_speakers': self.num_speakers,
        }
        base_config = super(DprnnBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DprnnBlockModel(keras.models.Model):
    """Cannot use model.save() to save model. https://keras.io/zh/models/about-keras-models/ """

    def __init__(self, is_last_dprnn, num_overlapping_chunks, chunk_size, num_filters_in_encoder,
                 units_per_lstm, act_r, use_bias=True, num_speakers=1, batch_size=None,
                 **kwargs):
        super(DprnnBlockModel, self).__init__(**kwargs)
        self.is_last_dprnn = is_last_dprnn
        self.batch_size = batch_size
        self.num_overlapping_chunks = num_overlapping_chunks
        self.chunk_size = chunk_size
        self.num_filters_in_encoder = num_filters_in_encoder
        self.units_per_lstm = units_per_lstm
        self.act_r = act_r
        self.use_bias = use_bias,
        self.num_speakers = num_speakers

        if is_last_dprnn:
            self.fc_units = self.num_filters_in_encoder*num_speakers
        else:
            self.fc_units = self.num_filters_in_encoder

        self.intra_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.units_per_lstm//2,
                                                                      return_sequences=True,
                                                                      use_bias=use_bias,
                                                                      activation=act_r))
        self.intra_fc = keras.layers.Dense(units=self.num_filters_in_encoder, use_bias=use_bias)
        self.intra_ln = LayerNormalization(center=False, scale=False)

        self.inter_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.units_per_lstm//2,
                                                                      return_sequences=True,
                                                                      use_bias=use_bias,
                                                                      activation=act_r))
        self.inter_fc = keras.layers.Dense(units=self.fc_units, use_bias=use_bias)
        self.inter_ln = LayerNormalization(center=False, scale=False)

    def build(self, input_shape):
        super(DprnnBlockModel, self).build(input_shape)

    def call(self, T):
        # Intra-Chunk Processing
        T_shaped = tf.reshape(T, (self.batch_size*self.num_overlapping_chunks,
                              self.chunk_size, self.num_filters_in_encoder))
        U = self.intra_rnn(T_shaped)
        U = tf.reshape(U, (self.batch_size*self.num_overlapping_chunks*self.chunk_size, self.units_per_lstm))
        U_Hat = self.intra_fc(U)
        U_Hat = tf.reshape(U_Hat, (self.batch_size, self.num_overlapping_chunks *
                           self.chunk_size*self.num_filters_in_encoder))
        LN_U_Hat = self.intra_ln(U_Hat)
        LN_U_Hat = tf.reshape(LN_U_Hat, (self.batch_size, self.num_overlapping_chunks,
                              self.chunk_size, self.num_filters_in_encoder))
        T_Hat = Add()([T, LN_U_Hat])

        # Inter-Chunk Processing
        T_Hat = tf.transpose(T_Hat, [0, 2, 1, 3])
        T_Hat_shaped = tf.reshape(T_Hat, (self.batch_size*self.chunk_size,
                                  self.num_overlapping_chunks, self.num_filters_in_encoder))
        V = self.inter_rnn(T_Hat_shaped)
        V = tf.reshape(V, (self.batch_size*self.chunk_size*self.num_overlapping_chunks, self.units_per_lstm))
        V_Hat = self.inter_fc(V)
        V_Hat = tf.reshape(V_Hat, (self.batch_size, self.num_overlapping_chunks*self.fc_units*self.chunk_size))
        LN_V_Hat = self.inter_ln(V_Hat)
        T_Out = tf.reshape(LN_V_Hat, (self.batch_size, self.chunk_size, self.num_overlapping_chunks, self.fc_units))
        if not self.is_last_dprnn:
            T_Out = Add()([T_Hat, T_Out])
        T_Out = tf.transpose(T_Out, [0, 2, 1, 3])

        # model = keras.models.Model(inputs=T, outputs=T_Out)
        # return model
        return T_Out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'is_last_dprnn': self.is_last_dprnn,
            'num_overlapping_chunks': self.num_overlapping_chunks,
            'chunk_size': self.chunk_size,
            'num_filters_in_encoder': self.num_filters_in_encoder,
            'units_per_lstm': self.units_per_lstm,
            'use_bias': self.use_bias,
            'act_r': self.act_r,
            'num_speakers': self.num_speakers,
            'batch_size': self.batch_size,
        }
        base_config = super(DprnnBlockModel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def dprnn_block(T, is_last_dprnn, num_overlapping_chunks, chunk_size, num_filters_in_encoder,
                units_per_lstm, act_r, batch_size, use_bias=True, num_speakers=1, **kwargs):

    if is_last_dprnn:
        fc_units = num_filters_in_encoder*num_speakers
    else:
        fc_units = num_filters_in_encoder

    # T.shape == (bs, time_step=329, chunk_size=64, num_filters_in_encoder=64)
    # Intra-Chunk Processing
    T_shaped = Lambda(lambda x: K.reshape(x, (batch_size*num_overlapping_chunks,
                      chunk_size, num_filters_in_encoder)))(T)
    # (bs, time_step=329, chunk_size=64, num_filters_in_encoder=64)
    U = Bidirectional(LSTM(units=units_per_lstm//2,
                           return_sequences=True,
                           use_bias=use_bias,
                           activation=act_r))(T_shaped)
    # intra_rnn  # (bs*time_step=8*329, chunk_size=64, units_per_lstm=256)
    U = Lambda(lambda x: K.reshape(x, (batch_size*num_overlapping_chunks*chunk_size, units_per_lstm)))(U)
    U_Hat = Dense(units=num_filters_in_encoder, use_bias=use_bias)(U)  # intra_fc
    U_Hat = Lambda(lambda x: K.reshape(x, (batch_size, num_overlapping_chunks *
                                           chunk_size*num_filters_in_encoder)))(U_Hat)
    LN_U_Hat = LayerNormalization(center=False, scale=False)(U_Hat)  # intra_ln
    LN_U_Hat = Lambda(lambda x: K.reshape(x, (batch_size, num_overlapping_chunks,
                                              chunk_size, num_filters_in_encoder)))(LN_U_Hat)
    T_Hat = Add()([T, LN_U_Hat])  # (bs, time_step=329, chunk_size=64, num_filters_in_encoder=64)

    # Inter-Chunk Processing
    T_Hat = Permute((2, 1, 3))(T_Hat)
    T_Hat_shaped = Lambda(lambda x: K.reshape(x, (batch_size*chunk_size,
                                                  num_overlapping_chunks, num_filters_in_encoder)))(T_Hat)
    V = Bidirectional(LSTM(units=units_per_lstm//2,
                           return_sequences=True,
                           use_bias=use_bias,
                           activation=act_r))(T_Hat_shaped)  # inter_rnn
    V = Lambda(lambda x: K.reshape(x, (batch_size*chunk_size*num_overlapping_chunks, units_per_lstm)))(V)
    V_Hat = Dense(units=fc_units, use_bias=use_bias)(V)  # inter_fc
    V_Hat = Lambda(lambda x: K.reshape(x, (batch_size, num_overlapping_chunks*fc_units*chunk_size)))(V_Hat)
    LN_V_Hat = LayerNormalization(center=False, scale=False)(V_Hat)  # inter_ln
    T_Out = Lambda(lambda x: K.reshape(x, (batch_size, chunk_size, num_overlapping_chunks, fc_units)))(LN_V_Hat)
    if not is_last_dprnn:
        T_Out = Add()([T_Hat, T_Out])
    T_Out = Permute((2, 1, 3))(T_Out)  # (bs, time_step=329, chunk_size=64, num_filters_in_encoder=64)

    return T_Out
