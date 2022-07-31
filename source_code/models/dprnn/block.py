import tensorflow as tf
import keras

if keras.__version__ < '2.3.1':
    import keras_layer_normalization


class DprnnBlock(keras.layers.Layer):
    def __init__(self, is_last_dprnn, num_overlapping_chunks, chunk_size, num_filters_in_encoder,
                 units_per_lstm, act_r, use_bias=True, num_speakers=1,
                 **kwargs):
        super(DprnnBlock, self).__init__(**kwargs)
        # self.num_outputs = num_outputs
        self.is_last_dprnn = is_last_dprnn
        # self.batch_size = batch_size
        self.num_overlapping_chunks = num_overlapping_chunks
        self.chunk_size = chunk_size
        self.num_filters_in_encoder = num_filters_in_encoder
        self.units_per_lstm = units_per_lstm
        self.act_r = act_r
        self.use_bias=use_bias,
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
        if keras.__version__ >= '2.3.1':
            self.intra_ln = keras.layers.LayerNormalization(center=False, scale=False)
        else:
            self.intra_ln = keras_layer_normalization.LayerNormalization(center=False, scale=False)

        self.inter_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.units_per_lstm//2,
                                                                      return_sequences=True,
                                                                      use_bias=use_bias,
                                                                      activation=act_r))
        self.inter_fc = keras.layers.Dense(units=self.fc_units, use_bias=use_bias)
        if keras.__version__ >= '2.3.1':
            self.inter_ln = keras.layers.LayerNormalization(center=False, scale=False)
        else:
            self.inter_ln = keras_layer_normalization.LayerNormalization(center=False, scale=False)

    def call(self, T):
        self.batch_size = keras.backend.shape(T)[0]
        # Intra-Chunk Processing
        T_shaped = tf.reshape(T, (self.batch_size*self.num_overlapping_chunks,
                              self.chunk_size, self.num_filters_in_encoder))
        U = self.intra_rnn(T_shaped)
        U = tf.reshape(U, (self.batch_size*self.num_overlapping_chunks*self.chunk_size, self.units_per_lstm))
        # U = keras.layers.Reshape((self.num_overlapping_chunks*self.chunk_size, self.units_per_lstm))(U)
        U_Hat = self.intra_fc(U)
        U_Hat = tf.reshape(U_Hat, (self.batch_size, self.num_overlapping_chunks *
                           self.chunk_size*self.num_filters_in_encoder))
        # U_Hat = keras.layers.Reshape((self.num_overlapping_chunks*self.chunk_size*self.num_filters_in_encoder))(U_Hat)
        LN_U_Hat = self.intra_ln(U_Hat)
        LN_U_Hat = tf.reshape(LN_U_Hat, (self.batch_size, self.num_overlapping_chunks,
                              self.chunk_size, self.num_filters_in_encoder))
        # LN_U_Hat = keras.layers.Reshape((self.num_overlapping_chunks, self.chunk_size, self.num_filters_in_encoder))(LN_U_Hat)
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
