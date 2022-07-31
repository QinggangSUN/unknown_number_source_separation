import tensorflow as tf
import keras
if keras.__version__ < '2.3.1':
    import keras_layer_normalization


class DprnnBlock(keras.layers.Layer):
    def __init__(self, num_outputs, is_last_dprnn, tasnet_with_dprnn, **kwargs):
        super(DprnnBlock, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.is_last_dprnn = is_last_dprnn

        # Copy relevant fields from Tasnet object
        self.batch_size = tasnet_with_dprnn.batch_size
        self.num_overlapping_chunks = tasnet_with_dprnn.num_overlapping_chunks
        self.chunk_size = tasnet_with_dprnn.chunk_size
        self.num_filters_in_encoder = tasnet_with_dprnn.num_filters_in_encoder
        self.units_per_lstm = tasnet_with_dprnn.units_per_lstm

        if is_last_dprnn:
            self.fc_units = self.num_filters_in_encoder*tasnet_with_dprnn.num_speakers
        else:
            self.fc_units = self.num_filters_in_encoder

        self.intra_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.units_per_lstm//2, return_sequences=True))
        self.intra_fc = keras.layers.Dense(units=self.num_filters_in_encoder)
        if keras.__version__ >= '2.3.1':
            self.intra_ln = keras.layers.LayerNormalization(center=False, scale=False)
        else:
            self.intra_ln = keras_layer_normalization.LayerNormalization(center=False, scale=False)

        self.inter_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.units_per_lstm//2, return_sequences=True))
        self.inter_fc = keras.layers.Dense(units=self.fc_units)
        if keras.__version__ >= '2.3.1':
            self.inter_ln = keras.layers.LayerNormalization(center=False, scale=False)
        else:
            self.inter_ln = keras_layer_normalization.LayerNormalization(center=False, scale=False)

    def call(self, T):
        # Intra-Chunk Processing
        T_shaped = tf.reshape(T, (self.batch_size*self.num_overlapping_chunks, self.chunk_size, self.num_filters_in_encoder))
        U = self.intra_rnn(T_shaped)
        U = tf.reshape(U, (self.batch_size*self.num_overlapping_chunks*self.chunk_size, self.units_per_lstm))
        U_Hat = self.intra_fc(U)
        U_Hat = tf.reshape(U_Hat, (self.batch_size, self.num_overlapping_chunks*self.chunk_size*self.num_filters_in_encoder))
        LN_U_Hat = self.intra_ln(U_Hat)
        LN_U_Hat = tf.reshape(LN_U_Hat, (self.batch_size, self.num_overlapping_chunks, self.chunk_size, self.num_filters_in_encoder))
        T_Hat = T + LN_U_Hat

        # Inter-Chunk Processing
        T_Hat = tf.transpose(T_Hat, [0, 2, 1, 3])
        T_Hat_shaped = tf.reshape(T_Hat, (self.batch_size*self.chunk_size, self.num_overlapping_chunks, self.num_filters_in_encoder))
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
