# -*- coding: utf-8 -*-
# References:
#   https://github.com/helianvine/conv-tasnet/blob/master/model/modules_v2.py
#   https://github.com/ishine/tf2-ConvTasNet/blob/main/model/algorithm.py
#   https://github.com/kaparoo/Conv-TasNet-Archive/tree/master/conv_tasnet/layers
from error import Error, ParameterError
import sys

import tensorflow
if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow import keras
else:
    import tensorflow as tf
    import keras
from keras.layers import Add, Conv1D, Lambda, PReLU, ZeroPadding1D
if keras.__version__ >= '2.3.1':
    from keras.layers import LayerNormalization
else:
    from keras_layer_normalization import LayerNormalization

from .layers import DepthwiseConv1D, GlobalNormalization
sys.path.append("...")


def normal_layer(inputs, norm_type, axis=None, **kwargs):
    """ Normalization layer.
    Args:
        inputs (keras.Tensor): inputs.
        norm_type (str): type of the normalization.
        axis (list[int], optional): an list of integer specifying the axis that should be normalized. Defaults to None.
    Returns:
        keras.Tensor: tensor after normalization.
    """
    if norm_type == "gln":
        if axis is None:
            axis = [1, 2]
        outputs = GlobalNormalization(axis=axis, **kwargs)(inputs)
    elif norm_type == "cln":
        outputs = LayerNormalization()(inputs)
    return outputs


def conv_block(inputs, filters, kernel_size, use_residual=True):
    """1D Convolutional block.
    Args:
        inputs (Keras.Tensor): input vector of the block.
        filters (int): The number of channels in the internal layers.
        kernel_size (int): The convolution kernel size of the middle layer.
        use_residual (bool, optional): Whether use residual block/output. Defaults to True.
    """
    x = inputs
    x = Conv1D(filters, kernel_size)(x)
    x = PReLU(shared_axes=[1])(x)
    x = LayerNormalization()(x)
    x = Conv1D(filters, kernel_size)(x)
    x = PReLU(shared_axes=[1])(x)
    feature = LayerNormalization()(x)
    if use_residual:
        residual = Conv1D(filters, kernel_size)(feature)
    else:
        residual = None
    skip_out = Conv1D(filters, kernel_size)(feature)
    return residual, skip_out


def tcn_block(inputs, filters, residual_channels, skip_channels, kernel_size, dilation, causal,
              norm_type="gln", use_residual=True):
    """1D Temporal Convolutional block.
    Args:
        inputs (Keras.Tensor): input vector of the block.
        filters (int): The number of filters in the convolutional layers.
        residual_channels (int): The number of filters in the residual channel.
        skip_channels (int): The number of filters in the skip channel.
        kernel_size (int): The convolution kernel size of the middle layer.
        dilation (int): dilation rate of the convolutional layers.
        causal (bool): whether support causal.
        norm_type (str, optional): type of the normalization layer. Defaults to "gln".
        use_residual (bool, optional): Whether use residual block/output. Defaults to True.
    """
    x = inputs
    x = Conv1D(filters, 1)(x)
    x = PReLU(shared_axes=[1])(x)
    if norm_type:
        x = normal_layer(x, norm_type)
    padding_size = dilation * (kernel_size - 1)
    if causal:
        x = ZeroPadding1D((padding_size, 0))(x)
    else:
        x = ZeroPadding1D((padding_size // 2, padding_size // 2))(x)
    x = DepthwiseConv1D(kernel_size, dilation_rate=dilation)(x)  # (bs, fl, filters*(depth_multiplier=1))
    x = PReLU(shared_axes=[1])(x)
    if norm_type:
        x = normal_layer(x, norm_type)
    if use_residual:
        residual = Conv1D(residual_channels, 1)(x)
    else:
        residual = None
    skip_out = Conv1D(skip_channels, 1)(x)
    return residual, skip_out


class TemporalConvBlock(keras.layers.Layer):
    def __init__(self, filters, residual_channels, skip_channels, kernel_size, dilation, causal,
                 norm_type="gln", use_residual=True):
        super(TemporalConvBlock, self).__init__()
        self.filters = filters
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.causal = causal
        self.norm_type = norm_type
        self.use_residual = use_residual

    def call(self, inputs):
        x = inputs  # (bs, fl, feature)
        x = Conv1D(self.filters, 1)(x)  # (bs, fl, filters)
        x = PReLU(shared_axes=[1])(x)
        if self.norm_type:
            x = normal_layer(x, self.norm_type)
        padding_size = self.dilation * (self.kernel_size - 1)
        if self.causal:
            x = ZeroPadding1D((padding_size, 0))(x)
        else:
            x = ZeroPadding1D((padding_size // 2, padding_size // 2))(x)
        x = DepthwiseConv1D(self.kernel_size, dilation_rate=self.dilation)(x)  # (bs, fl, filters*(depth_multiplier=1))
        x = PReLU(shared_axes=[1])(x)
        if self.norm_type:
            x = normal_layer(x, self.norm_type)
        if self.use_residual:
            residual = Conv1D(self.residual_channels, self.kernel_size,
                              padding='same')(x)  # (bs, fl, residual_channels)
        else:
            residual = None
        skip_out = Conv1D(self.skip_channels, self.kernel_size, padding='same')(x)  # (bs, fl, skip_channels)
        return [residual, skip_out]

    def compute_output_shape(self, input_shape):
        return [input_shape[:-1]+(self.residual_channels,), input_shape[:-1]+(self.skip_channels,)]

    def get_config(self):
        config = {
            'filters': self.filters,
            'residual_channels': self.residual_channels,
            'skip_channels': self.skip_channels,
            'kernel_size': self.kernel_size,
            'dilation': self.dilation,
            'causal': self.causal,
            'norm_type': self.norm_type,
            'use_residual': self.use_residual,
        }
        base_config = super(TemporalConvBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def tasnet_branch(inputs, n_repeats, n_layer_each_repeat, n_filters, residual_channels, skip_channels,
                  kernel_size, causal, norm_type, use_residual=True):
    skip_outputs = []
    for _ in range(n_repeats):
        for i_layer in range(n_layer_each_repeat):
            dilation = 2**i_layer
            residual, skip = TemporalConvBlock(n_filters, residual_channels, skip_channels, kernel_size, dilation,
                                               causal=causal, norm_type=norm_type,
                                               use_residual=use_residual)(inputs)
            if residual is not None:  # residual is not None if use_residual is True
                inputs = Add()([inputs, residual])
            skip_outputs.append(skip)
    if len(skip_outputs) > 1:
        tcn_output = Add()(skip_outputs)
    else:  # len(skip_outputs) == 1
        tcn_output = skip_outputs[0]
    tcn_output = Add()([tcn_output, inputs])
    return tcn_output  # (bs, fl, skip_channels)


def tcn_branch(inputs, n_repeats, n_layer_each_repeat, n_filters, residual_channels, skip_channels,
               kernel_size, causal, norm_type, use_residual=True, use_skip=True, use_sum_repeats=False):
    if not use_residual and not use_skip:
        raise ParameterError('use_residual, use_skip cannot both False')

    skip_outputs = []
    for i_repeat in range(n_repeats):
        skip_outputs_repeat_i = []
        for i_layer in range(n_layer_each_repeat):
            dilation = 2**i_layer
            residual, skip = tcn_block(inputs, n_filters, residual_channels, skip_channels, kernel_size, dilation,
                                       causal, norm_type, use_residual)

            if use_residual:  # residual is not None if use_residual is True
                inputs = Add()([inputs, residual])
            if use_skip:
                skip_outputs_repeat_i.append(skip)
        if len(skip_outputs_repeat_i) > 1:
            sum_skips_repeat_i = Add()(skip_outputs_repeat_i)
        elif len(skip_outputs_repeat_i) == 1:
            sum_skips_repeat_i = skip_outputs_repeat_i[0]
        else:  # use_skip is False, len(skip_outputs) == 0
            sum_skips_repeat_i = None
        if use_skip:
            skip_outputs.append(sum_skips_repeat_i)
            if use_sum_repeats:
                inputs = Add()([inputs, sum_skips_repeat_i])
    if use_skip:
        if len(skip_outputs) > 1:  # n_repeats > 1
            tcn_output = Add()(skip_outputs)
        elif len(skip_outputs) == 1:  # n_repeats == 1
            tcn_output = skip_outputs[0]
        else:
            raise Error('n_repeats may less than 1')
        if use_sum_repeats:
            tcn_output = Add()([tcn_output, inputs])
    else:  # use_skip is False, len(skip_outputs) == 0
        tcn_output = inputs
    return tcn_output  # (bs, fl, skip_channels)


class TcnBranch(keras.layers.Layer):
    def __init__(self, n_repeats, n_layer_each_repeat, n_filters, residual_channels, skip_channels,
                 kernel_size, causal, norm_type, use_residual=True, use_skip=True, use_sum_repeats=False):
        super(TcnBranch, self).__init__()
        self.n_repeats = n_repeats
        self.n_layer_each_repeat = n_layer_each_repeat
        self.n_filters = n_filters
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.kernel_size = kernel_size
        self.causal = causal
        self.norm_type = norm_type
        self.use_residual = use_residual
        self.use_skip = use_skip
        self.use_sum_repeats = use_sum_repeats

    def call(self, inputs):
        return tcn_branch(inputs, self.n_repeats, self.n_layer_each_repeat, self.n_filters,
                          self.residual_channels, self.skip_channels,
                          self.kernel_size, self.causal, self.norm_type,
                          use_residual=self.use_residual, use_skip=self.use_skip, use_sum_repeats=self.use_sum_repeats)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            if self.residual_channels is not None:
                output_shape = input_shape[:1]+(self.residual_channels,)+input_shape[2:]
            else:
                output_shape = input_shape[:1] + (self.skip_channels,)+input_shape[2:]
        elif self.data_format == 'channels_last':
            if self.residual_channels is not None:
                output_shape = input_shape[:-1]+(self.residual_channels,)
            else:
                output_shape = input_shape[:-1] + (self.skip_channels,)
        return output_shape

    def get_config(self):
        config = {
            'n_repeats': self.n_repeats,
            'n_layer_each_repeat': self.n_layer_each_repeat,
            'n_filters': self.n_filters,
            'residual_channels': self.residual_channels,
            'skip_channels': self.skip_channels,
            'kernel_size': self.kernel_size,
            'causal': self.causal,
            'norm_type': self.norm_type,
            'use_residual': self.use_residual,
            'use_skip': self.use_skip,
            'use_sum_repeats': self.use_sum_repeats,
        }
        base_config = super(TcnBranch, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
