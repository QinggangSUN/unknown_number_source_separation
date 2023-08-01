# -*- coding: utf-8 -*-
# https://github.com/helianvine/conv-tasnet/
import tensorflow
if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow import keras
    # import keras
    import tensorflow.compat.v1.keras.backend as K
    from tensorflow.python.keras.utils import conv_utils
    from tensorflow.python.keras.utils.conv_utils import normalize_data_format
else:
    import tensorflow as tf
    import keras
    import keras.utils.conv_utils as conv_utils
    from keras import backend as K
    from keras.backend import normalize_data_format
if keras.__version__ >= '2.3.1':
    from keras.layers import LayerNormalization
else:
    from keras_layer_normalization import LayerNormalization
from keras.layers import Conv2DTranspose, DepthwiseConv2D, InputSpec, Lambda


class Conv1DTranspose(keras.layers.Layer):
    """Conv1DTranspose from keras.layers.Conv2DTranspose."""

    def __init__(self, filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(Conv1DTranspose, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.output_padding = output_padding
        self.data_format = data_format
        self.dilation_rate = dilation_rate
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.activity_regularizer = activity_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def call(self, inputs):
        outputs = K.expand_dims(inputs, axis=2)
        outputs = Conv2DTranspose(filters=self.filters,
                                  kernel_size=(self.kernel_size, 1),
                                  strides=(self.strides, 1),
                                  padding=self.padding,
                                  output_padding=self.output_padding,
                                  data_format=self.data_format,
                                  dilation_rate=(self.dilation_rate, 1),
                                  activation=self.activation,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  activity_regularizer=self.activity_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  bias_constraint=self.bias_constraint,)(outputs)
        outputs = K.squeeze(outputs, axis=2)
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = list(input_shape)
        if self.data_format == 'channels_first':
            c_axis, h_axis = 1, 2
        else:
            c_axis, h_axis = 2, 1

        kernel_h = self.kernel_size
        stride_h = self.strides
        if self.output_padding is None:
            out_pad_h = None
        else:
            out_pad_h = self.output_padding

        output_shape[c_axis] = self.filters
        output_shape[h_axis] = conv_utils.deconv_length(output_shape[h_axis],
                                                        stride_h,
                                                        kernel_h,
                                                        self.padding,
                                                        out_pad_h,
                                                        self.dilation_rate)
        return tuple(output_shape)

    def get_config(self):
        config = {
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'output_padding': self.output_padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': self.activation,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint,
        }
        base_config = super(Conv1DTranspose, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def depth_wise_conv1d(input_tensor, kernel_size, strides=1,
                      padding='valid',
                      depth_multiplier=1,
                      data_format=None,
                      activation=None,
                      use_bias=True,
                      depthwise_initializer='glorot_uniform',
                      bias_initializer='zeros',
                      depthwise_regularizer=None,
                      bias_regularizer=None,
                      activity_regularizer=None,
                      depthwise_constraint=None,
                      bias_constraint=None, **kwargs):
    """DepthwiseConv1D from keras.layers.DepthwiseConv2D."""

    output_tensor = Lambda(lambda x: K.expand_dims(x, axis=2))(input_tensor)
    output_tensor = DepthwiseConv2D(kernel_size=(kernel_size, 1), strides=(strides, 1),
                                    padding=padding,
                                    depth_multiplier=depth_multiplier,
                                    data_format=data_format,
                                    activation=activation,
                                    use_bias=use_bias,
                                    depthwise_initializer=depthwise_initializer,
                                    bias_initializer=bias_initializer,
                                    depthwise_regularizer=depthwise_regularizer,
                                    bias_regularizer=bias_regularizer,
                                    activity_regularizer=activity_regularizer,
                                    depthwise_constraint=depthwise_constraint,
                                    bias_constraint=bias_constraint,
                                    **kwargs
                                    )(output_tensor)
    output_tensor = Lambda(lambda x: K.squeeze(x, axis=2))(output_tensor)
    return output_tensor


class _Conv(keras.layers.Layer):
    """Abstract nD convolution layer (private, used as implementation base).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of outputs.
    If `use_bias` is True, a bias vector is created and added to the outputs.
    Finally, if `activation` is not `None`,
    it is applied to the outputs as well.

    # Arguments
        rank: An integer, the rank of the convolution,
            e.g. "2" for 2D convolution.
        filters: Integer, the dimensionality of the output space
            (i.e. the number of output filters in the convolution).
        kernel_size: An integer or tuple/list of n integers, specifying the
            dimensions of the convolution window.
        strides: An integer or tuple/list of n integers,
            specifying the strides of the convolution.
            Specifying any stride value != 1 is incompatible with specifying
            any `dilation_rate` value != 1.
        padding: One of `"valid"` or `"same"` (case-insensitive).
        data_format: A string,
            one of `"channels_last"` or `"channels_first"`.
            The ordering of the dimensions in the inputs.
            `"channels_last"` corresponds to inputs with shape
            `(batch, ..., channels)` while `"channels_first"` corresponds to
            inputs with shape `(batch, channels, ...)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        dilation_rate: An integer or tuple/list of n integers, specifying
            the dilation rate to use for dilated convolution.
            Currently, specifying any `dilation_rate` value != 1 is
            incompatible with specifying any `strides` value != 1.
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to the kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    """

    def __init__(self, rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(_Conv, self).__init__(**kwargs)
        self.rank = rank
        self.filters = filters
        self.kernel_size = conv_utils.normalize_tuple(kernel_size, rank, 'kernel_size')
        self.strides = conv_utils.normalize_tuple(strides, rank, 'strides')
        self.padding = conv_utils.normalize_padding(padding)
        self.data_format = normalize_data_format(data_format)
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, rank, 'dilation_rate')
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.input_spec = InputSpec(ndim=self.rank + 2)

    def build(self, input_shape):
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=self.rank + 2,
                                    axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs):
        if self.rank == 1:
            outputs = K.conv1d(
                inputs,
                self.kernel,
                strides=self.strides[0],
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate[0])
        if self.rank == 2:
            outputs = K.conv2d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)
        if self.rank == 3:
            outputs = K.conv3d(
                inputs,
                self.kernel,
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0],) + tuple(new_space) + (self.filters,)
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding=self.padding,
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            return (input_shape[0], self.filters) + tuple(new_space)

    def get_config(self):
        config = {
            'rank': self.rank,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'data_format': self.data_format,
            'dilation_rate': self.dilation_rate,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                keras.regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint)
        }
        base_config = super(_Conv, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DepthwiseConv(_Conv):
    """Depthwise convolution.
    Depthwise convolution is a type of convolution in which each input channel
    is convolved with a different kernel (called a depthwise kernel). You can
    understand depthwise convolution as the first step in a depthwise separable
    convolution.
    It is implemented via the following steps:
    - Split the input into individual channels.
    - Convolve each channel with an individual depthwise kernel with
      `depth_multiplier` output channels.
    - Concatenate the convolved outputs along the channels axis.
    Unlike a regular convolution, depthwise convolution does not mix
    information across different input channels.
    The `depth_multiplier` argument determines how many filter are applied to
    one input channel. As such, it controls the amount of output channels that
    are generated per input channel in the depthwise step.
    Args:
      kernel_size: A tuple or list of integers specifying the spatial dimensions
        of the filters. Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: A tuple or list of integers specifying the strides of the
        convolution. Can be a single integer to specify the same value for all
        spatial dimensions. Specifying any `stride` value != 1 is incompatible
        with specifying any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive). `"valid"` means
        no padding. `"same"` results in padding with zeros evenly to the
        left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      depth_multiplier: The number of depthwise convolution output channels for
        each input channel. The total number of depthwise convolution output
        channels will be equal to `filters_in * depth_multiplier`.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`.  The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch_size, height,
        width, channels)` while `channels_first` corresponds to inputs with
        shape `(batch_size, channels, height, width)`. It defaults to the
        `image_data_format` value found in your Keras config file at
        `~/.keras/keras.json`. If you never set it, then it will be
        'channels_last'.
      dilation_rate: An integer or tuple/list of 2 integers, specifying the
        dilation rate to use for dilated convolution. Currently, specifying any
        `dilation_rate` value != 1 is incompatible with specifying any `strides`
        value != 1.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      depthwise_initializer: Initializer for the depthwise kernel matrix (see
        `keras.initializers`). If None, the default initializer
        ('glorot_uniform') will be used.
      bias_initializer: Initializer for the bias vector (see
        `keras.initializers`). If None, the default initializer ('zeros') will
        be used.
      depthwise_regularizer: Regularizer function applied to the depthwise
        kernel matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector (see
        `keras.regularizers`).
      activity_regularizer: Regularizer function applied to the output of the
        layer (its 'activation') (see `keras.regularizers`).
      depthwise_constraint: Constraint function applied to the depthwise kernel
        matrix (see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector (see
        `keras.constraints`).
    Input shape:
      4D tensor with shape: `[batch_size, channels, rows, cols]` if
        data_format='channels_first'
      or 4D tensor with shape: `[batch_size, rows, cols, channels]` if
        data_format='channels_last'.
    Output shape:
      4D tensor with shape: `[batch_size, channels * depth_multiplier, new_rows,
        new_cols]` if `data_format='channels_first'`
        or 4D tensor with shape: `[batch_size,
        new_rows, new_cols, channels * depth_multiplier]` if
        `data_format='channels_last'`. `rows` and `cols` values might have
        changed due to padding.
    Returns:
      A tensor of rank 4 representing
      `activation(depthwiseconv2d(inputs, kernel) + bias)`.
    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.
    # ==============================================================================
    # Copyright 2015 The TensorFlow Authors. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    """

    def __init__(
        self,
        rank,
        kernel_size,
        strides=1,
        padding="valid",
        depth_multiplier=1,
        data_format=None,
        dilation_rate=1,
        activation=None,
        use_bias=True,
        depthwise_initializer="glorot_uniform",
        bias_initializer="zeros",
        depthwise_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        depthwise_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            rank,
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs,
        )
        self.depth_multiplier = depth_multiplier
        self.depthwise_initializer = keras.initializers.get(depthwise_initializer)
        self.depthwise_regularizer = keras.regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = keras.constraints.get(depthwise_constraint)
        self.bias_initializer = keras.initializers.get(bias_initializer)

    def build(self, input_shape):
        if len(input_shape) != self.rank + 2:
            raise ValueError(
                "Inputs to `DepthwiseConv` should have "
                f"rank {self.rank + 2}. "
                f"Received input_shape={input_shape}."
            )
        input_shape = tf.TensorShape(input_shape)
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        channel_axis = channel_axis
        if input_shape.dims[channel_axis].value is None:
            raise ValueError(
                "The channel dimension of the inputs to `DepthwiseConv` "
                "should be defined. "
                f"The input_shape received is {input_shape}, "
                f"where axis {channel_axis} (0-based) "
                "is the channel dimension, which found to be `None`."
            )
        input_dim = int(input_shape[channel_axis])
        depthwise_kernel_shape = self.kernel_size + (
            input_dim,
            self.depth_multiplier,
        )

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name="depthwise_kernel",
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                shape=(input_dim * self.depth_multiplier,),
                initializer=self.bias_initializer,
                name="bias",
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
            )
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(
            min_ndim=self.rank + 2, axes={channel_axis: input_dim}
        )
        self.built = True

    def call(self, inputs):
        raise NotImplementedError

    def get_config(self):
        config = super().get_config()
        config.pop('rank')
        config.pop("filters")
        config.pop("kernel_initializer")
        config.pop("kernel_regularizer")
        config.pop("kernel_constraint")
        config["depth_multiplier"] = self.depth_multiplier
        config["depthwise_initializer"] = keras.initializers.serialize(
            self.depthwise_initializer
        )
        config["depthwise_regularizer"] = keras.regularizers.serialize(
            self.depthwise_regularizer
        )
        config["depthwise_constraint"] = keras.constraints.serialize(
            self.depthwise_constraint
        )
        return config


class DepthwiseConv1D(DepthwiseConv):
    """Depthwise 1D convolution.
    Depthwise convolution is a type of convolution in which each input channel
    is convolved with a different kernel (called a depthwise kernel). You can
    understand depthwise convolution as the first step in a depthwise separable
    convolution.
    It is implemented via the following steps:
    - Split the input into individual channels.
    - Convolve each channel with an individual depthwise kernel with
      `depth_multiplier` output channels.
    - Concatenate the convolved outputs along the channels axis.
    Unlike a regular 1D convolution, depthwise convolution does not mix
    information across different input channels.
    The `depth_multiplier` argument determines how many filter are applied to
    one input channel. As such, it controls the amount of output channels that
    are generated per input channel in the depthwise step.
    Args:
      kernel_size: An integer, specifying the height and width of the 1D
        convolution window. Can be a single integer to specify the same value
        for all spatial dimensions.
      strides: An integer, specifying the strides of the convolution along the
        height and width. Can be a single integer to specify the same value for
        all spatial dimensions. Specifying any stride value != 1 is incompatible
        with specifying any `dilation_rate` value != 1.
      padding: one of `'valid'` or `'same'` (case-insensitive). `"valid"` means
        no padding. `"same"` results in padding with zeros evenly to the
        left/right or up/down of the input such that output has the same
        height/width dimension as the input.
      depth_multiplier: The number of depthwise convolution output channels for
        each input channel. The total number of depthwise convolution output
        channels will be equal to `filters_in * depth_multiplier`.
      data_format: A string, one of `channels_last` (default) or
        `channels_first`.  The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape `(batch_size, height,
        width, channels)` while `channels_first` corresponds to inputs with
        shape `(batch_size, channels, height, width)`. It defaults to the
        `image_data_format` value found in your Keras config file at
        `~/.keras/keras.json`. If you never set it, then it will be
        'channels_last'.
      dilation_rate: A single integer, specifying the dilation rate to use for
        dilated convolution. Currently, specifying any `dilation_rate`
        value != 1 is incompatible with specifying any stride value != 1.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      depthwise_initializer: Initializer for the depthwise kernel matrix (see
        `keras.initializers`). If None, the default initializer
        ('glorot_uniform') will be used.
      bias_initializer: Initializer for the bias vector (see
        `keras.initializers`). If None, the default initializer ('zeros') will
        be used.
      depthwise_regularizer: Regularizer function applied to the depthwise
        kernel matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector (see
        `keras.regularizers`).
      activity_regularizer: Regularizer function applied to the output of the
        layer (its 'activation') (see `keras.regularizers`).
      depthwise_constraint: Constraint function applied to the depthwise kernel
        matrix (see `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector (see
        `keras.constraints`).
    Input shape:
      3D tensor with shape: `[batch_size, channels, input_dim]` if
        data_format='channels_first'
      or 3D tensor with shape: `[batch_size, input_dim, channels]` if
        data_format='channels_last'.
    Output shape:
      3D tensor with shape:
       `[batch_size, channels * depth_multiplier, new_dims]`
        if `data_format='channels_first'`
        or 3D tensor with shape: `[batch_size,
        new_dims, channels * depth_multiplier]` if
        `data_format='channels_last'`. `new_dims` values might have
        changed due to padding.
    Returns:
      A tensor of rank 3 representing
      `activation(depthwiseconv1d(inputs, kernel) + bias)`.
    Raises:
      ValueError: if `padding` is "causal".
      ValueError: when both `strides` > 1 and `dilation_rate` > 1.
    # ==============================================================================
    # Copyright 2015 The TensorFlow Authors. All Rights Reserved.
    #
    # Licensed under the Apache License, Version 2.0 (the "License");
    # you may not use this file except in compliance with the License.
    # You may obtain a copy of the License at
    #
    #     http://www.apache.org/licenses/LICENSE-2.0
    #
    # Unless required by applicable law or agreed to in writing, software
    # distributed under the License is distributed on an "AS IS" BASIS,
    # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    # See the License for the specific language governing permissions and
    # limitations under the License.
    """

    def __init__(
            self,
            kernel_size,
            strides=1,
            padding="valid",
            depth_multiplier=1,
            data_format=None,
            dilation_rate=1,
            activation=None,
            use_bias=True,
            depthwise_initializer="glorot_uniform",
            bias_initializer="zeros",
            depthwise_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            depthwise_constraint=None,
            bias_constraint=None,
            **kwargs
    ):
        super().__init__(
            1,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=depth_multiplier,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            depthwise_initializer=depthwise_initializer,
            bias_initializer=bias_initializer,
            depthwise_regularizer=depthwise_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            depthwise_constraint=depthwise_constraint,
            bias_constraint=bias_constraint,
            **kwargs
        )

    def call(self, inputs):
        def convert_data_format(data_format, ndim):
            if data_format == "channels_last":
                if ndim == 3:
                    return "NWC"
                elif ndim == 4:
                    return "NHWC"
                elif ndim == 5:
                    return "NDHWC"
                else:
                    raise ValueError(
                        f"Input rank not supported: {ndim}. "
                        "Expected values are [3, 4, 5]"
                    )
            elif data_format == "channels_first":
                if ndim == 3:
                    return "NCW"
                elif ndim == 4:
                    return "NCHW"
                elif ndim == 5:
                    return "NCDHW"
                else:
                    raise ValueError(
                        f"Input rank not supported: {ndim}. "
                        "Expected values are [3, 4, 5]"
                    )
            else:
                raise ValueError(
                    f"Invalid data_format: {data_format}. "
                    'Expected values are ["channels_first", "channels_last"]'
                )

        if self.data_format == "channels_last":
            strides = (1,) + self.strides * 2 + (1,)
            spatial_start_dim = 1
        else:
            strides = (1, 1) + self.strides * 2
            spatial_start_dim = 2
        inputs = K.expand_dims(inputs, spatial_start_dim)
        depthwise_kernel = tf.expand_dims(self.depthwise_kernel, axis=0)
        dilation_rate = (1,) + self.dilation_rate

        # outputs = K.depthwise_conv2d(
        #     inputs,
        #     depthwise_kernel,
        #     strides=strides,
        #     padding=self.padding,
        #     dilation_rate=dilation_rate,
        #     data_format=self.data_format,
        # )
        outputs = tf.nn.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=strides,
            padding=self.padding.upper(),
            rate=dilation_rate,
            data_format=convert_data_format(
                self.data_format, ndim=4
            ),
        )

        # if self.use_bias:
        #     outputs = K.bias_add(
        #         outputs,
        #         self.bias,
        #         data_format=self.data_format,
        #     )
        if self.use_bias:
            outputs = tf.nn.bias_add(
                outputs,
                self.bias,
                data_format=convert_data_format(
                    self.data_format, ndim=4
                ),
            )
        outputs = K.squeeze(outputs, axis=spatial_start_dim)
        # outputs = tf.squeeze(outputs, [spatial_start_dim])

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == "channels_first":
            input_dim = input_shape[2]
            out_filters = input_shape[1] * self.depth_multiplier
        elif self.data_format == "channels_last":
            input_dim = input_shape[1]
            out_filters = input_shape[2] * self.depth_multiplier

        input_dim = conv_utils.conv_output_length(
            input_dim,
            self.kernel_size[0],
            self.padding,
            self.strides[0],
            self.dilation_rate[0],
        )
        if self.data_format == "channels_first":
            return (input_shape[0], out_filters, input_dim)
        elif self.data_format == "channels_last":
            return (input_shape[0], input_dim, out_filters)


class NormalLayer(keras.layers.Layer):
    """Normalization layer"""

    def __init__(self, norm_type, axis=None, **kwargs):
        super(NormalLayer, self).__init__(**kwargs)
        self.axis = axis
        self.norm_type = norm_type
        if norm_type == "gln":
            self.norm_layer = GlobalNormalization(axis=axis, **kwargs)
        elif norm_type == "cln":
            self.norm_layer = LayerNormalization()

    def build(self, input_shape):
        self.norm_layer.build(input_shape)

    def call(self, inputs):
        return self.norm_layer(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis,
            'norm_type': self.norm_type,
            'norm_layer': self.norm_layer,
        }
        base_config = super(NormalLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GlobalNormalization(keras.layers.Layer):
    """ Global Layer Normalization layer."""

    def __init__(self,
                 axis=[1, 2],
                 center=True,
                 scale=True,
                 epsilon=1e-8,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GlobalNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta, self.gamma = None, None

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.center:
            self.beta = self.add_weight(name='beta',
                                        shape=input_shape[-1:],
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        trainable=True)
        if self.scale:
            self.gamma = self.add_weight(name='gamma',
                                         shape=input_shape[-1:],
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         trainable=True)
        # Be sure to call this at the end
        super(GlobalNormalization, self).build(input_shape)

    def call(self, inputs):
        # means, variances = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        # x = self.gamma * (inputs - means) / tf.pow(variances + EPS, 0.5) + self.beta
        mean = K.mean(inputs, axis=self.axis, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=self.axis, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def get_config(self):
        config = {
            'axis': self.axis,
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
        }
        base_config = super(GlobalNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
