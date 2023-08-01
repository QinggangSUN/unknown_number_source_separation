# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 16:11:37 2023

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

References:
Daniel Stoller, Sebastian Ewert, Simon Dixon, Wave-U-Net: A Multi-Scale Neural Network for End-to-End Audio Source
Separation, https://doi.org/10.48550/arXiv.1806.03185
https://github.com/satvik-venkatesh/Wave-U-net-TF2
https://github.com/f90/Wave-U-Net
"""
import logging

import numpy as np
import tensorflow
if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    from tensorflow import keras
else:
    import tensorflow as tf
    import keras
from keras.layers import Layer  # , Concatenate, Conv1D, Input, Lambda, LeakyReLU


class AudioClipLayer(Layer):
    """Restrict outputs to [-1, 1]."""

    def __init__(self, **kwargs):
        """Initializes the instance attributes."""
        super(AudioClipLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """Create the state of the layer (weights)"""
        super(AudioClipLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        """Defines the computation from inputs to outputs"""
        return tf.maximum(tf.minimum(inputs, 1.0), -1.0)

    def compute_output_shape(self, input_shape):
        return input_shape


class InterpolationLayer(Layer):
    """Learned Interpolation layer """

    def __init__(self, padding="valid", **kwargs):
        """Initializes the instance attributes"""
        super(InterpolationLayer, self).__init__(**kwargs)
        self.padding = padding

    def build(self, input_shape):
        """Create the state of the layer (weights)"""
        w_dim = list(input_shape)[3]
        w_init = keras.initializers.RandomNormal(stddev=1.0)
        self.w = self.add_weight(
            'w',  # name
            (w_dim, ),  # shape
            dtype=np.float32,
            initializer=w_init,
            trainable=True
        )
        super(InterpolationLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        """Defines the computation from inputs to outputs"""

        w_scaled = tf.math.sigmoid(self.w)

        counter_w = 1 - w_scaled

        conv_weights = tf.expand_dims(tf.concat([tf.expand_dims(tf.linalg.diag(
            w_scaled), axis=0), tf.expand_dims(tf.linalg.diag(counter_w), axis=0)], axis=0), axis=0)

        intermediate_vals = tf.nn.conv2d(inputs, conv_weights, strides=[1, 1, 1, 1], padding=self.padding.upper())

        intermediate_vals = tf.transpose(intermediate_vals, [2, 0, 1, 3])
        out = tf.transpose(inputs, [2, 0, 1, 3])

        num_entries = out.shape.as_list()[0]
        out = tf.concat([out, intermediate_vals], axis=0)

        indices = list()

        num_outputs = (2*num_entries - 1) if self.padding == "valid" else 2*num_entries

        for idx in range(num_outputs):
            if idx % 2 == 0:
                indices.append(idx // 2)
            else:
                indices.append(num_entries + idx//2)
        out = tf.gather(out, indices)
        current_layer = tf.transpose(out, [1, 2, 0, 3])

        self.output_dim = current_layer.shape
        return current_layer

    def compute_output_shape(self, input_shape):
        output_shape = (self.output_dim.as_list()[0], self.output_dim.as_list()[1],
                        self.output_dim.as_list()[2], self.output_dim.as_list()[3])
        return output_shape

    def get_config(self):
        config = {
            'padding': self.padding,
        }
        base_config = super(InterpolationLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def crop(tensor, target_shape, match_feature_dim=True):
    """Crops a 3D tensor [batch_size, width, channels] along the width axes to a target shape.
    Performs a centre crop. If the dimension difference is uneven, crop last dimensions first.
    Args:
        tensor (keras.Tensor, shape==(batch_size, width, channels)): 3D tensor, that should be cropped.
        target_shape ((int, int, int)): Target shape (3D tensor) that the tensor should be cropped to.
        match_feature_dim (bool, optional): Whether restrict shape of tensor. Defaults to True.
            if True, only width axis (axis 2 including batch) can differ.
    Returns:
        [type]: Cropped tensor.
    """
    shape = np.array(tensor.get_shape().as_list())
    diff = shape[1:] - np.array(target_shape[1:])
    assert(diff[1] == 0 or not match_feature_dim)  # Only width axis can differ
    if diff[0] % 2 != 0:
        logging.warning("WARNING: Cropping with uneven number of extra entries on one side")
    assert diff[0] >= 0  # Only positive difference allowed
    if diff[0] == 0:
        return tensor
    crop_start = diff // 2
    crop_end = diff - crop_start

    return tensor[:, crop_start[1]:-crop_end[1], :]


class CropLayer(Layer):
    """A keras layer call function crop."""

    def __init__(self, match_feature_dim=True, **kwargs):
        """Initializes the instance attributes"""
        super(CropLayer, self).__init__(**kwargs)
        self.match_feature_dim = match_feature_dim

    def build(self, input_shape):
        """Create the state of the layer (weights)"""
        # pass
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `CropLayer` layer should be called on a list of 2 inputs')
        if None in input_shape:
            return
        super(CropLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        """Defines the computation from inputs to outputs"""
        x1, x2 = inputs
        outputs = crop(x1, x2.get_shape().as_list(), self.match_feature_dim)
        self.output_dim = outputs.get_shape().as_list()

        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = (self.output_dim[0], self.output_dim[1], self.output_dim[2])
        return output_shape

    def get_config(self):
        config = {
            'match_feature_dim': self.match_feature_dim,
        }
        base_config = super(CropLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class CropConcatLayer(Layer):
    """Copy-and-crop operation for two feature maps of different size.
    Crops the first input x1 equally along its borders so that its shape is equal to
    the shape of the second input x2, then concatenates them along the feature channel axis.
    """

    def __init__(self, match_feature_dim=True, **kwargs):
        """Initializes the instance attributes"""
        super(CropConcatLayer, self).__init__(**kwargs)
        self.match_feature_dim = match_feature_dim

    def build(self, input_shape):
        """Create the state of the layer (weights)"""
        # # pass
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `CropConcatLayer` layer should be called on a list of 2 inputs')
        super(CropConcatLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        """Defines the computation from inputs to outputs.
          x1, x2 = inputs
          param x1: First input that is cropped and combined with the second input
          param x2: Second input
          return: Combined feature map
        """
        x1, x2 = inputs
        if x2 is None:
            outputs = x1
        else:
            x1 = crop(x1, x2.get_shape().as_list(), self.match_feature_dim)
            outputs = tf.concat([x1, x2], axis=2)

        self.output_dim = outputs.get_shape().as_list()
        return outputs

    def compute_output_shape(self, input_shape):
        output_shape = (self.output_dim[0], self.output_dim[1], self.output_dim[2])
        return output_shape

    def get_config(self):
        config = {
            'match_feature_dim': self.match_feature_dim,
        }
        base_config = super(CropConcatLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
