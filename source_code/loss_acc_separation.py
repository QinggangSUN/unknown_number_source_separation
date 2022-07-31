# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 12:03:06 2018

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=too-many-function-args
import random as python_random
import keras
from keras import backend as K      # pylint: disable=unused-import
import numpy as np
import tensorflow as tf

python_random.seed(123)  # for reproducibility
np.random.seed(1337)  # for reproducibility


def samerate_loss(y_true, y_pred, delta=1e-10):
    """Samerate (cosine similarity) keras loss fuction using Tensorflow."""
    y_true, y_pred = tf.transpose(
        y_true, [0, 3, 2, 1]), tf.transpose(y_pred, [0, 3, 2, 1])

    dn = y_pred.get_shape()[-1].value     # pylint: disable=invalid-name
    y_true = tf.reshape(y_true, [-1, dn])
    y_pred = tf.reshape(y_pred, [-1, dn])

    y_true = tf.where(tf.math.greater_equal(y_true, 0.),
                      tf.clip_by_value(y_true, delta, 1.), y_true)
    y_true = tf.where(tf.math.less(y_true, 0.),
                      tf.clip_by_value(y_true, -1., -delta), y_true)
    y_pred = tf.where(tf.math.greater_equal(y_pred, 0.),
                      tf.clip_by_value(y_pred, delta, 1.), y_pred)
    y_pred = tf.where(tf.math.less(y_pred, 0.),
                      tf.clip_by_value(y_pred, -1., -delta), y_pred)

    ypt = tf.abs(tf.reduce_sum(tf.multiply(y_pred, y_true), axis=-1))
    ypp = tf.reduce_sum(tf.multiply(y_pred, y_pred), axis=-1)
    ytt = tf.reduce_sum(tf.multiply(y_true, y_true), axis=-1)
    denominator = tf.sqrt(tf.multiply(ypp, ytt))
    srs = tf.divide(ypt, denominator)
    sr = 1.0-tf.reduce_mean(tf.stack(srs), axis=-1)     # pylint: disable=invalid-name
    return sr


def samerate_acc(y_true, y_pred, delta=1e-10):
    """Samerate (cosine similarity) keras accuracy fuction using Tensorflow."""
    y_true, y_pred = tf.transpose(
        y_true, [0, 3, 2, 1]), tf.transpose(y_pred, [0, 3, 2, 1])

    dn = y_pred.get_shape()[-1].value     # pylint: disable=invalid-name
    y_true = tf.reshape(y_true, [-1, dn])
    y_pred = tf.reshape(y_pred, [-1, dn])

    y_true = tf.where(tf.math.greater_equal(y_true, 0.),
                      tf.clip_by_value(y_true, delta, 1.), y_true)
    y_true = tf.where(tf.math.less(y_true, 0.),
                      tf.clip_by_value(y_true, -1., -delta), y_true)
    y_pred = tf.where(tf.math.greater_equal(y_pred, 0.),
                      tf.clip_by_value(y_pred, delta, 1.), y_pred)
    y_pred = tf.where(tf.math.less(y_pred, 0.),
                      tf.clip_by_value(y_pred, -1., -delta), y_pred)

    ypt = tf.abs(tf.reduce_sum(tf.multiply(y_pred, y_true), axis=-1))
    ypp = tf.reduce_sum(tf.multiply(y_pred, y_pred), axis=-1)
    ytt = tf.reduce_sum(tf.multiply(y_true, y_true), axis=-1)
    denominator = tf.sqrt(tf.multiply(ypp, ytt))
    srs = tf.divide(ypt, denominator)
    sr = tf.reduce_mean(tf.stack(srs), axis=-1)     # pylint: disable=invalid-name
    return sr


def samerate_acc_d2_clip(y_true, y_pred, delta=1e-10):
    """Samerate (cosine similarity) keras accuracy fuction using Tensorflow."""

    dn = y_pred.get_shape()[-1].value     # pylint: disable=invalid-name
    y_true = tf.reshape(y_true, [-1, dn])
    y_pred = tf.reshape(y_pred, [-1, dn])

    y_true = tf.where(tf.math.greater_equal(y_true, 0.),
                      tf.clip_by_value(y_true, delta, 1.), y_true)
    y_true = tf.where(tf.math.less(y_true, 0.),
                      tf.clip_by_value(y_true, -1., -delta), y_true)
    y_pred = tf.where(tf.math.greater_equal(y_pred, 0.),
                      tf.clip_by_value(y_pred, delta, 1.), y_pred)
    y_pred = tf.where(tf.math.less(y_pred, 0.),
                      tf.clip_by_value(y_pred, -1., -delta), y_pred)

    ypt = tf.abs(tf.reduce_sum(tf.multiply(y_pred, y_true), axis=-1))
    ypp = tf.reduce_sum(tf.multiply(y_pred, y_pred), axis=-1)
    ytt = tf.reduce_sum(tf.multiply(y_true, y_true), axis=-1)
    denominator = tf.sqrt(tf.multiply(ypp, ytt))
    srs = tf.divide(ypt, denominator)
    sr = tf.reduce_mean(tf.stack(srs), axis=-1)     # pylint: disable=invalid-name
    return sr


def samerate_acc_d2(y_true, y_pred, delta=1e-8, n_output=4):
    """Cosine similarity same rate including denominator stabilizer using Tensorflow.
        Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, et al. What’s all the Fuss about Free Universal Sound
        Separation Data?. ICASSP 2021. https://doi.org/10.1109/ICASSP39728.2021.9414774."""
    dn = y_pred.get_shape()[-1].value     # pylint: disable=invalid-name
    y_true = tf.reshape(y_true, [-1, dn])
    y_pred = tf.reshape(y_pred, [-1, dn])

    axis = 0 if dn == n_output else -1  # output channel is last channel
    ypt = tf.abs(tf.reduce_sum(tf.multiply(y_pred, y_true), axis=axis))
    ypp = tf.reduce_sum(tf.multiply(y_pred, y_pred), axis=axis)
    ytt = tf.reduce_sum(tf.multiply(y_true, y_true), axis=axis)

    denominator = tf.sqrt(tf.multiply(ypp, ytt))+delta
    srs = tf.divide(ypt, denominator)
    sr = tf.reduce_mean(tf.stack(srs), axis=axis)     # pylint: disable=invalid-name
    return sr


def samerate_loss2(y_true, y_pred):
    """Samerate (cosine similarity) keras loss fuction using Tensorflow."""
    y_true, y_pred = tf.transpose(
        y_true, [0, 3, 2, 1]), tf.transpose(y_pred, [0, 3, 2, 1])

    dn = y_pred.get_shape()[-1].value     # pylint: disable=invalid-name
    y_true = tf.reshape(y_true, [-1, dn])
    y_pred = tf.reshape(y_pred, [-1, dn])

    ypt = tf.abs(tf.reduce_sum(tf.multiply(y_pred, y_true), axis=-1))
    ypp = tf.reduce_sum(tf.multiply(y_pred, y_pred), axis=-1)
    ytt = tf.reduce_sum(tf.multiply(y_true, y_true), axis=-1)
    denominator = tf.clip_by_value(tf.sqrt(tf.multiply(ypp, ytt)), 1e-30, 1e30)
    srs = tf.clip_by_value(tf.divide(ypt, denominator), 1e-30, 1.0)
    sr = 1.0-tf.reduce_mean(tf.stack(srs), axis=-1)     # pylint: disable=invalid-name
    return sr


def samerate_acc2(y_true, y_pred):
    """Samerate (cosine similarity) keras accuracy fuction using Tensorflow."""
    y_true, y_pred = tf.transpose(
        y_true, [0, 3, 2, 1]), tf.transpose(y_pred, [0, 3, 2, 1])

    dn = y_pred.get_shape()[-1].value     # pylint: disable=invalid-name
    y_true = tf.reshape(y_true, [-1, dn])
    y_pred = tf.reshape(y_pred, [-1, dn])

    ypt = tf.abs(tf.reduce_sum(tf.multiply(y_pred, y_true), axis=-1))
    ypp = tf.reduce_sum(tf.multiply(y_pred, y_pred), axis=-1)
    ytt = tf.reduce_sum(tf.multiply(y_true, y_true), axis=-1)
    denominator = tf.clip_by_value(tf.sqrt(tf.multiply(ypp, ytt)), 1e-30, 1e30)
    srs = tf.clip_by_value(tf.divide(ypt, denominator), 1e-30, 1.0)
    sr = tf.reduce_mean(tf.stack(srs), axis=-1)     # pylint: disable=invalid-name
    return sr


def vae_loss(z_mean=0., z_log_var=1.):
    """VAE loss, xent_loss(binary_crossentropy) + kl_loss, using keras."""
    def loss(y_true, y_pred):
        xent_loss = keras.metrics.binary_crossentropy(
            K.flatten(y_true), K.flatten(y_pred))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
        vae_loss_value = K.mean(xent_loss + kl_loss)
        return vae_loss_value
    return loss


def vae_loss_2(z_mean=0., z_log_var=1.):
    """VAE loss, xent_loss(binary_crossentropy) + kl_loss, using keras."""
    def loss(y_true, y_pred):
        xent_loss = keras.metrics.binary_crossentropy(
            K.flatten(y_true), K.flatten(y_pred))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss_value = K.mean(xent_loss + kl_loss)
        return vae_loss_value
    return loss
