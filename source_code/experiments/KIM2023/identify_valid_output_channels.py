# -*- coding: utf-8 -*-
"""Compute active channels for predict outputs."""

import tensorflow

if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
else:
    import tensorflow as tf

from experiments.metrics import cosine_similarity_tf
from experiments.KIM2023.compute_parameters import compute_cosine_similarity_pairs_outputs_tf


def compute_active_channels_st(valid_channel_frequency):
    """compute active channels with single target.
    Args:
        valid_channel_frequency (tf.Tensor): valid frequency of each channel.
    Returns:
        active_channels (tf.Tensor): one-hot index of active channel.
    """
    active_index = tf.argmax(valid_channel_frequency)
    active_channels = tf.squeeze(tf.cast(tf.one_hot([active_index, ], valid_channel_frequency.shape[0]), tf.bool))
    return active_channels


# At each stage with k remained output signals, we decide the input contains k speakers if the maximum of cosine
# similarities between the remained signals is less than a threshold ηk−1.
def body_func(similarity_output_pairs, combs, n_active, active_channels, valid_pairs_mask, eta,
              valid_channel_frequency):
    """body function in tf.while_loop.
    Args:
        similarity_output_pairs (tf.Tensor): cosine similarity between output pairs.
        combs (tf.Tensor): combination of index of channels.
        n_active (int): number of active channels.
        active_channels (tf.Tensor): one-hot index of active channel.
        valid_pairs_mask (tf.Tensor): mask for valid output pairs.
        eta (tf.Tensor): (eta_1, ..., eta_k).
        valid_channel_frequency (tf.Tensor): valid frequency of each channel.
    """
    valid_similarity_output_pairs = tf.boolean_mask(similarity_output_pairs, valid_pairs_mask)
    valid_combs = tf.boolean_mask(combs, valid_pairs_mask)

    # If not, one of the output signals that produce the maximum absolute cosine similarity is classified as an
    # invalid output and the cosine similarities related to that output are removed.
    index_pair_choose = tf.argmax(valid_similarity_output_pairs)
    pair_choose = tf.gather(valid_combs, index_pair_choose)
    # Between the two most similar signals, we discard the output channel which was less frequently selected as
    # a valid output for the validation set to select the output channel that is more likely to produce
    # well-separated sources for the unseen samples.
    invalid_channel_index = tf.cond(tf.greater_equal(tf.gather(valid_channel_frequency, pair_choose[0]),
                                                     tf.gather(valid_channel_frequency, pair_choose[1])),
                                    lambda: pair_choose[1],
                                    lambda: pair_choose[0])

    pairs_mask_invalid = []
    for i_c_ in range(combs.shape[0]):
        pair_i = combs[i_c_]
        pairs_mask_invalid.append(tf.cond(tf.reduce_any(tf.equal(invalid_channel_index, pair_i)),
                                          lambda: False, lambda: True))

    valid_pairs_mask = tf.logical_and(valid_pairs_mask, tf.convert_to_tensor(pairs_mask_invalid))
    n_active = n_active - 1
    num_channels_ = active_channels.shape[0]
    invalid_channel_onehot = tf.cast(tf.one_hot(invalid_channel_index, num_channels_), tf.bool)
    active_channels = tf.logical_and(active_channels, tf.logical_not(invalid_channel_onehot))
    return similarity_output_pairs, combs, n_active, active_channels, valid_pairs_mask, eta, valid_channel_frequency


# This procedure is repeated until the maximum similarity is less than the threshold or k becomes 2.
def cond_func(similarity_output_pairs, combs, n_active, active_channels_, valid_pairs_mask, eta,
              valid_channel_frequency):
    """condition function in tf.while_loop.
    Args:
        similarity_output_pairs (tf.Tensor): cosine similarity between output pairs.
        combs (tf.Tensor): combination of index of channels.
        n_active (int): number of active channels.
        active_channels_ (tf.Tensor): one-hot index of active channel.
        valid_pairs_mask (tf.Tensor): mask for valid output pairs.
        eta (tf.Tensor): eta_k.
        valid_channel_frequency (tf.Tensor): valid frequency of each channel.
    Returns:
        (bool): termination condition (false for stop).
    """
    valid_similarity_output_pairs = tf.multiply(similarity_output_pairs, tf.cast(valid_pairs_mask, tf.float32))
    return tf.logical_and(tf.greater(n_active, 2),
                          tf.greater_equal(tf.reduce_max(valid_similarity_output_pairs),
                                           tf.gather(eta, n_active - 2)))


def compute_active_channels_mt(active_channels, predict_outputs, eta, valid_channel_frequency):
    """compute active channels with multiple targets.
    Args:
        active_channels (tf.Tensor): index of active channel.
        predict_outputs (tf.Tensor): predict outputs of the model.
        eta (tf.Tensor): (eta_1, ..., eta_k).
        valid_channel_frequency (tf.Tensor): valid frequency of each channel.
    Returns:
        active_channels (tf.Tensor): index of active channel.
    """
    # Once the input turns out to be a mixture of multiple speeches, we initially assume all output channels are
    # active and evaluate the absolute values of the cosine similarities for all pairs of output signals.
    active_channels = [True, ] * len(active_channels)
    n_active = tf.reduce_sum(tf.cast(active_channels, tf.int32))
    similarity_output_pairs, valid_pairs_mask, combs = compute_cosine_similarity_pairs_outputs_tf(len(active_channels),
                                                                                                  predict_outputs)
    active_channels = tf.convert_to_tensor(active_channels)

    similarity_output_pairs, combs, n_active, active_channels, valid_pairs_mask, eta, valid_channel_frequency = \
        tf.while_loop(cond_func, body_func, (similarity_output_pairs, combs, n_active, active_channels,
                                             valid_pairs_mask, eta, valid_channel_frequency),
                      parallel_iterations=1,
                      maximum_iterations=2
                      )
    return active_channels


def determine_active_channels(input_mixture, predict_outputs, eta, valid_channel_frequency):
    """determine active channels.
    Args:
        input_mixture (tf.Tensor): input mixture.
        predict_outputs (tf.Tensor): predict outputs of the model.
        eta (tf.Tensor): (eta_1, ..., eta_k).
        valid_channel_frequency (tf.Tensor): valid frequency of each channel.
    Returns:
        active_channels (tf.Tensor): index of active channel.
    """
    # First, we determine if only one speaker is active by checking whether the magnitude of the cosine similarity
    # between the input and the output is higher than a threshold η1 for all output signals.
    num_channels = len(valid_channel_frequency)
    active_channels = []
    for i in range(num_channels):
        predict_output_i = predict_outputs[i, :]
        active_channels.append(tf.greater(cosine_similarity_tf(input_mixture, predict_output_i), eta[0]))

    predict_outputs = tf.convert_to_tensor(predict_outputs)
    eta = tf.convert_to_tensor(eta)
    valid_channel_frequency = tf.convert_to_tensor(valid_channel_frequency)

    active_channels = tf.cond(tf.equal(tf.reduce_sum(tf.cast(active_channels, tf.int32)), num_channels),
                              lambda: compute_active_channels_st(valid_channel_frequency),
                              lambda: compute_active_channels_mt(active_channels, predict_outputs, eta,
                                                                 valid_channel_frequency))
    return active_channels
