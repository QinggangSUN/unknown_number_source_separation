# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A model to sequentially separate variable number of sources.

We use a sequential neural network using TDCN++ and beamforming.
We handle a variable number of sources up to a maximum number.

This model works both for single channel and multi-channel
mixture_waveforms input data.
"""

import typing
import attr

import tensorflow
if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    import tensorflow as tf

# Define loss functions.
mse_loss = lambda source, separated: tf.nn.l2_loss(source - separated)


def _stabilized_log_base(x, base=10., stabilizer=1e-8):
    """Stabilized log with specified base."""
    logx = tf.log(x + stabilizer)
    logb = tf.log(tf.constant(base, dtype=logx.dtype))
    return logx / logb


def log_mse_loss(source, separated, max_snr=1e6, bias_ref_signal=None):
    """Negative log MSE loss, the negated log of SNR denominator."""
    err_pow = tf.reduce_sum(tf.square(source - separated), axis=-1)
    snrfactor = 10. ** (-max_snr / 10.)
    if bias_ref_signal is None:
        ref_pow = tf.reduce_sum(tf.square(source), axis=-1)
    else:
        ref_pow = tf.reduce_sum(tf.square(bias_ref_signal), axis=-1)
    bias = snrfactor * ref_pow
    return 10. * _stabilized_log_base(bias + err_pow)


def weights_for_nonzero_refs(source_waveforms, consider_as_zero=None):
    """Return shape (batch, source) weights for signals that are nonzero.

    Args:
      source_waveforms: A tensor (batch, source, samples), dtype=tf.float32.
      consider_as_zero: An optional tensor (batch, source), dtype=tf.bool which
        indicates some entries as being zero even if they are not exactly zero.
        This can be used to indicate some source types (e.g. noise) as being
        zero regardless of their norm.
    Returns:
      consider_nonzero: A tensor (batch, source), dtype=tf.bool of sources
        that will be considered as non-zero based on their norm being greater than
        1e-8 or they are not in consider_as_zero array if given.
    """
    source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
    consider_nonzero = tf.greater(source_norms, 1e-8)
    if consider_as_zero is not None:
        consider_nonzero = tf.logical_and(consider_nonzero,
                                          tf.logical_not(consider_as_zero))
    return consider_nonzero


def weights_for_signal_type(batch, signal_types, chosen_signal_types):
    """Return (batch, source) weights for signal types that are among chosen ones.

    Args:
      batch: Batch size dimension, integer.
      signal_types: A list of signal types, such as ['speech', 'speech', 'noise'].
      chosen_signal_types: A set that is a subset of set(signal_types) which
        includes source types that are going to be chosen in the output tensor.
    Returns:
      choose_source: A (batch, source) tensor of dtype=tf.bool which is 1 for
        sources that are among the chosen_signal_types, and 0 for others.
    """
    num_signals = len(signal_types)
    choose_source = [False] * num_signals
    if chosen_signal_types:
        for i, signal_type in enumerate(signal_types):
            if signal_type in chosen_signal_types:
                choose_source[i] = True
    choose_source = tf.convert_to_tensor(choose_source, dtype=tf.bool)
    choose_source = tf.reshape(choose_source, (1, num_signals))
    choose_source = tf.broadcast_to(choose_source, (batch, num_signals))
    return choose_source
