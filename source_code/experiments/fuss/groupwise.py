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
"""Apply per-source-type loss functions."""

import typing
import tensorflow
if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    import tensorflow as tf

from experiments.fuss.permutation_invariant import wrap as pit_wrap


def apply(loss_fns: typing.Dict[str, typing.Callable[..., typing.Any]],
          signal_names: typing.List[str],
          reference,
          estimate,
          permutation_invariant_losses: typing.List[str],
          batch_size, num_sources, frame_length):
    """Apply loss functions to the corresponding references and estimates.

    For each kind of signal, gather corresponding references and estimates, and
    apply the loss function.  Scatter-add the results into the loss.

    For elements of signals_names not in loss_fns, no loss will be applied.

    Args:
      loss_fns: dictionary of string -> loss_fn.
        Each string is a name to match elements of signal_names.
        Each loss_fn has the following signature:
        Args
          reference [batch, n_source, frame_length] tensor
          estimate [batch, n_source, frame_length] tensor
          batch [int] batch size
        Returns
          A [batch, grouped_source] tensor of dtype=tf.float32
      signal_names: list of names of each signal.
      reference: [batch, n_source, frame_length] tensor.
      estimate: [batch, n_source, frame_length] tensor.
      permutation_invariant_losses: List of losses to be permutation invariant.
      batch_size: batch size.
      num_sources: maximum number of the sources.
      frame_length: frame length of input.
    Returns:
      loss, A [batch, source] tensor of dtype=tf.float32
    """

    # batch = tf.shape(reference)[0]
    # n_source = tf.shape(reference)[1]
    # frame_length = tf.shape(reference)[2]
    batch = batch_size
    n_source = num_sources
    frame_length = frame_length
    reference = tf.reshape(reference, (batch, n_source, frame_length))
    estimate = tf.reshape(estimate, (batch, n_source, frame_length))

    # if tf.shape(reference)[:2] != tf.shape(estimate)[:2]: # error convert to tf.shape
    #     raise ValueError('First two axes (batch, source) of reference and estimate'
    #                      'must be equal, got {}, {}'.format(
    #         tf.shape(reference)[:2], tf.shape(estimate)[:2]))

    loss = tf.zeros(shape=reference.shape[:2], dtype=tf.float32)
    permuted_estimates = tf.zeros_like(reference)

    # For each kind of signal, e.g. 'speech', 'noise', gather subsets of reference
    # and estimate, apply loss function and scatter-add into the loss tensor.
    for name, loss_fn in loss_fns.items():
        idxs = [idx for idx, value in enumerate(signal_names) if value == name]
        idxs_0 = tf.tile(
            tf.expand_dims(tf.range(batch), 1),
            [1, len(idxs)])
        idxs_1 = tf.tile(
            tf.expand_dims(tf.constant(idxs, dtype=tf.int32), 0),
            [batch, 1])
        idxs_nd = tf.stack([idxs_0, idxs_1], axis=2)
        reference_key = tf.gather_nd(reference, idxs_nd)
        estimate_key = tf.gather_nd(estimate, idxs_nd)

        loss_fn = pit_wrap(
            loss_fn,
            enable=name in permutation_invariant_losses)
        loss_key, permuted_estimates_key = loss_fn(reference_key, estimate_key)

        loss = tf.tensor_scatter_add(loss, idxs_nd, loss_key)
        permuted_estimates = tf.tensor_scatter_add(
            permuted_estimates, idxs_nd, permuted_estimates_key)

    return loss, permuted_estimates
