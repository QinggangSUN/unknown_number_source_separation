# -*- coding: utf-8 -*-

import tensorflow

if tensorflow.__version__ >= '2.0':
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
else:
    import tensorflow as tf

from experiments.fuss.permutation_invariant import wrap as pit_wrap
from experiments.fuss.metrics import signal_to_noise_ratio_gain_invariant


def si_sdr_stabilizer_tf(s_sep, s_ref, epsilon=1e-8):
    """scale-invariant signal-to-noise ratio.
       [conv-tasnet](https://github.com/helianvine/conv-tasnet?tab=readme-ov-file#conv-tasnet)
       [Convolutional time domain audio separation network](https://arxiv.org/abs/1809.07454)
    Args:
        s_sep (tf.Tensor): degradation signal with shape [batch_size, speakers, length]
        s_ref (tf.Tensor): reference signal with shape [batch_size, speakers, length]
        epsilon (float): : epsilon stabilizer.
    """
    s_dot = tf.reduce_sum(s_ref * s_sep, axis=-1, keep_dims=True)  # [batch_size, C(speakers), 1]
    p_ref = tf.reduce_sum(s_ref ** 2, axis=-1, keep_dims=True)  # [batch_size, C(speakers), 1]
    s_target = s_dot * s_ref / (p_ref + epsilon)  # [batch_size, C(speakers), length]
    e_noise = s_sep - s_target  # [batch_size, C(speakers), length]
    s_target_norm = tf.reduce_sum(s_target ** 2, axis=-1)  # [batch_size, C(speakers)]
    e_noise_norm = tf.reduce_sum(e_noise ** 2, axis=-1)  # [batch_size, C(speakers)]
    si_sdr_loss = 10 * tf.log(s_target_norm / (e_noise_norm + epsilon))  # [batch_size, C(speakers)]
    si_sdr_loss = tf.reduce_mean(si_sdr_loss) / tf.log(10.0)
    return si_sdr_loss


def neg_si_sdr_stabilizer_loss_batch_tf(s_sep, s_ref, epsilon=1e-8):
    """scale-invariant signal-to-noise ratio.
       [conv-tasnet](https://github.com/helianvine/conv-tasnet?tab=readme-ov-file#conv-tasnet)
       [Convolutional time domain audio separation network](https://arxiv.org/abs/1809.07454)
    Args:
        s_sep (tf.Tensor): degradation signal with shape [batch_size, speakers, length]
        s_ref (tf.Tensor): reference signal with shape [batch_size, speakers, length]
        epsilon (float): : epsilon stabilizer.
    """
    s_dot = tf.reduce_sum(s_ref * s_sep, axis=-1, keep_dims=True)  # [batch_size, C(speakers), 1]
    p_ref = tf.reduce_sum(s_ref ** 2, axis=-1, keep_dims=True)  # [batch_size, C(speakers), 1]
    s_target = s_dot * s_ref / (p_ref + epsilon)  # [batch_size, C(speakers), length]
    e_noise = s_sep - s_target  # [batch_size, C(speakers), length]
    s_target_norm = tf.reduce_sum(s_target ** 2, axis=-1)  # [batch_size, C(speakers)]
    e_noise_norm = tf.reduce_sum(e_noise ** 2, axis=-1)  # [batch_size, C(speakers)]
    si_sdr_loss = - 10 * tf.log((s_target_norm + epsilon) / (e_noise_norm + epsilon))  # [batch_size, C(speakers)]
    return si_sdr_loss


def mse_loss_batch_tf(x, y):
    """MSE using TensorFlow.
    Args:
        x (tf.Tensor): x (batch, n_source, frame_length).
        y (tf.Tensor): y (batch, n_source, frame_length).
    Returns:
        mse (tf.Tensor): MSE (batch, n_source, ).
    """
    batch = x.shape[0]
    n_source = x.shape[1]
    mse = [tf.nn.l2_loss(x[i] - y[i]) for i in range(batch)]
    mse = tf.reshape(mse, (batch, n_source))
    return mse


def weights_for_nonzero_refs(source_waveforms):
    """Compute weights for signals that are nonzero.
    Args:
        source_waveforms (tf.Tensor): (n_channel, frame_length) source waveforms.
    Returns:
        (tf.Tensor): (source,) weights for signals that are nonzero.
    """
    source_norms = tf.sqrt(tf.reduce_mean(tf.square(source_waveforms), axis=-1))
    return tf.greater(source_norms, 1e-8)


def weights_for_active_seps(power_sources, power_separated):
    """Compute weights for active separated signals.
    Args:
        power_sources (tf.Tensor): (n_channel, frame_length) source waveforms.
        power_separated (tf.Tensor): (n_channel, frame_length) source waveforms.
    Returns:
        (tf.Tensor): (source,) weights for signals that are nonzero.
    """
    min_power = tf.reduce_min(power_sources, axis=-1, keepdims=True)
    return tf.greater(power_separated, 0.01 * min_power)


def cosine_similarity_tf(y_true, y_pred):
    """Cosine similarity using TensorFlow.
    Args:
        y_true (tf.Tensor): x (frame_length, ).
        y_pred (tf.Tensor): y (frame_length, ).
    Returns:
        result (tf.Tensor): cosine similarity between y_true and y_pred.
    """
    ypt = tf.abs(tf.reduce_sum(tf.multiply(y_pred, y_true), axis=-1))
    ypp = tf.reduce_sum(tf.multiply(y_pred, y_pred), axis=-1)
    ytt = tf.reduce_sum(tf.multiply(y_true, y_true), axis=-1)
    denominator = tf.sqrt(tf.multiply(ypp, ytt))
    result = tf.divide(ypt, denominator)
    return result


def cosine_similarity_stabilizer_tf(y_true, y_pred, epsi=1e-8):
    """Cosine similarity with stabilizer using TensorFlow. epsilon is same as in FUSS.
    Args:
        y_true (tf.Tensor): target tensor.
        y_pred (tf.Tensor): predict tensor.
        epsi (tf.Tensor): epsilon stabilizer.
    Returns:
        cosine_similarity (tf.Tensor): cosine similarity between y_true and y_pred.
    """
    p_t = tf.abs(tf.reduce_sum(tf.multiply(y_pred, y_true), axis=-1))
    p_p = tf.reduce_sum(tf.multiply(y_pred, y_pred), axis=-1)
    t_t = tf.reduce_sum(tf.multiply(y_true, y_true), axis=-1)

    numerator = tf.abs(p_t)
    denominator = tf.sqrt(tf.multiply(p_p, t_t)) + epsi
    cosine_similarity = numerator / denominator
    return cosine_similarity


def neg_cosine_similarity_stabilizer_tf(y_true, y_pred, epsi=1e-8):
    """negative cosine similarity with stabilizer using TensorFlow. epsilon is same as in FUSS.
    Args:
        y_true (tf.Tensor): target tensor (batch, 1, frame_length).
        y_pred (tf.Tensor): predict tensor (batch, 1, frame_length).
        epsi (tf.Tensor): epsilon stabilizer.
    Returns:
        (tf.Tensor): negative cosine similarity between y_true and y_pred.
    """
    return - cosine_similarity_stabilizer_tf(y_true, y_pred, epsi)


def log10_tf(x):
    """log10(x) using TensorFlow.
        https://github.com/tensorflow/tensorflow/issues/1666#issuecomment-202453841
    Args:
        x (dtype): x
    Returns:
        (dtype): log10(x)
    """
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def log10_stabilizer_tf(x, epsilon=1e-8):
    """log10(x) using TensorFlow.
        https://github.com/tensorflow/tensorflow/issues/1666#issuecomment-202453841
    Args:
        x (dtype): x
        epsilon (float): epsilon stabilizer.
    Returns:
        (dtype): log10(x)
    """
    numerator = tf.log(tf.clip_by_value(x, epsilon, 1.))
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator


def si_snr_tf(x, y):
    """SI-SNR in FUSS using TensorFlow.
    Args:
        x (tf.Tensor): x (frame_length, ).
        y (tf.Tensor): y (frame_length, ).
    Returns:
        si_snr (tf.Tensor): SI-SNR.
    """
    rho = cosine_similarity_tf(x, y)
    si_snr = 10 * log10_tf(tf.square(rho) / (1. - tf.square(rho)))
    return si_snr


def si_snr_loss_tf(x, y):
    """SI-SNR in FUSS using TensorFlow.
    Args:
        x (tf.Tensor): x (n_source, frame_length).
        y (tf.Tensor): y (n_source, frame_length).
    Returns:
        si_snr (tf.Tensor): SI-SNR (n_source, ).
    """
    n_source = x.shape[0]
    si_snr = [si_snr_tf(x[i], y[i]) for i in range(n_source)]
    si_snr = tf.reshape(si_snr, (n_source,))
    return si_snr


def neg_si_snr_loss_tf(x, y):
    """negative SI-SNR loss in FUSS using TensorFlow.
    Args:
        x (tf.Tensor): x (n_source, frame_length).
        y (tf.Tensor): y (n_source, frame_length).
    Returns:
        si_snr (tf.Tensor): SI-SNR (n_source, ).
    """
    n_source = x.shape[0]
    si_snr = [-si_snr_tf(x[i], y[i]) for i in range(n_source)]
    si_snr = tf.reshape(si_snr, (n_source,))
    return si_snr


def si_snr_stabilizer_tf(x, y, epsi=1e-8):
    """SI-SNR in FUSS with stabilizer using TensorFlow.
    Args:
        x (tf.Tensor): x (frame_length, ).
        y (tf.Tensor): y (frame_length, ).
        epsi (tf.Tensor): epsilon stabilizer.
    Returns:
        si_snr (tf.Tensor): SI-SNR.
    """
    rho = cosine_similarity_stabilizer_tf(x, y)
    si_snr = 10 * log10_tf((tf.square(rho) + epsi) / (1. - tf.square(rho) + epsi))
    return si_snr


def si_snr_stabilizer_loss_tf(x, y, epsi=1e-8):
    """SI-SNR in FUSS with stabilizer using TensorFlow.
    Args:
        x (tf.Tensor): x (n_source, frame_length).
        y (tf.Tensor): y (n_source, frame_length).
        epsi (tf.Tensor): epsilon stabilizer.
    Returns:
        si_snr (tf.Tensor): SI-SNR (n_source, ).
    """
    n_source = x.shape[0]
    si_snr = [si_snr_stabilizer_tf(x[i], y[i], epsi) for i in range(n_source)]
    si_snr = tf.reshape(si_snr, (n_source,))
    return si_snr


def neg_si_snr_stabilizer_loss_tf(x, y, epsi=1e-8):
    """negative SI-SNR in FUSS with stabilizer using TensorFlow.
    Args:
        x (tf.Tensor): x (n_source, frame_length).
        y (tf.Tensor): y (n_source, frame_length).
        epsi (tf.Tensor): epsilon stabilizer.
    Returns:
        si_snr (tf.Tensor): SI-SNR (n_source, ).
    """
    n_source = x.shape[0]
    si_snr = [-si_snr_stabilizer_tf(x[i], y[i], epsi) for i in range(n_source)]
    si_snr = tf.reshape(si_snr, (n_source,))
    return si_snr


def neg_si_snr_stabilizer_loss_batch_tf(x, y, epsi=1e-8):
    """negative SI-SNR in FUSS with stabilizer using TensorFlow.
        When x or y is zero, return -80 dB.
    Args:
        x (tf.Tensor): x (batch, n_source, frame_length).
        y (tf.Tensor): y (batch, n_source, frame_length).
        epsi (tf.Tensor): epsilon stabilizer.
    Returns:
        si_snr (tf.Tensor): SI-SNR (batch, n_source, ).
    """
    batch = x.shape[0]
    n_source = x.shape[1]
    si_snr = [neg_si_snr_stabilizer_loss_tf(x[i], y[i], epsi) for i in range(batch)]
    si_snr = tf.reshape(si_snr, (batch, n_source))
    return si_snr


def neg_si_snr_stabilizer_penalty_zero_tf(x, y, epsi=1e-8, eta=-300.):
    """SI-SNR in FUSS with penalty to zero using TensorFlow.
    Args:
        x (tf.Tensor): x (n_source, frame_length).
        y (tf.Tensor): y (n_source, frame_length).
        epsi (tf.Tensor): epsilon stabilizer.
        eta (tf.Tensor): loss for zero. When x or y is zero tensor, return eta as loss, otherwise, the loss will be 0.
    Returns:
        si_snr (tf.Tensor): SI-SNR (n_source, ).
    """
    si_snr = tf.cond(tf.logical_and(tf.greater(tf.reduce_sum(tf.square(x)), 0.),
                                    tf.greater(tf.reduce_sum(tf.square(y)), 0.), ),
                     lambda: -si_snr_stabilizer_tf(x, y, epsi),
                     lambda: -eta)
    return si_snr


def neg_si_snr_stabilizer_penalty_zero_loss_tf(x, y, epsi=1e-8, eta=-300.):
    """SI-SNR in FUSS with penalty to zero using TensorFlow.
    Args:
        x (tf.Tensor): x (n_source, frame_length).
        y (tf.Tensor): y (n_source, frame_length).
        epsi (tf.Tensor): epsilon stabilizer.
        eta (tf.Tensor): loss for zero. When x or y is zero tensor, return eta as loss, otherwise, the loss will be 0.
    Returns:
        si_snr (tf.Tensor): SI-SNR (n_source, ).
    """
    n_source = x.shape[0]
    si_snr = [neg_si_snr_stabilizer_penalty_zero_tf(x[i], y[i], epsi, eta) for i in range(n_source)]
    si_snr = tf.reshape(si_snr, (n_source,))
    return si_snr


def neg_si_snr_stabilizer_penalty_zero_loss_batch_tf(x, y, epsi=1e-8, eta=-300.):
    """SI-SNR in FUSS with penalty to zero using TensorFlow.
    Args:
        x (tf.Tensor): x (batch, n_source, frame_length).
        y (tf.Tensor): y (batch, n_source, frame_length).
        epsi (tf.Tensor): epsilon stabilizer.
        eta (tf.Tensor): loss for zero. When x or y is zero tensor, return eta as loss, otherwise, the loss will be 0.
    Returns:
        si_snr (tf.Tensor): SI-SNR (batch, n_source, ).
    """
    batch = x.shape[0]
    n_source = x.shape[1]
    si_snr = [neg_si_snr_stabilizer_penalty_zero_loss_tf(x[i], y[i], epsi, eta) for i in range(batch)]
    si_snr = tf.reshape(si_snr, (batch, n_source))
    return si_snr


def si_snr_pit_loss_matrix_tf(x, y):
    """SI-SNR loss matrix in FUSS with permutation invariant training using TensorFlow.
    Args:
        x (tf.Tensor): x (batch, n_source, frame_length).
        y (tf.Tensor): y (batch, n_source, frame_length).
    Returns:
        matrix_tensor (tf.Tensor): SI-SNR matrix (batch, n_source, n_source).
    """

    def loss_matrix(x_, y_):
        """SI-SNR loss matrix in FUSS with permutation invariant training using TensorFlow.
        Args:
            x_ (tf.Tensor): x (n_source, frame_length).
            y_ (tf.Tensor): y (n_source, frame_length).
        Returns:
            matrix_tensor (tf.Tensor): SI-SNR matrix (n_source, n_source).
        """
        n_source_ = x_.shape[0]
        matrix_ = []
        for i in range(n_source_):
            matrix_i = []
            for j in range(n_source_):
                matrix_i.append(si_snr_tf(x_[i], y_[j]))
            matrix_.append(matrix_i)
        matrix_tensor_ = tf.reshape(matrix_, [n_source_, n_source_])
        return matrix_tensor_

    batch = x.shape[0]
    n_source = x.shape[1]
    matrix = []
    for k in range(batch):
        matrix.append(loss_matrix(x[k], y[k]))
    matrix_tensor = tf.reshape(matrix, [n_source, n_source])

    return matrix_tensor


def compute_oracle_avg_si_snr(si_snr_channels, weights_active_refs, weights_active_seps, reward=20.0):
    """compute average oracle SI-SNR. 
    Args:
        si_snr_channels (tf.Tensor): si-snr from all matched targets without penalty.
        weights_active_refs (tf.Tensor): (n_channels,) (bool) active channels of references.
        weights_active_seps (tf.Tensor): (n_channels,) (bool) active channels of predict.
        reward (tf.Tensor): (1,) (float) constant reward value for false negative when oracle channels is empty.
    Returns:
        avg_si_snr (tf.Tensor): average oracle SI-SNR. When there is no oracle channel, penalty could be any positive
            value, because the first item will be zero.
    """
    weights_active_oracle = tf.logical_and(weights_active_refs, weights_active_seps)
    avg_si_snr = tf.cond(tf.greater(tf.reduce_sum(tf.cast(weights_active_oracle, tf.int32)), 0),
                         lambda: tf.reduce_mean(tf.boolean_mask(si_snr_channels, weights_active_oracle)),
                         lambda: reward)
    return avg_si_snr


class PenalizedSISNR(object):
    """Implementation of P-Si-SNR, as purposed in [1]. Use negative of the average SI-SNR from oracle source counting as
     the penalty term for miscounting.
    References:
        [1] Junzhe Zhu, Raymond A. Yeh, Mark Hasegawa-Johnson. "Multi-Decoder Dprnn: Source Separation for Variable
            Number of Speakers, in: 2021 IEEE International Conference on Acoustics, Speech and Signal Processing
             (ICASSP). IEEE, Toronto, ON, Canada: IEEE, Jun. 2021, pp. 3420–3424.
             https://doi.org/10.1109/ICASSP39728.2021.9414205
    """

    def __init__(self, penalty=None, reward=0.):
        super().__init__()
        self.penalty = penalty  # a constant value measuring from data
        self.reward = reward  # reward for false negative when there is no true positive
        self.result_dict = None

    def compute_fuss_permutation(self, source_waveforms, separated_waveforms):
        """Compute output align permutation for FUSS.
        Args:
            source_waveforms (tf.Tensor): (n_channel, frame_length) source waveforms.
            separated_waveforms (tf.Tensor): (n_channel, frame_length) separated waveforms without align.
        Returns:
            separated_waveforms (tf.Tensor): (n_channel, frame_length) separated waveforms after align.
        """
        # Align separated sources to reference sources.
        perm_inv_loss = pit_wrap(lambda tar, est: -signal_to_noise_ratio_gain_invariant(est, tar))
        _, separated_waveforms = perm_inv_loss(source_waveforms[tf.newaxis], separated_waveforms[tf.newaxis])
        separated_waveforms = separated_waveforms[0]  # Remove batch axis.
        return separated_waveforms

    def compute_fuss_num_active(self, separated_waveforms, source_waveforms):
        """compute number of active outputs for FUSS.
        Args:
            separated_waveforms (tf.Tensor): (n_channel, frame_length) separated waveforms after align.
            source_waveforms (tf.Tensor): (n_channel, frame_length) source waveforms.
        Returns:
            (dict): dictionary for results.
        """
        # Compute separated and source powers.
        power_separated = tf.reduce_mean(separated_waveforms ** 2, axis=-1)
        power_sources = tf.reduce_mean(source_waveforms ** 2, axis=-1)

        # Compute weights for active (separated, source) pairs where source is nonzero
        # and separated power is above threshold of the quietest source power - 20 dB.
        weights_active_refs = weights_for_nonzero_refs(source_waveforms)
        weights_active_seps = weights_for_active_seps(
            tf.boolean_mask(power_sources, weights_active_refs), power_separated)
        weights_active_pairs = tf.logical_and(weights_active_refs,
                                              weights_active_seps)

        # Compute SI-SNR.
        # sisnr_separated = signal_to_noise_ratio_gain_invariant(separated_waveforms, source_waveforms)
        sisnr_separated = si_snr_loss_tf(separated_waveforms, source_waveforms)
        # sisnr_separated = si_snr_stabilizer_loss_tf(separated_waveforms, source_waveforms)
        num_active_refs = tf.reduce_sum(tf.cast(weights_active_refs, tf.int32))
        num_active_seps = tf.reduce_sum(tf.cast(weights_active_seps, tf.int32))
        num_active_pairs = tf.reduce_sum(tf.cast(weights_active_pairs, tf.int32))

        # Compute under/equal/over separation.
        under_separation = tf.cast(tf.less(num_active_seps, num_active_refs), tf.int32)
        equal_separation = tf.cast(tf.equal(num_active_seps, num_active_refs), tf.int32)
        over_separation = tf.cast(tf.greater(num_active_seps, num_active_refs), tf.int32)

        self.result_dict = {'sisnr_separated': sisnr_separated,
                            'power_separated': power_separated,
                            'power_sources': power_sources,
                            'under_separation': under_separation,
                            'equal_separation': equal_separation,
                            'over_separation': over_separation,
                            'weights_active_refs': weights_active_refs,
                            'weights_active_seps': weights_active_seps,
                            'weights_active_pairs': weights_active_pairs,
                            'num_active_refs': num_active_refs,
                            'num_active_seps': num_active_seps,
                            'num_active_pairs': num_active_pairs}
        return self.result_dict

    def compute_p_si_snr_original(self, si_snr_channels, weights_active_refs, weights_active_seps, gt_nsrc, est_nsrc):
        """ Compute p-si-snr.
        Args:
            si_snr_channels (tf.Tensor): si-snr from all matched targets without penalty.
            weights_active_refs (tf.Tensor): (n_channels,) (bool) active channels of references.
            weights_active_seps (tf.Tensor): (n_channels,) (bool) active channels of predict.
            gt_nsrc (tf.Tensor): (int) ground truth number of targets.
            est_nsrc (tf.Tensor): (int) estimated number of targets.
        Returns:
            p_si_snr (float): p-si-snr.
        """
        weights_active_oracle = tf.logical_and(weights_active_refs, weights_active_seps)
        avg_si_snr = tf.reduce_mean(tf.boolean_mask(weights_active_oracle, si_snr_channels))
        penalty = self.penalty
        if penalty is not None:
            neg_penalty = -penalty
        else:
            neg_penalty = -avg_si_snr
        p_si_snr = (avg_si_snr * tf.cast(tf.minimum(est_nsrc, gt_nsrc), tf.float32) +
                    neg_penalty * tf.cast(tf.abs(est_nsrc - gt_nsrc), tf.float32)
                    ) / tf.cast(tf.maximum(est_nsrc, gt_nsrc), tf.float32)
        return p_si_snr

    def compute_p_si_snr_beta(self, si_snr_channels, weights_active_refs, weights_active_seps):
        """ Compute p-si-snr.
        Args:
            si_snr_channels (tf.Tensor): si-snr from all matched targets without penalty.
            weights_active_refs (tf.Tensor): (n_channels,) (bool) active channels of references.
            weights_active_seps (tf.Tensor): (n_channels,) (bool) active channels of predict.
        Returns:
            p_si_snr (float): p-si-snr.
        """
        avg_si_snr = compute_oracle_avg_si_snr(si_snr_channels, weights_active_refs, weights_active_seps)
        num_active_refs = tf.reduce_sum(tf.cast(weights_active_refs, tf.int32))
        num_active_seps = tf.reduce_sum(tf.cast(weights_active_seps, tf.int32))

        penalty = self.penalty
        if penalty is not None:  # use a constant value Pref
            neg_penalty = -penalty
        else:  # use average si_snr as Pref
            neg_penalty = -avg_si_snr

        def p_si_snr_positive(avg_si_snr_, neg_penalty_, num_active_refs_, num_active_seps_):
            p_si_snr_ = (avg_si_snr_ * tf.cast(tf.minimum(num_active_seps_, num_active_refs_), tf.float32) +
                         neg_penalty_ * tf.cast(tf.abs(num_active_seps_ - num_active_refs_), tf.float32)
                         ) / tf.cast(tf.maximum(num_active_seps_, num_active_refs_), tf.float32)
            return p_si_snr_

        def p_si_snr_negative(avg_si_snr_, num_active_refs_, num_active_seps_):
            p_si_snr_ = avg_si_snr_ * tf.cast(tf.maximum(num_active_refs_, num_active_seps_), tf.float32
                                              ) / tf.cast(tf.minimum(num_active_seps_, num_active_refs_), tf.float32)
            return p_si_snr_

        p_si_snr = tf.cond(tf.greater_equal(avg_si_snr, 0.),
                           lambda: p_si_snr_positive(avg_si_snr, neg_penalty, num_active_refs, num_active_seps),
                           lambda: p_si_snr_negative(avg_si_snr, num_active_refs, num_active_seps))

        return p_si_snr

    def compute_p_si_snr(self, si_snr_channels, weights_active_refs, weights_active_seps):
        """ Compute p-si-snr.
        Args:
            si_snr_channels (tf.Tensor): si-snr from all matched targets without penalty.
            weights_active_refs (tf.Tensor): (n_channels,) (bool) active channels of references.
            weights_active_seps (tf.Tensor): (n_channels,) (bool) active channels of predict.
        Returns:
            p_si_snr (float): p-si-snr.
        """
        result_dict = self.result_dict
        reward = self.reward
        avg_oracle_si_snr = compute_oracle_avg_si_snr(si_snr_channels, weights_active_refs, weights_active_seps, reward)
        result_dict.update({'avg_oracle_si_snr': avg_oracle_si_snr})
        weights_active_tp = tf.logical_and(weights_active_refs, weights_active_seps)
        weights_active_tn = tf.logical_and(weights_active_refs, tf.logical_not(weights_active_seps))
        weights_active_fp = tf.logical_and(tf.logical_not(weights_active_refs), weights_active_seps)
        weights_active_fn = tf.logical_and(tf.logical_not(weights_active_refs), tf.logical_not(weights_active_seps))
        num_tp = tf.reduce_sum(tf.cast(weights_active_tp, tf.int32))
        num_tn = tf.reduce_sum(tf.cast(weights_active_tn, tf.int32))
        num_fp = tf.reduce_sum(tf.cast(weights_active_fp, tf.int32))
        num_fn = tf.reduce_sum(tf.cast(weights_active_fn, tf.int32))
        num_sum = tf.reduce_sum([num_tp, num_tn, num_fp, num_fn])

        penalty = self.penalty
        if penalty is not None:  # use a constant value Pref
            neg_penalty = -penalty
        else:  # use average si_snr as Pref
            neg_penalty = -avg_oracle_si_snr

        def p_si_snr_positive(avg_si_snr_, neg_penalty_, num_tp_, num_fn_, num_tn_, num_fp_):
            p_si_snr_ = (avg_si_snr_ * tf.cast(tf.reduce_sum([num_tp_, num_fn_]), tf.float32) +
                         neg_penalty_ * tf.cast(tf.reduce_sum([num_tn_, num_fp_]), tf.float32)
                         ) / tf.cast(num_sum, tf.float32)
            return p_si_snr_

        def p_si_snr_negative(avg_si_snr_, neg_penalty_, num_tp_, num_fn_, num_tn_, num_fp_):
            p_si_snr_ = (avg_si_snr_ /
                         tf.cast(tf.reduce_sum([num_tp_, num_fn_]), tf.float32) *
                         tf.cast(tf.square(num_tp_), tf.float32) +
                         neg_penalty_ * tf.cast(tf.reduce_sum([num_tn_, num_fp_]), tf.float32)
                         ) / tf.cast(num_sum, tf.float32)
            return p_si_snr_

        p_si_snr = tf.cond(tf.greater_equal(avg_oracle_si_snr, 0.),
                           lambda: p_si_snr_positive(avg_oracle_si_snr, neg_penalty, num_tp, num_fn, num_tn, num_fp),
                           lambda: p_si_snr_negative(avg_oracle_si_snr, neg_penalty, num_tp, num_fn, num_tn, num_fp))
        result_dict.update({'p_si_snr': p_si_snr})

        result_dict.update({'num_tp': num_tp})
        result_dict.update({'num_tn': num_tn})
        result_dict.update({'num_fp': num_fp})
        result_dict.update({'num_fn': num_fn})

        self.result_dict = result_dict
        return p_si_snr

    def compute_fuss_p_si_snr(self, source_waveforms, separated_waveforms, pit=True):
        """compute p_si_snr for FUSS.
        Args:
            source_waveforms (tf.Tensor): (n_channel, frame_length) source waveforms.
            separated_waveforms (tf.Tensor): (n_channel, frame_length) separated waveforms without align.
            pit (bool, optional): whether to use permutation invariant training. Defaults to True.
        Returns:
            p_si_snr (float): p_si_snr.
            num_penalty_pairs (int): miscounting pairs.
            result_dict (dict): dictionary for results.
        """
        if pit:
            separated_waveforms = self.compute_fuss_permutation(source_waveforms, separated_waveforms)
        result_dict = self.compute_fuss_num_active(separated_waveforms, source_waveforms)
        weights_active_refs = result_dict['weights_active_refs']
        weights_active_seps = result_dict['weights_active_seps']
        sisnr_separated = result_dict['sisnr_separated']
        num_active_refs = result_dict['num_active_refs']
        num_active_seps = result_dict['num_active_seps']
        num_penalty_pairs = abs(num_active_refs - num_active_seps)
        p_si_snr = self.compute_p_si_snr(sisnr_separated, weights_active_refs, weights_active_seps)
        result_dict.update({'num_penalty_pairs': num_penalty_pairs})
        self.result_dict = result_dict
        return p_si_snr

    def compute_kim_num_active(self, source_waveforms, separated_waveforms, mixture_waveform, y_labels_real, hparams):
        """compute number of active outputs for KIM 2023.
        Args:
            source_waveforms (tf.Tensor): (n_channel, frame_length) source waveforms.
            separated_waveforms (tf.Tensor): (n_channel, frame_length) separated waveforms.
            mixture_waveform (tf.Tensor): (frame_length, ) mixture waveforms.
            y_labels_real (tf.Tensor): (n_channel, ) labels of mixture.
            hparams (HParams): Model hyperparameters.
        Returns:
            (dict): dictionary for results.
        """
        from experiments.fuss.groupwise import apply as groupwise_apply
        from experiments.KIM2023.identify_valid_output_channels import determine_active_channels

        num_sources = hparams.num_sources
        frame_length = hparams.frame_length
        source_waveforms = tf.reshape(source_waveforms, (num_sources, frame_length))
        separated_waveforms = tf.reshape(separated_waveforms, (num_sources, frame_length))
        mixture_waveform = tf.reshape(mixture_waveform, (1, frame_length))

        signal_types = hparams.signal_types
        unique_signal_types = list(set(signal_types))
        eta = hparams.eta
        valid_frequency_channel = hparams.valid_frequency_channel

        # determine_active_channels in KIM 2023
        separated_is_nonzero_mask = determine_active_channels(mixture_waveform, separated_waveforms, eta,
                                                              valid_frequency_channel)
        separated_is_nonzero_mask_tensor = tf.expand_dims(tf.cast(separated_is_nonzero_mask, tf.float32), axis=-1)
        separated_waveforms_nonzero = tf.multiply(separated_waveforms, separated_is_nonzero_mask_tensor)

        # Initial loss functions for resolving permutations.
        loss_fns = {signal_type: neg_cosine_similarity_stabilizer_tf for signal_type in unique_signal_types}
        # Permute separated to match references through loss_fns.
        _, permuted_estimates = groupwise_apply(
            loss_fns, signal_types, source_waveforms, separated_waveforms_nonzero, unique_signal_types,
            1, num_sources, frame_length)  # (num_samples, n_source, frame_length)

        def compute_y_label_estimates(permuted_estimates_, num_sources_):
            y_label_estimate = []
            for j in range(num_sources_):
                permuted_estimate_j = permuted_estimates_[j]
                y_label_estimate.append(tf.greater(tf.reduce_mean(tf.square(permuted_estimate_j)), 0.))
            y_label_estimate = tf.stack(y_label_estimate)
            return y_label_estimate

        permuted_estimates = tf.squeeze(permuted_estimates)  # (n_source, frame_length)
        # Compute weights for active (separated, source) pairs where source is nonzero
        weights_active_refs = tf.cast(y_labels_real, tf.bool)
        weights_active_seps = compute_y_label_estimates(permuted_estimates, num_sources)
        weights_active_pairs = tf.logical_and(weights_active_refs, weights_active_seps)

        # Compute SI-SNR.
        sisnr_separated = si_snr_loss_tf(permuted_estimates, source_waveforms)
        num_active_refs = tf.reduce_sum(tf.cast(weights_active_refs, tf.int32))
        num_active_seps = tf.reduce_sum(tf.cast(weights_active_seps, tf.int32))
        num_active_pairs = tf.reduce_sum(tf.cast(weights_active_pairs, tf.int32))

        # Compute under/equal/over separation.
        under_separation = tf.cast(tf.less(num_active_seps, num_active_refs), tf.int32)
        equal_separation = tf.cast(tf.equal(num_active_seps, num_active_refs), tf.int32)
        over_separation = tf.cast(tf.greater(num_active_seps, num_active_refs), tf.int32)

        self.result_dict = {'sisnr_separated': sisnr_separated,
                            'under_separation': under_separation,
                            'equal_separation': equal_separation,
                            'over_separation': over_separation,
                            'weights_active_refs': weights_active_refs,
                            'weights_active_seps': weights_active_seps,
                            'weights_active_pairs': weights_active_pairs,
                            'num_active_refs': num_active_refs,
                            'num_active_seps': num_active_seps,
                            'num_active_pairs': num_active_pairs}
        return self.result_dict

    def compute_kim_p_si_snr(self, source_waveforms, separated_waveforms, mixture_waveforms, y_labels_real, hparams):
        """compute p_si_snr for KIM 2023.
        Args:
            source_waveforms (tf.Tensor): (n_channel, frame_length) source waveforms.
            separated_waveforms (tf.Tensor): (n_channel, frame_length) separated waveforms without align.
            mixture_waveforms (tf.Tensor): (1, frame_length) mixture waveforms.
            y_labels_real (tf.Tensor): (n_channel, ) labels of mixture.
            hparams (HParams): Model hyperparameters.
        Returns:
            p_si_snr (float): p_si_snr.
        """
        result_dict = self.compute_kim_num_active(source_waveforms, separated_waveforms, mixture_waveforms,
                                                  y_labels_real, hparams)
        weights_active_refs = result_dict['weights_active_refs']
        weights_active_seps = result_dict['weights_active_seps']
        sisnr_separated = result_dict['sisnr_separated']
        num_active_refs = result_dict['num_active_refs']
        num_active_seps = result_dict['num_active_seps']
        num_penalty_pairs = abs(num_active_refs - num_active_seps)
        p_si_snr = self.compute_p_si_snr(sisnr_separated, weights_active_refs, weights_active_seps)
        result_dict.update({'num_penalty_pairs': num_penalty_pairs})
        self.result_dict = result_dict
        return p_si_snr
