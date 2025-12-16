# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 17:21:38 2025

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""

import librosa
import logging
import numpy as np
import scipy.signal
from sklearn.decomposition import FastICA
from coroica import CoroICA

from error import ParameterError


class FastIcaSklearn(FastICA):
    """FastICA using scikit-learn"""

    def get_config(self):
        return {'n_components': self.n_components,
                'algorithm': self.algorithm,
                'whiten': self.whiten,
                'fun': self.fun,
                'fun_args': self.fun_args,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'w_init': self.w_init,
                'random_state': self.random_state, }


class CoroIca(CoroICA):
    """CoroICA using coroica.
      N. Pfister, S. Weichwald, P. Bühlmann, and B. Schölkopf, “Robustifying Independent Component Analysis by Adjusting
      for Group-Wise Stationary Noise,” Journal of Machine Learning Research, vol. 20, no. 147, pp. 1–50, 2019.
    """

    def get_config(self):
        return {'n_components': self.n_components,
                'n_components_uwedge': self.n_components_uwedge,
                'rank_components': self.rank_components,
                'pairing': self.pairing,
                'max_matrices': self.max_matrices,
                'groupsize': self.groupsize,
                'partitionsize': self.partitionsize,
                'timelags': self.timelags,
                'instantcov': self.instantcov,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'minimize_loss': self.minimize_loss,
                'condition_threshold': self.condition_threshold,
                'random_state': self.random_state,
                'skip_sklearn_checks': self.skip_sklearn_checks, }


def interval_resample(original_data, num_virtual_channel, orig_sr, target_sr=None):
    target_sr = orig_sr * num_virtual_channel if target_sr is None else target_sr
    resample_data = librosa.resample(np.squeeze(original_data), orig_sr, target_sr, res_type='scipy')
    virtual_multichannel_data = librosa.util.frame(resample_data, num_virtual_channel, num_virtual_channel).transpose()
    return virtual_multichannel_data


def time_delay_iir(original_data, num_virtual_channel, n_time_delay, filter_param):
    original_data = np.squeeze(original_data)
    virtual_multichannel_data = [original_data, ]
    sig_delay = None
    for i in range(0, num_virtual_channel - 1):
        sig_delay_i = original_data if i == 0 else sig_delay
        sig_delay = np.roll(sig_delay_i, n_time_delay)
        b, a = scipy.signal.butter(filter_param[0], filter_param[1])
        sig_delay_filter = scipy.signal.lfilter(b, a, sig_delay)
        virtual_multichannel_data.append(sig_delay_filter)
    virtual_multichannel_data = np.asarray(virtual_multichannel_data).transpose()
    return virtual_multichannel_data


def train_ica_model(num_model, x_dataset, params, **kwargs):
    num_source = params['n_components']
    expand_channel_mode = kwargs['expand_channel_mode'] if 'expand_channel_mode' in kwargs else 'copy'
    s_predict = np.empty(x_dataset.shape[:-1] + (num_source,), dtype=np.float32)

    if num_model == 1000:
        ica_model = FastIcaSklearn(**params)
    elif num_model == 2000:
        ica_model = CoroIca(**params)
    else:
        raise ParameterError(f'invalid num_model {num_model}')

    for i, x_data_i in enumerate(x_dataset):
        if expand_channel_mode == 'copy':
            x_data_i = np.repeat(x_data_i, num_source, axis=-1)
        elif expand_channel_mode == 'interval_resample':
            """刘 佳;杨士莪;朴胜春. 多途环境下的单通道水声信号盲源分离[J]. 振动与冲击, 2012, 31(6): 15-18.
             LIU Jia;YANG Shi-e;PIAO Sheng-chun. Single channel blind source separation of underwater acoustic signal in
             multipath condition[J]. Journal of Vibration and Shock, 2012, 31(6): 15-18.
             https://jvs.sjtu.edu.cn/CN/Y2012/V31/I6/15."""

            num_virtual_channel = kwargs['num_virtual_channel']
            orig_sr = kwargs['orig_sr']
            x_data_i = interval_resample(x_data_i, num_virtual_channel, orig_sr)
        elif expand_channel_mode == 'time_delay_iir':
            """刘佳;杨士莪;朴胜春;黄益旺. 单观测通道船舶辐射噪声盲源分离[J]. 声学学报, 2011, 36(3): 265–270.
             LIU Jia, YANG Shi-e, PO Sheng-chun, HUANG Yi-wang. Blind source separation of ship-radiated noise using
             single observing channel[J]. Acta Acustica, 2011, 36(3): 265-270.
             DOI: 10.15949/j.cnki.0371-0025.2011.03.013."""

            num_virtual_channel = kwargs['num_virtual_channel']
            n_time_delay = kwargs['n_time_delay']
            filter_param = kwargs['filter_param']
            x_data_i = time_delay_iir(x_data_i, num_virtual_channel, n_time_delay, filter_param)
        else:
            raise ParameterError(f'invalid expand_channel_mode {expand_channel_mode}')

        try:
            s_predict_i = ica_model.fit_transform(x_data_i)
        except Exception as e:  # pylint: disable=broad-except
            logging.info('Failed to fit_transform', e)
            s_predict_i = np.zeros(x_dataset.shape[1:-1] + (num_source,), dtype=np.float32)
        s_predict[i] = s_predict_i

    config_dict = ica_model.get_config()

    return s_predict, config_dict
