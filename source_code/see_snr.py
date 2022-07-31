# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:10:42 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""

import logging
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from error import ParameterError
from file_operation import mkdir
from prepare_data import snr_np
from prepare_data_shipsear_recognition_mix_s0tos3 import PathSourceRoot, read_datas, save_datas


class DisplayMetric(object):
    """Display predict metric."""

    def __init__(self, s1_arr=None, s2_arr=None):
        if s1_arr:
            self._s1_arr = s1_arr
        if s2_arr:
            self._s2_arr = s2_arr

    def compute_metric(self, metric_func):
        """Compute all samples predict metic."""

        metrics = []
        for s1, s2 in zip(self._s1_arr, self._s2_arr):
            metrics.append(metric_func(s1, s2))

        self.metrics = np.asarray(metrics)
        return self.metrics

    def compute_mean_metrics(self, metrics=None):
        """Compute predict metric mean with channel."""

        if metrics is None:
            metrics = self.metrics
        self._mean_metric = np.mean(metrics, axis=-1)
        return self._mean_metric

    def display_static_metrics(self, metrics=None, set_name=None, form=None,
                               hist_bins=(0, 1, 21), path_save=None, save_name=None):
        """Display metrics static information."""
        metrics_ser = pd.Series(metrics, name=set_name)
        for item in form:
            if item == 'violin':
                metrics_df = pd.concat([metrics_ser], axis=1)
                metrics_df = pd.melt(metrics_df, value_vars=set_name,
                                     var_name=["set"], value_name="value")
                sns.violinplot(x="set", y="value", hue="set",
                               cut=0, scale="count", inner="box", data=metrics_df)
            elif item == 'hist':
                _, axes = plt.subplots(1, 1, sharex=True)
                sns.distplot(metrics_ser,
                             bins=np.linspace(hist_bins[0], hist_bins[1], hist_bins[2]),
                             ax=axes)
            else:
                raise ParameterError('Wrong display form.')

            if save_name:
                plt.savefig(os.path.join(path_save, f'{save_name}_{item}.svg'))
            plt.close()


def compute_snr(s_arr, sn_arr, path_metric, file_name, set_name, hist_bins, save_name, save_metric=True):
    """Compute and save snr."""
    dm = DisplayMetric(s_arr, sn_arr)
    snr_arr = dm.compute_metric(snr_np)
    snr_mean = dm.compute_mean_metrics(snr_arr)
    dm.display_static_metrics(snr_arr, set_name, ['violin', 'hist'],
                              hist_bins=hist_bins, path_save=path_metric, save_name=save_name)
    if save_metric:
        save_datas({'snr_mean': np.asarray(snr_mean), 'snr': snr_arr},
                   path_metric, file_name=file_name, **{'mode_batch': 'one_file_no_chunk'})
    return snr_arr, snr_mean


if __name__ == '__main__':
    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'
    PATH_CLASS_DATA = PathSourceRoot(PATH_DATA_ROOT, form_src='wav')
    PATH_SOURCE = PATH_CLASS_DATA.path_source_root
    S_NAMES = json.load(open(os.path.join(PATH_SOURCE, 'dirname.json'), 'r'))['dirname']
    S_LIST = read_datas(os.path.join(PATH_SOURCE, 's_hdf5'), S_NAMES[:4])
    PATH_SNR = '../result_snr'
    mkdir(PATH_SNR)

    # s1/s2
    compute_snr(S_LIST[1], S_LIST[2], PATH_SNR, 'SNR_1_2', 's_1_2',
                hist_bins=(-60, 60, 24), save_name='SNR_1_2')
    # s1/s3
    compute_snr(S_LIST[1], S_LIST[3], PATH_SNR, 'SNR_1_3', 's_1_3',
                hist_bins=(-60, 60, 24), save_name='SNR_1_3')
    # s2/s3
    compute_snr(S_LIST[2], S_LIST[3], PATH_SNR, 'SNR_2_3', 's_2_3',
                hist_bins=(-60, 60, 24), save_name='SNR_2_3')
