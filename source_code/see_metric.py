# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 21:43:44 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

See metrics for signal separation.

"""
# pylint: disable=line-too-long, too-many-arguments, too-many-locals, useless-object-inheritance
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import museval
import pandas as pd
import seaborn as sns

from error import ParameterError
from file_operation import list_dirs
from metric import samerate_acc_np, mse_np
from prepare_data_shipsear_recognition_mix_s0tos3 import save_datas


class DisplayMetric(object):
    """Display predict metric."""

    def __init__(self, z_list=None, zp_list=None, path_save=None):
        """__init__
        Args:
            z_list (list[np.ndarray], optional): list of target data. Defaults to None.
            zp_list (list[np.ndarray], optional): list of predict data. Defaults to None.
            path_save (str, optional): path where to save files. Defaults to None.
        """
        if z_list:
            self._z_list = z_list
        if zp_list:
            self._zp_list = zp_list
        self._path_save = path_save
        self.metrics = []  # [seti](n_src)(n_samples)
        self._mean_metric = None

    def compute_metric(self, metric_func, mode=1):
        """Compute all samples predict metric.
        Args:
            metric_func (function): function of compute metric.
            mode (int, optional): mode of data struct. Defaults to 1.
        Returns:
            metrics: metrics. list[list[np.ndarray]] when mode==1, list[np.ndarray] when mode==2.
        """
        metrics = []
        if mode == 1:  # [seti](n_src, n_samples, fl)
            for z_si, zp_si in zip(self._z_list, self._zp_list):
                metrics_si = []
                for z_i, zp_i in zip(z_si, zp_si):  # n_src
                    metrics_si.append(
                        [metric_func(zij, zpij) for zij, zpij in zip(z_i, zp_i)])
                metrics.append(np.asarray(metrics_si))
        elif mode == 2:  # [seti](n_sam, fl)
            for z_si, zp_si in zip(self._z_list, self._zp_list):
                metrics_si = [metric_func(z_i, zp_i) for z_i, zp_i in zip(z_si, zp_si)]
                metrics.append(np.asarray(metrics_si))

        self.metrics = metrics
        return self.metrics

    def display_cat_violin(self, metrics, channel_names, save_name=None):
        """Display predict metric violin by channel.
        Args:
            metrics (list[list[np.ndarray]]): metrics.
            channel_names (list[str]): list of channel names.
            save_name ([type], optional): [description]. Defaults to None.
        """
        metrics_list_df = [pd.DataFrame(m_i, columns=channel_names) for m_i in metrics]
        # metrics_df = pd.concat(
        #     metrics_list_df, keys=["train", "val", "test"], copy=False
        #     ).reset_index().rename(columns={"level_0": "set", "level_1": "num"})
        metrics_df = pd.concat(
            metrics_list_df, keys=channel_names, copy=False
        ).reset_index().rename(columns={"level_0": "set", "level_1": "num"})

        sns.catplot(x="source", y="value", hue="channel",
                    kind="violin", scale="count", cut=0,
                    data=pd.melt(
                        metrics_df, id_vars=["set", "num"], value_vars=channel_names,
                        var_name=["channel"], value_name="value"))

        if self._path_save:
            plt.savefig(os.path.join(self._path_save, f'{save_name}_cat_violin.svg'))
            plt.close()

    def compute_mean_metrics(self, metrics=None):
        """Compute predict metric mean with channel.
        Args:
            metrics (list[list[np.ndarray]], optional): metrics. Defaults to None.
        Returns:
            self._mean_metric (list[float]): list of the mean of the metrics.
        """
        if metrics is None:
            metrics = self.metrics
        self._mean_metric = [np.mean(m_i, axis=-1) for m_i in metrics]
        return self._mean_metric

    def display_static_metrics(self, metrics=None, set_names=None,
                               form=None, hist_bins=None, save_name=None):
        """Display metrics static information.
        Args:
            metrics (list[list[np.ndarray]], optional): metrics. Defaults to None.
            set_names (list[str], optional): list of names of sets, e.g. ['train', 'val', 'test']. Defaults to None.
            form (list[str], optional): type of metrics to display. Defaults to None.
            hist_bins (tuple(float, float, int), optional): paras for hist. Defaults to (0, 1, 21).
            save_name (str, optional): for name to save. Defaults to None.
        Raises:
            ParameterError: if item of form not in ['violin', 'hist'].
        """
        # metrics_list_df = [pd.DataFrame(
        #     m_i, columns=[set_names[i]]) for i, m_i in enumerate(mean_metrics)]
        metrics_list_ser = [
            pd.Series(m_i, name=set_names[i]) for i, m_i in enumerate(metrics)]
        for item in form:
            if item == 'violin':
                # metrics_df = pd.concat(metrics_list_df, axis=1)
                metrics_df = pd.concat(metrics_list_ser, axis=1)
                metrics_df = pd.melt(metrics_df, value_vars=set_names,
                                     var_name=["set"], value_name="value")
                sns.violinplot(x="set", y="value", hue="set",
                               cut=0, scale="count", inner="box", data=metrics_df)
            elif item == 'hist':
                if hist_bins is None:
                    value_min = min([metric_i.min() for metric_i in metrics_list_ser])
                    value_max = max([metric_i.max() for metric_i in metrics_list_ser])
                    hist_bins = (value_min, value_max, 10)

                _, axes = plt.subplots(1, len(set_names), sharex=True)
                for i, metrics_set_i in enumerate(metrics_list_ser):
                    if len(set_names) > 1:
                        sns.distplot(metrics_set_i,
                                     bins=np.linspace(hist_bins[0], hist_bins[1], hist_bins[2]),
                                     ax=axes[i])
                    else:
                        sns.distplot(metrics_set_i,
                                     bins=np.linspace(hist_bins[0], hist_bins[1], hist_bins[2]),
                                     ax=axes)

            else:
                raise ParameterError('Wrong display form.')

            if save_name:
                plt.savefig(os.path.join(self._path_save, f'{save_name}_{item}.svg'))
                plt.close()

    def display_mean_metrics(self, metrics=None, set_names=None,
                             form=None, hist_bins=None, save_name=None):
        """Display predict metric mean with channel.
        Args:
            metrics (list[list[np.ndarray]], optional): metrics. Defaults to None.
            set_names (list[str], optional): list of names of sets, e.g. ['train', 'val', 'test']. Defaults to None.
            form (list[str], optional): type of metrics to display. Defaults to None.
            hist_bins (tuple(float, float, int), optional): paras for hist. Defaults to (0, 1, 21).
            save_name (str, optional): for name to save. Defaults to None.
        """
        if metrics is None:
            mean_metrics = self.compute_mean_metrics()
        else:
            mean_metrics = self.compute_mean_metrics(metrics)

        self.display_static_metrics(mean_metrics, set_names, form, hist_bins, save_name)


def compute_mse(s_list, sp_list, path_metric, file_name, set_name, hist_bins, save_name, save_metric=True):
    """Compute and save mse.
    Args:
        s_list (list[np.ndarray,shape==(n_src,n_sams,n_channel=1,fl)]): list of target data.
        sp_list (list[np.ndarray,shape==(n_src,n_sams,n_channel=1,fl)]): list of predict data.
        path_metric (str): path where to save files.
        file_name (str): name of file to save.
        set_name (list[str]): name of the sources, e.g. ['s_1', 's_2', 's_3'].
        hist_bins (tuple(float, float, int)): paras for hist.
        save_name (str): name of file to save.
        save_metric (bool, optional): whether save metric to file. Defaults to True.
    Returns:
        mse_list (list[np.ndarray]): [metric](n_src, n_sams), list of mse.
        mse_mean (list[np.ndarray]): [metric], list of mean mse.
    """
    obj_dm = DisplayMetric(s_list, sp_list, path_metric)
    mse_list = obj_dm.compute_metric(mse_np, mode=2)
    mse_mean = obj_dm.compute_mean_metrics(mse_list)
    obj_dm.display_static_metrics(mse_list, set_name, ['violin', 'hist'],
                                  hist_bins=hist_bins, save_name=save_name)
    if save_metric:
        save_datas({'mse_mean': np.asarray(mse_mean), 'mse': np.asarray(mse_list)},
                   path_metric, file_name=file_name, **{'mode_batch': 'one_file_no_chunk'})
    return mse_list, mse_mean


def compute_sr(s_list, sp_list, path_metric, file_name, set_name, hist_bins, save_name, save_metric=True):
    """Compute and save sr.
    Args:
        s_list (list[np.ndarray,shape==(n_src,n_sams,n_channel=1,fl)]): list of target data.
        sp_list (list[np.ndarray,shape==(n_src,n_sams,n_channel=1,fl)]): list of predict data.
        path_metric (str): path where to save files.
        file_name (str): name of file to save.
        set_name (list[str]): name of the sources, e.g. ['s_1', 's_2', 's_3'].
        hist_bins (tuple(float, float, int), optional): paras for hist. Defaults to None.
        save_name (str): name of file to save.
        save_metric (bool, optional): whether save metric to file. Defaults to True.
    Returns:
        sr_list (list[np.ndarray]): [metric](n_src, n_sams), list of sr.
        sr_mean (list[np.ndarray]): [metric], list of mean sr.
    """
    obj_dm = DisplayMetric(s_list, sp_list, path_metric)
    sr_list = obj_dm.compute_metric(samerate_acc_np, mode=2)
    sr_mean = obj_dm.compute_mean_metrics(sr_list)
    obj_dm.display_static_metrics(sr_list, set_name, ['violin', 'hist'],
                                  hist_bins=hist_bins, save_name=save_name)
    if save_metric:
        save_datas({'sr_mean': np.asarray(sr_mean), 'sr': np.asarray(sr_list)},
                   path_metric, file_name=file_name, **{'mode_batch': 'one_file_no_chunk'})
    return sr_list, sr_mean


def save_metric_static(metric_arr, path_metric, metric_name, file_name, set_name, hist_bins, save_name,
                       is_save_metric=True):
    """Save metrics static information.
    Args:
        metric_arr (list[list[np.ndarray]], optional): metrics. Defaults to None.
        path_metric (str): path where to save files.
        metric_name (str): name of the metric.
        file_name (str): name of the file to save.
        set_name (list[str]): name of the sources, e.g. ['s_1', 's_2', 's_3'].
        hist_bins (tuple(float, float, int)): paras for hist.
        save_name (str): for name to save.
        is_save_metric (bool, optional): whether save to file. Defaults to True.
    Returns:
        metric_mean (list[float]): list of the mean of the metrics.
    """
    obj_dm = DisplayMetric(None, None, path_metric)
    obj_dm.display_static_metrics(metric_arr, set_name, ['violin', 'hist'],
                                  hist_bins=hist_bins, save_name=save_name)
    metric_mean = obj_dm.compute_mean_metrics(metric_arr)
    if is_save_metric:
        save_datas({f'{metric_name}_mean': np.asarray(metric_mean), metric_name: metric_arr},
                   path_metric, file_name=file_name, **{'mode_batch': 'one_file_no_chunk'})
    return metric_mean


def compute_sdr(s_list, sp_list, path_metric, file_name, set_name, hist_bins, save_name, is_save_metric=True):
    """Compute sdr related metrics.
    Args:
        s_list (list[np.ndarray,shape==(n_src,n_sams,n_channel=1,fl)]): list of target data.
        sp_list (list[np.ndarray,shape==(n_src,n_sams,n_channel=1,fl)]): list of predict data.
        path_metric (str): path where to save files.
        file_name (str): name of file to save.
        set_name (list[str]): name of the sources, e.g. ['s_1', 's_2', 's_3'].
        hist_bins (tuple(float, float, int), optional): paras for hist. Defaults to None.
        save_name (str): name of file to save.
        is_save_metric (bool, optional): whether save metric to file. Defaults to True.
    Returns:
        metric_list (list[np.ndarray]): [metric](n_src, n_sams), list of metrics.
        metric_mean_list (list[np.ndarray]): [metric], list of mean of metrics.
    """
    s_arr = np.asarray(s_list).transpose(1, 0, 3, 2)  # (n_src, n_sams, n_channel=1, fl)->(n_sams, n_src, fl, 1)
    sp_arr = np.asarray(sp_list).transpose(1, 0, 3, 2)

    n_sams = s_arr.shape[0]
    n_src = s_arr.shape[1]

    sdr_arr = np.empty((n_sams, n_src), dtype=np.float32)
    isr_arr = np.empty((n_sams, n_src), dtype=np.float32)
    sir_arr = np.empty((n_sams, n_src), dtype=np.float32)
    sar_arr = np.empty((n_sams, n_src), dtype=np.float32)
    for j, (s_j, sp_j) in enumerate(zip(s_arr, sp_arr)):
        sdr_arr[j], isr_arr[j], sir_arr[j], sar_arr[j] = np.squeeze(museval.evaluate(
            s_j, sp_j, win=np.inf, hop=np.inf, padding=False))
    sdr_arr = sdr_arr.transpose()  # (n_src, n_sams)
    isr_arr = isr_arr.transpose()
    sir_arr = sir_arr.transpose()
    sar_arr = sar_arr.transpose()

    metric_list = [sdr_arr, isr_arr, sir_arr, sar_arr]
    metric_mean_list = []
    for file_name_i, hist_bins_i, save_name_i, metric_arr_i in zip(file_name, hist_bins, save_name, metric_list):
        metric_mean_i = save_metric_static(metric_arr_i, path_metric, file_name_i, file_name_i,
                                           set_name, hist_bins_i, save_name_i, is_save_metric)
        metric_mean_list.append(metric_mean_i)

    return metric_list, metric_mean_list


def compute_si_sdr(s_list, sp_list, scaling, path_metric, file_name, set_name, hist_bins, save_name,
                   is_save_metric=True):
    """Compute si_sdr or sd_sdr related metrics.
    Args:
        s_list (list[np.ndarray,shape==(n_src,n_sams,n_channel=1,fl)]): list of target data.
        sp_list (list[np.ndarray,shape==(n_src,n_sams,n_channel=1,fl)]): list of predict data.
        scaling (bool): si_sdr True or sd_sdr False.
        path_metric (str): path where to save files.
        file_name (list[str]): list of metric name, e.g. ['sdr', 'isr', 'sir', 'sar'].
        set_name (list[str]): name of the sources, e.g. ['s_1', 's_2', 's_3'].
        hist_bins (tuple(float, float, int)): paras for hist.
        save_name (str): name of file to save.
        is_save_metric (bool, optional): whether save metric to file. Defaults to True.
    Returns:
        metric_list (list[np.ndarray]): [metric](n_src, n_sams), list of metrics.
        metric_mean_list (list[np.ndarray]): [metric], list of mean of metrics.
    """
    from metrics.si_sdr import ScaleInvariantSDR  # pylint: disable=import-outside-toplevel

    s_arr = np.asarray(s_list).transpose(1, 3, 2, 0)  # (n_src, n_sams, n_channel=1, fl)->(n_sams, fl, 1, n_src)
    sp_arr = np.asarray(sp_list).transpose(1, 3, 2, 0)

    n_sams = s_arr.shape[0]
    n_src = s_arr.shape[-1]

    sdr_arr = np.empty((n_sams, n_src), dtype=np.float32)
    sir_arr = np.empty((n_sams, n_src), dtype=np.float32)
    sar_arr = np.empty((n_sams, n_src), dtype=np.float32)

    for j, (s_j, sp_j) in enumerate(zip(s_arr, sp_arr)):
        si_sdr_class = ScaleInvariantSDR(s_j, sp_j, scaling=scaling)
        # sdr_arr[j], sir_arr[j], sar_arr[j] =
        score_dict = si_sdr_class.evaluate()
        scores = [value for key, value in score_dict.items()][1:]
        for k, score in enumerate(scores):
            sdr_arr[j][k] = score['SDR'][0]
            sir_arr[j][k] = score['SIR'][0]
            sar_arr[j][k] = score['SAR'][0]

    sdr_arr = sdr_arr.transpose()  # (n_src, n_sams)
    sir_arr = sir_arr.transpose()
    sar_arr = sar_arr.transpose()

    metric_list = [sdr_arr, sir_arr, sar_arr]
    metric_mean_list = []
    for file_name_i, hist_bins_i, save_name_i, metric_arr_i in zip(
            file_name, hist_bins, save_name, metric_list):
        metric_mean_i = save_metric_static(metric_arr_i, path_metric, file_name_i, file_name_i,
                                           set_name, hist_bins_i, save_name_i, is_save_metric)
        metric_mean_list.append(metric_mean_i)

    return metric_list, metric_mean_list


def metric_mean_mix_to_file(path_metric, sub_dirs, metric_keys, path_save):
    """Merge metric_mean files to .hdf5 file.
    Args:
        path_metric (str): path where to save files.
        sub_dirs (list[str]): name of the sets of datas.
        metric_keys (list[str]): name of the metrics.
        path_save (str): path to save merged file.
    """
    path_dirs_name = list_dirs(path_metric, False)
    for metric_key in metric_keys:
        for sub_dir in sub_dirs:
            f_h5 = h5py.File(os.path.join(path_save, f'{metric_key}_mean_{sub_dir}.hdf5'), 'a')
            for path_dir_name in path_dirs_name:
                grp_dir = f_h5.create_group(path_dir_name)
                path_metric_root = os.path.join(path_metric, path_dir_name, sub_dir, 'metric')
                for name in list_dirs(path_metric_root, False):
                    grp_dir[name] = np.asarray(h5py.File(os.path.join(path_metric_root,
                                                                      name,
                                                                      f'{metric_key}.hdf5'
                                                                      ), 'r'
                                                         )[f'{metric_key}_mean'], dtype=np.float32)
            f_h5.close()


def metric_mean_src_to_file(path_metric, sub_dirs, metric_keys, path_save):
    """Merge metric_mean files to .hdf5 file.
    Args:
        path_metric (str): path where to save files.
        sub_dirs (list[str]): name of the sets of datas.
        metric_keys (list[str]): name of the metrics.
        path_save (str): path to save merged file.
    """
    dirs_name = list_dirs(path_metric, False)
    for metric_key in metric_keys:
        for sub_dir in sub_dirs:
            f_h5 = h5py.File(os.path.join(path_save, f'{metric_key}_mean_{sub_dir}_src.hdf5'), 'a')
            for dir_name in dirs_name:
                path_metric_root = os.path.join(path_metric, dir_name, sub_dir, 'metric')
                f_h5[dir_name] = np.asarray(h5py.File(os.path.join(path_metric_root,
                                                                   f'{metric_key}.hdf5'
                                                                   ), 'r'
                                                      )[f'{metric_key}_mean'], dtype=np.float32)
            f_h5.close()


def metric_mean_to_csv(file_name, level=1):
    """Merge metric_mean files to .csv file.
    Args:
        file_name (str): file name of the .hdf5 metric.
        level (int, optional): level of the data saved. Defaults to 1.
    """
    with h5py.File(file_name, 'r') as h5_f:
        if level == 1:
            data_arr = np.vstack([np.asarray(h5_f[key]) for key in h5_f.keys()])
        elif level == 2:
            data_list = []
            for key in h5_f.keys():
                data_list_2 = []
                for key_2 in h5_f[key].keys():
                    data_list_2.append(np.asarray(h5_f[key][key_2]))
                data_list.append(np.concatenate(data_list_2))
            data_arr = np.vstack(data_list)
        file_name, _ = os.path.splitext(file_name)
        np.savetxt(f'{file_name}.csv', data_arr, delimiter=",")
