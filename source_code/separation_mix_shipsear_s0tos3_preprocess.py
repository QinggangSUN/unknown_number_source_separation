# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 10:54:10 2019

@author: sqg
"""
import os

import h5py
import json
import logging
import numpy as np
import pickle
from sklearn import preprocessing

from file_operation import mkdir
from prepare_data_shipsear_recognition_mix_s0tos3 import read_datas, save_datas
from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep
from recognition_mix_shipsear_s0tos3_preprocess import n_hot_labels, subset_nums_create, subset_x, y_sets_create

# pylint: disable=no-member, invalid-name
# pylint: disable=line-too-long, too-many-arguments, too-many-branches, too-many-locals, useless-object-inheritance


def sourceframes_mo_create(x_source_frames, scaler_save=None):
    """Scaler sources to max abs amplitude in each pair mixed data to one.
    Args:
        x_source_frames (list[np.ndarray, shape==(n_samples,1,fl)]): [n_sources] sources to be scalered.
        scaler_save (str): path to save scaler datas.
    Returns:
        x_source_frames_standard (list[np.ndarray, shape==(n_samples,1,fl)]): sources after scalered.
    """
    n_samples = x_source_frames[0].shape[0]
    n_sources = len(x_source_frames)
    scaler_arr = np.empty((n_samples,), dtype=np.float32)
    x_source_frames_standard = [np.empty(x_source_frames[0].shape, dtype=np.float32) for _ in range(n_sources)]
    for j in range(n_samples):
        x_j = np.asarray([x_source_frames[i][j] for i in range(n_sources)])
        max_am_j = np.max(np.abs(x_j))
        scaler_arr[j] = max_am_j
        for i in range(n_sources):
            x_source_frames_standard[i][j] = x_source_frames[i][j] / max_am_j
    if scaler_save is not None:
        path_save, file_name = os.path.split(scaler_save)
        mkdir(path_save)
        save_datas({file_name: scaler_arr}, path_save)
    return x_source_frames_standard


def z_labels_create(n_src):
    """Create labels of Z_train Z_val Z_test.
    Args:
        n_src (int): number of original sources, include s0.
    Returns:
        source_labels (np.ndarray,shape==(n_sources, n_src)): labels of mixed sources.
    Examples:
        >>> print(z_labels_create(4))
        [[1. 0. 0. 0.]
         [0. 1. 0. 0.]
         [0. 0. 1. 0.]
         [0. 0. 0. 1.]
         [0. 1. 1. 0.]
         [0. 1. 0. 1.]
         [0. 0. 1. 1.]
         [0. 1. 1. 1.]]
    """
    from recognition_mix_shipsear_s0tos3_preprocess import n_hot_labels
    labels_n_hot = np.asarray(n_hot_labels(n_src))  # n_hot labels
    source_labels = np.zeros((labels_n_hot.shape[0], labels_n_hot.shape[1] + 1))
    source_labels[:, 1:] = labels_n_hot
    source_labels[0][0] = 1
    return source_labels


def z_sets_create(x_sets_shape, nums, n_src, z_labels, x_source_frames, output_zero, path_save,
                  epsi=1e-6, save_batch=None):  # pylint: disable=too-many-arguments
    """Create z_train, z_val, z_test.
    Args:
        x_sets_shape (list[tuple(int)]): data shape of data sets.
        nums (list[list[tuple(int, int)]]): [n_set][n_samples](sourcei, numi), indexes of samples.
        n_src (int): number of original sources, include s0.
        z_labels (np.ndarray, shape==(n_sources, n_src)): labels of mixed sources.
        x_source_frames (list[np.ndarray, shape==(n_sams,1,fl)]): [n_sources] sources to be scalered.
        output_zero (str) {'zero','epsi'}: output data when no specific type signal in the mixed signal.
        path_save (str): where to save z sets.
        epsi (float, optional): value when output all epsi. Defaults to 1e-6.
        save_batch (int, optional): number of data each batch to save. Defaults to None.
    """
    if save_batch is None:
        for si, (nums_i, name_i) in enumerate(zip(nums, ['Z_train', 'Z_val', 'Z_test'])):
            if output_zero == 'zero':
                z_si = np.zeros(  # (nsamples, nsrc, featureshape)
                    x_sets_shape[si][0:1] + (n_src,) + x_sets_shape[si][1:], dtype=np.float32)
            elif output_zero == 'epsi':
                z_si = np.full(
                    x_sets_shape[si][0:1] + (n_src,) + x_sets_shape[si][1:], epsi, dtype=np.float32)
            for pi, pair_i in enumerate(nums_i):  # [n_samples](sourcei, numi) # pylint: disable=invalid-name
                label_i = z_labels[pair_i[0]]
                for lj in range(label_i.shape[0]):  # pylint: disable=invalid-name
                    if label_i[lj] == 1:
                        z_si[pi][lj] = x_source_frames[lj][pair_i[1]]
    elif save_batch == 1:
        for si, (nums_i, name_i) in enumerate(zip(nums, ['Z_train', 'Z_val', 'Z_test'])):
            for pi, pair_i in enumerate(nums_i):  # [n_samples](sourcei, numi) # pylint: disable=invalid-name
                if output_zero == 'zero':
                    z_si_pi = np.zeros(  # (1, nsrc, featureshape)
                        (1, n_src,) + x_sets_shape[si][1:], dtype=np.float32)
                elif output_zero == 'epsi':
                    z_si_pi = np.full(
                        (1, n_src,) + x_sets_shape[si][1:], epsi, dtype=np.float32)

                label_i = z_labels[pair_i[0]]
                for lj in range(label_i.shape[0]):  # pylint: disable=invalid-name
                    if label_i[lj] == 1:
                        z_si_pi[0][lj] = x_source_frames[lj][pair_i[1]]

                if pi == 0:
                    with h5py.File(os.path.join(path_save, f'{name_i}_{output_zero}.hdf5'), 'w') as f:
                        f.create_dataset(
                            'data', data=z_si_pi,
                            dtype=np.float32,
                            chunks=((z_si_pi.ndim - 1) * (1,) + z_si_pi.shape[-1:]),
                            maxshape=((None,) + z_si_pi.shape[1:]),
                            compression="gzip", compression_opts=9)
                else:
                    with h5py.File(os.path.join(path_save, f'{name_i}_{output_zero}.hdf5'), 'a') as f:
                        f['data'].resize(
                            (f['data'].shape[0] + z_si_pi.shape[0]), axis=0)
                        f['data'][-z_si_pi.shape[0]:] = z_si_pi


def load_source_frames(path_source_root, dir_names, **kwargs):
    """Load original sources.
    Args:
        path_source_root (str): path where s_hdf5 located.
        dir_names (list[str]): names of mixed sources, e.g. 's_1_2'.
    Returns:
        source_frames (list[np.ndarray, shape==(n_samples,1,fl)]): [n_sources] sources loaded.
    """
    if 'mode_read' in kwargs.keys():
        source_frames = read_datas(os.path.join(path_source_root, 's_hdf5'),
                                   dir_names, **{'mode': kwargs['mode_read']})
    else:
        source_frames = read_datas(os.path.join(path_source_root, 's_hdf5'), dir_names)

    return source_frames


def x_y_sets_create(path_class, rates_set, **kwargs):
    """Create X_train, X_val, X_test scalered data, and create Y_train, Y_val, Y_test data labels.
    Args:
        path_class (object class PathSourceSep): object of class to compute path.
        rates_set (list[float]): rates of datasets.
    """
    path_source = path_class.path_source
    path_source_root = path_class.path_source_root
    scaler_data = path_class.get_scaler_data()
    sub_set_way = path_class.sub_set_way

    dir_names = json.load(
        open(os.path.join(path_source_root, 'dirname.json'), 'r'))['dirname']

    x_source_frames = load_source_frames(path_source_root, dir_names)
    if scaler_data == 'max_one':
        x_source_frames = sourceframes_mo_create(x_source_frames,
                                                 os.path.join(path_source, 'scaler_mo_S'))

    x_source_frames = np.asarray(x_source_frames)
    logging.info('x_source_frames read and scaler finished')

    n_sources = x_source_frames.shape[0]  # = nmixsources
    # number of samples per mixsource
    n_samples = x_source_frames.shape[1]

    if not os.path.isfile(os.path.join(path_source_root, f'nums_{sub_set_way}.pickle')):
        subset_nums_create(
            path_source_root, sub_set_way, rates_set, n_samples, n_sources)
    with open(os.path.join(path_source_root, f'nums_{sub_set_way}.pickle'), 'rb') as f_rb:
        nums_rand = pickle.load(f_rb)

    x_sets = subset_x(x_source_frames, nums_rand)
    logging.info('x_sets created finished')

    # n_sources = len(dir_names) = N_SRC + 2^(N_SRC-1)-1-(N_SRC-1)
    n_src = int(np.log2(n_sources) + 1)  # number of original source
    y_labels = n_hot_labels(n_src)
    y_sets = y_sets_create(nums_rand, y_labels, n_src)
    logging.info('y_sets created finished')

    mkdir(path_source)
    save_datas(dict(zip(['X_train', 'X_val', 'X_test'], x_sets)), path_source)
    save_datas(dict(zip(['Y_train', 'Y_val', 'Y_test'], y_sets)), path_source, dtype=np.int32)


def z_data_create(path_class, output_zero='epsi'):
    """Create Z_train, Z_val, Z_test which is target of mixed sample to be separated.
    Args:
        path_class (object class PathSourceSep): object of class to compute path.
        output_zero (str, optional) {'zero','epsi'}: output data when no specific type signal in the mixed signal.
                                                     Defaults to 'epsi'.
    """
    path_source = path_class.path_source
    path_source_root = path_class.path_source_root
    scaler_data = path_class.get_scaler_data()

    x_sets = read_datas(path_source, ['X_train', 'X_val', 'X_test'])
    logging.info('data x_sets load finished')
    x_sets_shape = [x_i.shape for x_i in x_sets]

    with open(os.path.join(path_source_root, f'nums_{path_class.sub_set_way}.pickle'), 'rb') as f_rb:
        nums = pickle.load(f_rb)

    dir_names = json.load(open(os.path.join(path_source_root, 'dirname.json'), 'r'))['dirname']
    n_src = int(np.log2(len(dir_names)) + 1)  # number of original source

    z_labels = z_labels_create(n_src)

    x_sourceframes = read_datas(os.path.join(path_source_root, 's_hdf5'), dir_names)
    logging.info('data sourceframes load finished')
    if scaler_data == 'max_one':
        x_sourceframes = sourceframes_mo_create(x_sourceframes,
                                                os.path.join(path_source, 'scaler_mo_S'))

    z_sets_create(x_sets_shape, nums, n_src, z_labels, x_sourceframes, output_zero, path_source, save_batch=1)
    logging.info('z_sets_create finished')


def z_sets_ns_create(s_list, nums, scaler_data, path_save):
    """Create train, val, test data set through nums.
    Inputs:
        s_list (list[np.ndarray, shape==(n_sams, 1, frame_length)]): [n_src] samples of sources.
        scaler_data (str): way of scaler data.
        nums (list[int]): [n_sams], index of the samples per set of sources.
        path_save (str): path where to save z data sets.
    Returns:
        data_sources(list[list[np.ndarray, shape==(n_sams, 1, frame_length)]]): [n_src][set], data sets of sources.
    """
    def z_set_ns_create(source_arr, nums, set_names=None, save_name=None):
        """Create train, val, test data set through index nums.
        Inputs:
            source_arr (np.ndarray, shape==(n_sams, 1, frame_length)): samples of a source.
            nums (list[list[int]]): [n_sets][n_sams], index of the samples per set.
            set_names (list[str], optional): names of the sets of data. Default None.
            save_name (str): path+file_name to save data to files.
        Returns:
            source_sets(list[np.ndarray, shape==(n_sams, 1, frame_length)]): [set], data sets.
        """
        if set_names is None:
            set_names = ['train', 'val', 'test']

        source_sets = []
        for nums_set_i in nums:
            source_set_i = [source_arr[nums_i_j] for nums_i_j in nums_set_i]
            source_set_i = np.asarray(source_set_i)
            source_sets.append(source_set_i)

        if save_name is not None:
            path_save, file_name = os.path.split(save_name)
            for name_i, data_i in zip(set_names, source_sets):
                save_datas({f'{file_name}_{name_i}': data_i}, path_save)
        return source_sets  # [set](n_sams, 1, frame_length)

    if scaler_data == 'max_one':
        s_list = sourceframes_mo_create(s_list)

    data_sources = []
    for i, s_ci in enumerate(s_list):
        data_sources.append(
            z_set_ns_create(s_ci, nums, save_name=os.path.join(path_save, f'Z_{i}_ns')))

    return data_sources  # [n_src][set](n_sams, 1, frame_length)


def subset_seq(seq, n_sams):
    """Split index seqs to different sets.
    Args:
        seq (list[int]): index of the source.
        n_sams (list[int]): numbers of the sets.
    Returns:
        nums list[list[int]]: index of different sets.
    """
    nums = []
    i = 0
    for n_sam in n_sams:
        nums.append(seq[i:i+n_sam])
        i += n_sam

    return nums


if __name__ == '__main__':
    np.random.seed(1337)  # for reproducibility
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    PATH_ROOT = '../data/shipsEar/mix_separation'

    rates_set = [0.6, 0.2, 0.2]

    SCALER_DATA = 'max_one'
    # SCALER_DATA = 'or'
    SUB_SET_WAY = 'rand'
    # SUB_SET_WAY = 'order'

    # for feature original sample points
    PATH_CLASS = PathSourceRootSep(
        PATH_ROOT, form_src='wav', scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY)

    x_y_sets_create(PATH_CLASS, rates_set)

    z_data_create(PATH_CLASS, output_zero='zero')

    PATH_DATA_S = PATH_CLASS.path_source_root
    S_NAMES = json.load(open(os.path.join(PATH_DATA_S, 'dirname.json'), 'r'))['dirname']
    S_LIST = read_datas(os.path.join(PATH_DATA_S, 's_hdf5'), S_NAMES)

    N_SAMS = S_LIST[0].shape[0]
    if SUB_SET_WAY == 'rand':
        with open(os.path.join(PATH_DATA_S, 'randseq.pickle'), 'rb') as f_rb:
            nums_seq = pickle.load(f_rb)
    elif SUB_SET_WAY == 'order':
        nums_seq = list(range(N_SAMS))

    nums = subset_seq(nums_seq, [3055, 1018, 1020])

    PATH_DATA = PATH_CLASS.path_source
    z_sets_ns_create(S_LIST, nums, SCALER_DATA, PATH_DATA)

    logging.info('finished')
