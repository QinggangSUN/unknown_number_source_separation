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

from error import Error, ParameterError
from file_operation import mkdir, mycopyfile
from prepare_data_shipsear_recognition_mix_s0tos3 import read_datas, save_datas, save_datas_continue, SpectrumCreate
from recognition_mix_shipsear_s0tos3_preprocess import get_nums_rand_fname, n_hot_labels, subset_nums_create, subset_x
from recognition_mix_shipsear_s0tos3_preprocess import y_sets_create


# pylint: disable=no-member


def sourceframes_mo_create(x_source_frames, scaler_save=None):
    """Scaler sources to max abs amplitude in each pair mixed data to one.
    Args:
        x_source_frames (list[np.ndarray, shape==(n_samples,1,fl)]): [n_sources] sources to be scaled.
        scaler_save (str): path to save scaler datas.
    Returns:
        x_source_frames_standard (list[np.ndarray, shape==(n_samples,1,fl)]): sources after scaled.
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


def min_max_scaler_samples(source_data, feature_range=(0., 1.)):
    """Scaler data set to [0, 1].
    Args:
        source_data (np.ndarray, shape==(n_samples,)+feature_shape): source data.
        feature_range (tuple, optional): Desired range of transformed data. Defaults to (0., 1.).
    Returns:
        source_data_mm (np.ndarray, shape==(n_samples,)+feature_shape): data transformed.
        scaler_mm (obj, MinMaxScaler): object of MinMaxScaler, for get other stuff.
    """
    scaler_mm = preprocessing.MinMaxScaler(feature_range=feature_range)
    num_samples = source_data.shape[0]
    feature_shape = source_data.shape[1:]
    source_data_mm = scaler_mm.fit_transform(source_data.reshape(num_samples, -1).transpose()).transpose()
    source_data_mm = source_data_mm.reshape((num_samples,) + feature_shape)
    return source_data_mm, scaler_mm


def max_one_scaler_samples(source_data):
    """Scaler data set to [0, 1].
    Args:
        source_data (np.ndarray, shape==(n_samples,)+feature_shape): source data.
    Returns:
        source_data_mo (np.ndarray, shape==(n_samples,)+feature_shape): data transformed.
        scaler_mo (np.ndarray, shape==(n_samples,)+feature_shape): scale factors.
    """
    scaler_mo = np.empty((source_data.shape[0],), dtype=np.float32)
    source_data_mo = np.empty(source_data.shape, dtype=np.float32)
    feature_shape = source_data.shape[1:]
    for i, data_i in enumerate(source_data):
        data_i = data_i.flatten()
        max_am_i = np.max(np.abs(data_i))
        scaler_mo[i] = max_am_i
        source_data_mo_i = data_i / max_am_i
        source_data_mo[i] = source_data_mo_i.reshape(feature_shape)
    return source_data_mo, scaler_mo


def x_sets_mm_create(x_sets, feature_range=(0., 1.)):
    """Scaler data sets to [0, 1] on all sets.
    Args:
        x_sets (list[np.ndarray, shape==(n_samples,)+feature_shape]): sets of the features.
        feature_range (tuple(float, float)): scale data to [min, max].
    """

    n_samples = [x_set_i.shape[0] for x_set_i in x_sets]
    x_shapes = [tuple(x_set_i.shape) for x_set_i in x_sets]

    x_source_frames = np.vstack(x_sets)
    # scaler_mm = preprocessing.MinMaxScaler(feature_range=feature_range)  # to [0,1]
    # n_sample_all = x_source_frames.shape[0]
    # x_source_frames_mm = scaler_mm.fit_transform(x_source_frames.reshape(n_sample_all, -1).transpose()).transpose()
    x_source_frames_mm, _ = min_max_scaler_samples(x_source_frames, feature_range=feature_range)

    n_indexes = [sum(n_samples[:i]) for i in range(1, len(n_samples))]
    x_sets_mm = np.split(x_source_frames_mm, n_indexes)
    x_sets_mm = [x_set_i.reshape(x_shape_i) for x_set_i, x_shape_i in zip(x_sets_mm, x_shapes)]

    return x_sets_mm


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
        x_source_frames (list[np.ndarray, shape==(n_sams,1,fl)]): [n_sources] sources to be scaled.
        output_zero (str) {'zero','epsi'}: output data when no specific type signal in the mixed signal.
        path_save (str): where to save z sets.
        epsi (float, optional): value when output all epsi. Defaults to 1e-6.
        save_batch (int, optional): number of data each batch to save. Defaults to None.
    """
    if save_batch is None:
        for si, (nums_i, name_i) in enumerate(zip(nums, ['Z_train', 'Z_val', 'Z_test'])):
            if output_zero == 'zero':
                z_si = np.zeros(  # (nsamples, nsrc, feature-shape)
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
                    z_si_pi = np.zeros(  # (1, nsrc, feature-shape)
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
    """Create X_train, X_val, X_test scaled data, and create Y_train, Y_val, Y_test data labels.
    Args:
        path_class (object class PathSourceSep): object of class to compute path.
        rates_set (list[float]): rates of datasets.
    """
    path_source = path_class.path_source
    path_source_root = path_class.path_source_root
    scaler_data = path_class.get_scaler_data()
    sub_set_way = path_class.sub_set_way
    split_way = path_class.get_split_way()

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

    fname_nums = get_nums_rand_fname(sub_set_way, split_way)
    if not os.path.isfile(os.path.join(path_source_root, fname_nums)):
        subset_nums_create(
            path_source_root, sub_set_way, rates_set, n_samples, n_sources, split_way=split_way, path_class=path_class,
            **kwargs)
    with open(os.path.join(path_source_root, fname_nums), 'rb') as f_rb:
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
    sub_set_way = path_class.sub_set_way
    split_way = path_class.get_split_way()

    x_sets = read_datas(path_source, ['X_train', 'X_val', 'X_test'])
    logging.info('data x_sets load finished')
    x_sets_shape = [x_i.shape for x_i in x_sets]

    fname_nums = get_nums_rand_fname(sub_set_way, split_way)
    with open(os.path.join(path_source_root, fname_nums), 'rb') as f_rb:
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


def standar_feature(data, dim_target, min_shape=32):
    """Standardize data shape for network input and output.
    Args:
        data (np.ndarray): feature before standardizing.
        dim_target (int): dimension of the feature after standardizing.
        min_shape (int, optional): minimum value of all the dimensions of the feature. Defaults to 32.
    Returns:
        data_standard (np.ndarray): feature after standardizing.
    """
    if dim_target == 1:  # only for 1D network input
        data_standard = np.asarray(data).transpose((0, 2, 1))  # (n_samples, fl, 1)
    elif dim_target == 2:  # only for 2D network input
        data_standard = np.expand_dims(np.asarray(data), -1)  # (n_samples, t, fl, 1)

    if min_shape > 0 and dim_target == 2:  # only for 2D input padding, input size must >= (32, 32, 1)
        d1 = data_standard.shape[1]  # (n_samples, t, fl, 1)
        d2 = data_standard.shape[2]

        if d1 == 10546 and d2 == 3:  # win_length==4, hop_length==1
            data_standard = np.pad(data_standard,
                                   ((0, 0), (0, 1), (0, 0), (0, 0)),
                                   'constant', constant_values=(0, 0))
            data_standard = data_standard.reshape((data_standard.shape[0], 199, 53 * 3, 1))
        elif d1 == 5272 and d2 == 5:  # win_length==8, hop_length==2
            data_standard = data_standard.reshape((data_standard.shape[0], 659, 8 * 5, 1))
        elif d1 == 3514 and d2 == 7:  # win_length==12, hop_length==3
            data_standard = data_standard.reshape((data_standard.shape[0], 251, 14 * 7, 1))
        elif d1 == 2635 and d2 == 9:  # win_length==16, hop_length==4
            data_standard = data_standard.reshape((data_standard.shape[0], 31 * 5, 17 * 9, 1))
        elif d1 == 2108 and d2 == 11:  # win_length==20, hop_length==5
            data_standard = data_standard.reshape((data_standard.shape[0], 31 * 4, 17 * 11, 1))
        elif d1 == 810 and d2 == 27:  # win_length==52, hop_length==13
            data_standard = data_standard.reshape((data_standard.shape[0], 6 * 27, 5 * 27, 1))
        else:
            if d1 < min_shape:
                data_standard = np.pad(data_standard,
                                       ((0, 0), (0, min_shape - d1), (0, 0), (0, 0)),
                                       'constant', constant_values=(0, 0))
            if d2 < min_shape:
                data_standard = np.pad(data_standard,
                                       ((0, 0), (0, 0), (0, min_shape - d2), (0, 0)),
                                       'constant', constant_values=(0, 0))
    return data_standard


def data_feature_create(path_class_in, path_class_out, batch_save=0, **kwargs):
    """Create and save feature sources_frames.
    Args:
        path_class_in (object class PathSourceRootSep): object of class to compute path.
        path_class_out (object class PathSourceRootSep): object of class to compute path.
        batch_save (int, optional): each batch save batch_save samples. Defaults to 0 means save all samples.
    """
    path_source_in = path_class_in.path_source
    path_source_out = path_class_out.path_source
    mkdir(path_source_out)

    y_filenames = ['Y_train', 'Y_val', 'Y_test'] if 'y_filenames' not in kwargs.keys() else kwargs['y_filenames']
    y_filetype = '.hdf5' if 'y_filetype' not in kwargs.keys() else kwargs['y_filetype']
    for y_filename_i in y_filenames:
        if not os.path.isfile(os.path.join(path_source_out, y_filename_i + y_filetype)):
            mycopyfile(os.path.join(path_source_in, y_filename_i + y_filetype),
                       os.path.join(path_source_out, y_filename_i + y_filetype))

    x_filenames = ['X_train', 'X_val', 'X_test'] if 'x_filenames' not in kwargs.keys() else kwargs['x_filenames']
    if 'mode_read' in kwargs.keys():
        # from feature sources_wavmat to another feature
        sources_wavmat = read_datas(path_source_in, x_filenames, **{'mode': kwargs['mode_read']})
    else:
        sources_wavmat = read_datas(path_source_in, x_filenames)

    obj_spectrum_create = SpectrumCreate()

    form_src = path_class_out.get_form_src()
    dtype = np.complex64 if form_src in {'phase_spectrum'} else np.float32

    scaler_data = path_class_out.get_scaler_data()

    if batch_save == 0:
        # each set save a file
        x_sourceframe_sets = []
        for sources_wavmat_i in sources_wavmat:
            x_sourceframe_set_i = obj_spectrum_create.feature_create(sources_wavmat_i, path_class_out, form_src,
                                                                     **kwargs)
            x_sourceframe_sets.append(x_sourceframe_set_i)

        if scaler_data == 'mm':
            x_sourceframe_sets = x_sets_mm_create(x_sourceframe_sets)

        for set_i, x_sourceframe_set_i in enumerate(x_sourceframe_sets):
            save_datas(dict(zip([x_filenames[set_i]], [x_sourceframe_set_i])), path_source_out, dtype=dtype)
    else:
        mode_batch = 'batch_h5py' if 'mode_save' not in kwargs.keys() else kwargs['mode_save']
        dim_target = 2 if 'dim_target' not in kwargs.keys() else kwargs['dim_target']
        is_standard_feature = False if 'is_standard_feature' not in kwargs.keys() else kwargs['is_standard_feature']
        if scaler_data == 'or':
            for set_i, sources_wavmat_i in enumerate(sources_wavmat):
                for j in range(0, sources_wavmat_i.shape[0], batch_save):
                    if j + batch_save > sources_wavmat_i.shape[0]:
                        sources_i_j = obj_spectrum_create.feature_create(sources_wavmat_i[j:], path_class_out, form_src,
                                                                         **kwargs)
                    else:
                        sources_i_j = obj_spectrum_create.feature_create(sources_wavmat_i[j:j + batch_save],
                                                                         path_class_out, form_src, **kwargs)
                    if is_standard_feature:
                        sources_i_j = standar_feature(sources_i_j, dim_target)
                    save_datas_continue(dict(zip([x_filenames[set_i]], [sources_i_j])), path_source_out,
                                        **{'mode_batch': mode_batch, 'batch_num': batch_save, 'dtype': dtype})


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
        nums (list[list[int]]): index of different sets. [set][sams]
    """
    nums = []
    i = 0
    for n_sam in n_sams:
        nums.append(seq[i:i + n_sam])
        i += n_sam

    return nums


def load_randseq(sub_set_way, path_data_s, n_sams=None):
    """Load randseq from file.
    Args:
        sub_set_way (str): way of split data set.
        path_data_s (str): path where the file "randseq.pickle" saved.
        n_sams (int, optional): number of the samples. Defaults to None.
    Returns:
        randseq (np.ndarray): 1-D indexes.
    """
    if sub_set_way == 'rand':
        with open(os.path.join(path_data_s, 'randseq.pickle'), 'rb') as f_rb:
            randseq = pickle.load(f_rb)
    elif sub_set_way == 'order':
        randseq = list(range(n_sams))
    else:
        randseq = None
    return randseq


if __name__ == '__main__':
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep

    np.random.seed(1337)  # for reproducibility
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

    PATH_DATA_ROOT = '../data/shipsEar/mix_separation'

    rates_set = [0.6, 0.2, 0.2]

    SCALER_DATA = 'max_one'
    # SCALER_DATA = 'or'
    SUB_SET_WAY = 'rand'
    # SUB_SET_WAY = 'order'
    # SPLIT_WAY = None
    SPLIT_WAY = 'split'
    # for feature original sample points
    PATH_CLASS = PathSourceRootSep(PATH_DATA_ROOT, form_src='wav',
                                   scaler_data=SCALER_DATA, sub_set_way=SUB_SET_WAY, split_way=SPLIT_WAY)

    x_y_sets_create(PATH_CLASS, rates_set)

    z_data_create(PATH_CLASS, output_zero='zero')

    # For generate Z_0 to Z_7 data set
    PATH_DATA_S = PATH_CLASS.path_source_root
    S_NAMES = json.load(open(os.path.join(PATH_DATA_S, 'dirname.json'), 'r'))['dirname']
    S_LIST = read_datas(os.path.join(PATH_DATA_S, 's_hdf5'), S_NAMES)

    randseq = load_randseq(SUB_SET_WAY, PATH_DATA_S, n_sams=S_LIST[0].shape[0])
    index_seq_known_number = subset_seq(randseq, [3055, 1018, 1020])

    PATH_DATA = PATH_CLASS.path_source
    z_sets_ns_create(S_LIST, index_seq_known_number, SCALER_DATA, PATH_DATA)

    logging.info('finished')
