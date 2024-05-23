# -*- coding: utf-8 -*-
"""
Created on Wed March 18 20:34:30 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
# pylint: disable=import-outside-toplevel, line-too-long, useless-object-inheritance

import os

from sklearn import preprocessing
import numpy as np
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data
import prepare_data_shipsear_recognition_mix_s0tos3 as m_pre_data_shipsear


def n_hot_labels(nsrc):
    """Return a mixed sources n_hot labels matrix with input number of nsrc.
    Args:
        nsrc (int): The number of sources.
    Returns:
        list[[int]]: a 2d list with shape 2**(nsrc-1) * (nsrc-1)
    Examples:
        Input 4, return 8*3 mix labels matrix.
        >>> print(n_hot_labels(4))
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]
    """
    import numpy as np  # pylint: disable=import-outside-toplevel, redefined-outer-name
    from itertools import combinations  # pylint: disable=import-outside-toplevel
    labels = []  # n_hot labels
    labels.extend(np.zeros((1, nsrc - 1), dtype=int).tolist())  # s0
    labels.extend(np.eye(nsrc - 1, dtype=int).tolist())  # s1tos3
    for i in range(2, nsrc, 1):
        # e.g. [(1,2)(1,3)(2,3)(1,2,3)]
        index_ci = list(combinations(range(nsrc - 1), i))
        n_ci = len(index_ci)
        labels_ci = np.zeros((n_ci, nsrc - 1), dtype=int)
        for j in range(n_ci):  # e.g. (1,2)
            for index_ci_j_k in index_ci[j]:
                labels_ci[j, index_ci_j_k] = 1
        labels.extend(labels_ci.tolist())
    return labels


def get_nums_rand_fname(sub_set_way, split_way=None, file_type='pickle'):
    """Get file name of nums.pickle.
    Args:
        sub_set_way (str) ['rand', 'order']: way of splitting data sets.
        split_way (str, optional) [None, 'split', 'select']: way of splitting data strips. Defaults to None.
        file_type (str, optional) ['pickle', 'json', None]: file type of nums. Defaults to 'pickle'.
    Returns:
        fname_nums (str): file name of nums.
    """
    fname_nums = f'nums_{sub_set_way}'
    if split_way in {'split', 'select'}:
        fname_nums = f'{fname_nums}_{split_way}'
    if file_type in {'pickle', 'json'}:
        fname_nums = f'{fname_nums}.{file_type}'
    return fname_nums


def compute_subset_nums(fname_frame_info, mode='select',
                        rates_set=(0.6, 0.2, 0.2), target_num_sets=(3055, 1018, 1020),
                        source_index_select=2):
    """Compute subset nums for select mode.
    Args:
        fname_frame_info (str): File name of the "frame_info.json" file.
        mode (str, optional): Split mode. Defaults to 'select'.
        rates_set (tuple, optional): Rates of datasets.. Defaults to (0.6, 0.2, 0.2).
        target_num_sets (tuple(int), optional): force sample numbers of each set, compatible to other ways. Defaults to
            (3055, 1018, 1020).
        source_index_select (int, optional): Index of the source which selected to split sources. Defaults to 2.
    Returns:
        nums_order_select (list[list[int]]): [subset][data] indexes of file clips.
    """
    frame_info = read_data(os.path.dirname(fname_frame_info), os.path.splitext(os.path.basename(fname_frame_info))[0],
                           form_src='json')
    # source_names = frame_info['source_name']
    num_frames = frame_info['number_frame']

    def clip_nums(num_frames, num_clip):
        """Abandon extra frames of source over target number.
        Args:
            num_frames (list[int]): numbers of frames of every file.
            num_clip (int): target number of overall frames.
        Returns:
            num_frames_clip (list[int]): numbers of frames of every file after processing.
        """
        num_frames_clip = list()
        sum_num = 0
        for num_i in num_frames:
            if sum_num + num_i < num_clip:
                num_frames_clip.append(num_i)
                sum_num += num_i
            else:
                num_frames_clip.append(num_clip-sum_num)
                break
        return num_frames_clip

    def split_num_sets(num_frames, rates_set):
        """Split frames to different datasets.
        Args:
            num_frames (list[int]): numbers of frames of every file.
            rates_set (list[float]): rates of datasets.
        Returns:
            num_sets (list[list[int]]): [set][file] numbers of frames of every set.
        """
        n_sets = len(rates_set)
        num_sets = [list() for _ in range(n_sets)]  # [set][file]
        num_sum_sets = list()  # [file]
        for i, num_frame_i in enumerate(num_frames):
            num_sum_i = 0
            for j in range(n_sets):
                num_i_j = int(num_frame_i * rates_set[j])
                num_sets[j].append(num_i_j)
                num_sum_i += num_i_j
            num_sum_sets.append(num_sum_i)

            index_set = 0
            while num_sum_sets[i] != num_frame_i:
                if num_sum_sets[i] < num_frame_i:
                    num_sets[index_set][i] = num_sets[index_set][i] + 1
                    num_sum_sets[i] = num_sum_sets[i] + 1
                else:
                    num_sets[index_set][i] = num_sets[index_set][i] - 1
                    num_sum_sets[i] = num_sum_sets[i] - 1

                index_set = index_set + 1 if index_set < n_sets else 0

        return num_sets

    def adjust_num_sets(num_sets, target_num_sets):
        """Adjust num_sets to follow target numbers of datasets.
        Args:
            num_sets (list[list[int]]): [set][file] numbers of frames of every set.
            target_num_sets (list[int]): [set] target numbers of datasets.
        Returns:
            num_sets (list[list[int]]): [set][file] numbers of frames of every set after adjusting.
        """
        while 1:
            num_sum_sets = [np.sum(num_set_i) for num_set_i in num_sets]
            if np.sum(np.equal(num_sum_sets, target_num_sets)) == len(target_num_sets):
                break
            for i in range(len(rates_set)-1):
                index_file = 0
                while np.sum(num_sets[i]) != target_num_sets[i]:
                    if np.sum(num_sets[i]) < target_num_sets[i]:
                        num_sets[i][index_file] = num_sets[i][index_file] + 1
                        num_sets[i+1][index_file] = num_sets[i+1][index_file] - 1
                    else:
                        if num_sets[i][index_file] > 1:
                            num_sets[i][index_file] = num_sets[i][index_file] - 1
                            num_sets[i+1][index_file] = num_sets[i+1][index_file] + 1
                    index_file = index_file + 1 if index_file < len(num_sets[i])-1 else 0
        return num_sets

    def convert_num_to_index(num_frames_sets):
        """Convert numbers of clips to indexes.
        Args:
            num_frames_sets (list[list[int]]): [set][file] numbers of frames of every set.
        Returns:
            indexes (list[list[int]]): [subset][data] indexes of file clips.
        """
        indexes = [list() for _ in range(len(num_frames_sets))]
        n_sets = len(num_frames_sets)
        n_files = len(num_frames_sets[0])
        istart = 0
        for j in range(n_files):
            for i in range(n_sets):
                num_i_j = num_frames_sets[i][j]
                indexes[i] += list(range(istart, istart+num_i_j))
                istart += num_i_j
        return indexes

    if mode == 'select':
        num_sum_sets = int(np.sum(target_num_sets))
        num_frames_clip = [clip_nums(num_frame_i, num_sum_sets) for num_frame_i in num_frames]
        num_frames_select = num_frames_clip[source_index_select]
        num_frames_sets = split_num_sets(num_frames_select, rates_set)
        num_frames_sets = adjust_num_sets(num_frames_sets, target_num_sets)
        nums_order_select = convert_num_to_index(num_frames_sets)
        return nums_order_select


def subset_nums_create(path_source_root, sub_set_way, rates, n_samples, n_sources, split_way=None,
                       target_num_sets=None, source_index_select=None,
                       fname_frame_info=None, path_seg_root=None, path_class=None):
    """Save sub_set nums.
    Args:
        path_source_root (str): path root where data is.
        sub_set_way (str): ['order', 'rand'] way to subset data.
        rates (list[float]): The rates of each sub dataset, e.g. train val test.
        n_samples (int): Number of samples.
        n_sources (int): Number of mix sources.
        split_way (str, optional) [None, 'split', 'select{i}']: way of splitting data strips. Defaults to None.
        target_num_sets (list[int], optional) : force sample numbers of each set, compatible to other ways.
            Defaults to None.
        source_index_select (int): Index of the source which selected to split sources.
        fname_frame_info (str): File name of the "frame_info.json" file.
        path_seg_root (str): Path where the "frame_info.json" file saved.
        path_class (object class PathSourceRoot): object of class to compute path.
    """
    import os  # pylint: disable=import-outside-toplevel, redefined-outer-name
    import pickle  # pylint: disable=redefined-outer-name
    import json  # pylint: disable=redefined-outer-name
    from prepare_data import Subsets
    from prepare_data import shuffle_sets
    rss1 = Subsets(rates, n_samples)
    # nums: 3D list [sourcei][subseti][numi]
    if sub_set_way == 'rand':
        with open(os.path.join(path_source_root, 'randseq.pickle'), 'wb') as f_wb:
            pickle.dump(rss1.get_randseq(), f_wb)
        with open(os.path.join(path_source_root, 'randseq.json'), 'w', encoding='utf-8') as f_w:
            json.dump({'data': rss1.get_randseq()}, f_w)

        if split_way is None:
            nums_source = rss1.randsubsetsnums(n_samples)
            nums = [nums_source for _ in range(n_sources)]
        elif split_way == 'split':
            nums_source = rss1.rand_subsets_nums_with_randseq(n_samples)
            nums = [nums_source for _ in range(n_sources)]

    elif sub_set_way == 'order':
        if split_way is None or split_way == 'split':
            nums_source = rss1.ordersubsetsnums(n_samples)
            nums = [nums_source for _ in range(n_sources)]
        elif split_way == 'select':
            if fname_frame_info is None:
                if path_seg_root is None:
                    path_seg_root = path_class.path_seg_root
                fname_frame_info = os.path.join(path_seg_root, 'frame_info.json')
            nums_source = compute_subset_nums(fname_frame_info, mode=split_way,
                                              rates_set=rates, target_num_sets=target_num_sets,
                                              source_index_select=source_index_select)
            nums = [nums_source for _ in range(n_sources)]

    # return 2D list [subseti][(sourcei, numi)]
    nums_rand = shuffle_sets(nums)
    fname_nums = get_nums_rand_fname(sub_set_way, split_way, None)
    with open(os.path.join(path_source_root, f'{fname_nums}.pickle'), 'wb') as f_wb:
        pickle.dump(nums_rand, f_wb)
    with open(os.path.join(path_source_root, f'{fname_nums}.json'), 'w', encoding='utf-8') as f_w:
        json.dump({'data': nums_rand}, f_w)


def subset_x(source_frames, nums_rand):
    """Sub_set feature datasets x.
    Args:
        source_frames (np.ndarray,shape==(n_sources, n_samples)+feature_shape): sources scaled.
        nums_rand (list[pair(int, int)]): [n_set](n_source, index), index of rand data.
    Returns:
        x_sets (list[np.ndarray,shape==(n_samples,)+feature_shape]): feature datasets.
    """
    import numpy as np  # pylint: disable=redefined-outer-name
    x_sets = []
    for nums_i in nums_rand:
        x_sets_i = []
        for pair_i in nums_i:
            x_sets_i.append(source_frames[pair_i[0]][pair_i[1]])
        x_sets.append(x_sets_i)
    return x_sets


def y_sets_create(nums_rand, y_labels, n_src):
    """Create label data y_sets.
    Args:
        nums_rand (list[pair(int, int)]): [n_set](n_source, index), index of rand data.
        y_labels (list[list[int]): [n_sources][n_src] label of mix sources.
        n_src (int): number of original sources.
    """
    import numpy as np  # pylint:disable=redefined-outer-name
    y_sets = []
    for pair_si in nums_rand:
        label_i = [y_labels[pair_i[0]] for pair_i in pair_si]
        y_sets.append(
            np.asarray(label_i, dtype=np.int32).reshape((-1, 1, n_src - 1)))
    return y_sets


class XsetSourceFrames(object):
    """Read and scaler data x_sets."""

    def __init__(self, path_source_root, dir_names, **kwargs):
        self._path_source_root = path_source_root
        self._dir_names = dir_names

        # Load data x_sets.
        if 'mode_read' in kwargs.keys():
            self._source_frames = np.asarray(
                m_pre_data_shipsear.read_datas(
                    os.path.join(self._path_source_root, 's_hdf5'),
                    self._dir_names, **{'mode': kwargs['mode_read']}), dtype=np.float32)
        else:
            self._source_frames = np.asarray(
                m_pre_data_shipsear.read_datas(
                    os.path.join(self._path_source_root, 's_hdf5'), self._dir_names), dtype=np.float32)

        self.n_sources = self._source_frames.shape[0]  # = nmixsources
        # number of samples per mixsource
        self.n_samples = self._source_frames.shape[1]
        self.feature_shape = self._source_frames.shape[2:]

    def get_source_frames(self):
        """Get the x_sets data _source_frames.
        Returns:
            self._source_frames (np.ndarray,shape==(n_sources, n_samples)+feature_shape): sources to scaler.
        """
        return self._source_frames

    def sourceframes_mm_create(self, feature_range=(0., 1.)):
        """Scaler data feature.
        Args:
            self._source_frames (np.ndarray,shape==(n_sources, n_samples)+feature_shape): sources to scaler.
            self.n_sources (int): number of the sources.
            self.n_samples (int): number of samples per source.
        Returns:
            self._sourceframes_mm (np.ndarray,shape==(n_sources, n_samples)+feature_shape): sources scaled.
        """
        scaler_mm = preprocessing.MinMaxScaler(feature_range=feature_range)  # to [0,1]
        # return 2D np.array [num][feature]
        self._sourceframes_mm = scaler_mm.fit_transform(  # pylint: disable=attribute-defined-outside-init
            self._source_frames.reshape(
                self.n_sources * self.n_samples, -1).transpose()).transpose()
        # return 3D np.array [n_sources][n_samples][feature]
        self._sourceframes_mm = self._sourceframes_mm.reshape(  # pylint: disable=attribute-defined-outside-init
            (self.n_sources, self.n_samples) + self.feature_shape)
        return self._sourceframes_mm


if __name__ == '__main__':
    import os
    import logging
    import numpy as np
    import pickle
    import json

    from file_operation import mkdir
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep

    np.random.seed(1337)  # for reproducibility
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO)

    def data_create(path_class, rates_set, **kwargs):  # pylint: disable=too-many-locals
        """Create X_train, X_val, X_test scaled data, and
            create Y_train, Y_val, Y_test data labels.
        Args:
            path_class (object class PathSourceRoot): object of class to compute path.
            rates_set (list[float]): rates of datasets.
        """
        path_source = path_class.path_source
        path_source_root = path_class.path_source_root
        scaler_data = path_class.get_scaler_data()
        sub_set_way = path_class.sub_set_way

        dir_names = json.load(
            open(os.path.join(path_source_root, 'dirname.json'), 'r'))['dirname']

        x_source_frames_class = XsetSourceFrames(path_source_root, dir_names, **kwargs)
        if scaler_data == 'or':
            x_source_frames = x_source_frames_class.get_source_frames()
        elif scaler_data == 'mm':
            x_source_frames = x_source_frames_class.sourceframes_mm_create()
        logging.info('x_source_frames read and scaler finished')

        n_samples = x_source_frames_class.n_samples
        n_sources = x_source_frames_class.n_sources
        fname_nums = get_nums_rand_fname(sub_set_way)
        if not os.path.isfile(os.path.join(path_source_root, fname_nums)):
            subset_nums_create(
                path_source_root, sub_set_way, rates_set, n_samples, n_sources)
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
        m_pre_data_shipsear.save_datas(
            dict(zip(['X_train', 'X_val', 'X_test'], x_sets)), path_source,
            **{'mode_batch': 'batch_h5py', 'batch_num': 20}  # only use this with small memory
        )
        m_pre_data_shipsear.save_datas(
            dict(zip(['Y_train', 'Y_val', 'Y_test'], y_sets)), path_source, dtype=np.int32)

    PATH_ROOT = '../data/shipsEar/mix_separation'

    RATES_SET = [0.6, 0.2, 0.2]  # rates of train, val, test set

    # for feature original sample points
    PATH_CLASS = PathSourceRootSep(
        PATH_ROOT, form_src='wav', scaler_data='or', sub_set_way='rand')
    # PATH_ROOT, form_src='wav', scaler_data='mm', sub_set_way='order')

    data_create(PATH_CLASS, RATES_SET)

    logging.info('data preprocessing finished')
