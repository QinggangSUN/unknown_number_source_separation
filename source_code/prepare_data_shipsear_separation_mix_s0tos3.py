# -*- coding: utf-8 -*-
"""
Created on Thu May  6 19:47:54 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""

# pylint: disable=no-member
import logging
import os

from prepare_data_shipsear_recognition_mix_s0tos3 import PathSourceRoot

logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.INFO)

_SR = 52734
_IS_MONO = True
_FRAME_LENGTH = 10547    # ~200ms 10546.800 000 000 001
_FRAME_SHIFT = 10547


def get_sr():
    """Return const global variable _SR.
    Returns:
        _SR (int): sample rate.
    """
    return _SR


class PathSourceRootSep(PathSourceRoot):
    """Path to find sources."""

    def __init__(self, path_root, **kwargs):
        super().__init__(path_root, **kwargs)

    def _set_path_mix_root(self, value):
        """Calculate the 'path_mix_root' property.
        Args:
            value (str): set the path_mix_root.
        """
        if value:
            self._path_mix_root = value
            return
        self._path_mix_root = os.path.join(self._get_path_seg_root(), 's0tos3', 'mix_1to3sep')


def name_mix_src(indexes):
    """Creat mix source name.
    Args:
        indexes (tuple(int)): indexes of the sources.
    Returns:
        name_str (str): the name of the mix source.
    """
    items = ['s']
    for k in indexes:  # e.g. (1,2)
        items.append('_')
        items.append(str(k))
    name_str = ''.join(items)
    return name_str


def create_snrs(distributed='uniform', low=None, high=None, num=1, n_src=1):
    """Randomly generate SNR between sources.
    Args:
        distributed (str, optional): distributed of the SNR. Defaults to 'uniform'.
        low (float, optional): min value (dB). Defaults to None.
        high (float, optional): max value (dB). Defaults to None.
        num (int, optional): number of the samples. Defaults to 1.
        n_src (int, optional): number of the SNR source pair. Defaults to 1.
    Returns:
        snrs (np.ndarray): randomly generated SNR.
    """
    import numpy as np

    if distributed == 'uniform':
        snrs = np.random.uniform(low, high, (n_src, num))
    elif distributed == 'gauss':
        mu = np.average(low, high)
        sigma = (high - low) ** 2
        snrs = np.random.normal(mu, sigma, (n_src, num))

    return snrs


def data_mixwav_create(path_class, snrs=None, test_few=False):  # pylint: disable=too-many-locals
    """Create and save mixed sources original sampling point wavmat,
        you may run this function only onece.
    Args:
        path_class (object class PathSourceRootSep): object of class to compute path.
        snrs (np.ndarray, optional): randomly generated SNR. Defaults to None.
        test_few (bool, optional): whether return few datas for test. Defaults to False.
    """
    import gc
    import json
    import numpy as np
    from itertools import combinations
    from file_operation import mkdir, mycopyfile
    from prepare_data import balancesets, mix_add_snr
    from prepare_data_shipsear_recognition_mix_s0tos3 import read_datas, save_datas

    path_seg_root = path_class.path_seg_root
    path_seg = path_class.get_path_seg()
    # path_mix_root = path_class.path_mix_root
    # mkdir(path_mix_root)
    path_source_root = path_class.path_source_root
    mkdir(path_source_root)

    # read sources
    mycopyfile(os.path.join(path_seg_root, 'dirname.json'),
               os.path.join(path_source_root, 'dirname.json'))
    src_names = json.load(open(os.path.join(path_source_root, 'dirname.json'), 'r'))['dirname'][:4]

    source_frames = read_datas(path_seg, src_names)

    # balance sources
    source_frames = balancesets(source_frames)

    if test_few:
        source_frames = [si[:12] for si in source_frames]

    n_src = len(source_frames)
    n_sams = source_frames[0].shape[0]

    if snrs is None:
        snrs = create_snrs('uniform', -5, 5, n_sams, n_src-2)

    for j in range(n_sams):
        sources = [source_frames[i][j] for i in range(1, n_src)]
        _, sources_standard = mix_add_snr(sources, snrs[:, j].tolist())
        for k, source_standard_k in enumerate(sources_standard):
            source_frames[k+1][j] = source_standard_k

        del sources
        del sources_standard
        gc.collect()

    path_source_out = os.path.join(path_source_root, 's_hdf5')
    mkdir(path_source_out)
    dir_names = []
    data_list = []
    for i in range(1, n_src):  # mix 1 to 3 without 0
        if i == 1:
            index_ci = tuple(combinations(range(n_src), i))
            # e.g. ((0),(1),(2),(3)) ((1,2),(1,3),(2,3))
        else:
            index_ci = tuple(combinations(range(1, n_src), i))
            # e.g. ((0),(1),(2),(3)) ((1,2),(1,3),(2,3))

        for index_ci_j in index_ci:  # e.g. (1,2)
            dir_names.append(name_mix_src(index_ci_j))
            if len(index_ci_j) == 1:
                mix_cij_arr = source_frames[index_ci_j[0]]
            else:
                mix_cij_arr = np.sum([source_frames[k] for k in index_ci_j], axis=0)
            mix_cij_arr = mix_cij_arr.reshape(
                mix_cij_arr.shape[0:1]+(1,)+mix_cij_arr.shape[-1:])
            data_list.append(mix_cij_arr)
    with open(os.path.join(path_source_root, 'dirname.json'), 'w', encoding='utf-8') as f_w:
        json.dump({'dirname': dir_names}, f_w)

    save_datas(dict(zip(dir_names, data_list)), path_source_out)


if __name__ == '__main__':
    PATH_ROOT = '../data/shipsEar/mix_separation'

    PATH_CLASS = PathSourceRootSep(PATH_ROOT, form_src='wav')
    data_mixwav_create(PATH_CLASS, test_few=False)

    logging.info('finished')
