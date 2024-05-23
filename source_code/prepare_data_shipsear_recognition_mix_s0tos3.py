# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 20:34:30 2018

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

    balance sources first, then full combination mix without s0
"""
# pylint: disable=attribute-defined-outside-init, import-outside-toplevel, invalid-name
# pylint: disable=line-too-long, no-member, redefined-outer-name, reimported
# pylint: disable=too-many-arguments, too-many-branches, too-many-lines, too-many-locals, useless-object-inheritance,
# pylint: disable=too-many-statements

import logging
import os

import librosa
import numpy as np

from error import ParameterError
from feature_extract import feature_extract

_SR = 52734
_IS_MONO = True
_FRAME_LENGTH = 10547  # ~200ms 10546.800 000 000 001
_FRAME_SHIFT = 10547


def get_sr():
    """Return const global variable _SR."""
    return _SR


def get_mono():
    """Return const global variable _IS_MONO."""
    return _IS_MONO


def get_fl():
    """Return const global variable _FRAME_LENGTH."""
    return _FRAME_LENGTH


def get_fs():
    """Return const global variable _FRAME_SHIFT."""
    return _FRAME_SHIFT


# output mix features
_WIN_LENGTH = 1582  # 52734*0.03 = 1582.02
_HOP_LENGTH = 396  # 52734*0.03/4 = 395.505


def get_win_length():
    """Return const global variable _WIN_LENGTH."""
    return _WIN_LENGTH


def get_hop_length():
    """Return const global variable _HOP_LENGTH."""
    return _HOP_LENGTH


class PathSourceRoot(object):  # pylint: disable=too-many-instance-attributes
    """Path to find sources."""

    def __init__(self, path_root, **kwargs):
        self._path_root = path_root
        self._split_way = None

        for key, value in kwargs.items():
            if key == 'path_seg_root':
                self._path_seg_root = value
            elif key == 'frame_length':
                self._frame_length = value
            elif key == 'frame_shift':
                self._frame_shift = value
            elif key == 'path_mix_root':
                self._path_mix_root = value
            elif key == 'win_length':
                self._win_length = value
            elif key == 'hop_length':
                self._hop_length = value
            elif key == 'form_src':
                self._form_src = value
            elif key == 'sub_set_way':
                self.sub_set_way = value
            elif key == 'split_way':
                self._split_way = value
            elif key == 'scaler_data':
                self._scaler_data = value
            elif key == 'path_source':
                self._path_source = value
            elif key == 'n_mels':
                self._n_mels = value
            elif key == 'n_mfcc':
                self._n_mfcc = value
            elif key == 'high':
                self._high = value
            elif key == 'low':
                self._low = value
            elif key == 'cutoff':
                self._cutoff = value
            else:
                raise ParameterError('kwargs key invalid')

    def get_path_root(self):
        """Get read-only _path_root."""
        return self._path_root

    def get_path_raw(self):
        """Get path_raw from _path_root."""
        return os.path.join(self._path_root, 'raw')

    def _set_path_seg_root(self, value):
        """Calculate the 'path_seg_root' property."""
        if value:
            self._path_seg_root = value
            return
        if hasattr(self, '_frame_length'):
            frame_length = self._frame_length
        else:
            frame_length = get_fl()
        if hasattr(self, '_frame_shift'):
            frame_shift = self._frame_shift
        else:
            frame_shift = get_fs()
        self._path_seg_root = os.path.join(self._path_root, f'{frame_length}_{frame_shift}')

    def _get_path_seg_root(self):
        """Indirect accessor for 'path_seg_root' property."""
        if not hasattr(self, '_path_seg_root'):
            self._set_path_seg_root(None)
        return self._path_seg_root

    def get_path_seg_root(self):
        """Get the 'path_seg_root' property.
        Examples:
            example:
                >>> PATH_CLASS = PathSourceRoot('.')
                >>> print(PATH_CLASS.path_seg_root)
                ./10547_10547
        """
        return self._get_path_seg_root()

    def set_path_seg_root(self, value=None):
        """Set path_seg_root.
        Examples:
            example:
                >>> PATH_CLASS = PathSourceRoot('.')
                >>> PATH_CLASS.path_seg_root = './seg_root'
                >>> print(PATH_CLASS.path_seg_root)
                ./seg_root
        """
        return self._set_path_seg_root(value)

    path_seg_root = property(get_path_seg_root, set_path_seg_root)

    def get_path_seg(self):
        """Get path_seg from _path_seg_root."""
        return os.path.join(self._path_seg_root, 'wavhdf5')

    def _set_path_mix_root(self, value):
        """Calculate the 'path_mix_root' property."""
        if value:
            self._path_mix_root = value
            return
        self._path_mix_root = os.path.join(self._get_path_seg_root(), 's0tos3', 'mix_1to3')

    def _get_path_mix_root(self):
        """Indirect accessor for 'path_mix_root' property."""
        if not hasattr(self, '_path_mix_root'):
            self._set_path_mix_root(None)
        return self._path_mix_root

    def get_path_mix_root(self):
        """Get the 'path_mix_root' property.
        Examples:
            example:
                >>> PATH_CLASS = PathSourceRoot('.')
                >>> print(PATH_CLASS.path_mix_root)
                ./10547_10547/s0tos3/mix_1to3
            example:
                >>> PATH_CLASS = PathSourceRoot('.')
                >>> PATH_CLASS.path_seg_root = './seg_root'
                >>> print(PATH_CLASS.path_mix_root)
                ./seg_root/s0tos3/mix_1to3
        """
        return self._get_path_mix_root()

    def set_path_mix_root(self, value=None):
        """Set path_mix_root.
        Examples:
            example:
                >>> PATH_CLASS = PathSourceRoot('.')
                >>> PATH_CLASS.path_mix_root = './mix_root'
                >>> print(PATH_CLASS.path_mix_root)
                ./mix_root
            example:
                >>> PATH_CLASS = PathSourceRoot('.')
                >>> PATH_CLASS.path_mix_root = './mix_root'
                >>> PATH_CLASS.path_seg_root = './seg_root'
                >>> print(PATH_CLASS.path_mix_root)
                ./mix_root
        """
        self._set_path_mix_root(value)

    path_mix_root = property(get_path_mix_root, set_path_mix_root)

    def _set_path_source_root(self, form_src, **kwargs):
        """Calculate the 'path_source_root' property."""
        if form_src:
            self._form_src = form_src
        if not hasattr(self, '_form_src'):
            raise ParameterError(
                'Has to create _form_src by __init__ or _set_path_source')

        if self._form_src == 'wav':
            self._path_source_root = os.path.join(self._get_path_mix_root(
            ), 'wavmat')  # pylint: disable=attribute-defined-outside-init
        elif self._form_src in {'magspectrum', 'angspectrum', 'phase_spectrum', 'realspectrum', 'imgspectrum'}:
            if not hasattr(self, '_win_length'):
                self._win_length = get_win_length()
            if not hasattr(self, '_hop_length'):
                self._hop_length = get_hop_length()
            self._path_source_root = os.path.join(
                self._get_path_mix_root(),
                f'{self._form_src}_{self._win_length}_{self._hop_length}')  # pylint: disable=attribute-defined-outside-init
        elif self._form_src == 'logmelspectrum':
            if not hasattr(self, '_win_length'):
                self._win_length = get_win_length()
            if not hasattr(self, '_hop_length'):
                self._hop_length = get_hop_length()
            if not hasattr(self, '_n_mels'):
                raise ParameterError('need para n_mels')
            self._path_source_root = os.path.join(
                self._get_path_mix_root(),
                f'logmelspectrum_{self._win_length}_{self._hop_length}_{self._n_mels}')  # pylint: disable=attribute-defined-outside-init
        elif self._form_src == 'mfcc':
            if not hasattr(self, '_win_length'):
                self._win_length = get_win_length()
            if not hasattr(self, '_hop_length'):
                self._hop_length = get_hop_length()
            if not hasattr(self, '_n_mels'):
                raise ParameterError('need para n_mels')
            if not hasattr(self, '_n_mfcc'):
                raise ParameterError('need para n_mfcc')
            self._path_source_root = os.path.join(
                self._get_path_mix_root(), f'mfcc_{self._win_length}_{self._hop_length}_{self._n_mels}_{self._n_mfcc}')
        elif self._form_src == 'demon':
            if not hasattr(self, '_high'):
                raise ParameterError('need para cutoff')
            if not hasattr(self, '_low'):
                raise ParameterError('need para cutoff')
            if not hasattr(self, '_cutoff'):
                raise ParameterError('need para cutoff')
            self._path_source_root = os.path.join(
                self._get_path_mix_root(), f'demon_{int(self._high)}_{int(self._low)}_{int(self._cutoff)}')
        else:
            raise ParameterError('form_src invalid')

    def _get_path_source_root(self):
        """Indirect accessor for 'path_source_root' property."""
        if not hasattr(self, '_path_source_root'):
            self._set_path_source_root(None)
        return self._path_source_root

    def get_path_source_root(self):
        """Get the 'path_source_root' property.
        Examples:
            example:
                >>> PATH_CLASS = PathSourceRoot('.', form_src='wav')
                >>> print(PATH_CLASS.path_source_root)
                ./10547_10547/s0tos3/mix_1to3/wavmat
            example:
                >>> PATH_CLASS = PathSourceRoot('.', form_src='wav')
                >>> PATH_CLASS.path_mix_root = './mix_root'
                >>> print(PATH_CLASS.path_source_root)
                ./mix_root/wavmat
            example:
                >>> PATH_CLASS = PathSourceRoot('.', form_src='wav')
                >>> PATH_CLASS.path_seg_root = './seg_root'
                >>> print(PATH_CLASS.path_source_root)
                ./seg_root/s0tos3/mix_1to3/wavmat
        """
        return self._get_path_source_root()

    def set_path_source_root(self, value=None):
        """Set path_source_root.
        Examples:
            example:
                >>> PATH_CLASS = PathSourceRoot('.', form_src='wav')
                >>> PATH_CLASS.path_source_root = './src_root'
                >>> print(PATH_CLASS.path_source_root)
                ParameterError: form_src invaild
                (directly give path_source_root banded)
            example:
                >>> PATH_CLASS = PathSourceRoot('.', form_src='magspectrum')
                >>> PATH_CLASS.path_source_root = 'wav'
                >>> print(PATH_CLASS.path_source_root)
                ./10547_10547/s0tos3/mix_1to3/wavmat
            example:
                >>> PATH_CLASS = PathSourceRoot('.', form_src='magspectrum')
                >>> PATH_CLASS.path_source_root = 'wav'
                >>> PATH_CLASS.path_mix_root = './mix_root'
                >>> print(PATH_CLASS.path_source_root)
                ./10547_10547/s0tos3/mix_1to3/wavmat
        """
        self._set_path_source_root(value)

    path_source_root = property(get_path_source_root, set_path_source_root)

    def get_win_length(self):
        """Get _win_length."""
        return self._win_length

    def get_hop_length(self):
        """Get _hop_length."""
        return self._hop_length

    def _set_path_source(self, value, form_src, scaler_data, sub_set_way, split_way=None):
        """Calculate the 'path_source' property."""
        if value:
            self._path_source = value
            return

        if not hasattr(self, '_path_source_root'):
            self._set_path_source_root(form_src)

        if scaler_data:
            self._scaler_data = scaler_data
        if not hasattr(self, '_scaler_data'):
            raise ParameterError(
                'Has to create _scaler_data by __init__ or _set_path_source')

        if sub_set_way:
            self.sub_set_way = sub_set_way
        if not hasattr(self, 'sub_set_way'):
            raise ParameterError(
                'Has to create sub_set_way by __init__ or _set_path_source')

        if split_way:
            self._split_way = split_way

        if self._scaler_data == 'mm':
            file_name = f'min_max_scaler_{self.sub_set_way}'
        elif self._scaler_data == 'or':
            file_name = f'original_{self.sub_set_way}'
        elif self._scaler_data == 'max_one':
            file_name = f'max_one_{self.sub_set_way}'
        else:
            raise ParameterError('scaler_data invalid')

        if self._split_way:
            file_name = f'{file_name}_{self._split_way}'

        self._path_source = os.path.join(self._path_source_root, file_name)

    def _get_path_source(self):
        """Indirect accessor for 'path_source' property."""
        if not hasattr(self, '_path_source'):
            self._set_path_source(None, None, None, None, None)
        return self._path_source

    def get_path_source(self):
        """Get the 'path_source' property.
        Examples:
            example:
                >>> PATH_CLASS = PathSourceRoot(
                '.', form_src='wav', scaler_data='or', sub_set_way='rand')
                >>> print(PATH_CLASS.path_source)
                ./10547_10547/s0tos3/mix_1to3/wavmat/original_rand
            example:
                >>> PATH_CLASS = PathSourceRoot(
                '.', form_src='wav', scaler_data='or', sub_set_way='rand',
                frame_length=100, frame_shift=50)
                >>> print(PATH_CLASS.path_source)
                ./100_50/s0tos3/mix_1to3/wavmat/original_rand
            example:
                >>> PATH_CLASS = PathSourceRoot(
                '.', form_src='wav', scaler_data='or', sub_set_way='rand', split_way='split')
                >>> print(PATH_CLASS.path_source)
                ./10547_10547/s0tos3/mix_1to3/wavmat/original_rand_split
        """
        return self._get_path_source()

    def set_path_source(self, args):
        """Set path_source.
        Examples:
            example:
                >>> PATH_CLASS = PathSourceRoot('.')
                >>> PATH_CLASS.path_source = None, 'wav', 'or', 'order'
                >>> print(PATH_CLASS.path_source)
                ./10547_10547/s0tos3/mix_1to3/wavmat/original_order
            example:
                >>> PATH_CLASS = PathSourceRoot('./root')
                >>> PATH_CLASS.path_source = './path_source', None, None, None
                >>> print(PATH_CLASS.path_source)
                ./path_source
            example:
                >>> PATH_CLASS = PathSourceRoot('./root')
                >>> PATH_CLASS.path_mix_root = './mix_root'
                >>> PATH_CLASS.path_source = None, 'wav', 'or', 'order'
                >>> print(PATH_CLASS.path_source)
                ./mix_root/wavmat/original_order
            example:
                >>> PATH_CLASS = PathSourceRoot('./root')
                >>> PATH_CLASS.path_source_root = './src_root'
                >>> PATH_CLASS.path_source = None, 'wav', 'or', 'order'
                >>> print(PATH_CLASS.path_source)
                ParameterError: form_src invaild
            example:
                >>> PATH_CLASS = PathSourceRoot('./root')
                >>> PATH_CLASS.path_source_root = 'wav'
                >>> PATH_CLASS.path_source = None, 'magspectrum', 'or', 'order'
                >>> print(PATH_CLASS.path_source)
                ./root/10547_10547/s0tos3/mix_1to3/wavmat/original_order
        """
        try:
            value, form_src, scaler_data, sub_set_way, split_way = args
        except ValueError:
            raise ParameterError("Pass an iterable with five items")
        else:
            self._set_path_source(value, form_src, scaler_data, sub_set_way, split_way)

    path_source = property(get_path_source, set_path_source)

    def get_scaler_data(self):
        """Get read-only _scaler_data."""
        return self._scaler_data

    def get_form_src(self):
        """Get read-only _form_src."""
        return self._form_src

    def get_split_way(self):
        """Get read-only _form_src."""
        return self._split_way


def read_source(path, file_names, form_src='hdf5', data_type=None):
    """Read data file_names [file_names] from [path]."""
    import json
    import os
    import logging
    import numpy as np
    import h5py
    import scipy.io as sio
    logging.warning("DeprecationWarning: The 'read_source' function is deprecated, use 'read_datas' instead")

    def str_remove_end(file_names, file_type):
        """Remove files' name extension.
        Args:
            file_names (list[str]): list of file names.
            file_type (str): file name extension.
        Returns:
            file_names_small (list[str]): list of file names without extension.
        """
        logging_debug = False
        file_names_small = []
        for name_i in file_names:
            if name_i.endswith('.' + file_type):
                logging_debug = True
                file_names_small.append(name_i[:-len('.' + file_type)])
            else:
                file_names_small.append(name_i)
        if logging_debug:
            logging.debug('Remove file names extension.')
        return file_names_small

    if form_src == 'hdf5':
        if data_type:
            logging.warning('ignore data_type')
        file_names = str_remove_end(file_names, form_src)
        source_frames = [
            h5py.File(os.path.join(path, f'{name_i.rstrip()}.{form_src}'), 'r')['data'] for name_i in file_names]
    elif form_src == 'bin':
        if data_type is None:
            data_type = np.float32
        source_frames = [np.fromfile(
            os.path.join(path, f'{name_i.rstrip()}.{form_src}'), dtype=data_type).reshape(
            np.load(os.path.join(path, f'{name_i.rstrip()}_shape.npy'))) for name_i in file_names]
    elif form_src == 'mat':
        if data_type:
            logging.warning('ignore data_type')
        file_names = str_remove_end(file_names, form_src)
        source_frames = [sio.loadmat(
            os.path.join(path, f'{name_i.rstrip()}.{form_src}'))['data'] for name_i in file_names]
    elif form_src == 'json':
        if data_type:
            logging.warning('ignore data_type')
        source_frames = [json.load(
            open(os.path.join(path, f'{name_i.rstrip()}.{form_src}'), 'r')) for name_i in file_names]
    else:
        raise ParameterError('Invalid form_src keyword.')

    return source_frames  # return 3D np.array [source][num][feature]


def read_data(path, file_name, form_src='hdf5', dict_key='data', data_type=None, **kwargs):
    """Read data file_name from path.
    Args:
        path (str): path where to read data.
        file_name (str): file name.
        form_src (str, optional): file name extension. Defaults to 'hdf5'.
        dict_key (str, optional): load data dict keyword. Defaults to 'data'.
        data_type ([type], optional): data type after load. Defaults to None.
    Raises:
        ParameterError: kwargs['mode'], Invalid read_data mode.
        ParameterError: form_src, Invalid form_src keyword.
    Returns:
        data ([type]): data load.
    """
    import json
    import os
    import numpy as np
    import h5py
    import scipy.io as sio
    import tables

    def str_remove_end(file_name, file_type):
        """Remove files' name extension.
        Args:
            file_name (str): file name.
            file_type (str): file name extension.
        Returns:
            file_names_small (list[str]): list of file names without extension.
        """
        if file_name.endswith('.' + file_type):
            logging.debug('Remove file names extension.')
            file_name_small = file_name[:-len('.' + file_type)]
        else:
            file_name_small = file_name
        return file_name_small

    if form_src == 'hdf5':
        if data_type:
            logging.warning('ignore data_type')
        file_name = str_remove_end(file_name, form_src)
        if 'mode' not in kwargs.keys():
            data = h5py.File(os.path.join(path, f'{file_name.rstrip()}.{form_src}'), 'r')[dict_key]
        else:
            if kwargs['mode'] == 'pytables':
                with tables.open_file(os.path.join(path, f'{file_name.rstrip()}.{form_src}')) as f_r:
                    data = f_r[dict_key]
            elif kwargs['mode'] == 'np':
                with h5py.File(os.path.join(path, f'{file_name.rstrip()}.{form_src}'), 'r') as f_r:
                    data = np.asarray(f_r[dict_key])
            else:
                raise ParameterError('Invalid read_data mode.')
    elif form_src == 'bin':
        if data_type is None:
            data_type = np.float32
        data = np.fromfile(
            os.path.join(path, f'{file_name.rstrip()}.{form_src}'), dtype=data_type).reshape(
            np.load(os.path.join(path, f'{file_name.rstrip()}_shape.npy')))
    elif form_src == 'mat':
        if data_type:
            logging.warning('ignore data_type')
        file_name = str_remove_end(file_name, form_src)
        data = sio.loadmat(os.path.join(path, f'{file_name.rstrip()}.{form_src}'))[dict_key]
    elif form_src == 'json':
        if data_type:
            logging.warning('ignore data_type')
        data = json.load(open(os.path.join(path, f'{file_name.rstrip()}.{form_src}'), 'r'))
    else:
        raise ParameterError('Invalid form_src keyword.')

    return data


def read_datas(path, file_names, form_src='hdf5', dict_key='data', data_type=None, **kwargs):
    """Read data file_names [file_names] from [path].
    Args:
        path (str): path where to read data.
        file_names (list[str]): list file names to read.
        form_src (str, optional): file name extension. Defaults to 'hdf5'.
        dict_key (str, optional): load data dict keyword. Defaults to 'data'.
        data_type ([type], optional): data type after load. Defaults to None.
    Returns:
        datas (list[type]): datas load.
    """
    datas = [read_data(path, file_name, form_src, dict_key, data_type, **kwargs) for file_name in file_names]
    return datas


def data_save_reshape(data):
    """Reshape data to last dim is not 1.
    Args:
        data (np.ndarray): data to reshape.
    Returns:
        data (np.ndarray): data after reshape.
    Examples:
        data: np.ndarray, shape==(n, 1, fl, 1, 1)
        index_not_one = [0, 2]
        last_one = 2 != 4
        data = data.transpose((0, 1) + (3, 4) + (2, ))
        data.shape = (n, 1, 1, 1, fl)
    """
    index_not_one = [index for (index, value) in enumerate(data.shape) if value != 1]
    if index_not_one:
        last_one = index_not_one[-1]
        if last_one != data.ndim - 1:
            data = data.transpose(
                tuple(range(last_one)) + tuple(range(last_one + 1, data.ndim)) + (last_one,))
    return data


def compute_chunk_size(data, chunk_type, dims=None):
    """Compute chunk size for save .h5 file.
    Args:
        data (np.ndarray): data to save.
        chunk_type (str): chunk size selected.
        dims (int/list[int]): number of indexes.
    Returns:
        chunk_size (tuple(int)): chunk size to save.
    """
    if chunk_type == 'back_dim':
        chunk_size = (1,) + tuple(data[0].shape)
    elif chunk_type == 'last_dim':
        chunk_size = (data.ndim - 1) * (1,) + data.shape[-1:]
    elif chunk_type == 'back_n':
        chunk_size = dims * (1,) + data.shape[dims:]
    elif chunk_type == 'some_dim':
        chunk_size = list(data.shape)
        for dim_i in dims:
            chunk_size[dim_i] = 1
        chunk_size = tuple(chunk_size)
    return chunk_size


def save_datas(set_dict, path_save, **kwargs):
    """Save data dict set_dict[key] to [path_save] using file name [key].
    Args:
        set_dict (dict{str:np.ndarray}): dictionary of datas to be saved, key is the file name.
        path_save (str): path to save files.
    Raises:
        ParameterError: Invalid kwargs keyword.
        ParameterError: Invalid mode_batch keyword.
        ParameterError: Invalid form_save keyword.
    """
    import csv
    import h5py
    import json
    import numpy as np
    import os
    import pickle
    import scipy.io as sio
    import tables

    form_save = 'hdf5'
    dtype = np.dtype('float32')
    mode_batch = 'normal'
    save_key = 'data'
    save_key2 = 'sij'
    chunk_type = 'back_dim'
    mode_csv = 'array'
    batch_num = 200
    for key, value in kwargs.items():
        if key == 'form_save':
            form_save = value
        elif key == 'dtype':
            dtype = value
        elif key == 'mode_batch':
            mode_batch = value
        elif key == 'batch_num':
            batch_num = value
        elif key == 'file_name':
            file_name = value
        elif key == 'save_key':
            save_key = value
        elif key == 'save_key2':
            save_key2 = value
        elif key == 'chunk_type':
            chunk_type = value
        elif key == 'mode_csv':
            mode_csv = value
        else:
            raise ParameterError('Invalid kwargs keyword.')

    if form_save == 'hdf5':
        if dtype == 'str':  # 1D str list
            for name_i, data_i in set_dict.items():
                with h5py.File(os.path.join(path_save, f'{name_i}.hdf5'), 'a') as f_a:
                    d_f = f_a.create_dataset(name_i,
                                             shape=tuple(data_i.shape),
                                             dtype=h5py.string_dtype(encoding='utf-8', length=None), )
                    for j, str_j in enumerate(data_i):
                        d_f[j] = str_j
            return
        if mode_batch == 'normal':
            for name_i, data_i in set_dict.items():
                chunk_size_i = compute_chunk_size(data_i, chunk_type)
                with h5py.File(os.path.join(path_save, f'{name_i}.hdf5'), 'w') as f_w:
                    f_w.create_dataset(
                        save_key, data=data_i, dtype=dtype,
                        chunks=chunk_size_i,
                        compression="gzip", compression_opts=9)
        elif mode_batch == 'batch':  # warning, file will only reshape in this mode instead of creating new file
            filters = tables.Filters(complevel=9, complib='blosc')
            for name_i, data_i in set_dict.items():
                file_name_i = os.path.join(path_save, f'{name_i}.hdf5')
                chunk_size_i = compute_chunk_size(data_i, chunk_type)
                with tables.open_file(file_name_i, 'a') as f_a:
                    if save_key not in f_a.root:
                        data_earray = f_a.create_earray(f_a.root, save_key,
                                                        atom=tables.Atom.from_dtype(dtype),
                                                        shape=((0,) + data_i.shape[1:]),
                                                        chunkshape=chunk_size_i,
                                                        filters=filters)
                    else:
                        data_earray = getattr(f_a.root, save_key)
                    data_earray.append(data_i)
        elif mode_batch == 'batch_h5py':
            for name_i, data_i in set_dict.items():
                file_name_i = os.path.join(path_save, f'{name_i}.hdf5')
                chunk_size_i = compute_chunk_size(data_i, chunk_type)
                dim_0_i = len(data_i) if isinstance(data_i, (list, tuple)) else data_i.shape[0]
                if batch_num < dim_0_i:
                    for j in range(0, dim_0_i, batch_num):
                        if j + batch_num > dim_0_i:
                            data_j = np.asarray(data_i[j:], dtype=dtype)
                        else:
                            data_j = np.asarray(data_i[j:j + batch_num], dtype=dtype)
                        if j == 0:
                            with h5py.File(file_name_i, 'w') as f_w:
                                f_w.create_dataset(
                                    save_key, data=data_j, dtype=dtype,
                                    chunks=chunk_size_i,
                                    maxshape=((None,) + data_j.shape[1:]),
                                    compression="gzip", compression_opts=9)
                        else:
                            with h5py.File(file_name_i, 'a') as f_a:
                                f_a[save_key].resize((f_a[save_key].shape[0] + data_j.shape[0]), axis=0)
                                f_a[save_key][-data_j.shape[0]:] = data_j
                else:
                    with h5py.File(file_name_i, 'w') as f_w:
                        f_w.create_dataset(
                            save_key, data=data_i, dtype=dtype,
                            chunks=chunk_size_i,
                            compression="gzip", compression_opts=9)
        elif mode_batch == 'one_file_no_chunk':
            full_file_name = os.path.join(path_save, f'{file_name}.hdf5')
            with h5py.File(full_file_name, 'a') as f_a:
                for name_i, data_i in set_dict.items():
                    f_a.create_dataset(name_i, data=data_i, dtype=dtype, compression="gzip", compression_opts=9)
        else:
            raise ParameterError('Invalid mode_batch keyword.')
    elif form_save == 'mat':
        for name_i, data_i in set_dict.items():
            sio.savemat(os.path.join(path_save, f'{name_i}.mat'), {save_key2: data_i})
    elif form_save == 'npy':
        for name_i, data_i in set_dict.items():
            np.save(os.path.join(path_save, f'{name_i}.npy'), {save_key2: data_i})
    elif form_save == 'picke':
        for name_i, data_i in set_dict.items():
            with open(os.path.join(path_save, f'{name_i}.pickle'), 'wb') as f_wb:
                pickle.dump({save_key2: data_i}, f_wb)
    elif form_save == 'bin':
        for name_i, data_i in set_dict.items():
            data_i.tofile(os.path.join(path_save, f'{name_i}.bin'))
            np.save(os.path.join(path_save, f'{name_i}_shape.npy'), data_i.shape)
            with open(os.path.join(path_save, f'{name_i}_shape.json'), 'w', encoding='utf-8') as f_w:
                json.dump({save_key: data_i.shape}, f_w)
    elif form_save == 'json':
        with open(os.path.join(path_save, f'{file_name}.json'), 'w', encoding='utf-8') as f_w:
            json.dump(set_dict, f_w)
    elif form_save == 'csv':
        for name_i, data_i in set_dict.items():
            with open(os.path.join(path_save, f'{name_i}.csv'), 'w') as f_w:
                if mode_csv == 'array':
                    csv.writer(f_w).writerows(data_i)
                elif mode_csv == 'dict':
                    csv.DictWriter(f_w, name_i).writerows(data_i)
    else:
        raise ParameterError('Invalid form_save keyword.')


def save_datas_continue(set_dict, path_save, **kwargs):
    """Save data dict set_dict[key] to [path_save] using file name [key].
        Warning, different with 'save_datas', this function will reshape exist files instead of create new files.
    Args:
        set_dict (dict{str:np.ndarray}): dictionary of datas to be saved, key is the file name.
        path_save (str): path to save files.
    Raises:
        ParameterError: Invalid kwargs keyword.
        ParameterError: Invalid mode_batch keyword.
        ParameterError: Invalid form_save keyword.
    """
    import h5py
    import numpy as np
    import os
    import tables

    form_save = 'hdf5'
    dtype = np.dtype('float32')
    mode_batch = 'normal'
    save_key = 'data'
    chunk_type = 'back_dim'

    for key, value in kwargs.items():
        if key == 'form_save':
            form_save = value
        elif key == 'dtype':
            dtype = value
        elif key == 'mode_batch':
            mode_batch = value
        elif key == 'batch_num':
            batch_num = value
        elif key == 'save_key':
            save_key = value
        elif key == 'chunk_type':
            chunk_type = value
        else:
            raise ParameterError('Invalid kwargs keyword.')

    if form_save == 'hdf5':
        if dtype == 'str':  # 1D str list
            for name_i, data_i in set_dict.items():
                file_name_i = os.path.join(path_save, f'{name_i}.hdf5')
                if not os.path.isfile(file_name_i):
                    with h5py.File(file_name_i, 'w') as f_w:
                        d_f = f_w.create_dataset(name_i,
                                                 shape=tuple(data_i.shape),
                                                 dtype=h5py.string_dtype(encoding='utf-8', length=None), )
                        for j, str_j in enumerate(data_i):
                            d_f[j] = str_j
                else:
                    with h5py.File(file_name_i, 'a') as f_a:
                        f_a[name_i].resize((f_a[name_i].shape[0] + data_i.shape[0]), axis=0)
                        for j, str_j in enumerate(data_i):
                            f_a[name_i][-data_i.shape[0] + j] = str_j
            return
        if mode_batch == 'pytables':
            filters = tables.Filters(complevel=9, complib='blosc')
            for name_i, data_i in set_dict.items():
                file_name_i = os.path.join(path_save, f'{name_i}.hdf5')
                chunk_size_i = compute_chunk_size(data_i, chunk_type)
                with tables.open_file(file_name_i, 'a') as f_a:
                    if save_key not in f_a.root:
                        data_earray = f_a.create_earray(f_a.root, save_key,
                                                        atom=tables.Atom.from_dtype(dtype),
                                                        shape=((0,) + data_i.shape[1:]),
                                                        chunkshape=chunk_size_i,
                                                        filters=filters)
                    else:
                        data_earray = getattr(f_a.root, save_key)
                data_earray.append(data_i)
        if mode_batch == 'batch_pytables':
            filters = tables.Filters(complevel=9, complib='blosc')
            for name_i, data_i in set_dict.items():
                file_name_i = os.path.join(path_save, f'{name_i}.hdf5')
                chunk_size_i = compute_chunk_size(data_i, chunk_type)
                dim_0_i = len(data_i) if isinstance(data_i, (list, tuple)) else data_i.shape[0]
                with tables.open_file(file_name_i, 'a') as f_a:
                    if save_key not in f_a.root:
                        data_earray = f_a.create_earray(f_a.root, save_key,
                                                        atom=tables.Atom.from_dtype(dtype),
                                                        shape=((0,) + data_i.shape[1:]),
                                                        chunkshape=chunk_size_i,
                                                        filters=filters)
                    else:
                        data_earray = getattr(f_a.root, save_key)
                    if batch_num < dim_0_i:
                        for j in range(0, dim_0_i, batch_num):
                            if j + batch_num > dim_0_i:
                                data_j = np.asarray(data_i[j:], dtype=dtype)
                            else:
                                data_j = np.asarray(data_i[j:j + batch_num], dtype=dtype)
                            data_earray.append(data_j)
                    else:
                        data_earray.append(data_i)
        elif mode_batch == 'batch_h5py':
            for name_i, data_i in set_dict.items():
                file_name_i = os.path.join(path_save, f'{name_i}.hdf5')
                chunk_size_i = compute_chunk_size(data_i, chunk_type)
                dim_0_i = len(data_i) if isinstance(data_i, (list, tuple)) else data_i.shape[0]
                if batch_num < dim_0_i:
                    for j in range(0, dim_0_i, batch_num):
                        if j + batch_num > dim_0_i:
                            data_j = np.asarray(data_i[j:], dtype=dtype)
                        else:
                            data_j = np.asarray(data_i[j:j + batch_num], dtype=dtype)
                        if j == 0 and not os.path.isfile(file_name_i):
                            with h5py.File(file_name_i, 'w') as f_w:
                                f_w.create_dataset(
                                    save_key, data=data_j, dtype=dtype,
                                    chunks=chunk_size_i,
                                    maxshape=((None,) + data_j.shape[1:]),
                                    compression="gzip", compression_opts=9)
                        else:
                            with h5py.File(file_name_i, 'a') as f_a:
                                f_a[save_key].resize((f_a[save_key].shape[0] + data_j.shape[0]), axis=0)
                                f_a[save_key][-data_j.shape[0]:] = data_j
                else:
                    if os.path.isfile(file_name_i):
                        with h5py.File(file_name_i, 'a') as f_a:
                            f_a[save_key].resize((f_a[save_key].shape[0] + data_i.shape[0]), axis=0)
                            f_a[save_key][-data_i.shape[0]:] = data_i
                    else:
                        with h5py.File(file_name_i, 'w') as f_w:
                            f_w.create_dataset(
                                save_key, data=data_i, dtype=dtype,
                                chunks=chunk_size_i,
                                maxshape=((None,) + data_i.shape[1:]),
                                compression="gzip", compression_opts=9)
        else:
            raise ParameterError('Invalid mode_batch keyword.')
    else:
        raise ParameterError('Invalid form_save keyword.')


def save_process_batch(data, func, path_save, file_name, batch_num=200, save_key='data', mode_batch='batch',
                       chunk_type='back_dim', *args, **kwargs):
    """Process data by batch through func, save to path_save.
    Args:
        data (np.ndarray,shape==(nsam, - - )): data to save
        func (function): function to process data
        path_save (str): where to save data
        file_name (str): name of the saved file
        batch_num (int, optional): each batch process batch_num data
        save_key (str, optional): data save keyword. Defaults to 'data'.
        mode_batch (str, optional): use pytables(default) or h5py to save data. Defaults to 'batch'.
        chunk_type (str, optional): for compute chunk size to save .h5 file. Defaults to 'back_dim'.
    """
    import h5py
    import numpy as np
    import tables

    dtype = None
    for key, value in kwargs.items():
        if key == 'dtype':
            dtype = value
    if dtype is None:
        dtype = np.dtype('float32')

    for j in range(0, data.shape[0], batch_num):
        if j + batch_num > data.shape[0]:
            data_j = data[j:]
        else:
            data_j = data[j:j + batch_num]

        data_result = func(data_j, *args, **kwargs)
        chunk_size = compute_chunk_size(data_result, chunk_type)

        if mode_batch == 'batch':  # pytables
            with tables.open_file(os.path.join(path_save, f'{file_name}.hdf5'), 'a') as f_w:
                if save_key not in f_w.root:
                    data_earray = f_w.create_earray(f_w.root, save_key,
                                                    atom=tables.Atom.from_dtype(dtype),
                                                    shape=((0,) + data_result.shape[1:]),
                                                    chunkshape=chunk_size,
                                                    filters=tables.Filters(complevel=9, complib='blosc'))
                else:
                    data_earray = getattr(f_w.root, save_key)
                data_earray.append(data_result)
        elif mode_batch == 'batch_h5py':  # h5py
            if j == 0:
                with h5py.File(os.path.join(path_save, f'{file_name}.hdf5'), 'w') as f:
                    f.create_dataset(
                        save_key, data=data_result,
                        dtype=dtype,
                        chunks=chunk_size,
                        maxshape=((None,) + data_result.shape[1:]),
                        compression="gzip", compression_opts=9)
            else:
                with h5py.File(os.path.join(path_save, f'{file_name}.hdf5'), 'a') as f:
                    f[save_key].resize(
                        (f[save_key].shape[0] + data_result.shape[0]), axis=0)
                    f[save_key][-data_result.shape[0]:] = data_result
        else:
            raise ParameterError('Invalid mode_batch keyword.')


def concat_h5py(fname_1, fname_2, fname_new, create_h5py=False, axis=0, save_key='data', mode_batch='batch_h5py',
                batch_num=200, dtype=None):
    """Concatenate two data saved in h5py file.
    Args:
        fname_1 (str): full file name with path of data 1.
        fname_2 (str): full file name with path of data 2.
        fname_new (str): full file name with path of new data concatenated of data 1 and data 2.
        create_h5py (bool, optional): whether creating .hdf files. Defaults to False.
        axis (int, optional): axis of data array. Defaults to 0.
        save_key (str, optional): keyword of the data to save to file. Defaults to 'data'.
        mode_batch (str, optional): mode of saving data to file by batch. Defaults to 'batch_h5py'.
        batch_num (int, optional): number of the batch to save data. Defaults to 200.
        dtype (np.dtype, optional): type of the data to save. Defaults to None.
    Raises:
        ParameterError: Invalid save_key keyword.
        ParameterError: Invalid mode_batch keyword.
    """
    import h5py
    import numpy as np
    # import tables
    from file_operation import mycopyfile

    if dtype is None:
        dtype = np.dtype('float32')

    if create_h5py:
        path_src_1, src_name_1 = os.path.split(fname_1)
        data_1 = np.asarray(read_data(path_src_1, src_name_1, dict_key=save_key))
        chunk_size_1 = compute_chunk_size(data_1, 'back_n', axis + 1)
        with h5py.File(fname_new, 'w') as f_w:
            f_w.create_dataset(
                save_key, data=data_1, dtype=dtype,
                chunks=chunk_size_1, maxshape=(None,) * data_1.ndim,
                compression="gzip", compression_opts=9)
    else:
        mycopyfile(fname_1, fname_new)

    path_src_2, src_name_2 = os.path.split(fname_2)
    data_2 = read_data(path_src_2, src_name_2, dict_key=save_key)
    if mode_batch == 'normal':
        with h5py.File(fname_new, 'a') as f_a:
            f_a[save_key].resize((f_a[save_key].shape[axis] + data_2.shape[axis]), axis=axis)
            if axis == 0:
                f_a[save_key][-data_2.shape[0]:] = data_2
            elif axis == 1:
                f_a[save_key][:, -data_2.shape[1]:] = data_2
    elif mode_batch == 'batch':
        raise ParameterError('Invalid mode_batch keyword.')
    elif mode_batch == 'batch_h5py':
        for j in range(0, data_2.shape[axis], batch_num):
            j_end = data_2.shape[axis] if j + batch_num > data_2.shape[axis] else j + batch_num
            if axis == 0:
                data_j = data_2[j:j_end]
            elif axis == 1:
                data_j = data_2[:, j:j_end]

            with h5py.File(fname_new, 'a') as f_a:
                f_a[save_key].resize(
                    (f_a[save_key].shape[axis] + data_j.shape[axis]), axis=axis)
                if axis == 0:
                    f_a[save_key][-data_j.shape[0]:] = data_j
                elif axis == 1:
                    f_a[save_key][:, -data_j.shape[1]:] = data_j
    else:
        raise ParameterError('Invalid mode_batch keyword.')


def create_frame_info(wav_file_names, num_frames, src_name):
    assert(len(wav_file_names) == len(num_frames))
    info_list = []
    index_src = 0
    for wav_file_name, num_frame in zip(wav_file_names, num_frames):
        for i in range(num_frame):
            index_i = index_src + i
            info_dict_i = {'src': src_name, 'file_name': wav_file_name, 'index': index_i}
            info_list.append(info_dict_i)
        index_src += num_frame
    return info_list


def data_seg_create(path_class):
    """Create and save seg wavmats from raw data .wav files,
        you may run this function only once.
    Args:
        path_class (object class PathSourceRoot): object of class to compute path.
    """

    import json
    import numpy as np
    from feature_extract import feature_extract
    from file_operation import mkdir, read_wavs_to_np, walk_dirs_start_str

    # raw data files in dirs e.g. /s0, /s1 /s2 /s3
    path_seg_root = path_class.path_seg_root
    mkdir(path_seg_root)
    path_raw = path_class.get_path_raw()

    dir_names = walk_dirs_start_str(path_raw, 's')
    n_src = len(dir_names)

    source_frames = []
    wav_file_names = []
    num_frames = []
    for dir_i in dir_names:
        sources_wav_np_i, wav_file_names_i = read_wavs_to_np(dir_i, get_sr(), get_mono())
        wav_file_names.append([os.path.basename(wav_file_names_j) for wav_file_names_j in wav_file_names_i])
        # original wav sampling points
        sourceframes_np_i = feature_extract('sample_np',
                                            **{'sources': sources_wav_np_i,
                                               'fl': get_fl(),
                                               'fs': get_fs()})  # 1d list 2darray to 1d list 2darray
        num_frames_i = [frames_file.shape[0] for frames_file in sourceframes_np_i]
        num_frames.append(num_frames_i)
        # 1d list 2darray to 2darray (n_samples, fl)
        source_frames.append(np.vstack(np.asarray(sourceframes_np_i)))

    logging.debug('source_frames.shape')
    for sf_i in source_frames:
        logging.debug(sf_i.shape)

    dir_names_save = []
    for i in range(0, n_src, 1):
        dir_names_save.append(f's_{i}')
    with open(os.path.join(path_seg_root, 'dirname.json'), 'w', encoding='utf-8') as f_w:
        json.dump({'dirname': dir_names_save}, f_w)

    frame_info = {'source_name': dir_names_save, 'wav_file_name': wav_file_names, 'number_frame': num_frames}
    with open(os.path.join(path_seg_root, 'frame_info.json'), 'w', encoding='utf-8') as f_w:
        json.dump(frame_info, f_w)

    path_seg = path_class.get_path_seg()
    mkdir(path_seg)
    save_datas(dict(zip(dir_names_save, source_frames)), path_seg, dtype='float64')


def data_mixwav_create(path_class):  # pylint: disable=too-many-locals
    """Create and save mixed sources original sampling point wavmat,
        you may run this function only once.
    Args:
        path_class (object class PathSourceRoot): object of class to compute path.
    """
    import json
    from itertools import combinations
    from file_operation import mkdir, mycopyfile
    from prepare_data import balancesets, mixaddframes_np

    path_seg_root = path_class.path_seg_root
    path_seg = path_class.get_path_seg()
    # path_mix_root = path_class.path_mix_root
    # mkdir(path_mix_root)
    path_source_root = path_class.path_source_root
    mkdir(path_source_root)

    # read sources
    mycopyfile(os.path.join(path_seg_root, 'dirname.json'),
               os.path.join(path_source_root, 'dirname.json'))
    dir_names = json.load(open(os.path.join(path_source_root, 'dirname.json'), 'r'))['dirname']

    source_frames = read_source(path_seg, dir_names)

    # balance sources
    source_frames = balancesets(source_frames)

    n_src = len(source_frames)

    path_source_out = os.path.join(path_source_root, 's_hdf5')
    mkdir(path_source_out)
    dir_names = []
    data_list = []
    for i in range(1, n_src + 1, 1):  # mix 1 to 3 without 0
        if i == 1:
            index_ci = list(combinations(range(n_src), i))
        else:
            index_ci = list(combinations(range(1, n_src), i))

        for index_ci_j in index_ci:  # e.g. (1,2)
            items = ['s']
            for k in index_ci_j:
                items.append('_')
                items.append(str(k))
            pathout_j = ''.join(items)
            dir_names.append(pathout_j)

            mix_cij_arr = mixaddframes_np(
                [source_frames[k] for k in index_ci_j])
            mix_cij_arr = mix_cij_arr.reshape(
                mix_cij_arr.shape[0:1] + (1,) + mix_cij_arr.shape[-1:])
            data_list.append(mix_cij_arr)
    with open(os.path.join(path_source_root, 'dirname.json'), 'w', encoding='utf-8') as f_w:
        json.dump({'dirname': dir_names}, f_w)

    save_datas(dict(zip(dir_names, data_list)), path_source_out)


class SpectrumCreate(object):
    """Create a feature related to spectrum."""

    def __init__(self):
        pass

    def _spectrum_create(self, sources_wavmat, feature, win_length, hop_length, fix_length=False, window='hamming'):
        """Abstract method for create spectrum feature sources_frames."""
        source_frames = []
        for source_i in sources_wavmat:
            source_frames.append(feature_extract(
                feature, **{
                    'source': source_i.reshape(-1, ),
                    'window': window,
                    'win_length': win_length, 'hop_length': hop_length,
                    'n_fft': win_length, 'center': False,
                    'dtype': np.complex64, 'fix_length': fix_length}))  # 2D to 3D
        dtype = np.complex64 if feature in {'phase_spectrum'} else np.float32
        return np.asarray(source_frames, dtype=dtype)

    def magspectrum_create(self, sources_wavmat, win_length, hop_length, fix_length=False, window='hamming'):
        """Create magnitude (amplitude) spectrum feature sources_frames."""

        return self._spectrum_create(sources_wavmat, 'magspectrum', win_length, hop_length, fix_length, window)

    def angspectrum_create(self, sources_wavmat, win_length, hop_length, fix_length=False, window='hamming'):
        """Create angle (phase) spectrum feature sources_frames."""

        return self._spectrum_create(sources_wavmat, 'angspectrum', win_length, hop_length, fix_length, window)

    def phase_spectrum_create(self, sources_wavmat, win_length, hop_length, fix_length=False, window='hamming'):
        """Create phase spectrum feature sources_frames."""

        return self._spectrum_create(sources_wavmat, 'phase_spectrum', win_length, hop_length, fix_length, window)

    def realspectrum_create(self, sources_wavmat, win_length, hop_length, fix_length=False, window='hamming'):
        """Create real part of spectrum feature sources_frames."""

        return self._spectrum_create(sources_wavmat, 'realspectrum', win_length, hop_length, fix_length, window)

    def imgspectrum_create(self, sources_wavmat, win_length, hop_length, fix_length=False, window='hamming'):
        """Create image part of spectrum feature sources_frames."""

        return self._spectrum_create(sources_wavmat, 'imgspectrum', win_length, hop_length, fix_length, window)

    def logmelspectrum_create(self, sources, sr, n_mels, win_length=None, hop_length=None, window=None, mode=0):
        """Create Log-Mel Spectrogram feature sources_frames."""
        source_frames = []
        if mode == 0:  # input wavmat
            for source_i in sources:
                source_frames.append(feature_extract(
                    'logmelspectrum', **{
                        'source': source_i.reshape(-1, ),
                        'sr': sr, 'n_mels': n_mels, 'window': window,
                        'win_length': win_length, 'hop_length': hop_length,
                        'n_fft': win_length, 'center': False, 'dtype': np.float32}))  # 2D to 3D
        elif mode == 1:  # input stft spectrum
            for source_i in sources:
                source_frames.append(feature_extract(
                    'logmelspectrum', **{
                        'S': source_i.transpose() ** 2,
                        'sr': sr, 'n_mels': n_mels}))  # 2D to 3D
        return np.asarray(source_frames, dtype=np.float32)

    def mfcc_create(self, sources, sr, n_mfcc,
                    win_length=None, hop_length=None, window=None, n_mels=None, mode=0):
        """Create Log-Mel Spectrogram feature sources_frames."""
        source_frames = []
        if mode == 0:
            for source_i in sources:
                source_frames.append(feature_extract(
                    'mfcc', **{
                        'source': source_i.reshape(-1, ), 'sr': sr, 'n_mfcc': n_mfcc,
                        'n_fft': win_length, 'hop_length': hop_length, 'win_length': win_length,
                        'window': window, 'center': False,
                        'n_mels': n_mels, 'dtype': np.float32
                    }))  # 2D to 3D
        elif mode == 1:  # input log-power Mel spectrogram
            for source_i in sources:
                source_frames.append(feature_extract(
                    'mfcc', **{
                        'source': None, 'S': librosa.power_to_db(source_i.transpose()),
                        'sr': sr, 'n_mfcc': n_mfcc
                    }))  # 2D to 3D
        return np.asarray(source_frames, dtype=np.float32)

    def demon_create(self, sources, high=30000, low=20000, cutoff=1000.0, fs=200000, mode='square_law'):
        """Create Log-Mel Spectrogram feature sources_frames."""
        source_frames = []
        for source_i in sources:
            source_frames.append(feature_extract(
                'demon', **{
                    'source': source_i, 'high': high, 'low': low, 'cutoff': cutoff, 'fs': fs, 'mode': mode
                }))  # 2D to 3D
        return np.asarray(source_frames, dtype=np.float32)

    def feature_create(self, sources, path_class_out, form_src, **kwargs):
        if form_src == 'magspectrum':
            feature = self.magspectrum_create(
                sources,
                path_class_out.get_win_length(),
                path_class_out.get_hop_length(), kwargs['fix_length'], kwargs['window'])
        elif form_src == 'angspectrum':
            feature = self.angspectrum_create(
                sources,
                path_class_out.get_win_length(),
                path_class_out.get_hop_length(), kwargs['fix_length'], kwargs['window'])
        elif form_src == 'phase_spectrum':
            feature = self.phase_spectrum_create(
                sources,
                path_class_out.get_win_length(),
                path_class_out.get_hop_length(), kwargs['fix_length'], kwargs['window'])
        elif form_src == 'realspectrum':
            feature = self.realspectrum_create(
                sources,
                path_class_out.get_win_length(),
                path_class_out.get_hop_length(), kwargs['fix_length'], kwargs['window'])
        elif form_src == 'imgspectrum':
            feature = self.imgspectrum_create(
                sources,
                path_class_out.get_win_length(),
                path_class_out.get_hop_length(), kwargs['fix_length'], kwargs['window'])
        elif form_src == 'logmelspectrum':
            mode = 0 if 'mode' not in kwargs.keys() else kwargs['mode']
            if mode == 0:  # input wavmats
                feature = self.logmelspectrum_create(
                    sources, kwargs['sr'], kwargs['n_mels'],
                    path_class_out.get_win_length(),
                    path_class_out.get_hop_length(), kwargs['window'], mode=mode)
            elif mode == 1:  # inpute stft spectrum
                feature = self.logmelspectrum_create(
                    sources, kwargs['sr'], kwargs['n_mels'],
                    path_class_out.get_win_length(),
                    path_class_out.get_hop_length(), mode=mode)
        elif form_src == 'mfcc':
            mode = 0 if 'mode' not in kwargs.keys() else kwargs['mode']
            if mode == 0:  # input wavmats
                feature = self.mfcc_create(
                    sources, kwargs['sr'], kwargs['n_mfcc'],
                    path_class_out.get_win_length(), path_class_out.get_hop_length(),
                    kwargs['window'], kwargs['n_mels'], mode=mode)
            elif mode == 1:  # input log-power Mel spectrogram
                feature = self.mfcc_create(
                    sources, kwargs['sr'], kwargs['n_mfcc'], mode=mode)
        elif form_src == 'demon':
            feature = self.demon_create(sources, **kwargs)
        else:
            raise ParameterError('Invalid feature')
        return feature


if __name__ == '__main__':
    from prepare_data_shipsear_separation_mix_s0tos3 import PathSourceRootSep

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO)

    PATH_ROOT = '../data/shipsEar/mix_separation'

    # Create segment datas.
    PATH_CLASS = PathSourceRootSep(PATH_ROOT)
    data_seg_create(PATH_CLASS)

    logging.info('finished')
