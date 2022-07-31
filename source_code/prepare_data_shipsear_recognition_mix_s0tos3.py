# -*- coding: utf-8 -*-
"""
Created on Thu Aug  9 20:34:30 2018

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

    balance sources first, than full combination mix without s0
"""
# pylint: disable=attribute-defined-outside-init, import-outside-toplevel, invalid-name
# pylint: disable=line-too-long, no-member, redefined-outer-name, reimported
# pylint: disable=too-many-arguments, too-many-branches, too-many-lines, too-many-locals, useless-object-inheritance,
# pylint: disable=too-many-statements

import logging
import os

from error import ParameterError

_SR = 52734
_IS_MONO = True
_FRAME_LENGTH = 10547    # ~200ms 10546.800 000 000 001
_FRAME_SHIFT = 10547
# _FRAME_SHIFT = 2636     # 10547/4 = 2636.7


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
# _WIN_LENGTH = _FRAME_LENGTH
# _HOP_LENGTH = _FRAME_LENGTH
# _HOP_LENGTH = 2636
_WIN_LENGTH = 1582     # 52734*0.03 = 1582.02
_HOP_LENGTH = 396      # 52734*0.03/4 = 395.505
# _WIN_LENGTH = 1055     # 52734*0.02 = 1054.68
# _HOP_LENGTH = 264      # 52734*0.02/4 = 263.67
# _WIN_LENGTH = 4
# _HOP_LENGTH = 1


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
        elif self._form_src in {'magspectrum', 'angspectrum', 'realspectrum', 'imgspectrum'}:
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
            raise ParameterError('form_src invaild')

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

    def _set_path_source(self, value, form_src, scaler_data, sub_set_way):
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

        if self._scaler_data == 'mm':
            self._path_source = os.path.join(self._path_source_root, f'min_max_scaler_{self.sub_set_way}')
        elif self._scaler_data == 'or':
            self._path_source = os.path.join(self._path_source_root, f'original_{self.sub_set_way}')
        elif self._scaler_data == 'max_one':
            self._path_source = os.path.join(self._path_source_root, f'max_one_{self.sub_set_way}')
        else:
            raise ParameterError('scaler_data invaild')

    def _get_path_source(self):
        """Indirect accessor for 'path_source' property."""
        if not hasattr(self, '_path_source'):
            self._set_path_source(None, None, None, None)
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
            value, form_src, scaler_data, sub_set_way = args
        except ValueError:
            raise ParameterError("Pass an iterable with four items")
        else:
            self._set_path_source(value, form_src, scaler_data, sub_set_way)

    path_source = property(get_path_source, set_path_source)

    def get_scaler_data(self):
        """Get read-only _scaler_data."""
        return self._scaler_data

    def get_form_src(self):
        """Get read-only _form_src."""
        return self._form_src


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
            if name_i.endswith('.'+file_type):
                logging_debug = True
                file_names_small.append(name_i[:-len('.'+file_type)])
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
            file_names (list[str]): list of file names.
            file_type (str): file name extension.
        Returns:
            file_names_small (list[str]): list of file names without extension.
        """
        if file_name.endswith('.'+file_type):
            logging.debug('Remove file names extension.')
            file_name_small = file_name[:-len('.'+file_type)]
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
        if last_one != data.ndim-1:
            data = data.transpose(
                tuple(range(last_one))+tuple(range(last_one+1, data.ndim))+(last_one, ))
    return data


def compute_chunk_size(data, chunk_type, dims=None):
    """Compute chunk size for save .h5 file.
    Args:
        data (np.ndarray): data to save.
        chunk_type (str): chunk size selected.
    Returns:
        chunk_size (tuple(int)): chunk size to save.
    """
    if chunk_type == 'back_dim':
        chunk_size = (1,)+data.shape[1:]
    elif chunk_type == 'last_dim':
        chunk_size = (data.ndim-1)*(1,)+data.shape[-1:]
    elif chunk_type == 'back_n':
        chunk_size = dims*(1,)+data.shape[dims:]
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

    for key, value in kwargs.items():
        if key == 'form_save':
            form_save = value
        elif key == 'dtype':
            dtype = value
        elif key == 'mode_batch':
            mode_batch = value
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
                                             dtype=h5py.string_dtype(encoding='utf-8', length=None),)
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
        elif mode_batch == 'batch':
            filters = tables.Filters(complevel=9, complib='blosc')
            for name_i, data_i in set_dict.items():
                file_name_i = os.path.join(path_save, f'{name_i}.hdf5')
                chunk_size_i = compute_chunk_size(data_i, chunk_type)
                with tables.open_file(file_name_i, 'a') as f_w:
                    if save_key not in f_w.root:
                        data_earray = f_w.create_earray(f_w.root, save_key,
                                                        atom=tables.Atom.from_dtype(dtype),
                                                        shape=((0,)+data_i.shape[1:]),
                                                        chunkshape=chunk_size_i,
                                                        filters=filters)
                    else:
                        data_earray = getattr(f_w.root, save_key)
                    data_earray.append(data_i)
        elif mode_batch == 'batch_h5py':
            for name_i, data_i in set_dict.items():
                file_name_i = os.path.join(path_save, f'{name_i}.hdf5')
                chunk_size_i = compute_chunk_size(data_i, chunk_type)
                if not os.path.isfile(file_name_i):
                    with h5py.File(file_name_i, 'w') as f_w:
                        f_w.create_dataset(
                            save_key, data=data_i, dtype=dtype,
                            chunks=chunk_size_i,
                            maxshape=((None,)+data_i.shape[1:]),
                            compression="gzip", compression_opts=9)
                else:
                    with h5py.File(file_name_i, 'a') as f_a:
                        f_a[save_key].resize((f_a[save_key].shape[0]+data_i.shape[0]), axis=0)
                        f_a[save_key][-data_i.shape[0]:] = data_i
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
        if j+batch_num > data.shape[0]:
            data_j = data[j:]
        else:
            data_j = data[j:j+batch_num]

        data_result = func(data_j, *args, **kwargs)
        chunk_size = compute_chunk_size(data_result, chunk_type)

        if mode_batch == 'batch':  # pytables
            with tables.open_file(os.path.join(path_save, f'{file_name}.hdf5'), 'a') as f_w:
                if save_key not in f_w.root:
                    data_earray = f_w.create_earray(f_w.root, save_key,
                                                    atom=tables.Atom.from_dtype(dtype),
                                                    shape=((0,)+data_result.shape[1:]),
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
                        maxshape=((None,)+data_result.shape[1:]),
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
        chunk_size_1 = compute_chunk_size(data_1, 'back_n', axis+1)
        with h5py.File(fname_new, 'w') as f_w:
            f_w.create_dataset(
                save_key, data=data_1, dtype=dtype,
                chunks=chunk_size_1, maxshape=(None,)*data_1.ndim,
                compression="gzip", compression_opts=9)
    else:
        mycopyfile(fname_1, fname_new)

    path_src_2, src_name_2 = os.path.split(fname_2)
    data_2 = read_data(path_src_2, src_name_2, dict_key=save_key)
    if mode_batch == 'normal':
        with h5py.File(fname_new, 'a') as f_a:
            f_a[save_key].resize((f_a[save_key].shape[axis]+data_2.shape[axis]), axis=axis)
            if axis == 0:
                f_a[save_key][-data_2.shape[0]:] = data_2
            elif axis == 1:
                f_a[save_key][:, -data_2.shape[1]:] = data_2
    elif mode_batch == 'batch':
        raise ParameterError('Invalid mode_batch keyword.')
    elif mode_batch == 'batch_h5py':
        for j in range(0, data_2.shape[axis], batch_num):
            j_end = data_2.shape[axis] if j+batch_num > data_2.shape[axis] else j+batch_num
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


def data_seg_create(path_class):
    """Create and save seg wavmats from raw data .wav files,
        you may run this function only onece.
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
    for dir_i in dir_names:
        sources_wavi_np = read_wavs_to_np(dir_i, get_sr(), get_mono())
        # original wav sampling points
        sourceframesi_np = feature_extract('sample_np',
                                           **{'sources': sources_wavi_np,
                                              'fl': get_fl(),
                                              'fs': get_fs()})  # 1d list 2darray to 1d list 2darray
        # 1d list 2darray to 2darray (n_samples, fl)
        source_frames.append(np.vstack(np.asarray(sourceframesi_np)))

    logging.debug('source_frames.shape')
    for sf_i in source_frames:
        logging.debug(sf_i.shape)

    dir_names_save = []
    for i in range(0, n_src, 1):
        dir_names_save.append(f's_{i}')
    with open(os.path.join(path_seg_root, 'dirname.json'), 'w', encoding='utf-8') as f_w:
        json.dump({'dirname': dir_names_save}, f_w)

    path_seg = path_class.get_path_seg()
    mkdir(path_seg)
    save_datas(dict(zip(dir_names_save, source_frames)), path_seg, dtype='float64')


def data_mixwav_create(path_class):  # pylint: disable=too-many-locals
    """Create and save mixed sources original sampling point wavmat,
        you may run this function only onece.
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
    for i in range(1, n_src+1, 1):  # mix 1 to 3 without 0
        index_ci = []  # e.g.[0,1,2,3] [(1,2)(1,3)(2,3)]
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
                mix_cij_arr.shape[0:1]+(1,)+mix_cij_arr.shape[-1:])
            data_list.append(mix_cij_arr)
    with open(os.path.join(path_source_root, 'dirname.json'), 'w', encoding='utf-8') as f_w:
        json.dump({'dirname': dir_names}, f_w)

    save_datas(dict(zip(dir_names, data_list)), path_source_out)


if __name__ == '__main__':
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.INFO)

    PATH_ROOT = '../data/shipsEar/mix_separation'

    # Create segment datas.
    PATH_CLASS = PathSourceRoot(PATH_ROOT)
    data_seg_create(PATH_CLASS)

    # Create original wavmat mixed sources.
    PATH_CLASS = PathSourceRoot(PATH_ROOT, form_src='wav')
    data_mixwav_create(PATH_CLASS)

    logging.info('finished')
