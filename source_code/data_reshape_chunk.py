# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 15:24:46 2021

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
import gc
from file_operation import list_files_end_str
from prepare_data_shipsear_recognition_mix_s0tos3 import read_data, save_process_batch


def reshape_chunk_size(path_read, filename_read, key_data='data', path_save=None, filename_save=None):
    """Read and save .h5 file using a different chunk size.
    Args:
        path_read (str): path where read file.
        filename_read (str): name of the file to read.
        key_data (str, optional): keyword of the dict of data. Defaults to 'data'.
        path_save (str, optional): path where to save file. Defaults to None.
        filename_save (str, optional): name of the file to save. Defaults to None.
    """
    if path_save is None:
        path_save = path_read
        filename_save = f'{filename_read}_reshape'
    data_r = read_data(path_read, filename_read, form_src='hdf5', dict_key=key_data)
    # save_datas({filename_save: data_r}, path_save, **{'mode_batch': 'batch_h5py'})
    save_process_batch(data_r, lambda x: x, path_save, filename_save, mode_batch='batch_h5py')
    del data_r
    gc.collect()


def reshape_chunk_size_dir(path_read, file_extension='.hdf5', path_save=None):
    """Read and save .h5 files under path_read using a different chunk size.
    Args:
        path_read (str): path where read files.
        file_extension (str, optional): filename extension. Defaults to '.hdf5'.
        path_save (str, optional): path where to save files. Defaults to None.
    """
    file_names = list_files_end_str(path_read, file_extension, False)
    if path_save is None:
        for filename in file_names:
            reshape_chunk_size(path_read, filename)
    else:
        for filename in file_names:
            reshape_chunk_size(path_read, filename, path_save=path_save, filename_save=filename)
