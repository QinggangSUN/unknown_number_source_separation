# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 19:07:49 2018

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
import librosa
from error import Error, ParameterError


def subframe_np(sources, fl, fs, scale=False):  # pylint: disable=invalid-name
    """Cuting sources to frames using librosa.
    Args:
        sources (list[np.ndarray,shape==(length,)]): [n] n np array wav datas.
        fl (int): frame length to cut.
        fs (int): frame shift length to cut.
        scale (bool, optional): whether scaler data from [-1,1] to [0,1]. Defaults to False.
    Returns:
        frames (list[np.ndarray,shape==(n_samples, fl)]): frames after cut.
    """
    if scale is True:  # scale from [-1,1] to [0,1]
        for si in sources:  # pylint: disable=invalid-name
            si = si + 1.0  # pylint: disable=invalid-name
            si = si * 0.5  # pylint: disable=invalid-name
    if librosa.__version__ >= '0.7.1':
        frames = [librosa.util.frame(si, fl, fs, axis=0) for si in sources]  # pylint: disable=unexpected-keyword-arg
    else:
        frames = [librosa.util.frame(si, fl, fs).T for si in sources]  # (fl, n_samples)
    return frames


def feature_extract(feature, **kwargs):
    """Extrct a feature of wav frame.
    Args:
        feature (str): keyword of the feature.
    Raises:
        ParameterError: not enough parameters.
        ParameterError: source and S cannot all be None.
        ParameterError: Invalid feature type.
    Returns:
        features (func): function of extract feature.
    """
    if feature == 'sample_np':
        if 'sources' not in kwargs or 'fl' not in kwargs or 'fs' not in kwargs:
            raise ParameterError('not enough parameters')
        features = subframe_np(sources=kwargs['sources'], fl=kwargs['fl'], fs=kwargs['fs'])
    else:
        raise ParameterError('Invalid feature type.')
    return features
