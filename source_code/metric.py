# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 17:38:52 2020

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
import numpy as np


def samerate_acc_cut_np(y_true, y_pred, delta=1e-10):
    """Same rate (cosine similarity) including cut small value using np."""
    y_true = np.asarray(
        [delta if yi >= 0 and yi < delta else yi for yi in y_true], dtype=np.float32)
    y_true = np.asarray(
        [-delta if yi < 0 and yi > -delta else yi for yi in y_true], dtype=np.float32)
    y_pred = np.asarray(
        [delta if yi >= 0 and yi < delta else yi for yi in y_pred], dtype=np.float32)
    y_pred = np.asarray(
        [-delta if yi < 0 and yi > -delta else yi for yi in y_pred], dtype=np.float32)

    p_t = np.dot(y_pred, y_true)
    p_p = np.dot(y_pred, y_pred)
    t_t = np.dot(y_true, y_true)
    numerator = np.abs(p_t)
    denominator = np.sqrt(p_p*t_t)
    denominator = np.clip(denominator, 1e-30, 1e30)
    same_rate = numerator/denominator
    same_rate = np.clip(same_rate, 1e-30, 1.0)
    return same_rate


def samerate_acc_clip_np(y_true, y_pred):
    """Same rate (cosine similarity) including clip denominator using np."""
    p_t = np.dot(y_pred, y_true)
    p_p = np.dot(y_pred, y_pred)
    t_t = np.dot(y_true, y_true)
    numerator = np.abs(p_t)
    denominator = np.sqrt(p_p*t_t)
    denominator = np.clip(denominator, 1e-30, 1e30)
    same_rate = numerator/denominator
    same_rate = np.clip(same_rate, 1e-30, 1.0)
    return same_rate


def samerate_acc_stabilizer_np(y_true, y_pred, epsi=1e-8):
    """Cosine similarity same rate including denominator stabilizer using np.
        Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, et al. What’s all the Fuss about Free Universal Sound
        Separation Data?. ICASSP 2021. https://doi.org/10.1109/ICASSP39728.2021.9414774.
    """
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    p_t = np.dot(y_pred, y_true)
    p_p = np.dot(y_pred, y_pred)
    t_t = np.dot(y_true, y_true)
    numerator = np.abs(p_t)
    denominator = np.sqrt(p_p*t_t)+epsi
    same_rate = numerator/denominator
    return same_rate


def samerate_acc_np(y_true, y_pred):
    """Same rate (cosine similarity) using np."""
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)

    p_t = np.dot(y_pred, y_true)
    p_p = np.dot(y_pred, y_pred)
    t_t = np.dot(y_true, y_true)
    numerator = np.abs(p_t)
    denominator = np.sqrt(p_p * t_t)
    same_rate = numerator/denominator

    # return np.abs(np.dot(y_pred, y_true))/np.sqrt(np.dot(y_pred, y_pred)*np.dot(y_true, y_true))
    return same_rate


def si_snr_stabilizer_np(y_true, y_pred, epsi=1e-8):
    """SI-SNR based on cosine similarity including stabilizer using np.
        rho = abs(y_pred*y_true)/(|y_pred*y_pred|*|y_true*y_true|+epsi)
        SI-SNR ~= 10log10[(rho^2+epsi)/(1-rho^2+epsi)]
        Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, et al. What’s all the Fuss about Free Universal Sound
        Separation Data?. ICASSP 2021. https://doi.org/10.1109/ICASSP39728.2021.9414774.
    """
    rho = samerate_acc_stabilizer_np(y_true, y_pred, epsi)
    si_snr = 10 * np.log10((rho**2+epsi)/(1-rho**2+epsi))

    return si_snr


def si_snr_np(y_true, y_pred):
    """SI-SNR based on cosine similarity including stabilizer using np.
        rho = abs(y_pred*y_true)/(|y_pred*y_pred|*|y_true*y_true|)
        SI-SNR ~= 10log10[rho^2/(1-rho^2)]
        Scott Wisdom, Hakan Erdogan, Daniel P. W. Ellis, et al. What’s all the Fuss about Free Universal Sound
        Separation Data?. ICASSP 2021. https://doi.org/10.1109/ICASSP39728.2021.9414774.
    """
    rho = samerate_acc_np(y_true, y_pred)
    si_snr = 10 * np.log10(rho**2/(1-rho**2))

    return si_snr


def mse_np(y_true, y_pred):
    """MSE using numpy."""
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    square_errors = np.square(y_true-y_pred)  # pylint:disable=assignment-from-no-return
    mse = np.mean(square_errors)  # pylint:disable=unused-variable

    # return np.square(np.subtract(y_true, y_pred)).mean()
    return mse
