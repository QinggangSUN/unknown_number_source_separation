# -*- coding: utf-8 -*-
"""
Created on Sun May  8 16:42:18 2022

@author: SUN Qinggang

E-mail: sun10qinggang@163.com

"""
from file_operation import mkdir
from prepare_data_shipsear_separation_mix_s0tos3 import create_snrs
from see_metric_autoencoder import display_metric

if __name__ == '__main__':

    n_sams = 5093
    n_src = 2
    snrs = create_snrs('uniform', -5, 5, n_sams, n_src)
    snrs_1_2, snrs_1_3 = snrs[0], snrs[1]
    snrs_2_3 = snrs_1_3 - snrs_1_2

    snrs = [snrs_1_2, snrs_1_3, snrs_2_3]

    PATH_SNR = '../result_snr_separation'
    mkdir(PATH_SNR)

    display_metric(snrs, PATH_SNR, ['s_1_2', 's_1_3', 's_2_3'], hist_bins=(-10, 10, 100), save_name='SNR')
