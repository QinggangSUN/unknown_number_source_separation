# -*- coding: utf-8 -*-
"""
Created on Thu May 31 17:14:53 2018

prepare data from raw datasets
random split to train, validation, test sets

@author: SUN Qinggang
E-mail: sun10qinggang@163.com

"""
import logging
import numpy as np
np.random.seed(1337)

from error import Error, ParameterError


def balancesets(sets):
    """Cut to the min number of the element of the list.
    Args:
        sets (list[list[type]]): 2D list.
    Returns:
        sets (list[list[type]]): 2D list after balance numbers.
    """
    lis = [len(si) for si in sets]
    num = min(lis)

    for i in range(len(sets)):  # pylint: disable=consider-using-enumerate
        sets[i] = sets[i][0:num]
    return sets  # dimension not changed


class Subsets(object):
    """Split datas in to train, val, test subsets."""

    def __init__(self, rates, ndata):
        self.rates = rates
        self.ndata = ndata
        self.randseq = list(range(ndata))
        np.random.shuffle(self.randseq)

    def randsubsets(self, data):
        """Input one source list of data,
        output 2D list of data[subseti][datai]"""
        if not self.ndata == len(data):
            raise Exception("self.ndata != len(data)")
        randseq = self.randseq
        subs = []
        for i in range(len(self.rates)-1):
            numi = int(self.rates[i]*self.ndata)  # num of subsets[i]
            subs.append([data[randseq[j]]for j in range(numi)])
            randseq = randseq[numi:]
        subs.append([data[randseq_j] for randseq_j in randseq])
        return subs

    def ordersubsets(self, data):
        """Input one source list of data,
        output 2D list of data[subseti][datai]"""
        if not self.ndata == len(data):
            raise Exception("self.ndata != len(data)")
        subs = []
        istart = 0
        for i in range(len(self.rates)-1):
            numi = int(self.rates[i]*self.ndata)  # num of subsets[i]
            subs.append(data[istart:istart+numi])
            istart += numi
        subs.append(data[istart:])
        return subs

    def randsubsetsnums(self, ndata):
        """Input num of one source,
        output 2D list of [subseti][numi]"""
        randseq = list(range(ndata))
        np.random.shuffle(randseq)  # warning, this is different with self.randseq
        subs = []
        for i in range(len(self.rates)-1):
            numi = int(self.rates[i]*ndata)  # num of subsets[i]
            subs.append(randseq[:numi])
            randseq = randseq[numi:]
        subs.append(randseq)
        return subs

    def rand_subsets_nums_with_randseq(self, ndata):
        """Input num of one source,
        output 2D list of [subseti][numi]"""
        randseq = self.randseq.copy()
        subs = []
        for i in range(len(self.rates)-1):
            numi = int(self.rates[i]*ndata)  # num of subsets[i]
            subs.append(randseq[:numi])
            randseq = randseq[numi:]
        subs.append(randseq)
        return subs

    def ordersubsetsnums(self, ndata):
        """Input num of one source,
        output 2D list of [subseti][numi]"""
        subs = []
        istart = 0
        for i in range(len(self.rates)-1):
            numi = int(self.rates[i]*ndata)  # num of subsets[i]
            subs.append(list(range(istart, istart+numi)))
            istart += numi
        subs.append(list(range(istart, ndata)))
        return subs

    def select_subsets_nums(self, ndata, rates, num_sets=None):
        """Input num of one source,
        output 2D list of [subseti][numi]"""
        randseq = self.randseq.copy()
        subs = []
        for i in range(len(self.rates)-1):
            numi = int(self.rates[i]*ndata)  # num of subsets[i]
            subs.append(randseq[:numi])
            randseq = randseq[numi:]
        subs.append(randseq)
        return subs

    def get_randseq(self):
        return self.randseq


def shuffle_sets(ni_3d):
    """Input 3D nums [source][subset][numi], return shuffled sets."""
    nsrc = len(ni_3d)
    nsub = len(ni_3d[0])
    nsets = []
    for subj in range(nsub):
        ni_setj = []
        for si in range(nsrc):  # pylint: disable=invalid-name
            for nk in ni_3d[si][subj]:  # pylint: disable=invalid-name
                ni_setj.append((si, nk))
        nsamples = len(ni_setj)
        randseq = list(range(nsamples))
        np.random.shuffle(randseq)
        rand_ni_setj = [ni_setj[randseq[sami]]for sami in range(nsamples)]
        nsets.append(rand_ni_setj)
    return nsets  # return 2D list [subset][(source, numi)]


def mixaddframes_np(frames):
    """Input 2D list frames [source][frames][fl],
    output 1D list of mix using average add"""
    nsrc = len(frames)  # number of sources
    if nsrc == 1:
        return np.asarray(frames[0], dtype=np.float32)

    # mix = np.sum(frames, axis=0)/np.float32(nsrc)
    mix = np.average(frames, axis=0)
    return np.asarray(mix, dtype=np.float32)


def snr_np(x1, x2):
    """Computing SNR using numpy.

    Args:
        x1 (np.ndarray, shape==(1,)): signal.
        x2 (np.ndarray, shape==(1,)): noise.

    Returns:
        float: SNR in dB.
    """
    import numpy as np
    return 20 * np.log10(np.sum(x1**2) / np.sum(x2**2))


def mix_add_snr(x, snr):
    """Mix signals with snr.

    Args:
        x (list[np.ndarray, shape=(1,)]): sources to be mixed.
        snr (list[float]): SNR of source_0 to source_1,...,source_n.

    Returns:
        mix: the mixed signal.
        x_standard: all the signals after standard to the SNR.
    """

    import numpy as np

    assert len(x) == len(snr)+1
    x_standard = [x[0]]
    for i, x_i in enumerate(x[1:]):
        norm_x0 = np.linalg.norm(x[0], ord=2)
        norm_xi = np.linalg.norm(x_i, ord=2)
        mag = 10.0 ** (0.05 * snr[i])
        x_standard.append(x_i * norm_x0 / (norm_xi * mag))
    mix = np.sum(x_standard)

    return mix, x_standard


def ld3_to_ld2(ld3):
    """3D list to 2D list.
    Input: list[m][n_1...n_m][p], output: [n_1+...+n_m][p]
    """
    ld2 = []
    for ld3i in ld3:
        ld2 += ld3i
    return ld2


def list_transpose(ld2, warn=False):
    """Transpose a 2D list.
    Args:
        ld2 (list[list[]]): a 2D list.
        warn (bool, optional): whether generate a warnning. Default to False.
    Returns:
        ld2_t (list[list[]]): a 2D list transposed.
    Examples:
        >>> a = [[1, 2, 3], [4, 5], [7, 8, 9]]
        >>> print(list_transpose(a, warn=True))
        WARNING:root:length of sub list not equal
        [[1, 4, 7], [2, 5, 8], [3, 9]]
    """
    max_len = max([len(ld2_j) for ld2_j in ld2])
    if warn:
        if not len(set([len(ld2_j) for ld2_j in ld2])) == 1:
            logging.warning('length of sub list not equal')
    ld2_t = [[] for _ in range(max_len)]

    for ld2_j in ld2:
        for i in range(len(ld2_j)):
            ld2_t[i].append(ld2_j[i])
    return ld2_t


def nhot_3to4(nhot_3):
    """Input: np.ndarray shape (1,3)
    Return: np.ndarray shape (1, 4)."""
    nhot_4 = np.full((1, 4), 0)
    nhot_4[0, 1:] = nhot_3
    nhot_4[0, 0] = 0 if np.any(nhot_3) else 1
    return nhot_4


def filter_data(x, y, condiction='one'):
    """Filter specific_data."""
    if condiction == 'one':
        index = np.where(y == 1)[0]
        logging.debug(''.join(['filter_data index: ', str(index)]))
        x_filter = x[index]
    return x_filter
