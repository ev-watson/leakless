import math

import numpy as np


def alm_len_from_nsides(nsides):
    """
    helper function to find len of an alm_array from nsides
    assuming lmax = 3*nsides - 1
    :param nsides: int, resolution of map
    :return: int, len of alm array
    """
    lmax = 3*nsides - 1
    return (lmax+1) * (lmax+2) // 2


def nsides_from_alm_len(alm_len):
    """
    helper function to find nsides from alm array len
    :param alm_len: int, len of alm array
    :return: int, resolution of map
    """
    return (math.sqrt(8*alm_len+1)-1)//6


def recombine(channel_tensor):
    """
    recombines E and B mode real and imaginary channels
    :param channel_tensor: array-like, must be shape (4, npoints), channels must be in order: E.re, E.im, B.re, B.im
    :return: tuple of complex E and B alm arrays, each with shape (npoints,)
    """
    e_r = channel_tensor[0]
    e_i = channel_tensor[1]
    b_r = channel_tensor[2]
    b_i = channel_tensor[3]

    e_comb = e_r + 1j*e_i
    b_comb = b_r + 1j*b_i
    return e_comb.astype(np.complex128), b_comb.astype(np.complex128)
