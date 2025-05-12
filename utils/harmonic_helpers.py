import math


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
