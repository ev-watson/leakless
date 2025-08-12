import math


def alm_len_from_lmax(lmax):
    """
    helper function to find len of an alm_array from lmax
    :param lmax: int, max ell mode
    :return: int, len of alm array
    """
    return int((lmax + 1) * (lmax + 2) // 2)


def alm_len_from_nside(nside):
    """
    helper function to find len of an alm_array from nside
    assuming lmax = 3*nside - 1
    :param nside: int, resolution of map
    :return: int, len of alm array
    """
    return alm_len_from_lmax(3 * nside - 1)


def nside_from_alm_len(alm_len):
    """
    helper function to find nside from alm array len
    :param alm_len: int, len of alm array
    :return: int, resolution of map
    """
    return int((math.sqrt(8 * alm_len + 1) - 1) // 6)


def lmax_from_alm_len(alm_len):
    """
    helper function to find lmax from alm array len
    :param alm_len: int, len of alm array
    :return: int, max ell mode
    """
    return int(3 * nside_from_alm_len(alm_len) - 1)


def recombine(channel_tensor):
    """
    recombines E and B mode real and imaginary channels
    :param channel_tensor: array-like, must be shape (4, npoints), channels must be in order: E.re, E.im, B.re, B.im
    :return: tuple of complex E and B alm arrays, each with shape (npoints,)
    """
    e_r = channel_tensor[:, 0]
    e_i = channel_tensor[:, 1]
    b_r = channel_tensor[:, 2]
    b_i = channel_tensor[:, 3]

    e_comb = e_r + 1j * e_i
    b_comb = b_r + 1j * b_i
    return e_comb, b_comb


def alm_span_from_m_band(lmax: int, band: tuple[int, int]) -> tuple[int, int]:
    """
    From Healpy docs:
    In HEALPix C++ and healpy, a_lm coefficients are stored ordered by m. I.e. if l_max
    is 16, the first 16 elements are m=0,l=0..16, then the following 15 elements are
    m=1,l=1..16 and so on until the last element, the 153th, is m=16,l=16

    this function notes the size up to m is (m+1)(lmax+1) + m(m+1)/2 and uses this to compute
    starting and ending indices of alm array to capture desired m band
    Args:
        lmax: int, max ell mode
        band: tuple[int, int], tuple with first element mmin and second element mmax

    Returns:
        tuple[int, int], tuple of starting/ending indices of alm array to capture desired m band
    """
    mmin, mmax = band
    if not (0 <= mmin <= mmax <= lmax + 1):
        raise ValueError("Require 0 <= mmin <= mmax <= lmax for healpy packed ordering.")

    def S(m: int):
        if m <= 0: return 0
        return int((m + 1) * (lmax + 1 - (m / 2)))

    start = S(mmin)
    end = S(mmax)
    return start, end
