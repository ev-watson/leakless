import random

import camb
import healpy as hp
import numpy as np
from tqdm import tqdm

import config
from utils import sample_normal, print_block, alm_len_from_nsides


def generate_stack(seed, cls, nsides, mask):
    """
    Generate data stack for CNN model

    orders columns as
    [E_partial.real, E_partial.imag, B_partial.real, B_partial.imag, E_full.real, E_full.imag, B_full.real, B_full.imag]

    :param seed: int, seed for stack
    :param cls: list, list of angular power spectra in order [TT, EE, BB, TE]
    :param nsides: int, resolution of map
    :param mask: array, binary mask to be applied to full sky map
    :return: shape [8, N] np.ndarray
    """
    # seed before synfast call
    np.random.seed(seed)

    # full maps
    tqu = hp.synfast(cls.T, nside=nsides, new=True)
    teb = hp.map2alm(tqu)
    emode = teb[1]
    bmode = teb[2]
    emap = hp.alm2map(emode, nside=nsides)
    bmap = hp.alm2map(bmode, nside=nsides)
    e_mask_true = emap * mask
    b_mask_true = bmap * mask
    e_alm_true = hp.map2alm(e_mask_true)
    b_alm_true = hp.map2alm(b_mask_true)

    # partial maps
    tqu_mask = tqu * mask
    teb_mask = hp.map2alm(tqu_mask)
    e_alm_guess = teb_mask[1]
    b_alm_guess = teb_mask[2]

    # build stack
    stack = np.stack([e_alm_guess.real, e_alm_guess.imag,
                      b_alm_guess.real, b_alm_guess.imag,
                      e_alm_true.real, e_alm_true.imag,
                      b_alm_true.real, b_alm_true.imag,], axis=0)

    return stack


def main():
    nsides = config.NSIDES
    n_samples = config.STACK_SIZE

    print_block(f"NSIDES: {nsides} // {n_samples} samples")

    params = {
        'H0': (67.66, 0.42),
        'ombh2': (0.02242, 0.00014),
        'omch2': (0.11933, 0.00091),
        'tau': (0.0561, 0.0071),
        'As': (2.105e-09, 3.000e-11),
        'ns': (0.9665, 0.0038),
        'mnu': 0.06,
        'omk': 0,
        'halofit_version': 'mead',
        'lmax': 3000,
    }

    # sample param space
    sampled = sample_normal(params, best_fit=True)

    # generated lensed angular power spectrum
    pars = camb.set_params(**sampled)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
    lensedCL = powers['lensed_scalar']

    # prepare mask
    mask = hp.read_map("binary_GAL_mask_N1024.fits", field=1)
    low_mask = hp.ud_grade(mask, nside_out=nsides, dtype=np.int32)

    # ensures unique seeds
    seeds = random.sample(range(2 ** 32 - 1), k=n_samples)
    stacks = np.empty((n_samples, 8, alm_len_from_nsides(nsides)), dtype=np.float64)
    for i, seed in enumerate(tqdm(seeds, desc="Generating stacks") if config.MAC else range(len(seeds))):
        stacks[i] = generate_stack(seed, lensedCL, nsides=nsides, mask=low_mask)

    # save file
    np.save("stacks.npy", stacks)

    print_block("SAVED STACKS TO: stacks.npy")


if __name__ == "__main__":
    main()
