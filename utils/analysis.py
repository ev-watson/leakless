import sys

import healpy as hp
import numpy as np
import torch
from tqdm import tqdm

import config
from .harmonic_helpers import alm_len_from_lmax, nside_from_alm_len, lmax_from_alm_len, recombine
from .logging_utils import print_block
from .losses import calc_mae, calc_mape, SpectralBinLoss


def rho_metric(input_vector, output_vector):
    """
    calculates the deviation of the cross correlation coefficient between input and output (rho) from 1
    Args:
        input_vector: array-like, shape (ntrials, alm_len, 4)
        output_vector: array-like, shape (ntrials, alm_len, 4)

    Returns:
        deviation: float, 1-rho
    """
    ntrials = input_vector.shape[0]
    alm_len = input_vector.shape[-2]
    nside = nside_from_alm_len(alm_len)
    lmax = lmax_from_alm_len(alm_len)

    mask = hp.read_map(config.MASK_FILE, field=1)
    mask = hp.ud_grade(mask, nside_out=nside, dtype=np.int32)

    # recombines the imaginary and real parts to one number
    alm_len = alm_len_from_lmax(lmax)
    alm_b_in = np.zeros((ntrials, alm_len), dtype=np.complex128)
    alm_b_out = np.zeros((ntrials, alm_len), dtype=np.complex128)
    for i in range(ntrials):
        alm_b_in[i] = recombine(input_vector[i])[1]
        alm_b_out[i] = recombine(output_vector[i])[1]

    # make maps from alm arrays and calc cross coeff (rho)
    b_in = np.zeros((ntrials, lmax + 1), dtype=np.float64)
    b_out = np.zeros((ntrials, lmax + 1), dtype=np.float64)
    b_cross = np.zeros((ntrials, lmax + 1), dtype=np.float64)
    b_cross_coeff = np.zeros((ntrials, lmax + 1), dtype=np.float64)
    for i in range(ntrials):
        b_in_masked_map = hp.alm2map(alm_b_in[i], nside=nside) * mask
        b_out_masked_map = hp.alm2map(alm_b_out[i], nside=nside) * mask
        b_in[i] = hp.anafast(b_in_masked_map, lmax=lmax)
        b_out[i] = hp.anafast(b_out_masked_map, lmax=lmax)

        b_cross[i] = hp.anafast(b_in_masked_map, b_out_masked_map, lmax=lmax)

        b_cross_coeff[i] = b_cross[i] / np.sqrt(b_in[i] * b_out[i])

    # take averages of all trials for all ell modes
    cl_b_cross_coeff = np.mean(b_cross_coeff)

    return 1 - cl_b_cross_coeff


def print_analysis(metric_obj, g, t, ntrials, suppress, err, verbose):
    """
    Helper function for random tests

    :param metric_obj: obj that can be passed (g, t)
    :param g: guess vector
    :param t: target vector
    :param ntrials: int, number of trials
    :param suppress: bool, if true no print statements
    :param err: bool, enable printing to stderr as well
    :param verbose: bool, enable verbose output
    """
    metric = metric_obj(g, t)
    if not suppress:
        print_block(f"RANDOM INPUT TESTING TRIALS: {ntrials}", err=err)
        print_block(f"METRIC: {metric:.6g}", err=err)
        if verbose:
            print_block("PREDICTIONS:", err=err)
            print(g)
            print_block("TARGETS:", err=err)
            print(t)

    return metric


def leak_test(model, ntrials=100, batch_size=config.BATCH_SIZE, hopt=False, suppress=False, err=False, verbose=False):
    """
    Performs random input testing to evaluate the accuracy of GNN's acceleration prediction.
    ntrials will be rounded to nearest i
    :param model: PyTorch model
    :param ntrials: int, number of trials
    :param batch_size: int, batch size for prediction, default 16
    :param hopt: bool, returns mae instead of tensors for hyper-optimization
    :param suppress: bool, if true, suppresses print statements.
    :param err: bool, enable printing to stderr as well.
    :param verbose: bool, enable verbose output
    :return: tuple of (input, output, targets) tensors
    """
    if not suppress:
        print_block("BEGINNING RANDOM BATCH TESTING", err=err)

    neff = (ntrials // batch_size) * batch_size
    stack = np.load(config.DATA_FILE, mmap_mode='r')
    input_slice = slice(None, config.INPUT_DIM)
    target_slice = slice(config.INPUT_DIM, None)
    npoints = stack.shape[1]

    idx = np.random.randint(len(stack), size=neff)
    sampled = np.array([stack[i] for i in idx])  # shape [ntrials, npoints, 8]
    pred_vals = torch.empty((neff, npoints, config.INPUT_DIM), dtype=torch.float32)

    device = next(model.parameters()).device
    input_data = torch.from_numpy(sampled[..., input_slice]).to(device=device, dtype=torch.float32)
    target_data = torch.from_numpy(sampled[..., target_slice]).to(device=device, dtype=torch.float32)
    pred_vals = pred_vals.to(device)

    # Finds closest power of 2 that will make batch_size and num_batches as even as possible then multiplies by 2
    # batch_size = 2 ** (round(np.log2(np.sqrt(ntrials)))+1)
    batch_size = batch_size
    num_batches = neff // batch_size
    if not suppress:
        print_block(f"{neff} TRIALS, {num_batches} BATCHES of {batch_size} SIZE", err=err)
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, neff)
        batch_input = input_data[start_idx:end_idx]
        pred_vals[start_idx:end_idx] = model.predict(batch_input)

    if not suppress:
        print_block("TESTING COMPLETE, BEGINNING ANALYSIS", err=err)

    # keep torch for metric calc
    metric = print_analysis(rho_metric, pred_vals, target_data, neff, suppress, err, verbose)

    # drop torch
    input_vector = input_data.cpu().numpy()
    output_vector = pred_vals.cpu().numpy()
    target_vector = target_data.cpu().numpy()

    if hopt:
        return metric
    else:
        return input_vector, output_vector, target_vector


def model_analysis(model, ntrials, nside, lmax, mask, outstream=sys.stdout):
    """
    Takes in a model checkpoint, runs map and final power spectrum analysis, returns tensors for plotting
    Args:
        model: PyTorch model
        ntrials: int, number of trials
        nside: int, HEALPix nside
        lmax: int, HEALPix lmax
        mask: obj, mask to apply to maps
        outstream: output stream

    Returns: dict with all arrays necessary for plotting

    """
    # writes a text based illustration of the model to outstream
    with open(outstream, 'a') as f:
        print(model.hparams, file=f)
        print(model, file=f)

    neff = (ntrials // config.BATCH_SIZE) * config.BATCH_SIZE

    # grabs necessary tensors
    input_vector, output_vector, target_vector = leak_test(model, ntrials=neff)

    # recombines the imaginary and real parts to one number
    alm_len = alm_len_from_lmax(lmax)
    alm_e_in = np.zeros((neff, alm_len), dtype=np.complex128)
    alm_b_in = np.zeros((neff, alm_len), dtype=np.complex128)
    alm_e_out = np.zeros((neff, alm_len), dtype=np.complex128)
    alm_b_out = np.zeros((neff, alm_len), dtype=np.complex128)
    alm_e_targ = np.zeros((neff, alm_len), dtype=np.complex128)
    alm_b_targ = np.zeros((neff, alm_len), dtype=np.complex128)
    for i in range(neff):
        alm_e_in[i], alm_b_in[i] = recombine(input_vector[i])
        alm_e_out[i], alm_b_out[i] = recombine(output_vector[i])
        alm_e_targ[i], alm_b_targ[i] = recombine(target_vector[i])

    # make maps from alm arrays and calc cross coeff (rho)
    e_in = np.zeros((neff, lmax + 1), dtype=np.float64)
    b_in = np.zeros((neff, lmax + 1), dtype=np.float64)
    e_out = np.zeros((neff, lmax + 1), dtype=np.float64)
    b_out = np.zeros((neff, lmax + 1), dtype=np.float64)
    e_targ = np.zeros((neff, lmax + 1), dtype=np.float64)
    b_targ = np.zeros((neff, lmax + 1), dtype=np.float64)
    e_cross = np.zeros((neff, lmax + 1), dtype=np.float64)
    b_cross = np.zeros((neff, lmax + 1), dtype=np.float64)
    b_cross_coeff = np.zeros((neff, lmax + 1), dtype=np.float64)
    for i in tqdm(range(neff)):
        e_in_masked_map = hp.alm2map(alm_e_in[i], nside=nside) * mask
        b_in_masked_map = hp.alm2map(alm_b_in[i], nside=nside) * mask
        e_out_masked_map = hp.alm2map(alm_e_out[i], nside=nside) * mask
        b_out_masked_map = hp.alm2map(alm_b_out[i], nside=nside) * mask
        e_targ_masked_map = hp.alm2map(alm_e_targ[i], nside=nside) * mask
        b_targ_masked_map = hp.alm2map(alm_b_targ[i], nside=nside) * mask
        e_in[i] = hp.anafast(e_in_masked_map, lmax=lmax)
        b_in[i] = hp.anafast(b_in_masked_map, lmax=lmax)
        e_out[i] = hp.anafast(e_out_masked_map, lmax=lmax)
        b_out[i] = hp.anafast(b_out_masked_map, lmax=lmax)
        e_targ[i] = hp.anafast(e_targ_masked_map, lmax=lmax)
        b_targ[i] = hp.anafast(b_targ_masked_map, lmax=lmax)
        e_cross[i] = hp.anafast(e_in_masked_map, e_out_masked_map, lmax=lmax)
        b_cross[i] = hp.anafast(b_in_masked_map, b_out_masked_map, lmax=lmax)

        b_cross_coeff[i] = b_cross[i] / np.sqrt(b_in[i] * b_out[i])

    # take averages of all trials
    cl_e_in = np.mean(e_in, axis=0)
    cl_b_in = np.mean(b_in, axis=0)
    cl_e_out = np.mean(e_out, axis=0)
    cl_b_out = np.mean(b_out, axis=0)
    cl_e_targ = np.mean(e_targ, axis=0)
    cl_b_targ = np.mean(b_targ, axis=0)
    cl_e_cross = np.mean(e_cross, axis=0)
    cl_b_cross = np.mean(b_cross, axis=0)
    cl_b_cross_coeff = np.mean(b_cross_coeff, axis=0)
    cl_e_leak_std = np.std(e_in, axis=0)
    cl_b_leak_std = np.std(b_in, axis=0)
    cl_e_pred_std = np.std(e_out, axis=0)
    cl_b_pred_std = np.std(b_out, axis=0)
    cl_e_true_std = np.std(e_targ, axis=0)
    cl_b_true_std = np.std(b_targ, axis=0)
    cl_e_cross_std = np.std(e_cross, axis=0)
    cl_b_cross_std = np.std(b_cross, axis=0)
    cl_b_cross_coeff_std = np.std(b_cross_coeff, axis=0)

    output = {
        'e_in': cl_e_in,
        'b_in': cl_b_in,
        'e_out': cl_e_out,
        'b_out': cl_b_out,
        'e_targ': cl_e_targ,
        'b_targ': cl_b_targ,
        'e_cross': cl_e_cross,
        'b_cross': cl_b_cross,
        'rho': cl_b_cross_coeff,
        'e_in_std': cl_e_leak_std,
        'b_in_std': cl_b_leak_std,
        'e_out_std': cl_e_pred_std,
        'b_out_std': cl_b_pred_std,
        'e_targ_std': cl_e_true_std,
        'b_targ_std': cl_b_true_std,
        'e_cross_std': cl_e_cross_std,
        'b_cross_std': cl_b_cross_std,
        'rho_std': cl_b_cross_coeff_std,
    }

    return output
