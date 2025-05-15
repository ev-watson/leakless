import numpy as np
import torch

import config
from utils.logging_utils import print_block
from utils.losses import calc_mae, calc_mape


def print_analysis(g, t, ntrials, mape, suppress, err, verbose, axis=None):
    """
    Helper function for random tests
    """
    mae = calc_mae(g, t, axis=axis)
    if not suppress:
        print_block(f"RANDOM INPUT TESTING TRIALS: {ntrials}", err=err)
        print_block(f"MAE: {mae:.6g}", err=err)
        if verbose:
            print_block("PREDICTIONS:", err=err)
            print(g)
            print_block("TARGETS:", err=err)
            print(t)

    if mape:
        mape_val = calc_mape(g, t, axis=axis)
        if not suppress:
            print_block(f"MAPE: {mape_val:.6g}%", err=err)
        return mae, mape_val
    else:
        return mae


def leak_test(model, ntrials=100, batch_size=config.BATCH_SIZE, hopt=False, suppress=False, err=False, verbose=False, mean_axis=None):
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
    :param mean_axis: int, axis along which to calculate analysis statistics, None for entire array
    :return: tuple of (input, output, targets) tensors
    """
    if not suppress:
        print_block("BEGINNING RANDOM BATCH TESTING", err=err)

    stack = np.load(config.DATA_FILE)
    input_slice = slice(None, config.IN_CHANNELS)
    target_slice = slice(config.IN_CHANNELS, None)
    npoints = stack.shape[-1]

    idx = np.random.randint(len(stack), size=ntrials)
    sampled = np.array([stack[i] for i in idx])  # shape [ntrials, 8, npoints]
    pred_vals = torch.empty((ntrials, config.IN_CHANNELS, npoints))

    device = next(model.parameters()).device
    input_data = torch.from_numpy(sampled[:, input_slice, :]).to(device=device, dtype=torch.get_default_dtype())
    target_data = torch.from_numpy(sampled[:, target_slice, :]).to(device=device, dtype=torch.get_default_dtype())
    pred_vals = pred_vals.to(device)

    # Finds closest power of 2 that will make batch_size and num_batches as even as possible then multiplies by 2
    # batch_size = 2 ** (round(np.log2(np.sqrt(ntrials)))+1)
    batch_size = batch_size
    num_batches = int(np.ceil(ntrials / batch_size))
    if not suppress:
        print_block(f"{ntrials} TRIALS, {num_batches} BATCHES of {batch_size} SIZE", err=err)
    for batch_num in range(num_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, ntrials)
        batch_input = input_data[start_idx:end_idx]
        pred_vals[start_idx:end_idx] = model.predict(batch_input)

    if not suppress:
        print_block("TESTING COMPLETE, BEGINNING ANALYSIS", err=err)

    mae = print_analysis(pred_vals, target_data, ntrials, False, suppress, err, verbose, axis=mean_axis)
    if hopt:
        return mae
    else:
        return input_data.cpu().numpy(), pred_vals.cpu().numpy(), target_data.cpu().numpy()
