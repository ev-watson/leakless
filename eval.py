import argparse
import numpy as np
import torch
import pymaster as nmt

import config
from models import Leakless


def build_window(mask_path: str, apodization_deg: float):
    mask = nmt.mask_read_healpix(mask_path, nest=False)
    window = nmt.mask_apodization(mask, apodization_deg, apotype="C2")
    return window


def build_workspace(window, nside: int, n_bins: int):
    # Initialize binning and compute coupling matrix once
    binning = nmt.NmtBin(nside, n_bins)
    # Dummy field needed to build the workspace
    zeros = np.zeros(nside_to_size(nside))
    field = nmt.NmtField(window, [zeros, zeros], purify_b=True)
    workspace = nmt.NmtWorkspace()
    workspace.compute_coupling_matrix(field, field, binning)
    return workspace, binning


def nside_to_size(nside: int):
    return 12 * nside * nside


def compute_bandpower(window, workspace, binning, Q_map, U_map):
    field = nmt.NmtField(window, [Q_map, U_map], purify_b=True)
    cl = nmt.compute_full_master(field, field, workspace)
    # cl = [EE, EB, BE, BB]
    ells = binning.get_effective_ells()
    return ells, cl[3]


def evaluate(model, masked_path: str, true_path: str, mask_path: str,
             nside: int, n_bins: int, apod: float):
    # Load maps
    Q_mask, U_mask = load_q_u(masked_path)
    Q_true, U_true = load_q_u(true_path)

    # Predict full-sky Q/U
    Q_pred, U_pred = model.predict(np.stack([Q_mask, U_mask], axis=0))

    # Build estimator
    window = build_window(mask_path, apod)
    workspace, binning = build_workspace(window, nside, n_bins)

    # Compute true and predicted bandpowers
    ell, bb_true = compute_bandpower(window, workspace, binning, Q_true, U_true)
    _, bb_pred = compute_bandpower(window, workspace, binning, Q_pred, U_pred)

    # Metrics
    delta = bb_pred - bb_true
    frac_err = delta / bb_true

    # Print results
    print("ell, BB_true, BB_pred, delta, frac_err")
    for l, t, p, d, fe in zip(ell, bb_true, bb_pred, delta, frac_err):
        print(f"{l:.0f}, {t:.3e}, {p:.3e}, {d:.3e}, {fe:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate model predictions and test performance.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint file.")
    parser.add_argument("--data", type=str, required=True, help="Path to .npy data for prediction stage.")
    parser.add_argument("--test-data", type=str, required=True, help="Path to .npy test dataset.")
    parser.add_argument("--trials", type=int, default=1, help="Number of prediction trials to average.")
    parser.add_argument("--skip-predict", action="store_true", help="Skip the prediction evaluation stage.")
    parser.add_argument("--skip-test", action="store_true", help="Skip the test dataset evaluation stage.")
    args = parser.parse_args()

    model = Leakless.load_from_checkpoint(ckpt_path)
    data = np.load(config.DATA_FILE)

    if not args.skip_predict:
        data = load_data(args.data)
        evaluate_predictions(model, data, args.trials)
    else:
        print("Skipping prediction stage.")

    if not args.skip_test:
        test_data = load_data(args.test_data)
        evaluate_on_test(model, test_data)
    else:
        print("Skipping test stage.")


if __name__ == "__main__":
    main()
