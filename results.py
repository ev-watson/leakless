import argparse
import os

import camb
import healpy as hp
import numpy as np
from matplotlib import pyplot as plt

import config
from models import Leakless
from utils import model_analysis, sample_normal

print(f'Using healpy {hp.__version__} installed at {os.path.dirname(hp.__file__)}')
print(f'Using CAMB {camb.__version__} installed at {os.path.dirname(camb.__file__)}')

parser = argparse.ArgumentParser(description="Model Results Analysis")
parser.add_argument("--ckpt_path", "-c", type=str, default=None)
parser.add_argument("--ntrials", "-n", type=int, default=1000)
args = parser.parse_args()

os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# The defaults give one massive neutrino and helium set using BBN consistency
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

sampled = sample_normal(params, best_fit=True)

pars = camb.set_params(**sampled)
results = camb.get_results(pars)
powers = results.get_cmb_power_spectra(pars, CMB_unit='muK', raw_cl=True)
lensedCL = powers['lensed_scalar']
lensedcl_e = lensedCL.T[1]
lensedcl_b = lensedCL.T[2]

nside = config.NSIDE
lmax = config.LMAX
seed = config.SEED
np.random.seed(seed)

mask = hp.read_map(config.MASK_FILE, field=1)
low_mask = hp.ud_grade(mask, nside_out=nside, dtype=np.int32)
fsky = np.mean(mask)

model = Leakless.load_from_checkpoint(args.ckpt_path)

analysis_dict = model_analysis(model=model,
                               ntrials=args.ntrials,
                               nside=nside,
                               lmax=lmax,
                               mask=low_mask,
                               outstream='results/output.txt')

cl_e_in = analysis_dict['e_in']
cl_b_in = analysis_dict['b_in']
cl_e_out = analysis_dict['e_out']
cl_b_out = analysis_dict['b_out']
cl_e_targ = analysis_dict['e_targ']
cl_b_targ = analysis_dict['b_targ']
cl_e_cross = analysis_dict['e_cross']
cl_b_cross = analysis_dict['b_cross']
cl_b_cross_coeff = analysis_dict['rho']
cl_e_leak_std = analysis_dict['e_in_std']
cl_b_leak_std = analysis_dict['b_in_std']
cl_e_pred_std = analysis_dict['e_out_std']
cl_b_pred_std = analysis_dict['b_out_std']
cl_e_true_std = analysis_dict['e_targ_std']
cl_b_true_std = analysis_dict['b_targ_std']
cl_e_cross_std = analysis_dict['e_cross_std']
cl_b_cross_std = analysis_dict['b_cross_std']
cl_b_cross_coeff_std = analysis_dict['rho_std']

plt.figure(figsize=(6, 6))
plt.errorbar(np.arange(lmax + 1), cl_e_in / fsky, yerr=cl_e_leak_std / fsky, label='E_input (leak)', fmt='o')
plt.errorbar(np.arange(lmax + 1), cl_b_in / fsky, yerr=cl_b_leak_std / fsky, label='B_input (leak)', fmt='o')
plt.errorbar(np.arange(lmax + 1), cl_e_out / fsky, yerr=cl_e_pred_std / fsky, label='E_output', fmt='o')
plt.errorbar(np.arange(lmax + 1), cl_b_out / fsky, yerr=cl_b_pred_std / fsky, label='B_output', fmt='o')
plt.errorbar(np.arange(lmax + 1), cl_e_targ / fsky, yerr=cl_e_true_std / fsky, label='E_true', fmt='o')
plt.errorbar(np.arange(lmax + 1), cl_b_targ / fsky, yerr=cl_b_true_std / fsky, label='B_true', fmt='o')
plt.xscale('log')
plt.yscale('log')
plt.plot(lensedcl_e)
plt.plot(lensedcl_b)
plt.xlim(2, lmax)
plt.legend()
plt.savefig('results/figures/cl_plot.png')

plt.figure(figsize=(6, 6))
plt.errorbar(np.arange(lmax + 1), cl_e_cross / fsky, yerr=cl_e_cross_std / fsky, label='E_in_out_cross', fmt='o')
plt.errorbar(np.arange(lmax + 1), cl_b_cross / fsky, yerr=cl_b_cross_std / fsky, label='B_in_out_cross', fmt='o')
plt.xscale('log')
plt.yscale('log')
plt.plot(lensedcl_e)
plt.plot(lensedcl_b)
plt.xlim(2, lmax)
plt.legend()
plt.savefig('results/figures/in_out_cross_plot.png')

plt.figure(figsize=(6, 6))
plt.errorbar(np.arange(lmax + 1), cl_b_cross_coeff, yerr=cl_b_cross_coeff_std,
             label=rf'$\bar{{\rho_B}}\approx{np.mean(cl_b_cross_coeff):.3g}\pm{np.mean(cl_b_cross_coeff_std):.3g}$', fmt='o')
plt.xlim(2, lmax)
plt.legend()
plt.savefig('results/figures/rho_plot.png')
