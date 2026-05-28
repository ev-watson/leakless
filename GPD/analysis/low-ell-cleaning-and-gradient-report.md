# Low-Ell Cleaning, NaMaster Baseline, and Training-Gradient Report

Date: 2026-05-11

## Project Target

The scientific target is not merely estimating a B-mode power spectrum. The target is a model that accepts a partial-sky polarization observation and returns a leakage-cleaned partial-sky product. The working strategy is:

1. Generate a full-sky CMB realization.
2. Derive the full-sky E/B fields, which are leakage-free by construction.
3. Apply the experiment mask to those full-sky E/B fields.
4. Train the model to map the masked observed/partial-sky E/B coefficients to the masked leakage-free E/B coefficients.

The current `data_init.py` matches this strategy. Its input channels are E/B coefficients from masked Q/U maps:

```text
tqu_mask = tqu * mask
teb_mask = hp.map2alm(tqu_mask, lmax=lmax)
```

Its target channels are E/B coefficients after first decomposing the full-sky map and then cutting the true E/B maps:

```text
emap = hp.alm2map(emode, nside=nside)
bmap = hp.alm2map(bmode, nside=nside)
e_mask_true = emap * mask
b_mask_true = bmap * mask
e_alm_true = hp.map2alm(e_mask_true, lmax=lmax)
b_alm_true = hp.map2alm(b_mask_true, lmax=lmax)
```

So the model is learning a leakage-free cut-sky target, not a literal all-sky inpainted map. That is a good and defensible supervised target for the user-facing product: "give me the masked region back, but with E-to-B leakage removed."

## What NaMaster Pure-B Is

NaMaster is a pseudo-Cl power-spectrum estimation package. Its pure-B mode is not a map-cleaning neural network replacement. It does not take a partial-sky map and return a cleaned B map. Instead, it estimates B-mode bandpowers from masked Q/U maps while suppressing E-to-B mixing through pure-mode projection and mode-coupling correction.

The key conceptual distinction:

- Leakless output: a cleaned map/alm-like object, usable as a reconstructed field.
- NaMaster pure-B output: leakage-suppressed BB bandpowers, usable as a science estimator and benchmark.

NaMaster's `NmtField` accepts HEALPix spin-2 Q/U maps, and exposes `purify_b=True`. The docs state that spin>0 fields use Q/U Stokes maps with the HEALPix polarization convention, and that `purify_b` asks the field to purify B modes. The docs also warn that `masked_on_input=True` is not advisable with purification because it can bias spectra near mask edges. The pure-B example says the pure-B formalism requires a differentiable mask boundary, using C1/C2 apodization.

This lines up with the literature. Smith's pure pseudo-Cl estimator was designed because ordinary pseudo-Cl E/B mixing can dominate B-mode estimation on finite sky cuts. Grain, Tristram, and Stompor extended the pure pseudo-spectrum approach to cross-spectra and discuss residual leakage, apodization, and variance tradeoffs.

Sources:

- NaMaster `NmtField` API: https://namaster.readthedocs.io/en/latest/api/pymaster.field.html
- NaMaster pure E/B example: https://namaster.readthedocs.io/en/latest/source/sample_pureb.html
- Smith 2006, Pure pseudo-Cl estimators for CMB B-modes: https://arxiv.org/abs/astro-ph/0608662
- Grain, Tristram, Stompor 2009, pure pseudo-cross-spectrum approach: https://arxiv.org/abs/0903.2350

## NaMaster Compatibility With This Project

Local environment checked:

```text
pymaster 2.3.3
NSIDE = 32
LMAX = 95
npix = 12288
alm_len = 4656
stored sample shape = (4656, 8)
```

A smoke test succeeded using this conversion:

1. Load one `stacks.npy` sample.
2. Recombine project E/B real/imag channels into complex alms.
3. Convert E/B alms to Q/U maps with `hp.alm2map([T0, E, B], pol=True)`.
4. Build an apodized mask with `nmt.mask_apodization(..., apotype=config.MASK_APOTYPE)`.
5. Build `NmtField(mask, [Q, U], purify_b=True)`.
6. Build `NmtWorkspace` and compute BB bandpowers.

The local test passed for all dtype combinations:

```text
float64 mask/maps OK
float64 mask/float32 maps OK
float32 mask/maps OK
```

NaMaster internally accepts the project dimensions and accepts float32 inputs, though Healpy map transforms still produce/expect float64 or complex128 at the analysis boundary. This reinforces the dtype policy: float32 is fine for ML storage and tensors; Healpy/NaMaster analysis can promote to float64/complex128.

Important caveat: the cleanest NaMaster baseline should use the original masked Q/U maps plus mask, not only E/B alms reconstructed back to Q/U. The current stack stores E/B alms, so the test proves compatibility but is not the ideal baseline input. For a high-quality baseline, preserve or regenerate Q/U maps during evaluation, or add an evaluation-only path that rebuilds them directly inside `generate_stack`.

## What NaMaster Should Be Used For

NaMaster pure-B is an excellent baseline for the scientific output, but not a direct target replacement.

Use it for:

- BB bandpower baseline on the same masked Q/U input.
- Standard pseudo-Cl vs pure-B comparison.
- Model-output BB bandpower comparison.
- Cross-spectrum checks if independent simulations/noise splits are introduced.
- Low-ell uncertainty reference, because pure-B explicitly handles the ambiguous/pure-mode problem.

Do not use it as:

- A direct map cleaner.
- A full-sky inpainting method.
- A differentiable training loss, at least not through PyMaster itself.

The key evaluation table should have rows by ell band:

```text
band
standard pseudo-B input vs target BB error
NaMaster pure-B input vs target BB error
model output vs target BB error
model output-target rho
input-target rho
bootstrap confidence interval
shuffled/null baseline
```

That table would make the project claim much sharper: the model is useful if it beats pure-B in regimes where map-level reconstruction matters, or if it matches pure-B bandpowers while additionally returning a useful cleaned field.

## Why Bandpower Losses Often Underperform

The current coefficient-space loss has likely worked best because it is dense, stable, and phase-preserving. Every alm coefficient contributes a direct gradient, including the sign/phase information needed to reconstruct a particular realization.

Bandpower-style losses often fail for this problem for several reasons:

1. Bandpowers are quadratic summaries. Many wrong maps can have the same BB spectrum.
2. Low ell has few modes, so gradients are high-variance and realization-dependent.
3. Relative errors explode when target BB power is small.
4. Log-ratio losses can overemphasize tiny bands and destabilize training.
5. Healpy/NaMaster paths are NumPy/C and not differentiable through PyTorch.
6. Applying a mask in map space is nonlocal in harmonic space; naive alm bandpower loss does not fully represent the cut-sky coupling.
7. A pure spectrum objective may improve spectra while degrading map-level reconstruction.

This is why the right direction is probably not "replace MSE with bandpower loss." The safer direction is "keep coefficient reconstruction loss as the backbone, add a carefully bounded physics auxiliary."

## Recommended Training-Gradient Design

### Primary Loss: Weighted Coefficient Reconstruction

Keep coefficient-domain loss as the main term:

```text
L_coeff = mean_lm w_l * |B_pred_lm - B_target_lm|^2
```

For the current 4-channel output, apply this to all channels or to B channels with a larger weight. The key improvement is ell-aware weighting. Low ell should receive more deliberate weight, but not so much that the model chases cosmic-variance noise.

Suggested stable weighting:

```text
w_l = band_weight(l) / (sqrt(C_l_target_smooth) + floor)
```

or simpler:

```text
w_l = band_weight(l)
```

with manually bounded band weights. Avoid unbounded inverse-power weighting.

Why this should work: it keeps per-realization phase information while making low ell less invisible to the optimizer.

### Auxiliary Loss A: Differentiable Alm Bandpower Loss

Implement a pure PyTorch bandpower summary from packed alms:

```text
C_l(B) = sum_m |B_lm|^2 / (2l + 1)
```

Then use a bounded relative or log-cosh error by band:

```text
L_cl = mean_band alpha_band * logcosh((C_band_pred - C_band_target) / scale_band)
```

This can be differentiable without Healpy because the packed `(l,m)` index map is known and fixed for `LMAX=95`. Use scatter/add over an `ell_index` tensor.

This should be auxiliary only, e.g.

```text
L = L_coeff + lambda_cl * L_cl
lambda_cl in [0.01, 0.10]
```

### Auxiliary Loss B: Binned Cross-Correlation Loss

For confidence that the model learned the right realization rather than only the right spectrum:

```text
rho_band = C_cross_band / sqrt(C_pred_band * C_target_band + eps)
L_rho = mean_band alpha_band * (1 - rho_band)
```

This is closer to the current confidence language. It penalizes output that has the right amplitude but wrong phase/sign.

Use it carefully:

- Ignore ell 0-1.
- Use bins, not individual low ell.
- Clamp denominators with a meaningful floor.
- Keep weight small.

### Auxiliary Loss C: Low-Ell Curriculum

Instead of a new loss, use a schedule:

1. Warm up with ordinary coefficient MSE/L1.
2. Add low-ell band weights after validation stabilizes.
3. Add `L_cl` or `L_rho` with small lambda only after the model already reconstructs plausible maps.

This avoids early training collapse from noisy low-ell gradients.

## Recommended Next Experiments

### Experiment 1: Baseline Evaluation Harness

Build an evaluation script before changing training. It should evaluate:

- raw input pseudo-B,
- NaMaster pure-B,
- current/archived model output,
- target masked true B.

Outputs:

- per-band BB relative error,
- output-target rho,
- bootstrap confidence interval,
- shuffled-output null.

This experiment tells us whether low ell is a model failure, an estimator/metric failure, or a fundamental information-limit issue under the mask.

### Experiment 2: Current Loss Reproduction

Train or evaluate the current HRM with plain MSE/L1 and produce the same per-band metrics. The current repo has HRM code but no local `tlogs/leakless_hrm` checkpoint, so we need a fresh checkpoint before making HRM claims.

### Experiment 3: Weighted Coefficient Loss

Add only bounded low-ell coefficient weighting. This is the lowest-risk training change and preserves the behavior that worked historically.

### Experiment 4: Add Small Differentiable Spectral Auxiliary

Only after Experiment 3 has a baseline, add `lambda_cl` or `lambda_rho`.

Pass condition:

- low-ell band improves against current loss,
- high-ell confidence does not regress materially,
- output-target metrics beat shuffled/null,
- NaMaster pure-B comparison is included.

## Data Regeneration

`data_init.py` already allocates:

```text
stacks = np.empty(..., dtype=np.float32)
```

So a new run will regenerate `stacks.npy` as float32. I did not regenerate it here because the local data file is 1.4 GB and regeneration overwrites it. Given the job script requests six hours on the cluster, the safer path is to submit `job_scripts/run_data_init.sh` when ready.

## Concrete Recommendation

Do not replace the current loss with a pure bandpower objective. The strongest next step is:

1. Implement a NaMaster pure-B baseline/evaluation harness.
2. Reproduce current-loss behavior with the current HRM checkpoint path.
3. Try bounded low-ell coefficient weighting.
4. Add a small differentiable alm-bandpower or binned-rho auxiliary only if weighted coefficient loss is insufficient.

This respects the actual scientific target: cleaned map-like output, with BB bandpower recovery as a necessary but not sufficient diagnostic.
