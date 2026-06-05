# Progress report figures

Figures for `GPD/progress-report-renewal.tex` are exported from `workspace.ipynb`
(Sections 6–7: mask quicklook, first-sample validation).

Re-run the notebook in the `leakless` environment to regenerate:

```bash
jupyter nbconvert --execute workspace.ipynb --to notebook --inplace
```

Expected outputs:

| File | Notebook section |
|------|------------------|
| `mask_nside32.png` | Galactic mask at NSIDE=32 |
| `mask_apodized.png` | Apodized mask (NaMaster C2) |
| `b_map_input.png` | Partial-sky input B map |
| `b_map_target.png` | Leakage-free target B map |
| `power_spectra_first_sample.png` | E/B power spectra comparison |
