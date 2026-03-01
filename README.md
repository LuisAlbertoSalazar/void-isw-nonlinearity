# Search for Nonlinear ISW-Void Coupling in DESI DR1

**Result: Non-detection.** A marginal nonlinear component exists but does not reach significance under calibrated statistical methods.

## What this is

A search for nonlinear scaling in the relationship between cosmic void density contrast and the integrated Sachs-Wolfe (ISW) CMB temperature signal, using 2.89 million DESI DR1 spectroscopic galaxies (z = 0.4–0.8) and the Planck PR3 SMICA CMB map.

## Key results

| Test | Result |
|------|--------|
| Binned ΔBIC (8 bins) | 6.55 — **but method has 14.5% false positive rate** |
| Unbinned ΔBIC | 0.62 — not significant |
| Unbinned F-test | p = 0.015 — nominally significant |
| Density-shuffle z-score | 3.38 (p = 0.016) — nonlinearity tied to real density ordering |
| Mock CMB calibration | No calibrated method (FPR < 5%) finds significance |

**Bottom line:** The data contain a weak nonlinear component that is genuinely tied to density ordering, but it does not survive properly calibrated statistical testing. This is a null result with an interesting methodological lesson about false-positive calibration in binned void-ISW analyses.

## Files

### Papers
- `Salazar_2026_Nonlinear_ISW_Voids_Empirical_v2.docx` — Empirical analysis (v2.0, corrected)
- `Salazar_2026_Density_Dependent_Gravitational_Framework_v2.docx` — Theoretical framework (v2.0, corrected)

### Analysis code
- `tiu_empirical_test_fullscale.py` — Main pipeline: DESI DR1 query, void finding, CMB stacking, model comparison (v3.0)
- `robustness_stress_tests.py` — Three stress tests: unbinned regression, density shuffle, mock ΛCDM CMB
- `diagnose_binned_bias.py` — Nine-method false-positive calibration comparing binned, unbinned, CV, and F-test approaches

### Results
- `stress_test_results.json` — Output from the three stress tests
- `bias_diagnostic_results.json` — Output from the nine-method calibration
- `CHANGELOG.md` — Version history

## How to reproduce

**Step 1:** Run the main pipeline (requires internet for DESI TAP queries and Planck download, ~2-6 hours):
```bash
python tiu_empirical_test_fullscale.py
```

**Step 2:** Run the stress tests (uses cached data from Step 1, ~30-60 min):
```bash
python robustness_stress_tests.py
```

**Step 3:** Run the bias diagnostic (uses cached data from Step 1, ~10-30 min):
```bash
python diagnose_binned_bias.py
```

## Data sources

- **DESI DR1**: NOIRLab Astro Data Lab TAP service, CC BY 4.0
- **Planck PR3 SMICA**: IRSA Planck Archive

## Version history

- **v2.0 (March 2026):** Added stress tests, corrected statistical claims. See erratum in empirical paper Section 5.5.
- **v1.0 (February 2026):** Initial release. Statistical claims now superseded by v2.0.

## Author

Luis Alberto Salazar — lasalazar@alum.mit.edu

## License

CC BY 4.0
