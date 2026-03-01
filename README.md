# Nonlinear Density-Dependent ISW Signal in Cosmic Voids

**Author:** Luis Alberto Salazar, Independent Researcher
**Contact:** lasalazar@alum.mit.edu
**Date:** February 2026
**License:** CC BY 4.0

## Summary

This repository contains the complete research package for an empirical test of nonlinear density-dependent scaling in the integrated Sachs-Wolfe (ISW) signal of cosmic voids, using 2,891,109 galaxies from DESI Data Release 1 and the Planck PR3 SMICA CMB temperature map.

The results presented in this package were generated from a full-scale query of the DESI DR1 and Planck archives performed on February 28, 2026.

### Key Result

| Metric | Value |
|--------|-------|
| Galaxies used | 2,891,109 |
| Voids analyzed | 222 |
| Delta-BIC | 6.55 (strong evidence for nonlinearity) |
| F-test p-value | 0.014 |
| Null-test z-score | 3.69 |
| Eta estimate | 0.64 |
| Jackknife stability | 4.19 - 6.55 across sky quadrants |
| Bootstrap 95% CI | [-1.84, 22.96] |

## Repository Contents

### Papers

| File | Description |
|------|-------------|
| `Salazar_2026_Nonlinear_ISW_Voids_Empirical.docx` | Empirical paper presenting data analysis, results, and robustness tests |
| `Salazar_2026_Density_Dependent_Gravitational_Framework.docx` | Framework paper presenting theoretical motivation grounded in GR and the holographic principle |

### Code

| File | Description |
|------|-------------|
| `tiu_empirical_test_fullscale.py` | Complete analysis pipeline (v3.0). Fully reproducible using only public data. |
| `README_code.md` | Technical documentation for the analysis pipeline |

### Results

| File | Description |
|------|-------------|
| `tiu_fullscale_results.txt` | Comprehensive text report |
| `tiu_fullscale_results.json` | Machine-readable JSON with all diagnostics |
| `tiu_test_results.txt` | Summary results |

## Reproducibility

All results can be reproduced from scratch:

```bash
pip install numpy matplotlib astropy scipy requests
python tiu_empirical_test_fullscale.py
```

Data sources (all public, no account required):
- DESI DR1 (CC BY 4.0): https://datalab.noirlab.edu/
- Planck PR3 SMICA: https://irsa.ipac.caltech.edu/

## Citation

> Salazar, L. A. (2026). Evidence for Nonlinear Density-Dependent ISW Signal in Cosmic Voids Using DESI DR1 and Planck Data. Independent Researcher.

> Salazar, L. A. (2026). A Density-Dependent Gravitational Framework: Temporal Asymmetry, Information Costs, and the Void-ISW Anomaly. Independent Researcher.

## Acknowledgments

This research uses data from the Astro Data Lab at NSF NOIRLab and the Planck Legacy Archive. DESI data is released under CC BY 4.0.

## Contact

Luis Alberto Salazar
Independent Researcher
lasalazar@alum.mit.edu
