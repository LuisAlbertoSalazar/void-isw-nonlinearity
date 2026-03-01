# TIU Full-Scale Empirical Test v3.0

**Author:** Luis Alberto Salazar  
**Date:** February 2026  
**License:** CC BY 4.0  

## What Changed from v2.1

| Feature | v2.1 (Pilot) | v3.0 (Full-Scale) |
|---------|-------------|-------------------|
| Galaxies | 200,000 | Up to 18,700,000 (full DR1) |
| TAP queries | Single sync | Batched by RA slice + async fallback |
| CMB stacking | Loop-based | Vectorized (10-50x faster) |
| Robustness tests | None | Bootstrap, jackknife, null test, parameter sweep |
| Redshift analysis | Single bin | 4 sub-bins tested independently |
| Power analysis | Manual | Automated projection |
| Resume capability | None | Per-batch caching |
| Output | 1 plot + 1 txt | 2 plots + txt report + JSON |

## Quick Start

```bash
# Install dependencies (same as before)
pip install numpy matplotlib astropy scipy requests

# Quick test (200K galaxies, same as v2.1)
TIU_QUICK_MODE=1 python tiu_empirical_test_fullscale.py

# Full DR1 run (~18.7M galaxies)
python tiu_empirical_test_fullscale.py
```

## What the Robustness Suite Tests

| Test | What It Checks | Why It Matters |
|------|---------------|----------------|
| **Bootstrap** (1000×) | Confidence interval on ΔBIC | Is the result stable under resampling? |
| **Jackknife** (4 quadrants) | Spatial stability | Is one sky region driving the signal? |
| **Null test** (200×) | Random positions vs real voids | Could noise produce this ΔBIC? |
| **Void threshold sweep** | δ = −0.3 to −0.7 | Does the result depend on the void definition? |
| **NSIDE sweep** | 8, 16, 32 | Does pixel resolution change the answer? |
| **Aperture sweep** | 3°, 5°, 7° | Does the CMB averaging window matter? |
| **Redshift bins** | 4 bins from z=0.4 to z=0.8 | Does the signal evolve with redshift? |
| **Power analysis** | Void count projections | How many voids for ΔBIC > 6? |

## Runtime Estimates

| Mode | Galaxies | Est. Runtime | Disk |
|------|----------|-------------|------|
| Quick (`TIU_QUICK_MODE=1`) | 200K | 20-30 min | ~500 MB |
| Full DR1 | 18.7M | 2-6 hours | ~1 GB |

The TAP query phase dominates runtime for full DR1. Each RA slice is cached individually in `~/tiu_research/cache/`, so if a batch fails or the script is interrupted, it resumes from where it left off.

## Machine Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| RAM | 4 GB | 8 GB |
| Disk | 1 GB free | 2 GB free |
| CPU | Any modern | Multi-core helps with numpy |
| Network | Stable broadband | Stable broadband |

## Output Files

```
tiu_research/
├── cache/                              # Per-batch TAP query cache
│   ├── desi_batch_00_0_10.npz
│   ├── desi_batch_01_10_20.npz
│   └── ...
├── data/
│   ├── planck_cmb.fits                 # Cached Planck map (~400 MB)
│   └── desi_galaxies_18700000.npz      # Cached full galaxy catalog
├── plots/
│   ├── tiu_fullscale_results.png       # 9-panel diagnostic plot
│   └── tiu_nonlinearity_test.png       # 2-panel paper figure
└── results/
    ├── tiu_fullscale_results.txt       # Comprehensive text report
    ├── tiu_fullscale_results.json      # Machine-readable results
    └── tiu_test_results.txt            # Backward-compatible simple output
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TIU_QUICK_MODE` | `0` | Set to `1` for 200K pilot run |
| `TIU_MAX_ROWS` | `18700000` | Override galaxy count |
| `TIU_BATCH_SIZE` | `500000` | Rows per TAP batch |

## Interpretation Guide

| ΔBIC | Meaning |
|------|---------|
| > 6 | Strong evidence for TIU nonlinearity |
| 2 to 6 | Moderate evidence for TIU |
| −2 to 2 | Inconclusive |
| −6 to −2 | Moderate evidence for ΛCDM linearity |
| < −6 | Strong evidence for ΛCDM |

The bootstrap CI tells you how much to trust the point estimate. The null test tells you whether noise alone could produce the observed ΔBIC. The parameter sweep tells you whether the result is robust to analysis choices or an artifact of one specific configuration.

## Citation

> Salazar, L. A. (2026). Evidence for Nonlinear Density-Dependent ISW Signal in Cosmic Voids Using DESI DR1 and Planck Data. Independent Researcher.

> Salazar, L. A. (2026). A Density-Dependent Gravitational Framework: Temporal Asymmetry, Information Costs, and the Void-ISW Anomaly. Independent Researcher.
