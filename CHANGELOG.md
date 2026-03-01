# Changelog

## v2.0 — March 2026

**Major revision: corrected statistical claims after stress testing.**

- Added three robustness stress tests (unbinned regression, density-shuffle null, mock ΛCDM CMB)
- Added nine-method false-positive calibration diagnostic
- **Withdrawn:** "strong evidence for nonlinearity" claim from v1.0
- **Corrected:** Binned ΔBIC = 6.55 was computed with a method having 14.5% false positive rate
- **Finding:** No calibrated statistical method detects a significant nonlinear signal
- **Finding:** Density-shuffle test confirms a marginal nonlinear component tied to density ordering (z = 3.38, p = 0.016)
- Updated both papers to v2.0 with erratum sections
- Added robustness_stress_tests.py and diagnose_binned_bias.py scripts
- Added stress_test_results.json and bias_diagnostic_results.json

## v1.0 — February 2026

- Initial release
- Reported ΔBIC = 6.55 favoring quadratic model (now known to be inflated by binned pipeline bias)
- Null test z-score = 3.69 (tested wrong hypothesis — void positions vs random, not density ordering)
