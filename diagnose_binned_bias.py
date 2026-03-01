#!/usr/bin/env python3
"""
================================================================================
BINNED PIPELINE BIAS DIAGNOSTIC
================================================================================
Version: 1.0
Author:  Luis Alberto Salazar (lasalazar@alum.mit.edu)
Date:    March 2026

PURPOSE:
    The stress tests revealed that the binned analysis pipeline has an ~11%
    false positive rate for ΔBIC > 2 on null CMB mocks. This script:

    1. DIAGNOSES why the bias exists (bin count, fitting method, error estimation)
    2. TESTS alternative approaches that reduce the false positive rate
    3. IDENTIFIES a corrected methodology that you can defend to reviewers

    The goal: find an analysis method where the FALSE POSITIVE rate on null
    mocks is < 5%, AND then see what the REAL data says under that method.

REQUIRES:
    Cached data from tiu_empirical_test_fullscale.py (galaxies + Planck map)

RUN:
    python diagnose_binned_bias.py
================================================================================
"""

import os, sys, warnings, time, json
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from astropy.io import fits

# ============================================================
# PATHS AND CONFIG
# ============================================================

WORK_DIR    = os.path.join(os.path.expanduser("~"), "tiu_research")
DATA_DIR    = os.path.join(WORK_DIR, "data")
PLOTS_DIR   = os.path.join(WORK_DIR, "plots")
RESULTS_DIR = os.path.join(WORK_DIR, "results")

for d in [WORK_DIR, DATA_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

DEFAULT_VOID_NSIDE     = 16
DEFAULT_VOID_THRESHOLD = -0.5
DEFAULT_CMB_APERTURE   = 5.0
DEFAULT_GALACTIC_MASK  = 20.0

N_MOCK_REALIZATIONS = 200  # More mocks for better false-positive estimates

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] [{level}] {msg}", flush=True)

def banner(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


# ============================================================
# HEALPIX (minimal, from main pipeline)
# ============================================================

def nside2npix(nside):
    return 12 * nside * nside

def ang2pix_ring(nside, theta, phi):
    npix = nside2npix(nside)
    theta = np.asarray(theta, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64) % (2 * np.pi)
    z = np.cos(theta)
    za = np.abs(z)
    tt = phi / (np.pi / 2)
    pixels = np.zeros_like(theta, dtype=np.int64)
    eq = za <= 2.0 / 3.0
    if np.any(eq):
        temp = nside * (0.5 + tt[eq] / 4)
        jp = np.floor(temp - z[eq] * nside * 0.75).astype(np.int64)
        jm = np.floor(temp + z[eq] * nside * 0.75).astype(np.int64)
        ir = nside + 1 + jp - jm
        kshift = 1 - (ir % 2)
        ip = np.floor((jp + jm - nside + kshift + 1) / 2).astype(np.int64) + 1
        ip = ip - 4 * nside * np.floor(ip / (4 * nside)).astype(np.int64)
        pixels[eq] = 2 * nside * (nside - 1) + (ir - 1) * 4 * nside + ip - 1
    north = (~eq) & (z > 0)
    if np.any(north):
        tp = tt[north] % 1.0
        tmp = nside * np.sqrt(3 * (1 - za[north]))
        jp = np.floor(tp * tmp).astype(np.int64)
        jm = np.floor((1 - tp) * tmp).astype(np.int64)
        ir = jp + jm + 1
        ip = np.floor(tt[north] * ir + 1).astype(np.int64)
        ip = ip - 4 * np.floor(ip / (4 * ir)).astype(np.int64)
        pixels[north] = 2 * ir * (ir - 1) + ip - 1
    south = (~eq) & (z <= 0)
    if np.any(south):
        tp = tt[south] % 1.0
        tmp = nside * np.sqrt(3 * (1 - za[south]))
        jp = np.floor(tp * tmp).astype(np.int64)
        jm = np.floor((1 - tp) * tmp).astype(np.int64)
        ir = jp + jm + 1
        ip = np.floor(tt[south] * ir + 1).astype(np.int64)
        ip = ip - 4 * np.floor(ip / (4 * ir)).astype(np.int64)
        pixels[south] = npix - 2 * ir * (ir + 1) + ip - 1
    return np.clip(pixels, 0, npix - 1)

def pix2ang_ring(nside, pix):
    npix = nside2npix(nside)
    pix = np.asarray(pix, dtype=np.int64)
    theta = np.zeros_like(pix, dtype=np.float64)
    phi = np.zeros_like(pix, dtype=np.float64)
    ncap = 2 * nside * (nside - 1)
    n = pix < ncap
    if np.any(n):
        p = pix[n]
        ip = np.floor((np.sqrt(1 + 2 * p) + 1) / 2).astype(np.int64)
        iphi = p - 2 * ip * (ip - 1) + 1
        theta[n] = np.arccos(1 - ip * ip / (3.0 * nside * nside))
        phi[n] = (iphi - 0.5) * np.pi / (2.0 * ip)
    eq = (~n) & (pix < npix - ncap)
    if np.any(eq):
        p = pix[eq] - ncap
        ir = np.floor(p / (4 * nside)).astype(np.int64) + nside
        iphi = (p % (4 * nside)) + 1
        fodd = 0.5 * (1 + ((ir + nside) % 2))
        theta[eq] = np.arccos((2 * nside - ir) / (1.5 * nside))
        phi[eq] = (iphi - fodd) * np.pi / (2.0 * nside)
    s = pix >= npix - ncap
    if np.any(s):
        p = npix - 1 - pix[s]
        ip = np.floor((np.sqrt(1 + 2 * p) + 1) / 2).astype(np.int64)
        iphi = p - 2 * ip * (ip - 1) + 1
        theta[s] = np.pi - np.arccos(1 - ip * ip / (3.0 * nside * nside))
        phi[s] = (iphi - 0.5) * np.pi / (2.0 * ip)
    return theta, phi

def ang2vec(theta, phi):
    st = np.sin(theta)
    return np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


# ============================================================
# DATA LOADING
# ============================================================

def load_cached_galaxies():
    for max_rows in [18700000, 2891109, 200000]:
        cache = os.path.join(DATA_DIR, f"desi_galaxies_{max_rows}.npz")
        if os.path.exists(cache):
            d = np.load(cache)
            log(f"Loaded {len(d['ra']):,} cached galaxies")
            return d["ra"], d["dec"], d["z"]
    for f in os.listdir(DATA_DIR):
        if f.startswith("desi_galaxies") and f.endswith(".npz"):
            d = np.load(os.path.join(DATA_DIR, f))
            if "ra" in d:
                log(f"Loaded {len(d['ra']):,} cached galaxies from {f}")
                return d["ra"], d["dec"], d["z"]
    return None, None, None

def load_planck(galactic_mask=DEFAULT_GALACTIC_MASK):
    cmb_file = os.path.join(DATA_DIR, "planck_cmb.fits")
    if not os.path.exists(cmb_file):
        return None, None, None
    with fits.open(cmb_file) as hdul:
        temp_map = hdul[1].data.field(0).flatten().astype(np.float64)
    npix = len(temp_map)
    nside = int(np.sqrt(npix / 12))
    pix_idx = np.arange(npix)
    theta, _ = pix2ang_ring(nside, pix_idx)
    lat = np.pi / 2 - theta
    mask = ((np.abs(lat) > np.radians(galactic_mask))
            & np.isfinite(temp_map) & (np.abs(temp_map) < 1))
    return temp_map, mask, nside

def find_voids(ra, dec, nside=DEFAULT_VOID_NSIDE,
               threshold=DEFAULT_VOID_THRESHOLD, galactic_mask=DEFAULT_GALACTIC_MASK):
    npix = nside2npix(nside)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra) % (2 * np.pi)
    pixels = ang2pix_ring(nside, theta, phi)
    counts = np.bincount(pixels, minlength=npix).astype(np.float64)
    pix_theta, pix_phi = pix2ang_ring(nside, np.arange(npix))
    pix_lat = np.abs(np.degrees(np.pi / 2 - pix_theta))
    gal_ok = pix_lat > galactic_mask
    observed = (counts > 0) & gal_ok
    mean_count = np.mean(counts[observed])
    delta = np.full(npix, np.nan)
    delta[observed] = (counts[observed] - mean_count) / mean_count
    void_mask = observed & (delta < threshold)
    void_pixels = np.where(void_mask)[0]
    vt, vp = pix2ang_ring(nside, void_pixels)
    vra = np.degrees(vp)
    vdec = 90.0 - np.degrees(vt)
    vdelta = delta[void_pixels]
    idx = np.argsort(vdelta)
    return vra[idx], vdec[idx], vdelta[idx]

def stack_cmb(cmb_map, mask, nside_cmb, vra, vdec, vdelta,
              aperture=DEFAULT_CMB_APERTURE):
    ap_rad = np.radians(aperture)
    npix = len(cmb_map)
    all_theta, all_phi = pix2ang_ring(nside_cmb, np.arange(npix))
    all_vecs = ang2vec(all_theta, all_phi)
    valid = mask & np.isfinite(cmb_map)
    valid_idx = np.where(valid)[0]
    valid_vecs = all_vecs[valid_idx]
    valid_temps = cmb_map[valid_idx]
    result_delta, result_temp = [], []
    for i in range(len(vra)):
        tv = np.radians(90.0 - vdec[i])
        pv = np.radians(vra[i])
        cv = np.array([np.sin(tv)*np.cos(pv), np.sin(tv)*np.sin(pv), np.cos(tv)])
        cos_dist = valid_vecs @ cv
        in_aperture = cos_dist >= np.cos(ap_rad)
        if np.sum(in_aperture) >= 10:
            result_temp.append(np.mean(valid_temps[in_aperture]))
            result_delta.append(vdelta[i])
        if (i + 1) % 50 == 0 or i == len(vra) - 1:
            print(f"\r  Stacking: {i+1}/{len(vra)} ({len(result_temp)} valid)",
                  end="", flush=True)
    print()
    return np.array(result_delta), np.array(result_temp)


# ============================================================
# MOCK CMB GENERATOR (from stress tests)
# ============================================================

def generate_mock_cmb(nside):
    npix = nside2npix(nside)
    nside_low = max(nside // 8, 4)
    npix_low = nside2npix(nside_low)
    low_map = np.random.randn(npix_low) * 30e-6
    theta_high, phi_high = pix2ang_ring(nside, np.arange(npix))
    pix_low = ang2pix_ring(nside_low, theta_high, phi_high)
    cmb_map = low_map[pix_low]
    cmb_map += np.random.randn(npix) * 10e-6
    return cmb_map


# ============================================================
# MULTIPLE FITTING METHODS TO COMPARE
# ============================================================

def method_original_binned(delta, temp_uk, n_bins=8):
    """Original method: equal-count bins, weighted chi2 BIC."""
    edges = np.unique(np.percentile(delta, np.linspace(0, 100, n_bins + 1)))
    bc, bm, be = [], [], []
    for i in range(len(edges) - 1):
        in_bin = (delta >= edges[i]) & (delta < edges[i + 1])
        n_in = np.sum(in_bin)
        if n_in >= 3:
            bc.append(np.mean(delta[in_bin]))
            bm.append(np.mean(temp_uk[in_bin]))
            be.append(np.std(temp_uk[in_bin]) / np.sqrt(n_in))
    bc, bm, be = np.array(bc), np.array(bm), np.array(be)
    be[be == 0] = 1e-10
    if len(bc) < 4:
        return np.nan

    lc = np.polyfit(bc, bm, 1, w=1/be)
    lx = np.sum(((bm - np.polyval(lc, bc)) / be)**2)
    qc = np.polyfit(bc, bm, 2, w=1/be)
    qx = np.sum(((bm - np.polyval(qc, bc)) / be)**2)

    n = len(bc)
    bic_lin = lx + 2 * np.log(n)
    bic_quad = qx + 3 * np.log(n)
    return bic_lin - bic_quad


def method_unbinned_bic(delta, temp_uk):
    """Unbinned: fit directly to individual voids, RSS-based BIC."""
    n = len(delta)
    lin_rss = np.sum((temp_uk - np.polyval(np.polyfit(delta, temp_uk, 1), delta))**2)
    quad_rss = np.sum((temp_uk - np.polyval(np.polyfit(delta, temp_uk, 2), delta))**2)
    bic_lin = n * np.log(lin_rss / n) + 2 * np.log(n)
    bic_quad = n * np.log(quad_rss / n) + 3 * np.log(n)
    return bic_lin - bic_quad


def method_unbinned_aic(delta, temp_uk):
    """Unbinned: AIC instead of BIC (less penalty for complexity)."""
    n = len(delta)
    lin_rss = np.sum((temp_uk - np.polyval(np.polyfit(delta, temp_uk, 1), delta))**2)
    quad_rss = np.sum((temp_uk - np.polyval(np.polyfit(delta, temp_uk, 2), delta))**2)
    aic_lin = n * np.log(lin_rss / n) + 2 * 2
    aic_quad = n * np.log(quad_rss / n) + 2 * 3
    return aic_lin - aic_quad


def method_unbinned_aicc(delta, temp_uk):
    """Unbinned: corrected AIC (AICc) — better for small samples."""
    n = len(delta)
    lin_rss = np.sum((temp_uk - np.polyval(np.polyfit(delta, temp_uk, 1), delta))**2)
    quad_rss = np.sum((temp_uk - np.polyval(np.polyfit(delta, temp_uk, 2), delta))**2)

    k_lin, k_quad = 2, 3
    aic_lin = n * np.log(lin_rss / n) + 2 * k_lin
    aic_quad = n * np.log(quad_rss / n) + 2 * k_quad

    # AICc correction
    if n - k_lin - 1 > 0:
        aic_lin += 2 * k_lin * (k_lin + 1) / (n - k_lin - 1)
    if n - k_quad - 1 > 0:
        aic_quad += 2 * k_quad * (k_quad + 1) / (n - k_quad - 1)

    return aic_lin - aic_quad


def method_ftest_pvalue(delta, temp_uk):
    """Unbinned: return -log10(F-test p-value) as the test statistic.
    Higher values = stronger evidence for quadratic."""
    n = len(delta)
    lin_rss = np.sum((temp_uk - np.polyval(np.polyfit(delta, temp_uk, 1), delta))**2)
    quad_rss = np.sum((temp_uk - np.polyval(np.polyfit(delta, temp_uk, 2), delta))**2)

    k_lin, k_quad = 2, 3
    dof_quad = n - k_quad
    if dof_quad > 0 and lin_rss > quad_rss:
        f_stat = ((lin_rss - quad_rss) / (k_quad - k_lin)) / (quad_rss / dof_quad)
        f_pval = 1 - stats.f.cdf(f_stat, k_quad - k_lin, dof_quad)
        return -np.log10(max(f_pval, 1e-20))
    return 0.0


def method_loocv(delta, temp_uk):
    """Leave-one-out cross-validation: compare prediction error.
    Returns (linear_MSE - quadratic_MSE). Positive favors quadratic.
    This has NO overfitting bias because each point is predicted
    from a model fit to ALL other points."""
    n = len(delta)
    lin_errors = np.zeros(n)
    quad_errors = np.zeros(n)

    for i in range(n):
        # Leave out point i
        d_train = np.delete(delta, i)
        t_train = np.delete(temp_uk, i)

        # Fit on remaining n-1 points
        lin_c = np.polyfit(d_train, t_train, 1)
        quad_c = np.polyfit(d_train, t_train, 2)

        # Predict the held-out point
        lin_errors[i] = (temp_uk[i] - np.polyval(lin_c, delta[i]))**2
        quad_errors[i] = (temp_uk[i] - np.polyval(quad_c, delta[i]))**2

    lin_mse = np.mean(lin_errors)
    quad_mse = np.mean(quad_errors)

    # Return difference: positive means quadratic predicts better
    return lin_mse - quad_mse


def method_kfold_cv(delta, temp_uk, k=10):
    """K-fold cross-validation. More efficient than LOOCV.
    Returns (linear_MSE - quadratic_MSE). Positive favors quadratic."""
    n = len(delta)
    indices = np.random.permutation(n)
    fold_size = n // k

    lin_errors = []
    quad_errors = []

    for fold in range(k):
        test_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size],
                                     indices[(fold + 1) * fold_size:]])

        d_train, t_train = delta[train_idx], temp_uk[train_idx]
        d_test, t_test = delta[test_idx], temp_uk[test_idx]

        lin_c = np.polyfit(d_train, t_train, 1)
        quad_c = np.polyfit(d_train, t_train, 2)

        lin_errors.extend((t_test - np.polyval(lin_c, d_test))**2)
        quad_errors.extend((t_test - np.polyval(quad_c, d_test))**2)

    return np.mean(lin_errors) - np.mean(quad_errors)


def method_binned_16(delta, temp_uk):
    """Binned with 16 bins instead of 8."""
    return method_original_binned(delta, temp_uk, n_bins=16)


def method_binned_12(delta, temp_uk):
    """Binned with 12 bins instead of 8."""
    return method_original_binned(delta, temp_uk, n_bins=12)


# ============================================================
# FALSE POSITIVE RATE CALIBRATION
# ============================================================

def calibrate_false_positive_rates(vra, vdec, vdelta, nside_mock=256,
                                    n_mocks=N_MOCK_REALIZATIONS):
    """
    Run each method on null CMB mocks and measure the false positive rate.
    A method is calibrated if its FPR at some threshold is < 5%.
    """
    banner("CALIBRATING FALSE POSITIVE RATES ON NULL MOCKS")

    methods = {
        "original_binned_8":  method_original_binned,
        "binned_12":          method_binned_12,
        "binned_16":          method_binned_16,
        "unbinned_bic":       method_unbinned_bic,
        "unbinned_aic":       method_unbinned_aic,
        "unbinned_aicc":      method_unbinned_aicc,
        "ftest_neglog10p":    method_ftest_pvalue,
        "loocv_mse_diff":     method_loocv,
        "kfold_cv_mse_diff":  method_kfold_cv,
    }

    # Storage for null distributions
    null_distributions = {name: [] for name in methods}

    log(f"Running {n_mocks} mock CMB realizations across {len(methods)} methods...")

    for i in range(n_mocks):
        mock_cmb = generate_mock_cmb(nside_mock)
        npix_mock = len(mock_cmb)
        theta_mock, _ = pix2ang_ring(nside_mock, np.arange(npix_mock))
        lat_mock = np.abs(np.pi / 2 - theta_mock)
        mock_mask = (np.abs(np.degrees(lat_mock)) > DEFAULT_GALACTIC_MASK) & np.isfinite(mock_cmb)

        rd, rt = stack_cmb(mock_cmb, mock_mask, nside_mock, vra, vdec, vdelta)
        if len(rd) < 30:
            continue

        rt_uk = rt * 1e6

        for name, method in methods.items():
            try:
                val = method(rd, rt_uk)
                if np.isfinite(val):
                    null_distributions[name].append(val)
            except Exception:
                pass

        if (i + 1) % 25 == 0:
            print(f"\r  Mock {i+1}/{n_mocks}", end="", flush=True)

    print()

    # ---- Compute false positive rates ----
    results = {}
    log(f"\n{'Method':<25} {'Mean':>8} {'Std':>8} {'FPR>0':>8} {'FPR>2':>8} {'FPR>6':>8}")
    log(f"{'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for name in methods:
        vals = np.array(null_distributions[name])
        if len(vals) < 10:
            continue

        mean_v = np.mean(vals)
        std_v = np.std(vals)

        # For methods where positive = favors quadratic
        fpr_0 = np.mean(vals > 0)
        fpr_2 = np.mean(vals > 2)
        fpr_6 = np.mean(vals > 6)

        # For F-test, the threshold is different (-log10(0.05) ≈ 1.3)
        if name == "ftest_neglog10p":
            fpr_sig = np.mean(vals > 1.301)  # p < 0.05
            fpr_0 = fpr_sig  # Override: "false positive" = p < 0.05

        # For CV methods, any positive value means quadratic predicted better
        if "cv" in name or "loocv" in name:
            fpr_2 = np.mean(vals > np.percentile(vals, 95))  # top 5%
            fpr_6 = np.mean(vals > np.percentile(vals, 99))  # top 1%

        log(f"{name:<25} {mean_v:>8.3f} {std_v:>8.3f} "
            f"{fpr_0:>8.1%} {fpr_2:>8.1%} {fpr_6:>8.1%}")

        results[name] = {
            "null_mean": float(mean_v),
            "null_std": float(std_v),
            "null_values": vals.tolist(),
            "fpr_gt_0": float(fpr_0),
            "fpr_gt_2": float(fpr_2),
            "fpr_gt_6": float(fpr_6),
            "n_mocks": len(vals),
        }

    return results, null_distributions


def evaluate_real_data(delta, temp_uk, null_results):
    """Run all methods on real data and compare to null distributions."""
    banner("EVALUATING REAL DATA WITH ALL METHODS")

    methods = {
        "original_binned_8":  method_original_binned,
        "binned_12":          method_binned_12,
        "binned_16":          method_binned_16,
        "unbinned_bic":       method_unbinned_bic,
        "unbinned_aic":       method_unbinned_aic,
        "unbinned_aicc":      method_unbinned_aicc,
        "ftest_neglog10p":    method_ftest_pvalue,
        "loocv_mse_diff":     method_loocv,
        "kfold_cv_mse_diff":  method_kfold_cv,
    }

    log(f"\n{'Method':<25} {'Real val':>10} {'Null mean':>10} "
        f"{'Z-score':>10} {'P-value':>10} {'Calibrated?':>12}")
    log(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*12}")

    real_results = {}

    for name, method in methods.items():
        if name not in null_results:
            continue

        try:
            real_val = method(delta, temp_uk)
        except Exception:
            continue

        null_vals = np.array(null_results[name]["null_values"])
        null_mean = np.mean(null_vals)
        null_std = np.std(null_vals)

        z_score = (real_val - null_mean) / null_std if null_std > 0 else 0
        p_val = np.mean(null_vals >= real_val)

        # A method is "calibrated" if its FPR > 2 is < 5%
        fpr_2 = null_results[name]["fpr_gt_2"]
        calibrated = fpr_2 < 0.05

        log(f"{name:<25} {real_val:>10.4f} {null_mean:>10.4f} "
            f"{z_score:>10.2f} {p_val:>10.4f} "
            f"{'YES' if calibrated else 'NO':>12}")

        real_results[name] = {
            "real_value": float(real_val),
            "null_mean": float(null_mean),
            "null_std": float(null_std),
            "z_score": float(z_score),
            "p_value_vs_null": float(p_val),
            "calibrated": calibrated,
            "significant": p_val < 0.05,
        }

    return real_results


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def make_comparison_plots(null_distributions, real_results):
    """Plot null distributions and real values for each method."""
    methods_to_plot = [name for name in null_distributions
                       if len(null_distributions[name]) > 10]

    n_methods = len(methods_to_plot)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle("Null Distribution Calibration: Each Method vs Mock CMB",
                 fontsize=13, fontweight="bold")

    for idx, name in enumerate(methods_to_plot):
        ax = axes[idx // n_cols, idx % n_cols]
        vals = np.array(null_distributions[name])

        ax.hist(vals, bins=40, alpha=0.6, color="steelblue", edgecolor="white")

        if name in real_results:
            rv = real_results[name]["real_value"]
            ax.axvline(rv, color="red", lw=2, ls="--",
                       label=f"Real = {rv:.2f}")
            p = real_results[name]["p_value_vs_null"]
            cal = real_results[name]["calibrated"]
            sig = real_results[name]["significant"]
            status = "SIG" if sig else "n.s."
            cal_str = "calibrated" if cal else "BIASED"
            ax.set_title(f"{name}\np={p:.3f} ({status}, {cal_str})", fontsize=9)
        else:
            ax.set_title(name, fontsize=9)

        ax.axvline(0, color="gray", lw=1, ls=":")
        ax.legend(fontsize=7)
        ax.set_xlabel("Test statistic")
        ax.set_ylabel("Count")

    # Hide unused axes
    for idx in range(n_methods, n_rows * n_cols):
        axes[idx // n_cols, idx % n_cols].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plot_path = os.path.join(PLOTS_DIR, "bias_diagnostic_all_methods.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Plot saved: {plot_path}")
    return plot_path


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    banner("BINNED PIPELINE BIAS DIAGNOSTIC v1.0")
    print("  Goal: find an analysis method where false positive rate < 5%")
    print("  Then evaluate the real data under that method")
    print()

    # ---- Load data ----
    banner("LOADING DATA")
    ra, dec, z = load_cached_galaxies()
    if ra is None:
        print("ERROR: No cached galaxy data. Run main pipeline first.")
        sys.exit(1)

    cmb_map, cmb_mask, nside_cmb = load_planck()
    if cmb_map is None:
        print("ERROR: No cached Planck map. Run main pipeline first.")
        sys.exit(1)

    # ---- Reproduce voids and stacking ----
    banner("REPRODUCING VOID FINDING & STACKING")
    vra, vdec, vdelta = find_voids(ra, dec)
    log(f"Voids: {len(vra)}")
    rd, rt = stack_cmb(cmb_map, cmb_mask, nside_cmb, vra, vdec, vdelta)
    log(f"Valid measurements: {len(rd)}")
    rt_uk = rt * 1e6

    # ---- Calibrate on null mocks ----
    null_results, null_dists = calibrate_false_positive_rates(vra, vdec, vdelta)

    # ---- Evaluate real data ----
    real_results = evaluate_real_data(rd, rt_uk, null_results)

    # ---- Plots ----
    banner("GENERATING PLOTS")
    plot_path = make_comparison_plots(null_dists, real_results)

    # ---- Identify best calibrated method ----
    banner("RECOMMENDATION")

    calibrated_significant = []
    calibrated_nonsignificant = []
    uncalibrated = []

    for name, res in real_results.items():
        if res["calibrated"] and res["significant"]:
            calibrated_significant.append((name, res))
        elif res["calibrated"]:
            calibrated_nonsignificant.append((name, res))
        else:
            uncalibrated.append((name, res))

    if calibrated_significant:
        log("METHODS THAT ARE BOTH CALIBRATED AND SIGNIFICANT:")
        for name, res in calibrated_significant:
            log(f"  {name}: real={res['real_value']:.4f}, "
                f"z={res['z_score']:.2f}, p={res['p_value_vs_null']:.4f}")
        log("")
        log("These methods have a <5% false positive rate on null mocks")
        log("AND detect a significant signal in the real data.")
        log("Use these to report your result.")
    else:
        log("NO method is both calibrated (<5% FPR) AND significant (p<0.05).")
        log("This means the signal does not survive proper calibration.")

    if calibrated_nonsignificant:
        log("")
        log("CALIBRATED but NOT significant:")
        for name, res in calibrated_nonsignificant:
            log(f"  {name}: real={res['real_value']:.4f}, "
                f"z={res['z_score']:.2f}, p={res['p_value_vs_null']:.4f}")

    if uncalibrated:
        log("")
        log("UNCALIBRATED methods (>5% FPR, cannot be trusted):")
        for name, res in uncalibrated:
            log(f"  {name}: FPR>2 = {null_results[name]['fpr_gt_2']:.1%}")

    # ---- Save everything ----
    banner("SAVING RESULTS")
    output = {
        "date": datetime.now().isoformat(),
        "n_mocks": N_MOCK_REALIZATIONS,
        "n_voids": len(rd),
        "null_calibration": {k: {kk: vv for kk, vv in v.items() if kk != "null_values"}
                              for k, v in null_results.items()},
        "real_data_results": real_results,
        "calibrated_and_significant": [name for name, _ in calibrated_significant],
        "calibrated_not_significant": [name for name, _ in calibrated_nonsignificant],
        "uncalibrated": [name for name, _ in uncalibrated],
    }

    json_path = os.path.join(RESULTS_DIR, "bias_diagnostic_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    log(f"Results: {json_path}")

    elapsed = time.time() - start_time
    banner("COMPLETE")
    print(f"  Runtime: {elapsed/60:.1f} minutes")
    print(f"  Results: {json_path}")
    print(f"  Plot:    {plot_path}")
    print()


if __name__ == "__main__":
    main()
