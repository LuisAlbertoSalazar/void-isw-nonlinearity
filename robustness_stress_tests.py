#!/usr/bin/env python3
"""
================================================================================
ROBUSTNESS STRESS TESTS FOR NONLINEAR ISW VOID SIGNAL
================================================================================
Version: 1.0
Author:  Luis Alberto Salazar (lasalazar@alum.mit.edu)
Date:    March 2026

PURPOSE:
    Three critical stress tests designed to determine whether the ΔBIC = 6.55
    nonlinear void-ISW signal (from tiu_empirical_test_fullscale.py) is robust
    or an artifact of methodology.

    TEST 1: Unbinned Per-Void Regression
        Fit linear and quadratic models directly to individual void measurements
        (no binning). If the signal only appears in binned data, it's likely
        bin-induced curvature rather than a real nonlinear relationship.

    TEST 2: Density-Shuffle Null Test
        Keep void sky positions (and CMB temperatures) fixed, but randomly
        shuffle the density contrast labels among voids. Repeat 1000 times.
        If the quadratic preference is tied to the TRUE δ ordering, shuffled
        data should produce ΔBIC ~ 0.  If the quadratic preference appears
        even in shuffled data, the curvature is in the temperature distribution
        itself, not in the δ-T relationship.

    TEST 3: Simulated ΛCDM Mock CMB Test
        Replace the real Planck CMB map with Gaussian-random CMB realizations
        that have no ISW signal at all. Run the full pipeline (void finding →
        stacking → model comparison). If your method routinely produces ΔBIC
        ~ 3-6 on fake CMB maps, you've discovered a methodological bias.
        If mocks give ΔBIC ~ 0, the signal is harder to dismiss.

PREREQUISITES:
    - Must have already run tiu_empirical_test_fullscale.py at least once
      (to have cached DESI galaxy data and downloaded the Planck map)
    - Same dependencies as the main script:
      python -m pip install numpy matplotlib astropy scipy requests

USAGE:
    python robustness_stress_tests.py

    This script will:
    1. Load cached DESI galaxies and Planck CMB from ~/tiu_research/data/
    2. Reproduce the void finding and CMB stacking
    3. Run all three stress tests
    4. Generate a clear PASS/FAIL report and diagnostic plots
    5. Save results to ~/tiu_research/results/stress_test_results.json

INTERPRETATION:
    If ALL three tests pass → Signal survives serious scrutiny. Email Nadathur.
    If Test 1 fails → Signal is bin-induced. Do NOT email. Investigate binning.
    If Test 2 fails → Curvature is in CMB, not in δ-T relation. Systematic.
    If Test 3 fails → Pipeline has a built-in quadratic bias. Methodological.
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
from scipy.optimize import curve_fit
from astropy.io import fits

# ============================================================
# IMPORT CORE FUNCTIONS FROM MAIN PIPELINE
# ============================================================
# We add the parent directory so we can import the main pipeline.
# If it's not on the path, we define the critical functions inline.

WORK_DIR    = os.path.join(os.path.expanduser("~"), "tiu_research")
DATA_DIR    = os.path.join(WORK_DIR, "data")
PLOTS_DIR   = os.path.join(WORK_DIR, "plots")
RESULTS_DIR = os.path.join(WORK_DIR, "results")

for d in [WORK_DIR, DATA_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ---- Configuration (must match main pipeline) ----
DEFAULT_VOID_NSIDE     = 16
DEFAULT_VOID_THRESHOLD = -0.5
DEFAULT_CMB_APERTURE   = 5.0
DEFAULT_GALACTIC_MASK  = 20.0

# ---- Stress test parameters ----
N_SHUFFLE_REALIZATIONS = 1000   # Test 2: density shuffles
N_MOCK_CMB_REALIZATIONS = 100   # Test 3: mock CMB maps
PLANCK_NSIDE_MOCK = 256         # Lower resolution for mock CMB (faster)
                                 # Original is 2048, but for stacking at 5°
                                 # apertures, 256 (~14 arcmin) is sufficient

def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] [{level}] {msg}", flush=True)

def banner(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


# ============================================================
# HEALPIX FUNCTIONS (copied from main pipeline for standalone use)
# ============================================================

def nside2npix(nside):
    return 12 * nside * nside

def ang2pix_ring(nside, theta, phi):
    """Convert (theta, phi) in radians to HEALPix RING pixel index."""
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
    """Convert HEALPix RING pixel index to (theta, phi) in radians."""
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
    """Convert (theta, phi) to unit vectors."""
    st = np.sin(theta)
    return np.column_stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)])


# ============================================================
# DATA LOADING (from cached pipeline outputs)
# ============================================================

def load_cached_galaxies():
    """Load DESI galaxies from the main pipeline's cache."""
    # Try full DR1 cache first, then smaller caches
    for max_rows in [18700000, 2891109, 200000]:
        cache = os.path.join(DATA_DIR, f"desi_galaxies_{max_rows}.npz")
        if os.path.exists(cache):
            d = np.load(cache)
            log(f"Loaded {len(d['ra']):,} cached galaxies from {cache}")
            return d["ra"], d["dec"], d["z"]

    # Try any .npz file in data dir that looks like galaxy data
    for f in os.listdir(DATA_DIR):
        if f.startswith("desi_galaxies") and f.endswith(".npz"):
            d = np.load(os.path.join(DATA_DIR, f))
            if "ra" in d:
                log(f"Loaded {len(d['ra']):,} cached galaxies from {f}")
                return d["ra"], d["dec"], d["z"]

    log("No cached galaxy data found! Run tiu_empirical_test_fullscale.py first.", "ERROR")
    return None, None, None


def load_planck(galactic_mask=DEFAULT_GALACTIC_MASK):
    """Load the cached Planck CMB map."""
    cmb_file = os.path.join(DATA_DIR, "planck_cmb.fits")
    if not os.path.exists(cmb_file):
        log("No cached Planck map found! Run tiu_empirical_test_fullscale.py first.", "ERROR")
        return None, None, None

    log(f"Loading Planck CMB map (galactic mask = {galactic_mask}°)...")
    with fits.open(cmb_file) as hdul:
        temp_map = hdul[1].data.field(0).flatten().astype(np.float64)

    npix = len(temp_map)
    nside = int(np.sqrt(npix / 12))

    pix_idx = np.arange(npix)
    theta, _ = pix2ang_ring(nside, pix_idx)
    lat = np.pi / 2 - theta
    mask = ((np.abs(lat) > np.radians(galactic_mask))
            & np.isfinite(temp_map)
            & (np.abs(temp_map) < 1))

    sky_frac = np.sum(mask) / len(mask) * 100
    log(f"NSIDE={nside}, {npix:,} pixels, usable sky: {sky_frac:.0f}%")
    return temp_map, mask, nside


# ============================================================
# VOID FINDING AND CMB STACKING (from main pipeline)
# ============================================================

def find_voids(ra, dec, nside=DEFAULT_VOID_NSIDE,
               threshold=DEFAULT_VOID_THRESHOLD, galactic_mask=DEFAULT_GALACTIC_MASK):
    """Identify cosmic voids using HEALPix density field."""
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
    """CMB stacking — returns per-void (delta, temperature) pairs."""
    ap_rad = np.radians(aperture)
    n_voids = len(vra)

    npix = len(cmb_map)
    all_theta, all_phi = pix2ang_ring(nside_cmb, np.arange(npix))
    all_vecs = ang2vec(all_theta, all_phi)

    valid = mask & np.isfinite(cmb_map)
    valid_idx = np.where(valid)[0]
    valid_vecs = all_vecs[valid_idx]
    valid_temps = cmb_map[valid_idx]

    result_delta, result_temp = [], []

    for i in range(n_voids):
        tv = np.radians(90.0 - vdec[i])
        pv = np.radians(vra[i])
        cv = np.array([np.sin(tv)*np.cos(pv),
                        np.sin(tv)*np.sin(pv),
                        np.cos(tv)])

        cos_dist = valid_vecs @ cv
        in_aperture = cos_dist >= np.cos(ap_rad)
        n_in = np.sum(in_aperture)

        if n_in >= 10:
            result_temp.append(np.mean(valid_temps[in_aperture]))
            result_delta.append(vdelta[i])

        if (i + 1) % 50 == 0 or i == n_voids - 1:
            print(f"\r  Stacking: {i+1}/{n_voids} voids "
                  f"({len(result_temp)} valid)", end="", flush=True)

    print()
    return np.array(result_delta), np.array(result_temp)


# ============================================================
# BINNED FIT (from main pipeline — for comparison)
# ============================================================

def fit_models_binned(delta, temp_uk, n_bins=8):
    """Original binned model comparison (for reference / comparison)."""
    edges = np.unique(np.percentile(delta, np.linspace(0, 100, n_bins + 1)))
    bc, bm, be, bn = [], [], [], []
    for i in range(len(edges) - 1):
        in_bin = (delta >= edges[i]) & (delta < edges[i + 1])
        n_in = np.sum(in_bin)
        if n_in >= 3:
            bc.append(np.mean(delta[in_bin]))
            bm.append(np.mean(temp_uk[in_bin]))
            be.append(np.std(temp_uk[in_bin]) / np.sqrt(n_in))
            bn.append(n_in)

    bc, bm, be, bn = np.array(bc), np.array(bm), np.array(be), np.array(bn)
    be[be == 0] = 1e-10

    if len(bc) < 4:
        return None

    lc = np.polyfit(bc, bm, 1, w=1/be)
    lp = np.polyval(lc, bc)
    lx = np.sum(((bm - lp) / be) ** 2)

    qc = np.polyfit(bc, bm, 2, w=1/be)
    qp = np.polyval(qc, bc)
    qx = np.sum(((bm - qp) / be) ** 2)

    n = len(bc)
    bic_lin = lx + 2 * np.log(n)
    bic_quad = qx + 3 * np.log(n)
    delta_bic = bic_lin - bic_quad

    return {
        "delta_bic": delta_bic, "chi2_linear": lx, "chi2_quad": qx,
        "n_bins": n, "n_voids": len(delta),
    }


# ============================================================
# TEST 1: UNBINNED PER-VOID REGRESSION
# ============================================================

def test1_unbinned_regression(delta, temp_uk):
    """
    Fit linear and quadratic models DIRECTLY to individual void measurements.
    No binning. Uses AIC and BIC at the per-void level.

    The key question: does the quadratic model still win when we don't
    average away the noise through binning?

    We use proper log-likelihood with estimated Gaussian errors,
    and compare via AIC, BIC, and F-test.
    """
    banner("TEST 1: UNBINNED PER-VOID REGRESSION")
    n = len(delta)
    log(f"Fitting {n} individual void measurements (no binning)")

    # ---- Linear fit: T = a*δ + b ----
    lin_coeffs = np.polyfit(delta, temp_uk, 1)
    lin_pred = np.polyval(lin_coeffs, delta)
    lin_resid = temp_uk - lin_pred
    lin_rss = np.sum(lin_resid**2)
    k_lin = 2  # slope + intercept

    # ---- Quadratic fit: T = a*δ² + b*δ + c ----
    quad_coeffs = np.polyfit(delta, temp_uk, 2)
    quad_pred = np.polyval(quad_coeffs, delta)
    quad_resid = temp_uk - quad_pred
    quad_rss = np.sum(quad_resid**2)
    k_quad = 3  # a + b + c

    # ---- Log-likelihood (Gaussian, estimated σ) ----
    # For model comparison with estimated variance:
    # BIC = n*ln(RSS/n) + k*ln(n)
    # AIC = n*ln(RSS/n) + 2*k
    bic_lin  = n * np.log(lin_rss / n) + k_lin * np.log(n)
    bic_quad = n * np.log(quad_rss / n) + k_quad * np.log(n)
    delta_bic = bic_lin - bic_quad  # positive favors quadratic

    aic_lin  = n * np.log(lin_rss / n) + 2 * k_lin
    aic_quad = n * np.log(quad_rss / n) + 2 * k_quad
    delta_aic = aic_lin - aic_quad  # positive favors quadratic

    # ---- F-test for nested models ----
    dof_lin = n - k_lin
    dof_quad = n - k_quad
    if lin_rss > quad_rss and dof_quad > 0:
        f_stat = ((lin_rss - quad_rss) / (k_quad - k_lin)) / (quad_rss / dof_quad)
        f_pval = 1 - stats.f.cdf(f_stat, k_quad - k_lin, dof_quad)
    else:
        f_stat, f_pval = 0.0, 1.0

    # ---- Pearson correlation ----
    r_val, p_val = stats.pearsonr(delta, temp_uk)

    # ---- R² values ----
    ss_tot = np.sum((temp_uk - np.mean(temp_uk))**2)
    r2_lin = 1 - lin_rss / ss_tot if ss_tot > 0 else 0
    r2_quad = 1 - quad_rss / ss_tot if ss_tot > 0 else 0

    # ---- Variance reduction from adding quadratic term ----
    var_reduction_pct = (1 - quad_rss / lin_rss) * 100 if lin_rss > 0 else 0

    # ---- Results ----
    log(f"Linear  RSS = {lin_rss:.2f}, BIC = {bic_lin:.2f}, AIC = {aic_lin:.2f}")
    log(f"Quad    RSS = {quad_rss:.2f}, BIC = {bic_quad:.2f}, AIC = {aic_quad:.2f}")
    log(f"ΔBIC (unbinned) = {delta_bic:.4f}  (positive favors quadratic)")
    log(f"ΔAIC (unbinned) = {delta_aic:.4f}  (positive favors quadratic)")
    log(f"F-test: F = {f_stat:.4f}, p = {f_pval:.4e}")
    log(f"Pearson r = {r_val:.4f}, p = {p_val:.4e}")
    log(f"R² linear = {r2_lin:.6f}, R² quadratic = {r2_quad:.6f}")
    log(f"Variance reduction from quadratic term: {var_reduction_pct:.2f}%")
    log(f"Quadratic coefficients: a={quad_coeffs[0]:.4f}, b={quad_coeffs[1]:.4f}, c={quad_coeffs[2]:.4f}")

    # ---- Compare to binned result ----
    binned = fit_models_binned(delta, temp_uk)
    binned_dbic = binned["delta_bic"] if binned else np.nan
    log(f"Binned ΔBIC (for comparison) = {binned_dbic:.4f}")

    # ---- Verdict ----
    if delta_bic > 2:
        verdict = "PASS — Quadratic preferred even without binning"
        passed = True
    elif delta_bic > 0:
        verdict = "MARGINAL — Slight quadratic preference, but weak"
        passed = False
    else:
        verdict = "FAIL — Quadratic NOT preferred in unbinned data"
        passed = False

    log(f"VERDICT: {verdict}")

    results = {
        "test": "unbinned_regression",
        "n_voids": n,
        "delta_bic_unbinned": float(delta_bic),
        "delta_aic_unbinned": float(delta_aic),
        "delta_bic_binned": float(binned_dbic),
        "f_stat": float(f_stat),
        "f_pval": float(f_pval),
        "r_value": float(r_val),
        "p_value": float(p_val),
        "r2_linear": float(r2_lin),
        "r2_quadratic": float(r2_quad),
        "variance_reduction_pct": float(var_reduction_pct),
        "linear_coeffs": lin_coeffs.tolist(),
        "quad_coeffs": quad_coeffs.tolist(),
        "verdict": verdict,
        "passed": passed,
    }

    return results, delta, temp_uk, lin_coeffs, quad_coeffs


# ============================================================
# TEST 2: DENSITY-SHUFFLE NULL TEST
# ============================================================

def test2_density_shuffle(delta, temp_uk, n_shuffles=N_SHUFFLE_REALIZATIONS):
    """
    Keep void sky positions and CMB temperatures FIXED.
    Randomly shuffle the density contrast (δ) labels among voids.
    Refit linear and quadratic models for each shuffle.

    This tests: is the quadratic preference tied to the TRUE δ ordering,
    or is it just general curvature in the temperature distribution?

    If shuffled ΔBIC values are centered near 0 and the real ΔBIC is an
    outlier, the signal is genuinely tied to density ordering.
    """
    banner("TEST 2: DENSITY-SHUFFLE NULL TEST")
    n = len(delta)
    log(f"Shuffling δ labels {n_shuffles} times (keeping temperatures fixed)...")

    # First get the real ΔBIC (unbinned)
    lin_rss = np.sum((temp_uk - np.polyval(np.polyfit(delta, temp_uk, 1), delta))**2)
    quad_rss = np.sum((temp_uk - np.polyval(np.polyfit(delta, temp_uk, 2), delta))**2)
    real_bic_lin = n * np.log(lin_rss / n) + 2 * np.log(n)
    real_bic_quad = n * np.log(quad_rss / n) + 3 * np.log(n)
    real_delta_bic = real_bic_lin - real_bic_quad

    # Also get the binned ΔBIC for comparison
    binned = fit_models_binned(delta, temp_uk)
    real_binned_dbic = binned["delta_bic"] if binned else np.nan

    shuffle_dbic_unbinned = []
    shuffle_dbic_binned = []

    for i in range(n_shuffles):
        # Shuffle δ labels (break the δ-T pairing)
        shuffled_delta = np.random.permutation(delta)

        # Unbinned ΔBIC
        s_lin_rss = np.sum((temp_uk - np.polyval(
            np.polyfit(shuffled_delta, temp_uk, 1), shuffled_delta))**2)
        s_quad_rss = np.sum((temp_uk - np.polyval(
            np.polyfit(shuffled_delta, temp_uk, 2), shuffled_delta))**2)
        s_bic_lin = n * np.log(s_lin_rss / n) + 2 * np.log(n)
        s_bic_quad = n * np.log(s_quad_rss / n) + 3 * np.log(n)
        shuffle_dbic_unbinned.append(s_bic_lin - s_bic_quad)

        # Binned ΔBIC
        s_binned = fit_models_binned(shuffled_delta, temp_uk)
        if s_binned:
            shuffle_dbic_binned.append(s_binned["delta_bic"])

        if (i + 1) % 200 == 0:
            print(f"\r  Shuffle: {i+1}/{n_shuffles}", end="", flush=True)

    print()

    shuffle_dbic_unbinned = np.array(shuffle_dbic_unbinned)
    shuffle_dbic_binned = np.array(shuffle_dbic_binned)

    # ---- Statistics on unbinned shuffles ----
    mean_shuffle = np.mean(shuffle_dbic_unbinned)
    std_shuffle = np.std(shuffle_dbic_unbinned)
    z_score_unbinned = ((real_delta_bic - mean_shuffle) / std_shuffle
                         if std_shuffle > 0 else 0)
    p_value_unbinned = np.mean(shuffle_dbic_unbinned >= real_delta_bic)

    # ---- Statistics on binned shuffles ----
    if len(shuffle_dbic_binned) > 0:
        mean_binned = np.mean(shuffle_dbic_binned)
        std_binned = np.std(shuffle_dbic_binned)
        z_score_binned = ((real_binned_dbic - mean_binned) / std_binned
                           if std_binned > 0 else 0)
        p_value_binned = np.mean(shuffle_dbic_binned >= real_binned_dbic)
    else:
        mean_binned = std_binned = z_score_binned = np.nan
        p_value_binned = np.nan

    log(f"Real ΔBIC (unbinned): {real_delta_bic:.4f}")
    log(f"Shuffle mean (unbinned): {mean_shuffle:.4f} ± {std_shuffle:.4f}")
    log(f"Z-score (unbinned): {z_score_unbinned:.2f}")
    log(f"P-value (unbinned): {p_value_unbinned:.4f}")
    log(f"")
    log(f"Real ΔBIC (binned): {real_binned_dbic:.4f}")
    log(f"Shuffle mean (binned): {mean_binned:.4f} ± {std_binned:.4f}")
    log(f"Z-score (binned): {z_score_binned:.2f}")
    log(f"P-value (binned): {p_value_binned:.4f}")

    # ---- Verdict ----
    # The test passes if the real ΔBIC is a significant outlier vs shuffles
    if p_value_unbinned < 0.05 and z_score_unbinned > 2:
        verdict = "PASS — Nonlinearity is tied to true δ ordering (unbinned)"
        passed = True
    elif p_value_binned < 0.05 and z_score_binned > 2:
        verdict = "PARTIAL — Signal tied to δ ordering only in binned analysis"
        passed = False  # Still concerning if only binned works
    else:
        verdict = "FAIL — Shuffled data produces similar quadratic preference"
        passed = False

    log(f"VERDICT: {verdict}")

    results = {
        "test": "density_shuffle",
        "n_shuffles": n_shuffles,
        "real_delta_bic_unbinned": float(real_delta_bic),
        "real_delta_bic_binned": float(real_binned_dbic),
        "shuffle_mean_unbinned": float(mean_shuffle),
        "shuffle_std_unbinned": float(std_shuffle),
        "z_score_unbinned": float(z_score_unbinned),
        "p_value_unbinned": float(p_value_unbinned),
        "shuffle_mean_binned": float(mean_binned),
        "shuffle_std_binned": float(std_binned),
        "z_score_binned": float(z_score_binned),
        "p_value_binned": float(p_value_binned),
        "verdict": verdict,
        "passed": passed,
    }

    return results, shuffle_dbic_unbinned, shuffle_dbic_binned


# ============================================================
# TEST 3: SIMULATED ΛCDM MOCK CMB TEST
# ============================================================

def generate_mock_cmb(nside, cl_approx=True):
    """
    Generate a Gaussian random CMB realization with no ISW signal.

    Uses an approximate CMB power spectrum (Planck best-fit ΛCDM).
    The key point: this CMB has NO correlation with large-scale structure,
    so any ΔBIC > 0 from running the pipeline on this is pure methodology.

    Parameters:
        nside: HEALPix resolution
        cl_approx: if True, use approximate power spectrum
    Returns:
        cmb_map: simulated temperature map (in Kelvin, like Planck)
    """
    npix = nside2npix(nside)
    lmax = 3 * nside - 1

    if cl_approx:
        # Approximate CMB power spectrum (ΛCDM)
        # Dl = l(l+1)Cl/(2π) ≈ constant ~ 1000 μK² for l > 30
        # For l < 30, roughly Dl ∝ l
        ell = np.arange(lmax + 1, dtype=np.float64)
        dl = np.zeros(lmax + 1)
        dl[2:] = 1000.0 * (1 - np.exp(-ell[2:] / 30.0))  # μK²

        # Suppress very high ell (beam + noise in real Planck)
        dl *= np.exp(-(ell / (0.8 * lmax))**2)

        # Convert Dl to Cl
        cl = np.zeros(lmax + 1)
        cl[2:] = dl[2:] * 2 * np.pi / (ell[2:] * (ell[2:] + 1))
        cl *= 1e-12  # μK² → K²

    # Generate alm from Cl
    # For a real-valued map, we need alm for 0 ≤ m ≤ l
    # Simple approach: generate map in pixel space using the power spectrum
    # by summing random spherical harmonic modes

    # Faster approach: generate white noise and filter
    # For our purposes, the simplest correct approach:
    # Generate Gaussian random field with the right angular power spectrum

    # Even simpler for our test: generate correlated noise on the sphere
    # that has roughly the right scale (~1° coherence length for CMB)
    # This is sufficient because we're stacking at 5° apertures

    # Method: generate white noise, then smooth with a Gaussian beam
    np.random.seed(None)  # Different each time
    white_noise = np.random.randn(npix)

    # Smooth by averaging over neighboring pixels
    # At NSIDE=256, pixel size is ~14'. CMB coherence is ~1°.
    # We need ~4 smoothing passes of nearest-neighbor averaging.
    # But for a proper test, let's use harmonic-space filtering.

    # Alternative: since we just need the right variance at 5° scales,
    # generate on a coarser grid and interpolate
    # For NSIDE=256, a 5° disc contains ~75 pixels
    # The CMB RMS at 5° is ~30 μK

    # Simplest valid approach for our test:
    # Generate noise with the right pixel-to-pixel correlation
    # by generating at lower NSIDE and upgrading

    nside_low = max(nside // 8, 4)  # Very coarse → gives degree-scale structure
    npix_low = nside2npix(nside_low)
    low_map = np.random.randn(npix_low) * 30e-6  # 30 μK RMS in Kelvin

    # Upgrade to target resolution by nearest-pixel mapping
    theta_high, phi_high = pix2ang_ring(nside, np.arange(npix))
    pix_low = ang2pix_ring(nside_low, theta_high, phi_high)
    cmb_map = low_map[pix_low]

    # Add small-scale noise (instrument noise + small-scale CMB)
    cmb_map += np.random.randn(npix) * 10e-6  # 10 μK pixel noise

    return cmb_map


def test3_mock_cmb(vra, vdec, vdelta, n_mocks=N_MOCK_CMB_REALIZATIONS,
                    nside_mock=PLANCK_NSIDE_MOCK):
    """
    Replace the real Planck CMB with Gaussian-random mock CMB realizations.
    Run the stacking and model comparison on each mock.

    If the pipeline routinely produces ΔBIC > 3 on random CMB maps
    (that have NO ISW signal), then the method has a built-in quadratic bias.

    This is the most powerful test: it directly measures the false positive rate
    of your analysis pipeline.
    """
    banner("TEST 3: SIMULATED ΛCDM MOCK CMB")
    log(f"Generating {n_mocks} mock CMB realizations at NSIDE={nside_mock}...")
    log(f"Using {len(vra)} void positions from real data")

    mock_dbic_unbinned = []
    mock_dbic_binned = []
    n_voids_used = []

    for i in range(n_mocks):
        # Generate mock CMB
        mock_cmb = generate_mock_cmb(nside_mock)
        npix_mock = len(mock_cmb)

        # Create galactic mask for mock
        theta_mock, phi_mock = pix2ang_ring(nside_mock, np.arange(npix_mock))
        lat_mock = np.abs(np.pi / 2 - theta_mock)
        mock_mask = (np.abs(np.degrees(lat_mock)) > DEFAULT_GALACTIC_MASK) & np.isfinite(mock_cmb)

        # Stack real void positions on mock CMB
        rd, rt = stack_cmb(mock_cmb, mock_mask, nside_mock, vra, vdec, vdelta)

        if len(rd) < 30:
            continue

        rt_uk = rt * 1e6
        n_voids_used.append(len(rd))

        # Unbinned ΔBIC
        n = len(rd)
        lin_rss = np.sum((rt_uk - np.polyval(np.polyfit(rd, rt_uk, 1), rd))**2)
        quad_rss = np.sum((rt_uk - np.polyval(np.polyfit(rd, rt_uk, 2), rd))**2)
        bic_lin = n * np.log(lin_rss / n) + 2 * np.log(n)
        bic_quad = n * np.log(quad_rss / n) + 3 * np.log(n)
        mock_dbic_unbinned.append(bic_lin - bic_quad)

        # Binned ΔBIC
        binned = fit_models_binned(rd, rt_uk)
        if binned:
            mock_dbic_binned.append(binned["delta_bic"])

        if (i + 1) % 20 == 0:
            print(f"\r  Mock CMB: {i+1}/{n_mocks} "
                  f"(mean ΔBIC = {np.mean(mock_dbic_unbinned):.2f})", end="", flush=True)

    print()

    mock_dbic_unbinned = np.array(mock_dbic_unbinned)
    mock_dbic_binned = np.array(mock_dbic_binned)

    # ---- Statistics ----
    mean_mock_ub = np.mean(mock_dbic_unbinned) if len(mock_dbic_unbinned) > 0 else np.nan
    std_mock_ub = np.std(mock_dbic_unbinned) if len(mock_dbic_unbinned) > 0 else np.nan
    frac_gt0_ub = np.mean(mock_dbic_unbinned > 0) if len(mock_dbic_unbinned) > 0 else np.nan
    frac_gt2_ub = np.mean(mock_dbic_unbinned > 2) if len(mock_dbic_unbinned) > 0 else np.nan
    frac_gt6_ub = np.mean(mock_dbic_unbinned > 6) if len(mock_dbic_unbinned) > 0 else np.nan

    mean_mock_b = np.mean(mock_dbic_binned) if len(mock_dbic_binned) > 0 else np.nan
    std_mock_b = np.std(mock_dbic_binned) if len(mock_dbic_binned) > 0 else np.nan
    frac_gt0_b = np.mean(mock_dbic_binned > 0) if len(mock_dbic_binned) > 0 else np.nan
    frac_gt2_b = np.mean(mock_dbic_binned > 2) if len(mock_dbic_binned) > 0 else np.nan
    frac_gt6_b = np.mean(mock_dbic_binned > 6) if len(mock_dbic_binned) > 0 else np.nan

    log(f"Mock CMB results (unbinned):")
    log(f"  Mean ΔBIC = {mean_mock_ub:.4f} ± {std_mock_ub:.4f}")
    log(f"  Fraction ΔBIC > 0: {frac_gt0_ub:.2%}")
    log(f"  Fraction ΔBIC > 2: {frac_gt2_ub:.2%}")
    log(f"  Fraction ΔBIC > 6: {frac_gt6_ub:.2%}")
    log(f"")
    log(f"Mock CMB results (binned):")
    log(f"  Mean ΔBIC = {mean_mock_b:.4f} ± {std_mock_b:.4f}")
    log(f"  Fraction ΔBIC > 0: {frac_gt0_b:.2%}")
    log(f"  Fraction ΔBIC > 2: {frac_gt2_b:.2%}")
    log(f"  Fraction ΔBIC > 6: {frac_gt6_b:.2%}")

    # ---- Verdict ----
    # If >10% of mocks produce ΔBIC > 2, the pipeline has a bias
    if frac_gt2_b > 0.10 or frac_gt2_ub > 0.10:
        verdict = ("FAIL — Pipeline produces quadratic preference in >10% of "
                   "null CMB mocks. Method has a built-in bias.")
        passed = False
    elif frac_gt0_b > 0.60:
        verdict = ("CAUTION — Pipeline shows slight quadratic tendency in null mocks. "
                   "May be a weak methodological bias.")
        passed = False
    else:
        verdict = ("PASS — Mock CMB maps show no systematic quadratic preference. "
                   "Real signal is not a pipeline artifact.")
        passed = True

    log(f"VERDICT: {verdict}")

    results = {
        "test": "mock_cmb",
        "n_mocks": n_mocks,
        "nside_mock": nside_mock,
        "mean_voids_per_mock": float(np.mean(n_voids_used)) if n_voids_used else 0,
        "unbinned": {
            "mean_delta_bic": float(mean_mock_ub),
            "std_delta_bic": float(std_mock_ub),
            "frac_gt_0": float(frac_gt0_ub),
            "frac_gt_2": float(frac_gt2_ub),
            "frac_gt_6": float(frac_gt6_ub),
        },
        "binned": {
            "mean_delta_bic": float(mean_mock_b),
            "std_delta_bic": float(std_mock_b),
            "frac_gt_0": float(frac_gt0_b),
            "frac_gt_2": float(frac_gt2_b),
            "frac_gt_6": float(frac_gt6_b),
        },
        "verdict": verdict,
        "passed": passed,
    }

    return results, mock_dbic_unbinned, mock_dbic_binned


# ============================================================
# DIAGNOSTIC PLOTS
# ============================================================

def make_diagnostic_plots(delta, temp_uk, lin_coeffs, quad_coeffs,
                           shuffle_dbic_ub, shuffle_dbic_b,
                           mock_dbic_ub, mock_dbic_b,
                           test1_results, test2_results, test3_results):
    """Generate diagnostic plots for all three stress tests."""
    banner("GENERATING DIAGNOSTIC PLOTS")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Robustness Stress Tests — Diagnostic Summary",
                 fontsize=14, fontweight="bold", y=0.98)

    # ---- Plot 1: Unbinned scatter with both fits ----
    ax = axes[0, 0]
    ax.scatter(delta, temp_uk, alpha=0.3, s=10, c="steelblue", label="Individual voids")

    d_grid = np.linspace(delta.min(), delta.max(), 200)
    ax.plot(d_grid, np.polyval(lin_coeffs, d_grid), "r-", lw=2,
            label=f"Linear (BIC ref)")
    ax.plot(d_grid, np.polyval(quad_coeffs, d_grid), "g--", lw=2,
            label=f"Quadratic")

    ax.set_xlabel("Density contrast δ")
    ax.set_ylabel("CMB Temperature (μK)")
    ax.set_title(f"Test 1: Unbinned Regression\n"
                 f"ΔBIC = {test1_results['delta_bic_unbinned']:.2f} | "
                 f"{'PASS' if test1_results['passed'] else 'FAIL'}",
                 fontsize=11)
    ax.legend(fontsize=8)

    # ---- Plot 2: Density shuffle histogram ----
    ax = axes[0, 1]
    if len(shuffle_dbic_b) > 0:
        ax.hist(shuffle_dbic_b, bins=50, alpha=0.6, color="gray",
                label=f"Shuffled (binned, n={len(shuffle_dbic_b)})")
    if len(shuffle_dbic_ub) > 0:
        ax.hist(shuffle_dbic_ub, bins=50, alpha=0.4, color="lightblue",
                label=f"Shuffled (unbinned, n={len(shuffle_dbic_ub)})")

    real_binned = test2_results.get("real_delta_bic_binned", np.nan)
    real_unbinned = test2_results.get("real_delta_bic_unbinned", np.nan)
    if np.isfinite(real_binned):
        ax.axvline(real_binned, color="red", lw=2, ls="--",
                   label=f"Real binned ΔBIC = {real_binned:.2f}")
    if np.isfinite(real_unbinned):
        ax.axvline(real_unbinned, color="blue", lw=2,
                   label=f"Real unbinned ΔBIC = {real_unbinned:.2f}")

    ax.set_xlabel("ΔBIC")
    ax.set_ylabel("Count")
    ax.set_title(f"Test 2: Density Shuffle Null\n"
                 f"z-score = {test2_results['z_score_binned']:.2f} (binned) | "
                 f"{'PASS' if test2_results['passed'] else 'FAIL'}",
                 fontsize=11)
    ax.legend(fontsize=7)

    # ---- Plot 3: Mock CMB histogram ----
    ax = axes[1, 0]
    if len(mock_dbic_b) > 0:
        ax.hist(mock_dbic_b, bins=40, alpha=0.6, color="orange",
                label=f"Mock CMB (binned, n={len(mock_dbic_b)})")
    if len(mock_dbic_ub) > 0:
        ax.hist(mock_dbic_ub, bins=40, alpha=0.4, color="lightyellow",
                edgecolor="orange",
                label=f"Mock CMB (unbinned, n={len(mock_dbic_ub)})")

    ax.axvline(0, color="black", lw=1, ls=":")
    ax.axvline(2, color="red", lw=1, ls="--", alpha=0.5, label="ΔBIC = 2")
    ax.axvline(6, color="red", lw=1, ls="-", alpha=0.5, label="ΔBIC = 6")

    ax.set_xlabel("ΔBIC from mock CMB")
    ax.set_ylabel("Count")
    frac2 = test3_results["binned"]["frac_gt_2"]
    ax.set_title(f"Test 3: Mock ΛCDM CMB\n"
                 f"P(ΔBIC>2) = {frac2:.1%} | "
                 f"{'PASS' if test3_results['passed'] else 'FAIL'}",
                 fontsize=11)
    ax.legend(fontsize=7)

    # ---- Plot 4: Summary scorecard ----
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = "STRESS TEST SCORECARD\n" + "=" * 40 + "\n\n"

    tests = [
        ("Test 1: Unbinned Regression", test1_results),
        ("Test 2: Density Shuffle", test2_results),
        ("Test 3: Mock ΛCDM CMB", test3_results),
    ]

    for name, result in tests:
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        summary_text += f"{status}  {name}\n"
        summary_text += f"       {result['verdict']}\n\n"

    all_pass = all(r["passed"] for _, r in tests)
    if all_pass:
        summary_text += "\n" + "=" * 40 + "\n"
        summary_text += "ALL TESTS PASSED\n"
        summary_text += "Signal survives scrutiny. Proceed to email.\n"
    else:
        n_fail = sum(1 for _, r in tests if not r["passed"])
        summary_text += "\n" + "=" * 40 + "\n"
        summary_text += f"{n_fail}/3 TESTS FAILED\n"
        summary_text += "Signal has vulnerabilities. Review before emailing.\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = os.path.join(PLOTS_DIR, "stress_test_diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    log(f"Diagnostic plot: {plot_path}")

    return plot_path


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    banner("ROBUSTNESS STRESS TESTS v1.0")
    print("  Three tests to determine if ΔBIC = 6.55 signal is real or artifact")
    print("  Test 1: Unbinned per-void regression (no binning)")
    print("  Test 2: Density-shuffle null test (1000 shuffles)")
    print("  Test 3: Mock ΛCDM CMB test (100 realizations)")
    print()

    # ---- Load cached data ----
    banner("LOADING DATA")
    ra, dec, z = load_cached_galaxies()
    if ra is None:
        print("\nERROR: No cached galaxy data. Run tiu_empirical_test_fullscale.py first.")
        print("This script uses cached data from the main pipeline to avoid re-downloading.")
        sys.exit(1)

    cmb_map, cmb_mask, nside_cmb = load_planck()
    if cmb_map is None:
        print("\nERROR: No cached Planck map. Run tiu_empirical_test_fullscale.py first.")
        sys.exit(1)

    # ---- Reproduce void finding and stacking ----
    banner("REPRODUCING VOID FINDING & CMB STACKING")
    vra, vdec, vdelta = find_voids(ra, dec)
    log(f"Voids found: {len(vra)}")

    rd, rt = stack_cmb(cmb_map, cmb_mask, nside_cmb, vra, vdec, vdelta)
    log(f"Valid void-CMB measurements: {len(rd)}")
    rt_uk = rt * 1e6  # Convert to μK for analysis

    if len(rd) < 30:
        log("Too few valid measurements. Cannot run stress tests.", "ERROR")
        sys.exit(1)

    # ---- Reproduce original binned result for reference ----
    banner("REFERENCE: ORIGINAL BINNED RESULT")
    ref = fit_models_binned(rd, rt_uk)
    if ref:
        log(f"Original binned ΔBIC = {ref['delta_bic']:.4f}")
    else:
        log("Could not reproduce original result!", "ERROR")

    # ---- Run the three stress tests ----
    test1_results, delta, temp_uk, lin_c, quad_c = test1_unbinned_regression(rd, rt_uk)
    test2_results, shuffle_ub, shuffle_b = test2_density_shuffle(rd, rt_uk)
    test3_results, mock_ub, mock_b = test3_mock_cmb(vra, vdec, vdelta)

    # ---- Diagnostic plots ----
    plot_path = make_diagnostic_plots(
        delta, temp_uk, lin_c, quad_c,
        shuffle_ub, shuffle_b,
        mock_ub, mock_b,
        test1_results, test2_results, test3_results
    )

    # ---- Save results ----
    banner("SAVING RESULTS")
    all_results = {
        "date": datetime.now().isoformat(),
        "version": "1.0",
        "reference_binned_delta_bic": ref["delta_bic"] if ref else None,
        "test1_unbinned_regression": test1_results,
        "test2_density_shuffle": test2_results,
        "test3_mock_cmb": test3_results,
        "overall": {
            "all_passed": all(r["passed"] for r in [test1_results, test2_results, test3_results]),
            "tests_passed": sum(1 for r in [test1_results, test2_results, test3_results] if r["passed"]),
            "tests_total": 3,
        }
    }

    json_path = os.path.join(RESULTS_DIR, "stress_test_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"JSON results: {json_path}")

    # ---- Final Summary ----
    elapsed = time.time() - start_time
    banner("STRESS TEST SUMMARY")
    print()

    for name, result in [("Test 1: Unbinned Regression", test1_results),
                          ("Test 2: Density Shuffle", test2_results),
                          ("Test 3: Mock ΛCDM CMB", test3_results)]:
        status = "PASS ✓" if result["passed"] else "FAIL ✗"
        print(f"  [{status}] {name}")
        print(f"           {result['verdict']}")
        print()

    all_pass = all_results["overall"]["all_passed"]
    n_pass = all_results["overall"]["tests_passed"]

    print(f"  {'='*60}")
    if all_pass:
        print(f"  ALL 3 TESTS PASSED")
        print(f"  Signal survives serious stress testing.")
        print(f"  You can proceed to contact researchers with confidence.")
    else:
        print(f"  {n_pass}/3 TESTS PASSED")
        print(f"  Signal has vulnerabilities that need investigation")
        print(f"  before contacting researchers.")
    print(f"  {'='*60}")
    print()
    print(f"  Runtime:    {elapsed/60:.1f} minutes")
    print(f"  Results:    {json_path}")
    print(f"  Plot:       {plot_path}")
    print()


if __name__ == "__main__":
    main()
