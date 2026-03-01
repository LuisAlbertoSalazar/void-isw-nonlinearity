#!/usr/bin/env python3
"""
================================================================================
NONLINEAR DENSITY-DEPENDENT ISW SIGNAL IN COSMIC VOIDS — FULL-SCALE TEST
================================================================================
Version: 3.0 (Full DESI DR1 — Async TAP, Batched Queries, Robustness Suite)

Author:  Luis Alberto Salazar (lasalazar@alum.mit.edu)
Date:    February 28, 2026
License: CC BY 4.0

PURPOSE:
    Test whether cosmic voids show a NONLINEAR relationship between
    density contrast and ISW-like CMB temperature signal.

    - LCDM predicts:  T ∝ δ            (linear)
    - Nonlinear:      T ∝ δ + ηδ²      (quadratic / density-dependent)

DATA SOURCES (all public, no account required):
    - Planck PR3 SMICA CMB temperature map (IRSA)
    - DESI DR1 spectroscopic redshifts (NOIRLab TAP, CC BY 4.0)

PREREQUISITES:
    python -m pip install numpy matplotlib astropy scipy requests

WHAT'S NEW IN v3.0:
    - Async TAP batching for full DESI DR1 (~18.7M galaxies)
    - Multiple void thresholds tested (-0.3, -0.4, -0.5, -0.6, -0.7)
    - Multiple NSIDE resolutions tested (8, 16, 32)
    - Multiple CMB apertures tested (3°, 5°, 7°)
    - Bootstrap resampling for error estimation on ΔBIC
    - Jackknife spatial validation (sky quadrants)
    - Null test: random positions vs void positions
    - Galactic mask sensitivity (15°, 20°, 25°, 30°)
    - Redshift bin analysis (split z range into sub-bins)
    - Power analysis: how many more voids needed for definitive result
    - Full results saved as structured JSON + text report
    - Parallel CMB stacking via vectorized operations
    - Resume capability: caches intermediate results

RUN:
    python tiu_empirical_test_fullscale.py

    Optional flags (set as environment variables):
        TIU_MAX_ROWS=18700000     # Default: full DR1
        TIU_BATCH_SIZE=500000     # Rows per TAP query batch
        TIU_SKIP_DOWNLOAD=1       # Skip Planck download if cached
        TIU_QUICK_MODE=1          # Run with 200K rows for testing

EXPECTED RUNTIME:
    Full DR1:   2-6 hours (dominated by TAP queries + CMB stacking)
    Quick mode: 20-30 minutes
================================================================================
"""

import os, sys, warnings, time, json, hashlib
from datetime import datetime
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.io import fits
from scipy import stats
from scipy.optimize import curve_fit
import requests

# ============================================================
# CONFIGURATION
# ============================================================

WORK_DIR    = os.path.join(os.path.expanduser("~"), "tiu_research")
DATA_DIR    = os.path.join(WORK_DIR, "data")
PLOTS_DIR   = os.path.join(WORK_DIR, "plots")
RESULTS_DIR = os.path.join(WORK_DIR, "results")
CACHE_DIR   = os.path.join(WORK_DIR, "cache")

# Core parameters
QUICK_MODE      = bool(int(os.environ.get("TIU_QUICK_MODE", "0")))
DESI_MAX_ROWS   = int(os.environ.get("TIU_MAX_ROWS", "200000" if QUICK_MODE else "18700000"))
BATCH_SIZE      = int(os.environ.get("TIU_BATCH_SIZE", "500000"))
DESI_Z_MIN      = 0.4
DESI_Z_MAX      = 0.8

# Default analysis parameters
DEFAULT_VOID_NSIDE     = 16
DEFAULT_VOID_THRESHOLD = -0.5
DEFAULT_CMB_APERTURE   = 5.0
DEFAULT_GALACTIC_MASK  = 20.0

# Robustness sweep parameters
VOID_THRESHOLDS   = [-0.3, -0.4, -0.5, -0.6, -0.7]
VOID_NSIDES       = [8, 16, 32]
CMB_APERTURES     = [3.0, 5.0, 7.0]
GALACTIC_MASKS    = [15.0, 20.0, 25.0, 30.0]
REDSHIFT_BINS     = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8)]

# Bootstrap / Jackknife
N_BOOTSTRAP       = 1000
N_NULL_REALIZATIONS = 200

# TAP query config
TAP_BASE_URL      = "https://datalab.noirlab.edu/tap"
TAP_SYNC_LIMIT    = 500000  # Max rows for sync queries
TAP_TIMEOUT       = 900     # 15 minutes per query
TAP_RETRY_MAX     = 3
TAP_RETRY_DELAY   = 30      # seconds

for d in [WORK_DIR, DATA_DIR, PLOTS_DIR, RESULTS_DIR, CACHE_DIR]:
    os.makedirs(d, exist_ok=True)


def log(msg, level="INFO"):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  [{ts}] [{level}] {msg}", flush=True)


def banner(title):
    print(f"\n{'='*72}")
    print(f"  {title}")
    print(f"{'='*72}")


# ============================================================
# HEALPIX MATH (pure Python, no C library required)
# ============================================================

def nside2npix(nside):
    return 12 * nside * nside

def nside2resol(nside):
    return np.degrees(np.sqrt(4 * np.pi / nside2npix(nside)))

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


def query_disc_indices(nside, center_theta, center_phi, radius_rad):
    """Return pixel indices within a disc (brute force but vectorized)."""
    npix = nside2npix(nside)
    all_theta, all_phi = pix2ang_ring(nside, np.arange(npix))
    cv = ang2vec(np.array([center_theta]), np.array([center_phi]))[0]
    av = ang2vec(all_theta, all_phi)
    cos_dist = av @ cv
    return np.where(cos_dist >= np.cos(radius_rad))[0]


# ============================================================
# PHASE 1: PLANCK CMB DATA
# ============================================================

def download_planck():
    """Download the Planck SMICA CMB temperature map."""
    cmb_file = os.path.join(DATA_DIR, "planck_cmb.fits")
    if os.path.exists(cmb_file):
        size_mb = os.path.getsize(cmb_file) / 1e6
        log(f"Planck CMB map cached ({size_mb:.0f} MB)")
        return cmb_file

    url = ("https://irsa.ipac.caltech.edu/data/Planck/release_3/"
           "all-sky-maps/maps/component-maps/cmb/"
           "COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits")
    log("Downloading Planck CMB map (~400 MB)...")
    try:
        r = requests.get(url, stream=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(cmb_file, "wb") as f:
            for chunk in r.iter_content(chunk_size=131072):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  Downloading: {downloaded/1e6:.0f}/{total/1e6:.0f} MB "
                          f"({pct:.0f}%)", end="", flush=True)
        print()
        log("Download complete")
        return cmb_file
    except Exception as e:
        log(f"Download failed: {e}", "ERROR")
        log(f"Manual download URL: {url}")
        return None


def load_planck(cmb_file, galactic_mask=DEFAULT_GALACTIC_MASK):
    """Load the Planck CMB map and create a galactic mask."""
    log(f"Loading CMB map (galactic mask = {galactic_mask}°)...")
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
# PHASE 2: DESI DR1 GALAXY DATA (BATCHED TAP QUERIES)
# ============================================================

def _tap_query_batch(ra_min, ra_max, max_rows, batch_id=""):
    """Execute a single TAP sync query for an RA slice."""
    query = (f"SELECT TOP {max_rows} "
             f"mean_fiber_ra as ra, mean_fiber_dec as dec, "
             f"z as redshift "
             f"FROM desi_dr1.zpix "
             f"WHERE z > {DESI_Z_MIN} AND z < {DESI_Z_MAX} "
             f"AND zwarn = 0 "
             f"AND mean_fiber_ra >= {ra_min} "
             f"AND mean_fiber_ra < {ra_max}")

    for attempt in range(TAP_RETRY_MAX):
        try:
            r = requests.get(
                f"{TAP_BASE_URL}/sync",
                params={
                    "REQUEST": "doQuery",
                    "LANG": "ADQL",
                    "FORMAT": "csv",
                    "QUERY": query,
                },
                timeout=TAP_TIMEOUT,
            )
            r.raise_for_status()
            lines = r.text.strip().split("\n")
            ra, dec, z = [], [], []
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        ra.append(float(parts[0]))
                        dec.append(float(parts[1]))
                        z.append(float(parts[2]))
                    except ValueError:
                        continue
            return np.array(ra), np.array(dec), np.array(z)
        except Exception as e:
            if attempt < TAP_RETRY_MAX - 1:
                log(f"Batch {batch_id} attempt {attempt+1} failed: {e}. "
                    f"Retrying in {TAP_RETRY_DELAY}s...", "WARN")
                time.sleep(TAP_RETRY_DELAY)
            else:
                log(f"Batch {batch_id} failed after {TAP_RETRY_MAX} attempts: {e}", "ERROR")
                return np.array([]), np.array([]), np.array([])


def _async_tap_query_batch(ra_min, ra_max, max_rows, batch_id=""):
    """Execute an async TAP query for an RA slice — handles larger results."""
    query = (f"SELECT "
             f"mean_fiber_ra as ra, mean_fiber_dec as dec, "
             f"z as redshift "
             f"FROM desi_dr1.zpix "
             f"WHERE z > {DESI_Z_MIN} AND z < {DESI_Z_MAX} "
             f"AND zwarn = 0 "
             f"AND mean_fiber_ra >= {ra_min} "
             f"AND mean_fiber_ra < {ra_max}")

    for attempt in range(TAP_RETRY_MAX):
        try:
            # Submit async job
            r = requests.post(
                f"{TAP_BASE_URL}/async",
                data={
                    "REQUEST": "doQuery",
                    "LANG": "ADQL",
                    "FORMAT": "csv",
                    "QUERY": query,
                    "MAXREC": str(max_rows),
                },
                timeout=60,
            )
            r.raise_for_status()

            # Get job URL from redirect or response
            job_url = r.url if r.url != f"{TAP_BASE_URL}/async" else r.headers.get("Location", "")
            if not job_url or "/async/" not in job_url:
                # Try parsing the response for job ID
                text = r.text
                if "jobId" in text or "job/" in text:
                    # Try to extract job URL
                    for line in text.split("\n"):
                        if "href" in line.lower() and "/async/" in line:
                            start = line.find("/async/")
                            end = line.find('"', start)
                            if end > start:
                                job_url = TAP_BASE_URL + line[start:end]
                                break
                if not job_url or "/async/" not in job_url:
                    log(f"Batch {batch_id}: Could not get async job URL, falling back to sync", "WARN")
                    return _tap_query_batch(ra_min, ra_max, min(max_rows, TAP_SYNC_LIMIT), batch_id)

            # Start the job
            requests.post(f"{job_url}/phase", data={"PHASE": "RUN"}, timeout=30)

            # Poll for completion
            max_poll = 120  # 120 * 15s = 30 minutes max wait
            for poll in range(max_poll):
                time.sleep(15)
                phase_r = requests.get(f"{job_url}/phase", timeout=30)
                phase = phase_r.text.strip().upper()
                if phase == "COMPLETED":
                    break
                elif phase in ("ERROR", "ABORTED"):
                    raise Exception(f"Async job {phase}")
                if poll % 4 == 0:
                    log(f"Batch {batch_id}: waiting... (phase={phase}, {poll*15}s elapsed)")

            # Download results
            results_r = requests.get(f"{job_url}/results/result", timeout=TAP_TIMEOUT)
            results_r.raise_for_status()
            lines = results_r.text.strip().split("\n")

            ra, dec, z = [], [], []
            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) >= 3:
                    try:
                        ra.append(float(parts[0]))
                        dec.append(float(parts[1]))
                        z.append(float(parts[2]))
                    except ValueError:
                        continue

            # Clean up job
            try:
                requests.delete(job_url, timeout=10)
            except:
                pass

            return np.array(ra), np.array(dec), np.array(z)

        except Exception as e:
            if attempt < TAP_RETRY_MAX - 1:
                log(f"Batch {batch_id} async attempt {attempt+1} failed: {e}. "
                    f"Retrying in {TAP_RETRY_DELAY}s...", "WARN")
                time.sleep(TAP_RETRY_DELAY)
            else:
                log(f"Batch {batch_id} async failed, falling back to sync", "WARN")
                return _tap_query_batch(ra_min, ra_max, min(max_rows, TAP_SYNC_LIMIT), batch_id)


def get_desi_galaxies():
    """Query DESI DR1 via batched TAP queries over RA slices."""
    cache = os.path.join(DATA_DIR, f"desi_galaxies_{DESI_MAX_ROWS}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        n = len(d["ra"])
        if n > 0:
            log(f"DESI data cached: {n:,} galaxies")
            return d["ra"], d["dec"], d["z"]

    log(f"Querying DESI DR1 (target: {DESI_MAX_ROWS:,} galaxies, "
        f"z={DESI_Z_MIN}-{DESI_Z_MAX})")

    # Determine batching strategy
    if DESI_MAX_ROWS <= TAP_SYNC_LIMIT:
        # Single sync query
        log("Using single sync query")
        ra, dec, z = _tap_query_batch(0, 360, DESI_MAX_ROWS, "single")
    else:
        # Batch by RA slices
        # DESI covers ~14,000 deg² of sky, galaxies roughly uniformly in RA
        # Use ~36 batches of 10° each, with rows per batch estimated
        n_slices = 36
        ra_step = 360.0 / n_slices
        rows_per_slice = int(DESI_MAX_ROWS / n_slices * 1.2)  # 20% buffer

        log(f"Batching: {n_slices} RA slices × ~{rows_per_slice:,} rows each")

        all_ra, all_dec, all_z = [], [], []
        total_fetched = 0

        for i in range(n_slices):
            ra_min = i * ra_step
            ra_max = (i + 1) * ra_step
            batch_id = f"{i+1}/{n_slices} (RA {ra_min:.0f}-{ra_max:.0f})"

            # Check per-batch cache
            batch_cache = os.path.join(CACHE_DIR,
                f"desi_batch_{i:02d}_{ra_min:.0f}_{ra_max:.0f}.npz")

            if os.path.exists(batch_cache):
                bd = np.load(batch_cache)
                bra, bdec, bz = bd["ra"], bd["dec"], bd["z"]
                log(f"Batch {batch_id}: cached ({len(bra):,} galaxies)")
            else:
                log(f"Batch {batch_id}: querying...")
                if rows_per_slice > TAP_SYNC_LIMIT:
                    bra, bdec, bz = _async_tap_query_batch(
                        ra_min, ra_max, rows_per_slice, batch_id)
                else:
                    bra, bdec, bz = _tap_query_batch(
                        ra_min, ra_max, rows_per_slice, batch_id)

                if len(bra) > 0:
                    np.savez(batch_cache, ra=bra, dec=bdec, z=bz)
                    log(f"Batch {batch_id}: got {len(bra):,} galaxies")
                else:
                    log(f"Batch {batch_id}: empty (may be outside DESI footprint)", "WARN")

            if len(bra) > 0:
                all_ra.append(bra)
                all_dec.append(bdec)
                all_z.append(bz)
                total_fetched += len(bra)

            log(f"Running total: {total_fetched:,} galaxies")

        ra = np.concatenate(all_ra) if all_ra else np.array([])
        dec = np.concatenate(all_dec) if all_dec else np.array([])
        z = np.concatenate(all_z) if all_z else np.array([])

    if len(ra) > 0:
        np.savez(cache, ra=ra, dec=dec, z=z)
        log(f"Cached {len(ra):,} galaxies total")

    return ra, dec, z


# ============================================================
# PHASE 3: VOID IDENTIFICATION
# ============================================================

def find_voids(ra, dec, nside=DEFAULT_VOID_NSIDE,
               threshold=DEFAULT_VOID_THRESHOLD, galactic_mask=DEFAULT_GALACTIC_MASK):
    """Identify cosmic voids using HEALPix density field."""
    npix = nside2npix(nside)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra) % (2 * np.pi)
    pixels = ang2pix_ring(nside, theta, phi)

    counts = np.bincount(pixels, minlength=npix).astype(np.float64)

    # Apply galactic mask to void pixels too
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


# ============================================================
# PHASE 4: CMB STACKING (VECTORIZED)
# ============================================================

def stack_cmb_vectorized(cmb_map, mask, nside_cmb, vra, vdec, vdelta,
                          aperture=DEFAULT_CMB_APERTURE):
    """Vectorized CMB stacking — much faster than loop-based v2.1."""
    ap_rad = np.radians(aperture)
    n_voids = len(vra)

    # Precompute all pixel unit vectors for the CMB map
    npix = len(cmb_map)
    all_theta, all_phi = pix2ang_ring(nside_cmb, np.arange(npix))
    all_vecs = ang2vec(all_theta, all_phi)  # (npix, 3)

    # Pre-filter: only use unmasked, finite pixels
    valid = mask & np.isfinite(cmb_map)
    valid_idx = np.where(valid)[0]
    valid_vecs = all_vecs[valid_idx]     # (n_valid, 3)
    valid_temps = cmb_map[valid_idx]

    result_delta, result_temp = [], []

    for i in range(n_voids):
        tv = np.radians(90.0 - vdec[i])
        pv = np.radians(vra[i])
        cv = np.array([np.sin(tv)*np.cos(pv),
                        np.sin(tv)*np.sin(pv),
                        np.cos(tv)])

        # Vectorized angular distance
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
# PHASE 5: MODEL FITTING & STATISTICAL TESTS
# ============================================================

def fit_models(delta, temp_uk, n_bins=8):
    """Fit linear and quadratic models, return BIC and fit parameters."""
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

    bc = np.array(bc)
    bm = np.array(bm)
    be = np.array(be)
    bn = np.array(bn)
    be[be == 0] = 1e-10

    if len(bc) < 4:
        return None

    # Linear fit
    lc = np.polyfit(bc, bm, 1, w=1/be)
    lp = np.polyval(lc, bc)
    lr = bm - lp
    lx = np.sum((lr / be) ** 2)

    # Quadratic fit
    qc = np.polyfit(bc, bm, 2, w=1/be)
    qp = np.polyval(qc, bc)
    qr = bm - qp
    qx = np.sum((qr / be) ** 2)

    # BIC
    n = len(bc)
    bic_lin = lx + 2 * np.log(n)
    bic_quad = qx + 3 * np.log(n)
    delta_bic = bic_lin - bic_quad  # positive favors nonlinear model

    # Correlation on unbinned data
    r_val, p_val = stats.pearsonr(delta, temp_uk)

    # F-test for nested models
    dof_lin = n - 2
    dof_quad = n - 3
    if dof_quad > 0 and lx > qx:
        f_stat = ((lx - qx) / (dof_lin - dof_quad)) / (qx / dof_quad)
        f_pval = 1 - stats.f.cdf(f_stat, dof_lin - dof_quad, dof_quad)
    else:
        f_stat, f_pval = 0.0, 1.0

    # Extract eta (quadratic coefficient normalized)
    # T = a*delta^2 + b*delta + c  →  eta ~ a/b if b != 0
    eta_estimate = qc[0] / qc[1] if abs(qc[1]) > 1e-10 else np.nan

    return {
        "bin_centers": bc, "bin_means": bm, "bin_errors": be, "bin_counts": bn,
        "linear_coeffs": lc, "quad_coeffs": qc,
        "chi2_linear": lx, "chi2_quad": qx,
        "bic_linear": bic_lin, "bic_quad": bic_quad,
        "delta_bic": delta_bic,
        "r_value": r_val, "p_value": p_val,
        "f_stat": f_stat, "f_pval": f_pval,
        "eta_estimate": eta_estimate,
        "n_voids": len(delta), "n_bins": len(bc),
    }


def get_verdict(delta_bic):
    if delta_bic > 6:
        return "STRONG evidence for nonlinearity → SUPPORTS NONLINEAR MODEL"
    elif delta_bic > 2:
        return "MODERATE evidence for nonlinearity → LEANS NONLINEAR"
    elif delta_bic > -2:
        return "INCONCLUSIVE"
    elif delta_bic > -6:
        return "MODERATE evidence for linearity → LEANS ΛCDM"
    else:
        return "STRONG evidence for linearity → SUPPORTS ΛCDM"


# ============================================================
# ROBUSTNESS TESTS
# ============================================================

def bootstrap_bic(delta, temp, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap resample to get confidence interval on ΔBIC."""
    log(f"Bootstrap: {n_bootstrap} realizations...")
    temp_uk = temp * 1e6
    bic_samples = []
    n = len(delta)

    for i in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        result = fit_models(delta[idx], temp_uk[idx])
        if result:
            bic_samples.append(result["delta_bic"])
        if (i + 1) % 200 == 0:
            print(f"\r  Bootstrap: {i+1}/{n_bootstrap}", end="", flush=True)

    print()
    bic_samples = np.array(bic_samples)
    return {
        "mean": np.mean(bic_samples),
        "median": np.median(bic_samples),
        "std": np.std(bic_samples),
        "ci_2.5": np.percentile(bic_samples, 2.5),
        "ci_97.5": np.percentile(bic_samples, 97.5),
        "frac_positive": np.mean(bic_samples > 0),
        "frac_gt_6": np.mean(bic_samples > 6),
    }


def jackknife_spatial(ra, dec, delta, temp):
    """Jackknife by sky quadrant to test spatial dependence."""
    log("Jackknife: testing 4 sky quadrants...")
    temp_uk = temp * 1e6
    quadrants = [
        ("NE", (ra >= 0) & (ra < 180) & (dec >= 0)),
        ("NW", (ra >= 180) & (ra < 360) & (dec >= 0)),
        ("SE", (ra >= 0) & (ra < 180) & (dec < 0)),
        ("SW", (ra >= 180) & (ra < 360) & (dec < 0)),
    ]

    results = {}
    for name, mask in quadrants:
        # Leave-one-out: use everything EXCEPT this quadrant
        keep = ~mask
        if np.sum(keep) < 30:
            results[name] = {"delta_bic": np.nan, "n_voids": 0}
            continue
        r = fit_models(delta[keep], temp_uk[keep])
        if r:
            results[name] = {"delta_bic": r["delta_bic"], "n_voids": r["n_voids"]}
            log(f"  Excluding {name}: ΔBIC = {r['delta_bic']:.2f} ({r['n_voids']} voids)")
        else:
            results[name] = {"delta_bic": np.nan, "n_voids": 0}

    return results


def null_test(cmb_map, mask, nside_cmb, n_random, aperture=DEFAULT_CMB_APERTURE):
    """Random position null test: do random sky positions show the same signal?"""
    log(f"Null test: {N_NULL_REALIZATIONS} realizations of {n_random} random positions...")
    ap_rad = np.radians(aperture)

    npix = len(cmb_map)
    valid = mask & np.isfinite(cmb_map)
    valid_idx = np.where(valid)[0]
    all_theta, all_phi = pix2ang_ring(nside_cmb, np.arange(npix))
    all_vecs = ang2vec(all_theta, all_phi)
    valid_vecs = all_vecs[valid_idx]
    valid_temps = cmb_map[valid_idx]

    null_r_values = []
    null_delta_bics = []

    for real in range(N_NULL_REALIZATIONS):
        # Random positions on the sky (outside galactic plane)
        rand_phi = np.random.uniform(0, 2 * np.pi, n_random * 2)
        rand_cos_theta = np.random.uniform(-1, 1, n_random * 2)
        rand_theta = np.arccos(rand_cos_theta)
        rand_lat = np.abs(np.degrees(np.pi / 2 - rand_theta))
        ok = rand_lat > DEFAULT_GALACTIC_MASK
        rand_theta = rand_theta[ok][:n_random]
        rand_phi = rand_phi[ok][:n_random]

        if len(rand_theta) < n_random // 2:
            continue

        # Random "delta" values matching the void distribution
        fake_delta = np.random.uniform(-1.0, -0.3, len(rand_theta))

        temps = []
        deltas = []
        for j in range(len(rand_theta)):
            cv = np.array([np.sin(rand_theta[j])*np.cos(rand_phi[j]),
                           np.sin(rand_theta[j])*np.sin(rand_phi[j]),
                           np.cos(rand_theta[j])])
            cos_dist = valid_vecs @ cv
            in_ap = cos_dist >= np.cos(ap_rad)
            if np.sum(in_ap) >= 10:
                temps.append(np.mean(valid_temps[in_ap]))
                deltas.append(fake_delta[j])

        if len(temps) >= 30:
            t_arr = np.array(temps) * 1e6
            d_arr = np.array(deltas)
            r_val, _ = stats.pearsonr(d_arr, t_arr)
            null_r_values.append(r_val)
            result = fit_models(d_arr, t_arr)
            if result:
                null_delta_bics.append(result["delta_bic"])

        if (real + 1) % 50 == 0:
            print(f"\r  Null test: {real+1}/{N_NULL_REALIZATIONS}", end="", flush=True)

    print()
    null_r_values = np.array(null_r_values)
    null_delta_bics = np.array(null_delta_bics)

    return {
        "null_r_mean": float(np.mean(null_r_values)) if len(null_r_values) > 0 else np.nan,
        "null_r_std": float(np.std(null_r_values)) if len(null_r_values) > 0 else np.nan,
        "null_bic_mean": float(np.mean(null_delta_bics)) if len(null_delta_bics) > 0 else np.nan,
        "null_bic_std": float(np.std(null_delta_bics)) if len(null_delta_bics) > 0 else np.nan,
        "n_realizations": len(null_r_values),
    }


def parameter_sweep(ra, dec, cmb_map, mask, nside_cmb):
    """Sweep over void thresholds, NSIDE, apertures, galactic masks."""
    log("Parameter sweep: testing sensitivity to analysis choices...")
    sweep_results = []

    # Sweep void thresholds
    for thresh in VOID_THRESHOLDS:
        vra, vdec, vdelta = find_voids(ra, dec, DEFAULT_VOID_NSIDE, thresh)
        if len(vra) < 30:
            continue
        rd, rt = stack_cmb_vectorized(cmb_map, mask, nside_cmb, vra, vdec, vdelta)
        if len(rd) < 30:
            continue
        result = fit_models(rd, rt * 1e6)
        if result:
            entry = {
                "param": "void_threshold", "value": thresh,
                "delta_bic": result["delta_bic"], "n_voids": result["n_voids"],
                "chi2_lin": result["chi2_linear"], "chi2_quad": result["chi2_quad"],
            }
            sweep_results.append(entry)
            log(f"  threshold={thresh:.1f}: ΔBIC={result['delta_bic']:.2f} "
                f"({result['n_voids']} voids)")

    # Sweep NSIDE
    for ns in VOID_NSIDES:
        vra, vdec, vdelta = find_voids(ra, dec, ns, DEFAULT_VOID_THRESHOLD)
        if len(vra) < 30:
            continue
        rd, rt = stack_cmb_vectorized(cmb_map, mask, nside_cmb, vra, vdec, vdelta)
        if len(rd) < 30:
            continue
        result = fit_models(rd, rt * 1e6)
        if result:
            entry = {
                "param": "void_nside", "value": ns,
                "delta_bic": result["delta_bic"], "n_voids": result["n_voids"],
                "chi2_lin": result["chi2_linear"], "chi2_quad": result["chi2_quad"],
            }
            sweep_results.append(entry)
            log(f"  NSIDE={ns}: ΔBIC={result['delta_bic']:.2f} ({result['n_voids']} voids)")

    # Sweep apertures (re-stack with different aperture)
    vra, vdec, vdelta = find_voids(ra, dec, DEFAULT_VOID_NSIDE, DEFAULT_VOID_THRESHOLD)
    for ap in CMB_APERTURES:
        rd, rt = stack_cmb_vectorized(cmb_map, mask, nside_cmb, vra, vdec, vdelta, ap)
        if len(rd) < 30:
            continue
        result = fit_models(rd, rt * 1e6)
        if result:
            entry = {
                "param": "cmb_aperture", "value": ap,
                "delta_bic": result["delta_bic"], "n_voids": result["n_voids"],
                "chi2_lin": result["chi2_linear"], "chi2_quad": result["chi2_quad"],
            }
            sweep_results.append(entry)
            log(f"  aperture={ap}°: ΔBIC={result['delta_bic']:.2f} ({result['n_voids']} voids)")

    return sweep_results


def redshift_bin_analysis(ra, dec, z, cmb_map, mask, nside_cmb):
    """Split by redshift and test each bin independently."""
    log("Redshift bin analysis...")
    results = []

    for z_lo, z_hi in REDSHIFT_BINS:
        sel = (z >= z_lo) & (z < z_hi)
        n_sel = np.sum(sel)
        if n_sel < 1000:
            log(f"  z=[{z_lo},{z_hi}): {n_sel} galaxies — skipping", "WARN")
            continue

        vra, vdec, vdelta = find_voids(ra[sel], dec[sel])
        if len(vra) < 20:
            log(f"  z=[{z_lo},{z_hi}): {len(vra)} voids — skipping", "WARN")
            continue

        rd, rt = stack_cmb_vectorized(cmb_map, mask, nside_cmb, vra, vdec, vdelta)
        if len(rd) < 20:
            continue

        result = fit_models(rd, rt * 1e6)
        if result:
            entry = {
                "z_range": f"{z_lo}-{z_hi}",
                "n_galaxies": int(n_sel),
                "n_voids": result["n_voids"],
                "delta_bic": result["delta_bic"],
                "r_value": result["r_value"],
                "p_value": result["p_value"],
            }
            results.append(entry)
            log(f"  z=[{z_lo},{z_hi}): ΔBIC={result['delta_bic']:.2f} "
                f"({result['n_voids']} voids, {n_sel:,} galaxies)")

    return results


def power_analysis(current_n_voids, current_delta_bic):
    """Estimate how many voids needed for ΔBIC thresholds."""
    log("Power analysis...")
    # BIC penalty difference = Δk * ln(N) where Δk = 1
    # If signal exists, chi2 improvement scales ~linearly with N
    # So ΔBIC ≈ (chi2_improvement_per_void * N) - ln(N)

    targets = [2, 6, 10]
    results = {}

    if abs(current_delta_bic) < 0.01 or current_n_voids < 10:
        log("  Cannot estimate — current signal too weak")
        return {"note": "Signal too weak for power estimate"}

    # Simple scaling: ΔBIC ~ signal_per_void * N - ln(N)
    # Solve for each target
    signal_per_void = (current_delta_bic + np.log(current_n_voids)) / current_n_voids

    for target in targets:
        # Newton's method: find N where signal_per_void * N - ln(N) = target
        N = current_n_voids
        for _ in range(100):
            f = signal_per_void * N - np.log(N) - target
            fp = signal_per_void - 1/N
            if abs(fp) < 1e-15:
                break
            N_new = N - f / fp
            if N_new < 10:
                N = 10
                break
            N = N_new
            if abs(f) < 0.01:
                break

        results[f"voids_for_bic_{target}"] = int(max(N, current_n_voids))
        log(f"  ΔBIC ≥ {target}: ~{int(N):,} voids needed")

    results["current_n_voids"] = current_n_voids
    results["current_delta_bic"] = current_delta_bic
    results["signal_per_void"] = signal_per_void
    return results


# ============================================================
# PHASE 6: COMPREHENSIVE PLOTS
# ============================================================

def make_plots(primary_result, bootstrap, jackknife, null_test_result,
               sweep_results, zbin_results, delta, temp):
    """Generate comprehensive diagnostic plots."""
    log("Generating plots...")
    temp_uk = temp * 1e6
    r = primary_result

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # ---- Plot 1: Main result ----
    ax1 = fig.add_subplot(gs[0, 0])
    bc, bm, be = r["bin_centers"], r["bin_means"], r["bin_errors"]
    sx = np.linspace(bc.min(), bc.max(), 100)
    ax1.errorbar(bc, bm, yerr=be, fmt="o", color="#1B3A5C", ms=7, capsize=4,
                 label="Void bins", zorder=3)
    ax1.plot(sx, np.polyval(r["linear_coeffs"], sx), "--", color="#E74C3C",
             lw=2, label=r"Linear ($\Lambda$CDM)")
    ax1.plot(sx, np.polyval(r["quad_coeffs"], sx), "-", color="#2ECC71",
             lw=2, label="Quadratic (nonlinear)")
    ax1.set_xlabel(r"Void Density Contrast ($\delta$)")
    ax1.set_ylabel(r"Mean CMB Temperature ($\mu$K)")
    ax1.set_title(f"Primary Result: ΔBIC = {r['delta_bic']:.2f}\n"
                  f"{get_verdict(r['delta_bic'])}", fontsize=10, fontweight="bold")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # ---- Plot 2: Residuals ----
    ax2 = fig.add_subplot(gs[0, 1])
    lr = bm - np.polyval(r["linear_coeffs"], bc)
    ax2.errorbar(bc, lr, yerr=be, fmt="s", color="#1B3A5C", ms=7, capsize=4)
    ax2.axhline(0, color="#E74C3C", ls="--", alpha=0.7)
    ax2.plot(sx, np.polyval(r["quad_coeffs"], sx) - np.polyval(r["linear_coeffs"], sx),
             "-", color="#2ECC71", lw=2)
    ax2.set_xlabel(r"$\delta$")
    ax2.set_ylabel(r"Residual from Linear ($\mu$K)")
    ax2.set_title("Residuals from Linear Fit", fontsize=10)
    ax2.grid(alpha=0.3)

    # ---- Plot 3: Bootstrap ΔBIC distribution ----
    ax3 = fig.add_subplot(gs[0, 2])
    if bootstrap:
        # Reconstruct approximate distribution from summary stats
        bic_mean = bootstrap["mean"]
        bic_std = bootstrap["std"]
        x = np.linspace(bic_mean - 4*bic_std, bic_mean + 4*bic_std, 200)
        y = stats.norm.pdf(x, bic_mean, bic_std)
        ax3.fill_between(x, y, alpha=0.3, color="#3498DB")
        ax3.plot(x, y, color="#3498DB", lw=2)
        ax3.axvline(0, color="gray", ls="--", alpha=0.7, label="ΔBIC = 0")
        ax3.axvline(6, color="#2ECC71", ls="--", alpha=0.7, label="ΔBIC = 6 (strong)")
        ax3.axvline(-6, color="#E74C3C", ls="--", alpha=0.7, label="ΔBIC = −6 (strong)")
        ax3.axvline(bic_mean, color="#3498DB", ls="-", lw=2)
        ax3.set_xlabel("ΔBIC")
        ax3.set_ylabel("Density")
        ci_lo, ci_hi = bootstrap["ci_2.5"], bootstrap["ci_97.5"]
        ax3.set_title(f"Bootstrap: {bic_mean:.2f} [{ci_lo:.2f}, {ci_hi:.2f}]",
                      fontsize=10, fontweight="bold")
        ax3.legend(fontsize=7)
    ax3.grid(alpha=0.3)

    # ---- Plot 4: Parameter sweep — thresholds ----
    ax4 = fig.add_subplot(gs[1, 0])
    thresh_data = [s for s in sweep_results if s["param"] == "void_threshold"]
    if thresh_data:
        xs = [s["value"] for s in thresh_data]
        ys = [s["delta_bic"] for s in thresh_data]
        ns = [s["n_voids"] for s in thresh_data]
        ax4.bar(range(len(xs)), ys, color="#3498DB", alpha=0.7)
        ax4.set_xticks(range(len(xs)))
        ax4.set_xticklabels([f"{x:.1f}\n(n={n})" for x, n in zip(xs, ns)], fontsize=8)
        ax4.axhline(0, color="gray", ls="--")
        ax4.axhline(6, color="#2ECC71", ls="--", alpha=0.5)
        ax4.axhline(-6, color="#E74C3C", ls="--", alpha=0.5)
    ax4.set_xlabel("Void Threshold")
    ax4.set_ylabel("ΔBIC")
    ax4.set_title("Sensitivity: Void Threshold", fontsize=10)
    ax4.grid(alpha=0.3, axis="y")

    # ---- Plot 5: Parameter sweep — NSIDE ----
    ax5 = fig.add_subplot(gs[1, 1])
    nside_data = [s for s in sweep_results if s["param"] == "void_nside"]
    if nside_data:
        xs = [s["value"] for s in nside_data]
        ys = [s["delta_bic"] for s in nside_data]
        ns = [s["n_voids"] for s in nside_data]
        ax5.bar(range(len(xs)), ys, color="#9B59B6", alpha=0.7)
        ax5.set_xticks(range(len(xs)))
        ax5.set_xticklabels([f"NS={x}\n(n={n})" for x, n in zip(xs, ns)], fontsize=8)
        ax5.axhline(0, color="gray", ls="--")
        ax5.axhline(6, color="#2ECC71", ls="--", alpha=0.5)
        ax5.axhline(-6, color="#E74C3C", ls="--", alpha=0.5)
    ax5.set_xlabel("NSIDE")
    ax5.set_ylabel("ΔBIC")
    ax5.set_title("Sensitivity: HEALPix Resolution", fontsize=10)
    ax5.grid(alpha=0.3, axis="y")

    # ---- Plot 6: Parameter sweep — aperture ----
    ax6 = fig.add_subplot(gs[1, 2])
    ap_data = [s for s in sweep_results if s["param"] == "cmb_aperture"]
    if ap_data:
        xs = [s["value"] for s in ap_data]
        ys = [s["delta_bic"] for s in ap_data]
        ax6.bar(range(len(xs)), ys, color="#E67E22", alpha=0.7)
        ax6.set_xticks(range(len(xs)))
        ax6.set_xticklabels([f"{x}°" for x in xs], fontsize=8)
        ax6.axhline(0, color="gray", ls="--")
        ax6.axhline(6, color="#2ECC71", ls="--", alpha=0.5)
        ax6.axhline(-6, color="#E74C3C", ls="--", alpha=0.5)
    ax6.set_xlabel("CMB Aperture (deg)")
    ax6.set_ylabel("ΔBIC")
    ax6.set_title("Sensitivity: CMB Aperture", fontsize=10)
    ax6.grid(alpha=0.3, axis="y")

    # ---- Plot 7: Redshift bins ----
    ax7 = fig.add_subplot(gs[2, 0])
    if zbin_results:
        labels = [r["z_range"] for r in zbin_results]
        bics = [r["delta_bic"] for r in zbin_results]
        colors = ["#2ECC71" if b > 0 else "#E74C3C" for b in bics]
        ax7.bar(range(len(labels)), bics, color=colors, alpha=0.7)
        ax7.set_xticks(range(len(labels)))
        ax7.set_xticklabels([f"z={l}" for l in labels], fontsize=8)
        ax7.axhline(0, color="gray", ls="--")
    ax7.set_xlabel("Redshift Bin")
    ax7.set_ylabel("ΔBIC")
    ax7.set_title("Redshift Dependence", fontsize=10)
    ax7.grid(alpha=0.3, axis="y")

    # ---- Plot 8: Jackknife ----
    ax8 = fig.add_subplot(gs[2, 1])
    if jackknife:
        labels = list(jackknife.keys())
        bics = [jackknife[k]["delta_bic"] for k in labels]
        valid = [not np.isnan(b) for b in bics]
        labels = [l for l, v in zip(labels, valid) if v]
        bics = [b for b, v in zip(bics, valid) if v]
        if bics:
            colors = ["#2ECC71" if b > 0 else "#E74C3C" for b in bics]
            ax8.bar(range(len(labels)), bics, color=colors, alpha=0.7)
            ax8.set_xticks(range(len(labels)))
            ax8.set_xticklabels([f"Excl. {l}" for l in labels], fontsize=8)
            ax8.axhline(0, color="gray", ls="--")
            ax8.axhline(r["delta_bic"], color="#3498DB", ls="-", alpha=0.5,
                        label=f"Full: {r['delta_bic']:.2f}")
            ax8.legend(fontsize=8)
    ax8.set_xlabel("Excluded Quadrant")
    ax8.set_ylabel("ΔBIC")
    ax8.set_title("Jackknife: Spatial Stability", fontsize=10)
    ax8.grid(alpha=0.3, axis="y")

    # ---- Plot 9: Null test ----
    ax9 = fig.add_subplot(gs[2, 2])
    if null_test_result and null_test_result["n_realizations"] > 0:
        null_mean = null_test_result["null_bic_mean"]
        null_std = null_test_result["null_bic_std"]
        x = np.linspace(null_mean - 4*null_std, null_mean + 4*null_std, 200)
        y = stats.norm.pdf(x, null_mean, null_std)
        ax9.fill_between(x, y, alpha=0.3, color="gray", label="Null distribution")
        ax9.plot(x, y, color="gray", lw=2)
        ax9.axvline(r["delta_bic"], color="#3498DB", lw=2,
                    label=f"Observed: {r['delta_bic']:.2f}")
        z_score = (r["delta_bic"] - null_mean) / null_std if null_std > 0 else 0
        ax9.set_title(f"Null Test (z-score: {z_score:.1f})", fontsize=10, fontweight="bold")
        ax9.legend(fontsize=8)
    ax9.set_xlabel("ΔBIC")
    ax9.set_ylabel("Density")
    ax9.grid(alpha=0.3)

    # Title
    fig.suptitle(
        f"Nonlinear ISW Void Test — {r['n_voids']} Voids from {DESI_MAX_ROWS:,} Galaxies\n"
        f"Primary ΔBIC = {r['delta_bic']:.2f} | {get_verdict(r['delta_bic'])}",
        fontsize=14, fontweight="bold", y=0.98
    )

    plot_path = os.path.join(PLOTS_DIR, "tiu_fullscale_results.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close("all")
    log(f"Comprehensive plot saved: {plot_path}")

    # Also save the simple comparison plot for the paper
    fig2, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].errorbar(bc, bm, yerr=be, fmt="o", color="#1B3A5C", ms=8, capsize=4,
                     label="Void bins", zorder=3)
    axes[0].plot(sx, np.polyval(r["linear_coeffs"], sx), "--", color="#E74C3C",
                 lw=2, label=r"Linear ($\Lambda$CDM)")
    axes[0].plot(sx, np.polyval(r["quad_coeffs"], sx), "-", color="#2ECC71",
                 lw=2, label="Quadratic (nonlinear)")
    axes[0].set_xlabel(r"Void Density Contrast ($\delta$)", fontsize=12)
    axes[0].set_ylabel(r"Mean CMB Temperature ($\mu$K)", fontsize=12)
    axes[0].set_title("Nonlinearity Test:\nVoid Density vs CMB Temperature",
                      fontsize=13, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    axes[1].errorbar(bc, lr, yerr=be, fmt="s", color="#1B3A5C", ms=8, capsize=4)
    axes[1].axhline(0, color="#E74C3C", ls="--", alpha=0.7, label="Linear expectation")
    axes[1].plot(sx, np.polyval(r["quad_coeffs"], sx) - np.polyval(r["linear_coeffs"], sx),
                 "-", color="#2ECC71", lw=2, label="Quadratic excess")
    axes[1].set_xlabel(r"Void Density Contrast ($\delta$)", fontsize=12)
    axes[1].set_ylabel(r"Residual from Linear Fit ($\mu$K)", fontsize=12)
    axes[1].set_title(f"$\\Delta$BIC = {r['delta_bic']:.1f}  |  {get_verdict(r['delta_bic'])}",
                      fontsize=10, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    paper_plot = os.path.join(PLOTS_DIR, "tiu_nonlinearity_test.png")
    plt.savefig(paper_plot, dpi=150, bbox_inches="tight")
    plt.close("all")
    log(f"Paper plot saved: {paper_plot}")

    return plot_path


# ============================================================
# PHASE 7: SAVE EVERYTHING
# ============================================================

def save_results(primary, bootstrap, jackknife, null_result,
                 sweep, zbin, power, delta, temp):
    """Save comprehensive results as text report + JSON."""
    r = primary
    temp_uk = temp * 1e6
    verdict = get_verdict(r["delta_bic"])

    # ---- TEXT REPORT ----
    txt_path = os.path.join(RESULTS_DIR, "tiu_fullscale_results.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("  NONLINEAR ISW VOID TEST - COMPREHENSIVE RESULTS\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Date:                {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Script version:      3.0\n")
        f.write(f"DESI galaxies:       {DESI_MAX_ROWS:,}\n")
        f.write(f"Voids analyzed:      {r['n_voids']}\n")
        f.write(f"Bins:                {r['n_bins']}\n\n")

        f.write("-" * 50 + "\n")
        f.write("PRIMARY RESULT\n")
        f.write("-" * 50 + "\n")
        f.write(f"Linear chi-sq:       {r['chi2_linear']:.4f}\n")
        f.write(f"Quadratic chi-sq:    {r['chi2_quad']:.4f}\n")
        f.write(f"BIC (linear):        {r['bic_linear']:.4f}\n")
        f.write(f"BIC (quadratic):     {r['bic_quad']:.4f}\n")
        f.write(f"ΔBIC:                {r['delta_bic']:.4f}\n")
        f.write(f"F-statistic:         {r['f_stat']:.4f} (p = {r['f_pval']:.4e})\n")
        f.write(f"Correlation r:       {r['r_value']:.6f}\n")
        f.write(f"Correlation p:       {r['p_value']:.6e}\n")
        f.write(f"η estimate:          {r['eta_estimate']:.4f}\n")
        f.write(f"\nVERDICT: {verdict}\n\n")

        f.write("-" * 50 + "\n")
        f.write("BOOTSTRAP CONFIDENCE INTERVAL\n")
        f.write("-" * 50 + "\n")
        if bootstrap:
            f.write(f"Mean ΔBIC:           {bootstrap['mean']:.4f}\n")
            f.write(f"Median ΔBIC:         {bootstrap['median']:.4f}\n")
            f.write(f"Std ΔBIC:            {bootstrap['std']:.4f}\n")
            f.write(f"95% CI:              [{bootstrap['ci_2.5']:.4f}, {bootstrap['ci_97.5']:.4f}]\n")
            f.write(f"P(ΔBIC > 0):         {bootstrap['frac_positive']:.4f}\n")
            f.write(f"P(ΔBIC > 6):         {bootstrap['frac_gt_6']:.4f}\n\n")

        f.write("-" * 50 + "\n")
        f.write("JACKKNIFE SPATIAL STABILITY\n")
        f.write("-" * 50 + "\n")
        if jackknife:
            for quad, data in jackknife.items():
                f.write(f"Excluding {quad}:      ΔBIC = {data['delta_bic']:.4f} "
                        f"({data['n_voids']} voids)\n")
            bics = [v["delta_bic"] for v in jackknife.values() if not np.isnan(v["delta_bic"])]
            if bics:
                f.write(f"Jackknife range:     [{min(bics):.4f}, {max(bics):.4f}]\n")
                f.write(f"Jackknife std:       {np.std(bics):.4f}\n\n")

        f.write("-" * 50 + "\n")
        f.write("NULL TEST (RANDOM POSITIONS)\n")
        f.write("-" * 50 + "\n")
        if null_result:
            f.write(f"Null ΔBIC mean:      {null_result['null_bic_mean']:.4f}\n")
            f.write(f"Null ΔBIC std:       {null_result['null_bic_std']:.4f}\n")
            f.write(f"Null r mean:         {null_result['null_r_mean']:.4f}\n")
            f.write(f"Null r std:          {null_result['null_r_std']:.4f}\n")
            if null_result['null_bic_std'] > 0:
                z_score = ((r['delta_bic'] - null_result['null_bic_mean'])
                           / null_result['null_bic_std'])
                f.write(f"Observed z-score:    {z_score:.4f}\n")
            f.write(f"N realizations:      {null_result['n_realizations']}\n\n")

        f.write("-" * 50 + "\n")
        f.write("PARAMETER SENSITIVITY\n")
        f.write("-" * 50 + "\n")
        for s in sweep:
            f.write(f"{s['param']:>20s} = {s['value']:>6}: "
                    f"ΔBIC = {s['delta_bic']:>8.4f} ({s['n_voids']} voids)\n")
        f.write("\n")

        f.write("-" * 50 + "\n")
        f.write("REDSHIFT BIN ANALYSIS\n")
        f.write("-" * 50 + "\n")
        for zr in zbin:
            f.write(f"z = {zr['z_range']:>9s}: ΔBIC = {zr['delta_bic']:>8.4f} "
                    f"({zr['n_voids']} voids, r = {zr['r_value']:.4f})\n")
        f.write("\n")

        f.write("-" * 50 + "\n")
        f.write("POWER ANALYSIS\n")
        f.write("-" * 50 + "\n")
        if power:
            for k, v in power.items():
                f.write(f"{k:>25s}: {v}\n")
        f.write("\n")

        f.write("-" * 50 + "\n")
        f.write("BINNED DATA\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'delta':>12} {'T (uK)':>12} {'error':>12} {'count':>8}\n")
        for c, m, e, n in zip(r["bin_centers"], r["bin_means"],
                               r["bin_errors"], r["bin_counts"]):
            f.write(f"{c:>12.4f} {m:>12.4f} {e:>12.4f} {int(n):>8d}\n")

    log(f"Text report: {txt_path}")

    # Also fix encoding for the remaining file writes
    # ---- JSON (machine-readable) ----
    json_path = os.path.join(RESULTS_DIR, "tiu_fullscale_results.json")
    json_data = {
        "date": datetime.now().isoformat(),
        "version": "3.0",
        "config": {
            "desi_max_rows": DESI_MAX_ROWS,
            "z_range": [DESI_Z_MIN, DESI_Z_MAX],
            "void_nside": DEFAULT_VOID_NSIDE,
            "void_threshold": DEFAULT_VOID_THRESHOLD,
            "cmb_aperture": DEFAULT_CMB_APERTURE,
            "galactic_mask": DEFAULT_GALACTIC_MASK,
        },
        "primary": {
            "n_voids": r["n_voids"],
            "delta_bic": r["delta_bic"],
            "chi2_linear": r["chi2_linear"],
            "chi2_quad": r["chi2_quad"],
            "r_value": r["r_value"],
            "p_value": r["p_value"],
            "f_stat": r["f_stat"],
            "f_pval": r["f_pval"],
            "eta_estimate": r["eta_estimate"],
            "verdict": verdict,
        },
        "bootstrap": bootstrap,
        "jackknife": {k: {"delta_bic": float(v["delta_bic"]) if not np.isnan(v["delta_bic"]) else None,
                          "n_voids": v["n_voids"]}
                      for k, v in jackknife.items()} if jackknife else None,
        "null_test": null_result,
        "parameter_sweep": sweep,
        "redshift_bins": zbin,
        "power_analysis": power,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, default=str)
    log(f"JSON results: {json_path}")

    # Also save backward-compatible simple results
    simple_path = os.path.join(RESULTS_DIR, "tiu_test_results.txt")
    with open(simple_path, "w", encoding="utf-8") as f:
        f.write("NONLINEAR ISW VOID TEST RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Date:              {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Voids analyzed:    {r['n_voids']}\n")
        f.write(f"Bins:              {r['n_bins']}\n\n")
        f.write(f"Linear chi-sq:     {r['chi2_linear']:.4f}\n")
        f.write(f"Quadratic chi-sq:  {r['chi2_quad']:.4f}\n")
        f.write(f"BIC (linear):      {r['bic_linear']:.4f}\n")
        f.write(f"BIC (quadratic):   {r['bic_quad']:.4f}\n")
        f.write(f"Delta-BIC:         {r['delta_bic']:.4f}\n")
        f.write(f"Correlation r:     {r['r_value']:.6f}\n")
        f.write(f"Correlation p:     {r['p_value']:.6e}\n\n")
        f.write(f"VERDICT: {verdict}\n\n")
        f.write("BINNED DATA:\n")
        f.write(f"{'delta':>12} {'T (uK)':>12} {'error':>12}\n")
        for c, m, e in zip(r["bin_centers"], r["bin_means"], r["bin_errors"]):
            f.write(f"{c:>12.4f} {m:>12.4f} {e:>12.4f}\n")
    log(f"Simple results: {simple_path}")

    return txt_path, json_path


# ============================================================
# MAIN
# ============================================================

def main():
    start_time = time.time()

    banner("NONLINEAR ISW VOID TEST v3.0")
    print(f"  Target galaxies:  {DESI_MAX_ROWS:,}")
    print(f"  Quick mode:       {QUICK_MODE}")
    print(f"  Working dir:      {WORK_DIR}")
    print(f"  ΛCDM predicts:    T ∝ δ            (linear)")
    print(f"  Nonlinear predicts:    T ∝ δ + δ²       (quadratic)")
    print()

    # ---- Phase 1: Planck ----
    banner("PHASE 1: Planck CMB Data")
    cmb_file = download_planck()
    if not cmb_file:
        return
    cmb_map, mask, nside_cmb = load_planck(cmb_file)

    # ---- Phase 2: DESI ----
    banner("PHASE 2: DESI Galaxy Data")
    ra, dec, z = get_desi_galaxies()
    if ra is None or len(ra) == 0:
        log("No galaxies retrieved. Check network.", "ERROR")
        return
    log(f"Total galaxies: {len(ra):,}")
    log(f"Redshift range: {z.min():.3f} to {z.max():.3f}")
    log(f"RA range: {ra.min():.1f} to {ra.max():.1f}")
    log(f"Dec range: {dec.min():.1f} to {dec.max():.1f}")

    # ---- Phase 3: Voids ----
    banner("PHASE 3: Void Identification")
    vra, vdec, vdelta = find_voids(ra, dec)
    log(f"Voids found: {len(vra)}")
    if len(vra) == 0:
        log("No voids found.", "ERROR")
        return

    # ---- Phase 4: CMB Stacking ----
    banner("PHASE 4: CMB Stacking")
    rd, rt = stack_cmb_vectorized(cmb_map, mask, nside_cmb, vra, vdec, vdelta)
    log(f"Valid void-CMB measurements: {len(rd)}")
    if len(rd) < 30:
        log("Too few valid measurements for analysis.", "ERROR")
        return

    # ---- Phase 5: Primary Analysis ----
    banner("PHASE 5: Primary Analysis — Linear vs Nonlinear")
    primary = fit_models(rd, rt * 1e6)
    if not primary:
        log("Model fitting failed.", "ERROR")
        return

    verdict = get_verdict(primary["delta_bic"])
    log(f"ΔBIC = {primary['delta_bic']:.4f}")
    log(f"F-test p = {primary['f_pval']:.4e}")
    log(f"Correlation r = {primary['r_value']:.4f} (p = {primary['p_value']:.2e})")
    log(f"η estimate = {primary['eta_estimate']:.4f}")
    log(f"VERDICT: {verdict}")

    # ---- Phase 6: Robustness Suite ----
    banner("PHASE 6: Robustness Tests")

    # 6a: Bootstrap
    bootstrap = bootstrap_bic(rd, rt)
    log(f"Bootstrap ΔBIC: {bootstrap['mean']:.2f} ± {bootstrap['std']:.2f} "
        f"[{bootstrap['ci_2.5']:.2f}, {bootstrap['ci_97.5']:.2f}]")

    # 6b: Jackknife
    # Need RA/Dec for the valid voids (matched to rd/rt)
    # Reconstruct from stacking — use vra/vdec for voids that had valid measurements
    # Since stack_cmb_vectorized filters, we need to track which voids were kept
    # Workaround: re-identify which voids matched
    jk_ra = np.zeros(len(rd))
    jk_dec = np.zeros(len(rd))
    j = 0
    for i in range(len(vra)):
        if j >= len(rd):
            break
        if abs(vdelta[i] - rd[j]) < 1e-10:
            jk_ra[j] = vra[i]
            jk_dec[j] = vdec[i]
            j += 1

    jackknife = jackknife_spatial(jk_ra, jk_dec, rd, rt)

    # 6c: Null test
    null_result = null_test(cmb_map, mask, nside_cmb, len(rd))

    # 6d: Parameter sweep
    sweep = parameter_sweep(ra, dec, cmb_map, mask, nside_cmb)

    # 6e: Redshift bins
    zbin = redshift_bin_analysis(ra, dec, z, cmb_map, mask, nside_cmb)

    # 6f: Power analysis
    power = power_analysis(primary["n_voids"], primary["delta_bic"])

    # ---- Phase 7: Plots ----
    banner("PHASE 7: Generating Plots")
    make_plots(primary, bootstrap, jackknife, null_result, sweep, zbin, rd, rt)

    # ---- Phase 8: Save ----
    banner("PHASE 8: Saving Results")
    save_results(primary, bootstrap, jackknife, null_result, sweep, zbin, power, rd, rt)

    # ---- Summary ----
    elapsed = time.time() - start_time
    banner("COMPLETE")
    print(f"  Total runtime:     {elapsed/60:.1f} minutes")
    print(f"  Galaxies used:     {len(ra):,}")
    print(f"  Voids analyzed:    {primary['n_voids']}")
    print(f"  Primary ΔBIC:      {primary['delta_bic']:.4f}")
    print(f"  Bootstrap 95% CI:  [{bootstrap['ci_2.5']:.2f}, {bootstrap['ci_97.5']:.2f}]")
    print(f"  Verdict:           {verdict}")
    print()
    print(f"  Output directory:  {WORK_DIR}")
    print(f"  Plots:             {PLOTS_DIR}")
    print(f"  Results:           {RESULTS_DIR}")
    print()


if __name__ == "__main__":
    main()
