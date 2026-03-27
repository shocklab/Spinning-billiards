#!/usr/bin/env python3
"""Unified precomputation of FTLE data for all figures.

Generates a single .npz file containing per-trajectory FTLE values
for Stadium and Sinai at 50 α values with 100,000 ICs each,
plus a geometry scan for the stadium at 5 values of the flat-section
length a (10,000 ICs each).

Usage:
    python3 precompute_data.py          # Run everything
    python3 precompute_data.py --main   # Only main geometries (Stadium + Sinai)
    python3 precompute_data.py --geo    # Only geometry scan
"""

import os
import sys
import time
import numpy as np
from spinning_billiards import (
    lyapunov_ensemble, STADIUM, SINAI, warmup
)

# ── Parameters ──
N_ALPHA = 50
ALPHA_VALUES = np.linspace(0.0, 1.0, N_ALPHA)
N_STEPS = 50_000          # collisions per trajectory
N_ICS_MAIN = 100_000      # ICs for main geometries
N_ICS_GEO = 10_000        # ICs for geometry scan extras

# Stadium: geo_p1 = a (half-length of flat section), geo_p2 = 0.0
# Sinai:   geo_p1 = L (box half-side), geo_p2 = R (obstacle radius)
MAIN_GEOS = [
    ("Stadium", STADIUM, 1.0, 0.0),
    ("Sinai",   SINAI,   2.0, 1.0),
]

A_GEO_VALUES = np.array([0.2, 0.5, 1.0, 2.0, 4.0])

OUTDIR = os.path.join(os.path.dirname(__file__), "experiment_results")
OUTFILE = os.path.join(OUTDIR, "unified_billiards_data.npz")


def _fmt_time(seconds):
    """Format seconds into h:mm:ss or m:ss."""
    if seconds >= 3600:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h}:{m:02d}:{s:02d}"
    else:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m}:{s:02d}"


def run_main_geometries():
    """Compute per-trajectory FTLE for Stadium and Sinai at 100K ICs."""
    results = {}

    for gname, geo, p1, p2 in MAIN_GEOS:
        print(f"\n{'='*60}")
        print(f"  {gname}: {N_ICS_MAIN:,} ICs × {N_ALPHA} α values × {N_STEPS:,} collisions")
        print(f"{'='*60}")

        ftle_all = np.empty((N_ALPHA, N_ICS_MAIN))
        t_geo_start = time.perf_counter()

        for j, a in enumerate(ALPHA_VALUES):
            t0 = time.perf_counter()
            lcns = lyapunov_ensemble(N_STEPS, N_ICS_MAIN, a, geo, p1, p2,
                                     perturb_mag=1e-7, u_max_frac=0.5)
            dt = time.perf_counter() - t0
            ftle_all[j, :] = lcns

            elapsed = time.perf_counter() - t_geo_start
            remaining = elapsed / (j + 1) * (N_ALPHA - j - 1)
            print(f"  [{gname}] α = {a:.2f}  [{j+1}/{N_ALPHA}]  "
                  f"{dt:.1f}s this α | "
                  f"elapsed {_fmt_time(elapsed)} | "
                  f"est. remaining {_fmt_time(remaining)}")

        total = time.perf_counter() - t_geo_start
        print(f"  {gname} complete: {_fmt_time(total)}")

        # Compute summary stats
        finite_mask = np.isfinite(ftle_all)
        means = np.array([np.mean(ftle_all[j][finite_mask[j]])
                          for j in range(N_ALPHA)])
        stds = np.array([np.std(ftle_all[j][finite_mask[j]])
                         for j in range(N_ALPHA)])
        sems = np.array([stds[j] / max(1, np.sqrt(np.sum(finite_mask[j])))
                         for j in range(N_ALPHA)])

        results[f'{gname}_ftle'] = ftle_all
        results[f'{gname}_mean'] = means
        results[f'{gname}_std'] = stds
        results[f'{gname}_sem'] = sems

    return results


def run_geometry_scan():
    """Compute λ(α) for stadium at different flat-section lengths a."""
    print(f"\n{'='*60}")
    print(f"  Geometry scan: {len(A_GEO_VALUES)} a-values × "
          f"{N_ALPHA} α × {N_ICS_GEO:,} ICs × {N_STEPS:,} collisions")
    print(f"{'='*60}")

    geo_means = np.empty((len(A_GEO_VALUES), N_ALPHA))
    geo_sems = np.empty((len(A_GEO_VALUES), N_ALPHA))

    t_scan_start = time.perf_counter()
    total_combos = len(A_GEO_VALUES) * N_ALPHA

    for k, a_geo in enumerate(A_GEO_VALUES):
        t_a_start = time.perf_counter()

        for j, alpha in enumerate(ALPHA_VALUES):
            t0 = time.perf_counter()
            lcns = lyapunov_ensemble(N_STEPS, N_ICS_GEO, alpha, STADIUM,
                                     a_geo, 0.0,
                                     perturb_mag=1e-7, u_max_frac=0.5)
            lcns = lcns[np.isfinite(lcns)]
            geo_means[k, j] = np.mean(lcns)
            geo_sems[k, j] = np.std(lcns) / max(1, np.sqrt(len(lcns)))
            dt = time.perf_counter() - t0

            combo_idx = k * N_ALPHA + j + 1
            elapsed = time.perf_counter() - t_scan_start
            remaining = elapsed / combo_idx * (total_combos - combo_idx)

            if (j + 1) % 10 == 0 or j == 0 or j == N_ALPHA - 1:
                print(f"  [GeoScan a={a_geo}] α = {alpha:.2f}  "
                      f"[{combo_idx}/{total_combos}]  "
                      f"{dt:.1f}s | elapsed {_fmt_time(elapsed)} | "
                      f"est. remaining {_fmt_time(remaining)}")

        dt_a = time.perf_counter() - t_a_start
        print(f"  a={a_geo} complete: {_fmt_time(dt_a)}")

    total = time.perf_counter() - t_scan_start
    print(f"  Geometry scan complete: {_fmt_time(total)}")

    return {
        'a_geo_values': A_GEO_VALUES,
        'GeoScan_mean': geo_means,
        'GeoScan_sem': geo_sems,
    }


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else '--all'

    print("Spinning billiards unified precomputation")
    print(f"Output: {OUTFILE}")
    warmup()

    data = {'alpha_values': ALPHA_VALUES}

    if mode in ('--all', '--main'):
        data.update(run_main_geometries())

    if mode in ('--all', '--geo'):
        data.update(run_geometry_scan())

    # Save
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(OUTFILE, **data)
    fsize = os.path.getsize(OUTFILE) / 1e6
    print(f"\nSaved: {OUTFILE} ({fsize:.1f} MB)")
    print("Keys:", list(data.keys()))


if __name__ == '__main__':
    main()
