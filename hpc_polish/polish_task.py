"""HPC array task for the high-statistics 'polish' campaign.

Five datasets, selected by the DATASET environment variable; the cell and
chunk come from SLURM_ARRAY_TASK_ID. Every task writes one compressed npz of
per-IC results into $RESULTS_DIR, so all downstream figures can be rebuilt at
any binning without re-running anything.

  ablang    2 geometries x 41 alpha, 4 ablation modes on paired ICs,
            1e5 ICs/cell in 8 chunks x 5e4 collisions.
            Per IC and mode: lambda, mean 1/cos(theta) at curved walls
            (truncated at |cos| >= 1/80), grazing fraction; plus a pooled
            200-bin |cos(theta)| histogram per mode.
  geoscan   5 stadium half-lengths x 41 alpha, full coupling,
            1e5 ICs/cell in 4 chunks. Per-IC lambda.
  rscan     6 Sinai radii x 41 alpha, full coupling,
            65,536 ICs/cell in 2 chunks. Per-IC lambda.
  spectrum  2 geometries x 21 alpha, full 5D Benettin spectrum,
            16,384 ICs/cell in 8 chunks. Per-IC 5-exponent spectra.
  kac       21 alpha x 3 disjoint boxes on the Sinai obstacle,
            128 orbits x 2e6 collisions. Per-orbit visit counts + mu(S).

Seeds are deterministic per (dataset, cell, chunk) and recorded in each npz.
"""

import math
import os
import time

import numpy as np
from numba import njit, prange

from spinning_billiards import (
    STADIUM, SINAI,
    WALL_CIRCLE, WALL_CAP_LEFT, WALL_CAP_RIGHT,
    _next_collision, _reflect, lyapunov_ensemble, lyapunov_spectrum_ensemble,
)
from compute_spin_ablation import MODES, _alpha_for, draw_ics
from kac_return_times import (
    _count_visits_ens, make_ics_in_S, mu_S, u_max_of, sinai_boundary_length,
)

N_BINS = 200
SEC_CLIP = 80.0          # truncate 1/cos at 80 (|cos| >= 1/80)

ALPHA41 = np.round(np.linspace(0.0, 1.0, 41), 6)
ALPHA21 = np.round(np.linspace(0.0, 1.0, 21), 6)
ALPHA101 = np.round(np.linspace(0.0, 1.0, 101), 6)
# fine grid for the initial drop, uniform in sqrt(alpha) (the natural
# small-alpha variable: rotation angle 4*arctan(sqrt(alpha)))
ALPHA_FINE = np.round(0.05 * (np.arange(1, 10) / 10.0) ** 2, 6)


# ----------------------------------------------------------------------
# ablang kernel: Benettin lambda + curved-wall angle statistics per IC
# ----------------------------------------------------------------------

@njit(cache=True)
def _lyap_angle_stats(n_steps, x0, y0, vx0, vy0, u0, alpha, mode,
                      geo, p1, p2, d0, hist):
    rx = x0; ry = y0; rvx = vx0; rvy = vy0; ru = u0
    dp = np.empty(5)
    for k in range(5):
        dp[k] = np.random.randn()
    nrm = 0.0
    for k in range(5):
        nrm += dp[k] * dp[k]
    nrm = math.sqrt(nrm)
    for k in range(5):
        dp[k] = dp[k] / nrm * d0
    px = rx + dp[0]; py = ry + dp[1]
    pvx = rvx + dp[2]; pvy = rvy + dp[3]; pu = ru + dp[4]

    T = 0.0; S = 0.0
    n_curved = 0; sec_sum = 0.0; n_graz = 0

    for i in range(n_steps):
        xn, yn, dt, tx, ty, nx, ny, w = _next_collision(rx, ry, rvx, rvy,
                                                        geo, p1, p2)
        curved = (w == WALL_CIRCLE or w == WALL_CAP_LEFT or w == WALL_CAP_RIGHT)
        if curved:
            sp = math.sqrt(rvx * rvx + rvy * rvy)
            if sp > 1e-13:
                c = abs((rvx * nx + rvy * ny) / sp)
                if c > 1.0:
                    c = 1.0
                n_curved += 1
                sec_sum += 1.0 / max(c, 1.0 / SEC_CLIP)
                if c < 0.1:
                    n_graz += 1
                b = int(c * N_BINS)
                if b >= N_BINS:
                    b = N_BINS - 1
                hist[b] += 1.0
        ae = _alpha_for(w, alpha, mode)
        rvx2, rvy2, ru2 = _reflect(rvx, rvy, ru, ae, tx, ty, nx, ny)
        rx2 = xn; ry2 = yn

        xn2, yn2, dt2, tx2, ty2, nx2, ny2, w2 = _next_collision(
            px, py, pvx, pvy, geo, p1, p2)
        ae2 = _alpha_for(w2, alpha, mode)
        pvx2, pvy2, pu2 = _reflect(pvx, pvy, pu, ae2, tx2, ty2, nx2, ny2)

        T += dt
        sep = math.sqrt((rx2 - xn2)**2 + (ry2 - yn2)**2 +
                        (rvx2 - pvx2)**2 + (rvy2 - pvy2)**2 + (ru2 - pu2)**2)
        if sep < 1e-30:
            sep = 1e-30
        S += math.log(sep / d0)
        b = d0 / sep
        px = rx2 + b * (xn2 - rx2); py = ry2 + b * (yn2 - ry2)
        pvx = rvx2 + b * (pvx2 - rvx2); pvy = rvy2 + b * (pvy2 - rvy2)
        pu = ru2 + b * (pu2 - ru2)
        rx = rx2; ry = ry2; rvx = rvx2; rvy = rvy2; ru = ru2

    lam = S / T if T > 0 else np.nan
    secm = sec_sum / n_curved if n_curved > 0 else np.nan
    graz = n_graz / n_curved if n_curved > 0 else np.nan
    return lam, secm, graz


@njit(parallel=True, cache=True)
def _ablang_chunk(n_steps, ics, alpha, mode, geo, p1, p2):
    n = ics.shape[0]
    lam = np.empty(n); sec = np.empty(n); grz = np.empty(n)
    hists = np.zeros((n, N_BINS))
    for k in prange(n):
        h = np.zeros(N_BINS)
        l, s, g = _lyap_angle_stats(n_steps, ics[k, 0], ics[k, 1], ics[k, 2],
                                    ics[k, 3], ics[k, 4], alpha, mode,
                                    geo, p1, p2, 1e-7, h)
        lam[k] = l; sec[k] = s; grz[k] = g
        for b in range(N_BINS):
            hists[k, b] = h[b]
    return lam, sec, grz, hists


def run_ablang(task):
    GEOS = [("sinai", SINAI, 2.0, 1.0), ("stadium", STADIUM, 1.0, 0.0)]
    N_IC, N_CHUNK, N_STEPS = 100_000, 4, 50_000
    n_cells = len(GEOS) * len(ALPHA41)
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < n_cells
    gi, ai = divmod(cell, len(ALPHA41))
    gname, geo, p1, p2 = GEOS[gi]
    a = float(ALPHA41[ai])
    n = N_IC // N_CHUNK
    seed = 100003 * 1 + 1009 * cell + chunk

    ics = draw_ics(n, geo, p1, p2, a, 0.5, seed)
    out = {}
    for mname, mcode in MODES.items():
        np.random.seed(seed + 7 * mcode + 1)   # perturbation directions
        lam, sec, grz, hists = _ablang_chunk(N_STEPS, ics, a, mcode,
                                             geo, p1, p2)
        out[f"lam_{mname}"] = lam.astype(np.float32)
        out[f"sec_{mname}"] = sec.astype(np.float32)
        out[f"graz_{mname}"] = grz.astype(np.float32)
        out[f"hist_{mname}"] = hists.sum(axis=0)
    fn = f"{os.environ['RESULTS_DIR']}/ablang_{gname}_a{ai:02d}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, geometry=gname, seed=seed,
                        n_steps=N_STEPS, sec_clip=SEC_CLIP, **out)
    print("wrote", fn, flush=True)


def run_geoscan(task):
    A_GEO = [0.2, 0.5, 1.0, 2.0, 4.0]
    N_IC, N_CHUNK, N_STEPS = 100_000, 1, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    gi, ai = divmod(cell, len(ALPHA41))
    a_geo = A_GEO[gi]; a = float(ALPHA41[ai])
    n = N_IC // N_CHUNK
    lc = lyapunov_ensemble(N_STEPS, n, a, STADIUM, a_geo, 0.0,
                           perturb_mag=1e-7, u_max_frac=0.5)
    fn = f"{os.environ['RESULTS_DIR']}/geoscan_a{a_geo}_i{ai:02d}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, a_geo=a_geo, n_steps=N_STEPS,
                        lam=lc.astype(np.float32))
    print("wrote", fn, flush=True)


def run_rscan(task):
    RS = [0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
    N_IC, N_CHUNK, N_STEPS = 65_536, 1, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    ri, ai = divmod(cell, len(ALPHA41))
    R = RS[ri]; a = float(ALPHA41[ai])
    n = N_IC // N_CHUNK
    lc = lyapunov_ensemble(N_STEPS, n, a, SINAI, 2.0, R,
                           perturb_mag=1e-7, u_max_frac=0.5)
    fn = f"{os.environ['RESULTS_DIR']}/rscan_R{R}_i{ai:02d}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, R=R, n_steps=N_STEPS,
                        lam=lc.astype(np.float32))
    print("wrote", fn, flush=True)


def run_spectrum(task):
    GEOS = [("sinai", SINAI, 2.0, 1.0), ("stadium", STADIUM, 1.0, 0.0)]
    N_IC, N_CHUNK, N_STEPS = 16_384, 8, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    gi, ai = divmod(cell, len(ALPHA21))
    gname, geo, p1, p2 = GEOS[gi]
    a = float(ALPHA21[ai])
    n = N_IC // N_CHUNK
    sp = lyapunov_spectrum_ensemble(N_STEPS, n, a, geo, p1, p2,
                                    perturb_mag=1e-7, u_max_frac=0.5)
    fn = f"{os.environ['RESULTS_DIR']}/spectrum_{gname}_i{ai:02d}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, geometry=gname, n_steps=N_STEPS,
                        spectra=sp.astype(np.float32))
    print("wrote", fn, flush=True)


def run_kac(task):
    N_ORBITS, N_COLL, L, R = 128, 2_000_000, 2.0, 1.0
    ai, bi = divmod(task, 3)
    a = float(ALPHA21[ai])
    if a == 0.0:
        a = 0.02          # u_max diverges at alpha = 0; use the smallest cell
    um = u_max_of(a)
    boxes = [
        (0.0, math.pi / 2, -0.5, 0.5, -0.3 * um, 0.3 * um),
        (math.pi, 1.5 * math.pi, 0.0, 1.0, -0.6 * um, 0.6 * um),
        (math.pi / 2, math.pi, -1.0, 0.0, 0.1 * um, 0.7 * um),
    ]
    box = boxes[bi]
    rng = np.random.default_rng(555 + task)
    ics = make_ics_in_S(N_ORBITS, a, L, R, box, rng)
    hits = _count_visits_ens(N_COLL, ics, a, L, R, *box)
    phi_lo, phi_hi, w_lo, w_hi, u_lo, u_hi = box
    arc = R * (phi_hi - phi_lo) / sinai_boundary_length(L, R)
    m = mu_S(a, arc, w_lo, w_hi, u_lo, u_hi)
    fn = f"{os.environ['RESULTS_DIR']}/kac_i{ai:02d}_b{bi}.npz"
    np.savez_compressed(fn, alpha=a, box_index=bi, box=np.array(box),
                        n_coll=N_COLL, mu_S=m, hits=hits)
    print("wrote", fn, flush=True)


# ----------------------------------------------------------------------
# Final ("go big") campaign, 2026-07-10 evening: one order of magnitude
# above the morning run on every dataset, planned as the last one.
# ----------------------------------------------------------------------

def run_ablang10(task):
    """2 geo x 41 alpha x 4 modes, 1e6 paired ICs per cell, 20 chunks."""
    GEOS = [("sinai", SINAI, 2.0, 1.0), ("stadium", STADIUM, 1.0, 0.0)]
    N_IC, N_CHUNK, N_STEPS = 1_000_000, 20, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(GEOS) * len(ALPHA41)
    gi, ai = divmod(cell, len(ALPHA41))
    gname, geo, p1, p2 = GEOS[gi]
    a = float(ALPHA41[ai])
    n = N_IC // N_CHUNK
    seed = 700003 + 1009 * cell + chunk

    ics = draw_ics(n, geo, p1, p2, a, 0.5, seed)
    out = {}
    for mname, mcode in MODES.items():
        np.random.seed(seed + 7 * mcode + 1)   # perturbation directions
        lam, sec, grz, hists = _ablang_chunk(N_STEPS, ics, a, mcode,
                                             geo, p1, p2)
        out[f"lam_{mname}"] = lam.astype(np.float32)
        out[f"sec_{mname}"] = sec.astype(np.float32)
        out[f"graz_{mname}"] = grz.astype(np.float32)
        out[f"hist_{mname}"] = hists.sum(axis=0)
    fn = (f"{os.environ['RESULTS_DIR']}/"
          f"ablang10_{gname}_a{ai:02d}_c{chunk:02d}.npz")
    np.savez_compressed(fn, alpha=a, geometry=gname, seed=seed,
                        n_steps=N_STEPS, sec_clip=SEC_CLIP, **out)
    print("wrote", fn, flush=True)


def run_ftle(task):
    """Stadium + Sinai per-IC FTLE/lambda: 101 alpha x 1e6 ICs, 5 chunks.
    One dataset feeding Fig 2 (mean lambda), Fig 3 (FTLE distributions),
    Fig 5 (chaotic fraction with threshold bands) and the island figure's
    measured regular fraction."""
    GEOS = [("sinai", SINAI, 2.0, 1.0), ("stadium", STADIUM, 1.0, 0.0)]
    N_IC, N_CHUNK, N_STEPS = 1_000_000, 5, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(GEOS) * len(ALPHA101)
    gi, ai = divmod(cell, len(ALPHA101))
    gname, geo, p1, p2 = GEOS[gi]
    a = float(ALPHA101[ai])
    n = N_IC // N_CHUNK
    lc = lyapunov_ensemble(N_STEPS, n, a, geo, p1, p2,
                           perturb_mag=1e-7, u_max_frac=0.5)
    fn = f"{os.environ['RESULTS_DIR']}/ftle_{gname}_i{ai:03d}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, geometry=gname, n_steps=N_STEPS,
                        lam=lc.astype(np.float32))
    print("wrote", fn, flush=True)


def run_geoscan10(task):
    A_GEO = [0.2, 0.5, 1.0, 2.0, 4.0]
    N_IC, N_CHUNK, N_STEPS = 1_000_000, 5, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(A_GEO) * len(ALPHA41)
    gi, ai = divmod(cell, len(ALPHA41))
    a_geo = A_GEO[gi]; a = float(ALPHA41[ai])
    n = N_IC // N_CHUNK
    lc = lyapunov_ensemble(N_STEPS, n, a, STADIUM, a_geo, 0.0,
                           perturb_mag=1e-7, u_max_frac=0.5)
    fn = (f"{os.environ['RESULTS_DIR']}/"
          f"geoscan10_a{a_geo}_i{ai:02d}_c{chunk}.npz")
    np.savez_compressed(fn, alpha=a, a_geo=a_geo, n_steps=N_STEPS,
                        lam=lc.astype(np.float32))
    print("wrote", fn, flush=True)


def run_rscan10(task):
    """Adds R = 1.35 to fill the gap in the recovery-amplitude trend."""
    RS = [0.3, 0.5, 0.8, 1.0, 1.2, 1.35, 1.5]
    N_IC, N_CHUNK, N_STEPS = 655_360, 4, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(RS) * len(ALPHA41)
    ri, ai = divmod(cell, len(ALPHA41))
    R = RS[ri]; a = float(ALPHA41[ai])
    n = N_IC // N_CHUNK
    lc = lyapunov_ensemble(N_STEPS, n, a, SINAI, 2.0, R,
                           perturb_mag=1e-7, u_max_frac=0.5)
    fn = f"{os.environ['RESULTS_DIR']}/rscan10_R{R}_i{ai:02d}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, R=R, n_steps=N_STEPS,
                        lam=lc.astype(np.float32))
    print("wrote", fn, flush=True)


def run_spectrum10(task):
    GEOS = [("sinai", SINAI, 2.0, 1.0), ("stadium", STADIUM, 1.0, 0.0)]
    N_IC, N_CHUNK, N_STEPS = 65_536, 16, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(GEOS) * len(ALPHA21)
    gi, ai = divmod(cell, len(ALPHA21))
    gname, geo, p1, p2 = GEOS[gi]
    a = float(ALPHA21[ai])
    n = N_IC // N_CHUNK
    sp = lyapunov_spectrum_ensemble(N_STEPS, n, a, geo, p1, p2,
                                    perturb_mag=1e-7, u_max_frac=0.5)
    fn = (f"{os.environ['RESULTS_DIR']}/"
          f"spectrum10_{gname}_i{ai:02d}_c{chunk:02d}.npz")
    np.savez_compressed(fn, alpha=a, geometry=gname, n_steps=N_STEPS,
                        spectra=sp.astype(np.float32))
    print("wrote", fn, flush=True)


def run_kac10(task):
    """41-alpha grid (restoration zone resolved) x 3 boxes, 512 orbits
    per box in 4 chunks of 128."""
    N_ORBITS, N_COLL, L, R = 128, 2_000_000, 2.0, 1.0
    cell, chunk = divmod(task, 4)
    assert cell < len(ALPHA41) * 3
    ai, bi = divmod(cell, 3)
    a = float(ALPHA41[ai])
    if a == 0.0:
        a = 0.02          # u_max diverges at alpha = 0; use the smallest cell
    um = u_max_of(a)
    boxes = [
        (0.0, math.pi / 2, -0.5, 0.5, -0.3 * um, 0.3 * um),
        (math.pi, 1.5 * math.pi, 0.0, 1.0, -0.6 * um, 0.6 * um),
        (math.pi / 2, math.pi, -1.0, 0.0, 0.1 * um, 0.7 * um),
    ]
    box = boxes[bi]
    rng = np.random.default_rng(750003 + task)
    ics = make_ics_in_S(N_ORBITS, a, L, R, box, rng)
    hits = _count_visits_ens(N_COLL, ics, a, L, R, *box)
    phi_lo, phi_hi, w_lo, w_hi, u_lo, u_hi = box
    arc = R * (phi_hi - phi_lo) / sinai_boundary_length(L, R)
    m = mu_S(a, arc, w_lo, w_hi, u_lo, u_hi)
    fn = f"{os.environ['RESULTS_DIR']}/kac10_i{ai:02d}_b{bi}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, box_index=bi, box=np.array(box),
                        n_coll=N_COLL, mu_S=m, hits=hits)
    print("wrote", fn, flush=True)


# --- fine-alpha extension of the final campaign: 9 sqrt-ramped points in
# --- (0, 0.05) for the four datasets whose lambda(alpha) knee is unresolved.

def run_ablang_fine(task):
    GEOS = [("sinai", SINAI, 2.0, 1.0), ("stadium", STADIUM, 1.0, 0.0)]
    N_IC, N_CHUNK, N_STEPS = 1_000_000, 20, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(GEOS) * len(ALPHA_FINE)
    gi, ai = divmod(cell, len(ALPHA_FINE))
    gname, geo, p1, p2 = GEOS[gi]
    a = float(ALPHA_FINE[ai])
    n = N_IC // N_CHUNK
    seed = 760003 + 1009 * cell + chunk
    ics = draw_ics(n, geo, p1, p2, a, 0.5, seed)
    out = {}
    for mname, mcode in MODES.items():
        np.random.seed(seed + 7 * mcode + 1)
        lam, sec, grz, hists = _ablang_chunk(N_STEPS, ics, a, mcode,
                                             geo, p1, p2)
        out[f"lam_{mname}"] = lam.astype(np.float32)
        out[f"sec_{mname}"] = sec.astype(np.float32)
        out[f"graz_{mname}"] = grz.astype(np.float32)
        out[f"hist_{mname}"] = hists.sum(axis=0)
    fn = (f"{os.environ['RESULTS_DIR']}/"
          f"ablangf_{gname}_a{ai}_c{chunk:02d}.npz")
    np.savez_compressed(fn, alpha=a, geometry=gname, seed=seed,
                        n_steps=N_STEPS, sec_clip=SEC_CLIP, **out)
    print("wrote", fn, flush=True)


def run_ftle_fine(task):
    GEOS = [("sinai", SINAI, 2.0, 1.0), ("stadium", STADIUM, 1.0, 0.0)]
    N_IC, N_CHUNK, N_STEPS = 1_000_000, 5, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(GEOS) * len(ALPHA_FINE)
    gi, ai = divmod(cell, len(ALPHA_FINE))
    gname, geo, p1, p2 = GEOS[gi]
    a = float(ALPHA_FINE[ai])
    n = N_IC // N_CHUNK
    lc = lyapunov_ensemble(N_STEPS, n, a, geo, p1, p2,
                           perturb_mag=1e-7, u_max_frac=0.5)
    fn = f"{os.environ['RESULTS_DIR']}/ftlef_{gname}_i{ai}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, geometry=gname, n_steps=N_STEPS,
                        lam=lc.astype(np.float32))
    print("wrote", fn, flush=True)


def run_geoscan_fine(task):
    A_GEO = [0.2, 0.5, 1.0, 2.0, 4.0]
    N_IC, N_CHUNK, N_STEPS = 1_000_000, 5, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(A_GEO) * len(ALPHA_FINE)
    gi, ai = divmod(cell, len(ALPHA_FINE))
    a_geo = A_GEO[gi]; a = float(ALPHA_FINE[ai])
    n = N_IC // N_CHUNK
    lc = lyapunov_ensemble(N_STEPS, n, a, STADIUM, a_geo, 0.0,
                           perturb_mag=1e-7, u_max_frac=0.5)
    fn = (f"{os.environ['RESULTS_DIR']}/"
          f"geoscanf_a{a_geo}_i{ai}_c{chunk}.npz")
    np.savez_compressed(fn, alpha=a, a_geo=a_geo, n_steps=N_STEPS,
                        lam=lc.astype(np.float32))
    print("wrote", fn, flush=True)


def run_rscan_fine(task):
    RS = [0.3, 0.5, 0.8, 1.0, 1.2, 1.35, 1.5]
    N_IC, N_CHUNK, N_STEPS = 655_360, 4, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(RS) * len(ALPHA_FINE)
    ri, ai = divmod(cell, len(ALPHA_FINE))
    R = RS[ri]; a = float(ALPHA_FINE[ai])
    n = N_IC // N_CHUNK
    lc = lyapunov_ensemble(N_STEPS, n, a, SINAI, 2.0, R,
                           perturb_mag=1e-7, u_max_frac=0.5)
    fn = f"{os.environ['RESULTS_DIR']}/rscanf_R{R}_i{ai}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, R=R, n_steps=N_STEPS,
                        lam=lc.astype(np.float32))
    print("wrote", fn, flush=True)


# --- R-scan top-up to an even 1e6 ICs per point (655,360 + 344,640) ---

def run_rscan_top(task):
    RS = [0.3, 0.5, 0.8, 1.0, 1.2, 1.35, 1.5]
    N_IC, N_CHUNK, N_STEPS = 344_640, 2, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(RS) * len(ALPHA41)
    ri, ai = divmod(cell, len(ALPHA41))
    R = RS[ri]; a = float(ALPHA41[ai])
    n = N_IC // N_CHUNK
    lc = lyapunov_ensemble(N_STEPS, n, a, SINAI, 2.0, R,
                           perturb_mag=1e-7, u_max_frac=0.5)
    fn = f"{os.environ['RESULTS_DIR']}/rscantop_R{R}_i{ai:02d}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, R=R, n_steps=N_STEPS,
                        lam=lc.astype(np.float32))
    print("wrote", fn, flush=True)


def run_rscanf_top(task):
    RS = [0.3, 0.5, 0.8, 1.0, 1.2, 1.35, 1.5]
    N_IC, N_CHUNK, N_STEPS = 344_640, 2, 50_000
    cell, chunk = divmod(task, N_CHUNK)
    assert cell < len(RS) * len(ALPHA_FINE)
    ri, ai = divmod(cell, len(ALPHA_FINE))
    R = RS[ri]; a = float(ALPHA_FINE[ai])
    n = N_IC // N_CHUNK
    lc = lyapunov_ensemble(N_STEPS, n, a, SINAI, 2.0, R,
                           perturb_mag=1e-7, u_max_frac=0.5)
    fn = f"{os.environ['RESULTS_DIR']}/rscanftop_R{R}_i{ai}_c{chunk}.npz"
    np.savez_compressed(fn, alpha=a, R=R, n_steps=N_STEPS,
                        lam=lc.astype(np.float32))
    print("wrote", fn, flush=True)


RUNNERS = {"ablang": run_ablang, "geoscan": run_geoscan, "rscan": run_rscan,
           "spectrum": run_spectrum, "kac": run_kac,
           "ablang10": run_ablang10, "ftle": run_ftle,
           "geoscan10": run_geoscan10, "rscan10": run_rscan10,
           "spectrum10": run_spectrum10, "kac10": run_kac10,
           "ablang_fine": run_ablang_fine, "ftle_fine": run_ftle_fine,
           "geoscan_fine": run_geoscan_fine, "rscan_fine": run_rscan_fine,
           "rscan_top": run_rscan_top, "rscanf_top": run_rscanf_top}

if __name__ == "__main__":
    ds = os.environ["DATASET"]
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    t0 = time.perf_counter()
    RUNNERS[ds](task)
    print(f"[{ds} task {task}] done in {time.perf_counter()-t0:.0f}s")
