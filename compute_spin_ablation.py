"""Where does spin act? Ablate the spin coupling by wall type.

Both referees for CHA26-AR-00869 argue that the Q = v_par - alpha*u mechanism
cannot explain chaos suppression, because flat-wall collisions do not generate
chaos (Bunimovich, Nonlinearity 31 (2018) R78). The paper's mechanism section
attributes suppression to Q being conserved along runs of same-orientation
(flat) walls.

Attributing Benettin log-stretch to individual collisions is ill-posed:
stretching accrues along free flight, and growth seeded by a curved wall is
realised on the segment after it. (See compute_wall_decomposition.py, whose
'end' and 'start' conventions disagree by an order of magnitude -- that
quantity should not be used to draw conclusions.)

This script instead runs a clean ablation. The collision law takes alpha_flat
at flat walls and alpha_curved at curved walls:

    full     : alpha_flat = alpha_curved = alpha        (the paper's dynamics)
    curved   : alpha_flat = 0,  alpha_curved = alpha    (spin only at curved walls)
    flat     : alpha_flat = alpha, alpha_curved = 0     (spin only at flat walls)
    none     : alpha_flat = alpha_curved = 0            (spinless control)

Each variant conserves E = v^2/2 + alpha_eff * u^2 / 2, because a specular
collision changes neither |v| nor u. So all four are legitimate dynamics on
the same energy shell, and lambda is comparable across them.

Pre-registered decision rule (committed before looking at output):

  * If lambda_curved(alpha) tracks lambda_full(alpha) and lambda_flat(alpha)
    stays near lambda(0), spin acts at the curved walls. The Q-on-flat-runs
    mechanism is dead, and the referees are right.
  * If lambda_flat(alpha) reproduces most of the suppression, the referees'
    premise is wrong: flat-wall spin coupling matters even though flat walls
    generate no chaos at alpha = 0. Q can be rehabilitated, and this is the
    single most valuable figure in the revision.
  * If neither alone reproduces it but both together do, the suppression is
    cooperative and needs the arrival-angle analysis (theta at curved walls).

We additionally record the incidence-angle distribution at curved walls, since
the per-collision expansion of a dispersing billiard scales with 1/cos(theta):
if spin suppresses chaos by reshaping arrival angles, it shows up here.
"""

import argparse
import math
import os
import time

import numpy as np
from numba import njit, prange

from spinning_billiards import (
    CIRCLE, RECTANGLE, STADIUM, SINAI,
    WALL_CIRCLE, WALL_CAP_LEFT, WALL_CAP_RIGHT,
    _next_collision, _reflect,
    _random_ic_circle, _random_ic_rectangle, _random_ic_stadium,
    _random_ic_sinai, _random_velocity_with_spin,
)

MODE_FULL, MODE_CURVED, MODE_FLAT, MODE_NONE = 0, 1, 2, 3
N_THETA_BINS = 40


@njit(cache=True)
def _is_curved(w):
    return w == WALL_CIRCLE or w == WALL_CAP_LEFT or w == WALL_CAP_RIGHT


@njit(cache=True)
def _alpha_for(w, alpha, mode):
    curved = _is_curved(w)
    if mode == MODE_FULL:
        return alpha
    elif mode == MODE_CURVED:
        return alpha if curved else 0.0
    elif mode == MODE_FLAT:
        return 0.0 if curved else alpha
    else:
        return 0.0


@njit(cache=True)
def _step(x, y, vx, vy, u, alpha, mode, geo_type, p1, p2):
    xn, yn, dt, tx, ty, nx, ny, w = _next_collision(x, y, vx, vy,
                                                    geo_type, p1, p2)
    a_eff = _alpha_for(w, alpha, mode)
    vxn, vyn, un = _reflect(vx, vy, u, a_eff, tx, ty, nx, ny)
    return xn, yn, vxn, vyn, un, dt, w, nx, ny


@njit(cache=True)
def _lyap_ablated(n_steps, x0, y0, vx0, vy0, u0, alpha, mode,
                  geo_type, p1, p2, perturb_mag, theta_hist):
    """Benettin under the ablated collision law. Fills theta_hist in place
    with |cos(theta)| at curved-wall collisions (theta from inward normal)."""
    rx = x0; ry = y0; rvx = vx0; rvy = vy0; ru = u0

    dp = np.empty(5)
    for k in range(5):
        dp[k] = np.random.randn()
    nrm = 0.0
    for k in range(5):
        nrm += dp[k] * dp[k]
    nrm = math.sqrt(nrm)
    for k in range(5):
        dp[k] = dp[k] / nrm * perturb_mag

    px = rx + dp[0]; py = ry + dp[1]
    pvx = rvx + dp[2]; pvy = rvy + dp[3]; pu = ru + dp[4]

    cumtime = 0.0
    sum_log_beta = 0.0
    n_curved = 0

    for i in range(n_steps):
        rx2, ry2, rvx2, rvy2, ru2, dt_r, w, nx, ny = _step(
            rx, ry, rvx, rvy, ru, alpha, mode, geo_type, p1, p2)
        px2, py2, pvx2, pvy2, pu2, dt_p, _, _, _ = _step(
            px, py, pvx, pvy, pu, alpha, mode, geo_type, p1, p2)

        cumtime += dt_r

        if _is_curved(w):
            n_curved += 1
            sp = math.sqrt(rvx * rvx + rvy * rvy)
            if sp > 1e-12:
                c = abs((rvx * nx + rvy * ny) / sp)
                if c > 1.0:
                    c = 1.0
                b = int(c * N_THETA_BINS)
                if b >= N_THETA_BINS:
                    b = N_THETA_BINS - 1
                theta_hist[b] += 1.0

        sep = math.sqrt((rx2 - px2)**2 + (ry2 - py2)**2 +
                        (rvx2 - pvx2)**2 + (rvy2 - pvy2)**2 +
                        (ru2 - pu2)**2)
        if sep < 1e-30:
            sep = 1e-30

        beta = perturb_mag / sep
        sum_log_beta += math.log(beta)

        px2 = rx2 + beta * (px2 - rx2)
        py2 = ry2 + beta * (py2 - ry2)
        pvx2 = rvx2 + beta * (pvx2 - rvx2)
        pvy2 = rvy2 + beta * (pvy2 - rvy2)
        pu2 = ru2 + beta * (pu2 - ru2)

        rx = rx2; ry = ry2; rvx = rvx2; rvy = rvy2; ru = ru2
        px = px2; py = py2; pvx = pvx2; pvy = pvy2; pu = pu2

    lam = -sum_log_beta / cumtime if cumtime > 0 else 0.0
    rate_curved = n_curved / cumtime if cumtime > 0 else 0.0
    return lam, cumtime, n_curved, rate_curved


@njit(cache=True)
def _draw_ic(geo_type, p1, p2, alpha, u_max_frac):
    if geo_type == CIRCLE:
        x0, y0 = _random_ic_circle()
    elif geo_type == RECTANGLE:
        x0, y0 = _random_ic_rectangle(p1, p2)
    elif geo_type == STADIUM:
        x0, y0 = _random_ic_stadium(p1)
    else:
        x0, y0 = _random_ic_sinai(p1, p2)
    vx0, vy0, u0 = _random_velocity_with_spin(alpha, u_max_frac)
    return x0, y0, vx0, vy0, u0


@njit(cache=True)
def draw_ics(n_ics, geo_type, p1, p2, alpha, u_max_frac, seed):
    """Draw ICs once, inside numba, with numba's RNG explicitly seeded.

    The four ablation modes must run on *identical* initial conditions: the
    measurement is the difference between them, and pairing removes the
    ensemble variance from that difference. np.random.seed() called from
    Python does not seed numba's per-thread RNG, so the ICs are generated
    here (serially, seeded) and passed into the parallel kernel.
    """
    np.random.seed(seed)
    ics = np.empty((n_ics, 5))
    for k in range(n_ics):
        x0, y0, vx0, vy0, u0 = _draw_ic(geo_type, p1, p2, alpha, u_max_frac)
        ics[k, 0] = x0; ics[k, 1] = y0
        ics[k, 2] = vx0; ics[k, 3] = vy0; ics[k, 4] = u0
    return ics


@njit(parallel=True, cache=True)
def ablation_ensemble_ics(n_steps, ics, alpha, mode, geo_type, p1, p2,
                          perturb_mag):
    n_ics = ics.shape[0]
    lam = np.zeros(n_ics)
    rate_c = np.zeros(n_ics)
    frac_c = np.zeros(n_ics)
    hists = np.zeros((n_ics, N_THETA_BINS))

    for k in prange(n_ics):
        h = np.zeros(N_THETA_BINS)
        l, t, nc, rc = _lyap_ablated(
            n_steps, ics[k, 0], ics[k, 1], ics[k, 2], ics[k, 3], ics[k, 4],
            alpha, mode, geo_type, p1, p2, perturb_mag, h)
        lam[k] = l
        rate_c[k] = rc
        frac_c[k] = nc / n_steps
        for b in range(N_THETA_BINS):
            hists[k, b] = h[b]

    return lam, rate_c, frac_c, hists


def ablation_ensemble(n_steps, n_ics, alpha, mode, geo_type, p1, p2,
                      perturb_mag, u_max_frac, seed=0):
    ics = draw_ics(n_ics, geo_type, p1, p2, alpha, u_max_frac, seed)
    return ablation_ensemble_ics(n_steps, ics, alpha, mode,
                                 geo_type, p1, p2, perturb_mag)


GEOMETRIES = {
    "Stadium": (STADIUM, 1.0, 0.0),
    "Sinai": (SINAI, 2.0, 1.0),
    "Rectangle": (RECTANGLE, 1.0, 1.0),
}
MODES = {"full": MODE_FULL, "curved": MODE_CURVED,
         "flat": MODE_FLAT, "none": MODE_NONE}


def run(geometry, alphas, n_ics, n_steps, u_max_frac, perturb_mag, outdir, seed):
    geo_type, p1, p2 = GEOMETRIES[geometry]
    n_a = len(alphas)
    out = {}
    for mname in MODES:
        out[f"lambda_{mname}"] = np.zeros(n_a)
        out[f"sem_lambda_{mname}"] = np.zeros(n_a)
    out["rate_curved_full"] = np.zeros(n_a)
    out["frac_curved_full"] = np.zeros(n_a)
    out["cos_theta_hist_full"] = np.zeros((n_a, N_THETA_BINS))
    # Paired suppression shares, measured against the spinless-dynamics
    # control 'none' (which still carries the IC sampler's speed reduction).
    for k in ("drop_full", "drop_flat", "drop_curved"):
        out[k] = np.zeros(n_a)
        out[f"sem_{k}"] = np.zeros(n_a)
    out["share_flat"] = np.zeros(n_a)
    out["share_curved"] = np.zeros(n_a)

    for j, a in enumerate(alphas):
        t0 = time.perf_counter()
        line = f"  a={a:.3f} "
        # One IC set per alpha, shared by all four modes: the comparison
        # between modes is paired, so ensemble variance cancels in the
        # differences we care about.
        ics = draw_ics(n_ics, geo_type, p1, p2, a, u_max_frac, seed + j)
        lams = {}
        for mname, mcode in MODES.items():
            lam, rc, fc, hists = ablation_ensemble_ics(
                n_steps, ics, a, mcode, geo_type, p1, p2, perturb_mag)
            lams[mname] = lam
            if mname == "full":
                gf = np.isfinite(lam)
                out["rate_curved_full"][j] = rc[gf].mean()
                out["frac_curved_full"][j] = fc[gf].mean()
                h = hists[gf].sum(axis=0)
                s = h.sum()
                out["cos_theta_hist_full"][j] = h / s if s > 0 else h

        g = np.ones(n_ics, dtype=bool)
        for lam in lams.values():
            g &= np.isfinite(lam)
        m = max(int(g.sum()), 1)
        for mname, lam in lams.items():
            out[f"lambda_{mname}"][j] = lam[g].mean()
            out[f"sem_lambda_{mname}"][j] = lam[g].std() / math.sqrt(m)
            line += f" {mname}={out[f'lambda_{mname}'][j]:.5f}"

        d_full = lams["none"][g] - lams["full"][g]
        d_flat = lams["none"][g] - lams["flat"][g]
        d_curv = lams["none"][g] - lams["curved"][g]
        for k, d in (("drop_full", d_full), ("drop_flat", d_flat),
                     ("drop_curved", d_curv)):
            out[k][j] = d.mean()
            out[f"sem_{k}"][j] = d.std() / math.sqrt(m)
        denom = d_full.mean()
        if abs(denom) > 1e-9:
            out["share_flat"][j] = d_flat.mean() / denom
            out["share_curved"][j] = d_curv.mean() / denom

        print(line + f"  ({time.perf_counter()-t0:.1f}s)", flush=True)

    os.makedirs(outdir, exist_ok=True)
    tag = f"{geometry.lower()}_umf{u_max_frac:.2f}_seed{seed}"
    path = os.path.join(outdir, f"spin_ablation_{tag}.npz")
    np.savez(path, alpha=np.asarray(alphas), n_ics=n_ics, n_steps=n_steps,
             u_max_frac=u_max_frac, perturb_mag=perturb_mag, seed=seed, **out)
    print(f"  wrote {path}")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default="Sinai", choices=sorted(GEOMETRIES))
    ap.add_argument("--n-alpha", type=int, default=21)
    ap.add_argument("--n-ics", type=int, default=512)
    ap.add_argument("--n-steps", type=int, default=50_000)
    ap.add_argument("--u-max-frac", type=float, default=0.5)
    ap.add_argument("--perturb-mag", type=float, default=1e-7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="experiment_results/spin_ablation")
    args = ap.parse_args()

    alphas = np.linspace(0.0, 1.0, args.n_alpha)
    print(f"[{args.geometry}] {args.n_ics} ICs x {args.n_steps:,} collisions, "
          f"u_max_frac={args.u_max_frac}, seed={args.seed}")
    run(args.geometry, alphas, args.n_ics, args.n_steps,
        args.u_max_frac, args.perturb_mag, args.outdir, args.seed)


if __name__ == "__main__":
    main()
