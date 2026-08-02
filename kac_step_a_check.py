"""Measure n_c and kappa: collisions and time between successive obstacle hits.

Used by the Kac appendix notes (Sec. 8). Two independent configurations:
  seed 5,  64 orbits x 2e5 collisions   (run 1)
  seed 7, 128 orbits x 3e5 collisions   (run 2, quoted in the notes)
Licensed rows (alpha <= 0.125, alpha = 1) reproduce between runs to 0.1%.
At alpha = 0.5 the runs disagree by 7% (n_c) / 14% (kappa): inside the
fragmented window these are seed-dependent mixture averages, not physical
single-component quantities.

Anchors: n_c(0) = 1/mu(S_obs) = 3.5465;  kappa(0) = pi*A/(2*pi*R) = 6.4292
(measured 3.5464 and 6.4294; each within 0.005%).
Regular orbits never strike the obstacle and are excluded via their nan.
"""
import math
import numpy as np
from numba import njit, prange
from spinning_billiards import SINAI, WALL_CIRCLE, _next_collision, _reflect


@njit(cache=True)
def kappa_one(n_coll, x, y, vx, vy, u, alpha, L, R):
    t_since = 0.0; n_since = 0
    tot_t = 0.0; tot_n = 0; nvis = 0
    started = False
    for i in range(n_coll):
        xn, yn, dt, tx, ty, nx, ny, wall = _next_collision(x, y, vx, vy, SINAI, L, R)
        vx, vy, u = _reflect(vx, vy, u, alpha, tx, ty, nx, ny)
        x, y = xn, yn
        t_since += dt; n_since += 1
        if wall == WALL_CIRCLE:
            if started:
                tot_t += t_since; tot_n += n_since; nvis += 1
            started = True
            t_since = 0.0; n_since = 0
    if nvis == 0:
        return np.nan, np.nan
    return tot_t / nvis, tot_n / nvis


@njit(parallel=True, cache=True)
def kappa_ens(n_coll, ics, alpha, L, R):
    m = ics.shape[0]
    kt = np.empty(m); kn = np.empty(m)
    for k in prange(m):
        a, b = kappa_one(n_coll, ics[k, 0], ics[k, 1], ics[k, 2],
                         ics[k, 3], ics[k, 4], alpha, L, R)
        kt[k] = a; kn[k] = b
    return kt, kn


def run(seed, n_orbits, n_coll, alphas=(0.0, 0.05, 0.10, 0.125, 0.5, 1.0),
        L=2.0, R=1.0):
    rng = np.random.default_rng(seed)
    print(f"seed={seed}  {n_orbits} orbits x {n_coll} collisions")
    print(f"{'alpha':>6} {'kappa':>8} {'n_c':>8}")
    for alpha in alphas:
        ics = np.empty((n_orbits, 5))
        for k in range(n_orbits):
            while True:
                x, y = rng.uniform(-L, L, 2)
                if x * x + y * y > R * R * 1.05:
                    break
            um = 1.0 / math.sqrt(alpha) if alpha > 0 else 0.0
            uu = rng.uniform(-0.5 * um, 0.5 * um) if alpha > 0 else rng.uniform(-1, 1)
            v = math.sqrt(max(1.0 - alpha * uu * uu, 1e-9))
            th = rng.uniform(0, 2 * math.pi)
            ics[k] = (x, y, v * math.cos(th), v * math.sin(th), uu)
        kt, kn = kappa_ens(n_coll, ics, alpha, L, R)
        print(f"{alpha:6.3f} {np.nanmean(kt):8.4f} {np.nanmean(kn):8.4f}")


if __name__ == "__main__":
    A = 16.0 - math.pi
    print(f"anchors: 1/mu(S_obs) = {(16 + 2*math.pi)/(2*math.pi):.4f}   "
          f"Santalo pi*A/(2*pi*R) = {A/2:.4f}")
    run(5, 64, 200_000)
    run(7, 128, 300_000)
