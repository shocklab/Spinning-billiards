"""Per-collision stretch of the flat-only ablation variant vs spinless control.

Checks the Sec. VIII sentence about the flat-only stretch. Measured (this
script + master_ablang lambda values, alpha = 1):
    per collision:  Sinai 0.478,  stadium 0.505
    per unit time:  Sinai 0.407,  stadium 0.425
The manuscript's 0.43 matches the stadium per-TIME fall, not a per-collision
fall; the sentence needs relabelling.
"""
import numpy as np
from numba import njit, prange
from spinning_billiards import STADIUM, SINAI, _next_collision, _reflect
from compute_spin_ablation import _alpha_for, draw_ics, MODE_FLAT, MODE_NONE


@njit(cache=True)
def nu_variant(n_coll, x, y, vx, vy, u, alpha, mode, geo, p1, p2):
    T = 0.0
    for i in range(n_coll):
        xn, yn, dt, tx, ty, nx, ny, wall = _next_collision(x, y, vx, vy, geo, p1, p2)
        ae = _alpha_for(wall, alpha, mode)
        vx, vy, u = _reflect(vx, vy, u, ae, tx, ty, nx, ny)
        x, y = xn, yn
        T += dt
    return n_coll / T


@njit(parallel=True, cache=True)
def nu_ens(n_coll, ics, alpha, mode, geo, p1, p2):
    m = ics.shape[0]
    out = np.empty(m)
    for k in prange(m):
        out[k] = nu_variant(n_coll, ics[k, 0], ics[k, 1], ics[k, 2], ics[k, 3],
                            ics[k, 4], alpha, mode, geo, p1, p2)
    return out


if __name__ == "__main__":
    d = np.load("experiment_results/polish/master_ablang.npz")
    lam, al = d["lam"], np.asarray(d["alpha"], float)
    j1 = int(np.argmin(np.abs(al - 1.0)))
    for gi, (name, geo, p1, p2) in enumerate((("sinai", SINAI, 2.0, 1.0),
                                              ("stadium", STADIUM, 1.0, 0.0))):
        ics = draw_ics(64, geo, p1, p2, 1.0, 0.5, 3)
        nu_fo = np.nanmean(nu_ens(100_000, ics, 1.0, MODE_FLAT, geo, p1, p2))
        nu_ct = np.nanmean(nu_ens(100_000, ics, 1.0, MODE_NONE, geo, p1, p2))
        lam_fo, lam_ct = np.nanmean(lam[gi, 2, j1]), np.nanmean(lam[gi, 3, j1])
        print(f"{name}: lam_fo={lam_fo:.4f} lam_ctrl={lam_ct:.4f} "
              f"nu_fo={nu_fo:.4f} nu_ctrl={nu_ct:.4f}")
        print(f"   per-collision ratio {(lam_fo/nu_fo)/(lam_ct/nu_ct):.3f}   "
              f"per-time ratio {lam_fo/lam_ct:.3f}")
