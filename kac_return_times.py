"""Kac's-lemma estimate of the accessible phase-space volume.

Reviewer 2 (CHA26-AR-00869):

    "the reduction of the arguments in Rev 17 to the chaotic phase space
    fraction might not be correct. These arguments rest on Kac's lemma (for
    iterative maps) that states that the mean return time to a compact set S in
    phase space is given by the ratio of the accessible phase space volume to
    the volume of S. [...] One possibility might be to directly study the actual
    return times to a well-chosen area of phase space of chaotic trajectories,
    assuming ergodicity, to asses the accessible phase space volume without any
    knowledge of the regular phase space."

That is what this does.

Invariant measure of the collision map
--------------------------------------
The flow preserves dx dy dtheta du on the energy shell (u is constant during
free flight; the direction angle is constant; the motion is a translation). The
induced measure on the boundary cross-section carries the flux factor
v.n = v cos(theta). With spin, energy moves between translation and rotation, so
the speed depends on u:

    v(u) = sqrt(2E - alpha u^2),    |u| <= sqrt(2E/alpha)

Hence, taking 2E = 1 and substituting w = sin(theta) (so cos(theta) dtheta = dw),

    dmu  =  v(u) ds dw du / Z,        w in (-1, 1),  |u| <= u_max = 1/sqrt(alpha)
    Z    =  L_tot * 2 * I(-u_max, u_max),    I(a,b) = int_a^b sqrt(1 - alpha u^2) du

The manuscript states (Sec. II B) that the map "preserves the extended Liouville
measure cos(theta) ds dtheta du", i.e. a flat density in u. That is wrong: it
omits v(u). Fitting rho(u) ~ (1 - alpha u^2)^p to a 6e6-collision orbit gives
p = 0.49-0.64, converging to the predicted p = 1/2, and decisively excludes
p = 0. See `verify_measure()` below.

Kac's lemma
-----------
For an ergodic component C carrying normalised invariant measure, and a set
S contained in C, the mean return time to S (counted in collisions) is

    <tau_S>  =  mu(C) / mu(S)

with mu the invariant measure normalised over the whole shell. Equivalently, the
visit frequency of a single orbit in C is f_visit = mu(S)/mu(C). So

    mu(C)  =  mu(S) * <tau_S>  =  mu(S) / f_visit

which is the accessible phase-space volume, obtained without ever identifying
the regular component. mu(S) is known analytically from dmu above.

Validation the estimator must pass
----------------------------------
1. f_acc must be INDEPENDENT of the choice of S. This is the strong internal
   check: three disjoint windows must agree.
2. f_acc -> 1 as alpha -> 0 in the Sinai billiard, which is ergodic at alpha = 0.
3. Every S must lie inside the chaotic sea; we verify by launching ICs from S
   and confirming their FTLEs are all positive.

Note on the alpha -> 0 limit: u_max = 1/sqrt(alpha) diverges, because at
alpha = 0 the ball can spin arbitrarily fast at no energy cost and u becomes a
conserved label. The normalised measure on the shell is therefore singular at
alpha = 0, and "phase-space fraction" is not continuous there. We evaluate at
alpha > 0 throughout.
"""

import argparse
import math
import os

import numpy as np
from numba import njit, prange

from spinning_billiards import (
    SINAI, STADIUM, WALL_CIRCLE, WALL_CAP_LEFT, WALL_CAP_RIGHT,
    _next_collision, _reflect, lyapunov,
)


# --------------------------------------------------------------------------
#  Analytic pieces of the invariant measure
# --------------------------------------------------------------------------

def I_v(a, b, alpha):
    """int_a^b sqrt(1 - alpha u^2) du, exactly."""
    if alpha <= 0:
        return b - a
    r = math.sqrt(alpha)

    def F(u):
        u = max(-1.0 / r, min(1.0 / r, u))
        return 0.5 * u * math.sqrt(max(1.0 - alpha * u * u, 0.0)) + \
            0.5 / r * math.asin(max(-1.0, min(1.0, u * r)))
    return F(b) - F(a)


def u_max_of(alpha):
    return 1.0 / math.sqrt(alpha)


def sinai_boundary_length(L, R):
    return 8.0 * L + 2.0 * math.pi * R          # square perimeter + obstacle


def mu_S(alpha, arc_frac, w_lo, w_hi, u_lo, u_hi):
    """Normalised invariant measure of the box S.

    arc_frac : fraction of TOTAL boundary length occupied by the s-window
    """
    um = u_max_of(alpha)
    u_part = I_v(u_lo, u_hi, alpha) / I_v(-um, um, alpha)
    w_part = (w_hi - w_lo) / 2.0
    return arc_frac * w_part * u_part


# --------------------------------------------------------------------------
#  Orbit: count visits to S on the Sinai obstacle
# --------------------------------------------------------------------------

@njit(cache=True)
def _count_visits(n_coll, x, y, vx, vy, u, alpha, L, R,
                  phi_lo, phi_hi, w_lo, w_hi, u_lo, u_hi):
    """Run one orbit; count post-collision states landing in S.

    S = {wall = obstacle} x {phi in [phi_lo,phi_hi)} x {w in [w_lo,w_hi)}
        x {u in [u_lo,u_hi)}      with w = sin(theta), theta from inward normal.
    """
    hits = 0
    for i in range(n_coll):
        xn, yn, dt, tx, ty, nx, ny, wall = _next_collision(x, y, vx, vy,
                                                           SINAI, L, R)
        vx, vy, u = _reflect(vx, vy, u, alpha, tx, ty, nx, ny)
        x, y = xn, yn

        if wall == WALL_CIRCLE:
            sp = math.sqrt(vx * vx + vy * vy)
            if sp > 1e-13:
                w = (vx * tx + vy * ty) / sp          # sin(theta)
                phi = math.atan2(y, x)
                if phi < 0.0:
                    phi += 2.0 * math.pi
                if (phi_lo <= phi < phi_hi and w_lo <= w < w_hi
                        and u_lo <= u < u_hi):
                    hits += 1
    return hits


@njit(parallel=True, cache=True)
def _count_visits_ens(n_coll, ics, alpha, L, R,
                      phi_lo, phi_hi, w_lo, w_hi, u_lo, u_hi):
    n = ics.shape[0]
    out = np.zeros(n, dtype=np.int64)
    for k in prange(n):
        out[k] = _count_visits(n_coll, ics[k, 0], ics[k, 1], ics[k, 2],
                               ics[k, 3], ics[k, 4], alpha, L, R,
                               phi_lo, phi_hi, w_lo, w_hi, u_lo, u_hi)
    return out


@njit(cache=True)
def _seed_in_S(alpha, L, R, phi, w, u):
    """Post-collision state on the obstacle at angle phi, with sin(theta)=w."""
    x = R * math.cos(phi)
    y = R * math.sin(phi)
    nx, ny = math.cos(phi), math.sin(phi)        # outward from centre = into domain
    tx, ty = ny, -nx
    v = math.sqrt(max(1.0 - alpha * u * u, 1e-12))
    c = math.sqrt(max(1.0 - w * w, 0.0))
    vx = v * (c * nx + w * tx)
    vy = v * (c * ny + w * ty)
    return x, y, vx, vy, u


def make_ics_in_S(n, alpha, L, R, box, rng):
    phi_lo, phi_hi, w_lo, w_hi, u_lo, u_hi = box
    ics = np.empty((n, 5))
    for k in range(n):
        phi = rng.uniform(phi_lo, phi_hi)
        w = rng.uniform(w_lo, w_hi)
        u = rng.uniform(u_lo, u_hi)
        ics[k] = _seed_in_S(alpha, L, R, phi, w, u)
    return ics


# --------------------------------------------------------------------------
#  Driver
# --------------------------------------------------------------------------

def f_accessible(alpha, box, n_coll, n_orbits, L=2.0, R=1.0, seed=0):
    """Kac estimate of the accessible (chaotic-component) volume fraction."""
    rng = np.random.default_rng(seed)
    ics = make_ics_in_S(n_orbits, alpha, L, R, box, rng)
    hits = _count_visits_ens(n_coll, ics, alpha, L, R, *box)

    total = n_coll * n_orbits
    f_visit = hits.sum() / total
    if f_visit == 0:
        return np.nan, np.nan, 0.0

    phi_lo, phi_hi, w_lo, w_hi, u_lo, u_hi = box
    arc = R * (phi_hi - phi_lo) / sinai_boundary_length(L, R)
    m = mu_S(alpha, arc, w_lo, w_hi, u_lo, u_hi)

    f_acc = m / f_visit
    # per-orbit spread -> uncertainty
    fv = hits / n_coll
    fv = fv[fv > 0]
    f_acc_sem = (m / fv).std() / math.sqrt(max(len(fv), 1)) if len(fv) > 1 else 0.0
    return f_acc, f_acc_sem, f_visit


def check_S_is_chaotic(alpha, box, n_steps, n_ic, L=2.0, R=1.0, seed=1,
                       thresh=0.01):
    rng = np.random.default_rng(seed)
    ics = make_ics_in_S(n_ic, alpha, L, R, box, rng)
    lams = []
    for k in range(n_ic):
        _, lcn = lyapunov(n_steps, ics[k, 0], ics[k, 1], ics[k, 2], ics[k, 3],
                          ics[k, 4], alpha, SINAI, L, R, 1e-7)
        lams.append(lcn[-1])
    lams = np.array(lams)
    return float((lams > thresh).mean()), float(lams.min()), float(lams.mean())


def sampler_coverage(alpha, frac=0.5):
    """Fraction of the shell's invariant measure with |u| <= frac * u_max.

    This is the region the paper's IC sampler draws from.
    """
    um = u_max_of(alpha)
    return I_v(-frac * um, frac * um, alpha) / I_v(-um, um, alpha)
