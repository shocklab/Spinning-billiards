"""
Unified simulation framework for spinning billiards.

Simulates billiard dynamics with spin coupling across four geometries:
Circle, Rectangle, Stadium (Bunimovich), and Sinai.

All core loops are Numba JIT-compiled for performance.
"""

import numpy as np
import math
from numba import njit, prange

# ---------------------------------------------------------------------------
# Geometry type constants
# ---------------------------------------------------------------------------
CIRCLE = 0
RECTANGLE = 1
STADIUM = 2
SINAI = 3

# Wall/surface identifiers (used in Poincaré sections)
WALL_TOP = 0
WALL_LEFT = 1
WALL_BOTTOM = 2
WALL_RIGHT = 3
WALL_CIRCLE = 4        # circle boundary or Sinai obstacle
WALL_CAP_LEFT = 5      # stadium left semicircle
WALL_CAP_RIGHT = 6     # stadium right semicircle

# Small epsilon to avoid self-intersection
_EPS = 1e-12


# ===========================================================================
#  Collision detection — one function per geometry
# ===========================================================================

@njit(cache=True)
def _collision_circle(x, y, vx, vy):
    """Next collision with unit circle (ball inside)."""
    v2 = vx * vx + vy * vy
    b = (x * vx + y * vy) / v2
    c = (x * x + y * y - 1.0) / v2
    disc = b * b - c
    t = -b + math.sqrt(disc)
    xn = x + t * vx
    yn = y + t * vy
    # Tangent T = (-yn, xn), Normal n = (-xn, -yn)  [inward, unit on S^1]
    return xn, yn, t, -yn, xn, -xn, -yn, WALL_CIRCLE


@njit(cache=True)
def _collision_rectangle(x, y, vx, vy, L, H):
    """Next collision with rectangle [-L,L] x [-H,H]."""
    t_min = 1e30
    wall = -1

    # Right wall x = L
    if vx > _EPS:
        t_cand = (L - x) / vx
        if t_cand > _EPS and t_cand < t_min:
            t_min = t_cand
            wall = WALL_RIGHT
    # Left wall x = -L
    if vx < -_EPS:
        t_cand = (-L - x) / vx
        if t_cand > _EPS and t_cand < t_min:
            t_min = t_cand
            wall = WALL_LEFT
    # Top wall y = H
    if vy > _EPS:
        t_cand = (H - y) / vy
        if t_cand > _EPS and t_cand < t_min:
            t_min = t_cand
            wall = WALL_TOP
    # Bottom wall y = -H
    if vy < -_EPS:
        t_cand = (-H - y) / vy
        if t_cand > _EPS and t_cand < t_min:
            t_min = t_cand
            wall = WALL_BOTTOM

    xn = x + t_min * vx
    yn = y + t_min * vy

    # Tangent and inward normal (T x n = +z convention)
    if wall == WALL_TOP:
        return xn, yn, t_min, -1.0, 0.0, 0.0, -1.0, wall
    elif wall == WALL_LEFT:
        return xn, yn, t_min, 0.0, -1.0, 1.0, 0.0, wall
    elif wall == WALL_BOTTOM:
        return xn, yn, t_min, 1.0, 0.0, 0.0, 1.0, wall
    else:  # WALL_RIGHT
        return xn, yn, t_min, 0.0, 1.0, -1.0, 0.0, wall


@njit(cache=True)
def _circle_hit(x, y, vx, vy, cx, cy, R):
    """Solve for ray-circle intersection; return (t, valid).

    Returns the smallest positive t such that |pos(t) - center|^2 = R^2.
    For ball OUTSIDE the circle (Sinai obstacle): use -b - sqrt(disc).
    For ball INSIDE the circle (circle billiard, stadium caps): use -b + sqrt(disc).
    This helper returns BOTH roots and lets the caller choose.
    """
    dx = x - cx
    dy = y - cy
    v2 = vx * vx + vy * vy
    b = (dx * vx + dy * vy) / v2
    c = (dx * dx + dy * dy - R * R) / v2
    disc = b * b - c
    if disc < 0.0:
        return -1.0, -1.0, False
    sq = math.sqrt(disc)
    t_minus = -b - sq   # smaller root
    t_plus = -b + sq    # larger root
    return t_minus, t_plus, True


@njit(cache=True)
def _collision_stadium(x, y, vx, vy, L):
    """Next collision with Bunimovich stadium.

    Geometry: flat walls y = ±1 for |x| ≤ L, semicircular caps of radius 1
    centered at (±L, 0).
    """
    t_min = 1e30
    wall = -1
    xn = x
    yn = y
    tx = 0.0; ty = 0.0; nx = 0.0; ny = 0.0

    # --- Flat walls ---
    # Top wall y = 1
    if vy > _EPS:
        t_cand = (1.0 - y) / vy
        if t_cand > _EPS:
            x_hit = x + t_cand * vx
            if -L - _EPS <= x_hit <= L + _EPS and t_cand < t_min:
                t_min = t_cand
                wall = WALL_TOP
                xn = x_hit; yn = 1.0
                tx = -1.0; ty = 0.0; nx = 0.0; ny = -1.0
    # Bottom wall y = -1
    if vy < -_EPS:
        t_cand = (-1.0 - y) / vy
        if t_cand > _EPS:
            x_hit = x + t_cand * vx
            if -L - _EPS <= x_hit <= L + _EPS and t_cand < t_min:
                t_min = t_cand
                wall = WALL_BOTTOM
                xn = x_hit; yn = -1.0
                tx = 1.0; ty = 0.0; nx = 0.0; ny = 1.0

    # --- Left semicircular cap: circle center (-L, 0), radius 1 ---
    t_m, t_p, valid = _circle_hit(x, y, vx, vy, -L, 0.0, 1.0)
    if valid and t_p > _EPS:
        # Ball is inside stadium, so use the larger root (t_plus)
        t_cand = t_p
        if t_cand > _EPS and t_cand < t_min:
            xh = x + t_cand * vx
            yh = y + t_cand * vy
            if xh <= -L + _EPS:
                t_min = t_cand
                wall = WALL_CAP_LEFT
                xn = xh; yn = yh
                # Tangent and inward normal for circle centered at (-L, 0)
                dx = xn + L; dy = yn
                tx = -dy; ty = dx     # T = (-dy, dx)
                nx = -dx; ny = -dy    # n = (-dx, -dy) pointing inward

    # --- Right semicircular cap: circle center (L, 0), radius 1 ---
    t_m, t_p, valid = _circle_hit(x, y, vx, vy, L, 0.0, 1.0)
    if valid and t_p > _EPS:
        t_cand = t_p
        if t_cand > _EPS and t_cand < t_min:
            xh = x + t_cand * vx
            yh = y + t_cand * vy
            if xh >= L - _EPS:
                t_min = t_cand
                wall = WALL_CAP_RIGHT
                xn = xh; yn = yh
                dx = xn - L; dy = yn
                tx = -dy; ty = dx
                nx = -dx; ny = -dy

    return xn, yn, t_min, tx, ty, nx, ny, wall


@njit(cache=True)
def _collision_sinai(x, y, vx, vy, L, R):
    """Next collision in Sinai billiard: square [-L,L]^2 with circular
    obstacle of radius R at origin."""
    t_min = 1e30
    wall = -1
    xn = x; yn = y
    tx = 0.0; ty = 0.0; nx = 0.0; ny = 0.0

    # --- Four walls ---
    # Top y = L
    if vy > _EPS:
        t_cand = (L - y) / vy
        if t_cand > _EPS and t_cand < t_min:
            t_min = t_cand
            wall = WALL_TOP
            xn = x + t_cand * vx; yn = L
            tx = -1.0; ty = 0.0; nx = 0.0; ny = -1.0
    # Bottom y = -L
    if vy < -_EPS:
        t_cand = (-L - y) / vy
        if t_cand > _EPS and t_cand < t_min:
            t_min = t_cand
            wall = WALL_BOTTOM
            xn = x + t_cand * vx; yn = -L
            tx = 1.0; ty = 0.0; nx = 0.0; ny = 1.0
    # Left x = -L
    if vx < -_EPS:
        t_cand = (-L - x) / vx
        if t_cand > _EPS and t_cand < t_min:
            t_min = t_cand
            wall = WALL_LEFT
            xn = -L; yn = y + t_cand * vy
            tx = 0.0; ty = -1.0; nx = 1.0; ny = 0.0
    # Right x = L
    if vx > _EPS:
        t_cand = (L - x) / vx
        if t_cand > _EPS and t_cand < t_min:
            t_min = t_cand
            wall = WALL_RIGHT
            xn = L; yn = y + t_cand * vy
            tx = 0.0; ty = 1.0; nx = -1.0; ny = 0.0

    # --- Central circle obstacle (ball outside, bounces off exterior) ---
    t_m, t_p, valid = _circle_hit(x, y, vx, vy, 0.0, 0.0, R)
    if valid and t_m > _EPS and t_m < t_min:
        xh = x + t_m * vx
        yh = y + t_m * vy
        t_min = t_m
        wall = WALL_CIRCLE
        xn = xh; yn = yh
        # Normal points outward (away from center, into domain)
        inv = 1.0 / R
        nx = xn * inv; ny = yn * inv
        # Tangent: T x n = +z  =>  T = (ny, -nx) = (yn/R, -xn/R)
        tx = yn * inv; ty = -xn * inv

    return xn, yn, t_min, tx, ty, nx, ny, wall


# ===========================================================================
#  Reflection law (universal for all geometries)
# ===========================================================================

@njit(cache=True)
def _reflect(vx, vy, u, alpha, tx, ty, nx, ny):
    """Apply spin-coupled reflection.

    Parameters
    ----------
    vx, vy : velocity before collision
    u : spin before collision
    alpha : dimensionless moment of inertia parameter
    tx, ty : unit tangent at collision point
    nx, ny : unit inward normal at collision point

    Returns
    -------
    vx_new, vy_new, u_new
    """
    vT = vx * tx + vy * ty           # tangential velocity component
    vperp = -(vx * nx + vy * ny)     # normal component (positive = into table)

    if alpha == 0.0:
        # No spin coupling: specular reflection
        vx_new = vx + 2.0 * vperp * nx
        vy_new = vy + 2.0 * vperp * ny
        return vx_new, vy_new, u

    c1 = (1.0 - alpha) / (1.0 + alpha)
    c2 = 2.0 * alpha / (1.0 + alpha)
    c3 = 2.0 / (1.0 + alpha)

    vparr = c1 * vT - c2 * u
    u_new = -c1 * u - c3 * vT

    vx_new = vparr * tx + vperp * nx
    vy_new = vparr * ty + vperp * ny

    return vx_new, vy_new, u_new


# ===========================================================================
#  Main simulation loop
# ===========================================================================

@njit(cache=True)
def _next_collision(x, y, vx, vy, geo_type, geo_p1, geo_p2):
    """Dispatch to geometry-specific collision detection."""
    if geo_type == CIRCLE:
        return _collision_circle(x, y, vx, vy)
    elif geo_type == RECTANGLE:
        return _collision_rectangle(x, y, vx, vy, geo_p1, geo_p2)
    elif geo_type == STADIUM:
        return _collision_stadium(x, y, vx, vy, geo_p1)
    else:  # SINAI
        return _collision_sinai(x, y, vx, vy, geo_p1, geo_p2)


@njit(cache=True)
def simulate(n_collisions, x0, y0, vx0, vy0, u0, alpha,
             geo_type, geo_p1, geo_p2):
    """Run billiard simulation for n_collisions.

    Parameters
    ----------
    n_collisions : int
    x0, y0, vx0, vy0, u0 : initial conditions
    alpha : spin coupling parameter (0 = specular)
    geo_type : CIRCLE, RECTANGLE, STADIUM, or SINAI
    geo_p1, geo_p2 : geometry parameters
        CIRCLE:    ignored (use 0.0, 0.0)
        RECTANGLE: L (half-width), H (half-height)
        STADIUM:   L (half-length of flat section), ignored
        SINAI:     L (half-side of square), R (obstacle radius)

    Returns
    -------
    xs, ys, vxs, vys, us, ts, walls : arrays of length n_collisions+1
        ts[i] = cumulative time at collision i
        walls[i] = which surface was hit at collision i
    """
    N = n_collisions + 1
    xs = np.empty(N)
    ys = np.empty(N)
    vxs = np.empty(N)
    vys = np.empty(N)
    us = np.empty(N)
    ts = np.empty(N)
    walls = np.empty(N, dtype=np.int32)

    xs[0] = x0; ys[0] = y0
    vxs[0] = vx0; vys[0] = vy0
    us[0] = u0; ts[0] = 0.0; walls[0] = -1

    for i in range(1, N):
        xn, yn, dt, tx, ty, nx, ny, w = _next_collision(
            xs[i-1], ys[i-1], vxs[i-1], vys[i-1], geo_type, geo_p1, geo_p2)
        vxn, vyn, un = _reflect(vxs[i-1], vys[i-1], us[i-1], alpha, tx, ty, nx, ny)

        xs[i] = xn; ys[i] = yn
        vxs[i] = vxn; vys[i] = vyn
        us[i] = un
        ts[i] = ts[i-1] + dt
        walls[i] = w

    return xs, ys, vxs, vys, us, ts, walls


# ===========================================================================
#  Lyapunov exponent computation (Benettin algorithm)
# ===========================================================================

@njit(cache=True)
def _step_one_collision(x, y, vx, vy, u, alpha, geo_type, geo_p1, geo_p2):
    """Advance one collision. Returns new state and elapsed time."""
    xn, yn, dt, tx, ty, nx, ny, w = _next_collision(
        x, y, vx, vy, geo_type, geo_p1, geo_p2)
    vxn, vyn, un = _reflect(vx, vy, u, alpha, tx, ty, nx, ny)
    return xn, yn, vxn, vyn, un, dt


@njit(cache=True)
def lyapunov(n_steps, x0, y0, vx0, vy0, u0, alpha,
             geo_type, geo_p1, geo_p2, perturb_mag=1e-7):
    """Compute Lyapunov characteristic number via Benettin renormalization.

    Uses the correct 5D state vector [x, y, vx, vy, u].

    Parameters
    ----------
    n_steps : number of renormalization steps (= collisions)
    perturb_mag : initial perturbation magnitude in state space

    Returns
    -------
    t_cum : cumulative time array (length n_steps)
    lcn : LCN estimate at each step (length n_steps)
    """
    t_cum = np.empty(n_steps)
    lcn = np.empty(n_steps)

    # Reference trajectory state
    rx = x0; ry = y0; rvx = vx0; rvy = vy0; ru = u0

    # Generate random perturbation direction in 5D
    dp = np.empty(5)
    dp[0] = np.random.randn()
    dp[1] = np.random.randn()
    dp[2] = np.random.randn()
    dp[3] = np.random.randn()
    dp[4] = np.random.randn()
    norm = math.sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2 + dp[3]**2 + dp[4]**2)
    for k in range(5):
        dp[k] = dp[k] / norm * perturb_mag

    # Perturbed trajectory state
    px = rx + dp[0]; py = ry + dp[1]
    pvx = rvx + dp[2]; pvy = rvy + dp[3]
    pu = ru + dp[4]

    cumtime = 0.0
    sum_log_beta = 0.0

    for i in range(n_steps):
        # Evolve both trajectories by one collision
        rx2, ry2, rvx2, rvy2, ru2, dt_r = _step_one_collision(
            rx, ry, rvx, rvy, ru, alpha, geo_type, geo_p1, geo_p2)
        px2, py2, pvx2, pvy2, pu2, dt_p = _step_one_collision(
            px, py, pvx, pvy, pu, alpha, geo_type, geo_p1, geo_p2)

        # Use reference trajectory's time for cumulative time
        cumtime += dt_r

        # Compute separation in state space
        sep = math.sqrt((rx2 - px2)**2 + (ry2 - py2)**2 +
                        (rvx2 - pvx2)**2 + (rvy2 - pvy2)**2 +
                        (ru2 - pu2)**2)

        if sep < 1e-30:
            sep = 1e-30

        # beta = perturb_mag / sep (rescaling factor)
        beta = perturb_mag / sep
        sum_log_beta += math.log(beta)

        # Renormalize: place perturbed trajectory at distance perturb_mag
        # from reference, in the direction of their separation
        px2 = rx2 + beta * (px2 - rx2)
        py2 = ry2 + beta * (py2 - ry2)
        pvx2 = rvx2 + beta * (pvx2 - rvx2)
        pvy2 = rvy2 + beta * (pvy2 - rvy2)
        pu2 = ru2 + beta * (pu2 - ru2)

        # Update states
        rx = rx2; ry = ry2; rvx = rvx2; rvy = rvy2; ru = ru2
        px = px2; py = py2; pvx = pvx2; pvy = pvy2; pu = pu2

        t_cum[i] = cumtime
        if cumtime > 0.0:
            lcn[i] = -sum_log_beta / cumtime
        else:
            lcn[i] = 0.0

    return t_cum, lcn


# ===========================================================================
#  Full Lyapunov spectrum (Benettin with Gram-Schmidt)
# ===========================================================================

@njit(cache=True)
def lyapunov_spectrum(n_steps, x0, y0, vx0, vy0, u0, alpha,
                      geo_type, geo_p1, geo_p2, perturb_mag=1e-7):
    """Compute full Lyapunov spectrum via Benettin-Gram-Schmidt.

    Tracks 5 perturbation vectors simultaneously, applying modified
    Gram-Schmidt orthogonalization after each collision step.

    Uses the 5D state vector [x, y, vx, vy, u].

    Parameters
    ----------
    n_steps : number of collision steps
    x0, y0, vx0, vy0, u0 : initial conditions
    alpha : spin coupling parameter
    geo_type : geometry type (CIRCLE, RECTANGLE, STADIUM, SINAI)
    geo_p1, geo_p2 : geometry parameters
    perturb_mag : perturbation magnitude for shadow trajectories

    Returns
    -------
    exponents : array of shape (5,) -- the 5 Lyapunov exponents in decreasing order
    t_total : float -- total elapsed time
    """
    ndim = 5

    # Reference trajectory state
    rx = x0; ry = y0; rvx = vx0; rvy = vy0; ru = u0

    # Initialize 5 perturbation directions as orthonormal unit vectors
    # Q[i, j] = i-th unit perturbation direction, j-th component
    Q = np.zeros((ndim, ndim))
    for i in range(ndim):
        Q[i, i] = 1.0

    # Accumulate log stretching factors for each exponent
    sum_log = np.zeros(ndim)
    cumtime = 0.0

    for step in range(n_steps):
        # --- Evolve reference trajectory by one collision ---
        rx2, ry2, rvx2, rvy2, ru2, dt_r = _step_one_collision(
            rx, ry, rvx, rvy, ru, alpha, geo_type, geo_p1, geo_p2)
        cumtime += dt_r

        # --- Evolve each perturbed trajectory ---
        # displacement[i, j] = displacement vector for i-th perturbation
        disp = np.zeros((ndim, ndim))
        for i in range(ndim):
            # Perturbed IC = reference + unit_direction * perturb_mag
            px = rx + Q[i, 0] * perturb_mag
            py = ry + Q[i, 1] * perturb_mag
            pvx = rvx + Q[i, 2] * perturb_mag
            pvy = rvy + Q[i, 3] * perturb_mag
            pu = ru + Q[i, 4] * perturb_mag

            # Evolve perturbed trajectory
            px2, py2, pvx2, pvy2, pu2, dt_p = _step_one_collision(
                px, py, pvx, pvy, pu, alpha, geo_type, geo_p1, geo_p2)

            # Displacement = perturbed - reference (after evolution)
            disp[i, 0] = px2 - rx2
            disp[i, 1] = py2 - ry2
            disp[i, 2] = pvx2 - rvx2
            disp[i, 3] = pvy2 - rvy2
            disp[i, 4] = pu2 - ru2

        # --- Modified Gram-Schmidt orthogonalization ---
        # Orthogonalize disp[0], disp[1], ..., disp[4] in order
        # After GS, Q[i] stores the unit vector for the i-th direction
        for i in range(ndim):
            # Subtract projections onto all previously orthogonalized vectors
            # Q[j] for j < i are already unit vectors from this step
            for j in range(i):
                # Compute dot product: disp[i] . Q[j]  (Q[j] is unit vector)
                dot = 0.0
                for d in range(ndim):
                    dot += disp[i, d] * Q[j, d]
                # Subtract projection
                for d in range(ndim):
                    disp[i, d] -= dot * Q[j, d]

            # Compute norm of disp[i]
            norm_i = 0.0
            for d in range(ndim):
                norm_i += disp[i, d] * disp[i, d]
            norm_i = math.sqrt(norm_i)

            if norm_i < 1e-30:
                norm_i = 1e-30

            # Record log of stretching factor (norm / perturb_mag)
            sum_log[i] += math.log(norm_i / perturb_mag)

            # Normalize to unit vector
            for d in range(ndim):
                Q[i, d] = disp[i, d] / norm_i

        # --- Update reference state ---
        rx = rx2; ry = ry2; rvx = rvx2; rvy = rvy2; ru = ru2

    # Compute Lyapunov exponents
    exponents = np.empty(ndim)
    if cumtime > 0.0:
        for i in range(ndim):
            exponents[i] = sum_log[i] / cumtime
    else:
        for i in range(ndim):
            exponents[i] = 0.0

    # Sort in decreasing order (bubble sort for 5 elements)
    for i in range(ndim):
        for j in range(i + 1, ndim):
            if exponents[j] > exponents[i]:
                tmp = exponents[i]
                exponents[i] = exponents[j]
                exponents[j] = tmp

    return exponents, cumtime


@njit(cache=True, parallel=True)
def lyapunov_spectrum_ensemble(n_steps, n_ics, alpha, geo_type, geo_p1, geo_p2,
                                perturb_mag=1e-7, u_max_frac=0.5):
    """Compute ensemble-averaged Lyapunov spectrum.

    Runs n_ics independent trajectories in parallel and returns the
    full spectrum for each.

    Parameters
    ----------
    n_steps : collisions per trajectory
    n_ics : number of initial conditions
    alpha : spin coupling parameter
    geo_type : geometry type (CIRCLE, RECTANGLE, STADIUM, SINAI)
    geo_p1, geo_p2 : geometry parameters
    perturb_mag : perturbation magnitude
    u_max_frac : fraction of max allowed |u| to sample from (0 to 1)

    Returns
    -------
    spectra : array of shape (n_ics, 5) -- spectrum for each IC (decreasing order)
    """
    spectra = np.empty((n_ics, 5))

    for k in prange(n_ics):
        # Generate random IC based on geometry
        if geo_type == CIRCLE:
            x0, y0 = _random_ic_circle()
        elif geo_type == RECTANGLE:
            x0, y0 = _random_ic_rectangle(geo_p1, geo_p2)
        elif geo_type == STADIUM:
            x0, y0 = _random_ic_stadium(geo_p1)
        else:
            x0, y0 = _random_ic_sinai(geo_p1, geo_p2)

        vx0, vy0, u0 = _random_velocity_with_spin(alpha, u_max_frac)

        exps, t_total = lyapunov_spectrum(
            n_steps, x0, y0, vx0, vy0, u0, alpha,
            geo_type, geo_p1, geo_p2, perturb_mag)

        for j in range(5):
            spectra[k, j] = exps[j]

    return spectra


# ===========================================================================
#  Ensemble-averaged Lyapunov exponent
# ===========================================================================

@njit(cache=True)
def _random_ic_circle():
    """Random initial condition inside the unit circle."""
    while True:
        x = 2.0 * np.random.random() - 1.0
        y = 2.0 * np.random.random() - 1.0
        if x * x + y * y < 0.99:
            break
    return x, y


@njit(cache=True)
def _random_ic_rectangle(L, H):
    """Random initial condition inside rectangle."""
    x = (2.0 * np.random.random() - 1.0) * L * 0.99
    y = (2.0 * np.random.random() - 1.0) * H * 0.99
    return x, y


@njit(cache=True)
def _random_ic_stadium(L):
    """Random initial condition inside stadium."""
    while True:
        x = (2.0 * np.random.random() - 1.0) * (L + 1.0) * 0.99
        y = (2.0 * np.random.random() - 1.0) * 0.99
        # Check inside stadium: either in rectangle or in caps
        if -L <= x <= L:
            break  # in rectangular part
        elif x < -L:
            if (x + L)**2 + y**2 < 0.99:
                break
        else:
            if (x - L)**2 + y**2 < 0.99:
                break
    return x, y


@njit(cache=True)
def _random_ic_sinai(L, R):
    """Random initial condition inside Sinai billiard (outside obstacle)."""
    while True:
        x = (2.0 * np.random.random() - 1.0) * L * 0.99
        y = (2.0 * np.random.random() - 1.0) * L * 0.99
        if x * x + y * y > (R * 1.01)**2:
            break
    return x, y


@njit(cache=True)
def _random_velocity_with_spin(alpha_param, u_max_frac=0.5):
    """Random velocity direction and spin, consistent with energy E = 0.5.

    Total energy: 0.5*v^2 + 0.5*alpha*u^2 = 0.5
    So: v^2 = 1 - alpha*u^2, need |u| <= 1/sqrt(alpha).
    """
    theta = 2.0 * math.pi * np.random.random()
    if alpha_param > 0.0:
        u_max = 1.0 / math.sqrt(alpha_param)
        u = (2.0 * np.random.random() - 1.0) * u_max * u_max_frac
        v = math.sqrt(max(1.0 - alpha_param * u * u, 0.01))
    else:
        u = 0.0
        v = 1.0
    vx = v * math.cos(theta)
    vy = v * math.sin(theta)
    return vx, vy, u


@njit(parallel=True, cache=True)
def lyapunov_ensemble(n_steps, n_ics, alpha_param, geo_type, geo_p1, geo_p2,
                      perturb_mag=1e-7, u_max_frac=0.5):
    """Compute ensemble-averaged Lyapunov exponent.

    Runs n_ics independent trajectories in parallel and returns
    the final LCN for each.

    Parameters
    ----------
    n_steps : collisions per trajectory
    n_ics : number of initial conditions
    alpha_param : spin coupling
    u_max_frac : fraction of max allowed |u| to sample from (0 to 1)

    Returns
    -------
    final_lcns : array of shape (n_ics,) with final LCN per trajectory
    """
    final_lcns = np.empty(n_ics)

    for k in prange(n_ics):
        # Generate random IC based on geometry
        if geo_type == CIRCLE:
            x0, y0 = _random_ic_circle()
        elif geo_type == RECTANGLE:
            x0, y0 = _random_ic_rectangle(geo_p1, geo_p2)
        elif geo_type == STADIUM:
            x0, y0 = _random_ic_stadium(geo_p1)
        else:
            x0, y0 = _random_ic_sinai(geo_p1, geo_p2)

        vx0, vy0, u0 = _random_velocity_with_spin(alpha_param, u_max_frac)

        t_cum, lcn = lyapunov(n_steps, x0, y0, vx0, vy0, u0, alpha_param,
                              geo_type, geo_p1, geo_p2, perturb_mag)
        final_lcns[k] = lcn[-1]

    return final_lcns


# ===========================================================================
#  Lyapunov vs alpha parameter sweep
# ===========================================================================

@njit(cache=True)
def lyapunov_alpha_sweep(alpha_values, n_steps, n_ics, geo_type, geo_p1, geo_p2,
                         perturb_mag=1e-7, u_max_frac=0.5):
    """Sweep alpha and compute ensemble-averaged LCN at each value.

    Returns
    -------
    means : array of shape (len(alpha_values),)
    stds : array of shape (len(alpha_values),)
    """
    n_alpha = len(alpha_values)
    means = np.empty(n_alpha)
    stds = np.empty(n_alpha)

    for j in range(n_alpha):
        a = alpha_values[j]
        lcns = np.empty(n_ics)
        for k in range(n_ics):
            if geo_type == CIRCLE:
                x0, y0 = _random_ic_circle()
            elif geo_type == RECTANGLE:
                x0, y0 = _random_ic_rectangle(geo_p1, geo_p2)
            elif geo_type == STADIUM:
                x0, y0 = _random_ic_stadium(geo_p1)
            else:
                x0, y0 = _random_ic_sinai(geo_p1, geo_p2)

            vx0, vy0, u0 = _random_velocity_with_spin(a, u_max_frac)
            t_cum, lcn = lyapunov(n_steps, x0, y0, vx0, vy0, u0, a,
                                  geo_type, geo_p1, geo_p2, perturb_mag)
            lcns[k] = lcn[-1]

        means[j] = np.mean(lcns)
        stds[j] = np.std(lcns)

    return means, stds


# ===========================================================================
#  Phase space separation analysis
# ===========================================================================

@njit(cache=True)
def phase_space_separation(n_collisions, n_ics, alpha_param,
                           geo_type, geo_p1, geo_p2, perturb_mag=1e-7):
    """Track ln(delta_n / delta_0) averaged over n_ics initial conditions.

    Returns
    -------
    mean_log_sep : array of shape (n_collisions,) — average ln(sep/sep0)
    """
    log_seps = np.zeros(n_collisions)

    for k in range(n_ics):
        if geo_type == CIRCLE:
            x0, y0 = _random_ic_circle()
        elif geo_type == RECTANGLE:
            x0, y0 = _random_ic_rectangle(geo_p1, geo_p2)
        elif geo_type == STADIUM:
            x0, y0 = _random_ic_stadium(geo_p1)
        else:
            x0, y0 = _random_ic_sinai(geo_p1, geo_p2)

        vx0, vy0, u0 = _random_velocity_with_spin(alpha_param, 0.5)

        # Perturbed IC
        dp = np.empty(5)
        for j in range(5):
            dp[j] = np.random.randn()
        norm = math.sqrt(dp[0]**2 + dp[1]**2 + dp[2]**2 + dp[3]**2 + dp[4]**2)
        for j in range(5):
            dp[j] *= perturb_mag / norm

        rx = x0; ry = y0; rvx = vx0; rvy = vy0; ru = u0
        px = x0 + dp[0]; py = y0 + dp[1]
        pvx = vx0 + dp[2]; pvy = vy0 + dp[3]; pu = u0 + dp[4]

        for i in range(n_collisions):
            rx, ry, rvx, rvy, ru, _ = _step_one_collision(
                rx, ry, rvx, rvy, ru, alpha_param, geo_type, geo_p1, geo_p2)
            px, py, pvx, pvy, pu, _ = _step_one_collision(
                px, py, pvx, pvy, pu, alpha_param, geo_type, geo_p1, geo_p2)

            sep = math.sqrt((rx - px)**2 + (ry - py)**2 +
                            (rvx - pvx)**2 + (rvy - pvy)**2 +
                            (ru - pu)**2)
            if sep < 1e-30:
                sep = 1e-30
            log_seps[i] += math.log(sep / perturb_mag)

    # Average over ICs
    for i in range(n_collisions):
        log_seps[i] /= n_ics

    return log_seps


# ===========================================================================
#  Poincaré section data
# ===========================================================================

@njit(cache=True)
def poincare_section(n_collisions, x0, y0, vx0, vy0, u0, alpha_param,
                     geo_type, geo_p1, geo_p2):
    """Compute Birkhoff coordinates at each collision.

    Returns
    -------
    s_vals : arc-length position along boundary
    vpar_vals : tangential velocity component at collision
    u_vals : spin at collision
    wall_ids : which surface was hit
    """
    s_vals = np.empty(n_collisions)
    vpar_vals = np.empty(n_collisions)
    u_vals = np.empty(n_collisions)
    wall_ids = np.empty(n_collisions, dtype=np.int32)

    rx = x0; ry = y0; rvx = vx0; rvy = vy0; ru = u0

    for i in range(n_collisions):
        xn, yn, dt, tx, ty, nx, ny, w = _next_collision(
            rx, ry, rvx, rvy, geo_type, geo_p1, geo_p2)
        vxn, vyn, un = _reflect(rvx, rvy, ru, alpha_param, tx, ty, nx, ny)

        # Tangential velocity of incoming ray (before reflection)
        vpar = rvx * tx + rvy * ty

        # Arc-length coordinate depends on geometry
        # Convention: s increases counterclockwise (or continuously along
        # the boundary) so that the Poincaré section is well-defined.
        if geo_type == CIRCLE:
            s = math.atan2(yn, xn)  # ∈ (-π, π]
            if s < 0.0:
                s += 2.0 * math.pi  # map to [0, 2π)
        elif geo_type == RECTANGLE:
            # Perimeter = 4L + 4H, going CCW from bottom-left corner
            L = geo_p1; H = geo_p2
            if w == WALL_BOTTOM:
                s = (xn + L)                         # [0, 2L]
            elif w == WALL_RIGHT:
                s = 2.0 * L + (yn + H)               # [2L, 2L+2H]
            elif w == WALL_TOP:
                s = 2.0 * L + 2.0 * H + (L - xn)    # [2L+2H, 4L+2H]
            else:  # LEFT
                s = 4.0 * L + 2.0 * H + (H - yn)    # [4L+2H, 4L+4H]
        elif geo_type == STADIUM:
            # Boundary: top flat → right cap → bottom flat → left cap
            # Total perimeter = 4L + 2π
            L_s = geo_p1
            if w == WALL_TOP:
                # Top wall: y=1, x from -L to L. s from 0 to 2L.
                s = xn + L_s
            elif w == WALL_CAP_RIGHT:
                # Right cap: center (L,0), radius 1. Angle from π/2 to -π/2.
                angle = math.atan2(yn, xn - L_s)  # ∈ (-π, π]
                s = 2.0 * L_s + (math.pi / 2.0 - angle)  # [2L, 2L+π]
            elif w == WALL_BOTTOM:
                # Bottom wall: y=-1, x from L to -L. s from 2L+π to 4L+π.
                s = 2.0 * L_s + math.pi + (L_s - xn)
            else:  # CAP_LEFT
                # Left cap: center (-L,0), radius 1.
                # Arc goes clockwise (in angle) from -π/2 through ±π to π/2.
                angle = math.atan2(yn, xn + L_s)  # ∈ (-π, π]
                s = 4.0 * L_s + math.pi + ((-math.pi / 2.0 - angle) % (2.0 * math.pi))
                # Total: [4L+π, 4L+2π]
        else:  # SINAI
            # Square boundary: perimeter = 8L, going CCW from (-L,-L)
            # Central circle: separate arc-length s = R * θ
            L_sq = geo_p1; R = geo_p2
            if w == WALL_CIRCLE:
                angle = math.atan2(yn, xn)
                if angle < 0.0:
                    angle += 2.0 * math.pi
                s = R * angle  # [0, 2πR]
            elif w == WALL_BOTTOM:
                s = (xn + L_sq)                              # [0, 2L]
            elif w == WALL_RIGHT:
                s = 2.0 * L_sq + (yn + L_sq)                # [2L, 4L]
            elif w == WALL_TOP:
                s = 4.0 * L_sq + (L_sq - xn)                # [4L, 6L]
            else:  # LEFT
                s = 6.0 * L_sq + (L_sq - yn)                # [6L, 8L]

        s_vals[i] = s
        vpar_vals[i] = vpar
        u_vals[i] = un
        wall_ids[i] = w

        rx = xn; ry = yn; rvx = vxn; rvy = vyn; ru = un

    return s_vals, vpar_vals, u_vals, wall_ids


# ===========================================================================
#  Energy diagnostic
# ===========================================================================

@njit(cache=True)
def check_energy(vxs, vys, us, alpha_param):
    """Compute kinetic energy at each collision: E = 0.5*v^2 + 0.5*alpha*u^2.

    Returns array of energies. Should be constant if physics is correct.
    """
    N = len(vxs)
    E = np.empty(N)
    for i in range(N):
        E[i] = 0.5 * (vxs[i]**2 + vys[i]**2) + 0.5 * alpha_param * us[i]**2
    return E


# ===========================================================================
#  Conserved quantity diagnostic
# ===========================================================================

@njit(cache=True)
def check_conserved_quantity(xs, ys, vxs, vys, us, walls, alpha_param,
                             geo_type, geo_p1, geo_p2):
    """Compute Q = v_parallel - alpha * u at each collision.

    v_parallel is the tangential velocity component, computed using the wall's
    tangent vector at the collision point. Q is conserved through each
    individual flat-wall collision but jumps at curved walls.

    Returns
    -------
    Q : array of shape (N,) — the conserved quantity at each collision
    is_flat : array of shape (N,) — 1.0 if flat wall, 0.0 if curved
    """
    N = len(xs)
    Q = np.empty(N)
    is_flat = np.empty(N)

    for i in range(N):
        w = walls[i]
        vx = vxs[i]
        vy = vys[i]
        u = us[i]

        # Determine tangent vector based on wall type
        if w == WALL_TOP:
            # T = (-1, 0)
            vpar = -vx
            is_flat[i] = 1.0
        elif w == WALL_BOTTOM:
            # T = (1, 0)
            vpar = vx
            is_flat[i] = 1.0
        elif w == WALL_LEFT:
            # T = (0, -1)
            vpar = -vy
            is_flat[i] = 1.0
        elif w == WALL_RIGHT:
            # T = (0, 1)
            vpar = vy
            is_flat[i] = 1.0
        elif w == WALL_CIRCLE:
            if geo_type == SINAI:
                # Sinai obstacle: T = (y/R, -x/R)
                R = geo_p2
                vpar = (vx * ys[i] - vy * xs[i]) / R
            else:
                # Circle billiard: T = (-y, x) on unit circle
                vpar = -vx * ys[i] + vy * xs[i]
            is_flat[i] = 0.0
        elif w == WALL_CAP_LEFT:
            # Stadium left cap centered at (-L, 0): T = (-(y), (x+L))
            L = geo_p1
            dx = xs[i] + L
            dy = ys[i]
            vpar = -vx * dy + vy * dx
            is_flat[i] = 0.0
        elif w == WALL_CAP_RIGHT:
            # Stadium right cap centered at (L, 0): T = (-(y), (x-L))
            L = geo_p1
            dx = xs[i] - L
            dy = ys[i]
            vpar = -vx * dy + vy * dx
            is_flat[i] = 0.0
        else:
            vpar = 0.0
            is_flat[i] = 0.0

        Q[i] = vpar - alpha_param * u

    return Q, is_flat


# ===========================================================================
#  Convenience wrappers (non-JIT, for easy calling)
# ===========================================================================

def simulate_circle(n_collisions, x0, y0, theta0, u0=0.0, alpha=0.0):
    """Simulate on unit circle. theta0 = initial velocity angle."""
    vx0 = math.cos(theta0)
    vy0 = math.sin(theta0)
    return simulate(n_collisions, x0, y0, vx0, vy0, u0, alpha,
                    CIRCLE, 0.0, 0.0)


def simulate_rectangle(n_collisions, x0, y0, theta0, u0=0.0, alpha=0.0,
                        L=2.0, H=1.0):
    """Simulate on rectangle [-L,L] x [-H,H]."""
    vx0 = math.cos(theta0)
    vy0 = math.sin(theta0)
    return simulate(n_collisions, x0, y0, vx0, vy0, u0, alpha,
                    RECTANGLE, L, H)


def simulate_stadium(n_collisions, x0, y0, theta0, u0=0.0, alpha=0.0, L=1.0):
    """Simulate on Bunimovich stadium with flat section half-length L."""
    vx0 = math.cos(theta0)
    vy0 = math.sin(theta0)
    return simulate(n_collisions, x0, y0, vx0, vy0, u0, alpha,
                    STADIUM, L, 0.0)


def simulate_sinai(n_collisions, x0, y0, theta0, u0=0.0, alpha=0.0,
                    L=2.0, R=1.0):
    """Simulate on Sinai billiard: square [-L,L]^2 with obstacle radius R."""
    vx0 = math.cos(theta0)
    vy0 = math.sin(theta0)
    return simulate(n_collisions, x0, y0, vx0, vy0, u0, alpha,
                    SINAI, L, R)


# ===========================================================================
#  Plotting utilities
# ===========================================================================

def plot_trajectory(xs, ys, geo_type, geo_p1=0.0, geo_p2=0.0,
                    ax=None, lw=0.3, color='black', title=None):
    """Plot a billiard trajectory with boundary."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=150)

    ax.plot(xs, ys, c=color, linewidth=lw)

    theta = np.linspace(0, 2 * np.pi, 300)

    if geo_type == CIRCLE:
        ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1)
    elif geo_type == RECTANGLE:
        L, H = geo_p1, geo_p2
        rect = plt.Rectangle((-L, -H), 2*L, 2*H, fill=False, ec='k', lw=1)
        ax.add_patch(rect)
    elif geo_type == STADIUM:
        L = geo_p1
        s = np.linspace(-1, 1, 200)
        ax.plot(L * s, np.ones_like(s), 'k-', lw=1)
        ax.plot(L * s, -np.ones_like(s), 'k-', lw=1)
        ax.plot(-np.cos(np.pi * s / 2) - L, np.sin(np.pi * s / 2), 'k-', lw=1)
        ax.plot(np.cos(np.pi * s / 2) + L, -np.sin(np.pi * s / 2), 'k-', lw=1)
    elif geo_type == SINAI:
        L, R = geo_p1, geo_p2
        rect = plt.Rectangle((-L, -L), 2*L, 2*L, fill=False, ec='k', lw=1)
        ax.add_patch(rect)
        ax.plot(R * np.cos(theta), R * np.sin(theta), 'k-', lw=1)

    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title)
    return ax


def plot_lcn(t_cum, lcn, ax=None, label=None, color=None):
    """Plot LCN vs time on log-log scale."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5), dpi=150)

    mask = (t_cum > 0) & (lcn > 0)
    if color is not None:
        ax.plot(np.log10(t_cum[mask]), np.log10(lcn[mask]),
                label=label, c=color, lw=0.8)
    else:
        ax.plot(np.log10(t_cum[mask]), np.log10(lcn[mask]),
                label=label, lw=0.8)

    ax.set_xlabel(r'$\log_{10}(t)$')
    ax.set_ylabel(r'$\log_{10}(\mathrm{LCN})$')
    if label:
        ax.legend()
    return ax


def plot_poincare(s_vals, vpar_vals, u_vals=None, ax=None,
                  color_by_u=False, s=0.3, title=None):
    """Plot Poincaré section (s, v_parallel) or colored by u."""
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=150)

    if color_by_u and u_vals is not None:
        sc = ax.scatter(s_vals, vpar_vals, c=u_vals, s=s, cmap='coolwarm',
                        edgecolors='none')
        plt.colorbar(sc, ax=ax, label='u (spin)')
    else:
        ax.scatter(s_vals, vpar_vals, s=s, c='black', edgecolors='none')

    ax.set_xlabel('s (arc length)')
    ax.set_ylabel(r'$v_{\parallel}$ (tangential velocity)')
    if title:
        ax.set_title(title)
    return ax


# ===========================================================================
#  Warm up JIT on import (optional, call explicitly)
# ===========================================================================

def warmup():
    """Force JIT compilation of all core functions."""
    for geo, p1, p2, x0, y0 in [
        (CIRCLE, 0.0, 0.0, 0.5, 0.1),
        (RECTANGLE, 2.0, 1.0, 0.5, 0.1),
        (STADIUM, 1.0, 0.0, 0.5, 0.1),
        (SINAI, 2.0, 1.0, 1.5, 1.5),
    ]:
        simulate(5, x0, y0, 0.7, 0.7, 0.0, 0.1, geo, p1, p2)
        lyapunov(5, x0, y0, 0.7, 0.7, 0.0, 0.1, geo, p1, p2)
    print("JIT warmup complete.")
