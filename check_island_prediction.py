"""Test the elliptic-island prediction against the real stadium dynamics.

Claim (response letter, R1.8): in the rescaled variable u~ = sqrt(alpha)*u the
two-collision map (floor then ceiling) on (vx, u~) is an exact rotation by
theta(alpha) = 4*arctan(sqrt(alpha)), so for alpha > 0 the marginal
bouncing-ball family becomes a genuine stability island: vx librates instead of
drifting, and the orbit stays on the flat walls forever unless its x-excursion
reaches the caps.

Because the flat-wall collision law is LINEAR in (vx, u~) and |vy| is invariant
along a flat-bounce sequence (energy: vx^2 + vy^2 + u~^2 = 1), the prediction
machinery is pure 2x2 linear algebra -- no billiard simulation:

    z_{k+1} = R_b R_t z_k,   x accrues tau*vx per flight,  tau = 2/|vy|.

Four tests:
  1. ROTATION NUMBER: measured per-period rotation angle of (vx, u~) at
     successive floor collisions in the REAL stadium map vs 4*arctan(sqrt(alpha)).
  2. TRAPPING BOUNDARY: in the (x0, A) plane (A = |z0|), measured escape
     (real dynamics, does the orbit ever hit a cap within N bounces) vs the
     linear-model prediction (max x-excursion crosses |x| = L).
  3. FTLE: initial conditions inside the predicted island must have lambda ~ 0;
     matched ICs outside must be chaotic.
  4. REGULAR FRACTION: Monte Carlo over the paper's IC sampler, predicting
     trapped/escaped per IC from the linear model seeded at the first flat-wall
     collision, compared with the paper's FTLE regular fraction
     (1 - fractions in experiment_results/chaotic_fraction_stadium.npz).
     Other island families would make the measured regular fraction exceed the
     prediction, so agreement from below is the expected signature.
"""

import math
import os

import numpy as np
from numba import njit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spinning_billiards import (
    STADIUM, WALL_TOP, WALL_BOTTOM, WALL_CAP_LEFT, WALL_CAP_RIGHT,
    _next_collision, _reflect, lyapunov,
    _random_ic_stadium,
)

L_FLAT = 1.0          # stadium geo_p1: flats span |x| <= 1, walls y = +-1


def reflection_matrices(alpha):
    """R_bottom, R_top acting on (vx, u~), rescaled u~ = sqrt(alpha)*u."""
    s = math.sqrt(alpha) if alpha > 0 else 0.0
    M = np.array([[(1 - alpha) / (1 + alpha), -2 * alpha / (1 + alpha)],
                  [-2 / (1 + alpha), -(1 - alpha) / (1 + alpha)]])
    S = np.diag([1.0, s if s > 0 else 1.0])
    Sinv = np.diag([1.0, 1.0 / s if s > 0 else 1.0])
    Rb = S @ M @ Sinv
    F = np.diag([-1.0, 1.0])
    Rt = F @ Rb @ F
    return Rb, Rt


def predicted_excursion(alpha, z0, n_periods=4096):
    """Max |x - x0| over a flat-bounce sequence, from the linear model only."""
    Rb, Rt = reflection_matrices(alpha)
    A = math.hypot(z0[0], z0[1])
    vy = math.sqrt(max(1.0 - A * A, 1e-15))
    tau = 2.0 / vy
    z = np.array(z0, float)
    x = 0.0
    worst = 0.0
    for _ in range(n_periods):
        x += tau * z[0]                      # flight floor -> ceiling
        worst = max(worst, abs(x))
        z = Rt @ z
        x += tau * z[0]                      # flight ceiling -> floor
        worst = max(worst, abs(x))
        z = Rb @ z
    return worst


@njit(cache=True)
def measured_escape(alpha, x0, vx0, ut0, n_bounce):
    """Real stadium dynamics: 0 if trapped for n_bounce collisions, else 1.
    Start just above the floor at (x0, -1+eps) with post-collision velocity."""
    s = math.sqrt(alpha) if alpha > 0 else 1.0
    u0 = ut0 / s if alpha > 0 else 0.0
    vy0 = math.sqrt(max(1.0 - vx0 * vx0 - ut0 * ut0, 1e-15))
    x = x0; y = -1.0 + 1e-9
    vx = vx0; vy = vy0; u = u0
    for i in range(n_bounce):
        xn, yn, dt, tx, ty, nx, ny, w = _next_collision(x, y, vx, vy,
                                                        STADIUM, L_FLAT, 0.0)
        if w == WALL_CAP_LEFT or w == WALL_CAP_RIGHT:
            return 1
        vx, vy, u = _reflect(vx, vy, u, alpha, tx, ty, nx, ny)
        x, y = xn, yn
    return 0


@njit(cache=True)
def measured_rotation(alpha, x0, A, phase, n_periods):
    """Angles of (vx, u~) at successive floor collisions in the real map."""
    s = math.sqrt(alpha)
    vx = A * math.cos(phase); ut = A * math.sin(phase)
    u = ut / s
    vy = math.sqrt(max(1.0 - vx * vx - ut * ut, 1e-15))
    x = x0; y = -1.0 + 1e-9
    angles = np.empty(n_periods)
    k = 0
    guard = 0
    while k < n_periods and guard < 10 * n_periods + 100:
        guard += 1
        xn, yn, dt, tx, ty, nx, ny, w = _next_collision(x, y, vx, vy,
                                                        STADIUM, L_FLAT, 0.0)
        if w == WALL_CAP_LEFT or w == WALL_CAP_RIGHT:
            return angles[:k]
        vx, vy, u = _reflect(vx, vy, u, alpha, tx, ty, nx, ny)
        x, y = xn, yn
        if w == WALL_BOTTOM:
            angles[k] = math.atan2(s * u, vx)
            k += 1
    return angles[:k]


def test_rotation_number(alphas):
    print("=== 1. rotation number: measured vs 4*arctan(sqrt(alpha)) ===")
    ok = True
    for a in alphas:
        ang = measured_rotation(a, 0.15, 0.05, 0.4, 400)
        if len(ang) < 10:
            print(f"  alpha={a:.2f}: escaped too soon ({len(ang)} periods)")
            ok = False
            continue
        d = np.unwrap(np.diff(ang))
        pred = 4 * math.atan(math.sqrt(a))
        meas = float(np.abs(d).mean())
        # compare modulo 2*pi (direction irrelevant)
        err = min(abs(meas - pred), abs(2 * math.pi - meas - pred))
        print(f"  alpha={a:.2f}: measured {meas:.6f}  predicted {pred:.6f} "
              f" |err|={err:.2e}  spread={np.abs(d).std():.2e}")
        ok &= err < 1e-6
    print(f"  -> {'EXACT' if ok else 'MISMATCH'}")
    return ok


def test_boundary(alpha, n_grid=81, n_bounce=20000, phase=0.4):
    print(f"\n=== 2. trapping boundary at alpha={alpha} "
          f"(phase={phase}, {n_bounce} bounces) ===")
    x0s = np.linspace(0.0, 0.98, n_grid)
    As = np.linspace(0.002, 0.5, n_grid)
    measured = np.zeros((n_grid, n_grid), dtype=np.int8)
    predicted = np.zeros_like(measured)
    for i, x0 in enumerate(x0s):
        for j, A in enumerate(As):
            z0 = (A * math.cos(phase), A * math.sin(phase))
            exc = predicted_excursion(alpha, z0, 2048)
            predicted[i, j] = 1 if abs(x0) + exc > L_FLAT else 0
            measured[i, j] = measured_escape(alpha, x0,
                                             A * math.cos(phase),
                                             A * math.sin(phase), n_bounce)
    agree = (measured == predicted).mean()
    print(f"  cell-wise agreement: {agree:.4f}  "
          f"(measured trapped {1-measured.mean():.3f}, "
          f"predicted trapped {1-predicted.mean():.3f})")
    return x0s, As, measured, predicted, agree


def test_ftle(alpha, n_steps=40000):
    print(f"\n=== 3. FTLE inside vs outside the island (alpha={alpha}) ===")
    s = math.sqrt(alpha)
    cases = {
        "inside  (x0=0.1, A=0.05)": (0.1, 0.05),
        "inside  (x0=0.3, A=0.10)": (0.3, 0.10),
        "outside (x0=0.9, A=0.45)": (0.9, 0.45),
        "outside (x0=0.0, A=0.80)": (0.0, 0.80),
    }
    for name, (x0, A) in cases.items():
        vx = A * math.cos(0.4); ut = A * math.sin(0.4); u = ut / s
        vy = math.sqrt(max(1 - vx * vx - ut * ut, 1e-12))
        _, lcn = lyapunov(n_steps, x0, -1 + 1e-9, vx, vy, u, alpha,
                          STADIUM, L_FLAT, 0.0, 1e-7)
        print(f"  {name}: lambda = {lcn[-1]:+.5f}")


@njit(cache=True)
def _first_flat_hit(x, y, vx, vy, u, alpha, max_hops=50):
    """Advance until the first TOP/BOTTOM collision; return post-collision
    state there, or a cap flag if a cap comes first."""
    for i in range(max_hops):
        xn, yn, dt, tx, ty, nx, ny, w = _next_collision(x, y, vx, vy,
                                                        STADIUM, L_FLAT, 0.0)
        vx, vy, u = _reflect(vx, vy, u, alpha, tx, ty, nx, ny)
        x, y = xn, yn
        if w == WALL_CAP_LEFT or w == WALL_CAP_RIGHT:
            return x, y, vx, vy, u, 1, w
        if w == WALL_TOP or w == WALL_BOTTOM:
            return x, y, vx, vy, u, 0, w
    return x, y, vx, vy, u, 1, -1


from numba import prange


@njit(cache=True)
def _excursion_nb(alpha, vx0, ut0, n_periods):
    """Max |x - x0| of the flat-bounce linear model; pure 2x2 arithmetic."""
    sa = math.sqrt(alpha)
    d = 1.0 + alpha
    # R_bottom on (vx, ut); R_top = F R_b F with F = diag(-1, 1)
    b11 = (1.0 - alpha) / d; b12 = -2.0 * sa / d
    b21 = -2.0 * sa / d;     b22 = -(1.0 - alpha) / d
    A2 = vx0 * vx0 + ut0 * ut0
    vy = math.sqrt(max(1.0 - A2, 1e-15))
    tau = 2.0 / vy
    vx = vx0; ut = ut0
    x = 0.0; worst = 0.0
    for _ in range(n_periods):
        x += tau * vx
        if abs(x) > worst:
            worst = abs(x)
        # top reflection: F R_b F
        nvx = b11 * vx - b12 * ut
        nut = -b21 * vx + b22 * ut
        vx = nvx; ut = nut
        x += tau * vx
        if abs(x) > worst:
            worst = abs(x)
        # bottom reflection
        nvx = b11 * vx + b12 * ut
        nut = b21 * vx + b22 * ut
        vx = nvx; ut = nut
    return worst


@njit(parallel=True, cache=True)
def _trapped_fraction_nb(alpha, n_mc, seed, n_periods):
    np.random.seed(seed)
    # draw all ICs serially (numba RNG), classify in parallel
    ics = np.empty((n_mc, 5))
    u_max = 1.0 / math.sqrt(alpha)
    for k in range(n_mc):
        x0, y0 = _random_ic_stadium(L_FLAT)
        th = 2.0 * math.pi * np.random.random()
        u = (2.0 * np.random.random() - 1.0) * 0.5 * u_max
        v = math.sqrt(max(1.0 - alpha * u * u, 0.01))
        ics[k, 0] = x0; ics[k, 1] = y0
        ics[k, 2] = v * math.cos(th); ics[k, 3] = v * math.sin(th)
        ics[k, 4] = u
    sa = math.sqrt(alpha)
    n_trap = 0
    for k in prange(n_mc):
        x, y, vx, vy, u, cap, w = _first_flat_hit(
            ics[k, 0], ics[k, 1], ics[k, 2], ics[k, 3], ics[k, 4], alpha)
        if cap:
            continue
        A = math.hypot(vx, sa * u)
        if A >= 0.999:
            continue
        exc = _excursion_nb(alpha, vx, sa * u, n_periods)
        if abs(x) + exc <= L_FLAT:
            n_trap += 1
    return n_trap / n_mc


def test_regular_fraction(alphas, n_mc=100_000, seed=3):
    """Predicted trapped fraction under the paper's IC sampler vs the paper's
    FTLE regular fraction, both at high statistics: the prediction from a
    10^5-sample Monte Carlo per alpha (numba), the measurement from the
    10^5-IC FTLE ensemble in unified_billiards_data.npz (50 alpha values,
    threshold 0.01 as in the chaotic-fraction analysis)."""
    print("\n=== 4. regular fraction: linear-island prediction vs FTLE data ===")
    pred = np.zeros(len(alphas))
    for ia, a in enumerate(alphas):
        pred[ia] = _trapped_fraction_nb(a, n_mc, seed + ia, 1024)
        print(f"  alpha={a:.2f}: predicted trapped fraction = {pred[ia]:.4f} "
              f"(+/- {math.sqrt(pred[ia]*(1-pred[ia])/n_mc):.4f})")

    du = np.load("experiment_results/unified_billiards_data.npz")
    a_data = du["alpha_values"]
    ftle = du["Stadium_ftle"]
    finite = np.isfinite(ftle)
    reg = np.array([np.mean(ftle[j][finite[j]] <= 0.01) for j in range(len(a_data))])
    return pred, a_data, reg


def main():
    alphas_rot = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    test_rotation_number(alphas_rot)

    x0s, As, meas, predb, agree = test_boundary(0.5)

    test_ftle(0.5)

    alphas_mc = list(np.round(np.linspace(0.05, 1.0, 20), 3))
    pred, a_data, reg_data = test_regular_fraction(alphas_mc)

    # ---- figure ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    ax = axes[0]
    ax.imshow(meas.T, origin="lower", aspect="auto",
              extent=[x0s[0], x0s[-1], As[0], As[-1]],
              cmap="RdYlBu_r", alpha=0.75, vmin=0, vmax=1)
    ax.contour(x0s, As, predb.T, levels=[0.5], colors="k", linewidths=1.6)
    ax.set_xlabel(r"floor position $x_0$")
    ax.set_ylabel(r"velocity-plane amplitude $A=|(v_x,\tilde u)|$")
    ax.set_title(r"Stadium, $\alpha=0.5$: escape (red) vs trapped (blue);"
                 "\nblack: zero-parameter linear-model boundary")

    ax = axes[1]
    ax.plot(a_data, reg_data, "o-", ms=3, lw=1.2, color="#c1272d",
            label="FTLE regular fraction (paper data)")
    ax.plot(alphas_mc, pred, "s-", ms=4, lw=1.2, color="#0057b7",
            label="predicted trapped fraction (linear island)")
    ax.set_xlabel(r"spin coupling $\alpha$")
    ax.set_ylabel("regular fraction")
    ax.legend(fontsize=8)
    ax.set_title("Bouncing-ball island vs measured regular fraction")

    fig.tight_layout()
    os.makedirs("plots_v2", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"plots_v2/fig_island_prediction.{ext}", dpi=180)
    print("\nwrote plots_v2/fig_island_prediction.pdf")

    np.savez("experiment_results/island_prediction.npz",
             alpha_mc=np.array(alphas_mc), predicted_regular=pred,
             alpha_data=a_data, regular_data=reg_data,
             boundary_x0=x0s, boundary_A=As,
             boundary_measured=meas, boundary_predicted=predb,
             boundary_agreement=agree)


if __name__ == "__main__":
    main()
