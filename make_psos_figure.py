"""Three-dimensional Poincare surface of section for the spinning stadium.

Referee 1 (CHA26-AR-00869) is right that a 3D section in (s, v_par, u) is
perfectly well defined even though no 2D invariant surface exists. Left panel:
the section for the stadium at alpha = 0.5, several orbits started inside the
predicted bouncing-ball island (solid torus around v_x = u = 0 over the flat
sections) and a few chaotic orbits filling the sea around it. Right panel:
the floor-collision points of the same section in the rescaled velocity plane
(v_par, u~): every trapped orbit lies on an exact circle (the invariant of the
two-bounce rotation), and the dashed circle is the largest amplitude for which
the linear model permits trapping at all -- computed from the model, not fit.
Every plotted island orbit is verified trapped in the real dynamics for 2e4
bounces before it is drawn.

Arclength convention (stadium, flats |x| <= L at y = +-1, unit-radius caps):
  s = 0 at (-L, -1), increasing along the floor (+x), around the right cap,
  along the ceiling (-x), and around the left cap; total length 4L + 2*pi.
"""

import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spinning_billiards import (
    STADIUM, WALL_TOP, WALL_BOTTOM, WALL_CAP_LEFT, WALL_CAP_RIGHT,
    _next_collision, _reflect,
)
from check_island_prediction import _excursion_nb, measured_escape

L = 1.0
ALPHA = 0.5
SA = math.sqrt(ALPHA)


def arclength(x, y, wall):
    if wall == WALL_BOTTOM:
        return (x + L)
    if wall == WALL_CAP_RIGHT:
        phi = math.atan2(y, x - L)            # -pi/2 .. pi/2
        return 2 * L + (phi + math.pi / 2)
    if wall == WALL_TOP:
        return 2 * L + math.pi + (L - x)
    if wall == WALL_CAP_LEFT:
        phi = math.atan2(y, x + L)            # pi/2 .. 3pi/2 (wraps)
        if phi < 0:
            phi += 2 * math.pi
        return 4 * L + math.pi + (phi - math.pi / 2)
    return np.nan


def orbit_section(x, y, vx, vy, u, n_coll):
    S = np.empty(n_coll); VP = np.empty(n_coll); U = np.empty(n_coll)
    W = np.empty(n_coll, dtype=np.int64)
    for i in range(n_coll):
        xn, yn, dt, tx, ty, nx, ny, w = _next_collision(x, y, vx, vy,
                                                        STADIUM, L, 0.0)
        vx, vy, u = _reflect(vx, vy, u, ALPHA, tx, ty, nx, ny)
        x, y = xn, yn
        S[i] = arclength(x, y, w)
        VP[i] = vx * tx + vy * ty
        U[i] = u
        W[i] = w
    return S, VP, U, W


def island_ic(x0, A, phase):
    vx = A * math.cos(phase)
    u = A * math.sin(phase) / SA
    vy = math.sqrt(max(1 - vx * vx - ALPHA * u * u, 1e-12))
    return x0, -1.0 + 1e-9, vx, vy, u


def model_max_amplitude(n_A=120, n_phase=64, n_periods=2048):
    """Largest |(v_par, u~)| for which the linear model permits trapping
    (optimal floor position x0 = 0 and optimal phase)."""
    amax = 0.0
    for A in np.linspace(0.01, 0.7, n_A):
        excs = [_excursion_nb(ALPHA, A * math.cos(p), A * math.sin(p),
                              n_periods)
                for p in np.linspace(0, math.pi, n_phase)]
        if min(excs) <= L:
            amax = A
    return amax


def best_phase(A, n_phase=128, n_periods=2048):
    ph = np.linspace(0, math.pi, n_phase)
    excs = [_excursion_nb(ALPHA, A * math.cos(p), A * math.sin(p), n_periods)
            for p in ph]
    return float(ph[int(np.argmin(excs))])


def main():
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.0, "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0,
    })
    fig = plt.figure(figsize=(7.05, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1.0], wspace=0.42)
    ax = fig.add_subplot(gs[0], projection="3d")
    ax2 = fig.add_subplot(gs[1])

    # chaotic sea: three orbits, grey, selected by measured FTLE > 0.05 so
    # that none secretly belongs to a regular family (the alpha = 0.5 stadium
    # is island-rich; randomly drawn ICs land in them surprisingly often).
    from spinning_billiards import lyapunov
    rng = np.random.default_rng(2)
    picked = 0
    while picked < 3:
        th = rng.uniform(0, 2 * math.pi)
        u = rng.uniform(-1.2, 1.2) / SA
        v = math.sqrt(max(1 - ALPHA * u * u, 0.01))
        _, lcn = lyapunov(20000, 0.3, 0.2, v * math.cos(th), v * math.sin(th),
                          u, ALPHA, STADIUM, L, 0.0, 1e-7)
        if lcn[-1] < 0.05:
            continue
        picked += 1
        S, VP, U, W = orbit_section(0.3, 0.2, v * math.cos(th),
                                    v * math.sin(th), u, 6000)
        ax.scatter(S[:4000], VP[:4000], U[:4000], s=0.5, c="0.65",
                   alpha=0.25, linewidths=0)
        m = W == WALL_BOTTOM
        ax2.scatter(VP[m], SA * U[m], s=0.6, c="0.65", alpha=0.3,
                    linewidths=0)

    # island orbits for the 3D panel: blues, verified trapped
    cmap = plt.cm.Blues
    cases3d = [(-0.5, 0.05, 0.4), (-0.5, 0.12, 2.0), (0.0, 0.05, 1.1),
               (0.0, 0.15, 2.6), (0.5, 0.08, 0.4), (0.5, 0.11, 1.7)]
    # additional orbits for the zoom, filling the family out to its boundary:
    # optimal phase at x0 = 0, so the largest permitted amplitudes are reachable
    zoom_extra = [(0.0, A, best_phase(A)) for A in (0.22, 0.30, 0.38, 0.46)]

    zoom_orbits = sorted(cases3d + zoom_extra, key=lambda c: c[1])
    for k, (x0, A, ph) in enumerate(zoom_orbits):
        esc = measured_escape(ALPHA, x0, A * math.cos(ph), A * math.sin(ph),
                              20000)
        if esc:
            print(f"  WARNING: orbit x0={x0} A={A} phase={ph:.3f} escapes; "
                  "skipped")
            continue
        S, VP, U, W = orbit_section(*island_ic(x0, A, ph), 1200)
        col = cmap(0.35 + 0.6 * k / max(1, len(zoom_orbits) - 1))
        if (x0, A, ph) in cases3d:
            ax.scatter(S, VP, U, s=1.2, color=col, alpha=0.8, linewidths=0)
        m = W == WALL_BOTTOM
        ax2.scatter(VP[m], SA * U[m], s=1.5, color=col, alpha=0.9,
                    linewidths=0)

    ax.set_xlabel(r"boundary arclength $s$")
    ax.set_ylabel(r"$v_\parallel$")
    ax.set_zlabel(r"$u$")
    ax.set_yticks([-1, 0, 1])
    ax.view_init(elev=18, azim=-64)

    # model boundary: largest amplitude any trapped bouncing orbit can have
    amax = model_max_amplitude()
    print(f"  model maximal trapped amplitude A_max = {amax:.4f}")
    th = np.linspace(0, 2 * math.pi, 256)
    ax2.plot(amax * np.cos(th), amax * np.sin(th), "k--", lw=1.0)
    ax2.set_xlim(-0.62, 0.62)
    ax2.set_ylim(-0.62, 0.62)
    ax2.set_aspect("equal")
    ax2.set_xlabel(r"$v_\parallel$")
    ax2.set_ylabel(r"$\tilde u$")

    fig.tight_layout(w_pad=2.0)
    for ext in ("pdf", "png"):
        fig.savefig(f"plots_v2/fig_psos_stadium.{ext}", dpi=180)
    print("wrote plots_v2/fig_psos_stadium.pdf")


if __name__ == "__main__":
    main()
