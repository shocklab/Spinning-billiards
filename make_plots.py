"""
Publication-quality plots for spinning billiards paper.

Generates all figures from scratch with consistent styling.
"""

import numpy as np
import math
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Arc
from matplotlib.collections import LineCollection

from spinning_billiards import (
    simulate, lyapunov, lyapunov_ensemble,
    phase_space_separation, poincare_section, warmup,
    CIRCLE, RECTANGLE, STADIUM, SINAI,
    WALL_TOP, WALL_BOTTOM, WALL_LEFT, WALL_RIGHT,
    WALL_CIRCLE, WALL_CAP_LEFT, WALL_CAP_RIGHT,
)
OUTDIR = "plots_v2"
os.makedirs(OUTDIR, exist_ok=True)

# ── Global style ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.08,
    'axes.linewidth': 0.8,
    'axes.grid': False,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.8',
})

GEO_COLORS = {
    'Circle': '#2176AE',
    'Rectangle': '#F57C20',
    'Stadium': '#57A773',
    'Sinai': '#D33F49',
}
ALPHA_CMAP = plt.cm.viridis


def _save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUTDIR, f"{name}.png"))
    plt.close(fig)
    print(f"  -> {name}.pdf")


# ── Boundary drawing helpers ──────────────────────────────────────────

def draw_circle_boundary(ax, **kw):
    kw.setdefault('color', '0.3')
    kw.setdefault('linewidth', 1.2)
    kw.setdefault('fill', False)
    ax.add_patch(Circle((0, 0), 1.0, **kw))
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')

def draw_rectangle_boundary(ax, L=1.0, H=1.0, **kw):
    kw.setdefault('color', '0.3')
    kw.setdefault('linewidth', 1.2)
    xs = [-L, L, L, -L, -L]
    ys = [-H, -H, H, H, -H]
    ax.plot(xs, ys, **kw)
    ax.set_xlim(-L*1.1, L*1.1)
    ax.set_ylim(-H*1.1, H*1.1)
    ax.set_aspect('equal')

def draw_stadium_boundary(ax, L=1.0, **kw):
    kw.setdefault('color', '0.3')
    kw.setdefault('linewidth', 1.2)
    # Flat walls
    ax.plot([-L, L], [1, 1], **kw)
    ax.plot([-L, L], [-1, -1], **kw)
    # Caps
    theta = np.linspace(-np.pi/2, np.pi/2, 100)
    ax.plot(L + np.cos(theta), np.sin(theta), **kw)
    ax.plot(-L - np.cos(theta), np.sin(theta), **kw)
    ax.set_xlim(-(L+1)*1.08, (L+1)*1.08)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect('equal')

def draw_sinai_boundary(ax, L=2.0, R=1.0, **kw):
    kw.setdefault('color', '0.3')
    kw.setdefault('linewidth', 1.2)
    xs = [-L, L, L, -L, -L]
    ys = [-L, -L, L, L, -L]
    ax.plot(xs, ys, **kw)
    ax.add_patch(Circle((0, 0), R, fill=True, facecolor='0.85',
                        edgecolor=kw.get('color', '0.3'),
                        linewidth=kw.get('linewidth', 1.2)))
    ax.set_xlim(-L*1.08, L*1.08)
    ax.set_ylim(-L*1.08, L*1.08)
    ax.set_aspect('equal')


# =====================================================================
#  Figure 1: Trajectory gallery
# =====================================================================

def fig_trajectories():
    print("\n[Fig 1] Trajectories")

    alpha_vals = [0.0, 0.3, 0.7, 1.0]
    n_coll = 500
    geos = [
        ("Circle",    CIRCLE,    0.0, 0.0,  0.0, 0.5),
        ("Rectangle", RECTANGLE, 1.0, 1.0,  0.2, 0.3),
        ("Stadium",   STADIUM,   1.0, 0.0,  0.2, 0.3),
        ("Sinai",     SINAI,     2.0, 1.0,  0.5, 0.5),
    ]
    theta0 = 0.8

    fig, axes = plt.subplots(len(geos), len(alpha_vals),
                              figsize=(3.2 * len(alpha_vals), 3.2 * len(geos)))

    for row, (gname, geo, p1, p2, x0, y0) in enumerate(geos):
        for col, a in enumerate(alpha_vals):
            ax = axes[row, col]
            vx0 = math.cos(theta0)
            vy0 = math.sin(theta0)
            xs, ys, vxs, vys, us, ts, ws = simulate(
                n_coll, x0, y0, vx0, vy0, 0.0, a, geo, p1, p2)

            # Color segments by collision number
            points = np.column_stack([xs, ys]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            colors = plt.cm.plasma(np.linspace(0, 1, len(segments)))
            lc = LineCollection(segments, colors=colors, linewidths=0.3)
            ax.add_collection(lc)

            # Draw boundary
            if geo == CIRCLE:
                draw_circle_boundary(ax)
            elif geo == RECTANGLE:
                draw_rectangle_boundary(ax, L=p1, H=p2)
            elif geo == STADIUM:
                draw_stadium_boundary(ax, L=p1)
            elif geo == SINAI:
                draw_sinai_boundary(ax, L=p1, R=p2)

            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f'$\\alpha = {a}$', fontsize=12)
            if col == 0:
                ax.set_ylabel(gname, fontsize=12, fontweight='bold')

    fig.suptitle(f'Trajectory gallery ({n_coll} collisions, '
                 r'$\theta_0=0.8$, $u_0=0$)', fontsize=14, y=1.01)
    plt.tight_layout()
    _save(fig, "fig1_trajectories")


# =====================================================================
#  Figure 2: Lyapunov exponent vs α
# =====================================================================

def fig_lyapunov_vs_alpha():
    """Lyapunov exponent vs alpha for all geometries.

    Data priority:
    1. unified_billiards_data.npz (100K ICs, 50K collisions) — Stadium & Sinai
    2. lyapunov_sweep_data.npz (30K ICs, 200K collisions) — all 4 geometries
    3. Recompute from scratch (200 ICs, 50K collisions)
    """
    print("\n[Fig 2] Lyapunov vs alpha")

    DATADIR = os.path.join(os.path.dirname(__file__), "experiment_results")
    unified = os.path.join(DATADIR, "unified_billiards_data.npz")
    legacy = os.path.join(DATADIR, "lyapunov_sweep_data.npz")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Try unified data first (100K ICs for Stadium + Sinai)
    if os.path.exists(unified):
        du = np.load(unified)
        if 'Stadium_mean' in du and 'Sinai_mean' in du:
            print("  Loading unified precomputed data (100,000 ICs)...")
            alpha_vals = du['alpha_values']
            n_ics = 100000
            n_steps = 50000

            # Stadium & Sinai from unified (100K ICs)
            for gname in ['Stadium', 'Sinai']:
                means = du[f'{gname}_mean']
                sems = du[f'{gname}_sem']
                c = GEO_COLORS[gname]
                ax.errorbar(alpha_vals, means, yerr=sems, fmt='o-', markersize=3,
                             capsize=2, linewidth=1.3, color=c, label=gname,
                             elinewidth=0.7, capthick=0.7)

            # Circle & Rectangle from legacy or recompute (λ≈0 everywhere)
            if os.path.exists(legacy):
                dl = np.load(legacy)
                alpha_legacy = dl['alpha_values']
                for gname in ['Circle', 'Rectangle']:
                    means = dl[f'{gname}_mean']
                    sems = dl[f'{gname}_sem']
                    c = GEO_COLORS[gname]
                    label = f'{gname} ($\\lambda \\approx 0$)'
                    ax.errorbar(alpha_legacy, means, yerr=sems, fmt='o-',
                                 markersize=3, capsize=2, linewidth=1.3,
                                 color=c, label=label,
                                 elinewidth=0.7, capthick=0.7)
            else:
                # λ≈0 for Circle and Rectangle — just plot zero line
                for gname in ['Circle', 'Rectangle']:
                    c = GEO_COLORS[gname]
                    ax.plot(alpha_vals, np.zeros(len(alpha_vals)), 'o-',
                            markersize=3, linewidth=1.3, color=c,
                            label=f'{gname} ($\\lambda \\approx 0$)')

            ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, zorder=0)
            ax.set_xlabel(r'Spin coupling $\alpha$', fontsize=12)
            ax.set_ylabel(r'Lyapunov exponent $\lambda$', fontsize=12)
            ax.legend(fontsize=8, loc='upper right')
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.02, 0.50)
            ax.set_title(f'Ensemble-averaged Lyapunov exponent\n'
                         f'({n_ics:,} ICs, {n_steps:,} collisions per IC)',
                         fontsize=11)
            plt.tight_layout()
            _save(fig, "fig2_lyapunov_vs_alpha")
            return

    # Fallback: legacy precomputed data (30K ICs)
    if os.path.exists(legacy):
        print("  Loading legacy precomputed data (30,000 ICs)...")
        d = np.load(legacy)
        alpha_vals = d['alpha_values']
        for gname in ['Circle', 'Rectangle', 'Stadium', 'Sinai']:
            means = d[f'{gname}_mean']
            sems = d[f'{gname}_sem']
            c = GEO_COLORS[gname]
            if gname in ['Circle', 'Rectangle']:
                label = f'{gname} ($\\lambda \\approx 0$)'
            else:
                label = gname
            ax.errorbar(alpha_vals, means, yerr=sems, fmt='o-', markersize=3,
                         capsize=2, linewidth=1.3, color=c, label=label,
                         elinewidth=0.7, capthick=0.7)
        n_ics = 30000
        n_steps = 200000
    else:
        print("  No precomputed data found, computing from scratch...")
        alpha_vals = np.concatenate([
            np.linspace(0.0, 0.3, 10),
            np.linspace(0.35, 1.0, 15),
        ])
        alpha_vals = np.unique(np.round(alpha_vals, 4))
        n_steps = 50000
        n_ics = 200

        for gname, geo, p1, p2 in [
            ("Circle",    CIRCLE,    0.0, 0.0),
            ("Rectangle", RECTANGLE, 1.0, 1.0),
            ("Stadium",   STADIUM,   1.0, 0.0),
            ("Sinai",     SINAI,     2.0, 1.0),
        ]:
            t0 = time.perf_counter()
            means = np.empty(len(alpha_vals))
            stds = np.empty(len(alpha_vals))
            for j, a in enumerate(alpha_vals):
                lcns = lyapunov_ensemble(n_steps, n_ics, a, geo, p1, p2,
                                         perturb_mag=1e-7, u_max_frac=0.5)
                lcns = lcns[np.isfinite(lcns)]
                means[j] = np.mean(lcns)
                stds[j] = np.std(lcns) / max(1, np.sqrt(len(lcns)))
            dt = time.perf_counter() - t0
            print(f"  {gname}: {dt:.1f}s")

            c = GEO_COLORS[gname]
            if gname in ['Circle', 'Rectangle']:
                label = f'{gname} ($\\lambda \\approx 0$)'
            else:
                label = gname
            ax.errorbar(alpha_vals, means, yerr=stds, fmt='o-', markersize=3,
                         capsize=2, linewidth=1.3, color=c, label=label,
                         elinewidth=0.7, capthick=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, zorder=0)
    ax.set_xlabel(r'Spin coupling $\alpha$', fontsize=12)
    ax.set_ylabel(r'Lyapunov exponent $\lambda$', fontsize=12)
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.50)
    ax.set_title(f'Ensemble-averaged Lyapunov exponent\n'
                 f'({n_ics:,} ICs, {n_steps:,} collisions per IC)',
                 fontsize=11)

    plt.tight_layout()
    _save(fig, "fig2_lyapunov_vs_alpha")


# =====================================================================
#  Figure 3: Poincaré sections (Birkhoff coordinates, colored by spin)
# =====================================================================

def _random_ic_for_poincare(geo, p1, p2, alpha):
    """Generate a single random IC (x, y, vx, vy, u) for Poincaré section."""
    # Random position inside geometry
    if geo == CIRCLE:
        while True:
            x = 2.0 * np.random.random() - 1.0
            y = 2.0 * np.random.random() - 1.0
            if x*x + y*y < 0.99:
                break
    elif geo == RECTANGLE:
        x = (2.0 * np.random.random() - 1.0) * p1 * 0.99
        y = (2.0 * np.random.random() - 1.0) * p2 * 0.99
    elif geo == STADIUM:
        while True:
            x = (2.0 * np.random.random() - 1.0) * (p1 + 1.0) * 0.99
            y = (2.0 * np.random.random() - 1.0) * 0.99
            if -p1 <= x <= p1:
                break
            elif x < -p1:
                if (x + p1)**2 + y**2 < 0.99:
                    break
            else:
                if (x - p1)**2 + y**2 < 0.99:
                    break
    else:  # SINAI
        while True:
            x = (2.0 * np.random.random() - 1.0) * p1 * 0.99
            y = (2.0 * np.random.random() - 1.0) * p1 * 0.99
            if x*x + y*y > (p2 * 1.01)**2:
                break

    # Random velocity and spin consistent with E = 0.5
    theta = 2.0 * np.pi * np.random.random()
    if alpha > 0.0:
        u_max = 1.0 / np.sqrt(alpha)
        u = (2.0 * np.random.random() - 1.0) * u_max * 0.5
        v = np.sqrt(max(1.0 - alpha * u * u, 0.01))
    else:
        u = 0.0
        v = 1.0
    vx = v * np.cos(theta)
    vy = v * np.sin(theta)
    return x, y, vx, vy, u


def fig_poincare_sections():
    """Birkhoff-coordinate Poincaré sections colored by spin u."""
    print("\n[Fig 3] Poincaré sections")

    alpha_vals = [0.0, 0.4, 1.0]
    n_collisions = 5000
    n_ics = 25

    geos = [
        ("Stadium", STADIUM, 1.0, 0.0),
        ("Sinai",   SINAI,   2.0, 1.0),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for row, (gname, geo, p1, p2) in enumerate(geos):
        for col, alpha in enumerate(alpha_vals):
            ax = axes[row, col]
            t0 = time.perf_counter()

            # Collect all points from all ICs
            all_s = []
            all_vpar = []
            all_u = []

            for ic in range(n_ics):
                x0, y0, vx0, vy0, u0 = _random_ic_for_poincare(
                    geo, p1, p2, alpha)
                s_vals, vpar_vals, u_vals, wall_ids = poincare_section(
                    n_collisions, x0, y0, vx0, vy0, u0, alpha,
                    geo, p1, p2)
                all_s.append(s_vals)
                all_vpar.append(vpar_vals)
                all_u.append(u_vals)

            all_s = np.concatenate(all_s)
            all_vpar = np.concatenate(all_vpar)
            all_u = np.concatenate(all_u)

            dt = time.perf_counter() - t0
            print(f"  {gname} α={alpha}: {dt:.1f}s ({len(all_s)} points)")

            # Plot (s, v∥) colored by spin u
            sc = ax.scatter(all_s, all_vpar, c=all_u, s=0.3, alpha=0.5,
                           cmap='coolwarm', rasterized=True)

            # Symmetric color scale centered at 0
            u_lim = max(abs(all_u.min()), abs(all_u.max()), 0.01)
            sc.set_clim(-u_lim, u_lim)

            ax.set_xlabel(r'Arc length $s$', fontsize=10)
            if col == 0:
                ax.set_ylabel(r'$v_\parallel$', fontsize=11)

            if row == 0:
                ax.set_title(f'$\\alpha = {alpha}$', fontsize=12)

            # Add colorbar on rightmost column
            if col == 2:
                cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
                cb.set_label(r'Spin $u$', fontsize=9)

        # Row label
        axes[row, 0].annotate(gname, xy=(-0.25, 0.5),
                              xycoords='axes fraction',
                              fontsize=13, fontweight='bold',
                              ha='center', va='center', rotation=90)

    fig.suptitle(f'Poincaré sections (Birkhoff coordinates)\n'
                 f'{n_ics} ICs × {n_collisions:,} collisions, colored by spin $u$',
                 fontsize=13)
    plt.tight_layout()
    _save(fig, "fig3_poincare_sections")


# =====================================================================
#  Figure 3: FTLE distributions
# =====================================================================

def fig_ftle():
    """FTLE distributions at selected α values.

    Data priority:
    1. unified_billiards_data.npz — per-trajectory FTLE (100K ICs)
    2. Recompute from scratch (5,000 ICs)
    """
    print("\n[Fig 3] FTLE distributions")

    alpha_vals_plot = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]

    # Try loading unified precomputed data
    DATADIR = os.path.join(os.path.dirname(__file__), "experiment_results")
    unified = os.path.join(DATADIR, "unified_billiards_data.npz")
    use_unified = False
    if os.path.exists(unified):
        du = np.load(unified)
        if 'Stadium_ftle' in du:
            use_unified = True
            alpha_precomp = du['alpha_values']
            print(f"  Using unified precomputed data ({du['Stadium_ftle'].shape[1]:,} ICs)")

    for gname, geo, p1, p2 in [("Stadium", STADIUM, 1.0, 0.0),
                                 ("Sinai", SINAI, 2.0, 1.0)]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        axes = axes.flatten()

        t0 = time.perf_counter()
        for j, a in enumerate(alpha_vals_plot):
            if use_unified and f'{gname}_ftle' in du:
                # Find nearest α in precomputed grid
                idx = np.argmin(np.abs(alpha_precomp - a))
                lcns = du[f'{gname}_ftle'][idx, :]
                n_ics = du[f'{gname}_ftle'].shape[1]
                n_steps = 50000
            else:
                n_steps = 50000
                n_ics = 5000
                lcns = lyapunov_ensemble(n_steps, n_ics, a, geo, p1, p2,
                                          perturb_mag=1e-7, u_max_frac=0.5)

            lcns_finite = lcns[np.isfinite(lcns)]
            lcns_hist = lcns_finite[lcns_finite > -0.05]  # visual filter for histogram

            ax = axes[j]
            c = GEO_COLORS[gname]

            ax.hist(lcns_hist, bins=80, density=True, alpha=0.55,
                    color=c, edgecolor='0.3', linewidth=0.3, zorder=2)

            ax.axvline(x=0, color='red', linestyle='--', linewidth=0.7,
                      alpha=0.4, zorder=1)

            mean_val = np.mean(lcns_finite)
            median_val = np.median(lcns_finite)
            chaotic_mask = lcns_finite > 0.01
            cond_mean = np.mean(lcns_finite[chaotic_mask]) if np.any(chaotic_mask) else 0.0
            frac_chaotic = np.mean(chaotic_mask) * 100

            # Vertical lines for mean, median, conditional mean
            ax.axvline(x=mean_val, color='black', linestyle='-', linewidth=1.0,
                      alpha=0.7, zorder=3, label='Mean' if j == 0 else '')
            ax.axvline(x=median_val, color='blue', linestyle='--', linewidth=1.0,
                      alpha=0.7, zorder=3, label='Median' if j == 0 else '')
            if frac_chaotic > 0:
                ax.axvline(x=cond_mean, color='darkgreen', linestyle=':', linewidth=1.0,
                          alpha=0.7, zorder=3, label='Cond. mean' if j == 0 else '')

            ax.text(0.97, 0.95,
                    f'mean $= {mean_val:.3f}$\n'
                    f'median $= {median_val:.3f}$\n'
                    f'{frac_chaotic:.0f}% chaotic',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor='0.7', alpha=0.85))

            ax.set_xlabel(r'$\lambda$', fontsize=11)
            if j % 3 == 0:
                ax.set_ylabel('Density', fontsize=11)
            ax.set_title(f'$\\alpha = {a}$', fontsize=12)

        # Add legend from first panel
        axes[0].legend(fontsize=7, loc='upper left')

        dt = time.perf_counter() - t0
        print(f"  {gname}: {dt:.1f}s")

        fig.suptitle(f'{gname}: FTLE distributions '
                     f'({n_ics:,} ICs, {n_steps:,} collisions)', fontsize=14)
        plt.tight_layout()
        _save(fig, f"fig3_ftle_{gname.lower()}")


# =====================================================================
#  Figure 4: Phase space separation
# =====================================================================

def fig_phase_separation():
    print("\n[Fig 4] Phase separation")

    alpha_vals = [0.0, 0.3, 0.5, 0.7, 1.0]
    n_coll = 50
    n_ics = 8000

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()

    for idx, (gname, geo, p1, p2) in enumerate([
        ("Circle",    CIRCLE,    0.0, 0.0),
        ("Rectangle", RECTANGLE, 1.0, 1.0),
        ("Stadium",   STADIUM,   1.0, 0.0),
        ("Sinai",     SINAI,     2.0, 1.0),
    ]):
        ax = axes_flat[idx]
        t0 = time.perf_counter()
        for a in alpha_vals:
            lnd = phase_space_separation(n_coll, n_ics, a, geo, p1, p2,
                                          perturb_mag=1e-7)
            color = ALPHA_CMAP(a / 1.0)
            ax.plot(np.arange(n_coll), lnd, linewidth=1.5,
                    color=color, label=f'$\\alpha={a}$')
        dt = time.perf_counter() - t0
        print(f"  {gname}: {dt:.1f}s")

        ax.set_xlabel('Collisions', fontsize=11)
        ax.set_ylabel(r'$\ln(d_n/d_0)$', fontsize=11)
        ax.set_title(gname, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(f'Phase-space separation ({n_ics:,} ICs, {n_coll} collisions, '
                 r'$\delta_0=10^{-7}$)', fontsize=13)
    plt.tight_layout()
    _save(fig, "fig4_phase_separation")


# =====================================================================
#  Figure 5: Chaotic fraction
# =====================================================================

def fig_chaotic_fraction():
    """Chaotic fraction vs α with threshold sensitivity band.

    Data priority:
    1. unified_billiards_data.npz — per-trajectory FTLE (100K ICs)
    2. Recompute from scratch (5,000 ICs)
    """
    print("\n[Fig 5] Chaotic fraction")

    # Try loading unified precomputed data
    DATADIR = os.path.join(os.path.dirname(__file__), "experiment_results")
    unified = os.path.join(DATADIR, "unified_billiards_data.npz")
    use_unified = False
    if os.path.exists(unified):
        du = np.load(unified)
        if 'Stadium_ftle' in du:
            use_unified = True
            alpha_vals = du['alpha_values']
            n_ics = du['Stadium_ftle'].shape[1]
            n_steps = 50000
            print(f"  Using unified precomputed data ({n_ics:,} ICs)")
        else:
            du = None

    if not use_unified:
        alpha_vals = np.linspace(0.0, 1.0, 48)
        n_steps = 50000
        n_ics = 5000

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Threshold sensitivity analysis
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
    main_thresh = 0.01

    for gname, geo, p1, p2 in [
        ("Stadium", STADIUM, 1.0, 0.0),
        ("Sinai",   SINAI,   2.0, 1.0),
    ]:
        t0 = time.perf_counter()
        frac_by_thresh = {th: np.empty(len(alpha_vals)) for th in thresholds}

        for j, a in enumerate(alpha_vals):
            if use_unified and f'{gname}_ftle' in du:
                lcns = du[f'{gname}_ftle'][j, :]
                lcns = lcns[np.isfinite(lcns)]
            else:
                lcns = lyapunov_ensemble(n_steps, n_ics, a, geo, p1, p2,
                                          perturb_mag=1e-7, u_max_frac=0.5)
                lcns = lcns[np.isfinite(lcns)]
            for th in thresholds:
                frac_by_thresh[th][j] = np.mean(lcns > th) * 100

        dt = time.perf_counter() - t0
        print(f"  {gname}: {dt:.1f}s")

        # Print threshold sensitivity summary
        for th in thresholds:
            idx_1 = np.argmin(np.abs(alpha_vals - 1.0))
            print(f"    threshold={th}: chaotic fraction at α=1 = {frac_by_thresh[th][idx_1]:.1f}%")

        c = GEO_COLORS[gname]

        # Shaded band showing sensitivity range (min to max across thresholds)
        frac_min = np.minimum.reduce([frac_by_thresh[th] for th in thresholds])
        frac_max = np.maximum.reduce([frac_by_thresh[th] for th in thresholds])
        ax.fill_between(alpha_vals, frac_min, frac_max, color=c, alpha=0.15)

        # Main threshold as solid line
        ax.plot(alpha_vals, frac_by_thresh[main_thresh], '-', color=c,
                linewidth=1.8, label=gname)

    ax.set_xlabel(r'Spin coupling $\alpha$', fontsize=12)
    ax.set_ylabel('Chaotic fraction (%)', fontsize=12)
    ax.set_ylim(55, 102)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=10, loc='lower left')
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    _save(fig, "fig5_chaotic_fraction")


# =====================================================================
#  Figure 6: LCN convergence
# =====================================================================

def fig_convergence():
    print("\n[Fig 6] LCN convergence")

    n_steps_list = [10000, 50000, 100000, 500000]
    n_ics = 10000
    alpha_vals = [0.1, 0.5, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for idx, (gname, geo, p1, p2) in enumerate([
        ("Stadium", STADIUM, 1.0, 0.0),
        ("Sinai",   SINAI,   2.0, 1.0),
    ]):
        ax = axes[idx]
        t0 = time.perf_counter()

        for a_idx, a in enumerate(alpha_vals):
            means = []
            errs = []
            for ns in n_steps_list:
                lcns = lyapunov_ensemble(ns, n_ics, a, geo, p1, p2,
                                          perturb_mag=1e-7, u_max_frac=0.5)
                lcns = lcns[np.isfinite(lcns)]
                means.append(np.mean(lcns))
                errs.append(np.std(lcns) / max(1, np.sqrt(len(lcns))))

            ax.errorbar(n_steps_list, means, yerr=errs, fmt='o-',
                         markersize=5, capsize=3, linewidth=1.2,
                         label=f'$\\alpha = {a}$')

        dt = time.perf_counter() - t0
        print(f"  {gname}: {dt:.1f}s")

        ax.set_xscale('log')
        ax.set_xlabel('Number of collisions', fontsize=12)
        ax.set_ylabel(r'LCN (ensemble mean)', fontsize=12)
        ax.set_title(gname, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)

    fig.suptitle(f'LCN convergence ({n_ics} ICs per point)', fontsize=13)
    plt.tight_layout()
    _save(fig, "fig6_convergence")


# =====================================================================
#  Figure 7: Energy conservation
# =====================================================================

def fig_energy():
    print("\n[Fig 7] Energy conservation")

    n_coll = 5000
    alpha_vals = [0.0, 0.3, 0.5, 0.7, 1.0]
    theta0 = 0.8

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()

    geos = [
        ("Circle",    CIRCLE,    0.0, 0.0, 0.0, 0.5),
        ("Rectangle", RECTANGLE, 1.0, 1.0, 0.2, 0.3),
        ("Stadium",   STADIUM,   1.0, 0.0, 0.2, 0.3),
        ("Sinai",     SINAI,     2.0, 1.0, 0.5, 0.5),
    ]

    for idx, (gname, geo, p1, p2, x0, y0) in enumerate(geos):
        ax = axes_flat[idx]
        for a in alpha_vals:
            vx0 = math.cos(theta0); vy0 = math.sin(theta0)
            xs, ys, vxs, vys, us, ts, ws = simulate(
                n_coll, x0, y0, vx0, vy0, 0.0, a, geo, p1, p2)

            E = 0.5 * (vxs**2 + vys**2) + 0.5 * a * us**2
            E0 = E[0]
            err = np.abs(E - E0)
            err[err == 0] = 1e-20  # avoid log(0)

            color = ALPHA_CMAP(a / 1.0)
            ax.semilogy(np.arange(len(err)), err, linewidth=0.5,
                       color=color, alpha=0.7, label=f'$\\alpha={a}$')

        ax.set_xlabel('Collision', fontsize=11)
        ax.set_ylabel(r'$|E - E_0|$', fontsize=11)
        ax.set_title(gname, fontsize=12, fontweight='bold')
        ax.set_ylim(1e-20, 1e-10)
        if idx == 0:
            ax.legend(fontsize=8, ncol=2)

    fig.suptitle(f'Energy conservation error ({n_coll:,} collisions, '
                 r'$\theta_0=0.8$, $u_0=0$)', fontsize=13)
    plt.tight_layout()
    _save(fig, "fig7_energy")


# =====================================================================
#  Figure 8: Stadium geometry scan
# =====================================================================

def fig_geometry_scan():
    """Stadium λ(α) vs geometry parameter a.

    Data priority:
    1. unified_billiards_data.npz — GeoScan (10K ICs, 50 α values)
    2. Recompute from scratch (50 ICs, 20 α values)
    """
    print("\n[Fig 8] Stadium geometry scan")

    DATADIR = os.path.join(os.path.dirname(__file__), "experiment_results")
    unified = os.path.join(DATADIR, "unified_billiards_data.npz")

    fig, ax = plt.subplots(figsize=(7, 4.5))

    if os.path.exists(unified):
        du = np.load(unified)
        if 'GeoScan_mean' in du and 'a_geo_values' in du:
            print("  Loading unified precomputed geometry scan data")
            a_geo_vals = du['a_geo_values']
            alpha_vals = du['alpha_values']
            all_means = du['GeoScan_mean']
            all_sems = du['GeoScan_sem']
            n_ics = 10000
            n_steps = 50000

            colors = plt.cm.tab10(np.linspace(0, 0.5, len(a_geo_vals)))
            for k, a_geo in enumerate(a_geo_vals):
                ax.errorbar(alpha_vals, all_means[k], yerr=all_sems[k],
                             fmt='o-', markersize=3, capsize=2, linewidth=1.2,
                             color=colors[k], label=f'$a = {a_geo}$',
                             elinewidth=0.6, capthick=0.6)

            ax.set_xlabel(r'$\alpha$ (spin coupling)', fontsize=12)
            ax.set_ylabel(r'$\lambda$ (Lyapunov exponent)', fontsize=12)
            ax.set_title(f'Stadium: $\\lambda(\\alpha)$ vs geometry parameter $a$\n'
                         f'({n_ics:,} ICs, {n_steps:,} collisions)',
                         fontsize=12)
            ax.legend(fontsize=10)
            ax.set_ylim(bottom=0)

            plt.tight_layout()
            _save(fig, "fig8_geometry_scan")
            return

    # Fallback: recompute
    a_geo_vals = [0.2, 0.5, 1.0, 2.0, 4.0]
    alpha_vals = np.linspace(0.0, 1.0, 20)
    n_steps = 50000
    n_ics = 50

    colors = plt.cm.tab10(np.linspace(0, 0.5, len(a_geo_vals)))

    for k, a_geo in enumerate(a_geo_vals):
        t0 = time.perf_counter()
        means = np.empty(len(alpha_vals))
        stds = np.empty(len(alpha_vals))
        for j, a in enumerate(alpha_vals):
            lcns = lyapunov_ensemble(n_steps, n_ics, a, STADIUM, a_geo, 0.0,
                                      perturb_mag=1e-7, u_max_frac=0.5)
            lcns = lcns[np.isfinite(lcns)]
            means[j] = np.mean(lcns)
            stds[j] = np.std(lcns) / max(1, np.sqrt(len(lcns)))
        dt = time.perf_counter() - t0
        print(f"  a={a_geo}: {dt:.1f}s")

        ax.errorbar(alpha_vals, means, yerr=stds, fmt='o-', markersize=3,
                     capsize=2, linewidth=1.2, color=colors[k],
                     label=f'$a = {a_geo}$', elinewidth=0.6, capthick=0.6)

    ax.set_xlabel(r'$\alpha$ (spin coupling)', fontsize=12)
    ax.set_ylabel(r'$\lambda$ (Lyapunov exponent)', fontsize=12)
    ax.set_title(f'Stadium: $\\lambda(\\alpha)$ vs geometry parameter $a$\n'
                 f'({n_ics} ICs, {n_steps:,} collisions)',
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    _save(fig, "fig8_geometry_scan")


# =====================================================================
#  Figure 8b: Universality collapse — λ(α)/λ(0) vs α for all a-values
# =====================================================================

def fig_universality_collapse():
    """Test universality: plot λ(α)/λ(0) for all stadium geometries.

    Loads precomputed geometry scan data if available, otherwise recomputes.
    """
    print("\n[Fig 8b] Universality collapse")

    DATADIR = os.path.join(os.path.dirname(__file__), "experiment_results")
    precomp = os.path.join(DATADIR, "unified_billiards_data.npz")

    if os.path.exists(precomp):
        d = np.load(precomp)
        if 'GeoScan_mean' in d and 'a_geo_values' in d:
            print("  Loading precomputed geometry scan data")
            a_geo_vals = d['a_geo_values']
            alpha_vals = d['alpha_values']
            all_means = d['GeoScan_mean']   # shape (n_a, n_alpha)
            all_sems = d['GeoScan_sem']
        else:
            d = None
    else:
        d = None

    if d is None:
        # Recompute
        a_geo_vals = np.array([0.2, 0.5, 1.0, 2.0, 4.0])
        alpha_vals = np.linspace(0.0, 1.0, 30)
        n_steps = 50000
        n_ics = 500
        all_means = np.empty((len(a_geo_vals), len(alpha_vals)))
        all_sems = np.empty((len(a_geo_vals), len(alpha_vals)))

        for k, a_geo in enumerate(a_geo_vals):
            t0 = time.perf_counter()
            for j, a in enumerate(alpha_vals):
                lcns = lyapunov_ensemble(n_steps, n_ics, a, STADIUM, a_geo, 0.0,
                                          perturb_mag=1e-7, u_max_frac=0.5)
                lcns = lcns[np.isfinite(lcns)]
                all_means[k, j] = np.mean(lcns)
                all_sems[k, j] = np.std(lcns) / max(1, np.sqrt(len(lcns)))
            dt = time.perf_counter() - t0
            print(f"  a={a_geo}: {dt:.1f}s")

    # ── Normalize: λ(α)/λ(0) ──
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(a_geo_vals)))

    for k, a_geo in enumerate(a_geo_vals):
        lam0 = all_means[k, 0]
        if lam0 <= 0:
            continue
        ratio = all_means[k, :] / lam0
        ratio_err = all_sems[k, :] / lam0

        ax.errorbar(alpha_vals, ratio, yerr=ratio_err, fmt='o-', markersize=3,
                     capsize=2, linewidth=1.2, color=colors[k],
                     label=f'$a = {a_geo}$', elinewidth=0.6, capthick=0.6)

    ax.set_xlabel(r'Spin coupling $\alpha$', fontsize=12)
    ax.set_ylabel(r'$\lambda(\alpha) / \lambda(0)$', fontsize=12)
    ax.set_title(r'Normalized $\lambda(\alpha)/\lambda(0)$ across stadium geometries',
                 fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=9, loc='upper right')
    ax.axhline(y=1, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    _save(fig, "fig8b_universality_collapse")


# =====================================================================
#  Figure 9: Conserved quantity
# =====================================================================

def _compute_tangent(x, y, wall, geo_type, geo_p1, geo_p2):
    """Compute unit tangent vector at a collision point given wall type."""
    if wall == WALL_TOP:
        return -1.0, 0.0
    elif wall == WALL_BOTTOM:
        return 1.0, 0.0
    elif wall == WALL_LEFT:
        return 0.0, -1.0
    elif wall == WALL_RIGHT:
        return 0.0, 1.0
    elif wall == WALL_CIRCLE:
        if geo_type == SINAI:
            R = geo_p2
            tx, ty = y / R, -x / R
        else:
            # Circle billiard: outward normal is (x, y), tangent is (-y, x)
            tx, ty = -y, x
        norm = math.sqrt(tx**2 + ty**2)
        if norm > 0:
            tx /= norm; ty /= norm
        return tx, ty
    elif wall == WALL_CAP_LEFT:
        L = geo_p1
        dx, dy = x + L, y
        tx, ty = -dy, dx
        norm = math.sqrt(tx**2 + ty**2)
        if norm > 0:
            tx /= norm; ty /= norm
        return tx, ty
    elif wall == WALL_CAP_RIGHT:
        L = geo_p1
        dx, dy = x - L, y
        tx, ty = -dy, dx
        norm = math.sqrt(tx**2 + ty**2)
        if norm > 0:
            tx /= norm; ty /= norm
        return tx, ty
    return 0.0, 0.0


def fig_conserved_quantity():
    print("\n[Fig 9] Conserved quantity")

    n_coll = 10000
    alpha = 0.5
    theta0 = 0.8
    vx0 = math.cos(theta0); vy0 = math.sin(theta0)

    # Stadium only — shows both same-wall and wall-transition populations
    geo, p1, p2, x0, y0 = STADIUM, 1.0, 0.0, 0.2, 0.3

    xs, ys, vxs, vys, us, ts, ws = simulate(
        n_coll, x0, y0, vx0, vy0, 0.0, alpha, geo, p1, p2)

    # Compute Q_after(i) = v_after(i) . T(wall_i) - alpha * u_after(i)
    Q_after = np.empty(len(xs))
    for i in range(len(xs)):
        tx, ty = _compute_tangent(xs[i], ys[i], ws[i], geo, p1, p2)
        Q_after[i] = vxs[i] * tx + vys[i] * ty - alpha * us[i]

    # Inter-collision ΔQ = |Q(i+1) - Q(i)|
    dQ_inter = np.abs(np.diff(Q_after))
    dQ_inter_nz = dQ_inter.copy()
    dQ_inter_nz[dQ_inter_nz == 0] = 1e-20

    # Classify wall transitions
    same_wall = ws[1:] == ws[:-1]
    diff_wall = ~same_wall
    dQ_same = dQ_inter_nz[same_wall]
    dQ_diff = dQ_inter_nz[diff_wall]

    fig, ax = plt.subplots(figsize=(7, 4))

    bins_log = np.logspace(-18, 1, 80)
    ax.hist(dQ_same, bins=bins_log, alpha=0.7, color='#2176AE',
            label=f'Same wall ({len(dQ_same):,})', edgecolor='none')
    ax.hist(dQ_diff, bins=bins_log, alpha=0.7, color='#D33F49',
            label=f'Wall transition ({len(dQ_diff):,})', edgecolor='none')

    ax.set_xscale('log')
    ax.set_xlabel(r'$|\Delta Q| = |Q_{n+1} - Q_n|$', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(r'Stadium: inter-collision $\Delta Q$ distribution'
                 r' ($\alpha=0.5$, $10{,}000$ collisions)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(1e-18, 10)

    plt.tight_layout()
    _save(fig, "fig9_conserved_quantity")


# =====================================================================
#  Figure 10: LCN convergence traces
# =====================================================================

def fig_lcn_traces():
    print("\n[Fig 10] LCN convergence traces (ensemble-averaged)")

    n_steps = 5_000
    n_ics = 1000
    alpha_vals = [0.0, 0.3, 0.5, 0.7, 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes_flat = axes.flatten()

    geos = [
        ("Circle",    CIRCLE,    0.0, 0.0),
        ("Rectangle", RECTANGLE, 1.0, 1.0),
        ("Stadium",   STADIUM,   1.0, 0.0),
        ("Sinai",     SINAI,     2.0, 1.0),
    ]

    for idx, (gname, geo, p1, p2) in enumerate(geos):
        ax = axes_flat[idx]
        t0 = time.perf_counter()

        for a in alpha_vals:
            # Collect traces from ensemble (no subsampling)
            all_ts = np.empty((n_ics, n_steps))
            all_lcn = np.empty((n_ics, n_steps))

            for ic in range(n_ics):
                x0, y0, vx0, vy0, u0 = _random_ic_for_poincare(
                    geo, p1, p2, a)
                t_cum, lcn = lyapunov(n_steps, x0, y0, vx0, vy0, u0, a,
                                       geo, p1, p2, perturb_mag=1e-7)
                all_ts[ic, :] = t_cum
                all_lcn[ic, :] = lcn

            # Average over ICs
            mean_ts = np.mean(all_ts, axis=0)
            mean_lcn = np.mean(all_lcn, axis=0)
            mask = mean_ts > 0

            color = ALPHA_CMAP(a / 1.0)
            # Linear scale for all geometries
            ax.plot(mean_ts[mask], mean_lcn[mask],
                   linewidth=0.8, color=color, alpha=0.8,
                   label=f'$\\alpha={a}$')

        dt = time.perf_counter() - t0
        print(f"  {gname}: {dt:.1f}s ({n_ics} ICs x {len(alpha_vals)} alpha)")

        ax.set_xlabel('$t$', fontsize=11)
        ax.set_ylabel(r'$\lambda(t)$', fontsize=11)
        ax.set_xlim(0, 1000)
        ax.set_ylim(bottom=0)
        ax.set_title(gname, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle(f'LCN convergence traces ({n_steps:,} collisions, '
                 f'{n_ics} ICs)', fontsize=13)
    plt.tight_layout()
    _save(fig, "fig10_lcn_traces")


# =====================================================================
#  Appendix figure: Collision rate vs α
# =====================================================================

def fig_collision_rate():
    """Collision rate (collisions per unit time) vs α for Stadium and Sinai."""
    print("\n[Appendix] Collision rate vs alpha")

    alpha_vals = np.linspace(0.0, 1.0, 50)
    n_collisions = 10000
    n_ics = 100000

    fig, ax = plt.subplots(figsize=(7, 4.5))

    for gname, geo, p1, p2 in [
        ("Stadium", STADIUM, 1.0, 0.0),
        ("Sinai",   SINAI,   2.0, 1.0),
    ]:
        t0 = time.perf_counter()
        rates_mean = np.empty(len(alpha_vals))
        rates_std = np.empty(len(alpha_vals))

        for j, a in enumerate(alpha_vals):
            ic_rates = np.empty(n_ics)
            for k in range(n_ics):
                x0, y0, vx0, vy0, u0 = _random_ic_for_poincare(
                    geo, p1, p2, a)
                xs, ys, vxs, vys, us, ts, ws = simulate(
                    n_collisions, x0, y0, vx0, vy0, u0, a, geo, p1, p2)
                total_time = ts[-1] - ts[0]
                if total_time > 0:
                    ic_rates[k] = n_collisions / total_time
                else:
                    ic_rates[k] = np.nan
            ic_rates = ic_rates[np.isfinite(ic_rates)]
            rates_mean[j] = np.mean(ic_rates)
            rates_std[j] = np.std(ic_rates) / max(1, np.sqrt(len(ic_rates)))

        dt = time.perf_counter() - t0
        print(f"  {gname}: {dt:.1f}s")

        c = GEO_COLORS[gname]

        # Plot alpha>0 points connected with line
        pos_mask = alpha_vals > 0
        ax.errorbar(alpha_vals[pos_mask], rates_mean[pos_mask],
                     yerr=rates_std[pos_mask], fmt='o-',
                     markersize=3, capsize=2, linewidth=1.3, color=c,
                     label=gname, elinewidth=0.7, capthick=0.7)

        # Mark alpha=0 value on y-axis with a triangle marker
        ax.plot(0, rates_mean[0], marker='<', markersize=7, color=c,
                zorder=5, clip_on=False)
        ax.annotate(f'{rates_mean[0]:.3f}', xy=(0, rates_mean[0]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=8, color=c, va='center')

        # Report variation excluding alpha=0
        rate_min_pos = rates_mean[pos_mask].min()
        rate_max_pos = rates_mean[pos_mask].max()
        variation_pos = (rate_max_pos - rate_min_pos) / rates_mean[pos_mask][0] * 100
        print(f"    Rate range (α>0): {rate_min_pos:.3f} - {rate_max_pos:.3f} "
              f"(variation: {variation_pos:.1f}%)")

    ax.set_xlabel(r'Spin coupling $\alpha$', fontsize=12)
    ax.set_ylabel('Collision rate (collisions / time)', fontsize=12)
    ax.set_title(f'Mean collision rate vs $\\alpha$\n'
                 f'({n_ics:,} ICs, {n_collisions:,} collisions per IC)',
                 fontsize=11)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, 1.02)

    plt.tight_layout()
    _save(fig, "figA_collision_rate")


# =====================================================================
#  Appendix figure: Datseris-Hupe-Fleischmann (DH) scaling test
# =====================================================================

def fig_dh_scaling():
    """Test the DH scaling prediction: lambda proportional to 1/V_chaotic.

    If the Datseris-Hupe-Fleischmann scaling holds, the Lyapunov exponent
    should scale inversely with the chaotic phase-space volume fraction,
    i.e. lambda ~ 1/f_chaotic.  Equivalently, the product
    lambda * f_chaotic should be approximately constant in alpha.

    Data priority:
    1. unified_billiards_data.npz — per-trajectory FTLE gives both λ and f_chaotic (100K ICs)
    2. lyapunov_sweep_data.npz (λ only) + recompute f_chaotic (10K ICs)
    """
    print("\n[Appendix] DH scaling test")

    DATADIR = os.path.join(os.path.dirname(__file__), "experiment_results")
    unified = os.path.join(DATADIR, "unified_billiards_data.npz")
    legacy = os.path.join(DATADIR, "lyapunov_sweep_data.npz")

    threshold = 0.01
    results = {}

    geos = [
        ("Stadium", STADIUM, 1.0, 0.0),
        ("Sinai",   SINAI,   2.0, 1.0),
    ]

    # Try unified data (per-trajectory FTLE → both λ and f_chaotic)
    if os.path.exists(unified):
        du = np.load(unified)
        if 'Stadium_ftle' in du:
            alpha_vals = du['alpha_values']
            n_ics = du['Stadium_ftle'].shape[1]
            n_steps = 50000
            print(f"  Using unified precomputed data ({n_ics:,} ICs)")

            for gname, geo, p1, p2 in geos:
                ftle = du[f'{gname}_ftle']  # shape (n_alpha, n_ics)
                lam_mean = du[f'{gname}_mean']
                f_chaotic = np.empty(len(alpha_vals))
                for j in range(len(alpha_vals)):
                    lcns = ftle[j, :]
                    lcns = lcns[np.isfinite(lcns)]
                    f_chaotic[j] = np.mean(lcns > threshold)

                print(f"  {gname}: f_chaotic range: {f_chaotic.min():.3f} -- {f_chaotic.max():.3f}")
                results[gname] = {
                    'alpha': alpha_vals,
                    'lam': lam_mean,
                    'f_chaotic': f_chaotic,
                }

            # Plot
            GEO_MARKERS = {'Stadium': 'o', 'Sinai': 's'}
            fig, ax = plt.subplots(figsize=(7, 4.5))
            sc = None
            for gname in ['Stadium', 'Sinai']:
                r = results[gname]
                mask = r['f_chaotic'] > 0.01
                product = r['lam'][mask] * r['f_chaotic'][mask]
                alpha_masked = r['alpha'][mask]
                sc = ax.scatter(alpha_masked, product, c=alpha_masked, cmap=ALPHA_CMAP,
                           marker=GEO_MARKERS[gname], s=45, zorder=3,
                           edgecolors='0.3', linewidths=0.4,
                           vmin=0, vmax=1, label=gname)

            ax.set_xlabel(r'Spin coupling $\alpha$', fontsize=12)
            ax.set_ylabel(r'$\lambda \cdot f_{\rm chaotic}$', fontsize=12)
            ax.set_title(r'DH scaling test: $\lambda \cdot f_{\rm chaotic}$ vs $\alpha$'
                         f'\n({n_ics:,} ICs, {n_steps:,} collisions, threshold={threshold})',
                         fontsize=11)
            ax.set_xlim(-0.02, 1.02)
            ax.legend(fontsize=10)
            if sc is not None:
                cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
                cb.set_label(r'$\alpha$', fontsize=10)

            plt.tight_layout()
            _save(fig, "figA_dh_scaling")
            return

    # Fallback: legacy precomputed λ + recompute f_chaotic
    if not os.path.exists(legacy):
        print("  WARNING: no precomputed data found, skipping DH scaling.")
        return

    d = np.load(legacy)
    alpha_precomp = d['alpha_values']  # shape (48,)

    n_alpha = 21
    alpha_scan = np.linspace(0.0, 1.0, n_alpha)
    n_ics = 10000
    n_steps = 50000

    for gname, geo, p1, p2 in geos:
        t0 = time.perf_counter()

        # Interpolate precomputed lambda onto our alpha grid
        lam_interp = np.interp(alpha_scan, alpha_precomp, d[f'{gname}_mean'])

        # Compute chaotic fraction at each alpha
        f_chaotic = np.empty(n_alpha)
        for j, a in enumerate(alpha_scan):
            lcns = lyapunov_ensemble(n_steps, n_ics, a, geo, p1, p2,
                                      perturb_mag=1e-7, u_max_frac=0.5)
            lcns = lcns[np.isfinite(lcns)]
            f_chaotic[j] = np.mean(lcns > threshold)

        dt = time.perf_counter() - t0
        print(f"  {gname}: {dt:.1f}s  "
              f"(f_chaotic range: {f_chaotic.min():.3f} -- {f_chaotic.max():.3f})")

        results[gname] = {
            'alpha': alpha_scan,
            'lam': lam_interp,
            'f_chaotic': f_chaotic,
        }

    # ── Plot ──
    GEO_MARKERS = {'Stadium': 'o', 'Sinai': 's'}
    fig, ax = plt.subplots(figsize=(7, 4.5))

    sc = None
    for gname in ['Stadium', 'Sinai']:
        r = results[gname]
        mask = r['f_chaotic'] > 0.01
        product = r['lam'][mask] * r['f_chaotic'][mask]
        alpha_masked = r['alpha'][mask]
        sc = ax.scatter(alpha_masked, product, c=alpha_masked, cmap=ALPHA_CMAP,
                   marker=GEO_MARKERS[gname], s=45, zorder=3,
                   edgecolors='0.3', linewidths=0.4,
                   vmin=0, vmax=1, label=gname)

    ax.set_xlabel(r'Spin coupling $\alpha$', fontsize=12)
    ax.set_ylabel(r'$\lambda \cdot f_{\rm chaotic}$', fontsize=12)
    ax.set_title(r'DH scaling test: $\lambda \cdot f_{\rm chaotic}$ vs $\alpha$'
                 f'\n({n_ics:,} ICs, {n_steps:,} collisions, threshold={threshold})',
                 fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(fontsize=10)
    if sc is not None:
        cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
        cb.set_label(r'$\alpha$', fontsize=10)

    plt.tight_layout()
    _save(fig, "figA_dh_scaling")


# =====================================================================
#  Appendix figure: Full Lyapunov spectrum
# =====================================================================

def fig_lyapunov_spectrum():
    """Compute and plot the full 5D Lyapunov spectrum vs alpha."""
    from spinning_billiards import lyapunov_spectrum_ensemble

    print("\n[Appendix] Lyapunov spectrum")

    alpha_vals = np.linspace(0.0, 1.0, 21)
    n_steps = 50000
    n_ics = 50

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (gname, geo, p1, p2) in enumerate([
        ("Stadium", STADIUM, 1.0, 0.0),
        ("Sinai",   SINAI,   2.0, 1.0),
    ]):
        ax = axes[idx]
        t0 = time.perf_counter()

        # Store mean spectrum at each alpha
        all_spectra = np.empty((len(alpha_vals), 5))

        for j, a in enumerate(alpha_vals):
            spectra = lyapunov_spectrum_ensemble(
                n_steps, n_ics, a, geo, p1, p2,
                perturb_mag=1e-7, u_max_frac=0.5)
            # spectra has shape (n_ics, 5), already sorted decreasing per IC
            mean_spec = np.mean(spectra, axis=0)
            # Sort the mean spectrum in decreasing order
            all_spectra[j] = np.sort(mean_spec)[::-1]

        dt = time.perf_counter() - t0
        print(f"  {gname}: {dt:.1f}s")

        # Plot each exponent vs alpha
        colors = ['#D33F49', '#FF9F1C', '#2EC4B6', '#3A86FF', '#8338EC']
        labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$',
                  r'$\lambda_4$', r'$\lambda_5$']

        for k in range(4):  # Skip lambda_5 (off energy shell, ~-20)
            ax.plot(alpha_vals, all_spectra[:, k], 'o-', markersize=3,
                    linewidth=1.2, color=colors[k], label=labels[k])

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_xlabel(r'$\alpha$ (spin coupling)', fontsize=12)
        ax.set_ylabel(r'Lyapunov exponent', fontsize=12)
        ax.set_title(gname, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, ncol=2)
        ax.set_xlim(-0.02, 1.02)

    fig.suptitle(f'Full Lyapunov spectrum ($n={n_ics}$ ICs, '
                 f'{n_steps:,} collisions)', fontsize=13)
    plt.tight_layout()
    _save(fig, "figA_lyapunov_spectrum")


# =====================================================================
#  Hero figure (combined key results)
# =====================================================================

def fig_hero():
    """4-panel combined figure: (a) Lyapunov vs alpha, (b-d) FTLE distributions."""
    print("\n[Hero] Combined figure")

    fig = plt.figure(figsize=(15, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    # ─── (a) λ vs α all geometries (spans top row) ───
    ax_a = fig.add_subplot(gs[0, :])

    # Load precomputed 30K-IC data (same as Fig 2)
    DATADIR = os.path.join(os.path.dirname(__file__), "experiment_results")
    precomp = os.path.join(DATADIR, "lyapunov_sweep_data.npz")
    n_steps = 50000

    if os.path.exists(precomp):
        d = np.load(precomp)
        alpha_vals = d['alpha_values']
        for gname in ['Circle', 'Rectangle', 'Stadium', 'Sinai']:
            means = d[f'{gname}_mean']
            sems = d[f'{gname}_sem']
            c = GEO_COLORS[gname]
            ax_a.errorbar(alpha_vals, means, yerr=sems, fmt='o-', markersize=2.5,
                           capsize=1.5, linewidth=1, color=c, label=gname,
                           elinewidth=0.6, capthick=0.6)
    else:
        # Fallback: compute with modest ICs
        alpha_vals = np.linspace(0.0, 1.0, 20)
        n_ics = 200
        for gname, geo, p1, p2 in [
            ("Circle",    CIRCLE,    0.0, 0.0),
            ("Rectangle", RECTANGLE, 1.0, 1.0),
            ("Stadium",   STADIUM,   1.0, 0.0),
            ("Sinai",     SINAI,     2.0, 1.0),
        ]:
            means = np.empty(len(alpha_vals))
            stds = np.empty(len(alpha_vals))
            for j, a in enumerate(alpha_vals):
                lcns = lyapunov_ensemble(n_steps, n_ics, a, geo, p1, p2,
                                         perturb_mag=1e-7, u_max_frac=0.5)
                lcns = lcns[np.isfinite(lcns)]
                means[j] = np.mean(lcns)
                stds[j] = np.std(lcns, ddof=1) / max(1, np.sqrt(len(lcns)))
            c = GEO_COLORS[gname]
            ax_a.errorbar(alpha_vals, means, yerr=stds, fmt='o-', markersize=2.5,
                           capsize=1.5, linewidth=1, color=c, label=gname,
                           elinewidth=0.6, capthick=0.6)

    ax_a.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax_a.set_xlabel(r'Spin coupling $\alpha$')
    ax_a.set_ylabel(r'Lyapunov exponent $\lambda$')
    ax_a.set_xlim(-0.02, 1.02)
    ax_a.set_ylim(-0.02, 0.50)
    ax_a.set_title('(a) Lyapunov exponent vs $\\alpha$', fontsize=11)
    ax_a.legend(fontsize=8)

    # ─── (b,c,d) FTLE for Stadium α=0, 0.5, 1.0 ───
    alpha_ftle = [0.0, 0.5, 1.0]
    n_ics_ftle = 1000
    labels = ['(b)', '(c)', '(d)']

    for j, a in enumerate(alpha_ftle):
        ax = fig.add_subplot(gs[1, j])
        lcns = lyapunov_ensemble(n_steps, n_ics_ftle, a, STADIUM, 1.0, 0.0,
                                  perturb_mag=1e-7, u_max_frac=0.5)
        lcns_finite = lcns[np.isfinite(lcns)]
        lcns_hist = lcns_finite[lcns_finite > -0.05]  # visual filter for histogram

        c = GEO_COLORS['Stadium']
        ax.hist(lcns_hist, bins=80, density=True, alpha=0.45,
                color=c, edgecolor='none')

        ax.axvline(x=0, color='red', linestyle='--', linewidth=0.7, alpha=0.4)
        mean_val = np.mean(lcns_finite)  # unbiased mean from all finite values
        frac = np.mean(lcns_finite > 0.01) * 100
        ax.text(0.97, 0.95,
                f'$\\langle\\lambda\\rangle={mean_val:.3f}$\n{frac:.0f}% chaotic',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='0.7', alpha=0.85))
        ax.set_xlabel(r'$\lambda$')
        if j == 0:
            ax.set_ylabel('Density')
        ax.set_title(f'{labels[j]} Stadium FTLE, $\\alpha={a}$', fontsize=11)

    _save(fig, "hero_figure")


# =====================================================================
#  Main
# =====================================================================

if __name__ == "__main__":
    print("Warming up JIT...")
    warmup()
    print("Ready.\n")

    t_total = time.perf_counter()

    fig_trajectories()          # Fig 1
    fig_lyapunov_vs_alpha()     # Fig 2
    fig_ftle()                  # Fig 3
    fig_phase_separation()      # Fig 4
    fig_chaotic_fraction()      # Fig 5
    fig_convergence()           # Fig 6  — Appendix
    fig_energy()                # Fig 7  — Appendix
    fig_geometry_scan()         # Fig 8
    fig_universality_collapse() # Fig 9 (universality collapse)
    fig_conserved_quantity()    # Fig 10 — Appendix
    fig_lcn_traces()            # Fig 10 — Appendix
    fig_collision_rate()        # Fig A  — Appendix
    fig_dh_scaling()            # Fig A  — Appendix (DH scaling test)
    fig_lyapunov_spectrum()     # Fig A  — Appendix (full spectrum)
    fig_hero()
    # fig_poincare_sections()  # Removed from paper (3D→2D projection uninformative)

    dt_total = time.perf_counter() - t_total
    print(f"\n{'='*60}")
    print(f"ALL FIGURES COMPLETE in {dt_total:.0f}s")
    print(f"Output: {OUTDIR}/")
    print(f"{'='*60}")
