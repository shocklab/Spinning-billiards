"""Regenerate the manuscript's inherited figures at print-legible type.

Each figure keeps its source layout, but all fonts are scaled by the factor
by which LaTeX shrinks it on the page (figure width / rendered width), so
labels land at ~8.5 pt in print. This implements the R1.10 promise ("all
figures have been redrawn at larger type") for the figures inherited from the
submitted version. Outputs are written as <name>_v2.pdf/png; the original
files are snapshotted first and restored afterwards, so the submitted
manuscript keeps its original figures byte-for-byte.

In-figure suptitles are suppressed (monkey-patch below): they duplicate the
LaTeX captions and shrink worst of all.
"""

import os
import shutil
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.figure
import matplotlib.pyplot as plt

matplotlib.figure.Figure.suptitle = lambda self, *a, **k: None

import make_plots as mp

COL, TEXT = 3.375, 7.05          # rendered widths: \columnwidth, ~0.85\textwidth
BACKUP = "plots_v2/_originals_backup"

JOBS = [
    (mp.fig_trajectories,          12.8 / COL,  ["fig1_trajectories"]),
    (mp.fig_lyapunov_vs_alpha,     7.0 / COL,   ["fig2_lyapunov_vs_alpha"]),
    (mp.fig_ftle,                  13.0 / TEXT, ["fig3_ftle_stadium", "fig3_ftle_sinai"]),
    (mp.fig_phase_separation,      11.0 / TEXT, ["fig4_phase_separation"]),
    (mp.fig_chaotic_fraction,      7.0 / COL,   ["fig5_chaotic_fraction"]),
    (mp.fig_geometry_scan,         7.0 / COL,   ["fig8_geometry_scan"]),
    (mp.fig_universality_collapse, 7.0 / COL,   ["fig8b_universality_collapse"]),
    (mp.fig_lcn_traces,            11.0 / TEXT, ["fig10_lcn_traces"]),
    (mp.fig_energy,                11.0 / TEXT, ["fig7_energy"]),
    (mp.fig_collision_rate,        7.0 / COL,   ["figA_collision_rate"]),
    (mp.fig_dh_scaling,            7.0 / COL,   ["figA_dh_scaling"]),
    (mp.fig_lyapunov_spectrum,     12.0 / TEXT, ["figA_lyapunov_spectrum"]),
]


def set_scaled(k):
    plt.rcParams.update({
        "font.size": 8.5 * k, "axes.titlesize": 9.0 * k,
        "axes.labelsize": 9.0 * k, "xtick.labelsize": 8.0 * k,
        "ytick.labelsize": 8.0 * k, "legend.fontsize": 7.5 * k,
    })


def snapshot(names):
    os.makedirs(BACKUP, exist_ok=True)
    for n in names:
        for ext in ("pdf", "png"):
            p = f"plots_v2/{n}.{ext}"
            if os.path.exists(p):
                shutil.copy2(p, f"{BACKUP}/{n}.{ext}")


def rename_and_restore(names):
    for n in names:
        for ext in ("pdf", "png"):
            p = f"plots_v2/{n}.{ext}"
            if os.path.exists(p):
                os.replace(p, f"plots_v2/{n}_v2.{ext}")
            b = f"{BACKUP}/{n}.{ext}"
            if os.path.exists(b):
                shutil.copy2(b, p)


def fig14_v2():
    """R-scan replot from the saved npz at column size."""
    set_scaled(1.0)
    d = np.load("experiment_results/sinai_R_scan.npz")
    fig, ax = plt.subplots(figsize=(3.4, 2.7))
    Rs = d["R_values"]
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(Rs)))
    for c, R in zip(colors, Rs):
        frac = float(np.mean(d[f"R{R}_curved_frac"]))
        ax.errorbar(d["alpha_vals"], d[f"R{R}_means"], yerr=d[f"R{R}_sems"],
                    fmt="o-", ms=2, lw=1.0, capsize=1.2, elinewidth=0.5,
                    color=c, label=f"$R={R}$ ({frac:.0%})")
    ax.set_xlabel(r"Spin coupling $\alpha$")
    ax.set_ylabel(r"Lyapunov exponent $\lambda$")
    ax.legend(ncol=2, columnspacing=0.9, handletextpad=0.5)
    ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"plots_v2/fig14_sinai_R_scan_v2.{ext}")
    plt.close(fig)
    print("  -> fig14_sinai_R_scan_v2.pdf")


def main():
    for func, k, names in JOBS:
        t0 = time.perf_counter()
        snapshot(names)
        set_scaled(k)
        func()
        rename_and_restore(names)
        print(f"  [{func.__name__}] scale {k:.2f} -> "
              f"{', '.join(n + '_v2' for n in names)} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    fig14_v2()
    print("ALL FIGURES REGENERATED")


if __name__ == "__main__":
    main()
