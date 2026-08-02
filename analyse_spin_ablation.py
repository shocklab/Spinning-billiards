"""Summarise the spin-ablation sweep and draw the referee-response figure.

Reads experiment_results/spin_ablation/*.npz (written by compute_spin_ablation.py)
and answers, for each geometry:

  1. Does spin at flat walls alone suppress chaos?  (referees say it cannot)
  2. Does spin at curved walls alone suppress chaos?
  3. Are the two effects additive?
  4. Is any of it robust to the initial-condition sampling (u_max_frac)?

Baseline throughout is lambda_none(alpha), the spinless dynamics run on the
same alpha-dependent initial conditions. That control absorbs the translational
speed reduction built into the IC sampler (v = sqrt(1 - alpha*u^2)), which is
worth ~4% of lambda at alpha = 1 and is NOT a spin effect.
"""

import glob
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATADIR = "experiment_results/spin_ablation"
OUTDIR = "plots_v2"
COLORS = {"full": "#1b1b1b", "curved": "#c1272d",
          "flat": "#0057b7", "none": "#888888"}
LABELS = {"full": r"full spin",
          "curved": r"spin at curved walls only",
          "flat": r"spin at flat walls only",
          "none": r"spinless control"}


def load():
    runs = {}
    for p in sorted(glob.glob(os.path.join(DATADIR, "spin_ablation_*.npz"))):
        m = re.match(r"spin_ablation_(\w+)_umf([\d.]+)_seed(\d+)\.npz",
                     os.path.basename(p))
        if not m:
            continue
        geo, umf, seed = m.group(1), float(m.group(2)), int(m.group(3))
        runs[(geo, umf)] = np.load(p)
    return runs


def table(runs):
    for (geo, umf), d in sorted(runs.items()):
        a = d["alpha"]
        print(f"\n=== {geo}  (u_max_frac = {umf}) ===")
        print(f"{'alpha':>6} {'none':>9} {'full':>9} {'flat':>9} {'curved':>9}"
              f" {'share_flat':>11} {'share_curv':>11}")
        for j in range(0, len(a), max(1, len(a) // 7)):
            print(f"{a[j]:6.2f} {d['lambda_none'][j]:9.5f} "
                  f"{d['lambda_full'][j]:9.5f} {d['lambda_flat'][j]:9.5f} "
                  f"{d['lambda_curved'][j]:9.5f} "
                  f"{d['share_flat'][j]:10.1%} {d['share_curved'][j]:10.1%}")
        j = -1
        sf, sc = d["share_flat"][j], d["share_curved"][j]
        print(f"  at alpha=1: flat-only reproduces {sf:.0%} of the suppression, "
              f"curved-only {sc:.0%}  (sum {sf+sc:.0%})")
        dfull = d["drop_full"][j]
        print(f"  drop_full = {dfull:.5f} +/- {d['sem_drop_full'][j]:.5f}   "
              f"drop_flat = {d['drop_flat'][j]:.5f} +/- {d['sem_drop_flat'][j]:.5f}   "
              f"drop_curved = {d['drop_curved'][j]:.5f} +/- {d['sem_drop_curved'][j]:.5f}")


def figure(runs, umf=0.5):
    geos = [g for (g, u) in runs if u == umf and g != "rectangle"]
    geos = sorted(set(geos))
    if not geos:
        print("no data for that u_max_frac")
        return
    fig, axes = plt.subplots(1, len(geos), figsize=(5.2 * len(geos), 4.2),
                             squeeze=False)
    for ax, geo in zip(axes[0], geos):
        d = runs[(geo, umf)]
        a = d["alpha"]
        for m in ("none", "flat", "curved", "full"):
            ax.errorbar(a, d[f"lambda_{m}"], yerr=d[f"sem_lambda_{m}"],
                        fmt="o-", ms=3, lw=1.3, capsize=2, elinewidth=0.7,
                        color=COLORS[m], label=LABELS[m])
        ax.set_xlabel(r"spin coupling $\alpha$")
        ax.set_ylabel(r"Lyapunov exponent $\lambda$")
        ax.set_title(geo.capitalize())
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.15)
    axes[0][0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Wall-class ablation of the spin coupling", fontsize=11)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUTDIR, f"fig_spin_ablation.{ext}"), dpi=180)
    print(f"\nwrote {OUTDIR}/fig_spin_ablation.pdf")


def robustness(runs):
    print("\n=== robustness of the flat-only share to IC sampling ===")
    print(f"{'geometry':>10} {'u_max_frac':>11} {'share_flat(a=1)':>16} "
          f"{'share_curved(a=1)':>18} {'lambda_full(a=1)':>17}")
    for (geo, umf), d in sorted(runs.items()):
        if geo == "rectangle":
            continue
        print(f"{geo:>10} {umf:11.2f} {d['share_flat'][-1]:15.1%} "
              f"{d['share_curved'][-1]:17.1%} {d['lambda_full'][-1]:17.5f}")

    rect = [(u, d) for (g, u), d in runs.items() if g == "rectangle"]
    if rect:
        u, d = rect[0]
        print(f"\n=== rectangle control (no curved walls, u_max_frac={u}) ===")
        print(f"  lambda_full  max over alpha = {d['lambda_full'].max():.2e}")
        print(f"  lambda_flat  max over alpha = {d['lambda_flat'].max():.2e}")
        print("  (both must be ~0: flat walls with spin generate no chaos)")


if __name__ == "__main__":
    runs = load()
    if not runs:
        raise SystemExit(f"no npz files in {DATADIR}")
    table(runs)
    robustness(runs)
    figure(runs, umf=0.5)
