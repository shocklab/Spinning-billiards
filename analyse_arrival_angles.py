"""Arrival-angle distributions at curved walls, across the ablation modes.

Referee 2 asks for "analysis of any actual change in defocusing or dispersing".
The per-collision expansion of a dispersing/defocusing wall scales with
1/cos(theta) at incidence angle theta, so if the flat-wall spin coupling
suppresses chaos by reshaping the angles at which the particle arrives at the
curved boundary, that must show up in P(cos theta).

The alternative channel is that flat-wall collisions rotate the tangent-space
perturbation into the u~ direction, which curvature cannot amplify. That channel
operates even if the arrival-angle distribution does not move at all.

This script measures P(|cos theta|) at curved-wall collisions for the four
ablation variants (full / curved-only / flat-only / none) on identical initial
conditions, and summarises with <sec theta> and the grazing fraction
P(|cos theta| < 0.1). If flat-only leaves the distribution at the spinless one
while lambda halves, the angle-metering story is refuted and the
tangent-rotation channel carries the suppression.
"""

import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compute_spin_ablation import (
    MODES, GEOMETRIES, draw_ics, ablation_ensemble_ics, N_THETA_BINS,
)

ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
N_ICS = 1024
N_STEPS = 50_000
UMF = 0.5
SEED = 4242

COLORS = {"none": "#888888", "flat": "#0057b7",
          "curved": "#c1272d", "full": "#1b1b1b"}
LABELS = {"none": "spinless", "flat": "spin at flat walls only",
          "curved": "spin at curved walls only", "full": "full spin"}


def run():
    centers = (np.arange(N_THETA_BINS) + 0.5) / N_THETA_BINS   # |cos theta|
    out = {}
    for gname, (geo, p1, p2) in GEOMETRIES.items():
        if gname == "Rectangle":
            continue
        for ia, a in enumerate(ALPHAS):
            ics = draw_ics(N_ICS, geo, p1, p2, a, UMF, SEED + ia)
            for mname, mcode in MODES.items():
                lam, rc, fc, hists = ablation_ensemble_ics(
                    N_STEPS, ics, a, mcode, geo, p1, p2, 1e-7)
                h = hists.sum(axis=0)
                h = h / h.sum()
                sec = float((h / np.maximum(centers, centers[0])).sum())
                graz = float(h[centers < 0.1].sum())
                out[(gname, a, mname)] = dict(
                    hist=h, lam=float(np.nanmean(lam)),
                    sec=sec, grazing=graz)
                print(f"  {gname:8s} a={a:.2f} {mname:7s} "
                      f"lam={out[(gname,a,mname)]['lam']:.5f} "
                      f"<sec>={sec:7.3f}  P(graz)={graz:.4f}", flush=True)
    return centers, out


def figure(centers, out):
    geos = ["Sinai", "Stadium"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    for col, g in enumerate(geos):
        ax = axes[0, col]
        for m in ("none", "curved", "flat", "full"):
            h = out[(g, 1.0, m)]["hist"]
            ax.step(centers, h * N_THETA_BINS, where="mid",
                    color=COLORS[m], lw=1.4, label=LABELS[m])
        ax.set_xlabel(r"$|\cos\theta|$ at curved-wall collisions")
        ax.set_ylabel("density")
        ax.set_title(f"{g}, $\\alpha = 1$")
        if col == 0:
            ax.legend(fontsize=8)

        ax = axes[1, col]
        for m in ("none", "curved", "flat", "full"):
            xs = [a for a in ALPHAS]
            ys = [out[(g, a, m)]["sec"] for a in ALPHAS]
            ax.plot(xs, ys, "o-", ms=4, lw=1.3, color=COLORS[m],
                    label=LABELS[m])
        ax.set_xlabel(r"spin coupling $\alpha$")
        ax.set_ylabel(r"$\langle 1/\cos\theta \rangle$ at curved walls")
        ax.set_title(f"{g}: mean expansion-strength factor")

    fig.suptitle("Arrival-angle statistics at the chaos-generating walls, "
                 "by ablation variant", fontsize=12)
    fig.tight_layout()
    os.makedirs("plots_v2", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"plots_v2/fig_arrival_angles.{ext}", dpi=180)
    print("wrote plots_v2/fig_arrival_angles.pdf")


def save(centers, out):
    keys = sorted(out.keys())
    np.savez("experiment_results/arrival_angles.npz",
             centers=centers,
             keys=np.array([f"{g}|{a}|{m}" for g, a, m in keys]),
             hists=np.array([out[k]["hist"] for k in keys]),
             lam=np.array([out[k]["lam"] for k in keys]),
             sec=np.array([out[k]["sec"] for k in keys]),
             grazing=np.array([out[k]["grazing"] for k in keys]),
             n_ics=N_ICS, n_steps=N_STEPS, u_max_frac=UMF, seed=SEED)


if __name__ == "__main__":
    centers, out = run()
    save(centers, out)
    figure(centers, out)
