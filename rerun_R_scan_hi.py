"""High-statistics rerun of the Sinai obstacle-radius scan (paper Fig. 12).

The submitted figure used 500 initial conditions per (R, alpha) point, the
weakest ensemble in the paper, and it feeds the one feature the text flags as
not fully understood (the non-monotonic recovery at R = 1.5). This rerun uses
8,192 ICs per point on the same alpha grid, a 16x variance reduction, writing
experiment_results/sinai_R_scan_hi.npz (the original npz is left untouched)
and replotting plots_v2/fig14_sinai_R_scan_v2 at print size.
"""

import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spinning_billiards import lyapunov_ensemble, SINAI

N_ICS = 8192
N_STEPS = 50_000
L = 2.0


def run():
    old = np.load("experiment_results/sinai_R_scan.npz")
    alpha_vals = old["alpha_vals"]
    R_values = old["R_values"]
    out = {"alpha_vals": alpha_vals, "R_values": R_values,
           "n_ics": N_ICS, "n_steps": N_STEPS}
    for R in R_values:
        t0 = time.perf_counter()
        means = np.empty(len(alpha_vals))
        sems = np.empty(len(alpha_vals))
        for j, a in enumerate(alpha_vals):
            lc = lyapunov_ensemble(N_STEPS, N_ICS, a, SINAI, L, float(R),
                                   perturb_mag=1e-7, u_max_frac=0.5)
            lc = lc[np.isfinite(lc)]
            means[j] = lc.mean()
            sems[j] = lc.std() / math.sqrt(len(lc))
        out[f"R{R}_means"] = means
        out[f"R{R}_sems"] = sems
        out[f"R{R}_curved_frac"] = old[f"R{R}_curved_frac"]
        print(f"  R={R}: done in {time.perf_counter()-t0:.0f}s "
              f"(lambda(0)={means[0]:.4f}, lambda(1)={means[-1]:.4f})",
              flush=True)
    np.savez("experiment_results/sinai_R_scan_hi.npz", **out)
    print("wrote experiment_results/sinai_R_scan_hi.npz")
    return out


def plot(d):
    plt.rcParams.update({
        "font.size": 8.5, "axes.titlesize": 9.0, "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 7.5,
    })
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
        fig.savefig(f"plots_v2/fig14_sinai_R_scan_v2.{ext}", dpi=200)
    print("replotted plots_v2/fig14_sinai_R_scan_v2.pdf")


if __name__ == "__main__":
    plot(run())
