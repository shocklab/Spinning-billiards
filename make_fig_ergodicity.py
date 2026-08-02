"""Figure for the Kac appendix notes: the Birkhoff ergodicity test.

Left: spread of per-orbit visit frequency over the Poisson sampling floor.
Right: per-orbit Kac estimates mu(S)<tau_S> for box 1 at three alphas, with
the alpha = 0.5 cluster masses annotated (76% above 0.9, 23% below 0.3).
Reads experiment_results/polish/master_kac.npz only.
"""
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 8.5, "axes.labelsize": 9, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "legend.fontsize": 7.5})
d = np.load("experiment_results/polish/master_kac.npz")
hits, mus, al = d["hits"], d["mu_S"], np.asarray(d["alpha"], float)
N = 2e6
fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.7))

ax = axes[0]
cols = ["#0057b7", "#c1272d", "#3a7d44"]
for bi in range(3):
    cv, fl = [], []
    for ai in range(len(al)):
        h = hits[ai, bi]; h = h[np.isfinite(h) & (h > 0)]
        cv.append(h.std() / h.mean()); fl.append(1 / math.sqrt(h.mean()))
    ax.semilogy(al, np.array(cv) / np.array(fl), "-", lw=1.2, color=cols[bi],
                label=f"box {bi+1}")
ax.axhline(1, color="0.35", ls="--", lw=0.8)
ax.text(0.30, 1.35, "single-component floor", fontsize=6.5, color="0.35")
ax.set_xlabel(r"Spin coupling $\alpha$")
ax.set_ylabel("spread / sampling floor")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.5, 2000)
ax.legend(handlelength=1.4, labelspacing=0.3, loc="center right")

ax = axes[1]
for a, c, lab in ((0.05, "#0057b7", r"$\alpha=0.05$"),
                  (0.50, "#c1272d", r"$\alpha=0.50$"),
                  (1.00, "#3a7d44", r"$\alpha=1.00$")):
    ai = int(np.argmin(np.abs(al - a)))
    h = hits[ai, 0]; h = h[np.isfinite(h) & (h > 0)]
    per = mus[ai, 0] * N / h
    ax.hist(per, bins=np.linspace(0, 1.4, 90), histtype="step", lw=1.3,
            color=c, label=lab, density=True)
ax.text(1.13, 22, "76%", color="#c1272d", ha="center", fontsize=8)
ax.text(0.10, 5.5, "23%", color="#c1272d", ha="center", fontsize=8)
ax.set_xlabel(r"per-orbit $\mu(S)\,\langle\tau_S\rangle$  (box 1)")
ax.set_ylabel("density")
ax.set_xlim(0, 1.4); ax.set_yscale("log")
ax.legend(handlelength=1.4, labelspacing=0.3)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"plots_v2/figA_ergodicity_test.{ext}", dpi=200)
print("wrote plots_v2/figA_ergodicity_test.pdf/.png")
