"""Replot the manuscript's data figures from the HPC master files.

Final-campaign edition (2026-07-11): masters now hold the 10x datasets
(1e6 ICs on the 41-alpha grids, 101-alpha FTLE, 7 Sinai radii at 655k ICs,
65,536-IC spectra, 512-orbit Kac boxes on the 41-alpha grid) plus the
fine sqrt-ramped alpha grid in (0, 0.05) for the initial-drop knee
(master_*_fine.npz). Curves are drawn on the merged coarse+fine grid.

Outputs (plots_v2/): fig2_lyapunov_vs_alpha_v2, fig5_chaotic_fraction_v2,
fig_spin_ablation, fig_arrival_angles, fig8_geometry_scan_v2,
fig8b_universality_collapse_v2, fig14_sinai_R_scan_v2,
figA_lyapunov_spectrum_v2, figA_kac_fragmentation, fig_island_prediction.

Figures are drawn at their rendered print width with fonts at final size.
Every number quoted in the revised text is printed by stats().
"""

import math
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
POLISH = os.path.join(ROOT, "experiment_results", "polish")
RESULTS = os.path.join(ROOT, "experiment_results")
OUT = os.path.join(ROOT, "plots_v2")

COL = 3.375                     # \columnwidth
W80 = 0.80 * 7.05               # 0.80\textwidth
W85 = 0.85 * 7.05               # 0.85\textwidth

MODE_COLORS = {"full": "#1b1b1b", "curved": "#c1272d",
               "flat": "#0057b7", "none": "#888888"}
MODE_LABELS = {"full": "full spin", "curved": "spin at curved walls only",
               "flat": "spin at flat walls only", "none": "spinless control"}
GEO_TITLES = {"sinai": "Sinai", "stadium": "Stadium"}
GEO_COLORS = {"Circle": "#2176AE", "Rectangle": "#F57C20",
              "Stadium": "#57A773", "Sinai": "#D33F49"}

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9.0, "axes.labelsize": 9.0,
    "xtick.labelsize": 8.0, "ytick.labelsize": 8.0, "legend.fontsize": 7.0,
})


def _save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print(f"  -> plots_v2/{name}.pdf")


def load(name):
    return np.load(os.path.join(POLISH, f"master_{name}.npz"))


def try_load(name):
    p = os.path.join(POLISH, f"master_{name}.npz")
    return np.load(p) if os.path.exists(p) else None


def merged_lam_stats(d_main, d_fine):
    """Per-alpha mean/sem of lam for main + fine masters, merged and sorted.
    lam axis layout: (..., alpha, ic). Returns (alpha, mean, sem)."""
    outs = []
    for d in (d_main, d_fine):
        if d is None:
            continue
        lam = d["lam"]
        m = np.nanmean(lam, axis=-1)
        s = np.nanstd(lam, axis=-1) / math.sqrt(lam.shape[-1])
        outs.append((np.asarray(d["alpha"], float), m, s))
    al = np.concatenate([o[0] for o in outs])
    mean = np.concatenate([o[1] for o in outs], axis=-1)
    sem = np.concatenate([o[2] for o in outs], axis=-1)
    order = np.argsort(al)
    return al[order], mean[..., order], sem[..., order]


# ----------------------------------------------------------------- fig 2
def fig2(d, dfine):
    al, mean, sem = merged_lam_stats(d, dfine)      # mean[geo, alpha]
    fig, ax = plt.subplots(figsize=(COL, 2.8))
    pos = al > 0            # the exactly-spinless ensemble is its own limit;
    # lambda(0) nearly coincides for the two geometries, so draw the stadium
    # marker as a ring behind the Sinai dot
    for gi, gname, mk in ((1, "Stadium", dict(ms=6.0, mfc="none", mew=1.2)),
                          (0, "Sinai", dict(ms=3.2))):
        c = GEO_COLORS[gname]
        ax.plot(al[pos], mean[gi][pos], "-", lw=1.3, color=c, label=gname)
        ax.plot([0.0], [mean[gi][~pos][0]], "o", color=c, **mk)
        ax.fill_between(al[pos], (mean[gi] - sem[gi])[pos],
                        (mean[gi] + sem[gi])[pos], color=c, alpha=0.3, lw=0)
    legacy = np.load(os.path.join(RESULTS, "lyapunov_sweep_data.npz"))
    for gname in ("Circle", "Rectangle"):
        ax.plot(legacy["alpha_values"], legacy[f"{gname}_mean"], "-",
                lw=1.1, color=GEO_COLORS[gname],
                label=f"{gname} ($\\lambda \\approx 0$)")
    ax.axhline(0, color="gray", ls="--", lw=0.5, zorder=0)
    ax.set_xlabel(r"Spin coupling $\alpha$")
    ax.set_ylabel(r"Lyapunov exponent $\lambda$")
    ax.legend(loc="upper right", handlelength=1.5, labelspacing=0.3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.50)
    fig.tight_layout()
    _save(fig, "fig2_lyapunov_vs_alpha_v2")
    return al, mean, sem


# ----------------------------------------------------------------- fig 5
def fig5(d, dfine):
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
    fig, ax = plt.subplots(figsize=(COL, 2.8))
    fch_at = {}
    for gi, gname in ((1, "Stadium"), (0, "Sinai")):
        als, fr = [], {th: [] for th in thresholds}
        for dd in (d, dfine):
            if dd is None:
                continue
            lam = dd["lam"][gi]
            for j, a in enumerate(np.asarray(dd["alpha"], float)):
                x = lam[j]
                x = x[np.isfinite(x)]
                als.append(a)
                for th in thresholds:
                    fr[th].append(np.mean(x > th) * 100)
        order = np.argsort(als)
        als = np.asarray(als)[order]
        fr = {th: np.asarray(v)[order] for th, v in fr.items()}
        c = GEO_COLORS[gname]
        lo = np.minimum.reduce([fr[th] for th in thresholds])
        hi = np.maximum.reduce([fr[th] for th in thresholds])
        ax.fill_between(als, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(als, fr[0.01], "-", color=c, lw=1.5, label=gname)
        fch_at[gname] = (als, fr[0.01])
    ax.set_xlabel(r"Spin coupling $\alpha$")
    ax.set_ylabel("Chaotic fraction (%)")
    ax.set_ylim(55, 102)
    ax.set_xlim(-0.02, 1.02)
    ax.legend(loc="lower left", handlelength=1.5, labelspacing=0.3)
    fig.tight_layout()
    _save(fig, "fig5_chaotic_fraction_v2")
    return fch_at


# ----------------------------------------------------------------- ablation
def fig_ablation(d, dfine):
    al, mean, sem = merged_lam_stats(d, dfine)      # mean[geo, mode, alpha]
    pos = al > 0
    i0 = int(np.argmin(al))
    fig, axes = plt.subplots(1, 2, figsize=(W80, 2.55))
    for gi, g in enumerate(("sinai", "stadium")):
        ax = axes[gi]
        for m in ("none", "flat", "curved", "full"):
            mj = ["full", "curved", "flat", "none"].index(m)
            ax.plot(al[pos], mean[gi, mj][pos], "-", lw=1.2,
                    color=MODE_COLORS[m], label=MODE_LABELS[m])
            ax.fill_between(al[pos], (mean[gi, mj] - sem[gi, mj])[pos],
                            (mean[gi, mj] + sem[gi, mj])[pos],
                            color=MODE_COLORS[m], alpha=0.25, lw=0)
        # at alpha = 0 every variant is the same spinless billiard: one marker
        ax.plot([0.0], [mean[gi, 0, i0]], "o", ms=4.0, mfc="none",
                mec="k", mew=1.1)
        ax.set_xlabel(r"spin coupling $\alpha$")
        if gi == 0:
            ax.set_ylabel(r"Lyapunov exponent $\lambda$")
        ax.set_title(GEO_TITLES[g])
        ax.set_xlim(-0.02, 1.02)
        ax.grid(alpha=0.15)
    axes[0].legend(loc="upper right", handlelength=1.6,
                   borderaxespad=0.4, labelspacing=0.3)
    fig.tight_layout()
    _save(fig, "fig_spin_ablation")
    return al, mean, sem


# ----------------------------------------------------------- arrival angles
def fig_arrival(d, dfine):
    hists, als = [], []
    for dd in (d, dfine):
        if dd is None:
            continue
        hists.append(dd["hist"])                    # [geo, mode, alpha, 200]
        als.append(np.asarray(dd["alpha"], float))
    al = np.concatenate(als)
    hist = np.concatenate(hists, axis=2)
    order = np.argsort(al)
    al = al[order]
    hist = hist[:, :, order]

    centers = (np.arange(200) + 0.5) / 200
    sec_b = 1.0 / np.maximum(centers, 1 / 80.0)
    pooled = (hist * sec_b).sum(axis=3) / hist.sum(axis=3)
    i1 = int(np.argmin(np.abs(al - 1.0)))
    reb = hist[:, :, i1].reshape(2, 4, 50, 4).sum(axis=3)
    cen50 = (np.arange(50) + 0.5) / 50

    fig, axes = plt.subplots(2, 2, figsize=(W80, 4.6))
    for gi, g in enumerate(("sinai", "stadium")):
        ax = axes[0, gi]
        for m in ("none", "curved", "flat", "full"):
            mj = ["full", "curved", "flat", "none"].index(m)
            h = reb[gi, mj] / reb[gi, mj].sum()
            ax.step(cen50, h * 50, where="mid", color=MODE_COLORS[m],
                    lw=1.1, label=MODE_LABELS[m])
        ax.set_xlabel(r"$|\cos\theta|$ at curved-wall collisions")
        if gi == 0:
            ax.set_ylabel("density")
            ax.legend(handlelength=1.6, labelspacing=0.3)
        ax.set_title(f"{GEO_TITLES[g]}, $\\alpha = 1$")

        ax = axes[1, gi]
        for m in ("none", "curved", "flat", "full"):
            mj = ["full", "curved", "flat", "none"].index(m)
            ax.plot(al, pooled[gi, mj], "-", lw=1.2, color=MODE_COLORS[m],
                    label=MODE_LABELS[m])
        ax.axhline(math.pi / 2, color="0.4", ls="--", lw=0.7, zorder=0)
        ax.set_ylim(math.pi / 2 * 0.95, math.pi / 2 * 1.05)
        ax.set_xlabel(r"spin coupling $\alpha$")
        if gi == 0:
            ax.set_ylabel(r"$\langle 1/\cos\theta \rangle$ at curved walls")
        ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, "fig_arrival_angles")
    return al, pooled, hist


# ------------------------------------------------------------ geometry scan
def fig_geoscan(d, dfine):
    ag = d["keys"]
    al, mean, sem = merged_lam_stats(d, dfine)      # mean[a_geo, alpha]
    colors = plt.cm.tab10(np.linspace(0, 0.5, len(ag)))
    pos = al > 0
    i0 = int(np.argmin(al))            # alpha = 0 lives on the coarse grid
    assert al[i0] == 0.0

    fig, ax = plt.subplots(figsize=(COL, 2.7))
    for k, a_geo in enumerate(ag):
        ax.plot(al[pos], mean[k][pos], "-", lw=1.2, color=colors[k],
                label=f"$a = {a_geo:g}$")
        ax.fill_between(al[pos], (mean[k] - sem[k])[pos],
                        (mean[k] + sem[k])[pos], color=colors[k],
                        alpha=0.3, lw=0)
        ax.plot([0.0], [mean[k, i0]], "o", ms=3.6, mfc="none",
                mec=colors[k], mew=1.0)
    ax.set_xlabel(r"Spin coupling $\alpha$")
    ax.set_ylabel(r"Lyapunov exponent $\lambda$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(handlelength=1.4, labelspacing=0.3)
    fig.tight_layout()
    _save(fig, "fig8_geometry_scan_v2")

    fig, ax = plt.subplots(figsize=(COL, 2.7))
    for k, a_geo in enumerate(ag):
        ax.plot(al[pos], (mean[k] / mean[k, i0])[pos], "-", lw=1.2,
                color=colors[k], label=f"$a = {a_geo:g}$")
    ax.plot([0.0], [1.0], "o", ms=3.6, mfc="none", mec="k", mew=1.0)
    ax.axhline(1, color="gray", lw=0.5, alpha=0.4)
    ax.set_xlabel(r"Spin coupling $\alpha$")
    ax.set_ylabel(r"$\lambda(\alpha) / \lambda(0)$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="lower left", handlelength=1.4, labelspacing=0.3)
    fig.tight_layout()
    _save(fig, "fig8b_universality_collapse_v2")
    return al, mean, sem


# ----------------------------------------------------------------- R scan
# Measured directly (600 orbits x 3500 collisions x alpha in {0, 0.5, 1});
# matches the obstacle's boundary-length share 2*pi*R/(16 + 2*pi*R), as the
# uniform arclength marginal of the invariant measure requires. The values
# previously stored in sinai_R_scan*.npz were wrong at both ends.
CURVED_FRAC = {0.3: 0.11, 0.5: 0.17, 0.8: 0.25, 1.0: 0.29, 1.2: 0.33,
               1.35: 0.35, 1.5: 0.37}


def fig_rscan(d, dfine):
    Rs = d["keys"]
    al, mean, sem = merged_lam_stats(d, dfine)      # mean[R, alpha]
    colors = plt.cm.viridis(np.linspace(0.0, 0.85, len(Rs)))
    pos = al > 0
    i0 = int(np.argmin(al))
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    for k, R in enumerate(Rs):
        ax.plot(al[pos], mean[k][pos], "-", lw=1.2, color=colors[k],
                label=f"$R={R:g}$ ({CURVED_FRAC[float(R)]:.0%})")
        ax.fill_between(al[pos], (mean[k] - sem[k])[pos],
                        (mean[k] + sem[k])[pos], color=colors[k],
                        alpha=0.3, lw=0)
        ax.plot([0.0], [mean[k, i0]], "o", ms=3.4, mfc="none",
                mec=colors[k], mew=1.0)
    ax.set_xlabel(r"Spin coupling $\alpha$")
    ax.set_ylabel(r"Lyapunov exponent $\lambda$")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(ncol=2, columnspacing=0.9, handletextpad=0.5,
              handlelength=1.4, labelspacing=0.3)
    fig.tight_layout()
    _save(fig, "fig14_sinai_R_scan_v2")
    return al, mean, sem


# --------------------------------------------------------------- spectrum
def fig_spectrum(d):
    al = d["alpha"]
    sp = d["spectra"]                                # [geo, alpha, ic, 5]
    colors = ["#D33F49", "#FF9F1C", "#2EC4B6", "#3A86FF"]
    labels = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$", r"$\lambda_4$"]
    fig, axes = plt.subplots(1, 2, figsize=(W85, 2.6))
    means = {}
    for col, g in enumerate(("stadium", "sinai")):
        gi = ["sinai", "stadium"].index(g)
        m = np.nanmean(sp[gi], axis=1)
        m = np.sort(m, axis=1)[:, ::-1]
        means[g] = m
        ax = axes[col]
        posm = np.asarray(al, float) > 0
        for k in range(4):
            ax.plot(np.asarray(al)[posm], m[posm, k], "-", lw=1.2,
                    color=colors[k], label=labels[k])
            ax.plot([0.0], [m[~posm, k][0]], "o", ms=2.4, color=colors[k])
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.set_xlabel(r"$\alpha$ (spin coupling)")
        if col == 0:
            ax.set_ylabel("Lyapunov exponent")
        ax.set_title(GEO_TITLES[g])
        ax.legend(ncol=2, handlelength=1.4, labelspacing=0.3,
                  columnspacing=0.9)
        ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    _save(fig, "figA_lyapunov_spectrum_v2")
    return means


# -------------------------------------------------------------------- Kac
def fig_kac(d):
    al, hits, mus = np.asarray(d["alpha"], float).copy(), d["hits"], d["mu_S"]
    al[al == 0.0] = 0.02      # the alpha = 0 cell is run at 0.02: the box's
    nc = float(d["n_coll"])   # u-extent is set by u_max, divergent at 0
    n_al = len(al)
    est = np.empty((n_al, 3))
    sem = np.empty_like(est)
    for ai in range(n_al):
        for bi in range(3):
            h = hits[ai, bi]
            h = h[np.isfinite(h) & (h > 0)]
            per = mus[ai, bi] * nc / h
            est[ai, bi] = mus[ai, bi] * nc / h.mean()
            sem[ai, bi] = per.std() / math.sqrt(len(per))
    fig, ax = plt.subplots(figsize=(COL, 2.7))
    box_cols = ["#0057b7", "#c1272d", "#3a7d44"]
    for bi in range(3):
        ax.plot(al, est[:, bi], "-", lw=1.2, color=box_cols[bi],
                label=f"box {bi + 1}")
        ax.fill_between(al, est[:, bi] - sem[:, bi], est[:, bi] + sem[:, bi],
                        color=box_cols[bi], alpha=0.3, lw=0)
    ax.axhline(1, color="0.4", ls="--", lw=0.7, zorder=0)
    ax.set_xlabel(r"Spin coupling $\alpha$")
    ax.set_ylabel(r"$f_{\rm acc} = \mu(S)\,\langle\tau_S\rangle$")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower left", handlelength=1.4, labelspacing=0.3)
    fig.tight_layout()
    _save(fig, "figA_kac_fragmentation")
    return est, sem


# ---------------------------------------------------- FTLE distributions
def fig_ftle_master(d):
    """FTLE histograms at six alpha values from the 1e6-IC master, keeping
    the layout of make_plots.fig_ftle (regenerated print version)."""
    show = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    al = np.asarray(d["alpha"], float)
    saved = {k: plt.rcParams[k] for k in
             ("font.size", "axes.titlesize", "axes.labelsize",
              "xtick.labelsize", "ytick.labelsize", "legend.fontsize")}
    k = 13.0 / 7.05
    plt.rcParams.update({
        "font.size": 8.5 * k, "axes.titlesize": 9.0 * k,
        "axes.labelsize": 9.0 * k, "xtick.labelsize": 8.0 * k,
        "ytick.labelsize": 8.0 * k, "legend.fontsize": 7.5 * k,
    })
    for gi, gname, color in ((1, "stadium", GEO_COLORS["Stadium"]),
                             (0, "sinai", GEO_COLORS["Sinai"])):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
        axes = axes.flatten()
        for j, a in enumerate(show):
            idx = int(np.argmin(np.abs(al - a)))
            lcns = d["lam"][gi, idx]
            lcns = lcns[np.isfinite(lcns)]
            ax = axes[j]
            ax.hist(lcns[lcns > -0.05], bins=80, density=True, alpha=0.55,
                    color=color, edgecolor="0.3", linewidth=0.3, zorder=2)
            ax.axvline(0, color="red", ls="--", lw=0.7, alpha=0.4, zorder=1)
            mean_v = float(np.mean(lcns))
            med_v = float(np.median(lcns))
            ch = lcns > 0.01
            cond = float(np.mean(lcns[ch])) if np.any(ch) else 0.0
            fch = float(np.mean(ch)) * 100
            ax.axvline(mean_v, color="black", ls="-", lw=1.0, alpha=0.7,
                       zorder=3, label="Mean" if j == 0 else "")
            ax.axvline(med_v, color="blue", ls="--", lw=1.0, alpha=0.7,
                       zorder=3, label="Median" if j == 0 else "")
            ax.axvline(cond, color="darkgreen", ls=":", lw=1.0, alpha=0.7,
                       zorder=3, label="Cond. mean" if j == 0 else "")
            ax.text(0.97, 0.95,
                    f"mean $= {mean_v:.3f}$\n"
                    f"median $= {med_v:.3f}$\n"
                    f"{fch:.0f}% chaotic",
                    transform=ax.transAxes, ha="right", va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="0.7", alpha=0.85))
            ax.set_xlabel(r"$\lambda$")
            if j % 3 == 0:
                ax.set_ylabel("Density")
            ax.set_title(f"$\\alpha = {a}$")
            print(f"  ftle {gname} a={a}: mean {mean_v:.3f} "
                  f"median {med_v:.3f} chaotic {fch:.1f}%")
        axes[0].legend(loc="lower left")
        fig.tight_layout()
        _save(fig, f"fig3_ftle_{gname}_v2")
    plt.rcParams.update(saved)


# ------------------------------------------------------------- DH scaling
def fig_dh(d, dfine):
    """lambda * f_chaotic product from the FTLE master (both quantities from
    the same per-IC data), threshold 0.01, merged coarse+fine grid."""
    fig, ax = plt.subplots(figsize=(COL, 2.8))
    for gi, gname in ((1, "Stadium"), (0, "Sinai")):
        als, lam_m, fch = [], [], []
        for dd in (d, dfine):
            if dd is None:
                continue
            lam = dd["lam"][gi]
            for j, a in enumerate(np.asarray(dd["alpha"], float)):
                x = lam[j]
                x = x[np.isfinite(x)]
                als.append(a)
                lam_m.append(x.mean())
                fch.append(np.mean(x > 0.01))
        order = np.argsort(als)
        als = np.asarray(als)[order]
        prod = (np.asarray(lam_m) * np.asarray(fch))[order]
        c = GEO_COLORS[gname]
        pos = als > 0
        ax.plot(als[pos], prod[pos], "-", lw=1.3, color=c, label=gname)
        mk = dict(ms=6.0, mfc="none", mew=1.2) if gname == "Stadium" \
            else dict(ms=3.2)
        ax.plot([0.0], [prod[~pos][0]], "o", color=c, **mk)
        print(f"  DH {gname}: product max/min = "
              f"{prod.max():.4f}/{prod.min():.4f} = {prod.max()/prod.min():.2f}")
    ax.set_xlabel(r"Spin coupling $\alpha$")
    ax.set_ylabel(r"$\lambda \cdot f_{\rm chaotic}$")
    ax.set_xlim(-0.02, 1.02)
    ax.legend(handlelength=1.5, labelspacing=0.3)
    fig.tight_layout()
    _save(fig, "figA_dh_scaling_v2")


# ----------------------------------------------------- island (right panel)
def fig_island(fch_stadium):
    """Rebuild fig_island_prediction: boundary panels from the stored npz,
    measured regular fraction from the new FTLE master."""
    z = np.load(os.path.join(RESULTS, "island_prediction.npz"))
    als, fch = fch_stadium                    # percent chaotic, threshold 0.01
    reg_al, reg = als, 1.0 - fch / 100.0

    fig, axes = plt.subplots(1, 2, figsize=(W80, 2.9))
    ax = axes[0]
    x0s, As = z["boundary_x0"], z["boundary_A"]
    ax.imshow(z["boundary_measured"].T, origin="lower", aspect="auto",
              extent=[x0s[0], x0s[-1], As[0], As[-1]],
              cmap="RdYlBu_r", alpha=0.75, vmin=0, vmax=1)
    ax.contour(x0s, As, z["boundary_predicted"].T, levels=[0.5],
               colors="k", linewidths=1.4)
    ax.set_xlabel(r"floor position $x_0$")
    ax.set_ylabel(r"amplitude $A=|(v_x,\tilde u)|$")

    ax = axes[1]
    ax.plot(reg_al, reg, "-", lw=1.3, color="#c1272d",
            label="FTLE regular fraction")
    ax.plot(z["alpha_mc"], z["predicted_regular"], "s-", ms=2.5, lw=1.1,
            color="#0057b7", label="predicted trapped fraction")
    ax.set_xlabel(r"spin coupling $\alpha$")
    ax.set_ylabel("regular fraction")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(bottom=0)
    ax.legend(handlelength=1.5, labelspacing=0.3)
    fig.tight_layout()
    _save(fig, "fig_island_prediction")


# ------------------------------------------------------------------- stats
def stats(abl, arr, geo, rsc, spec_means, kac_est, kac_sem,
          f2, fch_at, dgeo, drs):
    al_a, abl_mean, abl_sem = abl
    al_v, pooled, hist = arr
    al_g, geo_mean, geo_sem = geo
    al_r, r_mean, r_sem = rsc
    al_f, f2_mean, f2_sem = f2

    print("\n================ numbers for the revised text ================")

    i1 = int(np.argmin(np.abs(al_a - 1.0)))
    print("\n--- ablation, alpha = 1 (1e6 ICs) ---")
    for gi, g in enumerate(("sinai", "stadium")):
        none, full = abl_mean[gi, 3, i1], abl_mean[gi, 0, i1]
        flat, curv = abl_mean[gi, 2, i1], abl_mean[gi, 1, i1]
        drop = none - full
        print(f"  {g:8s} share_flat={(none - flat) / drop:.3f} "
              f"share_curved={(none - curv) / drop:.3f}   "
              f"lam_none/lam_full = {none / full:.2f}")

    print("\n--- fine-alpha knee (full coupling) ---")
    for gi, g in enumerate(("sinai", "stadium")):
        for t in (0.0, 0.0005, 0.0045, 0.0125, 0.0405):
            j = int(np.argmin(np.abs(al_a - t)))
            print(f"  {g:8s} lam({al_a[j]:.4f}) = {abl_mean[gi, 0, j]:.4f}", end="")
        print()

    print("\n--- arrival angles (pooled, 1e6) ---")
    dev = pooled / (math.pi / 2) - 1
    print(f"  max |<sec> - pi/2|/(pi/2): {np.abs(dev).max() * 100:.2f}%")
    graz = hist[..., :20].sum(axis=3) / hist.sum(axis=3)
    print(f"  grazing fraction: min {graz.min():.4f} max {graz.max():.4f}")

    print("\n--- geometry scan (a = 0.2 vs 0.5, 1e6) ---")
    diff = geo_mean[0] - geo_mean[1]
    sig = np.sqrt(geo_sem[0] ** 2 + geo_sem[1] ** 2)
    sgn = np.sign(diff)
    cross = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    for c in cross[-1:]:
        print(f"  crossing between alpha={al_g[c]:.3f} and {al_g[c + 1]:.3f}")
    i1g = int(np.argmin(np.abs(al_g - 1.0)))
    print(f"  at alpha=1: diff = {diff[i1g]:+.5f} +- {sig[i1g]:.5f} "
          f"({diff[i1g] / sig[i1g]:.0f} sigma)")
    print(f"  lam(0): a=0.2 {geo_mean[0, 0]:.4f}, a=0.5 {geo_mean[1, 0]:.4f}")

    print("\n--- R scan (655,360 ICs) ---")
    Rlist = [round(float(r), 2) for r in drs["keys"]]
    for R in (1.5, 1.35, 1.2, 1.0):
        k = Rlist.index(R)
        m, s = r_mean[k], r_sem[k]
        jmin = np.argmin(m)
        jmax = jmin + np.argmax(m[jmin:])
        rise = (m[jmax] - m[jmin]) / m[jmin]
        nsig = (m[jmax] - m[jmin]) / math.sqrt(s[jmin] ** 2 + s[jmax] ** 2)
        print(f"  R={R}: min {m[jmin]:.4f}({s[jmin] * 1e4:.0f}) at "
              f"alpha={al_r[jmin]:.3f}; max {m[jmax]:.4f} at "
              f"alpha={al_r[jmax]:.3f}; rise {rise * 100:.1f}% ({nsig:.0f} sigma)")

    print("\n--- spectrum (65,536 ICs) ---")
    for g in ("sinai", "stadium"):
        m = spec_means[g]
        print(f"  {g:8s} max|lam2| = {np.abs(m[:, 1]).max():.5f}   "
              f"max|lam1+lam4| = {np.abs(m[:, 0] + m[:, 3]).max():.6f}")

    print("\n--- Kac (512 orbits, 41 alpha) ---")
    a41 = np.round(np.linspace(0, 1, 41), 6)
    spread = kac_est.max(axis=1) - kac_est.min(axis=1)
    for t in (0.05, 0.10, 0.15):
        j = int(np.argmin(np.abs(a41 - t)))
        print(f"  alpha={t}: boxes {np.round(kac_est[j], 3)}")
    for ai in range(41):
        if spread[ai] > 3 * kac_sem[ai].max() and spread[ai] > 0.02:
            print(f"  onset: alpha={a41[ai]:.3f} (spread {spread[ai]:.3f})")
            break
    print(f"  max spread {spread.max():.3f} at alpha={a41[np.argmax(spread)]:.3f}")
    for t in (0.925, 0.95, 0.975, 1.0):
        j = int(np.argmin(np.abs(a41 - t)))
        print(f"  alpha={t}: boxes {np.round(kac_est[j], 3)} "
              f"spread {spread[j]:.3f}")

    print("\n--- fig2 / DH-table lambda + f_chaotic (exact grid points) ---")
    als, fch = fch_at["Sinai"]
    for t in (0.05, 0.10, 0.15, 1.00):
        j = int(np.argmin(np.abs(al_f - t)))
        jf = int(np.argmin(np.abs(als - t)))
        print(f"  alpha={t}: Sinai lam={f2_mean[0, j]:.4f} "
              f"f_ch={fch[jf] / 100:.3f}")
    print(f"  lam(1): Sinai {f2_mean[0, int(np.argmin(np.abs(al_f-1)))]:.4f}, "
          f"Stadium {f2_mean[1, int(np.argmin(np.abs(al_f-1)))]:.4f}")
    for g in ("Stadium", "Sinai"):
        als, fch = fch_at[g]
        j = int(np.argmin(np.abs(als - 1.0)))
        print(f"  f_ch(1.0) {g}: {fch[j]:.1f}%")


def main():
    dabl, dgeo, drs = load("ablang"), load("geoscan"), load("rscan")
    dsp, dkac, dftle = load("spectrum"), load("kac"), load("ftle")
    fabl, fgeo = try_load("ablang_fine"), try_load("geoscan_fine")
    frs, fftle = try_load("rscan_fine"), try_load("ftle_fine")

    f2 = fig2(dftle, fftle)
    fch_at = fig5(dftle, fftle)
    abl = fig_ablation(dabl, fabl)
    arr = fig_arrival(dabl, fabl)
    geo = fig_geoscan(dgeo, fgeo)
    rsc = fig_rscan(drs, frs)
    spec_means = fig_spectrum(dsp)
    kac_est, kac_sem = fig_kac(dkac)
    fig_island(fch_at["Stadium"])
    stats(abl, arr, geo, rsc, spec_means, kac_est, kac_sem,
          f2, fch_at, dgeo, drs)


if __name__ == "__main__":
    main()
