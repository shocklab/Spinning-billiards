"""Fetch the polish-campaign results from the HPC and build master npz files.

Run AFTER all spin2d arrays complete:

    python3 hpc_polish/fetch_and_aggregate.py            # rsync + aggregate
    python3 hpc_polish/fetch_and_aggregate.py --local    # aggregate only

Produces, under experiment_results/polish/:
    master_ablang.npz    lam/sec/graz[geo, mode, alpha, ic] float32 (memmapable),
                         hist[geo, mode, alpha, 200], alpha grid, meta
    master_geoscan.npz   lam[a_geo, alpha, ic]
    master_rscan.npz     lam[R, alpha, ic]
    master_spectrum.npz  spectra[geo, alpha, ic, 5]
    master_kac.npz       hits[alpha, box, orbit], mu_S[alpha, box]

Every master file records the SLURM manifest text and the per-task seeds, so
each number in the paper can be traced to a job id.
"""

import argparse
import glob
import os
import subprocess
import sys

import numpy as np

# Cluster location, taken from the environment so no account details are
# committed. Set these before fetching:
#   export SPIN2D_HOST=user@cluster.example.edu
#   export SPIN2D_REMOTE=/scratch/user/spin2d
HOST = os.environ.get("SPIN2D_HOST", "user@cluster.example.edu")
REMOTE = os.environ.get("SPIN2D_REMOTE", "/scratch/user/spin2d")
LOCAL = os.path.join(os.path.dirname(__file__), "..", "experiment_results", "polish")
MODES = ["full", "curved", "flat", "none"]
GEOS = ["sinai", "stadium"]
A41 = np.round(np.linspace(0, 1, 41), 6)
A21 = np.round(np.linspace(0, 1, 21), 6)


def fetch():
    os.makedirs(LOCAL, exist_ok=True)
    subprocess.run(["rsync", "-az",
                    f"{HOST}:{REMOTE}/results_ablang", f"{HOST}:{REMOTE}/results_geoscan",
                    f"{HOST}:{REMOTE}/results_rscan", f"{HOST}:{REMOTE}/results_spectrum",
                    f"{HOST}:{REMOTE}/results_kac", LOCAL], check=True)


def agg_ablang():
    d = os.path.join(LOCAL, "results_ablang")
    n_ic, n_chunk = 100_000, 4
    per = n_ic // n_chunk
    lam = np.full((2, 4, 41, n_ic), np.nan, np.float32)
    sec = np.full_like(lam, np.nan)
    grz = np.full_like(lam, np.nan)
    hist = np.zeros((2, 4, 41, 200))
    missing = []
    for gi, g in enumerate(GEOS):
        for ai in range(41):
            for c in range(n_chunk):
                f = f"{d}/ablang_{g}_a{ai:02d}_c{c}.npz"
                if not os.path.exists(f):
                    missing.append(os.path.basename(f)); continue
                z = np.load(f)
                sl = slice(c * per, (c + 1) * per)
                for mi, m in enumerate(MODES):
                    lam[gi, mi, ai, sl] = z[f"lam_{m}"]
                    sec[gi, mi, ai, sl] = z[f"sec_{m}"]
                    grz[gi, mi, ai, sl] = z[f"graz_{m}"]
                    hist[gi, mi, ai] += z[f"hist_{m}"]
    manifest = open(f"{d}/MANIFEST.txt").read() if os.path.exists(f"{d}/MANIFEST.txt") else ""
    np.savez_compressed(f"{LOCAL}/master_ablang.npz", lam=lam, sec=sec, graz=grz,
                        hist=hist, alpha=A41, geos=GEOS, modes=MODES,
                        manifest=manifest, missing=missing)
    print(f"ablang: {lam.size/4:.0f} slots, missing chunks: {len(missing)}")


def agg_simple(name, key_fmt, keys, n_ic, alphas, n_chunk=1):
    d = os.path.join(LOCAL, f"results_{name}")
    per = n_ic // n_chunk
    lam = np.full((len(keys), len(alphas), n_ic), np.nan, np.float32)
    missing = []
    for ki, k in enumerate(keys):
        for ai in range(len(alphas)):
            for c in range(n_chunk):
                f = f"{d}/{key_fmt.format(k=k, ai=ai, c=c)}"
                if not os.path.exists(f):
                    missing.append(os.path.basename(f)); continue
                z = np.load(f)
                lam[ki, ai, c * per:(c + 1) * per] = z["lam"]
    manifest = open(f"{d}/MANIFEST.txt").read() if os.path.exists(f"{d}/MANIFEST.txt") else ""
    np.savez_compressed(f"{LOCAL}/master_{name}.npz", lam=lam, keys=keys,
                        alpha=alphas, manifest=manifest, missing=missing)
    print(f"{name}: missing {len(missing)}")


def agg_spectrum():
    d = os.path.join(LOCAL, "results_spectrum")
    n_ic, n_chunk = 16_384, 8
    per = n_ic // n_chunk
    sp = np.full((2, 21, n_ic, 5), np.nan, np.float32)
    missing = []
    for gi, g in enumerate(GEOS):
        for ai in range(21):
            for c in range(n_chunk):
                f = f"{d}/spectrum_{g}_i{ai:02d}_c{c}.npz"
                if not os.path.exists(f):
                    missing.append(os.path.basename(f)); continue
                sp[gi, ai, c * per:(c + 1) * per] = np.load(f)["spectra"]
    manifest = open(f"{d}/MANIFEST.txt").read() if os.path.exists(f"{d}/MANIFEST.txt") else ""
    np.savez_compressed(f"{LOCAL}/master_spectrum.npz", spectra=sp, alpha=A21,
                        geos=GEOS, manifest=manifest, missing=missing)
    print(f"spectrum: missing {len(missing)}")


def agg_kac():
    d = os.path.join(LOCAL, "results_kac")
    hits = np.full((21, 3, 128), np.nan)
    mus = np.full((21, 3), np.nan)
    missing = []
    for ai in range(21):
        for bi in range(3):
            f = f"{d}/kac_i{ai:02d}_b{bi}.npz"
            if not os.path.exists(f):
                missing.append(os.path.basename(f)); continue
            z = np.load(f)
            hits[ai, bi] = z["hits"]
            mus[ai, bi] = z["mu_S"]
    np.savez_compressed(f"{LOCAL}/master_kac.npz", hits=hits, mu_S=mus,
                        alpha=A21, n_coll=2_000_000, missing=missing)
    print(f"kac: missing {len(missing)}")


# ----------------------------------------------------------------------
# Final ("go big") campaign of 2026-07-10 evening. Overwrites the master_*
# files with the 10x datasets; the morning per-task npz dirs remain intact.
# ----------------------------------------------------------------------

A101 = np.round(np.linspace(0, 1, 101), 6)
AFINE = np.round(0.05 * (np.arange(1, 10) / 10.0) ** 2, 6)
FINAL_DIRS = ["results_ablang10", "results_ftle", "results_geoscan10",
              "results_rscan10", "results_spectrum10", "results_kac10",
              "results_ablang_fine", "results_ftle_fine",
              "results_geoscan_fine", "results_rscan_fine"]


def fetch_final():
    os.makedirs(LOCAL, exist_ok=True)
    for d in FINAL_DIRS:
        subprocess.run(["rsync", "-az",
                        f"{HOST}:{REMOTE}/{d}", LOCAL], check=True)
        n = len(glob.glob(os.path.join(LOCAL, d, "*.npz")))
        print(f"fetched {d}: {n} npz")


def agg_ablang10():
    d = os.path.join(LOCAL, "results_ablang10")
    n_ic, n_chunk = 1_000_000, 20
    per = n_ic // n_chunk
    lam = np.full((2, 4, 41, n_ic), np.nan, np.float32)
    sec = np.full_like(lam, np.nan)
    grz = np.full_like(lam, np.nan)
    hist = np.zeros((2, 4, 41, 200))
    missing = []
    for gi, g in enumerate(GEOS):
        for ai in range(41):
            for c in range(n_chunk):
                f = f"{d}/ablang10_{g}_a{ai:02d}_c{c:02d}.npz"
                if not os.path.exists(f):
                    missing.append(os.path.basename(f)); continue
                z = np.load(f)
                sl = slice(c * per, (c + 1) * per)
                for mi, m in enumerate(MODES):
                    lam[gi, mi, ai, sl] = z[f"lam_{m}"]
                    sec[gi, mi, ai, sl] = z[f"sec_{m}"]
                    grz[gi, mi, ai, sl] = z[f"graz_{m}"]
                    hist[gi, mi, ai] += z[f"hist_{m}"]
    manifest = open(f"{d}/MANIFEST.txt").read() if os.path.exists(f"{d}/MANIFEST.txt") else ""
    np.savez_compressed(f"{LOCAL}/master_ablang.npz", lam=lam, sec=sec, graz=grz,
                        hist=hist, alpha=A41, geos=GEOS, modes=MODES,
                        manifest=manifest, missing=missing)
    print(f"ablang10: {lam.size/4:.0f} slots, missing chunks: {len(missing)}")


def agg_ftle():
    d = os.path.join(LOCAL, "results_ftle")
    n_ic, n_chunk = 1_000_000, 5
    per = n_ic // n_chunk
    lam = np.full((2, 101, n_ic), np.nan, np.float32)
    missing = []
    for gi, g in enumerate(GEOS):
        for ai in range(101):
            for c in range(n_chunk):
                f = f"{d}/ftle_{g}_i{ai:03d}_c{c}.npz"
                if not os.path.exists(f):
                    missing.append(os.path.basename(f)); continue
                lam[gi, ai, c * per:(c + 1) * per] = np.load(f)["lam"]
    manifest = open(f"{d}/MANIFEST.txt").read() if os.path.exists(f"{d}/MANIFEST.txt") else ""
    np.savez_compressed(f"{LOCAL}/master_ftle.npz", lam=lam, alpha=A101,
                        geos=GEOS, n_steps=50_000, manifest=manifest,
                        missing=missing)
    print(f"ftle: missing {len(missing)}")


def agg_spectrum10():
    d = os.path.join(LOCAL, "results_spectrum10")
    n_ic, n_chunk = 65_536, 16
    per = n_ic // n_chunk
    sp = np.full((2, 21, n_ic, 5), np.nan, np.float32)
    missing = []
    for gi, g in enumerate(GEOS):
        for ai in range(21):
            for c in range(n_chunk):
                f = f"{d}/spectrum10_{g}_i{ai:02d}_c{c:02d}.npz"
                if not os.path.exists(f):
                    missing.append(os.path.basename(f)); continue
                sp[gi, ai, c * per:(c + 1) * per] = np.load(f)["spectra"]
    manifest = open(f"{d}/MANIFEST.txt").read() if os.path.exists(f"{d}/MANIFEST.txt") else ""
    np.savez_compressed(f"{LOCAL}/master_spectrum.npz", spectra=sp, alpha=A21,
                        geos=GEOS, manifest=manifest, missing=missing)
    print(f"spectrum10: missing {len(missing)}")


def agg_kac10():
    d = os.path.join(LOCAL, "results_kac10")
    n_chunk, per = 4, 128
    hits = np.full((41, 3, n_chunk * per), np.nan)
    mus = np.full((41, 3), np.nan)
    missing = []
    for ai in range(41):
        for bi in range(3):
            for c in range(n_chunk):
                f = f"{d}/kac10_i{ai:02d}_b{bi}_c{c}.npz"
                if not os.path.exists(f):
                    missing.append(os.path.basename(f)); continue
                z = np.load(f)
                hits[ai, bi, c * per:(c + 1) * per] = z["hits"]
                mus[ai, bi] = z["mu_S"]
    np.savez_compressed(f"{LOCAL}/master_kac.npz", hits=hits, mu_S=mus,
                        alpha=A41, n_coll=2_000_000, missing=missing)
    print(f"kac10: missing {len(missing)}")


def agg_ablang_fine():
    d = os.path.join(LOCAL, "results_ablang_fine")
    n_ic, n_chunk = 1_000_000, 20
    per = n_ic // n_chunk
    lam = np.full((2, 4, 9, n_ic), np.nan, np.float32)
    sec = np.full_like(lam, np.nan)
    grz = np.full_like(lam, np.nan)
    hist = np.zeros((2, 4, 9, 200))
    missing = []
    for gi, g in enumerate(GEOS):
        for ai in range(9):
            for c in range(n_chunk):
                f = f"{d}/ablangf_{g}_a{ai}_c{c:02d}.npz"
                if not os.path.exists(f):
                    missing.append(os.path.basename(f)); continue
                z = np.load(f)
                sl = slice(c * per, (c + 1) * per)
                for mi, m in enumerate(MODES):
                    lam[gi, mi, ai, sl] = z[f"lam_{m}"]
                    sec[gi, mi, ai, sl] = z[f"sec_{m}"]
                    grz[gi, mi, ai, sl] = z[f"graz_{m}"]
                    hist[gi, mi, ai] += z[f"hist_{m}"]
    np.savez_compressed(f"{LOCAL}/master_ablang_fine.npz", lam=lam, sec=sec,
                        graz=grz, hist=hist, alpha=AFINE, geos=GEOS,
                        modes=MODES, missing=missing)
    print(f"ablang_fine: missing chunks: {len(missing)}")


def agg_ftle_fine():
    d = os.path.join(LOCAL, "results_ftle_fine")
    n_ic, n_chunk = 1_000_000, 5
    per = n_ic // n_chunk
    lam = np.full((2, 9, n_ic), np.nan, np.float32)
    missing = []
    for gi, g in enumerate(GEOS):
        for ai in range(9):
            for c in range(n_chunk):
                f = f"{d}/ftlef_{g}_i{ai}_c{c}.npz"
                if not os.path.exists(f):
                    missing.append(os.path.basename(f)); continue
                lam[gi, ai, c * per:(c + 1) * per] = np.load(f)["lam"]
    np.savez_compressed(f"{LOCAL}/master_ftle_fine.npz", lam=lam, alpha=AFINE,
                        geos=GEOS, n_steps=50_000, missing=missing)
    print(f"ftle_fine: missing {len(missing)}")


def aggregate_final():
    agg_ablang10()
    agg_ftle()
    agg_ablang_fine()
    agg_ftle_fine()
    agg_simple("geoscan_fine", "geoscanf_a{k}_i{ai}_c{c}.npz",
               [0.2, 0.5, 1.0, 2.0, 4.0], 1_000_000, AFINE, n_chunk=5)
    agg_simple("rscan_fine", "rscanf_R{k}_i{ai}_c{c}.npz",
               [0.3, 0.5, 0.8, 1.0, 1.2, 1.35, 1.5], 655_360, AFINE,
               n_chunk=4)
    agg_simple("geoscan10", "geoscan10_a{k}_i{ai:02d}_c{c}.npz",
               [0.2, 0.5, 1.0, 2.0, 4.0], 1_000_000, A41, n_chunk=5)
    # rename to the master the replot script reads
    os.replace(f"{LOCAL}/master_geoscan10.npz", f"{LOCAL}/master_geoscan.npz")
    agg_simple("rscan10", "rscan10_R{k}_i{ai:02d}_c{c}.npz",
               [0.3, 0.5, 0.8, 1.0, 1.2, 1.35, 1.5], 655_360, A41, n_chunk=4)
    os.replace(f"{LOCAL}/master_rscan10.npz", f"{LOCAL}/master_rscan.npz")
    agg_spectrum10()
    agg_kac10()
    print("FINAL masters written to", os.path.abspath(LOCAL))


def agg_rscan_combined():
    """Merge rscan10 (655,360 ICs) + rscan_top (344,640) into 1e6-IC masters,
    coarse and fine grids."""
    RS = [0.3, 0.5, 0.8, 1.0, 1.2, 1.35, 1.5]
    jobs = [("rscan", A41, "rscan10", "rscan10_R{k}_i{ai:02d}_c{c}.npz", 4,
             "rscan_top", "rscantop_R{k}_i{ai:02d}_c{c}.npz", 2),
            ("rscan_fine", AFINE, "rscan_fine", "rscanf_R{k}_i{ai}_c{c}.npz", 4,
             "rscanf_top", "rscanftop_R{k}_i{ai}_c{c}.npz", 2)]
    for mname, alphas, d1, f1, nc1, d2, f2, nc2 in jobs:
        lam = np.full((7, len(alphas), 1_000_000), np.nan, np.float32)
        missing = []
        for ki, k in enumerate(RS):
            for ai in range(len(alphas)):
                off = 0
                for dd, fmt, nch, per in ((d1, f1, nc1, 163_840),
                                          (d2, f2, nc2, 172_320)):
                    for c in range(nch):
                        f = os.path.join(LOCAL, f"results_{dd}",
                                         fmt.format(k=k, ai=ai, c=c))
                        if not os.path.exists(f):
                            missing.append(os.path.basename(f))
                        else:
                            lam[ki, ai, off:off + per] = np.load(f)["lam"]
                        off += per
        np.savez_compressed(f"{LOCAL}/master_{mname}.npz", lam=lam, keys=RS,
                            alpha=alphas, missing=missing)
        print(f"{mname} combined (1e6): missing {len(missing)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", action="store_true", help="skip rsync")
    ap.add_argument("--final", action="store_true",
                    help="fetch/aggregate the final 10x campaign")
    ap.add_argument("--topup", action="store_true",
                    help="fetch rscan top-up and rebuild combined R masters")
    args = ap.parse_args()
    if args.topup:
        if not args.local:
            for d in ("results_rscan_top", "results_rscanf_top"):
                subprocess.run(["rsync", "-az", f"{HOST}:{REMOTE}/{d}", LOCAL],
                               check=True)
                n = len(glob.glob(os.path.join(LOCAL, d, "*.npz")))
                print(f"fetched {d}: {n} npz")
        agg_rscan_combined()
        raise SystemExit
    if args.final:
        if not args.local:
            fetch_final()
        aggregate_final()
        raise SystemExit
    if not args.local:
        fetch()
    agg_ablang()
    agg_simple("geoscan", "geoscan_a{k}_i{ai:02d}_c{c}.npz",
               [0.2, 0.5, 1.0, 2.0, 4.0], 100_000, A41)
    agg_simple("rscan", "rscan_R{k}_i{ai:02d}_c{c}.npz",
               [0.3, 0.5, 0.8, 1.0, 1.2, 1.5], 65_536, A41)
    agg_spectrum()
    agg_kac()
    print("masters written to", os.path.abspath(LOCAL))
