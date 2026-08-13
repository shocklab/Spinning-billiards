# Spinning Billiards

Code accompanying the paper:

**Spinning Billiards and Chaos**
Jacob S. Lund, Jeff Murugan, Jonathan P. Shock

## Overview

An ordinary billiard ball is a point mass. Give it a moment of inertia and let
the wall grip it, and each collision exchanges tangential velocity with spin.
This code simulates that system: a one-parameter family of billiards indexed by
the dimensionless moment of inertia $\alpha = I/(mr^2) \in [0,1]$, where
$\alpha = 0$ is the familiar specular billiard and $\alpha = 1$ a thin ring.

Four geometries are implemented: the circle and rectangle, which are integrable
without spin, and the stadium and Sinai billiard, which are chaotic. Spin
weakens the chaos in both: the Lyapunov exponent falls by about two thirds in
the stadium and three quarters in the Sinai billiard, but never reaches zero.

## Contents

The repository holds code: the simulation engine, the scripts that draw every
figure in the paper, and the pipeline that produces the datasets they read.

It does not hold the datasets or the figures themselves. The full result set
runs to about 13 GB, which is too large for version control and is in any case
reproducible from what is here.

## Simulation engine

```
spinning_billiards.py     collision law, four geometries, Benettin Lyapunov
                          algorithm, trajectory integration (Numba JIT)
```

## Generating the figures

Most figures are drawn from aggregated "master" datasets:

```
hpc_polish/replot_from_masters.py   lambda(alpha), chaotic fraction, geometry
                                    scan, universality collapse, DH scaling,
                                    Kac fragmentation, Lyapunov spectrum,
                                    arrival angles, wall-class ablation
```

The remainder are standalone:

```
make_plots.py                trajectories, FTLE distributions, phase-space
                             separation, energy conservation, LCN convergence
                             traces, collision rate
regenerate_v2_figures.py     driver for the above; also the obstacle-radius scan
make_psos_figure.py          three-dimensional Poincare section
check_island_prediction.py   bouncing-ball island against the linear model
analyse_arrival_angles.py    incidence-angle statistics at curved walls
analyse_spin_ablation.py     wall-class ablation
rerun_R_scan_hi.py           obstacle-radius scan at high statistics
```

## Regenerating the data

```
precompute_data.py               baseline sweeps
compute_spin_ablation.py         wall-class ablation kernels and arrival angles
kac_return_times.py              invariant measure and Kac return times
hpc_polish/polish_task.py        the production campaign, one task per array
                                 element (written for SLURM)
hpc_polish/fetch_and_aggregate.py  aggregates task output into the masters
```

`fetch_and_aggregate.py` copies task output back from a cluster; set
`SPIN2D_HOST` and `SPIN2D_REMOTE` in your environment to point it at yours.

The production runs used 10^6 initial conditions per point at 110 values of
$\alpha$, and assume a SLURM cluster. `polish_task.py` runs a single task and
can be called directly if you want a smaller slice without a scheduler.

## Verification

Independent checks of the main quantities the paper reports:

```
kac_step_a_check.py           mean collisions and mean time between curved-wall
                              collisions, against two parameter-free anchors
verify_flat_only_stretch.py   per-collision and per-unit-time stretch ratios
                              for the flat-only ablation variant
make_fig_ergodicity.py        spread of per-orbit visit frequency against the
                              sampling floor
```

## Requirements

Python 3.8+, NumPy, SciPy, Matplotlib, Numba. Kernels are JIT-compiled, so the first call to each one pays a
compilation cost.

```bash
pip install numpy scipy matplotlib numba
```

## Reproducing a figure from scratch

1. Run the relevant generator under "Regenerating the data" to produce the
   per-task output.
2. Aggregate with `hpc_polish/fetch_and_aggregate.py`.
3. Draw with `hpc_polish/replot_from_masters.py`, or the standalone script.

Standalone figures (the Poincare section, the island check) run directly from
`spinning_billiards.py` and need no precomputed data.

## License

MIT. See [LICENSE](LICENSE).

The manuscript text and figures are not covered by this licence; copyright in those is held by the authors and, on publication, by the journal.
