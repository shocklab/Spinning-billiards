# Spinning Billiards

Code accompanying the paper:

**Spinning Billiards and Chaos**
Jacob S. Lund, Jeff Murugan, Jonathan P. Shock

## Overview

How does internal spin affect chaos in classical 2D billiards? We study a
one-parameter family parametrised by the dimensionless moment of inertia
$\alpha = I/(mr^2) \in [0,1]$, across four geometries: circle and rectangle
(integrable), stadium and Sinai (chaotic). Spin reduces chaos but does not
eliminate it.

The collision law couples the translational and rotational degrees of freedom.
In the rescaled spin variable $\tilde u = \sqrt{\alpha}\,u$ the energy becomes
Euclidean and every collision is an orthogonal transformation of
$(v_x, v_y, \tilde u)$, which is the fact most of the analysis rests on.

## What is here, and what is not

This repository contains **code only**: the simulation engine, the scripts that
generate every figure in the paper, and the pipeline that produces the datasets
those scripts read.

It does **not** contain the datasets or the figures. The full result set is
about 13 GB, too large for version control, and it is reproducible from the
scripts below. Earlier versions of this repository carried a small subset of
precomputed data and the figures built from it; both have been removed, since
they predate the current results.

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

Independent checks of the quantities the paper's argument turns on:

```
kac_step_a_check.py           mean collisions and mean time between curved-wall
                              collisions, against two parameter-free anchors
verify_flat_only_stretch.py   per-collision and per-unit-time stretch ratios
                              for the flat-only ablation variant
make_fig_ergodicity.py        spread of per-orbit visit frequency against the
                              sampling floor
```

## Requirements

Python 3.8+, NumPy, SciPy, Matplotlib, Numba. The Numba JIT does the heavy
lifting; the first call to any kernel pays a compilation cost.

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

Please contact the authors for licensing information.
