# Spinning Billiards

Code and data accompanying the paper:

**Spinning Billiards**
Jacob S. Lund, Jeff Murugan, Jonathan P. Shock

## Overview

This project studies how internal spin affects chaos in classical 2D billiard systems. We investigate a one-parameter family of spinning billiards parametrised by the dimensionless moment of inertia $\alpha = I/(mr^2) \in [0,1]$, finding that spin reduces but does not eliminate chaos across four geometries: circle, rectangle (integrable), stadium, and Sinai (chaotic).

## Repository Structure

```
spinning_billiards.py       # Core simulation engine (Numba JIT)
make_plots.py               # Generates all publication figures
precompute_data.py          # Produces precomputed datasets
spinning_billiards_clean.tex # Paper source (LaTeX, REVTeX 4-2)
plots_v2/                   # Publication-quality figures (PDF)
experiment_results/         # Precomputed data (.npz files)
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Numba

Install dependencies:
```bash
pip install numpy matplotlib numba
```

## Reproducing Figures

All figures in the paper can be regenerated from the precomputed data:

```bash
python make_plots.py
```

To recompute the underlying datasets from scratch (this may take several hours):

```bash
python precompute_data.py
```

This produces `unified_billiards_data.npz` (~76 MB), which is not included in the repository due to its size. The smaller per-figure datasets in `experiment_results/` are included and sufficient for `make_plots.py`.


## License

Please contact the authors for licensing information.
