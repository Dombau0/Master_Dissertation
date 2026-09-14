# High-Dimensional Sampling Algorithms Utilizing Wasserstein Gradient Flows

Code accompanying my MSc dissertation in Statistical Science at the University of Oxford, which investigates whether a **Crank-Nicolson-type discretization** of Wasserstein gradient flows (WGFs) can improve on the standard **implicit Euler (JKO)** discretization used in existing variational, neural-transport-map sampling algorithms.

We derive a variational Crank-Nicolson formulation for the Kullback-Leibler (KL) and Jensen-Shannon (JS) divergences and incorporate it into an existing primal-dual, neural-transport-map algorithm, without introducing any additional neural networks. The two discretizations are compared empirically on synthetic Gaussian mixture sampling and on MNIST image generation.

## Repository Structure

```
.
├── config/                    # Hydra configuration files
│   ├── gmm_config.yaml        # Base config for the Gaussian mixture experiments
│   ├── img_config.yaml        # Base config for the MNIST image experiments
│   └── gmm_dimensions/        # Per-dimension overrides (2, 4, 8, 16, 32, 64, 128 dims)
├── src_gmm/                   # Source code for the Gaussian mixture experiments
│   ├── gmm_main_module.py     # Entry point (Hydra CLI)
│   ├── gmm_train_module.py    # Training loop (implicit Euler / Crank-Nicolson)
│   ├── gmm_network_module.py  # Transport and dual network architectures (RMLP)
│   ├── gmm_data_module.py     # Target/initial distribution and data handling
│   ├── gmm_logger_module.py   # Metric logging, checkpointing, plotting
│   └── gmm_utils_module.py    # Seeding, device setup, output directory handling
├── src_img/                   # Source code for the MNIST image experiments
│   ├── img_main_module.py     # Entry point (Hydra CLI)
│   ├── img_train_module.py    # Training loop (implicit Euler / Crank-Nicolson)
│   ├── img_network_module.py  # Transport (U-Net) and dual (ResNet) network architectures
│   ├── img_data_module.py     # MNIST loading and preprocessing
│   ├── img_logger_module.py   # Metric logging, checkpointing, sample generation
│   └── img_utils_module.py    # Seeding, device setup, output directory handling
└── batch/                     # Windows batch scripts to reproduce the reported experiments
```

## Method Overview

Both experiment tracks train a **transport network** $T_\theta$ and a **dual (variational) network** $H_\lambda$ in an alternating, primal-dual fashion to approximate successive JKO steps of a Wasserstein gradient flow. Setting `crank_nicolson: True` in the config switches the primal update from the standard implicit Euler loss to the Crank-Nicolson loss, which additionally averages in the frozen dual network and reference distribution from the *previous* JKO step — without requiring any extra network or added inference cost.

## Requirements

The code was developed and tested with Python 3.10+ and relies on the following main packages:

- `torch`, `torchvision`, `torchinfo`
- `hydra-core`, `omegaconf`
- `numpy`, `pandas`, `scikit-learn`
- `matplotlib`, `seaborn`
- `einops`
- `POT` (Python Optimal Transport)

Install them with:

```bash
pip install torch torchvision torchinfo hydra-core omegaconf numpy pandas scikit-learn matplotlib seaborn einops pot
```

(A pinned `requirements.txt` / `environment.yaml` may be added in the future — until then, any reasonably recent versions of the above should work.)

## Usage

Both experiment tracks use [Hydra](https://hydra.cc/) for configuration management, so any parameter in `config/gmm_config.yaml` or `config/img_config.yaml` can be overridden directly from the command line.

### Gaussian mixture sampling

Run a single configuration, e.g. 4-dimensional target, implicit Euler, step size α = 0.1:

```bash
python src_gmm/gmm_main_module.py gmm_dimensions=gmm_dimensions_004 alpha=0.1 jko_steps=40 crank_nicolson=False
```

Run a Crank-Nicolson sweep over multiple seeds using Hydra's multirun (`-m`):

```bash
python src_gmm/gmm_main_module.py -m gmm_dimensions=gmm_dimensions_002 alpha=0.1 jko_steps=40 seed=0,1,2,3,4 crank_nicolson=True
```

### MNIST image sampling

```bash
python src_img/img_main_module.py alpha=5 jko_steps=30 crank_nicolson=True
```

### Reproducing the dissertation's experiments

The `batch/` folder contains the exact commands used to produce the results reported in the dissertation: one script per Gaussian mixture dimensionality (`gmm_run_XXX.bat`), plus a robustness sweep across dimensions (`robust.bat`). These are Windows batch scripts; on macOS/Linux, run the equivalent `python ...` lines directly, or adapt the scripts to shell scripts (`.sh`).

## Configuration

Key parameters (see `config/gmm_config.yaml` and `config/img_config.yaml` for the full list):

| Parameter | Description |
|---|---|
| `crank_nicolson` | `True` for the Crank-Nicolson discretization, `False` for standard implicit Euler |
| `alpha` | JKO step size |
| `jko_steps` | Number of outer JKO steps |
| `dims` / `gmm_dimensions` | Dimensionality of the Gaussian mixture target (GMM experiments only) |
| `seed` | Random seed |
| `dual_steps` / `primal_steps` | Number of dual / primal gradient steps per batch |
| `output_directory` | Root directory for logs, checkpoints, and figures |

## Outputs

Each run creates a subdirectory under `output_directory`, named after its key parameters (dimension, α, discretization scheme, seed, ...), containing logged metrics, saved network checkpoints, and diagnostic plots.

## Acknowledgments

This implementation builds directly on the variational, primal-dual Wasserstein gradient flow algorithm of Fan et al. (2022), *Variational Wasserstein Gradient Flow*, ICML.

## Citation

If you use this code, please cite the accompanying dissertation:

```bibtex
@mastersthesis{Baus2026HighDimensionalSampling,
  author = {Dominik Christian Baus},
  title  = {High-Dimensional Sampling Algorithms Utilizing Wasserstein Gradient Flows},
  school = {University of Oxford},
  year   = {2026}
}
```
