# Phase-Space Computation of Multiple Arrivals

In a refracting medium a wave can reach the same point along several paths, but classical grid solvers keep only the first arrival. The *escape equations* of [Fomel & Sethian (PNAS, 2002)](https://doi.org/10.1073/pnas.102476599) recover all arrivals by recasting ray tracing as a static Hamilton–Jacobi problem in phase space, where the solution is single-valued.

This repository contains two solvers for the escape equations, in 2D and 3D:

- **`src/mesh_algorithm/`**: the paper's mesh-based "one-pass" marching solver (Dijkstra-like, unconditionally stable), used as the classical reference. 
- **`src/pinn_solver/`**: a mesh-free physics-informed neural network (PINN) trained directly on the escape equations, with two momentum parametrizations (spherical angles / Cartesian unit vector) and an optional slowness-conditioned mode that solves a whole *family* of media with one network.

Accuracy is scored against an exact characteristic-ODE reference (`src/pinn_solver/evaluation/ode_reference.py`) that integrates each ray individually.

## Setup

Requires Python 3.14.

```sh
pip install pipenv
pipenv install
```

## Quick start

All training cases live in a single file, `jobs/config.yaml`. Its `template` case is runnable and documents every available option. The other cases are small working examples. Train the 2D constant-medium case (a few minutes on CPU):

```sh
PYTHONPATH=src pipenv run python -m pinn_solver.run \
    --config jobs/config.yaml --case constant_medium \
    --output-dir output/pinn/smoke --set device=cpu
```

The run directory `<output-dir>/<case>/` receives `metrics.json` (a small machine-readable summary), `model.pt`, and three diagnostic figures (training loss, û heatmaps, signed error vs the exact ODE reference). Any config knob can be overridden on the command line with `--set dotted.key=value`:

```sh
PYTHONPATH=src pipenv run python -m pinn_solver.run \
    --config jobs/config.yaml --case gaussian_lens_cart_3d \
    --output-dir output/pinn/champion \
    --set net.trunk=pirate --set net.exit_time_features=true \
    --set net.factored_eikonal=true --set net.per_channel_heads=true
```

Run the mesh reference solver on the same media:

```sh
PYTHONPATH=src pipenv run python -m mesh_algorithm.examples.constant_medium_2d
PYTHONPATH=src pipenv run python -m mesh_algorithm.examples.variable_medium_3d
```

Draw rays and wavefronts bending through a medium:

```sh
PYTHONPATH=src pipenv run python -m ray_tracing.examples.gaussian_lens
```

## Evaluating runs

`metrics.json` is a nested JSON summary (losses, residual health, and û accuracy vs the exact ODE reference). Two helper scripts read it:

```sh
python scripts/inspect_pinn_diagnostics.py <run_dir>       # one-screen summary
python scripts/compare_runs.py <run_a> <run_b>             # two-run diff table
PYTHONPATH=src pipenv run python scripts/eval_family.py \
    --config jobs/config.yaml --case gaussian_lens_family_cart_3d \
    --checkpoint <model.pt>                                # conditioned models
```

## Repository layout

| Path | Contents |
|---|---|
| `src/mesh_algorithm/` | Mesh marching solver + runnable examples |
| `src/pinn_solver/` | PINN: networks, PDE residuals, training, metrics |
| `src/ray_tracing/` | Lagrangian ray/wavefront tracer + demos |
| `jobs/` | `config.yaml` (all cases + a fully-commented template) and the SLURM job scripts |
| `scripts/` | Run evaluation and comparison utilities |
| `bin/` | `sbatch`, `squeue`, `sync_runs` — SLURM helpers over SSH |

## Running on a cluster

Copy `.env.template` to `.env` and fill in the SSH settings. The job scripts expect the repo cloned on the cluster with a CUDA-enabled environment (see modules in `template.sh`).

```sh
bin/sbatch jobs/template.sh        # submit one training job (pulls, then sbatch)
bin/sbatch jobs/seed_template.sh   # 5-seed array of the same case (mean±std)
bin/squeue                         # your queue (add --watch to refresh)
bin/sync_runs                      # mirror the cluster's output/ back home
```

Edit `CASE` inside `jobs/template.sh` (adding `--set` overrides as needed) to pick what runs. `jobs/seed_template.sh` shows the seed-sweep pattern.

## Acknowledgements

This code was developed for a bachelor thesis at the University of Basel. This public version is a slightly trimmed release of the thesis codebase.

[Claude Code](https://claude.com/claude-code) was used to assist in the development and extension of this codebase.

## License

MIT — see [LICENSE](LICENSE).
