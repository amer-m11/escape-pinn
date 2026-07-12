"""Train one PINN case from a YAML configuration.

Usage:

    python -m pinn_solver.run --config jobs/config.yaml \
        --case constant_medium --output-dir output/pinn/smoke

Any config knob can be overridden on the command line with
``--set dotted.key=value`` (e.g. ``--set net.trunk=pirate --set device=cpu``).

The run directory receives, under ``<output-dir>/<case>/``:

- ``model.pt``      — trained ``state_dict``
- ``metrics.json``  — compact run summary (see :mod:`pinn_solver.evaluation.metrics`)
- ``training/loss.png``, ``solution/escape.png``, ``solution/error_vs_ode.png``
  — the three diagnostic figures (see :mod:`pinn_solver.evaluation.visualization`)

Accuracy is measured against the exact characteristic-ODE reference
(:mod:`pinn_solver.evaluation.ode_reference`): a fixed Sobol sample over the whole
trained phase-space volume, integrated once and cached per
``(slowness, n_points, seed)``.
"""

import argparse
import os

import matplotlib.pyplot as plt
import torch
import yaml

from .configuration import TrainConfiguration
from .evaluation.metrics import compute_run_metrics, save_run_metrics
from .solver import PINNRunner
from .evaluation.visualization import (
    plot_all_angles,
    plot_all_angles_3d,
    plot_error_all_angles,
    plot_training_loss,
)


def _apply_overrides(case_dict: dict, overrides: list[str]) -> dict:
    """Apply ``dotted.key=value`` overrides to a raw case dict in place.

    Lets a single base YAML case be swept over architecture/encoding knobs from
    the job script (e.g. ``--set net.trunk=pirate``) without duplicating the
    whole case block. The value is parsed as a YAML scalar, so ``5``, ``1e-3``,
    ``true`` and ``pirate`` coerce to int/float/bool/str as expected. Missing
    intermediate dicts are created.
    """
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"--set expects dotted.key=value, got {item!r}")
        key, raw = item.split("=", 1)
        value = yaml.safe_load(raw)
        node = case_dict
        parts = key.split(".")
        for p in parts[:-1]:
            nxt = node.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                node[p] = nxt
            node = nxt
        node[parts[-1]] = value
    return case_dict


def _load_case(
    config_path: str, case: str, overrides: list[str] | None = None
) -> TrainConfiguration:
    """Load one top-level case from a YAML file into a TrainConfiguration."""
    with open(config_path) as f:
        doc = yaml.safe_load(f)
    if case not in doc:
        available = [k for k in doc if k != "version"]
        raise SystemExit(
            f"case {case!r} not found in {config_path!r}; available cases: {available}"
        )
    case_dict = doc[case]
    if overrides:
        case_dict = _apply_overrides(dict(case_dict), overrides)
    return TrainConfiguration.model_validate(case_dict)


def _build_volume_reference(cfg, n_points: int, seed: int, cache_dir: str):
    """Exact-ODE reference at a Sobol sample over the trained volume.

    Cached per (slowness, n_points, seed). Degrades to ``None`` (metric
    skipped) instead of failing the run when the reference cannot be built.
    """
    if n_points <= 0:
        return None
    try:
        if cfg.physical_size.dim == 3:
            from .evaluation.ode_reference import get_or_build_ode_reference_points_full

            ref = get_or_build_ode_reference_points_full(
                cfg, n_points=n_points, seed=seed, cache_dir=cache_dir
            )
        else:
            from .evaluation.ode_reference import get_or_build_ode_reference_points_2d

            ref = get_or_build_ode_reference_points_2d(
                cfg, n_points=n_points, seed=seed, cache_dir=cache_dir
            )
        n_fin = int(ref["exited"].sum())
        print(f"Volume reference: {n_fin}/{n_points} exact-ODE rays (seed {seed}).")
        return ref
    except Exception as exc:  # noqa: BLE001 — degrade, don't fail the run
        print(f"Volume reference unavailable ({exc!r}); accuracy metric skipped.")
        return None


def _save_plots(cfg, model, history, out_dir: str) -> None:
    """The three diagnostic figures: loss curve, û heatmaps, error vs ODE."""
    is_3d = cfg.physical_size.dim == 3
    L = cfg.physical_size.as_tuple()
    view = (
        model.spherical_view()
        if is_3d and cfg.parametrization == "cartesian"
        else model
    )

    os.makedirs(os.path.join(out_dir, "training"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "solution"), exist_ok=True)

    ax = plot_training_loss(history)
    _save_fig(ax.figure, os.path.join(out_dir, "training", "loss.png"))

    fig = plot_all_angles_3d(view, L) if is_3d else plot_all_angles(model, L)
    _save_fig(fig, os.path.join(out_dir, "solution", "escape.png"))

    try:
        fig = plot_error_all_angles(view if is_3d else model, cfg)
        _save_fig(fig, os.path.join(out_dir, "solution", "error_vs_ode.png"))
    except Exception as exc:  # noqa: BLE001 — plots must not fail the run
        print(f"Error plot unavailable ({exc!r}); skipped.")


def _save_fig(fig, path: str) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run one PINN training case from a YAML configuration."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument(
        "--case",
        required=True,
        help="Top-level key in the YAML (e.g. constant_medium, gaussian_lens_3d).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="dotted.key=value",
        help="Override a config knob (repeatable), e.g. --set net.trunk=pirate.",
    )
    parser.add_argument(
        "--volume-n-points",
        type=int,
        default=200_000,
        help=(
            "Size of the Sobol phase-space sample for the accuracy metric "
            "vs the exact ODE reference. 0 disables the metric. The reference "
            "is integrated once and cached per (slowness, N, seed)."
        ),
    )
    parser.add_argument(
        "--volume-seed",
        type=int,
        default=0,
        help="Seed for the Sobol sample (keep fixed for cross-run ranking).",
    )
    parser.add_argument(
        "--cache-dir",
        default="output/mesh_cache",
        help="Disk cache directory for the ODE reference tables.",
    )
    args = parser.parse_args()

    cfg = _load_case(args.config, args.case, args.overrides)
    print(f"Loaded config: {args.config}::{args.case}")
    if args.overrides:
        print(f"  overrides: {args.overrides}")

    # ── Train ──────────────────────────────────────────────────────────────
    runner = PINNRunner(cfg)
    print(f"  device={runner.device} (config: {cfg.device})  seed={cfg.seed}")
    result = runner.run()

    # ── Reference + conditioned-family eval binding ────────────────────────
    volume_reference = _build_volume_reference(
        cfg, args.volume_n_points, args.volume_seed, args.cache_dir
    )

    # Conditioned (family) model: bind a representative mean medium so the
    # diagnostics can call the net without a ``cond`` tensor. The per-medium
    # distribution eval is scripts/eval_family.py.
    from .configuration.slowness import _ConditionedFamily
    from .networks import bind_fixed_conditioning

    eval_model = result.model
    if isinstance(cfg.slowness, _ConditionedFamily):
        lo, hi = cfg.slowness._bounds()
        mean_cond = cfg.slowness.cond_for(
            cfg.physical_size.as_tuple(), (0.5 * (lo + hi)).unsqueeze(0)
        ).squeeze(0)
        eval_model = bind_fixed_conditioning(result.model, mean_cond)
        print(
            "Conditioned family: diagnostics use the mean descriptor. "
            "Run scripts/eval_family.py for the per-medium distribution."
        )

    # ── Metrics + plots + checkpoint ───────────────────────────────────────
    example_dir = os.path.join(args.output_dir, args.case)
    os.makedirs(example_dir, exist_ok=True)

    metrics = compute_run_metrics(
        run_name=args.case,
        cfg=cfg,
        model=eval_model,
        history=result.history,
        volume_reference=volume_reference,
        wall_time_seconds={
            k: v.get("time_s") for k, v in result.phase_stats.items()
        },
        resolved_seed=runner.resolved_seed,
        resolved_device=str(runner.device),
    )
    metrics_path = os.path.join(example_dir, "metrics.json")
    save_run_metrics(metrics, metrics_path)
    print(f"Saved: {metrics_path}")
    vol = metrics.get("vs_reference_volume") or {}
    if vol.get("available"):
        g = vol["global"]
        print(
            f"û vs exact ODE: L2={g['L2']:.5f}  p99={g['p99']:.5f}  "
            f"L_inf={g['L_inf']:.4f}  ({g['n_points']} points)"
        )

    _save_plots(cfg, eval_model, result.history, example_dir)

    model_path = os.path.join(example_dir, "model.pt")
    torch.save(result.model.state_dict(), model_path)
    print(f"Saved: {model_path}")


if __name__ == "__main__":
    main()
