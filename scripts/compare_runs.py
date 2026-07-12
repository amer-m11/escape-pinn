#!/usr/bin/env python
"""Side-by-side diff of two PINN runs' ``metrics.json`` files.

Usage:
    python scripts/compare_runs.py <a: run_dir|metrics.json> <b: ...>

Prints an aligned table of the flat metrics (config, wall time, losses,
BC/PDE residuals, accuracy vs the exact ODE reference) with the relative
change.
"""
import json
import sys
from pathlib import Path


def _load(target: str) -> dict:
    p = Path(target)
    if p.is_dir():
        candidates = sorted(p.rglob("metrics.json"))
        if not candidates:
            raise SystemExit(f"no metrics.json found under {p}")
        p = candidates[0]
    with open(p) as f:
        return json.load(f)


def _get(m: dict, path: str):
    node = m
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


ROWS = [
    ("run", "run.name"),
    ("arch", "config.network.arch_name"),
    ("params", "config.network.n_parameters"),
    ("dim", "config.dim"),
    ("parametrization", "config.parametrization"),
    ("adam epochs", "config.adam_epochs"),
    ("lbfgs steps", "config.lbfgs_steps"),
    ("wall total [s]", "wall_time_seconds.total"),
    ("loss total", "loss.final.total"),
    ("loss pde", "loss.final.pde"),
    ("loss bc", "loss.final.bc"),
    ("bc residual", "bc_residual.total"),
    ("pde nan frac", "pde_nan_fraction"),
    ("vs ODE L2", "vs_reference_volume.global.L2"),
    ("vs ODE p99", "vs_reference_volume.global.p99"),
    ("vs ODE L_inf", "vs_reference_volume.global.L_inf"),
    ("vs ODE converged", "vs_reference_volume.convergence.converged"),
]


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return format(v, ".5g")
    return str(v)


def _delta(a, b):
    if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return ""
    if isinstance(a, bool) or isinstance(b, bool) or a == 0:
        return ""
    return f"{(b - a) / abs(a) * 100:+.1f}%"


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = _load(sys.argv[1]), _load(sys.argv[2])

    label_w = max(len(label) for label, _ in ROWS) + 2
    col_w = 14
    print(f"{'':<{label_w}}{'A':>{col_w}}{'B':>{col_w}}{'Δ(B vs A)':>{col_w}}")
    for label, path in ROWS:
        va, vb = _get(a, path), _get(b, path)
        if va is None and vb is None:
            continue
        print(f"{label:<{label_w}}{_fmt(va):>{col_w}}{_fmt(vb):>{col_w}}"
              f"{_delta(va, vb):>{col_w}}")

    # Per-component BC residuals when both runs carry them.
    ca = _get(a, "bc_residual.per_component") or {}
    cb = _get(b, "bc_residual.per_component") or {}
    common = [k for k in ca if k in cb]
    if common:
        print("\nBC per component:")
        for k in common:
            print(f"  {k:<{label_w - 2}}{_fmt(ca[k]):>{col_w}}{_fmt(cb[k]):>{col_w}}"
                  f"{_delta(ca[k], cb[k]):>{col_w}}")


if __name__ == "__main__":
    main()
