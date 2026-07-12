#!/usr/bin/env python
"""One-screen summary of a PINN run's ``metrics.json``.

Usage:
    python scripts/inspect_pinn_diagnostics.py <run_dir | metrics.json>
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


def _fmt(v, spec=".4g"):
    return format(v, spec) if isinstance(v, (int, float)) else str(v)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    m = _load(sys.argv[1])

    run, cfg = m.get("run", {}), m.get("config", {})
    net = cfg.get("network", {})
    print(f"run={run.get('name')}  schema={m.get('schema_version')}  "
          f"{run.get('timestamp_utc')}")
    print(f"arch={net.get('arch_name')}  params={net.get('n_parameters')}  "
          f"dim={cfg.get('dim')}D  parametrization={cfg.get('parametrization')}  "
          f"device={cfg.get('device')}  seed={cfg.get('resolved_seed')}")

    wall = m.get("wall_time_seconds") or {}
    if wall:
        parts = "  ".join(f"{k}={_fmt(v)}s" for k, v in wall.items() if v is not None)
        print(f"wall_time:  {parts}")

    loss = (m.get("loss") or {}).get("final") or {}
    if loss:
        parts = "  ".join(f"{k}={_fmt(v)}" for k, v in loss.items())
        print(f"loss.final:  {parts}")

    bc = m.get("bc_residual") or {}
    if bc:
        print(f"BC residual: total={_fmt(bc.get('total'))}")
        comp = bc.get("per_component") or {}
        print("  " + "  ".join(f"{k}={_fmt(v)}" for k, v in comp.items()))

    pde = m.get("pde_residual") or {}
    if pde:
        print(f"PDE residual ({pde.get('n_points')} interior points, "
              f"nonfinite_fraction={_fmt(pde.get('nonfinite_fraction'))}):")
        for name, s in (pde.get("per_channel") or {}).items():
            print(f"  {name:<14} max={_fmt(s.get('max'))}  mean={_fmt(s.get('mean'))}")

    vol = m.get("vs_reference_volume") or {}
    if vol.get("available"):
        g = vol.get("global", {})
        conv = vol.get("convergence", {})
        print(f"û vs exact ODE ({vol.get('n_points_finite')} finite points):")
        print(f"  L2={_fmt(g.get('L2'), '.5g')}  p99={_fmt(g.get('p99'), '.5g')}  "
              f"L_inf={_fmt(g.get('L_inf'), '.4g')}  "
              f"converged={conv.get('converged')}")
    else:
        print(f"û vs exact ODE: unavailable ({vol.get('reason')})")


if __name__ == "__main__":
    main()
