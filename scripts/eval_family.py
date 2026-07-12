#!/usr/bin/env python
"""Evaluate a slowness-conditioned PINN over a family of media (generalization).

For each held-out (and optionally random) lens descriptor, build the
exact direct-ODE full-volume reference, bind that descriptor into the conditioned
model, and score it with the ``vs_reference_volume`` metric. Reports L2 / p99
as a distribution over the family.

Usage:
    PYTHONPATH=src pipenv run python scripts/eval_family.py \
        --config jobs/config.yaml --case gaussian_lens_family_cart_3d \
        --checkpoint output/.../model.pt --volume-n-points 50000 --random 4
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pinn_solver.configuration.slowness import _ConditionedFamily
from pinn_solver.evaluation.metrics import compute_run_metrics
from pinn_solver.networks import NetworkFactory, bind_fixed_conditioning
from pinn_solver.evaluation.ode_reference import (
    get_or_build_ode_reference_points_full,
)
from pinn_solver.run import _load_case
from pinn_solver.trainers import TrainHistory


def _descriptors(family, n_random, random_seed):
    """(tag, physical-descriptor) rows: every held-out tuple + N random draws."""
    dtype = torch.get_default_dtype()
    rows = [("held_out", torch.tensor(t, dtype=dtype)) for t in family.held_out]
    if n_random > 0:
        g = torch.Generator().manual_seed(random_seed)
        rows += [("random", p) for p in family.sample_params(n_random, generator=g, dtype=dtype)]
    return rows


def _summarize(vals):
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "worst": max(vals),
        "best": min(vals),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--checkpoint", required=True, help="conditioned model.pt (state_dict)")
    p.add_argument("--mesh-cache-dir", default="output/mesh_cache")
    p.add_argument("--volume-n-points", type=int, default=50000)
    p.add_argument("--volume-seed", type=int, default=0)
    p.add_argument("--random", type=int, default=4,
                   help="random in-range descriptors to score alongside the held-out set")
    p.add_argument("--random-seed", type=int, default=12345)
    p.add_argument("--n-grid", type=int, default=41)
    p.add_argument("--device", default="cpu")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   help="dotted config override (repeatable) — MUST match the "
                        "training --set flags so the rebuilt arch matches the checkpoint.")
    p.add_argument("--output", default=None, help="write the full per-medium + summary JSON here")
    args = p.parse_args()

    cfg = _load_case(args.config, args.case, args.overrides)
    family = cfg.slowness
    if not isinstance(family, _ConditionedFamily):
        raise SystemExit(
            f"eval_family requires a conditioned family slowness; got "
            f"{type(family).__name__}. Point --case at a *_family case."
        )
    if cfg.net.conditioning_dim == 0:
        cfg.net.conditioning_dim = family.cond_dim

    device = torch.device(args.device)
    model = NetworkFactory.create_network(
        cfg.net, cfg.physical_size, cfg.parametrization
    )
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    rows = _descriptors(family, args.random, args.random_seed)
    per_medium = []
    print(f"{'tag':<10}{'L2':>10}{'p99':>10}{'L_inf':>9}  descriptor")
    for i, (tag, params) in enumerate(rows):
        concrete = family.to_concrete(params)
        cfg_m = cfg.model_copy(update={"slowness": concrete})
        vref = get_or_build_ode_reference_points_full(
            cfg_m, n_points=args.volume_n_points, seed=args.volume_seed,
            cache_dir=args.mesh_cache_dir,
        )
        cond = family.cond_for(
            cfg_m.physical_size.as_tuple(), params.unsqueeze(0)
        ).squeeze(0).to(device)
        bound = bind_fixed_conditioning(model, cond)
        m = compute_run_metrics(
            run_name=f"{tag}_{i}", cfg=cfg_m, model=bound, history=TrainHistory(),
            volume_reference=vref, n_grid=args.n_grid, n_theta_eval=16,
        )
        vrv = m["vs_reference_volume"]
        g = vrv["global"]
        rec = {"tag": tag, "params": [round(v, 4) for v in params.tolist()],
               "L2": g["L2"], "p99": g["p99"], "L_inf": g["L_inf"]}
        per_medium.append(rec)
        print(f"{tag:<10}{g['L2']:>10.5f}{g['p99']:>10.5f}"
              f"{g['L_inf']:>9.3f}  {rec['params']}")

    def _by(tag, key):
        return [r[key] for r in per_medium if r["tag"] == tag]

    summary = {
        "held_out": {k: _summarize(_by("held_out", k)) for k in ("L2", "p99")},
        "random": {k: _summarize(_by("random", k)) for k in ("L2", "p99")},
    }
    print()
    for tag in ("held_out", "random"):
        s = summary[tag]["L2"]
        if s:
            print(f"{tag} L2: mean {s['mean']:.5f}  median {s['median']:.5f}  "
                  f"worst {s['worst']:.5f}  (n={s['n']})")

    out = {"config": args.config, "case": args.case, "checkpoint": args.checkpoint,
           "volume_n_points": args.volume_n_points, "per_medium": per_medium,
           "summary": summary}
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
