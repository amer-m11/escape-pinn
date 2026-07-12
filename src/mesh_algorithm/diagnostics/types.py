"""Solve metadata captured by the mesh solver (counters + phase codes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

SCHEMA_VERSION = 3

# Mesh algorithm version, stored on `SolveDiagnostics.solver_version` so a
# persisted diagnostics dict records which solver produced it.
SOLVER_VERSION = "v1"


class AcceptancePhase(IntEnum):
    """How a node reached ACCEPTED, or whether it never did."""

    NEVER_ACCEPTED = 0
    BOUNDARY = 1
    MARCH = 2
    FALLBACK = 3


class FailureReason(IntEnum):
    """Last failure reason recorded for a node."""

    NONE = 0
    NO_FACE_CANDIDATE = 1
    FACE_WITHOUT_ACCEPTED_CORNERS = 2
    FACE_INTERPOLATION_OUT_OF_BOUNDS = 3
    CONTINUATION_SPATIAL_OOB = 4
    CONTINUATION_PHI_OOB = 5
    CONTINUATION_EXHAUSTED = 6


@dataclass(frozen=True, slots=True)
class SolveDiagnostics:
    """Metadata captured during a single solver run."""

    total_nodes: int
    grid_shape: tuple[int, ...]
    boundary_accepted: int
    considered_seeded: int

    accepted_after_initialize: int
    accepted_in_march: int
    accepted_after_march: int
    remaining_after_march: int
    fallback_used: bool
    fallback_sweeps: int
    accepted_in_fallback: int
    accepted_after_fallback: int
    remaining_after_fallback: int
    final_considered: int
    final_far: int

    heap_pushes: int
    heap_pops: int
    stale_heap_pops: int
    update_attempts: int
    update_improvements: int
    """tentative value updates during march"""
    non_finite_accepted: int = 0
    """Accepted node with inf/nan value. Should always be zero."""

    wall_seconds: float = 0.0
    """total wall time"""
    initialize_seconds: float = 0.0
    march_seconds: float = 0.0
    fallback_seconds: float = 0.0

    # Slowness field statistics
    slowness_min: float = 0.0
    slowness_max: float = 0.0
    slowness_mean: float = 0.0

    solved_at_utc: str = ""
    """Timestamp of solve completion. Relevant for cached meshes"""
    git_commit: str | None = None
    """Git commit hash of the code that ran the solve. Relevant for cached meshes."""
    solver_version: str = SOLVER_VERSION

    continuation_max_depth: int = 0
    """Max ray tracing continuation depth."""
    continuation_total_cells: int = 0
    """Total ray tracing continuation cells across all rays."""

    clamp_events: int = 0
    """Number of intersections pulled back to the boundary during face interpolation."""
    clamp_accepted: int = 0
    """Number of ACCEPTED nodes whose winning characteristic used at least one clamp."""

    accepted_per_phase: list[int] = field(default_factory=list)
    """Accepted node count per phase: [after_initialize, after_march, after_fallback_sweep_1, ...]"""
    failure_reasons_enabled: bool = False
    phase_counts: dict[str, int] = field(default_factory=dict)
    failure_reason_counts: dict[str, int] = field(default_factory=dict)

    schema_version: int = SCHEMA_VERSION

    @property
    def pct_accepted_after_march(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return 100.0 * self.accepted_after_march / self.total_nodes

    @property
    def pct_accepted_final(self) -> float:
        if self.total_nodes == 0:
            return 0.0
        return 100.0 * self.accepted_after_fallback / self.total_nodes

    def summary_lines(self) -> list[str]:
        """Human-readable one-liners for printing after a solve."""
        return [
            f"nodes={self.total_nodes}  grid={self.grid_shape}",
            f"accepted: boundary={self.boundary_accepted}  "
            f"after_march={self.accepted_after_march} "
            f"({self.pct_accepted_after_march:.2f}%)  "
            f"final={self.accepted_after_fallback} "
            f"({self.pct_accepted_final:.2f}%)",
            f"fallback_used={self.fallback_used} (sweeps={self.fallback_sweeps})  "
            f"non_finite_accepted={self.non_finite_accepted}",
            f"wall={self.wall_seconds:.2f}s (init={self.initialize_seconds:.2f}s "
            f"march={self.march_seconds:.2f}s fallback={self.fallback_seconds:.2f}s)",
        ] + (
            [f"failure_reasons: {self.failure_reason_counts}"]
            if self.failure_reason_counts
            else []
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "schema_version": int(self.schema_version),
            "total_nodes": int(self.total_nodes),
            "grid_shape": [int(v) for v in self.grid_shape],
            "boundary_accepted": int(self.boundary_accepted),
            "considered_seeded": int(self.considered_seeded),
            "accepted_after_initialize": int(self.accepted_after_initialize),
            "accepted_in_march": int(self.accepted_in_march),
            "accepted_after_march": int(self.accepted_after_march),
            "remaining_after_march": int(self.remaining_after_march),
            "fallback_used": bool(self.fallback_used),
            "fallback_sweeps": int(self.fallback_sweeps),
            "accepted_in_fallback": int(self.accepted_in_fallback),
            "accepted_after_fallback": int(self.accepted_after_fallback),
            "remaining_after_fallback": int(self.remaining_after_fallback),
            "final_considered": int(self.final_considered),
            "final_far": int(self.final_far),
            "heap_pushes": int(self.heap_pushes),
            "heap_pops": int(self.heap_pops),
            "stale_heap_pops": int(self.stale_heap_pops),
            "update_attempts": int(self.update_attempts),
            "update_improvements": int(self.update_improvements),
            "non_finite_accepted": int(self.non_finite_accepted),
            "wall_seconds": float(self.wall_seconds),
            "initialize_seconds": float(self.initialize_seconds),
            "march_seconds": float(self.march_seconds),
            "fallback_seconds": float(self.fallback_seconds),
            "slowness_min": float(self.slowness_min),
            "slowness_max": float(self.slowness_max),
            "slowness_mean": float(self.slowness_mean),
            "solved_at_utc": str(self.solved_at_utc),
            "git_commit": self.git_commit,
            "solver_version": str(self.solver_version),
            "continuation_max_depth": int(self.continuation_max_depth),
            "continuation_total_cells": int(self.continuation_total_cells),
            "clamp_events": int(self.clamp_events),
            "clamp_accepted": int(self.clamp_accepted),
            "accepted_per_phase": [int(v) for v in self.accepted_per_phase],
            "failure_reasons_enabled": bool(self.failure_reasons_enabled),
            "phase_counts": {
                str(name): int(count) for name, count in self.phase_counts.items()
            },
            "failure_reason_counts": {
                str(name): int(count)
                for name, count in self.failure_reason_counts.items()
            },
            "pct_accepted_after_march": float(self.pct_accepted_after_march),
            "pct_accepted_final": float(self.pct_accepted_final),
        }

