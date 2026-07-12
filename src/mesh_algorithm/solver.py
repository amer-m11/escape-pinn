"""Dimension-agnostic one-pass marching algorithm solver.

The solver reports detailed diagnostics about the solve.
"""

import heapq
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from .diagnostics.types import (
    SOLVER_VERSION,
    AcceptancePhase,
    FailureReason,
    SolveDiagnostics,
)
from .grid import Grid
from .node_state import NodeState


def _current_git_commit() -> str | None:
    """Return the short HEAD commit, or None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except FileNotFoundError, subprocess.SubprocessError:
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


class EscapeSolver:
    """One-pass marching algorithm solver for the phase-space escape equations.

    Computes ``û`` (escape time), ``σ̂`` (escape parameter), ``ŷ`` (exit
    position), and the exit direction and reports diagnostics about the solve.
    """

    def __init__(self, grid: Grid, *, track_failure_reasons: bool = False):
        self.grid = grid
        self._heap = []
        self._track_failure_reasons = bool(track_failure_reasons)
        self._acceptance_phase = None
        self._failure_reasons = None
        self._last_diagnostics = None
        self._reset_diagnostics_counters()

    def solve(self):
        """Run the full algorithm: initialise boundaries, then march.

        In case of unaccepted nodes after the march, a fallback sweep
        tries to resolve them by re-trying every non-accepted node. It
        shouldn't trigger usually. A warning is printed if it does.

        Returns the grid (with û, σ̂, exit position/direction, node status).
        """
        self._reset_diagnostics_state()

        wall_start = time.perf_counter()
        t0 = wall_start
        self._initialize()
        t1 = time.perf_counter()
        self._initialize_seconds = t1 - t0
        self._accepted_after_initialize = self._count_state(NodeState.ACCEPTED)
        self._accepted_per_phase.append(self._accepted_after_initialize)

        self._march()
        t2 = time.perf_counter()
        self._march_seconds = t2 - t1
        self._accepted_after_march = self._count_state(NodeState.ACCEPTED)
        self._remaining_after_march = self._total_nodes - self._accepted_after_march
        self._accepted_per_phase.append(self._accepted_after_march)

        if np.any(self.grid.status != NodeState.ACCEPTED):
            self._fallback_used = True
            print(
                "Warning: some nodes left unaccepted after march, "
                "attempting to resolve with fallback sweep."
            )
            self._resolve_remaining()
        t3 = time.perf_counter()
        self._fallback_seconds = t3 - t2 if self._fallback_used else 0.0
        self._wall_seconds = t3 - wall_start

        self._finalize_diagnostics()

        assert self._last_diagnostics is not None
        remaining = self._last_diagnostics.remaining_after_fallback
        if remaining > 0:
            print(
                f"Warning: fallback left {remaining} of {self._total_nodes} nodes unresolved.",
                file=sys.stderr,
            )
        return self.grid

    # ===== Phases =====

    def _initialize(self):
        """Initializes the grid:
        1. All nodes start FAR.
        2. Outward-pointing boundary nodes set to ACCEPTED with û=σ̂=0.
        3. Nodes adjacent to ACCEPTED set to CONSIDERED and get a tentative
        value.
        """
        self._put_boundary_in_accepted()
        self._put_accepted_neighbors_in_considered()

    def _march(self):
        """Main marching loop.

        Pop the CONSIDERED node with smallest σ̂, mark ACCEPTED, relax its
        octant neighbors. Lazy-deletion: a node may appear multiple
        times on the heap with stale priorities -> skip it if already
        ACCEPTED.
        """
        g = self.grid
        heap = self._heap
        assert self._acceptance_phase is not None

        while heap:
            self._heap_pops += 1
            _, idx = heapq.heappop(heap)

            if g.status[idx] == NodeState.ACCEPTED:
                self._stale_heap_pops += 1
                continue

            g.status[idx] = NodeState.ACCEPTED
            if self._acceptance_phase[idx] == AcceptancePhase.NEVER_ACCEPTED:
                self._acceptance_phase[idx] = AcceptancePhase.MARCH
            if self._failure_reasons is not None:
                self._failure_reasons[idx] = FailureReason.NONE

            for nidx in g.octant_neighbors(idx):
                if g.status[nidx] == NodeState.ACCEPTED:
                    continue
                self.update(nidx)

    def _put_boundary_in_accepted(self):
        """Set û=σ̂=0 on every outward-pointing boundary node."""
        g = self.grid
        assert self._acceptance_phase is not None

        for idx in g.iter_indices():
            if not g.is_boundary(idx):
                continue
            if not g.points_outward(idx):
                continue
            g.accept_boundary_node(idx)
            self._acceptance_phase[idx] = AcceptancePhase.BOUNDARY
            if self._failure_reasons is not None:
                self._failure_reasons[idx] = FailureReason.NONE
            self._boundary_accepted += 1

    def _put_accepted_neighbors_in_considered(self):
        """Compute a tentative value for every FAR node adjacent to an
        ACCEPTED node, push to the heap.
        """
        g = self.grid
        for idx in g.iter_indices():
            if g.status[idx] != NodeState.FAR:
                continue
            if any(g.status[nidx] == NodeState.ACCEPTED for nidx in g.adjacent(idx)):
                if self.update(idx):
                    self._considered_seeded += 1

    # ===== Update =====

    def update(self, idx):
        """Tentatively update ``idx`` via the local cell characteristic.

        Only commits and pushes to the heap if the new σ̂ improves on the
        previous one for this node.
        """
        g = self.grid
        self._update_attempts += 1
        result = g.compute_characteristic(idx)
        depth = int(getattr(result, "depth", 0) or 0)
        if depth:
            self._continuation_total_cells += depth
            if depth > self._continuation_max_depth:
                self._continuation_max_depth = depth
        clamps = int(getattr(result, "clamps", 0) or 0)
        self._clamp_events += clamps
        if result.sigma < g.sigma[idx]:
            g.commit_characteristic(idx, result)
            heapq.heappush(self._heap, (result.sigma, idx))
            self._update_improvements += 1
            self._heap_pushes += 1
            self._clamp_touched[idx] = clamps > 0
            if self._failure_reasons is not None:
                self._failure_reasons[idx] = FailureReason.NONE
            return True
        if self._failure_reasons is not None:
            failure_reason = getattr(result, "failure_reason", None)
            if failure_reason is not None:
                self._failure_reasons[idx] = int(failure_reason)
        return False

    # ===== Fallback sweep for nodes the one-pass march couldn't resolve =====

    def _resolve_remaining(self):
        """Fallback sweep for nodes left unaccepted after the one-pass march.

        Gauss-Seidel re-tries every non-accepted node, calling the local
        cell characteristic without the octant restriction. Causality is
        preserved because all remaining nodes have σ̂ larger than every
        already-accepted value. If this fires, something might have regressed.
        """
        g = self.grid
        progress = True
        assert self._acceptance_phase is not None
        while progress:
            self._fallback_sweeps += 1
            progress = False
            for idx in g.iter_indices():
                if g.status[idx] == NodeState.ACCEPTED:
                    continue
                before = g.sigma[idx]
                self.update(idx)
                if g.sigma[idx] < before:
                    progress = True
            # Drain newly-added heap entries before the next sweep.
            while self._heap:
                self._heap_pops += 1
                _, idx = heapq.heappop(self._heap)
                if g.status[idx] == NodeState.ACCEPTED:
                    self._stale_heap_pops += 1
                    continue
                g.status[idx] = NodeState.ACCEPTED
                if self._acceptance_phase[idx] == AcceptancePhase.NEVER_ACCEPTED:
                    self._acceptance_phase[idx] = AcceptancePhase.FALLBACK
                if self._failure_reasons is not None:
                    self._failure_reasons[idx] = FailureReason.NONE
            # Snapshot for the convergence-timeline plot.
            self._accepted_per_phase.append(self._count_state(NodeState.ACCEPTED))

    # ===== Diagnostics =====

    @property
    def acceptance_phase(self) -> np.ndarray:
        """Indicates when each node was accepted: boundary, march, fallback, or never."""
        if self._acceptance_phase is None:
            raise RuntimeError("acceptance_phase is only available after solve()")
        return self._acceptance_phase

    @property
    def failure_reasons(self) -> np.ndarray | None:
        """Optional failure reason codes."""
        return self._failure_reasons

    @property
    def last_diagnostics(self) -> SolveDiagnostics | None:
        """Last diagnostics snapshot produced by ``solve()``."""
        return self._last_diagnostics

    def diagnostics(self) -> SolveDiagnostics:
        """Return the diagnostics from the last ``solve()`` run."""
        if self._last_diagnostics is not None:
            return self._last_diagnostics

        g = self.grid
        total_nodes = int(np.prod(g.status.shape))
        accepted = self._count_state(NodeState.ACCEPTED)
        considered = self._count_state(NodeState.CONSIDERED)
        far = self._count_state(NodeState.FAR)
        return SolveDiagnostics(
            total_nodes=total_nodes,
            grid_shape=tuple(int(v) for v in g.status.shape),
            boundary_accepted=0,
            considered_seeded=0,
            accepted_after_initialize=accepted,
            accepted_in_march=0,
            accepted_after_march=accepted,
            remaining_after_march=total_nodes - accepted,
            fallback_used=False,
            fallback_sweeps=0,
            accepted_in_fallback=0,
            accepted_after_fallback=accepted,
            remaining_after_fallback=total_nodes - accepted,
            final_considered=considered,
            final_far=far,
            heap_pushes=0,
            heap_pops=0,
            stale_heap_pops=0,
            update_attempts=0,
            update_improvements=0,
            failure_reasons_enabled=self._track_failure_reasons,
            phase_counts=self._phase_counts(),
            failure_reason_counts=self._failure_reason_counts(),
        )

    def _reset_diagnostics_counters(self) -> None:
        self._total_nodes = int(np.prod(self.grid.status.shape))
        self._boundary_accepted = 0
        self._considered_seeded = 0
        self._accepted_after_initialize = 0
        self._accepted_after_march = 0
        self._remaining_after_march = self._total_nodes
        self._fallback_used = False
        self._fallback_sweeps = 0
        self._heap_pushes = 0
        self._heap_pops = 0
        self._stale_heap_pops = 0
        self._update_attempts = 0
        self._update_improvements = 0
        self._wall_seconds = 0.0
        self._initialize_seconds = 0.0
        self._march_seconds = 0.0
        self._fallback_seconds = 0.0
        self._continuation_max_depth = 0
        self._continuation_total_cells = 0
        self._clamp_events = 0
        self._accepted_per_phase: list[int] = []

    def _reset_diagnostics_state(self) -> None:
        self._heap.clear()
        self._acceptance_phase = np.full(
            self.grid.status.shape,
            AcceptancePhase.NEVER_ACCEPTED,
            dtype=np.int8,
        )
        self._clamp_touched = np.zeros(self.grid.status.shape, dtype=bool)
        self._failure_reasons = None
        if self._track_failure_reasons:
            self._failure_reasons = np.full(
                self.grid.status.shape,
                FailureReason.NONE,
                dtype=np.int8,
            )
        self._last_diagnostics = None
        self._reset_diagnostics_counters()

    def _count_state(self, state: NodeState) -> int:
        return int(np.sum(self.grid.status == state))

    def _phase_counts(self) -> dict[str, int]:
        if self._acceptance_phase is None:
            return {
                "never_accepted": self._total_nodes,
                "boundary": 0,
                "march": 0,
                "fallback": 0,
            }
        return {
            "never_accepted": int(
                np.sum(self._acceptance_phase == AcceptancePhase.NEVER_ACCEPTED)
            ),
            "boundary": int(np.sum(self._acceptance_phase == AcceptancePhase.BOUNDARY)),
            "march": int(np.sum(self._acceptance_phase == AcceptancePhase.MARCH)),
            "fallback": int(np.sum(self._acceptance_phase == AcceptancePhase.FALLBACK)),
        }

    def _failure_reason_counts(self) -> dict[str, int]:
        if self._failure_reasons is None:
            return {}
        counts = {}
        for reason in FailureReason:
            if reason == FailureReason.NONE:
                continue
            count = int(np.sum(self._failure_reasons == int(reason)))
            if count > 0:
                counts[reason.name.lower()] = count
        return counts

    def _finalize_diagnostics(self) -> None:
        accepted_after_fallback = self._count_state(NodeState.ACCEPTED)
        final_considered = self._count_state(NodeState.CONSIDERED)
        final_far = self._count_state(NodeState.FAR)
        phase_counts = self._phase_counts()
        slowness_min, slowness_max, slowness_mean = self._slowness_stats()
        self._last_diagnostics = SolveDiagnostics(
            total_nodes=self._total_nodes,
            grid_shape=tuple(int(v) for v in self.grid.status.shape),
            boundary_accepted=self._boundary_accepted,
            considered_seeded=self._considered_seeded,
            accepted_after_initialize=self._accepted_after_initialize,
            accepted_in_march=phase_counts["march"],
            accepted_after_march=self._accepted_after_march,
            remaining_after_march=self._remaining_after_march,
            fallback_used=self._fallback_used,
            fallback_sweeps=self._fallback_sweeps,
            accepted_in_fallback=phase_counts["fallback"],
            accepted_after_fallback=accepted_after_fallback,
            remaining_after_fallback=self._total_nodes - accepted_after_fallback,
            final_considered=final_considered,
            final_far=final_far,
            heap_pushes=self._heap_pushes,
            heap_pops=self._heap_pops,
            stale_heap_pops=self._stale_heap_pops,
            update_attempts=self._update_attempts,
            update_improvements=self._update_improvements,
            non_finite_accepted=self._count_non_finite_accepted(),
            wall_seconds=float(self._wall_seconds),
            initialize_seconds=float(self._initialize_seconds),
            march_seconds=float(self._march_seconds),
            fallback_seconds=float(self._fallback_seconds),
            slowness_min=slowness_min,
            slowness_max=slowness_max,
            slowness_mean=slowness_mean,
            solved_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            git_commit=_current_git_commit(),
            solver_version=SOLVER_VERSION,
            continuation_max_depth=int(self._continuation_max_depth),
            continuation_total_cells=int(self._continuation_total_cells),
            accepted_per_phase=list(self._accepted_per_phase),
            clamp_events=int(self._clamp_events),
            clamp_accepted=int(
                np.sum(self._clamp_touched & (self.grid.status == NodeState.ACCEPTED))
            ),
            failure_reasons_enabled=self._track_failure_reasons,
            phase_counts=phase_counts,
            failure_reason_counts=self._failure_reason_counts(),
        )

    def _slowness_stats(self) -> tuple[float, float, float]:
        """Return (min, max, mean) of the slowness field if available."""
        n_arr = getattr(self.grid, "_n", None)
        if n_arr is None:
            return 0.0, 0.0, 0.0
        arr = np.asarray(n_arr)
        return float(arr.min()), float(arr.max()), float(arr.mean())

    def _count_non_finite_accepted(self) -> int:
        """Count ACCEPTED nodes with at least one non-finite escape field."""
        accepted = self.grid.status == NodeState.ACCEPTED
        bad = np.zeros_like(accepted, dtype=bool)
        for name in ("u", "sigma", "y1", "y2", "y3", "phi_exit", "theta_exit"):
            if hasattr(self.grid, name):
                bad |= ~np.isfinite(getattr(self.grid, name))
        return int(np.sum(bad & accepted))
