"""Structured diagnostics helpers for the mesh solver.

This package holds the solve metadata and phase coded acceptance
arrays that the final solved grid does not retain by itself.
"""

from .types import (
    SCHEMA_VERSION,
    AcceptancePhase,
    FailureReason,
    SolveDiagnostics,
)

__all__ = [
    "AcceptancePhase",
    "FailureReason",
    "SCHEMA_VERSION",
    "SolveDiagnostics",
]
